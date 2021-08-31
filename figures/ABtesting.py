import os
import argparse
import json
from cv2 import transform

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
import torch.nn.functional as F

from dataset import get_dataset, Subset
from utils.base import get_mlflow_model_by_name, SmartFormatter
from processing.pipeline_numpy import RawProcessingPipeline

from utils.hendrycks_robustness import Distortions

import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="AB testing, Show Results", formatter_class=SmartFormatter)

# Select experiment
parser.add_argument("--mode", type=str, default="ABShowImages", choices=('ABMakeTable', 'ABShowTable', 'ABShowImages', 'ABShowAllImages', 'CMakeTable', 'CShowTable', 'CShowImages', 'CShowAllImages'),
                    help='R|Choose operation to compute. \n'
                    'A) Lens2Logit image generation: \n  '
                    'ABMakeTable: Compute cross-validation metrics results \n  '
                    'ABShowTable: Plot cross-validation results on a table \n  '
                    'ABShowImages: Choose a training and testing image to compare different pipelines \n  '
                    'ABShowAllImages: Plot all possible pipelines \n'
                    'B) Hendrycks Perturbations, C-type dataset: \n  '
                    'CMakeTable: For each pipeline, it computes cross-validation metrics for different perturbations  \n  '
                    'CShowTable: Plot metrics for different pipelines and perturbations \n  '
                    'CShowImages: Plot an image with a selected a pipeline and perturbation\n  '
                    'CShowAllImages: Plot all possible perturbations for a fixed pipeline')

parser.add_argument("--dataset_name", type=str, default='Microscopy',
                    choices=['Microscopy', 'Drone', 'DroneSegmentation'], help='Choose dataset')
parser.add_argument("--augmentation", type=str, default='weak',
                    choices=['none', 'weak', 'strong'], help='Choose augmentation')
parser.add_argument("--N_runs", type=int, default=5, help='Number of k-fold splitting used in the training')
parser.add_argument("--download_model", default=False, action='store_true', help='Download Models in cache')

# Select pipelines
parser.add_argument("--dm_train", type=str, default='bilinear', choices=('bilinear', 'malvar2004',
                                                                         'menon2007'), help='Choose demosaicing for training processing model')
parser.add_argument("--s_train", type=str, default='sharpening_filter', choices=('sharpening_filter',
                                                                                 'unsharp_masking'), help='Choose sharpening for training processing model')
parser.add_argument("--dn_train", type=str, default='gaussian_denoising', choices=('gaussian_denoising',
                                                                                   'median_denoising'), help='Choose denoising for training processing model')
parser.add_argument("--dm_test", type=str, default='bilinear', choices=('bilinear', 'malvar2004',
                                                                        'menon2007'), help='Choose demosaicing for testing processing model')
parser.add_argument("--s_test", type=str, default='sharpening_filter', choices=('sharpening_filter',
                                                                                'unsharp_masking'), help='Choose sharpening for testing processing model')
parser.add_argument("--dn_test", type=str, default='gaussian_denoising', choices=('gaussian_denoising',
                                                                                  'median_denoising'), help='Choose denoising for testing processing model')

# Select Ctest parameters
parser.add_argument("--transform", type=str, default='identity', choices=('identity', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                                                                          'gaussian_blur', 'zoom_blur', 'contrast', 'brightness', 'saturate', 'elastic_transform'), help='Choose transformation to show for Ctesting')
parser.add_argument("--severity", type=int, default=1, choices=(1, 2, 3, 4, 5), help='Choose severity for Ctesting')

args = parser.parse_args()


class metrics:
    def __init__(self, confusion_matrix):
        self.cm = confusion_matrix
        self.N_classes = len(confusion_matrix)

    def accuracy(self):
        Tp = torch.diagonal(self.cm, 0).sum()
        N_elements = torch.sum(self.cm)
        return Tp / N_elements

    def precision(self):
        Tp_Fp = torch.sum(self.cm, 1)
        Tp_Fp[Tp_Fp == 0] = 1
        return torch.diagonal(self.cm, 0) / Tp_Fp

    def recall(self):
        Tp_Fn = torch.sum(self.cm, 0)
        Tp_Fn[Tp_Fn == 0] = 1
        return torch.diagonal(self.cm, 0) / Tp_Fn

    def f1_score(self):
        prod = (self.precision() * self.recall())
        sum = (self.precision() + self.recall())
        sum[sum == 0.] = 1.
        return 2 * (prod / sum)

    def over_N_runs(ms, N_runs):
        m, m2 = 0, 0

        for i in ms:
            m += i
        mu = m / N_runs

        for i in ms:
            m2 += (i - mu)**2

        sigma = torch.sqrt(m2 / (N_runs - 1))

        return mu.tolist(), sigma.tolist()


class ABtesting:
    def __init__(self,
                 dataset_name: str,
                 augmentation: str,
                 dm_train: str,
                 s_train: str,
                 dn_train: str,
                 dm_test: str,
                 s_test: str,
                 dn_test: str,
                 N_runs: int,
                 severity=1,
                 transform='identity',
                 download_model=False):
        self.experiment_name = 'ABtesting'
        self.dataset_name = dataset_name
        self.augmentation = augmentation
        self.dm_train = dm_train
        self.s_train = s_train
        self.dn_train = dn_train
        self.dm_test = dm_test
        self.s_test = s_test
        self.dn_test = dn_test
        self.N_runs = N_runs
        self.severity = severity
        self.transform = transform
        self.download_model = download_model

    def static_pip_val(self, debayer=None, sharpening=None, denoising=None, severity=None, transform=None, plot_mode=False):

        if debayer == None:
            debayer = self.dm_test
        if sharpening == None:
            sharpening = self.s_test
        if denoising == None:
            denoising = self.dn_test
        if severity == None:
            severity = self.severity
        if transform == None:
            transform = self.transform

        dataset = get_dataset(self.dataset_name)

        if self.dataset_name == "Drone" or self.dataset_name == "DroneSegmentation":
            mean = torch.tensor([0.35, 0.36, 0.35])
            std = torch.tensor([0.12, 0.11, 0.12])
        elif self.dataset_name == "Microscopy":
            mean = torch.tensor([0.91, 0.84, 0.94])
            std = torch.tensor([0.08, 0.12, 0.05])

        if not plot_mode:
            dataset.transform = Compose([RawProcessingPipeline(
                camera_parameters=dataset.camera_parameters,
                debayer=debayer,
                sharpening=sharpening,
                denoising=denoising,
            ), Distortions(severity=severity, transform=transform),
                Normalize(mean, std)])
        else:
            dataset.transform = Compose([RawProcessingPipeline(
                camera_parameters=dataset.camera_parameters,
                debayer=debayer,
                sharpening=sharpening,
                denoising=denoising,
            ), Distortions(severity=severity, transform=transform)])

        return dataset

    def ABclassification(self):

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        parent_run_name = f"{self.dataset_name}_{self.dm_train}_{self.s_train}_{self.dn_train}_{self.augmentation}"

        print(
            f'\nTraining pipeline:\n Dataset: {self.dataset_name}, Augmentation: {self.augmentation} \n Debayer: {self.dm_train}, Sharpening: {self.s_train}, Denoiser: {self.dn_train} \n')
        print(f'\nTesting pipeline:\n Dataset: {self.dataset_name}, Augmentation: {self.augmentation} \n Debayer: {self.dm_test}, Sharpening: {self.s_test}, Denoiser: {self.dn_test} \n Transform: {self.transform}, Severity: {self.severity}\n')

        accuracies, precisions, recalls, f1_scores = [], [], [], []

        os.system('rm -r /tmp/py*')

        for N_run in range(self.N_runs):

            print(f"Evaluating Run {N_run}")

            run_name = parent_run_name + '_' + str(N_run)

            state_dict, model = get_mlflow_model_by_name(self.experiment_name, run_name,
                                                         download_model=self.download_model)

            dataset = self.static_pip_val()
            valid_set = Subset(dataset, indices=state_dict['valid_indices'])
            valid_loader = DataLoader(valid_set, batch_size=1, num_workers=16, shuffle=False)

            model.eval()

            len_classes = len(dataset.classes)
            confusion_matrix = torch.zeros((len_classes, len_classes))

            for img, label in valid_loader:

                prediction = model(img.to(DEVICE)).detach().cpu()
                prediction = torch.argmax(prediction, dim=1)
                confusion_matrix[label, prediction] += 1  # Real value rows, Declared columns

            m = metrics(confusion_matrix)

            accuracies.append(m.accuracy())
            precisions.append(m.precision())
            recalls.append(m.recall())
            f1_scores.append(m.f1_score())

            os.system('rm -r /tmp/t*')

        accuracy = metrics.over_N_runs(accuracies, self.N_runs)
        precision = metrics.over_N_runs(precisions, self.N_runs)
        recall = metrics.over_N_runs(recalls, self.N_runs)
        f1_score = metrics.over_N_runs(f1_scores, self.N_runs)
        return dataset.classes, accuracy, precision, recall, f1_score

    def ABsegmentation(self):

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        parent_run_name = f"{self.dataset_name}_{self.dm_train}_{self.s_train}_{self.dn_train}_{self.augmentation}"

        print(
            f'\nTraining pipeline:\n Dataset: {self.dataset_name}, Augmentation: {self.augmentation} \n Debayer: {self.dm_train}, Sharpening: {self.s_train}, Denoiser: {self.dn_train} \n')
        print(f'\nTesting pipeline:\n Dataset: {self.dataset_name}, Augmentation: {self.augmentation} \n Debayer: {self.dm_test}, Sharpening: {self.s_test}, Denoiser: {self.dn_test} \n Transform: {self.transform}, Severity: {self.severity}\n')

        IoUs = []

        os.system('rm -r /tmp/py*')

        for N_run in range(self.N_runs):

            print(f"Evaluating Run {N_run}")

            run_name = parent_run_name + '_' + str(N_run)

            state_dict, model = get_mlflow_model_by_name(self.experiment_name, run_name,
                                                         download_model=self.download_model)

            dataset = self.static_pip_val()

            valid_set = Subset(dataset, indices=state_dict['valid_indices'])
            valid_loader = DataLoader(valid_set, batch_size=1, num_workers=16, shuffle=False)

            model.eval()

            IoU = 0

            for img, label in valid_loader:

                prediction = model(img.to(DEVICE)).detach().cpu()
                prediction = F.logsigmoid(prediction).exp().squeeze()
                IoU += smp.utils.metrics.IoU()(prediction, label)

            IoU = IoU / len(valid_loader)
            IoUs.append(IoU.item())

            os.system('rm -r /tmp/t*')

        IoU = metrics.over_N_runs(torch.tensor(IoUs), self.N_runs)
        return IoU

    def ABShowImages(self):

        path = 'results/ABtesting/imgs/'
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(
            path, f'{self.dataset_name}_{self.augmentation}_{self.dm_train[:2]}{self.s_train[0]}{self.dn_train[:2]}_{self.dm_test[:2]}{self.s_test[0]}{self.dn_test[:2]}')

        if not os.path.exists(path):
            os.makedirs(path)

        run_name = f"{self.dataset_name}_{self.dm_train}_{self.s_train}_{self.dn_train}_{self.augmentation}" + \
            '_' + str(0)

        state_dict, model = get_mlflow_model_by_name(self.experiment_name, run_name, download_model=self.download_model)

        model.augmentation = None

        for t in ([self.dm_train, self.s_train, self.dn_train, 'train_img'],
                  [self.dm_test, self.s_test, self.dn_test, 'test_img']):

            debayer, sharpening, denoising, img_type = t[0], t[1], t[2], t[3]

            dataset = self.static_pip_val(debayer=debayer, sharpening=sharpening, denoising=denoising, plot_mode=True)
            valid_set = Subset(dataset, indices=state_dict['valid_indices'])

            img, _ = next(iter(valid_set))

            plt.figure()
            plt.imshow(img.permute(1, 2, 0))
            if img_type == 'train_img':
                plt.title('Train Image')
                plt.savefig(os.path.join(path, f'img_train.png'))
                imgA = img
            else:
                plt.title('Test Image')
                plt.savefig(os.path.join(path, f'img_test.png'))

                for c, color in enumerate(['Red', 'Green', 'Blue']):
                    diff = torch.abs(imgA - img)
                    plt.figure()
                    # plt.imshow(diff.permute(1,2,0))
                    plt.imshow(diff[c, 50:200, 50:200], cmap=f'{color}s')
                    plt.title(f'|Train Image - Test Image| - {color}')
                    plt.colorbar()
                    plt.savefig(os.path.join(path, f'diff_{color}.png'))
                    plt.figure()
                    diff[diff == 0.] = 1e-5
                    # plt.imshow(torch.log(diff.permute(1,2,0)))
                    plt.imshow(torch.log(diff)[c])
                    plt.title(f'log(|Train Image - Test Image|) - color')
                    plt.colorbar()
                    plt.savefig(os.path.join(path, f'logdiff_{color}.png'))

            if self.dataset_name == 'DroneSegmentation':
                plt.figure()
                plt.imshow(model(img[None].cuda()).detach().cpu().squeeze())
                if img_type == 'train_img':
                    plt.savefig(os.path.join(path, f'mask_train.png'))
                else:
                    plt.savefig(os.path.join(path, f'mask_test.png'))

    def ABShowAllImages(self):
        if not os.path.exists('results/ABtesting'):
            os.makedirs('results/ABtesting')

        demosaicings = ['bilinear', 'malvar2004', 'menon2007']
        sharpenings = ['sharpening_filter', 'unsharp_masking']
        denoisings = ['median_denoising', 'gaussian_denoising']

        fig = plt.figure()
        columns = 4
        rows = 3

        i = 1

        for dm in demosaicings:
            for s in sharpenings:
                for dn in denoisings:

                    dataset = self.static_pip_val(self.dm_test, self.s_test,
                                                  self.dn_test, plot_mode=True)

                    img, _ = dataset[0]

                    fig.add_subplot(rows, columns, i)
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title(f'{dm}\n{s}\n{dn}', fontsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

                    i += 1

        plt.show()
        plt.savefig(f'results/ABtesting/ABpipelines.png')

    def CShowImages(self):

        path = 'results/Ctesting/imgs/'
        if not os.path.exists(path):
            os.makedirs(path)

        run_name = f"{self.dataset_name}_{self.dm_test}_{self.s_test}_{self.dn_test}_{self.augmentation}" + '_' + str(0)

        state_dict, model = get_mlflow_model_by_name(self.experiment_name, run_name, download_model=True)

        model.augmentation = None

        dataset = self.static_pip_val(self.dm_test, self.s_test, self.dn_test,
                                      self.severity, self.transform, plot_mode=True)
        valid_set = Subset(dataset, indices=state_dict['valid_indices'])

        img, _ = next(iter(valid_set))

        plt.figure()
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig(os.path.join(
            path, f'{self.dataset_name}_{self.augmentation}_{self.dm_train[:2]}{self.s_train[0]}{self.dn_train[:2]}_{self.transform}_sev{self.severity}'))

    def CShowAllImages(self):
        if not os.path.exists('results/Cimages'):
            os.makedirs('results/Cimages')

        transforms = ['identity', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                      'gaussian_blur', 'zoom_blur', 'contrast', 'brightness', 'saturate', 'elastic_transform']

        for i, t in enumerate(transforms):

            fig = plt.figure(figsize=(10, 6))
            columns = 5
            rows = 1

            for sev in range(1, 6):

                dataset = self.static_pip_val(severity=sev, transform=t, plot_mode=True)

                img, _ = dataset[0]

                fig.add_subplot(rows, columns, sev)
                plt.imshow(img.permute(1, 2, 0))
                plt.title(f'Severity: {sev}')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()

            if '_' in t:
                t = t.replace('_', ' ')
            t = t[0].upper() + t[1:]

            fig.suptitle(f'{t}', x=0.5, y=0.8, fontsize=24)
            plt.show()
            plt.savefig(f'results/Cimages/{i+1}_{t.lower()}.png')


def ABMakeTable(dataset_name: str, augmentation: str,
                N_runs: int, download_model: bool):

    demosaicings = ['bilinear', 'malvar2004', 'menon2007']
    sharpenings = ['sharpening_filter', 'unsharp_masking']
    denoisings = ['median_denoising', 'gaussian_denoising']

    path = 'results/ABtesting/tables'
    if not os.path.exists(path):
        os.makedirs(path)

    runs = {}
    i = 0

    for dm_train in demosaicings:
        for s_train in sharpenings:
            for dn_train in denoisings:
                for dm_test in demosaicings:
                    for s_test in sharpenings:
                        for dn_test in denoisings:
                            train_pip = [dm_train, s_train, dn_train]
                            test_pip = [dm_test, s_test, dn_test]
                            runs[f'run{i}'] = {
                                'dataset': dataset_name,
                                'augmentation': augmentation,
                                'train_pip': train_pip,
                                'test_pip': test_pip,
                                'N_runs': N_runs
                            }
                            ABclass = ABtesting(
                                dataset_name=dataset_name,
                                augmentation=augmentation,
                                dm_train=dm_train,
                                s_train=s_train,
                                dn_train=dn_train,
                                dm_test=dm_test,
                                s_test=s_test,
                                dn_test=dn_test,
                                N_runs=N_runs,
                                download_model=download_model
                            )

                            if dataset_name == 'DroneSegmentation':
                                IoU = ABclass.ABsegmentation()
                                runs[f'run{i}']['IoU'] = IoU
                            else:
                                classes, accuracy, precision, recall, f1_score = ABclass.ABclassification()
                                runs[f'run{i}']['classes'] = classes
                                runs[f'run{i}']['accuracy'] = accuracy
                                runs[f'run{i}']['precision'] = precision
                                runs[f'run{i}']['recall'] = recall
                                runs[f'run{i}']['f1_score'] = f1_score

                            with open(os.path.join(path, f'{dataset_name}_{augmentation}_runs.txt'), 'w') as outfile:
                                json.dump(runs, outfile)

                            i += 1


def ABShowTable(dataset_name: str, augmentation: str):

    path = 'results/ABtesting/tables'
    assert os.path.exists(path), 'No tables to plot'

    json_file = os.path.join(path, f'{dataset_name}_{augmentation}_runs.txt')

    with open(json_file, 'r') as run_file:
        runs = json.load(run_file)

        metrics = torch.zeros((2, 12, 12))
        classes = []

        i, j = 0, 0

        for r in range(len(runs)):

            run = runs['run' + str(r)]
            if dataset_name == 'DroneSegmentation':
                acc = run['IoU']
            else:
                acc = run['accuracy']
            if len(classes) < 12:
                class_list = run['test_pip']
                class_name = f'{class_list[0][:2]},{class_list[1][:1]},{class_list[2][:2]}'
                classes.append(class_name)
            mu, sigma = round(acc[0], 4), round(acc[1], 4)

            metrics[0, j, i] = mu
            metrics[1, j, i] = sigma

            i += 1

            if i == 12:
                i = 0
                j += 1

    differences = torch.zeros_like(metrics)

    diag_mu = torch.diagonal(metrics[0], 0)
    diag_sigma = torch.diagonal(metrics[1], 0)

    for r in range(len(metrics[0])):
        differences[0, r] = diag_mu[r] - metrics[0, r]
        differences[1, r] = torch.sqrt(metrics[1, r]**2 + diag_sigma[r]**2)

    # Plot with scatter

    for i, img in enumerate([metrics, differences]):

        x, y = torch.arange(12), torch.arange(12)
        x, y = torch.meshgrid(x, y)

        if i == 0:
            vmin = max(0.65, round(img[0].min().item(), 2))
            vmax = round(img[0].max().item(), 2)
            step = 0.02
        elif i == 1:
            vmin = round(img[0].min().item(), 2)
            if augmentation == 'none':
                vmax = min(0.15, round(img[0].max().item(), 2))
            if augmentation == 'weak':
                vmax = min(0.08, round(img[0].max().item(), 2))
            if augmentation == 'strong':
                vmax = min(0.05, round(img[0].max().item(), 2))
            step = 0.01

        vmin = int(vmin / step) * step
        vmax = int(vmax / step) * step

        fig = plt.figure(figsize=(10, 6.2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        marker_size = 350
        plt.scatter(x, y, c=torch.rot90(img[1][x, y], -1, [0, 1]), vmin=0.,
                    vmax=img[1].max(), cmap='viridis', s=marker_size * 2, marker='s')
        ticks = torch.arange(0., img[1].max(), 0.03).tolist()
        ticks = [round(tick, 2) for tick in ticks]
        cba = plt.colorbar(pad=0.06)
        cba.set_ticks(ticks)
        cba.ax.set_yticklabels(ticks)
        # cmap = plt.cm.get_cmap('tab20c').reversed()
        cmap = plt.cm.get_cmap('Reds')
        plt.scatter(x, y, c=torch.rot90(img[0][x, y], -1, [0, 1]), vmin=vmin,
                    vmax=vmax, cmap=cmap, s=marker_size, marker='s')
        ticks = torch.arange(vmin, vmax, step).tolist()
        ticks = [round(tick, 2) for tick in ticks]
        if ticks[-1] != vmax:
            ticks.append(vmax)
        cbb = plt.colorbar(pad=0.06)
        cbb.set_ticks(ticks)
        if i == 0:
            ticks[0] = f'<{str(ticks[0])}'
        elif i == 1:
            ticks[-1] = f'>{str(ticks[-1])}'
        cbb.ax.set_yticklabels(ticks)
        for x in range(12):
            for y in range(12):
                txt = round(torch.rot90(img[0], -1, [0, 1])[x, y].item(), 2)
                if str(txt) == '-0.0':
                    txt = '0.00'
                elif str(txt) == '0.0':
                    txt = '0.00'
                elif len(str(txt)) == 3:
                    txt = str(txt) + '0'
                else:
                    txt = str(txt)

                plt.text(x - 0.25, y - 0.1, txt, color='black', fontsize='x-small')

        ax.set_xticks(torch.linspace(0, 11, 12))
        ax.set_xticklabels(classes)
        ax.set_yticks(torch.linspace(0, 11, 12))
        classes.reverse()
        ax.set_yticklabels(classes)
        classes.reverse()
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        cba.set_label('Standard Deviation')
        plt.xlabel("Test pipelines")
        plt.ylabel("Train pipelines")
        plt.title(f'Dataset: {dataset_name}, Augmentation: {augmentation}')
        if i == 0:
            if dataset_name == 'DroneSegmentation':
                cbb.set_label('IoU')
                plt.savefig(os.path.join(path, f"{dataset_name}_{augmentation}_IoU.png"))
            else:
                cbb.set_label('Accuracy')
                plt.savefig(os.path.join(path, f"{dataset_name}_{augmentation}_accuracies.png"))
        elif i == 1:
            if dataset_name == 'DroneSegmentation':
                cbb.set_label('IoU_d-IoU')
            else:
                cbb.set_label('Accuracy_d - Accuracy')
            plt.savefig(os.path.join(path, f"{dataset_name}_{augmentation}_differences.png"))


def CMakeTable(dataset_name: str, augmentation: str, severity: int, N_runs: int, download_model: bool):

    path = 'results/Ctesting/tables'
    if not os.path.exists(path):
        os.makedirs(path)

    demosaicings = ['bilinear', 'malvar2004', 'menon2007']
    sharpenings = ['sharpening_filter', 'unsharp_masking']
    denoisings = ['median_denoising', 'gaussian_denoising']

    transformations = ['identity', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                       'gaussian_blur', 'zoom_blur', 'contrast', 'brightness', 'saturate', 'elastic_transform']

    runs = {}
    i = 0

    for dm in demosaicings:
        for s in sharpenings:
            for dn in denoisings:
                for t in transformations:
                    pip = [dm, s, dn]
                    runs[f'run{i}'] = {
                        'dataset': dataset_name,
                        'augmentation': augmentation,
                        'pipeline': pip,
                        'N_runs': N_runs,
                        'transform': t,
                        'severity': severity,
                    }
                    ABclass = ABtesting(
                        dataset_name=dataset_name,
                        augmentation=augmentation,
                        dm_train=dm,
                        s_train=s,
                        dn_train=dn,
                        dm_test=dm,
                        s_test=s,
                        dn_test=dn,
                        severity=severity,
                        transform=t,
                        N_runs=N_runs,
                        download_model=download_model
                    )

                    if dataset_name == 'DroneSegmentation':
                        IoU = ABclass.ABsegmentation()
                        runs[f'run{i}']['IoU'] = IoU
                    else:
                        classes, accuracy, precision, recall, f1_score = ABclass.ABclassification()
                        runs[f'run{i}']['classes'] = classes
                        runs[f'run{i}']['accuracy'] = accuracy
                        runs[f'run{i}']['precision'] = precision
                        runs[f'run{i}']['recall'] = recall
                        runs[f'run{i}']['f1_score'] = f1_score

                    with open(os.path.join(path, f'{dataset_name}_{augmentation}_runs.json'), 'w') as outfile:
                        json.dump(runs, outfile)

                    i += 1


def CShowTable(dataset_name, augmentation):

    path = 'results/Ctesting/tables'
    assert os.path.exists(path), 'No tables to plot'

    json_file = os.path.join(path, f'{dataset_name}_{augmentation}_runs.txt')

    transforms = ['identity', 'gauss_noise', 'shot', 'impulse', 'speckle',
                  'gauss_blur', 'zoom', 'contrast', 'brightness', 'saturate', 'elastic']

    pip = []

    demosaicings = ['bilinear', 'malvar2004', 'menon2007']
    sharpenings = ['sharpening_filter', 'unsharp_masking']
    denoisings = ['median_denoising', 'gaussian_denoising']

    for dm in demosaicings:
        for s in sharpenings:
            for dn in denoisings:
                pip.append(f'{dm[:2]},{s[0]},{dn[2]}')

    with open(json_file, 'r') as run_file:
        runs = json.load(run_file)

        metrics = torch.zeros((2, len(pip), len(transforms)))

        i, j = 0, 0

        for r in range(len(runs)):

            run = runs['run' + str(r)]
            if dataset_name == 'DroneSegmentation':
                acc = run['IoU']
            else:
                acc = run['accuracy']
            mu, sigma = round(acc[0], 4), round(acc[1], 4)

            metrics[0, j, i] = mu
            metrics[1, j, i] = sigma

            i += 1

            if i == len(transforms):
                i = 0
                j += 1

        # Plot with scatter

        img = metrics

        vmin = 0.
        vmax = 1.

        x, y = torch.arange(12), torch.arange(11)
        x, y = torch.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 6.2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        marker_size = 350
        plt.scatter(x, y, c=torch.rot90(img[1][x, y], -1, [0, 1]), vmin=0.,
                    vmax=img[1].max(), cmap='viridis', s=marker_size * 2, marker='s')
        ticks = torch.arange(0., img[1].max(), 0.03).tolist()
        ticks = [round(tick, 2) for tick in ticks]
        cba = plt.colorbar(pad=0.06)
        cba.set_ticks(ticks)
        cba.ax.set_yticklabels(ticks)
        # cmap = plt.cm.get_cmap('tab20c').reversed()
        cmap = plt.cm.get_cmap('Reds')
        plt.scatter(x, y, c=torch.rot90(img[0][x, y], -1, [0, 1]), vmin=vmin,
                    vmax=vmax, cmap=cmap, s=marker_size, marker='s')
        ticks = torch.arange(vmin, vmax, step).tolist()
        ticks = [round(tick, 2) for tick in ticks]
        if ticks[-1] != vmax:
            ticks.append(vmax)
        cbb = plt.colorbar(pad=0.06)
        cbb.set_ticks(ticks)
        if i == 0:
            ticks[0] = f'<{str(ticks[0])}'
        elif i == 1:
            ticks[-1] = f'>{str(ticks[-1])}'
        cbb.ax.set_yticklabels(ticks)
        for x in range(12):
            for y in range(12):
                txt = round(torch.rot90(img[0], -1, [0, 1])[x, y].item(), 2)
                if str(txt) == '-0.0':
                    txt = '0.00'
                elif str(txt) == '0.0':
                    txt = '0.00'
                elif len(str(txt)) == 3:
                    txt = str(txt) + '0'
                else:
                    txt = str(txt)

                plt.text(x - 0.25, y - 0.1, txt, color='black', fontsize='x-small')

        ax.set_xticks(torch.linspace(0, 11, 12))
        ax.set_xticklabels(transforms)
        ax.set_yticks(torch.linspace(0, 11, 12))
        pip.reverse()
        ax.set_yticklabels(pip)
        pip.reverse()
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        cba.set_label('Standard Deviation')
        plt.xlabel("Pipelines")
        plt.ylabel("Distortions")
        if dataset_name == 'DroneSegmentation':
            cbb.set_label('IoU')
            plt.savefig(os.path.join(path, f"{dataset_name}_{augmentation}_IoU.png"))
        else:
            cbb.set_label('Accuracy')
            plt.savefig(os.path.join(path, f"{dataset_name}_{augmentation}_accuracies.png"))


if __name__ == '__main__':

    if args.mode == 'ABMakeTable':
        ABMakeTable(args.dataset_name, args.augmentation, args.N_runs, args.download_model)
    elif args.mode == 'ABShowTable':
        ABShowTable(args.dataset_name, args.augmentation)
    elif args.mode == 'ABShowImages':
        ABclass = ABtesting(args.dataset_name, args.augmentation, args.dm_train,
                            args.s_train, args.dn_train, args.dm_test, args.s_test,
                            args.dn_test, args.N_runs, download_model=args.download_model)
        ABclass.ABShowImages()
    elif args.mode == 'ABShowAllImages':
        ABclass = ABtesting(args.dataset_name, args.augmentation, args.dm_train,
                            args.s_train, args.dn_train, args.dm_test, args.s_test,
                            args.dn_test, args.N_runs, download_model=args.download_model)
        ABclass.ABShowAllImages()
    elif args.mode == 'CMakeTable':
        CMakeTable(args.dataset_name, args.augmentation, args.severity, args.N_runs, args.download_model)
    elif args.mode == 'CShowTable':  # TODO test it
        CShowTable(args.dataset_name, args.augmentation, args.severity)
    elif args.mode == 'CShowImages':
        ABclass = ABtesting(args.dataset_name, args.augmentation, args.dm_train,
                            args.s_train, args.dn_train, args.dm_test, args.s_test,
                            args.dn_test, args.N_runs, args.severity, args.transform,
                            download_model=args.download_model)
        ABclass.CShowImages()
    elif args.mode == 'CShowAllImages':
        ABclass = ABtesting(args.dataset_name, args.augmentation, args.dm_train,
                            args.s_train, args.dn_train, args.dm_test, args.s_test,
                            args.dn_test, args.N_runs, args.severity, args.transform,
                            download_model=args.download_model)
        ABclass.CShowAllImages()
