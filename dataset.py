import os
import shutil
import rawpy
import random
from PIL import Image
import tifffile as tiff
import zipfile

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit

if not os.path.exists('README.md'):  # set pwd to root
    os.chdir('..')

from utils.dataset_utils import split_img, list_images_in_dir, load_image
from utils.base import np2torch, torch2np, b2_download_folder

IMAGE_FILE_TYPES = ['dng', 'png', 'tif', 'tiff']


def get_dataset(name, I_ratio=1.0):
    # DroneDataset
    if name in ('DC', 'Drone', 'DroneClassification', 'DroneDatasetClassificationTiled'):
        return DroneDatasetClassificationTiled(I_ratio=I_ratio)
    if name in ('DS', 'DroneSegmentation', 'DroneDatasetSegmentationTiled'):
        return DroneDatasetSegmentationTiled(I_ratio=I_ratio)

    # MicroscopyDataset
    if name in ('M', 'Microscopy', 'MicroscopyDataset'):
        return MicroscopyDataset(I_ratio=I_ratio)

    # for testing
    if name in ('DSF', 'DroneDatasetSegmentationFull'):
        return DroneDatasetSegmentationFull(I_ratio=I_ratio)
    if name in ('MRGB', 'MicroscopyRGB', 'MicroscopyDatasetRGB'):
        return MicroscopyDatasetRGB(I_ratio=I_ratio)

    raise ValueError(name)


class ImageFolderDataset(Dataset):
    """Creates a dataset of images in img_dir and corresponding masks in mask_dir.
    Corresponding mask files need to contain the filename of the image.
    Files are expected to be of the same filetype.

    Args:
        img_dir (str): path to image folder
        mask_dir (str): path to mask folder
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """

    task = 'classification'

    def __init__(self, img_dir, labels, transform=None, bits=1):

        self.img_dir = img_dir
        self.labels = labels

        self.images = list_images_in_dir(img_dir)

        assert len(self.images) == len(self.labels)

        self.transform = transform
        self.bits = bits

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{len(self.images)}]"
        for n, (img, label) in enumerate(zip(self.images, self.labels)):
            rep += f'\nimage: {img}\tlabel: {label}'
            if n > 10:
                rep += '\n...'
                break
        return rep

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        label = self.labels[idx]

        img = load_image(self.images[idx])
        img = img / (2**self.bits - 1)
        if self.transform is not None:
            img = self.transform(img)

        if len(img.shape) == 2:
            assert img.shape == (256, 256), f"Invalid size for {self.images[idx]}"
        else:
            assert img.shape == (3, 256, 256), f"Invalid size for {self.images[idx]}"

        return img, label


class ImageFolderDatasetSegmentation(Dataset):
    """Creates a dataset of images in `img_dir` and corresponding masks in `mask_dir`.
    Corresponding mask files need to contain the filename of the image.
    Files are expected to be of the same filetype.

    Args:
        img_dir (str): path to image folder
        mask_dir (str): path to mask folder
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """

    task = 'segmentation'

    def __init__(self, img_dir, mask_dir, transform=None, bits=1):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = list_images_in_dir(img_dir)
        self.masks = list_images_in_dir(mask_dir)

        check_image_folder_consistency(self.images, self.masks)

        self.transform = transform
        self.bits = bits

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDatasetSegmentation[{len(self.images)}]"
        for n, (img, mask) in enumerate(zip(self.images, self.masks)):
            rep += f'\nimage: {img}\tmask: {mask}'
            if n > 10:
                rep += '\n...'
                break
        return rep

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = load_image(self.images[idx])
        mask = load_image(self.masks[idx])

        img = img / (2**self.bits - 1)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask


class MultiIntensity(Dataset):
    """Wrap datasets with different intesities

    Args:
        datasets (list): list of datasets to wrap
    """

    def __init__(self, datasets):
        self.dataset = datasets[0]

        for d in range(1, len(datasets)):
            self.dataset.images = self.dataset.images + datasets[d].images
            self.dataset.labels = self.dataset.labels + datasets[d].labels

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f"Subset [{len(self.dataset)}] of " + repr(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class Subset(Dataset):
    """Define a subset of a dataset by only selecting given indices.

    Args:
        dataset (Dataset): full dataset
        indices (list): subset indices
    """

    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"Subset [{len(self)}] of " + repr(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class DroneDatasetSegmentationFull(ImageFolderDatasetSegmentation):
    """Dataset consisting of full-sized numpy images and masks. Images are normalized to range [0, 1].
    """

    black_level = [0.0625, 0.0626, 0.0625, 0.0626]
    white_balance = [2.86653646, 1., 1.73079425]
    colour_matrix = [1.50768983, -0.33571374, -0.17197604, -0.23048614,
                     1.70698738, -0.47650126, -0.03119153, -0.32803956, 1.35923111]
    camera_parameters = black_level, white_balance, colour_matrix

    def __init__(self, I_ratio=1.0, transform=None, force_download=False, bits=16):

        assert I_ratio in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

        img_dir = f'data/drone/images_full/raw_scale{int(I_ratio*100):03d}'
        mask_dir = 'data/drone/masks_full'

        download_drone_dataset(force_download)  # XXX: zip files and add checksum? date?

        super().__init__(img_dir=img_dir, mask_dir=mask_dir, transform=transform, bits=bits)


class DroneDatasetSegmentationTiled(ImageFolderDatasetSegmentation):
    """Dataset consisting of tiled numpy images and masks. Images are in range [0, 1]
    Args:
        tile_size (int, optional): size of the tiled images. Defaults to 256.
    """

    camera_parameters = DroneDatasetSegmentationFull.camera_parameters

    def __init__(self, I_ratio=1.0, transform=None):

        tile_size = 256

        img_dir = f'data/drone/images_tiles_{tile_size}/raw_scale{int(I_ratio*100):03d}'
        mask_dir = f'data/drone/masks_tiles_{tile_size}'

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            dataset_full = DroneDatasetSegmentationFull(I_ratio=I_ratio, bits=1)
            print("tiling dataset..")
            create_tiles_dataset(dataset_full, img_dir, mask_dir, tile_size=tile_size)

        super().__init__(img_dir=img_dir, mask_dir=mask_dir, transform=transform, bits=16)


class DroneDatasetClassificationTiled(ImageFolderDataset):

    camera_parameters = DroneDatasetSegmentationFull.camera_parameters

    def __init__(self, I_ratio=1.0, transform=None):

        random_state = 72
        tile_size = 256
        thr = 0.01

        img_dir = f'data/drone/classification/images_tiles_{tile_size}/raw_scale{int(I_ratio*100):03d}_thr_{thr}'
        mask_dir = f'data/drone/classification/masks_tiles_{tile_size}_thr_{thr}'
        df_path = f'data/drone/classification/dataset_tiles_{tile_size}_{random_state}_{thr}.csv'

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            dataset_full = DroneDatasetSegmentationFull(I_ratio=I_ratio, bits=1)
            print("tiling dataset..")
            create_tiles_dataset_binary(dataset_full, img_dir, mask_dir, random_state, thr, tile_size=tile_size)

        self.classes = ['car', 'no car']
        self.df = pd.read_csv(df_path)
        labels = self.df['label'].to_list()

        super().__init__(img_dir=img_dir, labels=labels, transform=transform, bits=16)

        images, class_labels = read_label_csv(self.df)
        self.images = [os.path.join(self.img_dir, image) for image in images]
        self.labels = class_labels


class MicroscopyDataset(ImageFolderDataset):
    """MicroscopyDataset raw images

    Args:
        I_ratio (float): Original image rescaled by this factor, possible values [0.01,0.05,0.1,0.25,0.5,0.75,1.0]
        raw (bool): Select rgb dataset or raw dataset
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """

    black_level = [9.834368023181512e-06, 9.834368023181512e-06, 9.834368023181512e-06, 9.834368023181512e-06]
    white_balance = [-0.6567, 1.9673, 3.5304]
    colour_matrix = [-2.0338, 0.0933, 0.4157, -0.0286, 2.6464, -0.0574, -0.5516, -0.0947, 2.9308]

    camera_parameters = black_level, white_balance, colour_matrix

    dataset_mean = [0.91, 0.84, 0.94]
    dataset_std = [0.08, 0.12, 0.05]

    def __init__(self, I_ratio=1.0, transform=None, bits=16, force_download=False):

        assert I_ratio in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

        download_microscopy_dataset(force_download=force_download)

        self.img_dir = f'data/microscopy/images/raw_scale{int(I_ratio*100):03d}'
        self.transform = transform
        self.bits = bits

        self.label_file = 'data/microscopy/labels/Ma190c_annotations.dat'

        self.valid_classes = ['BAS', 'EBO', 'EOS', 'KSC', 'LYA', 'LYT', 'MMZ', 'MOB',
                              'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMB', 'PMO', 'UNC']

        self.invalid_files = ['Ma190c_lame3_zone13_composite_Mcropped_2.tiff', ]

        images, class_labels = read_label_file(self.label_file)

        # filter classes with low appearance
        self.valid_classes = [class_label for class_label in self.valid_classes
                              if class_labels.count(class_label) > 4]

        # remove invalid classes and invalid files from (images, class_labels)
        images, class_labels = list(zip(*[
            (image, class_label)
            for image, class_label in zip(images, class_labels)
            if class_label in self.valid_classes and image not in self.invalid_files
        ]))

        self.classes = list(sorted({*class_labels}))

        # store full path
        self.images = [os.path.join(self.img_dir, image) for image in images]

        # reindex labels
        self.labels = [self.classes.index(class_label) for class_label in class_labels]


class MicroscopyDatasetRGB(MicroscopyDataset):
    """MicroscopyDataset RGB images

    Args:
        I_ratio (float): Original image rescaled by this factor, possible values [0.01,0.05,0.1,0.25,0.5,0.75,1.0]
        raw (bool): Select rgb dataset or raw dataset
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """
    camera_parameters = None

    dataset_mean = None
    dataset_std = None

    def __init__(self, I_ratio=1.0, transform=None, bits=16, force_download=False):
        super().__init__(I_ratio=I_ratio, transform=transform, bits=bits, force_download=force_download)
        self.images = [image.replace('raw', 'rgb') for image in self.images]  # XXX: hack


def read_label_file(label_file_path):

    images = []
    class_labels = []

    with open(label_file_path, "rb") as data:
        for line in data:
            file_name, class_label = line.decode("utf-8").split()
            image = file_name + '.tiff'
            images.append(image)
            class_labels.append(class_label)

    return images, class_labels


def read_label_csv(df):

    images = []
    class_labels = []

    for file_name, label in zip(df['file name'], df['label']):
        image = file_name + '.tif'
        images.append(image)
        class_labels.append(int(label))
    return images, class_labels


def download_drone_dataset(force_download):
    b2_download_folder('drone/images', 'data/drone/images_full', force_download=force_download)
    b2_download_folder('drone/masks', 'data/drone/masks_full', force_download=force_download)
    unzip_drone_images()


def download_microscopy_dataset(force_download):
    b2_download_folder('Data histopathology/WhiteCellsImages',
                       'data/microscopy/images', force_download=force_download)
    b2_download_folder('Data histopathology/WhiteCellsLabels',
                       'data/microscopy/labels', force_download=force_download)
    unzip_microscopy_images()


def unzip_microscopy_images():

    if os.path.isfile('data/microscopy/labels/.bzEmpty'):
        os.remove('data/microscopy/labels/.bzEmpty')

    for file in os.listdir('data/microscopy/images'):
        if file.endswith(".zip"):
            zip = zipfile.ZipFile(os.path.join('data/microscopy/images', file))
            zip.extractall('data/microscopy/images')
            os.remove(os.path.join('data/microscopy/images', file))


def unzip_drone_images():

    if os.path.isfile('data/drone/masks_full/.bzEmpty'):
        os.remove('data/drone/masks_full/.bzEmpty')

    for file in os.listdir('data/drone/images_full'):
        if file.endswith(".zip"):
            zip = zipfile.ZipFile(os.path.join('data/drone/images_full', file))
            zip.extractall('data/drone/images_full')
            os.remove(os.path.join('data/drone/images_full', file))


def create_tiles_dataset(dataset, img_dir, mask_dir, tile_size=256):
    for folder in [img_dir, mask_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    for n, (img, mask) in enumerate(dataset):
        tiled_img = split_img(img, ROIs=(tile_size, tile_size), step=(tile_size, tile_size))
        tiled_mask = split_img(mask, ROIs=(tile_size, tile_size), step=(tile_size, tile_size))
        tiled_img, tiled_mask = class_detection(tiled_img, tiled_mask)  # Remove images without cars in it
        for i, (sub_img, sub_mask) in enumerate(zip(tiled_img, tiled_mask)):
            tile_id = f"{n:02d}_{i:05d}"
            Image.fromarray(sub_img).save(os.path.join(img_dir, tile_id + '.tif'))
            Image.fromarray(sub_mask > 0).save(os.path.join(mask_dir, tile_id + '.png'))


def create_tiles_dataset_binary(dataset, img_dir, mask_dir, random_state, thr, tile_size=256):

    for folder in [img_dir, mask_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    ids = []
    labels = []

    for n, (img, mask) in enumerate(dataset):
        tiled_img = split_img(img, ROIs=(tile_size, tile_size), step=(tile_size, tile_size))
        tiled_mask = split_img(mask, ROIs=(tile_size, tile_size), step=(tile_size, tile_size))

        X_with, X_without, Y_with, Y_without = binary_class_detection(
            tiled_img, tiled_mask, random_state, thr)  # creates balanced arrays with class and without class

        for i, (sub_X_with, sub_Y_with) in enumerate(zip(X_with, Y_with)):
            tile_id = f"{n:02d}_{i:05d}"
            ids.append(tile_id)
            labels.append(0)
            Image.fromarray(sub_X_with).save(os.path.join(img_dir, tile_id + '.tif'))
            Image.fromarray(sub_Y_with > 0).save(os.path.join(mask_dir, tile_id + '.png'))
        for j, (sub_X_without, sub_Y_without) in enumerate(zip(X_without, Y_without)):
            tile_id = f"{n:02d}_{i+1+j:05d}"
            ids.append(tile_id)
            labels.append(1)
            Image.fromarray(sub_X_without).save(os.path.join(img_dir, tile_id + '.tif'))
            Image.fromarray(sub_Y_without > 0).save(os.path.join(mask_dir, tile_id + '.png'))
           # Image.fromarray(sub_mask).save(os.path.join(mask_dir, tile_id + '.png'))

    df = pd.DataFrame({'file name': ids, 'label': labels})

    df_loc = f'data/drone/classification/dataset_tiles_{tile_size}_{random_state}_{thr}.csv'
    df.to_csv(df_loc)

    return


def class_detection(X, Y):
    """Split dataset in images which has the class in the target

       Args:
            X (ndarray): input image
            Y (ndarray): target with segmentation map (images with {0,1} values where it is 1 when there is the class)
       Returns:
           X_with_class (ndarray): input regions with the selected class 
           Y_with_class (ndarray): target regions with the selected class 
           X_without_class (ndarray): input regions without the selected class 
           Y_without_class (ndarray): target regions without the selected class 
    """

    with_class = []
    without_class = []
    for i, img in enumerate(Y):
        if img.mean() == 0:
            without_class.append(i)
        else:
            with_class.append(i)

    X_with_class = np.delete(X, without_class, 0)
    Y_with_class = np.delete(Y, without_class, 0)

    return X_with_class, Y_with_class


def binary_class_detection(X, Y, random_seed, thr):
    """Splits subimages in subimages with the selected class and without the selected class by calculating the mean of the submasks; subimages with 0 < submask.mean()<=thr are disregared



       Args:
            X (ndarray): input image
            Y (ndarray): target with segmentation map (images with {0,1} values where it is 1 when there is the class)
            thr (flaot): sub images are not considered if 0 < sub_target.mean() <= thr 
            balanced (bool): number of returned sub images is equal for both classes if true 
            random_seed (None or int): selection of sub images in class with more elements according to random_seed if balanced
       Returns:
           X_with_class (ndarray): input regions with the selected class 
           Y_with_class (ndarray): target regions with the selected class 
           X_without_class (ndarray): input regions without the selected class 
           Y_without_class (ndarray): target regions without the selected class 
    """

    with_class = []
    without_class = []
    no_class = []

    for i, img in enumerate(Y):
        m = img.mean()
        if m == 0:
            without_class.append(i)
        else:
            if m > thr:
                with_class.append(i)
            else:
                no_class.append(i)

    N = len(with_class)
    M = len(without_class)
    random.seed(random_seed)
    if N <= M:
        random.shuffle(without_class)
        with_class.extend(without_class[:M - N])
    else:
        random.shuffle(with_class)
        without_class.extend(with_class[:N - M])

    X_with_class = np.delete(X, without_class + no_class, 0)
    X_without_class = np.delete(X, with_class + no_class, 0)
    Y_with_class = np.delete(Y, without_class + no_class, 0)
    Y_without_class = np.delete(Y, with_class + no_class, 0)

    return X_with_class, X_without_class, Y_with_class, Y_without_class


def make_dataloader(dataset, batch_size, shuffle=True):

    X, Y = dataset

    X, Y = np2torch(X), np2torch(Y)

    dataset = TensorDataset(X, Y)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset


def check_image_folder_consistency(images, masks):
    file_type_images = images[0].split('.')[-1].lower()
    file_type_masks = masks[0].split('.')[-1].lower()
    assert len(images) == len(masks), "images / masks length mismatch"
    for img_file, mask_file in zip(images, masks):
        img_name = img_file.split('/')[-1].split('.')[0]
        assert img_name in mask_file, f"image {img_file} corresponds to {mask_file}?"
        assert img_file.split('.')[-1].lower() == file_type_images, \
            f"image file {img_file} file type mismatch. Shoule be: {file_type_images}"
        assert mask_file.split('.')[-1].lower() == file_type_masks, \
            f"image file {mask_file} file type mismatch. Should be: {file_type_masks}"
