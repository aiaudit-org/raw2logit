import os
from collections import defaultdict

import torch
import torch.optim
from torchvision.models import resnet18
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

import pytorch_lightning as pl

import mlflow.pytorch


def resnet_model(model=resnet18, pretrained=True, in_channels=3, fc_out_features=2):
    resnet = model(pretrained=pretrained)
    # if not pretrained:  # TODO: add case for in_channels=4
    #     resnet.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = torch.nn.Linear(in_features=512, out_features=fc_out_features, bias=True)
    return resnet


class LitModel(pl.LightningModule):

    def __init__(self,
                 classifier,
                 loss,
                 lr=1e-3,
                 weight_decay=0,
                 loss_aux=None,
                 adv_training=False,
                 adv_parameters='all',
                 metrics=None,
                 processor=None,
                 augmentation=None,
                 is_segmentation_task=False,
                 augmentation_on_eval=False,
                 metrics_on_training=True,
                 freeze_classifier=False,
                 freeze_processor=False,
                 ):
        super().__init__()

        self.classifier = classifier
        self.processor = processor

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss
        self.loss_aux_fn = loss_aux
        self.adv_training = adv_training
        self.metrics = metrics
        self.augmentation = augmentation
        self.is_segmentation_task = is_segmentation_task
        self.augmentation_on_eval = augmentation_on_eval
        self.metrics_on_training = metrics_on_training

        self.freeze_classifier = freeze_classifier
        self.freeze_processor = freeze_processor

        self.unfreeze()
        if freeze_classifier:
            pl.LightningModule.freeze(self.classifier)
        if freeze_processor:
            pl.LightningModule.freeze(self.processor)

        if adv_training and adv_parameters != 'all':
            if adv_parameters != 'all':
                pl.LightningModule.freeze(self.processor)
                for name, p in self.processor.named_parameters():
                    if adv_parameters in name:
                        p.requires_grad = True

    def forward(self, x):
        x = self.processor(x)
        apply_augmentation_step = self.training or self.augmentation_on_eval
        if self.augmentation is not None and apply_augmentation_step:
            x = self.augmentation(x, retain_state=self.is_segmentation_task)
        x = self.classifier(x)
        return x

    def update_step(self, batch, step_name):
        x, y = batch
        # debug(self.processor)
        # debug(self.processor.parameters())
        # debug.pause()
        # print('type', type(self.processor).__name__)

        logits = self(x)

        apply_augmentation_mask = self.is_segmentation_task and (self.training or self.augmentation_on_eval)
        if self.augmentation is not None and apply_augmentation_mask:
            y = self.augmentation(y, mask_transform=True).contiguous()

        loss = self.loss_fn(logits, y)

        if self.loss_aux_fn is not None:
            loss_aux = self.loss_aux_fn(x)
            loss += loss_aux

        self.log(f'{step_name}_loss', loss, on_step=False, on_epoch=True)
        if self.loss_aux_fn is not None:
            self.log(f'{step_name}_loss_aux', loss_aux, on_step=False, on_epoch=True)

        if self.is_segmentation_task:
            y_hat = F.logsigmoid(logits).exp().squeeze()
        else:
            y_hat = torch.argmax(logits, dim=1)

        if self.metrics is not None:
            for metric in self.metrics:
                metric_name = metric.__name__ if hasattr(metric, '__name__') else type(metric).__name__
                if metric_name == 'accuracy' or not self.training or self.metrics_on_training:
                    m = metric(y_hat.cpu().detach(), y.cpu())
                    self.log(f'{step_name}_{metric_name}', m, on_step=False, on_epoch=True,
                             prog_bar=self.training or metric_name == 'accuracy')
                if metric_name == 'iou_score' or not self.training or self.metrics_on_training:
                    m = metric(y_hat.cpu().detach(), y.cpu())
                    self.log(f'{step_name}_{metric_name}', m, on_step=False, on_epoch=True,
                             prog_bar=self.training or metric_name == 'iou_score')
                elif metric_name == 'accuracy' or not self.training or self.metrics_on_training:
                    m = metric(y_hat.cpu().detach(), y.cpu())
                    self.log(f'{step_name}_{metric_name}', m, on_step=False, on_epoch=True,
                             prog_bar=self.training or metric_name == 'accuracy')

        return loss

    def training_step(self, batch, batch_idx):
        return self.update_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.update_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.update_step(batch, 'test')

    def train(self, mode=True):
        self.training = mode

        # don't update batchnorm in adversarial training
        self.processor.train(mode=mode and not self.freeze_processor and not self.adv_training)
        self.classifier.train(mode=mode and not self.freeze_classifier)
        return self

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        return self.optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num')
        return items


class TrackImagesCallback(pl.callbacks.base.Callback):
    def __init__(self, data_loader, reference_processor=None, track_every_epoch=False, track_processing=True, track_gradients=True, track_predictions=True, save_tensors=True):
        super().__init__()
        self.data_loader = data_loader

        self.track_every_epoch = track_every_epoch

        self.track_processing = track_processing
        self.track_gradients = track_gradients
        self.track_predictions = track_predictions
        self.save_tensors = save_tensors

        self.reference_processor = reference_processor

    def callback_track_images(self, model, save_loc):
        track_images(model,
                     self.data_loader,
                     reference_processor=self.reference_processor,
                     track_processing=self.track_processing,
                     track_gradients=self.track_gradients,
                     track_predictions=self.track_predictions,
                     save_tensors=self.save_tensors,
                     save_loc=save_loc,
                     )

    def on_fit_end(self, trainer, pl_module):
        if not self.track_every_epoch:
            save_loc = 'results'
            self.callback_track_images(trainer.model, save_loc)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if self.track_every_epoch:
            save_loc = f'results/epoch_{trainer.current_epoch + 1:04d}'
            self.callback_track_images(trainer.model, save_loc)


from utils.debug import debug


# @debug
def log_tensor(batch, path, save_tensors=True, nrow=8):
    if save_tensors:
        torch.save(batch, path)
        mlflow.log_artifact(path, os.path.dirname(path))

    img_path = path.replace('.pt', '.png')
    split = img_path.split('/')
    img_path = '/'.join(split[:-1]) + '/img_' + split[-1]  # insert 'img_'; make it easier to find in mlflow

    grid = make_grid(batch, nrow=nrow).squeeze()
    save_image(grid, img_path)
    mlflow.log_artifact(img_path, os.path.dirname(path))


def track_images(model, data_loader, reference_processor=None, track_processing=True, track_gradients=True, track_predictions=True, save_tensors=True, save_loc='results'):

    device = model.device
    processor = model.processor
    classifier = model.classifier

    if not hasattr(processor, 'stages'):    # 'static' or 'none' pipeline
        return

    os.makedirs(save_loc, exist_ok=True)

    # TODO: implement track_predictions

    # inputs_full = []
    labels_full = []
    logits_full = []
    stages_full = defaultdict(list)
    grads_full = defaultdict(list)
    diffs_full = defaultdict(list)

    track_differences = reference_processor is not None

    for inputs, labels in data_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        processed_rgb = processor(inputs)

        if track_differences:
            # debug(processor)
            processed_rgb_ref = reference_processor(inputs)

        if track_gradients or track_predictions:
            logits = classifier(processed_rgb)

            # NOTE: should zero grads for good measure
            loss = model.loss_fn(logits, labels)
            loss.backward()

            if track_predictions:
                labels_full.append(labels.cpu().detach())
                logits_full.append(logits.cpu().detach())
        # inputs_full.append(inputs.cpu().detach())

        for stage, batch in processor.stages.items():
            stages_full[stage].append(batch.cpu().detach())
            if track_differences:
                diffs_full[stage].append((reference_processor.stages[stage] - batch).cpu().detach())
            if track_gradients:
                grads_full[stage].append(batch.grad.cpu().detach())

    with torch.no_grad():

        stages = stages_full
        grads = grads_full
        diffs = diffs_full

        if track_processing:
            for stage, batch in stages.items():
                stages[stage] = torch.cat(batch)

        if track_differences:
            for stage, batch in diffs.items():
                diffs[stage] = torch.cat(batch)

        if track_gradients:
            for stage, batch in grads.items():
                grads[stage] = torch.cat(batch)

        for stage_nr, stage_name in enumerate(stages):
            if track_processing:
                batch = stages[stage_name]
                log_tensor(batch, os.path.join(save_loc, f'processing_{stage_nr}_{stage_name}.pt'), save_tensors)

            if track_differences:
                batch = diffs[stage_name]
                log_tensor(batch, os.path.join(save_loc, f'diffs_{stage_nr}_{stage_name}.pt'), False)

            if track_gradients:
                batch_grad = grads[stage_name]
                batch_grad = batch_grad.abs()
                batch_grad = (batch_grad - batch_grad.min()) / (batch_grad.max() - batch_grad.min())
                log_tensor(batch_grad, os.path.join(
                    save_loc, f'gradients_{stage_nr}_{stage_name}.pt'), save_tensors)

        # inputs = torch.cat(inputs_full)

        if track_predictions:  # and model.is_segmentation_task:
            labels = torch.cat(labels_full)
            logits = torch.cat(logits_full)
            masks = labels.unsqueeze(1)
            predictions = logits  # torch.sigmoid(logits).unsqueeze(1)
            #mask_vis = torch.cat((masks, predictions, masks * predictions), dim=1)
            #log_tensor(mask_vis, os.path.join(save_loc, f'masks.pt'), save_tensors)
            log_tensor(masks, os.path.join(save_loc, f'targets.pt'), save_tensors)
            log_tensor(predictions, os.path.join(save_loc, f'preds.pt'), save_tensors)
