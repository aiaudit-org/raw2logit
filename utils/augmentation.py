import random
import numpy as np

import torch
import torchvision.transforms as T


class RandomRotate90():  # Note: not the same as T.RandomRotation(90)
    def __call__(self, x):
        x = x.rot90(random.randint(0, 3), dims=(-1, -2))
        return x

    def __repr__(self):
        return self.__class__.__name__


class AddGaussianNoise():
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        # noise = torch.randn_like(x) * self.std
        # out = x + noise
        # debug(x)
        # debug(noise)
        # debug(out)
        return x + torch.randn_like(x) * self.std

    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'


def set_global_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


class ComposeState(T.Compose):
    def __init__(self, transforms):
        self.transforms = []
        self.mask_transforms = []

        for t in transforms:
            apply_for_mask = True
            if isinstance(t, tuple):
                t, apply_for_mask = t
            self.transforms.append(t)
            if apply_for_mask:
                self.mask_transforms.append(t)

        self.seed = None

    # @debug
    def __call__(self, x, retain_state=False, mask_transform=False):
        if self.seed is not None:   # retain previous state
            set_global_seed(self.seed)
        if retain_state:    # save state for next call
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            self.seed = None    # reset / ignore state

        transforms = self.transforms if not mask_transform else self.mask_transforms
        for t in transforms:
            x = t(x)
        return x


augmentation_weak = ComposeState([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    RandomRotate90(),
])


augmentation_strong = ComposeState([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    # (transform, apply_to_mask=True)
    (T.RandomApply([AddGaussianNoise(std=0.0005)], p=0.5), False),
    (T.RandomAdjustSharpness(0.5, p=0.5), False),
])


def get_augmentation(type):
    if type == 'none':
        return None
    if type == 'weak':
        return augmentation_weak
    if type == 'strong':
        return augmentation_strong


if __name__ == '__main__':
    import os
    if not os.path.exists('README.md'):
        os.chdir('..')

    # from utils.debug import debug
    from utils.dataset import get_dataset
    import matplotlib.pyplot as plt

    dataset = get_dataset('DS')  # drone segmentation
    img, mask = dataset[10]
    mask = (mask + 0.2) / 1.2

    plt.figure(figsize=(14, 8))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.suptitle('no augmentation')
    plt.show()

    from utils.base import np2torch, torch2np
    img, mask = np2torch(img), np2torch(mask)

    # from utils.augmentation import get_augmentation
    augmentation = get_augmentation('strong')

    set_global_seed(1)

    for i in range(1, 4):
        plt.figure(figsize=(14, 8))
        plt.subplot(121)
        plt.imshow(torch2np(augmentation(img.unsqueeze(0), retain_state=True)).squeeze())
        plt.subplot(122)
        plt.imshow(torch2np(augmentation(mask.unsqueeze(0), mask_transform=True)).squeeze())
        plt.suptitle(f'augmentation test {i}')
        plt.show()
