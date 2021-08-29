"""
Raw Image Pipeline
"""
__author__ = "Marco Aversa"

import numpy as np

from rawpy import *  # XXX: no * imports!
from scipy import ndimage
from scipy import fftpack
from scipy.signal import convolve2d

from skimage.filters import unsharp_mask
from skimage.color import rgb2yuv, yuv2rgb, rgb2hsv, hsv2rgb
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman, denoise_nl_means, denoise_bilateral, denoise_wavelet, estimate_sigma

import matplotlib.pyplot as plt

from colour_demosaicing import (demosaicing_CFA_Bayer_bilinear,
                                demosaicing_CFA_Bayer_Malvar2004,
                                demosaicing_CFA_Bayer_Menon2007)

import torch
import numpy as np

from utils.dataset import Subset
from torch.utils.data import DataLoader

from colour_demosaicing import (demosaicing_CFA_Bayer_bilinear,
                                demosaicing_CFA_Bayer_Malvar2004,
                                demosaicing_CFA_Bayer_Menon2007)

import matplotlib.pyplot as plt


class RawProcessingPipeline(object):

    """Applies the raw-processing pipeline from pipeline.py"""

    def __init__(self, camera_parameters, debayer='bilinear', sharpening='unsharp_masking', denoising='gaussian'):
        '''
        Args:
            camera_parameters (tuple): (black_level, white_balance, colour_matrix)
            debayer (str): specifies the algorithm used as debayer; choose from {'bilinear','malvar2004','menon2007'}
            sharpening (str): specifies the algorithm used for sharpening; choose from {'sharpening_filter','unsharp_masking'}
            denoising (str): specifies the algorithm used for denoising; choose from choose from {'gaussian_denoising','median_denoising','fft_denoising'}        
        '''

        self.camera_parameters = camera_parameters

        self.debayer = debayer
        self.sharpening = sharpening
        self.denoising = denoising

    def __call__(self, img):
        """
        Args:
            img (ndarry of dtype float.32): image of size (H,W)
        return:
            img (tensor of dtype float): image of size (3,H,W)
        """
        black_level, white_balance, colour_matrix = self.camera_parameters
        img = processing(img, black_level, white_balance, colour_matrix,
                         debayer=self.debayer, sharpening=self.sharpening, denoising=self.denoising)
        img = img.transpose(2, 0, 1)

        return torch.Tensor(img)


def processing(img, black_level, white_balance, colour_matrix, debayer="bilinear", sharpening="unsharp_masking",
               sharp_radius=1.0, sharp_amount=1.0, denoising="median_filter", median_kernel_size=3,
               gaussian_sigma=0.5, fft_fraction=0.3, weight_chambolle=0.01, weight_bregman=100,
               sigma_bilateral=0.6, gamma=2.2, bits=16):
    """Apply pipeline on a raw image

       Args:
           rawImg (ndarray): raw image
           debayer (str): debayer algorithm
           white_balance (None, ndarray): white balance array (if None it will take the default camera white balance array)
           colour_matrix (None, ndarray): colour matrix (if None it will take the default camera colour matrix) - Size: 3x3
           gamma (float): exponent for the non linear gamma correction.

       Returns:
           img (ndarray): post-processed image

    """

    # Remove Black Level
    img = remove_blacklv(img, black_level)

    # Apply demosaicing - We don't have access to these 3 functions
    if debayer == "bilinear":
        img = demosaicing_CFA_Bayer_bilinear(img)
    if debayer == "malvar2004":
        img = demosaicing_CFA_Bayer_Malvar2004(img)
    if debayer == "menon2007":
        img = demosaicing_CFA_Bayer_Menon2007(img)

    # White Balance Correction

    # Sunny images white balance array -> 2<r<2.8, g=1.0, 1.3<b<1.6
    # Tungsten images white balance array -> 1.3<r<1.7, g=1.0, 2.2<b<2.8
    # Shade images white balance array -> 2.4<r<3.2, g=1.0, 1.1<b<1.3

    img = wb_correction(img, white_balance)

    # Colour Correction
    img = colour_correction(img, colour_matrix)

    # Sharpening
    if sharpening == "sharpening_filter":  # Fixed sharpening
        img = sharpening_filter(img)
    if sharpening == "unsharp_masking":  # Higher is radius and amount, higher is the sharpening
        img = unsharp_masking(img, radius=sharp_radius, amount=sharp_amount, multichannel=True)

    # Denoising
    if denoising == "median_denoising":
        img = median_denoising(img, size=median_kernel_size)
    if denoising == "gaussian_denoising":
        img = gaussian_denoising(img, sigma=gaussian_sigma)
    if denoising == "fft_denoising":  # fft_fraction = [0.0001,0.5]
        img = fft_denoising(img, keep_fraction=fft_fraction, row_cut=False, column_cut=True)

    # We don't have access to these 3 functions
    if denoising == "tv_chambolle":  # lower is weight, less is the denoising
        img = denoise_tv_chambolle(img, weight=weight_chambolle, eps=0.0002, n_iter_max=200, multichannel=True)
    if denoising == "tv_bregman":  # lower is weight, more is the denoising
        img = denoise_tv_bregman(img, weight=weight_bregman, max_iter=100,
                                 eps=0.001, isotropic=True, multichannel=True)
#     if denoising == "wavelet":
#         img = denoise_wavelet(img.copy(), sigma=None, wavelet='db1', mode='soft', wavelet_levels=None, multichannel=True,
#                 convert2ycbcr=False, method='BayesShrink', rescale_sigma=True)
    if denoising == "bilateral":  # higher is sigma_spatial, more is the denoising
        img = denoise_bilateral(img, win_size=None, sigma_color=None, sigma_spatial=sigma_bilateral,
                                bins=10000, mode='constant', cval=0, multichannel=True)

    # Gamma Correction
    img = np.clip(img, 0, 1)
    img = adjust_gamma(img, gamma=gamma)

    return img


def get_camera_parameters(rawpyImg):
    black_level = rawpyImg.black_level_per_channel
    white_balance = rawpyImg.camera_whitebalance[:3]
    colour_matrix = rawpyImg.color_matrix[:, :3].flatten().tolist()

    return black_level, white_balance, colour_matrix


def remove_blacklv(rawImg, black_level):
    rawImg[0::2, 0::2] -= black_level[0]  # R
    rawImg[0::2, 1::2] -= black_level[1]  # G
    rawImg[1::2, 0::2] -= black_level[2]  # G
    rawImg[1::2, 1::2] -= black_level[3]  # B

    return rawImg


def wb_correction(img, white_balance):
    return img * white_balance


def colour_correction(img, colour_matrix):
    colour_matrix = np.array(colour_matrix).reshape(3, 3)
    return np.einsum('ijk,lk->ijl', img, colour_matrix)


def unsharp_masking(img, radius=1.0, amount=1.0,
                    multichannel=False, preserve_range=True):

    img = rgb2yuv(img)
    img[:, :, 0] = unsharp_mask(img[:, :, 0], radius=radius, amount=amount,
                                multichannel=multichannel, preserve_range=preserve_range)
    img = yuv2rgb(img)
    return img


def sharpening_filter(image, iterations=1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])):

    # https://towardsdatascience.com/image-processing-with-python-blurring-and-sharpening-for-beginners-3bcebec0583a

    img_yuv = rgb2yuv(image)

    for i in range(iterations):
        img_yuv[:, :, 0] = convolve2d(img_yuv[:, :, 0], kernel, 'same', boundary='fill', fillvalue=0)

    final_image = yuv2rgb(img_yuv)

    return final_image


def median_denoising(img, size=3):

    img = rgb2yuv(img)
    img[:, :, 0] = ndimage.median_filter(img[:, :, 0], size)
    img = yuv2rgb(img)

    return img


def gaussian_denoising(img, sigma=0.5):

    img = rgb2yuv(img)
    img[:, :, 0] = ndimage.gaussian_filter(img[:, :, 0], sigma)
    img = yuv2rgb(img)

    return img


def fft_denoising(img, keep_fraction=0.3, row_cut=False, column_cut=True):
    """ keep_fraction = 0.5 --> same image as input
        keep_fraction --> 0 --> remove all details """
#   http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

    im_fft = fftpack.fft2(img)

    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft

    # Set r and c to be the number of rows and columns of the array.
    r, c, _ = im_fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and r*(1-keep_fraction):
    if row_cut == True:
        im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0

    # Similarly with the columns:
    if column_cut == True:
        im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0

    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(im_fft2).real

    return im_new


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    img = (img ** invGamma)
    return img


def show_img(img, title="no_title", size=12, histo=True, bins=300, bits=16, x_range=-1):
    """Plot image and its histogram

       Args:
           img (ndarray): image to plot
           title (str): title of the plot
           histo (bool): True - Plot histrograms per channel of the image. False - Plot the curve of histogram in a continue way
           bins (int): number of bins of the histogram
           size (int): figure size
           bits (int): number of bits per pixel in the ndarray
           x_range (list): maximum x range of the histogram (if -1 it will be take all x values)
    """
    shape = img.shape

    fig = plt.figure(figsize=(size, size))

    # show original image
    fig.add_subplot(221)
    if len(shape) > 2 and img.max() > 255:
        img_to_show = (img.copy() * 255. / (2**bits - 1)).astype(int)
    else:
        img_to_show = img.copy().astype(int)
    plt.imshow(img_to_show)
    if title != "no_title":
        plt.title(title)

    fig.add_subplot(222)

    if len(shape) > 2:
        if histo == True:
            plt.hist(img[:, :, 0].flatten(), bins=bins, label="Channel1", color="red", alpha=0.5)
            plt.hist(img[:, :, 1].flatten(), bins=bins, label="Channel2", color="green", alpha=0.5)
            plt.hist(img[:, :, 2].flatten(), bins=bins, label="Channel3", color="blue", alpha=0.5)
            if x_range != -1:
                plt.xlim([x_range[0], x_range[1]])
        else:
            h1, b1 = np.histogram(img[:, :, 0].flatten(), bins=bins)
            h2, b2 = np.histogram(img[:, :, 1].flatten(), bins=bins)
            h3, b3 = np.histogram(img[:, :, 2].flatten(), bins=bins)
            plt.plot(b1[:-1], h1, label="Channel1", color="red", alpha=0.5)
            plt.plot(b2[:-1], h2, label="Channel2", color="green", alpha=0.5)
            plt.plot(b3[:-1], h3, label="Channel3", color="blue", alpha=0.5)

        plt.legend()
    else:
        if histo == True:
            plt.hist(img.flatten(), bins=bins)
            if x_range != -1:
                plt.xlim([x_range[0], x_range[1]])
        else:
            h, b = np.histogram(img.flatten(), bins=bins)
            plt.plot(b[:-1], h)

    plt.xlabel("Intensities")
    plt.ylabel("Counts")

    plt.show()


def get_statistics(dataset, train_indices, transform=None):
    """Calculates the mean and the standard deviation of a given sub train set of dataset

    Args:
        dataset (Subset of DroneDataset): 
        train_indices (tensor): indicies correponding to a subset of the dataset
        transform (Compose): list of transformations compatible with Compose to be applied before calculations
    return:
        mean (tensor of dtype float): size (C,1,1)
        std (tensor of dtype float): size (C,1,1)
    """

    trainset = Subset(dataset, indices=train_indices, transform=transform)
    dataloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    dataiter = iter(dataloader)

    images, labels = dataiter.next()

    if len(images.shape) == 3:
        mean, std = torch.mean(images, axis=(0, 1, 2)), torch.std(images, axis=(0, 1, 2))
        return mean, std
    else:
        mean, std = torch.mean(images, axis=(0, 2, 3))[:, None, None], torch.std(images, axis=(0, 2, 3))[:, None, None]
        return mean, std
