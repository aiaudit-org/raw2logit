import os
from numpy.lib.function_base import interp
import torch
import torch.nn as nn
if not os.path.exists('README.md'):
    os.chdir('..')

from processing.pipeline_numpy import processing as default_processing
from utils.base import np2torch, torch2np

import segmentation_models_pytorch as smp

K_G = torch.Tensor([[0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0]]) / 4

K_RB = torch.Tensor([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]]) / 4

M_RGB_2_YUV = torch.Tensor([[0.299, 0.587, 0.114],
                            [-0.14714119, -0.28886916, 0.43601035],
                            [0.61497538, -0.51496512, -0.10001026]])
M_YUV_2_RGB = torch.Tensor([[1.0000000000e+00, -4.1827794561e-09, 1.1398830414e+00],
                            [1.0000000000e+00, -3.9464232326e-01, -5.8062183857e-01],
                            [1.0000000000e+00, 2.0320618153e+00, -1.2232658220e-09]])

K_BLUR = torch.Tensor([[6.9625e-08, 2.8089e-05, 2.0755e-04, 2.8089e-05, 6.9625e-08],
                       [2.8089e-05, 1.1332e-02, 8.3731e-02, 1.1332e-02, 2.8089e-05],
                       [2.0755e-04, 8.3731e-02, 6.1869e-01, 8.3731e-02, 2.0755e-04],
                       [2.8089e-05, 1.1332e-02, 8.3731e-02, 1.1332e-02, 2.8089e-05],
                       [6.9625e-08, 2.8089e-05, 2.0755e-04, 2.8089e-05, 6.9625e-08]])
K_SHARP = torch.Tensor([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
DEFAULT_CAMERA_PARAMS = (
    [0., 0., 0., 0.],
    [1., 1., 1.],
    [1., 0., 0., 0., 1., 0., 0., 0., 1.],
)


class RawToRGB(nn.Module):
    """transforms a raw image with 1 channel to rgb with 3 channels 

    Args:
        reduce_size (bool, optional): if False, the output image will have the same height and width 
            as the raw input, i.e. (B, C, H, W), empty values are filled with zeros.
            if True, the output dimensions are reduced by half (B, C, H//2, W//2), 
        out_channels (int, optional): number of output channels. One of {3, 4}.
            for 3 channels, the two green channels are averaged.
        track_stages (bool, optional): whether or not to retain intermediary steps in processing
        normalize_mosaic (function, optional): applies normalization transformation to rgb image
    """

    def __init__(self, reduce_size=True, out_channels=3, track_stages=False, normalize_mosaic=None):
        super().__init__()
        self.stages = None
        self.buffer = None
        self.reduce_size = reduce_size
        self.out_channels = out_channels
        self.track_stages = track_stages
        self.normalize_mosaic = normalize_mosaic

    def forward(self, raw):
        self.stages = {}
        self.buffer = {}

        rgb = raw2rgb(raw, reduce_size=self.reduce_size, out_channels=self.out_channels)
        self.stages['demosaic'] = rgb
        if self.normalize_mosaic:
            rgb = self.normalize_mosaic(rgb)

        if self.track_stages and raw.requires_grad:
            for stage in self.stages.values():
                stage.retain_grad()

        self.buffer['processed_rgb'] = rgb

        return rgb


class NNProcessing(nn.Module):
    """Transforms raw images to processed rgb via a segmentation Unet

    Args:
        track_stages (bool, optional): whether or not to retain intermediary steps in processing
        normalize_mosaic (function, optional): applies normalization transformation to rgb image
        batch_norm_output (bool, optional): adds a BatchNorm layer to the end of the processing
    """

    def __init__(self, track_stages=False, normalize_mosaic=None, batch_norm_output=True):
        super().__init__()
        self.stages = None
        self.buffer = None
        self.track_stages = track_stages
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_depth=3,
            decoder_channels=[256, 128, 64],
            in_channels=3,
            classes=3,
        )
        self.batch_norm = None if not batch_norm_output else nn.BatchNorm2d(3, affine=False)
        self.normalize_mosaic = normalize_mosaic

    def forward(self, raw):
        self.stages = {}
        self.buffer = {}

        rgb = raw2rgb(raw)
        if self.normalize_mosaic:
            rgb = self.normalize_mosaic(rgb)
        self.stages['demosaic'] = rgb
        rgb = self.model(rgb)
        if self.batch_norm is not None:
            rgb = self.batch_norm(rgb)
        self.stages['rgb'] = rgb

        if self.track_stages and raw.requires_grad:
            for stage in self.stages.values():
                stage.retain_grad()

        self.buffer['processed_rgb'] = rgb

        return rgb


def append_additive_layer(processor):
    processor.additive_layer = nn.Parameter(torch.zeros((1, 3, 256, 256)))
    # processor.additive_layer = nn.Parameter(0.001 * torch.randn((1, 3, 256, 256)))


class ParametrizedProcessing(nn.Module):
    """Differentiable processing pipeline via torch transformations

    Args:
        camera_parameters (tuple(list), optional): applies given camera parameters in processing
        track_stages (bool, optional): whether or not to retain intermediary steps in processing
        batch_norm_output (bool, optional): adds a BatchNorm layer to the end of the processing
    """

    def __init__(self, camera_parameters=None, track_stages=False, batch_norm_output=True):
        super().__init__()
        self.stages = None
        self.buffer = None
        self.track_stages = track_stages

        if camera_parameters is None:
            camera_parameters = DEFAULT_CAMERA_PARAMS

        black_level, white_balance, colour_matrix = camera_parameters

        self.black_level = nn.Parameter(torch.as_tensor(black_level))
        self.white_balance = nn.Parameter(torch.as_tensor(white_balance).reshape(1, 3))
        self.colour_correction = nn.Parameter(torch.as_tensor(colour_matrix).reshape(3, 3))

        self.gamma_correct = nn.Parameter(torch.Tensor([2.2]))

        self.debayer = Debayer()

        self.sharpening_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sharpening_filter.weight.data[0][0] = K_SHARP.clone()

        self.gaussian_blur = nn.Conv2d(1, 1, kernel_size=5, padding=2, padding_mode='reflect', bias=False)
        self.gaussian_blur.weight.data[0][0] = K_BLUR.clone()

        self.batch_norm = nn.BatchNorm2d(3, affine=False) if batch_norm_output else None

        self.register_buffer('M_RGB_2_YUV', M_RGB_2_YUV.clone())
        self.register_buffer('M_YUV_2_RGB', M_YUV_2_RGB.clone())

        self.additive_layer = None  # this can be added in later

    def forward(self, raw):
        assert raw.ndim == 3, f"needs dims (B, H, W), got {raw.shape}"

        self.stages = {}
        self.buffer = {}

        # self.stages['raw'] = raw

        rgb = raw2rgb(raw, black_level=self.black_level, reduce_size=False)
        rgb = rgb.contiguous()
        self.stages['demosaic'] = rgb

        rgb = self.debayer(rgb)
        # self.stages['debayer'] = rgb

        rgb = torch.einsum('bchw,kc->bchw', rgb, self.white_balance).contiguous()
        rgb = torch.einsum('bchw,kc->bkhw', rgb, self.colour_correction).contiguous()
        self.stages['color_correct'] = rgb

        yuv = torch.einsum('bchw,kc->bkhw', rgb, self.M_RGB_2_YUV).contiguous()
        yuv[:, [0], ...] = self.sharpening_filter(yuv[:, [0], ...])

        if self.track_stages:    # keep stage in computational graph for grad information
            rgb = torch.einsum('bchw,kc->bkhw', yuv.clone(), self.M_YUV_2_RGB).contiguous()
            self.stages['sharpening'] = rgb
            yuv = torch.einsum('bchw,kc->bkhw', rgb, self.M_RGB_2_YUV).contiguous()

        yuv[:, [0], ...] = self.gaussian_blur(yuv[:, [0], ...])
        rgb = torch.einsum('bchw,kc->bkhw', yuv, self.M_YUV_2_RGB).contiguous()
        self.stages['gaussian'] = rgb

        rgb = torch.clip(rgb, 1e-5, 1)
        self.stages['clipped'] = rgb

        rgb = torch.exp((1 / self.gamma_correct) * torch.log(rgb))
        self.stages['gamma_correct'] = rgb

        if self.additive_layer is not None:
            rgb = rgb + self.additive_layer
            self.stages['noise'] = rgb

        if self.batch_norm is not None:
            rgb = self.batch_norm(rgb)

        if self.track_stages and raw.requires_grad:
            for stage in self.stages.values():
                stage.retain_grad()

        self.buffer['processed_rgb'] = rgb

        return rgb


class Debayer(nn.Conv2d):
    """Separates the mosaiced raw image into its channels and interpolates bilinearly. Output is of same size as input.
    """

    def __init__(self):
        super().__init__(3, 3, kernel_size=3, padding=1, padding_mode='reflect', bias=False)    # pipeline_numpy uses 'replicate'
        self.weight.data.fill_(0)
        self.weight.data[0, 0] = K_RB.clone()
        self.weight.data[1, 1] = K_G.clone()
        self.weight.data[2, 2] = K_RB.clone()


def raw2rgb(raw, black_level=None, reduce_size=True, out_channels=3):
    """Transforms a raw image with 1 channel to rgb with 3 channels 

    Args:
        raw (Tensor): raw Tensor of shape (B, H, W)
        black_level (iterable, optional): RGGB black level values to subtract
        reduce_size (bool, optional): if False, the output image will have the same height and width 
            as the raw input, i.e. (B, C, H, W), empty values are filled with zeros.
            if True, the output dimensions are reduced by half (B, C, H//2, W//2), 
        out_channels (int, optional): number of output channels. One of {3, 4}.
            The two green channels are averaged if out_channels == 3.
    """
    assert out_channels in [3, 4]
    if black_level is None:
        black_level = [0, 0, 0, 0]
    Bch, H, W = raw.shape
    R = raw[:, 0::2, 0::2] - black_level[0]     # R
    G1 = raw[:, 0::2, 1::2] - black_level[1]    # G
    G2 = raw[:, 1::2, 0::2] - black_level[2]    # G
    B = raw[:, 1::2, 1::2] - black_level[3]     # B
    if reduce_size:
        rgb = torch.zeros((Bch, out_channels, H // 2, W // 2), device=raw.device)
        if out_channels == 3:
            rgb[:, 0, :, :] = R
            rgb[:, 1, :, :] = (G1 + G2) / 2
            rgb[:, 2, :, :] = B
        elif out_channels == 4:
            rgb[:, 0, :, :] = R
            rgb[:, 1, :, :] = G1
            rgb[:, 2, :, :] = G2
            rgb[:, 3, :, :] = B
    else:
        rgb = torch.zeros((Bch, out_channels, H, W), device=raw.device)
        if out_channels == 3:
            rgb[:, 0, 0::2, 0::2] = R
            rgb[:, 1, 0::2, 1::2] = G1
            rgb[:, 1, 1::2, 0::2] = G2
            rgb[:, 2, 1::2, 1::2] = B
        elif out_channels == 4:
            rgb[:, 0, 0::2, 0::2] = R
            rgb[:, 1, 0::2, 1::2] = G1
            rgb[:, 2, 1::2, 0::2] = G2
            rgb[:, 3, 1::2, 1::2] = B
    return rgb


# pipeline validation
if __name__ == "__main__":

    import torch
    import numpy as np

    if not os.path.exists('README.md'):
        os.chdir('..')

    import matplotlib.pyplot as plt
    from dataset import get_dataset
    from utils.base import np2torch, torch2np

    from utils.debug import debug
    from processing.pipeline_numpy import processing as default_processing

    raw_dataset = get_dataset('DS')
    loader = torch.utils.data.DataLoader(raw_dataset, batch_size=1)
    batch_raw, batch_mask = next(iter(loader))

    # torch proc
    camera_parameters = raw_dataset.camera_parameters
    black_level = camera_parameters[0]

    proc = ParametrizedProcessing(camera_parameters)

    batch_rgb = proc(batch_raw)
    rgb = batch_rgb[0]

    # numpy proc
    raw_img = batch_raw[0]
    numpy_raw = torch2np(raw_img)

    default_rgb = default_processing(numpy_raw, *camera_parameters,
                                     sharpening='sharpening_filter', denoising='gaussian_denoising')

    rgb_valid = np2torch(default_rgb)

    print("pipeline norm difference:", (rgb - rgb_valid).norm().item())

    rgb_mosaic = raw2rgb(batch_raw, reduce_size=False).squeeze()
    rgb_reduced = raw2rgb(batch_raw, reduce_size=True).squeeze()

    plt.figure(figsize=(16, 8))
    plt.subplot(151)
    plt.title('Raw')
    plt.imshow(torch2np(raw_img))
    plt.subplot(152)
    plt.title('RGB Mosaic')
    plt.imshow(torch2np(rgb_mosaic))
    plt.subplot(153)
    plt.title('RGB Reduced')
    plt.imshow(torch2np(rgb_reduced))
    plt.subplot(154)
    plt.title('Torch Pipeline')
    plt.imshow(torch2np(rgb))
    plt.subplot(155)
    plt.title('Default Pipeline')
    plt.imshow(torch2np(rgb_valid))
    plt.show()

    # assert rgb.allclose(rgb_valid)
