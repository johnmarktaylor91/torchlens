# FAITHFUL PORT of cagladbahadir/LOUPE @ master (original framework: TensorFlow 1.x / Keras)
# https://github.com/cagladbahadir/LOUPE
#
# LOUPE ("Learning-based Optimization of the Under-sampling Pattern in MRI", IPMI 2019,
# Bahadir/Dalca/Sabuncu, arXiv:1901.01960) jointly learns a probabilistic k-space
# under-sampling mask and a UNet reconstruction network. The original repo
# (loupe/models.py, loupe/layers.py) is TF1-era Keras (tf.fft/tf.ifft,
# keras.layers.normalization) and does not import cleanly against the installed
# base env (modern tf.keras has moved/removed those APIs); it is transcribed here
# FAITHFULLY, layer-for-layer, into self-contained torch:
#   - ConcatenateZero -> concat a zero imaginary channel (2-channel real/imag repr)
#   - FFT / IFFT       -> torch.fft.fft2 / ifft2 on the complex-valued (real, imag) pair
#   - ProbMask         -> learned per-pixel logit weights -> sigmoid(slope * w), "v2"
#                          RescaleProbMap-normalised to a target sparsity
#   - RandomMask        -> uniform[0,1] threshold tensor (fresh draw every forward pass,
#                           matching the original's re-sampling behaviour)
#   - ThresholdRandomMask -> sigmoid(sample_slope * (prob - thresh)) soft-binarisation
#   - UnderSample       -> multiply k-space by the sampled (real-valued, broadcast over
#                           the 2 channels) mask
#   - ComplexAbs        -> magnitude of the complex-valued under-sampled image
#   - UNet ("_unet_from_tensor") -> the exact same hard-coded 5-level encoder / 4-level
#                          decoder (Conv-LeakyReLU-BatchNorm x2 per level, average-pool
#                          down, upsample + concat skip up) as models.py
#   - final output = complex_abs(undersampled image) + unet(undersampled image), matching
#     "add_tensor = Add()([abs_tensor, unet_tensor])" in loupe_model(model_type='v2')
#
# Only mechanical adaptations were made: Keras Layer classes -> torch.nn.Module classes,
# tf.fft/tf.sigmoid -> torch.fft/torch.sigmoid, NHWC -> NCHW tensor layout. No architectural
# mechanism was added, removed, or altered relative to the original loupe_model(model_type='v2').

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ProbMask(nn.Module):
    """Learned probability mask (loupe/layers.py: ProbMask), logit-space weights.

    v2 default initializer: logit of Uniform(eps, 1-eps) rescaled by 1/slope, matching
    `_logit_slope_random_uniform` in the original.
    """

    def __init__(self, height, width, slope=5.0, eps=0.01):
        super().__init__()
        self.slope = slope
        u = torch.empty(height, width).uniform_(eps, 1.0 - eps)
        logit_init = -torch.log(1.0 / u - 1.0) / slope
        self.mult = nn.Parameter(logit_init)

    def forward(self, x):
        # x: (B, C, H, W) k-space tensor; mask is broadcast over batch/channel
        return (
            torch.sigmoid(self.slope * self.mult)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(x.shape[0], 1, x.shape[2], x.shape[3])
        )


class RescaleProbMap(nn.Module):
    """RescaleProbMap: rescale a probability map to hit a target sparsity exactly."""

    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, x):
        xbar = x.mean()
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (r <= 1).float()
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)


class RandomMask(nn.Module):
    """RandomMask: fresh Uniform(0,1) threshold field, same shape as the prob map."""

    def forward(self, x):
        return torch.rand_like(x)


class ThresholdRandomMask(nn.Module):
    """ThresholdRandomMask: soft-binarise prob_mask against the random threshold."""

    def __init__(self, slope=12.0):
        super().__init__()
        self.slope = slope

    def forward(self, prob_mask, thresh):
        return torch.sigmoid(self.slope * (prob_mask - thresh))


class UnderSample(nn.Module):
    """UnderSample: multiply (real, imag) k-space channels by the sampled mask."""

    def forward(self, kspace, mask):
        # kspace: (B, 2, H, W) real/imag ; mask: (B, 1, H, W)
        return kspace * mask


class ComplexAbs(nn.Module):
    """ComplexAbs: magnitude of a (real, imag) 2-channel tensor -> 1-channel."""

    def forward(self, x):
        real, imag = x[:, 0:1], x[:, 1:2]
        return torch.sqrt(real * real + imag * imag)


def _conv_block(in_ch, out_ch, kern):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kern, padding=kern // 2),
        nn.LeakyReLU(0.3),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, kern, padding=kern // 2),
        nn.LeakyReLU(0.3),
        nn.BatchNorm2d(out_ch),
    )


class LoupeUNet(nn.Module):
    """The exact 5-down/4-up hard-coded UNet from loupe/models.py `_unet_from_tensor`."""

    def __init__(self, in_ch=2, filt=8, kern=3, out_ch=2):
        super().__init__()
        self.conv1 = _conv_block(in_ch, filt, kern)
        self.conv2 = _conv_block(filt, filt * 2, kern)
        self.conv3 = _conv_block(filt * 2, filt * 4, kern)
        self.conv4 = _conv_block(filt * 4, filt * 8, kern)
        self.conv5 = _conv_block(filt * 8, filt * 16, kern)

        self.conv6 = _conv_block(filt * 16 + filt * 8, filt * 8, kern)
        self.conv7 = _conv_block(filt * 8 + filt * 4, filt * 4, kern)
        self.conv8 = _conv_block(filt * 4 + filt * 2, filt * 2, kern)
        self.conv9 = _conv_block(filt * 2 + filt, filt, kern)
        self.conv9_out = nn.Conv2d(filt, out_ch, 1)

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        c5 = self.conv5(p4)

        u1 = F.interpolate(c5, scale_factor=2, mode="nearest")
        c6 = self.conv6(torch.cat([c4, u1], dim=1))
        u2 = F.interpolate(c6, scale_factor=2, mode="nearest")
        c7 = self.conv7(torch.cat([c3, u2], dim=1))
        u3 = F.interpolate(c7, scale_factor=2, mode="nearest")
        c8 = self.conv8(torch.cat([c2, u3], dim=1))
        u4 = F.interpolate(c8, scale_factor=2, mode="nearest")
        c9 = self.conv9(torch.cat([c1, u4], dim=1))
        return self.conv9_out(c9)


class LoupeModel(nn.Module):
    """LOUPE v2: learned probabilistic under-sampling mask + UNet reconstruction.

    Faithful port of loupe_model(model_type='v2'): input -> concat-zero -> FFT ->
    learned+sparsity-rescaled prob mask -> random-threshold soft sampling -> under-sample
    k-space -> IFFT -> complex_abs + UNet(under-sampled image) residual sum.
    """

    def __init__(
        self, height=64, width=64, sparsity=0.25, filt=8, kern=3, pmask_slope=5.0, sample_slope=12.0
    ):
        super().__init__()
        self.concat_zero_pad = True
        self.prob_mask = ProbMask(height, width, slope=pmask_slope)
        self.rescale = RescaleProbMap(sparsity)
        self.random_mask = RandomMask()
        self.threshold = ThresholdRandomMask(slope=sample_slope)
        self.undersample = UnderSample()
        self.complex_abs = ComplexAbs()
        self.unet = LoupeUNet(in_ch=2, filt=filt, kern=kern, out_ch=2)

    def forward(self, x):
        # x: (B, 1, H, W) real-valued magnitude image
        two_channel = torch.cat([x, torch.zeros_like(x)], dim=1)  # ConcatenateZero
        kspace = torch.fft.fft2(torch.complex(two_channel[:, 0], two_channel[:, 1]))
        kspace = torch.stack([kspace.real, kspace.imag], dim=1).float()  # FFT layer

        prob_mask = self.prob_mask(kspace)
        prob_mask = self.rescale(prob_mask)
        thresh = self.random_mask(prob_mask)
        sampled_mask = self.threshold(prob_mask, thresh)

        under_kspace = self.undersample(kspace, sampled_mask)
        under_complex = torch.fft.ifft2(torch.complex(under_kspace[:, 0], under_kspace[:, 1]))
        under_img = torch.stack([under_complex.real, under_complex.imag], dim=1).float()  # IFFT

        abs_tensor = self.complex_abs(under_img)
        unet_tensor = self.unet(under_img)
        # match the original: Add()([abs_tensor, unet_tensor]) where unet_tensor has
        # `output_nb_feats=input_shape[-1]` = 1 channel in single-coil magnitude mode.
        return abs_tensor + unet_tensor[:, 0:1]


def build_loupe():
    return LoupeModel(height=64, width=64, sparsity=0.25, filt=8, kern=3)


def example_input_loupe():
    return torch.rand(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("LOUPE", "build_loupe", "example_input_loupe", "2019", "ported-pytorch"),
]
