# SOURCE: vendored from YangLabHKUST/SpatialScope @ master (src/SCGrad/)
#
# SpatialScope's spot-level gene-expression imputation stage uses SCGradNN, a
# WaveGrad-style dual-stream 1D diffusion U-Net: one downsampling stream consumes the
# noised signal `mu`, the other consumes the conditioning signal `x_sep`, and FiLM
# (feature-wise linear modulation) blocks conditioned on the diffusion noise level tie
# the two streams together across an encoder/decoder with skip connections. All classes
# (`SCGradNN`, `UBlock`/`UpsamplingBlock`, `DBlock`/`DownsamplingBlock`,
# `Conv1dWithInitialization`, `FeatureWiseLinearModulation_NoScalarCond`,
# `FeatureWiseAffine`, `PositionalEncoding`, `Upsample`, `Downsample`,
# `InterpolationBlock`, `BaseModule`) are transcribed verbatim from
# src/SCGrad/{nn.py,upsampling.py,downsampling.py,linear_modulation.py,
# interpolation.py,nn_layers.py,base.py}, merged into one file and with the internal
# `from SCGrad.xxx import yyy` package imports flattened to plain intra-file
# references (this is the only import change; no architecture code was altered).

import numpy as np
import torch


LINEAR_SCALE = 5000


# ---------------------------------------------------------------------------
# src/SCGrad/base.py
# ---------------------------------------------------------------------------


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# src/SCGrad/nn_layers.py
# ---------------------------------------------------------------------------


class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


# ---------------------------------------------------------------------------
# src/SCGrad/interpolation.py
# ---------------------------------------------------------------------------


class InterpolationBlock(BaseModule):
    def __init__(self, scale_factor, mode="linear", align_corners=False, downsample=False):
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor
            if not self.downsample
            else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        return outputs


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, scale_factor, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=scale_factor,
                padding=3,
            )

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, scale_factor, remain_dim, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if remain_dim is None:
            remain_dim = 0
        if self.with_conv:
            self.conv = torch.nn.ConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=scale_factor,
                padding=3,
                output_padding=remain_dim,
            )

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# src/SCGrad/linear_modulation.py
# ---------------------------------------------------------------------------


class PositionalEncoding(BaseModule):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4**exponents
        exponents = LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class FeatureWiseLinearModulation_NoScalarCond(BaseModule):
    def __init__(self, in_channels, out_channels, input_dscaled_by):
        super(FeatureWiseLinearModulation_NoScalarCond, self).__init__()
        self.signal_conv = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2),
            ]
        )
        self.positional_encoding = PositionalEncoding(in_channels)
        self.scale_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.shift_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, noise_level):
        outputs = self.signal_conv(x)
        # NOTE: matches the real repo -- the _NoScalarCond variant intentionally does
        # NOT add the positional encoding (the plain FeatureWiseLinearModulation class
        # in the same file does; SCGradNN.__init__ selects the NoScalarCond FiLM).
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class FeatureWiseAffine(BaseModule):
    def __init__(self, num_features):
        super(FeatureWiseAffine, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm1d(num_features)

    def forward(self, x, scale, shift):
        x = self.instance_norm(x)
        outputs = scale * x + shift
        return outputs


# ---------------------------------------------------------------------------
# src/SCGrad/downsampling.py
# ---------------------------------------------------------------------------


class ConvolutionBlock(BaseModule):
    def __init__(self, in_channels, out_channels, dilation):
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs


class DownsamplingBlock(BaseModule):
    def __init__(
        self, in_channels, out_channels, factor, dilations, downsampling=True, remain_dim=None
    ):
        super(DownsamplingBlock, self).__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels=in_channels, scale_factor=factor)
            sampling_layer2 = Downsample(in_channels=out_channels, scale_factor=factor)
        else:
            sampling_layer1 = Upsample(
                in_channels=in_channels, scale_factor=factor, remain_dim=remain_dim
            )
            sampling_layer2 = Upsample(
                in_channels=out_channels, scale_factor=factor, remain_dim=remain_dim
            )

        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        self.main_branch = torch.nn.Sequential(
            *(
                [
                    sampling_layer1,
                ]
                + [
                    ConvolutionBlock(in_size, out_size, dilation)
                    for in_size, out_size, dilation in zip(in_sizes, out_sizes, dilations)
                ]
            )
        )
        self.residual_branch = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                sampling_layer2,
            ]
        )

    def forward(self, x):
        outputs = self.main_branch(x)
        outputs = outputs + self.residual_branch(x)
        return outputs


# ---------------------------------------------------------------------------
# src/SCGrad/upsampling.py
# ---------------------------------------------------------------------------


class BasicModulationBlock(BaseModule):
    """
    Linear modulation part of UBlock, represented by sequence of the following layers:
        - Feature-wise Affine
        - LReLU
        - 3x1 Conv
    """

    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine(n_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs


class UpsamplingBlock(BaseModule):
    def __init__(
        self, in_channels, out_channels, factor, dilations, downsampling=False, remain_dim=None
    ):
        super(UpsamplingBlock, self).__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels=in_channels, scale_factor=factor)
            sampling_layer2 = Downsample(in_channels=out_channels, scale_factor=factor)
        else:
            sampling_layer1 = Upsample(
                in_channels=in_channels, scale_factor=factor, remain_dim=remain_dim
            )
            sampling_layer2 = Upsample(
                in_channels=out_channels, scale_factor=factor, remain_dim=remain_dim
            )
        self.first_block_main_branch = torch.nn.ModuleDict(
            {
                "upsampling": torch.nn.Sequential(
                    *[
                        torch.nn.LeakyReLU(0.2),
                        sampling_layer1,
                        Conv1dWithInitialization(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=dilations[0],
                            dilation=dilations[0],
                        ),
                    ]
                ),
                "modulation": BasicModulationBlock(out_channels, dilation=dilations[1]),
            }
        )
        self.first_block_residual_branch = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                sampling_layer2,
            ]
        )
        self.second_block_main_branch = torch.nn.ModuleDict(
            {
                f"modulation_{idx}": BasicModulationBlock(out_channels, dilation=dilations[2 + idx])
                for idx in range(2)
            }
        )

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch["upsampling"](x)
        outputs = self.first_block_main_branch["modulation"](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch["modulation_0"](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch["modulation_1"](residual, scale, shift)
        return outputs


# ---------------------------------------------------------------------------
# src/SCGrad/nn.py
# ---------------------------------------------------------------------------


def cal_final_dim(input_dim, factor):
    output_dim = (((input_dim + 4) - 7) // factor) + 1
    return output_dim


def remain_dim(input_dim, factor):
    remain = ((input_dim + 4) - 7) % factor
    return remain


def cal_final_updim(output_dim, factor):
    output_dim_ori = output_dim
    output_dim = int((output_dim - 1) / factor + 1)
    output_padding = output_dim_ori - ((output_dim - 1) * factor + 1)
    return output_dim, output_padding


class SCGradNN(BaseModule):
    def __init__(
        self,
        input_dim1: int,
        down_block_dim: list = [32, 128, 256, 512],
        factors: list = [3, 4, 5, 1],
        downsampling_dilations: list = [
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
        ],
        upsampling_dilations: list = [
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 1, 2],
        ],
        seed=182822,
    ):
        super(SCGradNN, self).__init__()

        self.input_dim1 = input_dim1
        self.down_block_dim = down_block_dim
        self.factors = factors
        self.downsampling_dilations = [[1, 2, 4]] * len(self.down_block_dim)
        self.upsampling_dilations = [[1, 2, 1, 2]] * len(self.down_block_dim)

        cal_factor = self.factors[:-1]
        output_dim = self.input_dim1
        self.output_padding = []
        for fac in cal_factor:
            output_dim, out_padding = cal_final_updim(output_dim, fac)
            self.output_padding.append(out_padding)

        # U_net left stream
        self.left_ublock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=self.down_block_dim[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.left_ublocks = torch.nn.ModuleList(
            [
                UpsamplingBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                    downsampling=True,
                )
                for in_size, out_size, factor, dilations in zip(
                    self.down_block_dim,
                    self.down_block_dim[1:] + [self.down_block_dim[-1]],
                    self.factors,
                    self.upsampling_dilations,
                )
            ]
        )

        # Building downsampling branch (starting from signal)
        self.left_dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=down_block_dim[0],
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.left_dblocks = torch.nn.ModuleList(
            [
                DownsamplingBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                )
                for in_size, out_size, factor, dilations in zip(
                    self.down_block_dim,
                    self.down_block_dim[1:] + [self.down_block_dim[-1]],
                    self.factors,
                    self.downsampling_dilations,
                )
            ]
        )

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = self.down_block_dim[1:] + [self.down_block_dim[-1]]
        film_out_sizes = self.down_block_dim[1:] + [self.down_block_dim[-1]]
        film_factors = [1] + self.factors[1:][::-1]
        self.left_films = torch.nn.ModuleList(
            [
                FeatureWiseLinearModulation_NoScalarCond(
                    in_channels=in_size,
                    out_channels=out_size,
                    input_dscaled_by=np.product(film_factors[: i + 1]),
                )
                for i, (in_size, out_size) in enumerate(zip(film_in_sizes, film_out_sizes))
            ]
        )

        # U_net right stream
        self.right_ublocks = torch.nn.ModuleList(
            [
                UpsamplingBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                    remain_dim=remain_dim,
                )
                for in_size, out_size, factor, dilations, remain_dim in zip(
                    (np.array(self.down_block_dim[1:]) * 2)[::-1],
                    self.down_block_dim[:-1][::-1],
                    self.factors[:-1][::-1],
                    self.upsampling_dilations[:-1],
                    self.output_padding[::-1],
                )
            ]
        )

        self.right_ublock_postconv = Conv1dWithInitialization(
            in_channels=self.down_block_dim[0],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.right_dblocks = torch.nn.ModuleList(
            [
                DownsamplingBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                    downsampling=False,
                    remain_dim=remain_dim,
                )
                for in_size, out_size, factor, dilations, remain_dim in zip(
                    (np.array(self.down_block_dim[1:]) * 2)[::-1],
                    self.down_block_dim[:-1][::-1],
                    self.factors[:-1][::-1],
                    self.downsampling_dilations[:-1],
                    self.output_padding[::-1],
                )
            ]
        )

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = self.down_block_dim[:-1][::-1]
        film_out_sizes = self.down_block_dim[:-1][::-1]
        film_factors = [1] + self.factors[1:][::-1]
        self.right_films = torch.nn.ModuleList(
            [
                FeatureWiseLinearModulation_NoScalarCond(
                    in_channels=in_size,
                    out_channels=out_size,
                    input_dscaled_by=np.product(film_factors[: i + 1]),
                )
                for i, (in_size, out_size) in enumerate(zip(film_in_sizes, film_out_sizes))
            ]
        )

    def forward(self, x_sep, mu, noise_level):
        """
        Computes forward pass of neural network.
        :param x_sep (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_x_sep, T//hop_length]
        :param mu (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        x_sep = x_sep.unsqueeze(1)
        assert len(x_sep.shape) == 3  # B, 1, T
        mu = mu.unsqueeze(1)
        assert len(mu.shape) == 3  # B, 1, T

        # left stream
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        hs_u, hs_d = [], []
        left_dblock_outputs = self.left_dblock_preconv(mu)
        for dblock, film in zip(self.left_dblocks, self.left_films):
            left_dblock_outputs = dblock(left_dblock_outputs)
            scale, shift = film(x=left_dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
            hs_d.append(left_dblock_outputs)

        left_ublock_outputs = self.left_ublock_preconv(x_sep)
        for i, ublock in enumerate(self.left_ublocks):
            scale, shift = statistics[i]
            left_ublock_outputs = ublock(x=left_ublock_outputs, scale=scale, shift=shift)
            hs_u.append(left_ublock_outputs)

        _, _ = hs_u.pop(), hs_d.pop()

        # right stream
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        right_dblock_outputs = left_dblock_outputs
        for dblock, film in zip(self.right_dblocks, self.right_films):
            right_dblock_outputs = dblock(torch.cat([right_dblock_outputs, hs_d.pop()], dim=1))
            scale, shift = film(x=right_dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])

        # Upsampling stream
        right_ublock_outputs = left_ublock_outputs
        for i, ublock in enumerate(self.right_ublocks):
            scale, shift = statistics[i]
            right_ublock_outputs = ublock(
                x=torch.cat([right_ublock_outputs, hs_u.pop()], dim=1), scale=scale, shift=shift
            )
        outputs = self.right_ublock_postconv(right_ublock_outputs)
        return outputs.squeeze(1)


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_scgradnn():
    # Real usage feeds full-length gene-expression / signal vectors (T can be large);
    # down_block_dim/factors match the repo defaults (downsample factors 3*4*5=60 along
    # the down-stream). input_dim1 must stay large enough that InstanceNorm1d never
    # sees a spatial dim of 1 partway through the U-Net (it requires >1 spatial element
    # in training mode) -- 480 keeps every intermediate feature map length > 1.
    return SCGradNN(input_dim1=480)


def example_input_scgradnn():
    B, T = 2, 480
    x_sep = torch.randn(B, T)
    mu = torch.randn(B, T)
    noise_level = torch.rand(B, 1)
    return (x_sep, mu, noise_level)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SpatialScope-SCGrad", "build_scgradnn", "example_input_scgradnn", 2023, "vendored"),
]
