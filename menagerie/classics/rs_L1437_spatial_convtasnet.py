# SOURCE: vendored from claritychallenge/clarity @ baf2b2c99f2e799ba5b0e24d0ad8a9f7ca0370d6
# https://raw.githubusercontent.com/claritychallenge/clarity/main/clarity/enhancer/dnn/mc_conv_tasnet.py
#
# Spatial Conv-TasNet ("mc_conv_tasnet" / multi-channel Conv-TasNet, itself adapted by
# the Clarity Challenge team from kaituoxu/Conv-TasNet): a Conv-TasNet variant with two
# parallel front-end encoders -- a single-channel SpectralEncoder over the reference
# microphone and a multi-channel SpatialEncoder over all microphones -- whose latents
# are concatenated before the shared TemporalConvNet mask-estimation separator
# (stacked dilated depthwise-separable TemporalBlocks with channelwise/global layer
# norm), followed by a Decoder that reconstructs the extracted source via masked
# overlap-and-add. `ConvTasNet`/`SpectralEncoder`/`SpatialEncoder`/`Decoder`/
# `TemporalConvNet`/`TemporalBlock`/`DepthwiseSeparableConv`/`Chomp1d`/`chose_norm`/
# `ChannelwiseLayerNorm`/`GlobalLayerNorm`/`overlap_and_add` are transcribed verbatim
# from `clarity/enhancer/dnn/mc_conv_tasnet.py`; only the module-level docstring/typing
# import style was left as-is and the CPU device default is used for tracing.

import math
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

EPS: Final = 1e-8


def overlap_and_add(signal, frame_step, device):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown,
            and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to
            frame_length.
        device: Whether to use 'cpu' or 'cuda' for processing.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of
            signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/
        contrib/signal/python/ops/reconstruction_ops.py
    """
    signal = signal.to(device)

    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame, device=device).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length, device=device)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class ConvTasNet(nn.Module):
    def __init__(
        self,
        N_spec,
        N_spat,
        L,
        B,
        H,
        P,
        X,
        R,
        C,
        num_channels,
        norm_type="cLN",
        causal=False,
        mask_nonlinear="relu",
        device=None,
    ):
        """
        Args:
            N_spec: Number of filters in spectral autoencoder
            N_spat: Number of filters in spatial autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 x 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()

        # Hyper-parameter
        (
            self.N_spec,
            self.N_spat,
            self.L,
            self.B,
            self.H,
            self.P,
            self.X,
            self.R,
            self.C,
        ) = (N_spec, N_spat, L, B, H, P, X, R, C)
        self.num_channels = num_channels
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.spectral_encoder = SpectralEncoder(L, N_spec)
        self.spatial_encoder = SpatialEncoder(L, N_spat, num_channels)

        self.separator = TemporalConvNet(
            N_spec, N_spat, B, H, P, X, R, C, norm_type, causal, mask_nonlinear
        )
        self.decoder = Decoder(N_spec, L, device)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, num_channels, T], M is batch size, T is #samples

        Returns:
            est_source: [M, C, T]
        """
        mixture_spectral = self.spectral_encoder(torch.narrow(mixture, 1, 0, 1).squeeze(1))
        mixture_spatial = self.spatial_encoder(mixture)

        mixture_w = torch.cat((mixture_spectral, mixture_spatial), 1)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_spectral, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class SpectralEncoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""

    def __init__(self, L, N):
        super().__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples

        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class SpatialEncoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""

    def __init__(self, L, N, num_channels):
        super().__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(num_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, num_channels, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L, device=None):
        super().__init__()
        # device for overlap_and add
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Hyper-parameter
        self.N, self.L = N, L
        # Components
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]

        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L // 2, device=self.device)  # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        N_spec,
        N_spat,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
    ):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 x 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.C, self.N_spec = C, N_spec
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N_spec + N_spat)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N_spec + N_spat, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for _r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * N_spec, 1, bias=False)
        # Put together
        self.network = nn.Sequential(
            layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1
        )

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet

        Args:
            mixture_w: [M, N, K], M is batch size

        Returns:
            est_mask: [M, C, N, K]
        """
        M, _N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, self.N_spec, K)  # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = F.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        return out + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, : -self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
    C is channel size and K is sequence length.

    Args:
        normn_type ():
        channel_size ():
    Returns:

    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    if norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    # norm_type == "BN":
    # # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
    # along M and K, so this BN usage is right.
    return nn.BatchNorm1d(channel_size)


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def build_spatial_convtasnet():
    model = ConvTasNet(
        N_spec=16,
        N_spat=8,
        L=16,
        B=8,
        H=16,
        P=3,
        X=2,
        R=1,
        C=2,
        num_channels=4,
        norm_type="cLN",
        causal=False,
        mask_nonlinear="relu",
        device="cpu",
    )
    model.eval()
    return model


def example_input_spatial_convtasnet():
    # [Batch, num_channels(mics), num_samples]
    return torch.randn(1, 4, 2000)


MENAGERIE_ENTRIES = [
    (
        "SpatialConvTasNet",
        "build_spatial_convtasnet",
        "example_input_spatial_convtasnet",
        2022,
        MENAGERIE_ZOO,
    ),
]
