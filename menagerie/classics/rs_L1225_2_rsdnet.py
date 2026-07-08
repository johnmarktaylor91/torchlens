# SOURCE: vendored from fmu2/nlos3d @ main
# https://github.com/fmu2/nlos3d/blob/main/libs/encoder.py
# https://github.com/fmu2/nlos3d/blob/main/libs/modules.py
#
# Mu, Mo, Peng, Liu, Nam, Raghavan, Velten & Li, "Physics to the Rescue: Deep
# Non-line-of-sight Reconstruction for High-speed Imaging," ICCP/TPAMI 2022
# (the "D-NLOS" candidate). Official PyTorch implementation. `RSDNet` is the
# repo's differentiable "RSD encoder": a learnable Conv3d in_block, a
# Rayleigh-Sommerfeld-diffraction propagation layer (`RSDEfficient`, implemented
# as a physics-derived, non-learnable FFT-based operator registered as
# buffers) that lifts time-domain transient features into the 3D space
# (depth/height/width) volume, and a learnable Conv3d out_block. This is the
# genuine encoder used by the paper's supervised/unsupervised/joint models
# (see `libs/model.py: make_encoder_decoder_model` /
# `make_encoder_renderer_model`, `config['encoder']['type'] == 'rsdnet'`,
# e.g. `configs/unsup/train_ch1_128.yaml`). The repo's differentiable
# transient-volume RENDERER (`libs/renderer.py`, used only downstream of the
# encoder to re-render predicted geometry into images) depends on a custom
# CUDA/C++ extension (`libs/clib`, ray-AABB intersection + sorted-array
# merging) that requires `python setup.py build_ext --inplace`; RSDNet itself
# has no such dependency (`libs/encoder.py` and `libs/modules.py` import only
# torch/numpy/scipy), so it is vendored standalone here without the renderer.
#
# Transcribed verbatim: `RSDBase`/`RSDEfficient` (RSD kernel) from
# `libs/encoder.py`, and `RSDNet` (RSD encoder) plus its helper dependencies
# `make_actv`, `make_norm3d`, `MaxNorm`, `ResConv3d`, `ResBlock3d` from
# `libs/modules.py`. Only edits: dropped the unused `RSD` (non-efficient,
# zero-padded) variant, `FFCNet`, `FRN`, `UNet` classes and their 2D-specific
# helper siblings (`make_norm2d`, `Blur2d`, `ResConv2d`, `ResBlock2d`, up/down
# -sample blocks, FFC blocks) that `RSDNet` does not use; added
# build_/example_input_ staging helpers below. One mechanical import fix:
# `scipy.signal.gaussian` was removed in scipy>=1.13 (moved under
# `scipy.signal.windows.gaussian`, same function) -- repointed the import,
# no architectural line altered.

import numpy as np
import scipy.signal.windows as ssignal_windows
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


def make_actv(actv):
    if actv == "relu":
        return nn.ReLU(inplace=True)
    elif actv == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == "exp":
        return lambda x: torch.exp(x)
    elif actv == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif actv == "tanh":
        return lambda x: torch.tanh(x)
    elif actv == "softplus":
        return lambda x: torch.log(1 + torch.exp(x - 1))
    elif actv == "linear":
        return nn.Identity()
    else:
        raise NotImplementedError("invalid activation function: {:s}".format(actv))


def make_norm3d(name, plane, affine=True, per_channel=True):
    if name == "batch":
        return nn.BatchNorm3d(plane, affine=affine)
    elif name == "instance":
        return nn.InstanceNorm3d(plane, affine=affine)
    elif name == "max":
        return MaxNorm(per_channel=per_channel)
    elif name == "none":
        return nn.Identity()
    else:
        raise NotImplementedError("invalid normalization function: {:s}".format(name))


class MaxNorm(nn.Module):
    """Per-channel normalization by max value"""

    def __init__(self, per_channel=True, eps=1e-8):
        super(MaxNorm, self).__init__()

        self.per_channel = per_channel
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): raw RSD output.

        Returns:
            x (float tensor, (bs, c, d, h, w)): normalized RSD output.
        """
        assert x.dim() == 5, "input should be a 5D tensor, got {:d}D".format(x.dim())

        if self.per_channel:
            x = F.normalize(x, p=float("inf"), dim=(-3, -2, -1))
        else:
            x = F.normalize(x, p=float("inf"), dim=(-4, -3, -2, -1))
        return x


class ResConv3d(nn.Module):
    """Residual block with 3D conv layers"""

    def __init__(
        self,
        in_plane,  # number of input planes
        plane,  # number of intermediate and output planes
        stride=1,  # stride of first conv layer
        actv="leaky_relu",  # activation function
        norm="none",  # normalization function
        affine=True,  # if True, apply learnable affine transform in norm
    ):
        super(ResConv3d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == "none" or not affine else False

        self.conv1 = nn.Conv3d(in_plane, plane, 3, stride, 1, padding_mode="replicate", bias=bias)
        self.norm1 = make_norm3d(norm, plane, affine)
        self.conv2 = nn.Conv3d(plane, plane, 3, 1, 1, padding_mode="replicate", bias=bias)
        self.norm2 = make_norm3d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv3d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm3d(norm, plane, affine)

        self.actv = make_actv(actv)

    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_plane,
        plane,
        stride,
        n_layers,
        actv="relu",
        norm="none",
        affine=False,
    ):
        super(ResBlock3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(ResConv3d(in_plane, plane, stride, actv, norm, affine))
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class RSDBase(nn.Module):
    """Rayleigh-Sommerfield diffraction kernel"""

    def __init__(
        self,
        t=256,  # time dimension of input volume
        d=32,  # depth dimension of output volume
        h=64,  # height dimension of input/output volume
        w=64,  # width dimension of input/output volume
        in_plane=6,  # number of input planes
        wall_size=2,  # wall size (unit: m)
        bin_len=0.02,  # distance covered by a bin (unit: m)
        zmin=0,  # min reconstruction depth w.r.t. the wall (unit: m)
        zmax=2,  # max reconstruction depth w.r.t. the wall (unit: m)
        scale_coef=1,  # scale coefficient for virtual wavelength
        n_cycles=4,  # number of cycles for virtual wavelet
        ratio=0.1,  # relative magnitude under which a frequency is discarded
        actv="linear",  # activation function
        norm="max",  # normalization function
        per_channel=True,  # if True, perform per-channel normalization
        affine=False,  # if True, apply a learnable affine transform in norm
        efficient=False,  # if True, use memory-efficient implementation
        **kwargs,
    ):
        super(RSDBase, self).__init__()
        assert t % 2 == 0, "time dimension must be even"

        self.t = t
        self.d = d
        self.h = h
        self.w = w
        self.in_plane = in_plane
        self.out_plane = in_plane

        self.wall_size = wall_size
        self.bin_len = bin_len
        self.zmin = zmin
        self.zmax = zmax

        self.scale_coef = scale_coef
        self.n_cycles = n_cycles
        self.ratio = ratio

        bin_resolution = bin_len / 3e8  # temporal bin resolution (unit: sec)
        sampling_freq = 1 / bin_resolution  # temporal sampling frequency

        # define virtual wave
        wall_spacing = wall_size / h  # sample spacing on the wall (unit: m)
        lambda_limit = 2 * wall_spacing  # smallest achievable wavelength
        wavelength = scale_coef * lambda_limit

        wave = self._define_wave(wavelength)
        fwave = np.abs(np.fft.fft(wave) / t)[: len(wave) // 2 + 1]
        coef_ratio = fwave / np.max(fwave)

        # retain spectrum [lambda - delta, lambda + delta]
        freq_idx = np.where(coef_ratio > ratio)[0]
        print("{:d}/{:d} frequencies kept in RSD".format(len(freq_idx), len(fwave)))
        freqs = sampling_freq * freq_idx / t
        omegas = 2 * np.pi * freqs  # angular frequencies
        coefs = fwave[freq_idx]  # weight cofficients

        # define RSD kernel
        zdim = np.linspace(zmin, zmax, d + 1)
        zdim = (zdim[:-1] + zdim[1:]) / 2  # mid-point rule

        rsd, tgrid = self._define_rsd(zdim, omegas)
        if not efficient:
            rsd = np.pad(rsd, ((0, 0), (0, 0), (0, h), (0, w)))  # (o, d, h*2, w*2)
        frsd = np.fft.fft2(rsd)  # (o, d, h(*2), w(*2))

        # define phase term in IFFT
        omegas = omegas.reshape(-1, 1, 1, 1)  # (o, 1, 1, 1)
        tgrid = (zdim / 3e8).reshape(1, -1, 1, 1)  # (1, d, 1, 1)
        phase = np.exp(1j * omegas * tgrid)  # (o, d, h/1, w/1)

        # parameters associated with virtual wave
        freq_idx = torch.from_numpy(freq_idx)  # (o,)
        self.register_buffer("freq_idx", freq_idx, persistent=False)

        coefs = torch.from_numpy(coefs.astype(np.float32))
        coefs = coefs.reshape(-1, 1, 1)  # (o, 1, 1)
        self.register_buffer("coefs", coefs, persistent=False)

        # parameters associated with RSD propagation
        frsd = torch.from_numpy(frsd.astype(np.complex64))  # (o, d, h(*2), w(*2))
        self.register_buffer("frsd", frsd, persistent=False)

        phase = torch.from_numpy(phase.astype(np.complex64))  # (o, d, h, w)
        self.register_buffer("phase", phase, persistent=False)

        self.actv = make_actv(actv)
        self.norm = make_norm3d(norm, in_plane, affine, per_channel)

    def _define_wave(self, wavelength):
        # discrete samples of the virtual wavelet
        samples = round((self.n_cycles * wavelength) / self.bin_len)
        n_cycles = samples * self.bin_len / wavelength
        idx = np.arange(samples) + 1

        # complex-valued sinusoidal wave modulated by gaussian envelope
        sinusoid = np.exp(1j * 2 * np.pi * n_cycles * idx / samples)
        # scipy>=1.13 moved this out of the top-level `scipy.signal` namespace
        # (upstream code calls `scipy.signal.gaussian`, which is removed);
        # `scipy.signal.windows.gaussian` is the same function, mechanical fix.
        win = ssignal_windows.gaussian(samples, (samples - 1) / 2 * 0.3)
        wave = sinusoid * win

        # pad wave to the same length as time-domain histograms
        if len(wave) < self.t:
            wave = np.pad(wave, (0, self.t - len(wave)))
        return wave

    def _define_rsd(self, zdim, omegas):
        width = self.wall_size / 2
        ydim = np.linspace(width, -width, self.h + 1)
        xdim = np.linspace(-width, width, self.w + 1)
        ydim = (ydim[:-1] + ydim[1:]) / 2  # mid-point rule
        xdim = (xdim[:-1] + xdim[1:]) / 2
        [zgrid, ygrid, xgrid] = np.meshgrid(zdim, ydim, xdim, indexing="ij")

        # a grid of distance between wall center and scene points
        # (assume light source lies at wall center)
        dgrid = np.sqrt((xgrid**2 + ygrid**2) + zgrid**2)  # (d, h, w)
        tgrid = zgrid / 3e8  # (d, h, w)

        # RSD kernel (falloff term is ignored)
        dgrid = dgrid.reshape(1, len(zdim), self.h, self.w)  # (1, d, h, w)
        omegas = omegas.reshape(-1, 1, 1, 1)  # (o, 1, 1, 1)
        rsd = np.exp(1j * omegas / 3e8 * dgrid) / dgrid  # (o, d, h, w)
        return rsd, tgrid

    def forward(self, x, sqrt=True):
        raise NotImplementedError("RSD forward pass not implemented")


class RSDEfficient(RSDBase):
    """
    NOTE: this implementation does not zero-pad RSD kernel for efficiency.
    This results in sparser frequency sampling (4x memory saving) and
    slightly noiser reconstruction results (with aliasing).
    """

    def __init__(self, **kwargs):
        super(RSDEfficient, self).__init__(efficient=True, **kwargs)

    def forward(self, x, sqrt=True):
        """
        Args:
            x (float tensor, (bs, c, t, h, w)): input time-domain features.
            sqrt (bool): if True, take the square root before normalization.

        Returns:
            x (float tensor, (bs, c, d, h, w)): output space-domain features.
        """
        bs, c, t, h, w = x.shape
        assert t == self.t, "time dimension should be {:d}, got {:d}".format(self.t, t)
        assert h == self.h, "height dimension should be {:d}, got {:d}".format(self.h, h)
        assert w == self.w, "width dimension should be {:d}, got {:d}".format(self.w, w)
        assert c == self.in_plane, "feature dimension should be {:d}, got {:d}".format(
            self.in_plane, c
        )

        # propagate each feature dimension independently
        tdata = x.flatten(0, 1)  # (bs*c, t, h, w)

        ## Step 1: convert measurement into FDH
        fdata = fft.rfft(tdata, dim=1)  # (bs*c, t//2+1, h, w)
        fdata = fdata[:, self.freq_idx]  # (bs*c, o, h, w)

        ## Step 2: define source phasor field
        phasor = self.coefs * fdata  # (bs*c, o, h, w)
        fsrc = fft.fftn(phasor, s=[-1, -1])  # (bs*c, o, h, w)

        ## Step 3: RSD propagation
        # WARNING: PyTorch is buggy when distributing complex tensors
        # here is a temporary workaround
        frsd, phase = self.frsd, self.phase
        if frsd.dim() == 5:
            frsd = torch.complex(frsd[..., 0], frsd[..., 1])
        if phase.dim() == 5:
            phase = torch.complex(phase[..., 0], phase[..., 1])
        fdst = fsrc.unsqueeze(2) * frsd
        fdst = phase * fdst  # (bs*c, o, d, h, w)
        fvol = torch.sum(fdst, 1)  # (bs*c, d, h, w)
        tvol = fft.ifftn(fvol, s=[-1, -1])
        tvol = fft.ifftshift(tvol, dim=(-2, -1))

        ## Step 4: post-process data
        tvol = torch.abs(tvol)  # (bs*c, d, h, w)
        if not sqrt:
            tvol = tvol**2

        x = tvol.reshape(bs, c, self.d, h, w)
        x = self.actv(self.norm(x))
        return x


class RSDNet(nn.Module):
    """RSD encoder"""

    def __init__(
        self,
        in_plane=1,  # number of input planes
        plane=6,  # number of planes prior to propagation
        in_block=True,  # if True, learn conv block before RSD
        ds=False,  # if True, down-sample the input
        rsd_layer=None,  # RSD kernel
        actv="leaky_relu",  # activation function
        norm="none",  # normalization function
        affine=False,  # if True, apply learnable affine transform in norm
        **kwargs,
    ):
        super(RSDNet, self).__init__()

        bias = True if norm == "none" or not affine else False
        stride = 2 if ds else 1

        if in_block:
            self.in_block = nn.Sequential(
                nn.Conv3d(in_plane, plane, 3, stride, 1, bias=bias),
                make_norm3d(norm, plane, affine),
                make_actv(actv),
                nn.Conv3d(plane, plane, 3, 1, 1, bias=bias),
                make_norm3d(norm, plane, affine),
                make_actv(actv),
            )
            in_plane = plane
        else:
            self.in_block = nn.Identity()

        self.rsd_layer = rsd_layer
        self.out_block = nn.Sequential(
            nn.Conv3d(in_plane, plane, 3, 1, 1, bias=bias),
            make_norm3d(norm, plane, affine),
            make_actv(actv),
            nn.Conv3d(plane, plane, 3, 1, 1),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.rsd_layer(x)
        x = self.out_block(x)
        return x


def build_rsdnet():
    """Tiny-config RSD encoder (small t/h/w/d so the FFT-based RSD kernel
    stays cheap to build+trace; `in_block` down-samples h,w by 2 with
    stride-2 3D convs before the RSD propagation layer, matching the repo's
    `configs/unsup/train_ch1_128.yaml` shape)."""
    rsd_layer = RSDEfficient(
        t=32,
        d=6,
        h=8,
        w=8,
        in_plane=6,
        wall_size=2,
        bin_len=0.02,
        zmin=0.5,
        zmax=2,
        scale_coef=1,
        actv="linear",
        norm="none",
        affine=False,
    )
    return RSDNet(
        in_plane=1,
        plane=6,
        in_block=True,
        ds=True,
        rsd_layer=rsd_layer,
        actv="leaky_relu",
        norm="none",
        affine=False,
    )


def example_input_rsdnet():
    # (bs, in_plane, t_in, h_in, w_in): `in_block`'s stride-2 3D convs halve
    # ALL three spatial dims (t, h, w alike), so t_in must be 2x the RSD
    # layer's t=32 too (not just h,w) to land on (t=32, h=8, w=8) post-conv.
    return (torch.randn(1, 1, 64, 16, 16),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "D-NLOS RSD Encoder (Physics-to-the-Rescue)",
        "build_rsdnet",
        "example_input_rsdnet",
        2022,
        "computational-imaging",
    ),
]
