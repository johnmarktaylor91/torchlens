# SOURCE: vendored from facebookresearch/fastMRI @ main
# (fastmri/models/varnet.py::NormUnet, ::SensitivityModel, ::VarNet, ::VarNetBlock;
#  fastmri/models/unet.py::Unet, ::ConvBlock, ::TransposeConvBlock;
#  fastmri/fftc.py::fft2c_new, ::ifft2c_new, ::fftshift, ::ifftshift, ::roll, ::roll_one_dim;
#  fastmri/math.py::complex_mul, ::complex_conj, ::complex_abs;
#  fastmri/coil_combine.py::rss, ::rss_complex;
#  fastmri/data/transforms.py::batched_mask_center, ::mask_center)
"""End-to-End Variational Network (E2E-VarNet) for accelerated MRI reconstruction
(Sriram et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction",
MICCAI 2020, arXiv:2004.06688). Official repo: https://github.com/facebookresearch/fastMRI
(``fastmri/`` @ main).

The full pipeline: a ``SensitivityModel`` (IFFT of multicoil masked k-space -> a
U-Net-based coil-sensitivity estimate, RSS-normalized), followed by ``num_cascades``
stacked ``VarNetBlock`` refinement cascades. Each cascade applies a soft data-consistency
term against the originally sampled k-space plus a learned regularizer: it reduces the
current multicoil k-space estimate to a single-coil image via sensitivity-weighted coil
combination, denoises with a normalized U-Net (``NormUnet``), then re-expands back to
multicoil k-space with the sensitivity maps before the DC correction. The final image is
the RSS combination of the last cascade's k-space estimate in image space.

The ``fastmri`` package's complex-as-real-pair helpers (``fft2c``/``ifft2c`` centered FFT
via ``torch.fft``, ``complex_mul``/``complex_conj``/``complex_abs``, ``rss``/``rss_complex``)
and ``fastmri.data.transforms.batched_mask_center`` are vendored verbatim alongside the
model classes since the ``fastmri`` pip package is not installed in this environment; they
are pure-torch utility functions with no architectural content. No layer, channel count, or
forward-pass control-flow was changed from the official repo.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# fastmri/fftc.py (verbatim, pure torch)
# ---------------------------------------------------------------------------


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm))
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(data), dim=(-2, -1), norm=norm))
    data = fftshift(data, dim=[-3, -2])

    return data


# ---------------------------------------------------------------------------
# fastmri/math.py (verbatim, pure torch)
# ---------------------------------------------------------------------------


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


# ---------------------------------------------------------------------------
# fastmri/coil_combine.py (verbatim, pure torch)
# ---------------------------------------------------------------------------


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt((data**2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt(complex_abs_sq(data).sum(dim))


# ---------------------------------------------------------------------------
# fastmri/data/transforms.py::mask_center, ::batched_mask_center (verbatim)
# ---------------------------------------------------------------------------


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (not x.shape[0] == mask_to.shape[0]):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


# ---------------------------------------------------------------------------
# fastmri/models/unet.py (verbatim, pure torch)
# ---------------------------------------------------------------------------


class Unet(nn.Module):
    """PyTorch implementation of a U-Net model (Ronneberger et al. 2015)."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


# ---------------------------------------------------------------------------
# fastmri/models/varnet.py (verbatim)
# ---------------------------------------------------------------------------


class NormUnet(nn.Module):
    """Normalized U-Net: a regular U-Net with input normalization for numerical
    stability."""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """Model for learning sensitivity estimation from k-space data. Applies an IFFT
    to multichannel k-space data and then a U-Net to the coil images to estimate coil
    sensitivities."""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        images, batches = self.chans_to_batch_dim(ifft2c_new(masked_kspace))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class VarNet(nn.Module):
    """A full variational network model: soft data consistency combined with a
    U-Net regularizer, applied over ``num_cascades`` stacked ``VarNetBlock`` layers."""

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return rss(complex_abs(ifft2c_new(kspace_pred)), dim=1)


class VarNetBlock(nn.Module):
    """Model block for end-to-end variational network: soft data consistency combined
    with the input model as a regularizer."""

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fft2c_new(complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return complex_mul(ifft2c_new(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term


MENAGERIE_ZOO = "vendored-pytorch"


def build_e2e_varnet():
    return VarNet(num_cascades=2, sens_chans=4, sens_pools=2, chans=4, pools=2)


def example_input_e2e_varnet():
    # (batch, coils, h, w, 2) complex-as-real-pair k-space; multiple of 16 for the
    # NormUnet's pad/unpad logic, small coil/spatial dims for a tiny trace.
    batch, coils, h, w = 1, 2, 16, 16
    masked_kspace = torch.randn(batch, coils, h, w, 2)
    # mask: True where k-space was sampled, broadcastable over (batch, 1, 1, w, 1)
    mask = torch.ones(batch, 1, 1, w, 1, dtype=torch.bool)
    mask[:, :, :, ::2, :] = False
    mask[:, :, :, w // 2 - 2 : w // 2 + 2, :] = True
    return (masked_kspace, mask)


MENAGERIE_ENTRIES = [
    ("E2E-VarNet", "build_e2e_varnet", "example_input_e2e_varnet", 2020, "vendored-pytorch"),
]
