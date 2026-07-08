# SOURCE: vendored from vios-s/CMRxRECON_Challenge_EDIPO @ main
# https://raw.githubusercontent.com/vios-s/CMRxRECON_Challenge_EDIPO/main/models/cinenet.py
# https://raw.githubusercontent.com/vios-s/CMRxRECON_Challenge_EDIPO/main/models/datalayer.py
# https://raw.githubusercontent.com/vios-s/CMRxRECON_Challenge_EDIPO/main/models/unet.py
# https://raw.githubusercontent.com/vios-s/CMRxRECON_Challenge_EDIPO/main/utils/fft.py
#
# CineNet (Kuestner et al.-lineage cascaded-U-Net dynamic-MRI reconstruction network,
# as implemented by the VIOS-S group for the MICCAI CMRxRecon challenge's EDIPO
# submission): an unrolled-optimization reconstruction network that alternates
# `CineNetBlock` regularizer stages (U-Net(s) applied in the spatio-temporal x-f/y-f
# planes, XT plane, 2D-per-frame, or full 3D volume, selected by `dynamic_type`) with
# `DCLayer` k-space data-consistency stages, matching the cascaded-CNN-with-DC-blocks
# family used across cardiac cine MRI reconstruction. The classes below (`CineNet`,
# `CineNetBlock`, `DCLayer`, `Unet`, `ConvBlock`, `TransposeConvBlock`, plus the
# `fft1c`/`ifft1c`/`fftshift`/`ifftshift`/`roll`/`roll_one_dim` FFT helpers from
# utils/fft.py and utils/math.py that `datalayer.py` needs) are copied verbatim from
# the official repo files. The only changes: the `sys.path.append("..")` +
# `from utils.fft import *` / `from models import Unet` package-relative imports are
# replaced by inlining the actual helper/Unet code into this single file (same
# functions, same bodies, no architectural change), and the unused `np`/`Tuple`/
# `Callable` imports from cinenet.py are dropped.
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

MENAGERIE_ZOO = "vendored-pytorch"


# ---- utils/math.py + utils/fft.py (verbatim) ----


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft1c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 1 dimensional Fast Fourier Transform.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-2])
    data = torch.view_as_real(torch.fft.fft(torch.view_as_complex(data), dim=-1, norm=norm))
    data = fftshift(data, dim=[-2])

    return data


def ifft1c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 1-dimensional Inverse Fast Fourier Transform.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-2])
    data = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(data), dim=-1, norm=norm))
    data = fftshift(data, dim=[-2])

    return data


def fftnc(data, norm="ortho", dim=(-4, -3)):
    """
    Apply centered 2 dimensional FFT for complex valued data (2 channels real-valued).
    """
    assert data.shape[-1] == 2, (
        "The last dimension should have a size of 2 corresponding to the real and imaginary parts"
    )

    data = ifftshift(data, dim=dim)
    data = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(data), dim=(dim[0] + 1, dim[1] + 1), norm=norm)
    )
    data = fftshift(data, dim=dim)

    return data


def ifftnc(data, norm="ortho", dim=(-4, -3)):
    assert data.shape[-1] == 2, (
        "The last dimension should have a size of 2 corresponding to the real and imaginary parts"
    )
    data = ifftshift(data, dim=dim)
    data = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(data), dim=(dim[0] + 1, dim[1] + 1), norm=norm)
    )
    data = fftshift(data, dim=dim)

    return data


# ---- models/unet.py (verbatim) ----


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234-241.
    Springer, 2015.
    """

    def __init__(
        self,
        chans: int = 32,
        num_pool_layers: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        dims: int = 2,
    ):
        super().__init__()

        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.dims = dims

        assert dims in [2, 3], "Dimensions must be either 2 or 3"

        if dims == 2:
            conv_op = nn.Conv2d
        if dims == 3:
            conv_op = nn.Conv3d

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, dims)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, dims))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob, dims)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, dims))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, dims))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, dims))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob, dims),
                conv_op(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.dims == 2:
            pool_op = F.avg_pool2d
        if self.dims == 3:
            pool_op = F.avg_pool3d

        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = pool_op(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad if needed to handle odd input dimensions
            if self.dims == 2:
                padding = [0, 0, 0, 0]
            if self.dims == 3:
                padding = [0, 0, 0, 0, 0, 0]

            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if self.dims == 3:
                if output.shape[-3] != downsample_layer.shape[-3]:
                    padding[5] = 1  # padding temporal end
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding)

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, dims: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.dims = dims

        if self.dims == 2:
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            drop_op = nn.Dropout2d

        if self.dims == 3:
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
            drop_op = nn.Dropout3d

        self.layers = nn.Sequential(
            conv_op(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            drop_op(drop_prob),
            conv_op(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            drop_op(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int, dims: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dims = dims

        if self.dims == 2:
            up_conv_op = nn.ConvTranspose2d
            norm_op = nn.InstanceNorm2d

        if self.dims == 3:
            up_conv_op = nn.ConvTranspose3d
            norm_op = nn.InstanceNorm3d

        self.layers = nn.Sequential(
            up_conv_op(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


# ---- models/datalayer.py (verbatim) ----


class DCLayer(nn.Module):
    """
    Data Consistency layer from DC-CNN, apply for single coil mainly
    """

    def __init__(self, lambda_init=None, learnable=True):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(DCLayer, self).__init__()
        if lambda_init is None:
            import math

            lambda_init = math.log(math.exp(1) - 1.0) / 1.0
        self.learnable = learnable
        self.lambda_ = nn.Parameter(torch.ones(1) * lambda_init, requires_grad=self.learnable)

    def forward(self, x, y, mask):
        A_x = fftnc(x)
        k_dc = (1 - mask) * A_x + mask * (
            self.lambda_.to(x.device) * A_x + (1 - self.lambda_.to(x.device)) * y
        )
        x_dc = ifftnc(k_dc)
        return x_dc

    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4g}, learnable={self.learnable}"


# ---- models/cinenet.py (verbatim) ----


class CineNet(nn.Module):
    def __init__(
        self,
        num_cascades: int,
        chans: int,
        pools: int,
        dynamic_type: str,
        weight_sharing: bool,
        datalayer: nn.Module,
        save_space: bool,
        reset_cache: bool,
    ):
        super().__init__()

        self.num_cascades = num_cascades

        if dynamic_type in ["XF", "XT"]:
            if weight_sharing:
                self.model = Unet(chans, pools, dims=2)
            else:
                self.model = nn.ModuleList([Unet(chans, pools, dims=2), Unet(chans, pools, dims=2)])
        elif dynamic_type == "3D":
            self.model = Unet(chans, pools, dims=3)
        else:
            self.model = Unet(chans, pools, dims=2)

        self.gradR = torch.nn.ModuleList(
            [
                CineNetBlock(self.model, dynamic_type, weight_sharing)
                for _ in range(self.num_cascades)
            ]
        )
        self.gradD = torch.nn.ModuleList([datalayer for _ in range(self.num_cascades)])

        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space

        self.reset_cache = reset_cache

    def forward(self, x, y, mask):
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
            y: Input k-space, shape (b, w, h, t, ch)
            mask: Input mask, shape (b, w, h, ch)
        Returns:
            x: Reconstructed image, shape (b, t, w, h, ch)
        """

        x = x.float()  # [b, w, h, t, ch]
        y = y.float()  # [b, t, w, h, ch]
        mask = mask.unsqueeze(1).float()  # [b, 1, w, h, 1]
        x_all = [x]
        x_half_all = []
        for i in range(self.num_cascades):
            x_thalf = x - self.gradR[i % self.num_cascades](x)  # [b, t, w, h, ch]
            x_half_all.append(x_thalf)  # [b, t, w, h, ch]
            x = self.gradD[i % self.num_cascades](x_thalf, y, mask)  # [b, w, h, t, ch]
            x_all.append(x)

        return x_all[-1]

    def forward_save_space(self, x, y, mask):
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
            y: Input k-space, shape (b, w, h, t, ch)
            mask: Input mask, shape (b, w, h, ch)
        Returns:
            x: Reconstructed image, shape (b, t, w, h, ch)
        """
        x = x.float()  # [b, w, h, t, ch]
        y = y.float()  # [b, w, h, t, ch]
        mask = mask.unsqueeze(3).float()  # [b, w, h, t, ch]

        for i in range(self.num_cascades):
            x_thalf = x - self.gradR[i % self.num_cascades](x)  # [b, w, h, t, ch]
            x = self.gradD[i % self.num_cascades](x_thalf, y, mask)  # [b, w, h, t, ch]

            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()

        return x


class CineNetBlock(nn.Module):
    def __init__(self, model: nn.Module, dynamic_type: str, weight_sharing: bool):
        super().__init__()

        self.model = model
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing

    def xfyf_transform(self, image_combined: torch.Tensor) -> torch.Tensor:
        """
        Separate input into two volumes in the rotated planes x-f and y-f
        (or x-t, y-t if in 'XT' dynamics mode). After being processed by
        their respective U-Nets, the volumes are then combined back into one.
        """
        b, h, w, t, ch = image_combined.shape

        # Subtract the image temporal average for numerical stability
        image_temp = image_combined.clone()
        image_mean = torch.stack(t * [torch.mean(image_temp, dim=-2)], dim=-2)

        x = image_combined - image_mean

        if self.dynamic_type == "XF":
            # Apply temporal FFT
            x = fft1c(x)
            x = x.permute(0, 3, 1, 2, 4)  # b,t,h,w,2

        # Reshape to xf, yf planes
        xf = (
            x.clone().permute(0, 2, 4, 3, 1).reshape(b * h, 2, w, t)
        )  # [b, h, ch, w, t] -> [b*h, ch, w, t]
        yf = (
            x.clone().permute(0, 3, 4, 2, 1).reshape(b * w, 2, h, t)
        )  # [b, w, ch, h, t] -> [b*w, ch, h, t]

        # UNet opearting on temporal transformed xf, yf-domain
        if self.weight_sharing:
            xf = self.model(xf)
            yf = self.model(yf)
        else:
            model_xf, model_yf = self.model
            xf = model_xf(xf)
            yf = model_yf(yf)

        # Reshape from xf, yf
        xf_r = xf.view(b, h, 1, 2, w, t).permute(0, 5, 2, 1, 4, 3)  # b,t,1,h,w,2
        yf_r = yf.view(b, w, 1, 2, h, t).permute(0, 5, 2, 4, 1, 3)  # b,t,1,h,w,2

        out = 0.5 * (xf_r + yf_r)

        if self.dynamic_type == "XF":
            # Apply temporal IFFT
            out = out.permute(0, 2, 3, 4, 1, 5)  # b,1,h,w,t,2
            out = ifft1c(out)
            out = out.permute(0, 4, 1, 2, 3, 5)  # b,t,1,h,w,2

        # Residual connection
        image_mean = image_mean.permute(0, 3, 1, 2, 4).unsqueeze(2)
        out = out + image_mean

        return out.squeeze(2).permute(0, 2, 3, 1, 4)

    def forward(self, image_pred: torch.Tensor) -> torch.Tensor:
        b, h, w, t, ch = image_pred.shape
        x = image_pred.clone()

        if self.dynamic_type in ["XF", "XT"]:
            model_out = self.xfyf_transform(x)  # [b, t, h, w, ch]

        elif self.dynamic_type == "2D":
            # Batch dimension b=1. Make first dimension time so
            # that each slice is trained independently. This is
            # similar to static MRI reconstruction.
            image_in = image_pred.permute(0, 3, 4, 2, 1).reshape(b * t, ch, h, w)
            model_out = self.model(image_in).reshape(b, t, h, w, ch)  # [b, t, h, w, ch]

        elif self.dynamic_type == "3D":
            # In this mode the whole spatio-temporal volume is
            # processed by a 3D U-Net at once.
            image_in = image_pred.permute(0, 4, 3, 2, 1).reshape(b, ch, t, h, w)
            model_out = self.model(image_in).reshape(b, t, h, w, ch)  # [b, t, h, w, ch]

        else:
            raise ValueError(f"Unknown dynamic type {self.dynamic_type}")

        return model_out  # [b, t, h, w, ch]


def build_cinenet():
    # Tiny menagerie-scale config (chans=4, pools=2, num_cascades=1 vs. the
    # repo's larger chans=16-18/pools=3/num_cascades=4-5 defaults) using
    # dynamic_type='XF' -- the module's OWN default (see the real
    # `CineNetModule.__init__(..., dynamic_type: str = 'XF', ...)` /
    # `add_model_specific_args` in pl_modules/cinenet_module.py) -- with
    # weight_sharing=True and a real DCLayer for data consistency, matching the
    # actual CineNet(...) construction used there.
    return CineNet(
        num_cascades=1,
        chans=4,
        pools=2,
        dynamic_type="XF",
        weight_sharing=True,
        datalayer=DCLayer(learnable=True),
        save_space=False,
        reset_cache=False,
    )


def example_input_cinenet():
    # x: zero-filled image estimate (b, h, w, t, ch=2 real/imag); y: undersampled
    # k-space (b, h, w, t, ch=2); mask: sampling mask (b, w, h, ch=1), matching the
    # real forward() signature `forward(self, x, y, mask)`. h=w=t is required here:
    # the top-level forward()'s `mask.unsqueeze(1)` broadcast against the (b,h,w,t,ch)
    # tensors in DCLayer only lines up when spatial dims equal the temporal dim (the
    # real repo's production shapes are non-cubic, e.g. (1,512,256,12,2), which hits
    # this same broadcast; menagerie-scale tracing sidesteps it with a cubic shape).
    torch.manual_seed(0)
    b, h, w, t, ch = 1, 8, 8, 8, 2
    x = torch.randn(b, h, w, t, ch)
    y = torch.randn(b, h, w, t, ch)
    mask = torch.randint(0, 2, (b, h, w, ch)).float()
    return (x, y, mask)


MENAGERIE_ENTRIES = [
    (
        "CineNet (cascaded U-Net dynamic-MRI reconstruction)",
        "build_cinenet",
        "example_input_cinenet",
        2024,
        "vendored-pytorch",
    ),
]
