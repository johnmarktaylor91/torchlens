# SOURCE: vendored from cq615/Deep-MRI-Reconstruction @ master
#
# https://github.com/cq615/Deep-MRI-Reconstruction
# https://raw.githubusercontent.com/cq615/Deep-MRI-Reconstruction/master/cascadenet_pytorch/model_pytorch.py
# https://raw.githubusercontent.com/cq615/Deep-MRI-Reconstruction/master/cascadenet_pytorch/kspace_pytorch.py
#
# "Deep Cascade CNN" / DC-CNN / "D5C5" (Schlemper, Caballero, Hajnal, Price, Rueckert,
# "A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction",
# IEEE TMI 2018 -- the official author PyTorch port of the original Theano/Lasagne repo).
# `DnCn` chains `nc` (default 5) residual CNN de-aliasing blocks (`conv_block`, each `nd`
# stacked Conv2d+LeakyReLU layers, default nd=5 -- hence "D5C5") with a k-space
# `DataConsistencyInKspace` re-projection layer after every block, so the network
# alternates "denoise in image domain" / "re-impose measured k-space samples" exactly as
# in the paper. `DnCn`, `conv_block`, and `DataConsistencyInKspace` (plus its
# `data_consistency` helper) are transcribed verbatim from the real repo files above.
#
# COMPAT FIX (not an architecture change): the original `DataConsistencyInKspace.perform`
# calls the pre-1.8 `torch.fft(x, 2, normalized=...)` / `torch.ifft(...)` API, which
# operated on real tensors with a trailing size-2 (real, imag) axis. That API was removed
# in modern torch (2.8 here). `_fft2c` / `_ifft2c` below reproduce the identical ortho-
# normalized 2D FFT over the same two spatial axes using the current `torch.fft.fft2` /
# `torch.fft.ifft2` on `torch.view_as_complex` / `torch.view_as_real` round-trips -- same
# math, same axes, same normalization, only the removed-API call is swapped out.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From cascadenet_pytorch/kspace_pytorch.py (+ torch.fft removed-API compat shim)
# ---------------------------------------------------------------------------
def _fft2c(x, normalized):
    # x: real tensor (..., 2) where last axis is (real, imag); FFT2 over the two axes
    # immediately preceding the trailing (real, imag) axis -- matches the old
    # torch.fft(x, signal_ndim=2, normalized=...) contract.
    norm = "ortho" if normalized else "backward"
    xc = torch.view_as_complex(x.contiguous())
    kc = torch.fft.fft2(xc, dim=(-2, -1), norm=norm)
    return torch.view_as_real(kc)


def _ifft2c(x, normalized):
    norm = "ortho" if normalized else "backward"
    xc = torch.view_as_complex(x.contiguous())
    kc = torch.fft.ifft2(xc, dim=(-2, -1), norm=norm)
    return torch.view_as_real(kc)


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm="ortho"):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == "ortho"
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = _fft2c(x, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = _ifft2c(out, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


# ---------------------------------------------------------------------------
# From cascadenet_pytorch/model_pytorch.py
# ---------------------------------------------------------------------------
def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl="lrelu", conv_dim=2, n_out=None):
    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == "relu" else lrelu

    layers = [conv_1, nll()]
    for i in range(nd - 2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DnCn(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyInKspace(norm="ortho"))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


def build_dccnn_tiny() -> DnCn:
    # real repo default hyperparameters: n_channels=2 (real/imag), nc=5 cascades,
    # nd=5 conv layers per cascade ("D5C5"); nf kept at the paper's default (32) since
    # it is a per-conv channel width, not an architectural knob.
    return DnCn(n_channels=2, nc=5, nd=5).eval()


def example_input_dccnn_tiny():
    # Real model needs 3 tensors: x (zero-filled image estimate, real/imag as channel
    # dim 2), k (fully / under-sampled k-space, same shape), m (binary sampling mask,
    # broadcastable to same shape) -- all shape (n, 2, nx, ny) per the docstring in
    # DataConsistencyInKspace.perform.
    n, nx, ny = 1, 32, 32
    x = torch.randn(n, 2, nx, ny)
    k = torch.randn(n, 2, nx, ny)
    m = torch.randint(0, 2, (n, 2, nx, ny)).float()
    return (x, k, m)


MENAGERIE_ENTRIES = [
    ("DeepCascadeCNN", "build_dccnn_tiny", "example_input_dccnn_tiny", 2018, "vendored-pytorch"),
]
