# SOURCE: vendored from j-duan/VS-Net @ master
# https://github.com/j-duan/VS-Net/blob/master/architecture.py
# https://github.com/j-duan/VS-Net/blob/master/data/transforms.py (complex_multiply helper)
#
# VS-Net: variable splitting network for accelerated parallel MRI reconstruction.
# The real entry point is vs_net.py: `from architecture import network`, then
# `model = network(alfa=1, beta=1, cascades=5)` is trained/evaluated directly.
# Architecture is reproduced verbatim from architecture.py; the ONLY change is
# porting the two `torch.fft(x, 2, normalized=True)` / `torch.ifft(...)` calls
# (removed pre-1.8 API) to the modern `torch.fft.fftn`/`ifftn` on
# `torch.view_as_complex` tensors, which is numerically the same 2D orthonormal
# FFT over the last two spatial dims of a (..., 2)-real/imag tensor. No
# architectural change: same cascade of CNN denoiser + data-consistency (k-space
# fidelity term against undersampling mask/coil sensitivities) + weighted-average
# blocks as the original.
import torch
import torch.nn as nn


def complex_multiply(x, y, u, v):
    """(x+iy) * (u+iv) = (x*u - y*v) + (x*v + y*u)i, returned stacked on a new last dim."""
    z1 = x * u - y * v
    z2 = x * v + y * u
    return torch.stack((z1, z2), dim=-1)


def _fft2_last2(data):
    """Port of the original `torch.fft(data, 2, normalized=True)` (old pre-1.8 API):
    orthonormal 2D FFT over dims (-3, -2) of a tensor whose last dim (size 2) holds
    (real, imag). Uses the modern complex-tensor torch.fft API for identical math.
    """
    complex_data = torch.view_as_complex(data.contiguous())
    complex_out = torch.fft.fftn(complex_data, dim=(-2, -1), norm="ortho")
    return torch.view_as_real(complex_out)


def _ifft2_last2(data):
    """Port of the original `torch.ifft(data, 2, normalized=True)`."""
    complex_data = torch.view_as_complex(data.contiguous())
    complex_out = torch.fft.ifftn(complex_data, dim=(-2, -1), norm="ortho")
    return torch.view_as_real(complex_out)


class dataConsistencyTerm(nn.Module):
    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self, x, k0, mask, sensitivity):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = complex_multiply(
            x[..., 0].unsqueeze(1),
            x[..., 1].unsqueeze(1),
            sensitivity[..., 0],
            sensitivity[..., 1],
        )

        k = _fft2_last2(x)

        v = self.noise_lvl
        if v is not None:  # noisy case
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0

        # ### backward op ### #
        x = _ifft2_last2(out)

        Sx = complex_multiply(x[..., 0], x[..., 1], sensitivity[..., 0], -sensitivity[..., 1]).sum(
            dim=1
        )

        SS = complex_multiply(
            sensitivity[..., 0],
            sensitivity[..., 1],
            sensitivity[..., 0],
            -sensitivity[..., 1],
        ).sum(dim=1)

        return Sx, SS


class weightedAverageTerm(nn.Module):
    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        x = self.para * cnn + (1 - self.para) * Sx
        return x


class cnn_layer(nn.Module):
    def __init__(self):
        super(cnn_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1, bias=True),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class network(nn.Module):
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(network, self).__init__()

        self.cascades = cascades
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []

        for i in range(cascades):
            conv_blocks.append(cnn_layer())
            dc_blocks.append(dataConsistencyTerm(alfa))
            wa_blocks.append(weightedAverageTerm(beta))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)

    def forward(self, x, k, m, c):
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_vsnet():
    # Small cascade depth (2 instead of 5) keeps this trace-friendly while
    # exercising every distinct block type (cnn_layer, dataConsistencyTerm,
    # weightedAverageTerm) at least twice.
    return network(alfa=1, beta=1, cascades=2).eval()


def example_input_vsnet():
    # x: undersampled complex image (B, H, W, 2). k: undersampled k-space
    # (B, H, W, 2). m: sampling mask (B, H, W, 1) broadcastable against k-space.
    # c: coil sensitivity maps (B, coils, H, W, 2).
    b, h, w, coils = 1, 16, 16, 2
    x = torch.randn(b, h, w, 2)
    k = torch.randn(b, h, w, 2)
    m = (torch.rand(b, h, w, 1) > 0.5).float()
    c = torch.randn(b, coils, h, w, 2)
    return (x, k, m, c)


MENAGERIE_ENTRIES = [
    ("VS-Net", "build_vsnet", "example_input_vsnet", 2019, MENAGERIE_ZOO),
]
