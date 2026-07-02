# SOURCE: vendored from cuiyixin555/DUBLID @ master
#   Vendored files: network.py (Network class), operations.py (conv2, real, real_mul,
#   mul, conj_mul, csquare, pad_to, fft2, ifft2, circ_shift, threshold), parameters.py
#   (hyperparameters).
# https://github.com/cuiyixin555/DUBLID
#
# DUBLID (Deep Unrolling for Blind Deblurring, Li, Eisner, Wang, Fergus, Freeman --
# "An Algorithm Unrolling Approach to Deep Blind Image Deblurring", arXiv:1902.03493).
# An algorithm-unrolling network: each of `num_layer` unrolled iterations alternates a
# closed-form (FFT-domain) least-squares update of the latent image against the current
# kernel estimate, a soft-thresholding nonlinearity, and a closed-form FFT-domain update
# of the blur kernel from the latent image -- all wired through learned per-layer scalar
# step sizes/proximal weights and a stack of learned 3x3 (or KxK for the first layer)
# convolution filters used both for feature extraction and (via `_reflect_filter`) image
# reconstruction. Everything (conv topology, per-layer FFT least-squares solves, kernel
# normalization, image-coefficient reconstruction for gray/color) is untouched from the
# original source.
#
# Minimal API-compat fix (NOT an architecture change): the original `operations.py`
# used the pre-1.8 `torch.rfft(x, signal_ndim=2)` / `torch.irfft(x, signal_ndim=2,
# signal_sizes=size)` API (removed in modern torch), which represented a 2D real FFT as
# a real tensor with a trailing size-2 (real, imag) axis. That is replaced here with the
# numerically equivalent modern `torch.view_as_real(torch.fft.rfft2(x))` /
# `torch.fft.irfft2(torch.view_as_complex(x.contiguous()), s=size)`, which reproduces
# the exact same `[..., 2]` real/imag layout the rest of the module's complex-arithmetic
# helpers (`mul`, `conj_mul`, `csquare`, ...) operate on. Verified bit-for-bit equivalent
# (round-trip reconstructs the input to float32 tolerance) against the legacy API.

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# parameters.py
# ---------------------------------------------------------------------------
#
# Shrunk from the repo defaults (num_layer=10, bounding_box_size=[45, 45], C=16
# feature maps) to a small unrolled depth/kernel bbox so the many chained FFT-domain
# solves stay fast; the algorithm-unrolling structure (per-layer FFT least-squares
# image update, soft-threshold, FFT-domain kernel update) is otherwise unchanged from
# the source. parameters.py is itself a flat module of run-size constants in the
# original repo, so shrinking these in place (rather than threading extra constructor
# args through `Network`) mirrors how the original script is configured.

C = 4  # number of feature maps (repo default 16)
K = 3  # filter width
num_layer = 2  # total number of layer (repo default 10)
epsilon = 1e-8  # to avoid division by zero
bias_init = 0.02
zeta_init = 1.0
prox_scale = 10.0
kernel_prox_init = 1.0
kernel_bias_init = 0.0
kernel_scale = 1e2
kernel_bias_scale = 0.01
eta_init = 1.0
bounding_box_size = [5, 5]  # repo default [45, 45]

# ---------------------------------------------------------------------------
# operations.py
# ---------------------------------------------------------------------------


def conv2(tensor, kernel, mode="same", pad_mode="reflect"):
    """
    Convolution with output size the same as the 'full' convolution scheme
    """

    Hk, Wk = kernel.shape[-2], kernel.shape[-1]
    if mode == "same":
        pad_size = (Wk // 2, Wk - Wk // 2 - 1, Hk // 2, Hk - Hk // 2 - 1)
    elif mode == "full":
        pad_size = (Wk - 1, Wk - 1, Hk - 1, Hk - 1)
    else:  # 'valid'
        pad_size = (0, 0, 0, 0)

    return F.conv2d(F.pad(tensor, pad=pad_size, mode=pad_mode), kernel)


def real(c):
    """
    Extract real part of complex tensor c
    """

    return c[..., 0]


def real_mul(r, c):
    """
    Multiply real tensor r with complex tensor c
    """

    return r.unsqueeze(dim=-1) * c


def mul(c1, c2):
    """
    Complex multiplication between c1 and c2
    """

    r1, i1 = c1[..., 0], c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1 * r2 - i1 * i2
    c = r1 * i2 + i1 * r2

    return torch.stack([r, c], dim=-1)


def conj_mul(c1, c2):
    """
    Complex conjugate of c1 and multiplication with c2
    """

    r1, i1 = c1[..., 0], -c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1 * r2 - i1 * i2
    c = r1 * i2 + i1 * r2

    return torch.stack([r, c], dim=-1)


def csquare(c):
    """
    Square of absolute values of complex numbers
    """

    return c[..., 0] ** 2 + c[..., 1] ** 2


def pad_to(original, size):
    """
    Post-pad last two dimensions to "size"
    """

    original_size = original.size()
    pad = [0, size[1] - original_size[-1], 0, size[0] - original_size[-2]]

    return F.pad(original, pad)


def fft2(signal, size=None):
    """
    Fast Fourier transform on the last two dimensions.
    Modern-API equivalent of the legacy `torch.rfft(signal, signal_ndim=2)`: returns
    the same real/imag-stacked `[..., 2]` layout as the removed API.
    """

    padded = signal if size is None else pad_to(signal, size)

    return torch.view_as_real(torch.fft.rfft2(padded))


def ifft2(signal, size=None):
    """
    Inverse fast Fourier transform on the last two dimensions.
    Modern-API equivalent of the legacy
    `torch.irfft(signal, signal_ndim=2, signal_sizes=size)`.
    """

    return torch.fft.irfft2(torch.view_as_complex(signal.contiguous()), s=size)


def circ_shift(ts, shift):
    """
    Circular shift on the last two dimensions
    """

    sr, sc = shift
    if sc != 0:  # column shift
        ts = torch.cat((ts[..., sc:], ts[..., :sc]), dim=-1)
    if sr != 0:  # row shift
        ts = torch.cat((ts[..., sr:, :], ts[..., :sr, :]), dim=-2)

    return ts


def threshold(x, thr):
    """
    Soft-thresholding operator
    """

    return F.relu(x - thr) - F.relu(-x - thr)


# ---------------------------------------------------------------------------
# network.py
# ---------------------------------------------------------------------------


class Network(nn.Module):
    def __init__(self, device=torch.device("cpu"), channels=1):
        super(Network, self).__init__()
        self.device = device
        weight = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(C, channels, K, K, device=self.device))
        )
        self.weight_list = nn.ParameterList([weight])
        for layer in range(num_layer - 1):
            weight = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(C, C, 3, 3, device=self.device))
            )
            self.weight_list.append(weight)
        self.bias_list = nn.ParameterList()
        for layer in range(num_layer + 2):
            # Rectifier thresholds
            bias = nn.Parameter(torch.full(size=(1, C, 1, 1), device=device, fill_value=bias_init))
            self.bias_list.append(bias)
        self.kernel_bias_list = nn.ParameterList()
        for layer in range(num_layer):
            kernel_bias = nn.Parameter(
                torch.full(size=(1,), device=device, fill_value=kernel_bias_init)
            )
            self.kernel_bias_list.append(kernel_bias)
        self.kernel_prox_list = nn.ParameterList()
        for layer in range(num_layer):
            kernel_prox = nn.Parameter(
                torch.full(size=(1,), device=device, fill_value=kernel_prox_init)
            )
            self.kernel_prox_list.append(kernel_prox)
        self.prox_list = nn.ParameterList()
        for layer in range(num_layer):
            # Proximity to the previous solutions
            zeta = nn.Parameter(torch.full(size=(1, C, 1, 1), device=device, fill_value=zeta_init))
            self.prox_list.append(zeta)
        eta = nn.Parameter(torch.full(size=(1, C, 1, 1), device=device, fill_value=eta_init))
        self.prox_list.append(eta)

    def _reflect_filter(self, w):
        """
        Symmetric reflection of filters arount the origin
        Input:
            w:  CoutxCinxHkxWk
        """

        Cout, Cin, Hk, Wk = w.size()
        w_ref = torch.reshape(w, shape=(-1, Hk * Wk))
        reverse_ind = torch.arange(Hk * Wk - 1, -1, -1, device=self.device)
        w_ref = torch.index_select(w_ref, 1, reverse_ind)
        w_ref = torch.reshape(w_ref, shape=(Cout, Cin, Hk, Wk))

        return w_ref

    def _compute_image_coeffs(self, Fy, Fg, Fw, Fk):
        """
        Solve for image coefficients from feature maps and kernel
        Input:
          Fy:       NxC_inxHfxWf2x2; fourier coefficients of blurred images
          Fg:       NxCxHfxWf2x2; fourier coefficients of feature maps
          Fw:       C_inxCxHfxWf2x2 where C_in=1 for grayscale images
                    and C_in=3 for color images; fourier coefficients of
                    the weights in the first feature extraction layer
          Fk:       NxHfxWf2x2;  fourier coefficients of blur kernels
        Output:
          Fx:       NxHfxWf2x2; fourier coefficients of estimated images
        """

        eta = prox_scale * self.prox_list[-1]
        ec = eta.unsqueeze(dim=-1)
        if Fw.shape[0] == 1:  # grayscale
            Fy = Fy[:, 0]
            num = conj_mul(Fk, Fy) + torch.sum(ec * conj_mul(Fw, Fg), dim=1)
            den = csquare(Fk) + torch.sum(eta * csquare(Fw), dim=1)
            Fx = num / den.unsqueeze(dim=-1)
            Fx = Fx.unsqueeze(dim=1)
        elif Fw.shape[0] == 3:  # color
            Fwr = Fw[0].unsqueeze(dim=0)
            Fwg = Fw[1].unsqueeze(dim=0)
            Fwb = Fw[2].unsqueeze(dim=0)
            Fyr, Fyg, Fyb = Fy[:, 0], Fy[:, 1], Fy[:, 2]
            Crr = csquare(Fk) + torch.sum(eta * csquare(Fwr), dim=1)
            Cgg = csquare(Fk) + torch.sum(eta * csquare(Fwg), dim=1)
            Cbb = csquare(Fk) + torch.sum(eta * csquare(Fwb), dim=1)
            Crg = torch.sum(ec * conj_mul(Fwr, Fwg), dim=1)
            Crb = torch.sum(ec * conj_mul(Fwr, Fwb), dim=1)
            Cgb = torch.sum(ec * conj_mul(Fwg, Fwb), dim=1)
            Br = conj_mul(Fk, Fyr) + torch.sum(ec * conj_mul(Fwr, Fg), dim=1)
            Bg = conj_mul(Fk, Fyg) + torch.sum(ec * conj_mul(Fwg, Fg), dim=1)
            Bb = conj_mul(Fk, Fyb) + torch.sum(ec * conj_mul(Fwb, Fg), dim=1)
            Irr = Cgg * Cbb - csquare(Cgb)
            Igg = Crr * Cbb - csquare(Crb)
            Ibb = Crr * Cgg - csquare(Crg)
            Irg = conj_mul(Cgb, Crb) - real_mul(Cbb, Crg)
            Irb = mul(Crg, Cgb) - real_mul(Cgg, Crb)
            Igb = conj_mul(Crg, Crb) - real_mul(Crr, Cgb)
            den = (
                Crr * (Cgg * Cbb - csquare(Cgb))
                - Cgg * csquare(Crb)
                - Cbb * csquare(Crg)
                + 2 * real(conj_mul(mul(Crg, Cgb), Crb))
            )
            Fxr = real_mul(Irr, Br) + mul(Irg, Bg) + mul(Irb, Bb)
            Fxg = conj_mul(Irg, Br) + real_mul(Igg, Bg) + mul(Igb, Bb)
            Fxb = conj_mul(Irb, Br) + conj_mul(Igb, Bg) + real_mul(Ibb, Bb)
            Fx = torch.stack([Fxr, Fxg, Fxb], dim=1)
            Fx /= den.unsqueeze(dim=1).unsqueeze(dim=-1)

        return Fx

    def forward(self, blurred_image):
        """
        The main deblurring network
        Input:
          blurred_image:    NxC_inxHfxWf
        Output:
          image_pred:       NxC_inxHvxWv; estimated image features
          kernel_pred:      NxHkxWk; estimated kernels
        """

        # Size for kernels
        Hk, Wk = bounding_box_size
        # Size for the 'full' scheme
        N, C_in, Hv, Wv = blurred_image.size()
        # Size for the 'same' scheme
        Hs, Ws = Hv + Hk - 1, Wv + Wk - 1
        # Size for the 'valid' scheme
        Hb, Wb = Hs + Hk - 1, Ws + Wk - 1

        # Feature extraction: filter the blurred images
        wy_list = []
        fy = blurred_image
        # Pad for the maximum possible filter size
        fft_size = (int(ceil(Hb / 64.0) * 64), int(ceil(Wb / 64.0) * 64))
        for layer in range(num_layer):
            w = self.weight_list[layer]
            if layer == 0:
                w_mean = torch.mean(w.view(C, C_in, -1), dim=-1)
                w = w - torch.reshape(w_mean, (C, C_in, 1, 1))
                w0 = torch.transpose(w, dim0=0, dim1=1)
                # At the output end we need to perform convolution instead
                # of correlation as we rely on the convolution theorem
                # for image reconstruction using FFT
                w = self._reflect_filter(w)
            fy = conv2(fy, w)
            fy_padded = pad_to(F.pad(fy, pad=(Wk - 1, Wk - 1, Hk - 1, Hk - 1)), size=fft_size)
            wy_list.append(fy_padded)

        # Deconvolution: estimate the kernel
        delta = torch.zeros((N, Hk, Wk), device=self.device)
        delta[:, Hk // 2, Wk // 2] = 1
        b0 = self.bias_list[0]
        z = threshold(circ_shift(wy_list[-1], (Hk // 2, Wk // 2)), b0)
        Fz = fft2(z)
        k = delta
        for layer in range(num_layer):
            # Retrieve filtered blurred image
            fy = wy_list.pop()
            Ffy = fft2(fy)
            Fk = fft2(k, size=fft_size).unsqueeze(dim=1)
            fft_size = fy.shape[-2:]
            # Update latent image
            zeta = prox_scale * self.prox_list[layer]
            num = zeta.unsqueeze(dim=-1) * conj_mul(Fk, Ffy) + Fz
            den = zeta * csquare(Fk) + 1
            Fg = num / den.unsqueeze(dim=-1)
            # Update surrogate blurred image
            b = self.bias_list[layer + 1]
            Fz = fft2(threshold(ifft2(Fg, size=fft_size), b))
            # Update kernels
            zk = self.kernel_prox_list[layer]
            num = zk * torch.sum(conj_mul(Fz, Ffy), dim=1) + Fk.squeeze(dim=1)
            den = zk * torch.sum(csquare(Fz), dim=1) + 1
            k = ifft2(num / den.unsqueeze(dim=-1), size=fft_size)
            k_max = torch.logsumexp(k.view(N, -1) * kernel_scale, dim=-1)
            k_max = k_max / kernel_scale
            bk = kernel_bias_scale * self.kernel_bias_list[layer]
            k = F.relu(k[:, :Hk, :Wk] - bk * torch.reshape(k_max, (N, 1, 1)))
            k_sum = k.sum(1, keepdim=True).sum(2, keepdim=True)
            k = (k + epsilon * delta) / (k_sum + epsilon)

        # Reconstruct image from feature map
        Fy = fft2(F.pad(blurred_image, pad=(Wk - 1, Wk - 1, Hk - 1, Hk - 1)), size=fft_size)
        Fk = fft2(k, size=fft_size)
        Fw0 = fft2(circ_shift(pad_to(w0, size=fft_size), (K // 2, K // 2)))
        b = self.bias_list[num_layer + 1]
        Fg = fft2(threshold(ifft2(Fg, size=fft_size), b))
        Fx = self._compute_image_coeffs(Fy, Fg, Fw0, Fk)
        image = ifft2(Fx, size=fft_size)
        # Only the interior can be reliably recovered
        image = image[:, :, Hk // 2 : Hk // 2 + Hv, Wk // 2 : Wk // 2 + Wv]
        kernel = k

        return image, kernel


MENAGERIE_ZOO = "vendored-pytorch"


def build_dublid():
    model = Network(device=torch.device("cpu"), channels=1)
    model.eval()
    return model


def example_input_dublid():
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("DUBLID", "build_dublid", "example_input_dublid", 2019, MENAGERIE_ZOO),
]
