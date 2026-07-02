# FAITHFUL PORT of https://github.com/rixez/pytorch_mri_variationalnetwork @ 50749046ccca36f29df523e3d547dee90fe20c7f
# (original framework: PyTorch 0.4/1.x, pre-`torch.fft` module)
#
# Ported files:
#   - models.py    -> `RBFActivation`(+`RBFActivationFunction`), `VnMriReconCell`, `VariationalNetwork`
#     (the Hammernik et al. "Learning a Variational Network for Reconstruction of Accelerated
#     MRI Data" architecture: `num_stages` unrolled gradient-descent cells, each a learned
#     multi-channel convolutional regularizer with a trainable radial-basis-function (RBF)
#     activation, plus a data-consistency term driven by the real MRI forward/adjoint
#     k-space operators).
#   - fft_utils.py -> `fftc2d`/`ifftc2d`/`complex_mul`/`conj` (the centered 2-D FFT/adjoint
#     machinery used inside `mri_forward_op`/`mri_adjoint_op`).
#   - misc_utils.py -> `zero_mean_norm_ball` (the zero-mean/unit-norm convolution-kernel
#     projection used at initialization when `do_prox_map=True`).
#
# Why a PORT and not a vendor: the real `fft_utils.py` calls the legacy
# `torch.fft(x, 2, normalized=True)` / `torch.ifft(x, 2, normalized=True)` free functions
# (torch<1.8's `signal_ndim`+`normalized` FFT API). That API was REMOVED in torch 1.8+ in
# favor of the `torch.fft` submodule -- `torch.fft(...)` is no longer callable at all on any
# torch this package supports, so the file cannot run verbatim. Every other mechanism
# (unrolled cell structure, RBF activation forward+backward, forward/adjoint k-space
# operators, zero-mean-norm-ball kernel projection, `pytorch_lightning.LightningModule`
# wrapping dropped since we only need `nn.Module.forward` for tracing) is transcribed
# unchanged; only the two FFT calls are updated to the modern equivalent
# (`torch.fft.fft2(..., norm="ortho")` / `torch.fft.ifft2(..., norm="ortho")` applied to a
# `torch.view_as_complex` reinterpretation of the same last-dim-2 real/imaginary layout,
# which is the direct successor of the old `normalized=True` 2-D FFT semantics).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "ported-pytorch"

DEFAULT_OPTS = {
    "kernel_size": 11,
    "features_in": 1,
    "features_out": 24,
    "do_prox_map": True,
    "pad": 11,
    "vmin": -1.0,
    "vmax": 1.0,
    "lamb_init": 1.0,
    "num_act_weights": 31,
    "init_type": "linear",
    "init_scale": 0.04,
    "sampling_pattern": "cartesian",
    "num_stages": 10,
    "seed": 1,
    "activation": "rbf",
    "loss_type": "complex",
}


# ---- fft_utils.py (ported: modern torch.fft replacing removed torch.fft()/torch.ifft()) ----


def fftc2d(x):
    """Centered 2-D Fourier transform on axes (-3, -2), shape [..., H, W, 2] (real/imag).

    Ported from the original `torch.fft(x, 2, normalized=True)` (torch<1.8 API, removed in
    modern torch) to `torch.fft.fft2` with `norm="ortho"` -- the direct modern successor of
    the old `normalized=True` semantics -- applied via `view_as_complex`/`view_as_real` to
    preserve the exact same last-dim-2 real/imaginary tensor layout used throughout this file.
    """
    xc = torch.view_as_complex(x.contiguous())
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.fft2(xc, dim=(-2, -1), norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return torch.view_as_real(xc)


def ifftc2d(x):
    """Centered inverse 2-D Fourier transform, the adjoint of `fftc2d` (see its docstring
    for the removed-API porting note)."""
    xc = torch.view_as_complex(x.contiguous())
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.ifft2(xc, dim=(-2, -1), norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return torch.view_as_real(xc)


def conj(x):
    """Complex conjugate of a 2-channel complex torch tensor."""
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_mul(x, y):
    """Complex multiply of two 2-channel complex torch tensors."""
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


# ---- misc_utils.py (verbatim) ----


def zero_mean_norm_ball(
    x, zero_mean=True, normalize=True, norm_bound=1.0, norm="l2", mask=None, axis=(0, ...)
):
    """Project onto zero-mean and norm-one ball (convolution kernel proximal map)."""
    if mask is None:
        shape = []
        for i in range(len(x.shape)):
            if i in axis:
                shape.append(x.shape[i])
            else:
                shape.append(1)
        mask = torch.ones(shape, dtype=torch.float32)

    x_masked = x * mask

    if zero_mean:
        x_mean = torch.mean(x_masked, dim=axis, keepdim=True)
        x_zm = x_masked - x_mean
    else:
        x_zm = x_masked

    if normalize:
        if norm == "l2":
            magnitude = torch.sqrt(torch.sum(torch.square(x_zm), dim=axis, keepdim=True))
            x_proj = x_zm / magnitude * norm_bound
        else:
            raise ValueError("Norm '%s' not defined." % norm)

    return x_proj


# ---- models.py (verbatim, with the two FFT call sites routed through the ported fft_utils) ----


class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """Forward pass for RBF activation.

        input: NxCxHxW
        w: 1 x C x 1 x 1 x #kernels
        mu: #kernels
        sigma: scalar
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:, :, :, :, i] * torch.exp(-torch.square(input - mu[i]) / (2 * sigma**2))
            output += tmp
            rbf_grad_input += tmp * (-(input - mu[i])) / (sigma**2)
        del tmp
        ctx.save_for_backward(input, w, mu, sigma, rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        grad_input = grad_output * rbf_grad_input

        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output * torch.exp(-torch.square(input - mu[i]) / (2 * sigma**2))).sum(
                (0, 2, 3)
            )
            grad_w[:, :, :, :, i] = tmp.view(w.shape[0:-1])

        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    """RBF activation function with trainable weights."""

    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(
            kwargs["vmin"], kwargs["vmax"], kwargs["num_act_weights"], dtype=np.float32
        )
        mu = np.linspace(
            kwargs["vmin"], kwargs["vmax"], kwargs["num_act_weights"], dtype=np.float32
        )
        self.sigma = 2 * kwargs["vmax"] / (kwargs["num_act_weights"] - 1)
        self.sigma = torch.tensor(self.sigma)
        if kwargs["init_type"] == "linear":
            w_0 = kwargs["init_scale"] * x_0
        elif kwargs["init_type"] == "tv":
            w_0 = kwargs["init_scale"] * np.sign(x_0)
        elif kwargs["init_type"] == "relu":
            w_0 = kwargs["init_scale"] * np.maximum(x_0, 0)
        elif kwargs["init_type"] == "student-t":
            alpha = 100
            w_0 = kwargs["init_scale"] * np.sqrt(alpha) * x_0 / (1 + 0.5 * alpha * x_0**2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs["init_type"])
        w_0 = np.reshape(w_0, (1, 1, 1, 1, kwargs["num_act_weights"]))
        w_0 = np.repeat(w_0, kwargs["features_out"], 1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self, x):
        if not self.mu.device == x.device:
            self.mu = self.mu.to(x.device)
            self.sigma = self.sigma.to(x.device)

        output = self.rbf_act(x, self.w, self.mu, self.sigma)

        return output


class VnMriReconCell(nn.Module):
    """One cell of the variational network: a learned multi-channel convolutional
    regularizer (RBF-activated) plus an MRI data-consistency gradient step."""

    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        conv_kernel = np.random.randn(
            options["features_out"],
            options["features_in"],
            options["kernel_size"],
            options["kernel_size"],
            2,
        ).astype(np.float32) / np.sqrt(options["kernel_size"] ** 2 * 2 * options["features_in"])
        conv_kernel -= np.mean(conv_kernel, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel = torch.from_numpy(conv_kernel)
        if options["do_prox_map"]:
            conv_kernel = zero_mean_norm_ball(conv_kernel, axis=(1, 2, 3, 4))

        self.conv_kernel = torch.nn.Parameter(conv_kernel)

        if self.options["activation"] == "rbf":
            self.activation = RBFActivation(**options)
        elif self.options["activation"] == "relu":
            self.activation = torch.nn.ReLU()
        self.lamb = torch.nn.Parameter(torch.tensor(options["lamb_init"], dtype=torch.float32))

    def mri_forward_op(self, u, coil_sens, sampling_mask, os=False):
        """Forward pass with kspace (2x the size).

        u: NxHxWx2 complex image
        coil_sens: NxCxHxWx2 coil sensitivity map
        sampling_mask: NxHxW sampling mask
        """
        if os:
            pad_u = torch.tensor((sampling_mask.shape[1] * 0.25 + 1), dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1] * 0.25 - 1), dtype=torch.int16)
            u_pad = F.pad(u, [0, 0, 0, 0, pad_u, pad_l])
        else:
            u_pad = u
        u_pad = u_pad.unsqueeze(1)
        coil_imgs = complex_mul(u_pad, coil_sens)  # NxCxHxWx2

        Fu = fftc2d(coil_imgs)

        mask = sampling_mask.unsqueeze(1)  # Nx1xHxW
        mask = mask.unsqueeze(4)  # Nx1xHxWx1
        mask = mask.repeat([1, 1, 1, 1, 2])  # Nx1xHxWx2

        kspace = mask * Fu  # NxCxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask, os=False):
        """Adjoint op: kspace -> coil-combined undersampled image.

        f: NxCxHxWx2 multi-channel kspace
        """
        mask = sampling_mask.unsqueeze(1)  # Nx1xHxW
        mask = mask.unsqueeze(4)  # Nx1xHxWx1
        mask = mask.repeat([1, 1, 1, 1, 2])  # Nx1xHxWx2

        Finv = ifftc2d(mask * f)  # NxCxHxWx2
        img = torch.sum(complex_mul(Finv, conj(coil_sens)), 1)

        if os:
            pad_u = torch.tensor((sampling_mask.shape[1] * 0.25 + 1), dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1] * 0.25 - 1), dtype=torch.int16)
            img = img[:, pad_u:-pad_l, :, :]

        return img

    def forward(self, inputs):
        u_t_1 = inputs["u_t"]  # NxHxWx2
        f = inputs["f"]
        c = inputs["coil_sens"]
        m = inputs["sampling_mask"]

        u_t_1 = u_t_1.unsqueeze(1)  # Nx1xHxWx2
        pad = self.options["pad"]
        u_t_real = u_t_1[:, :, :, :, 0]
        u_t_imag = u_t_1[:, :, :, :, 1]

        u_t_real = F.pad(u_t_real, [pad, pad, pad, pad], mode="reflect")
        u_t_imag = F.pad(u_t_imag, [pad, pad, pad, pad], mode="reflect")
        u_k_real = F.conv2d(u_t_real, self.conv_kernel[:, :, :, :, 0], stride=1, padding=5)
        u_k_imag = F.conv2d(u_t_imag, self.conv_kernel[:, :, :, :, 1], stride=1, padding=5)
        u_k = u_k_real + u_k_imag
        f_u_k = self.activation(u_k)
        u_k_T_real = F.conv_transpose2d(f_u_k, self.conv_kernel[:, :, :, :, 0], stride=1, padding=5)
        u_k_T_imag = F.conv_transpose2d(f_u_k, self.conv_kernel[:, :, :, :, 1], stride=1, padding=5)

        u_k_T_real = u_k_T_real.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag.unsqueeze(-1)
        u_k_T = torch.cat((u_k_T_real, u_k_T_imag), dim=-1)

        Ru = u_k_T[:, 0, pad:-pad, pad:-pad, :]  # NxHxWx2
        Ru = Ru / self.options["features_out"]

        if self.options["sampling_pattern"] == "cartesian":
            os = False
        elif (
            "sampling_pattern" not in self.options
            or self.options["sampling_pattern"] == "cartesian_with_os"
        ):
            os = True
        else:
            os = False

        Au = self.mri_forward_op(u_t_1[:, 0, :, :, :], c, m, os)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m, os)
        Du = At_Au_f * self.lamb
        u_t = u_t_1[:, 0, :, :, :] - Ru - Du
        output = {"u_t": u_t, "f": f, "coil_sens": c, "sampling_mask": m}
        return output  # NxHxWx2


class VariationalNetwork(nn.Module):
    """`num_stages` unrolled `VnMriReconCell`s (real forward pass only --
    `pytorch_lightning.LightningModule` training-loop machinery is dropped)."""

    def __init__(self, **kwargs):
        super().__init__()
        options = dict(DEFAULT_OPTS)

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        cell_list = []
        for _i in range(options["num_stages"]):
            cell_list.append(VnMriReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)

    def forward(self, inputs):
        output = self.cell_list(inputs)
        return output["u_t"]


# ---- tiny build/example (architecture unmodified from the real repo, small spatial size) ----


def build_mri_varnet():
    """Tiny VariationalNetwork (2 unrolled stages, 4x4 image, 2 coils, small kernel) for
    tracing."""
    torch.manual_seed(0)
    model = VariationalNetwork(
        kernel_size=3,
        features_out=4,
        pad=3,
        num_act_weights=7,
        num_stages=2,
        sampling_pattern="cartesian",
    )
    model.eval()
    return model


def example_input_mri_varnet():
    """Matches VariationalNetwork.forward: dict of {u_t, f, coil_sens, sampling_mask}
    (N=1, H=W=4, C=2 coils), the real multi-tensor input schema used by `KneeDataset`."""
    torch.manual_seed(0)
    N, H, W, C = 1, 4, 4, 2
    u_t = torch.randn(N, H, W, 2, dtype=torch.float32)
    f = torch.randn(N, C, H, W, 2, dtype=torch.float32)
    coil_sens = torch.randn(N, C, H, W, 2, dtype=torch.float32)
    sampling_mask = torch.ones(N, H, W, dtype=torch.float32)
    return {"u_t": u_t, "f": f, "coil_sens": coil_sens, "sampling_mask": sampling_mask}


MENAGERIE_ENTRIES = [
    (
        "MRI Variational Network (VarNet)",
        "build_mri_varnet",
        "example_input_mri_varnet",
        2018,
        MENAGERIE_ZOO,
    ),
]
