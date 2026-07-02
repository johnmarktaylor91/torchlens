# SOURCE: vendored from https://github.com/bo-10000/MoDL_PyTorch @ master
#   Vendored files (real model-definition code only; training/eval CLI (`train.py`,
#   `test.py`), the dataset loader (`datasets/modl_dataset.py`), config YAMLs, and
#   TensorBoard/checkpoint workspace dirs are dropped):
#     - models/modl.py -> `conv_block`, `cnn_denoiser`, `myAtA`, `myCG`,
#       `data_consistency`, `MoDL` (verbatim; the unrolled MoDL architecture itself)
#     - utils.py -> only `r2c`, `c2r` (verbatim; complex<->real channel-stacking
#       helpers that `myCG` calls on the traced forward path -- the file's other
#       contents, `Logger`/`set_seeds`/`fft_new`/plotting helpers, are training/eval-only
#       and dropped)
#
#   Import fix only (no architectural change): `from utils import r2c, c2r` is
#   flattened into this single file.
#
# MoDL (Aggarwal, Mani & Jacob, IEEE TMI 2019, "MoDL: Model-Based Deep Learning
# Architecture for Inverse Problems") is a model-based deep-learning reconstruction
# network for accelerated (undersampled) parallel-MRI. It unrolls a regularized
# least-squares reconstruction into `k_iters` alternating steps: a learned CNN
# denoiser (residual stack of Conv2d+BatchNorm2d+ReLU blocks, weight-shared across
# iterations) proposes a denoised image, then a data-consistency (DC) step solves
# the normal equations `(A^H A + lambda I) x = A^H y + lambda * denoised` via a
# differentiable conjugate-gradient (CG) solver built from the explicit multi-coil
# MRI forward model `A` (coil-sensitivity-weighted 2D FFT with k-space undersampling
# mask). `lambda` is a learned scalar. Architecture is reproduced verbatim.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- utils.py: only r2c/c2r (verbatim) ----


def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img


def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j * real_img[1]
    elif axis == 1:
        complex_img = real_img[:, 0] + 1j * real_img[:, 1]
    else:
        raise NotImplementedError
    return complex_img


# ---- models/modl.py (verbatim) ----


# CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()
    )


class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(2, 64)

        for _ in range(n_layers - 2):
            layers += conv_block(64, 64)

        layers += nn.Sequential(nn.Conv2d(64, 2, 3, padding=1), nn.BatchNorm2d(2))

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x  # (2, nrow, ncol)
        dw = self.nw(x) + idt  # (2, nrow, ncol)
        return dw


# CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """

    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm  # complex (B x ncoil x nrow x ncol)
        self.mask = mask  # complex (B x nrow x ncol)
        self.lam = lam

    def forward(self, im):  # step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_coil = self.csm * im  # split coil images (B x ncoil x nrow x ncol)
        k_full = torch.fft.fft2(im_coil, norm="ortho")  # convert into k-space
        k_u = k_full * self.mask  # undersampling
        im_u_coil = torch.fft.ifft2(k_u, norm="ortho")  # convert into image domain
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=1)  # coil combine (B x nrow x ncol)
        return im_u + self.lam * im


def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1)  # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj() * r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj() * Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj() * r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)


class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k  # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec


# model =======================
class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # dw
            z_k = self.dw(x_k)  # (2, nrow, ncol)
            # dc
            x_k = self.dc(z_k, x0, csm, mask)  # (2, nrow, ncol)
        return x_k


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_modl():
    """MoDL at tiny size for tracing (3 conv layers, 2 unrolled CG/denoiser
    iterations). Architecture is unmodified from the real repo."""
    torch.manual_seed(0)
    model = MoDL(n_layers=3, k_iters=2)
    model.eval()
    return model


def example_input_modl():
    """Matches MoDL.forward(x0, csm, mask): a zero-filled reconstruction, coil
    sensitivity maps, and an undersampling mask for a tiny 2-coil, 16x16 scenario."""
    torch.manual_seed(0)
    b, ncoil, nrow, ncol = 1, 2, 16, 16
    x0 = torch.randn(b, 2, nrow, ncol)
    csm = torch.randn(b, ncoil, nrow, ncol, dtype=torch.cfloat)
    mask = torch.ones(b, nrow, ncol, dtype=torch.cfloat)
    return (x0, csm, mask)


MENAGERIE_ENTRIES = [
    ("MoDL", "build_modl", "example_input_modl", 2019, MENAGERIE_ZOO),
]
