# SOURCE: vendored from vicky-k-19/ADMM-DAD @ main
# https://github.com/vicky-k-19/ADMM-DAD/blob/main/admm_mnist.py
# ADMM-DAD (ICASSP 2022, arXiv:2110.06986): analysis-operator deep denoiser unrolled
# via ADMM for compressed sensing reconstruction. `DAD` is the decoder-network class:
# a learned redundant analysis operator Phi combined with a fixed random sensing
# matrix A, unrolled for `admm_iterations` ADMM steps with a learned soft-threshold
# (ShrinkageActivation) at each step. Transcribed verbatim from the real model class
# in admm_mnist.py (pure torch.nn/torch.einsum/torch.linalg, no custom CUDA ops).
# Only changes: the module-level `normalize` global (set via argparse in the source
# script) is hardcoded here to "sqrt_m" (the source script's own default), and the
# training/plotting/argparse scaffolding is dropped -- the DAD model class itself is
# untouched. `torch.lu`/`torch.lu_unpack` are the source's own calls (deprecated but
# functional in torch 2.x).
import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

# Source script's own default for `--normalization` (admm_mnist.py parse_args()).
_NORMALIZE = "sqrt_m"


class ShrinkageActivation(nn.Module):
    def __init__(self):
        super(ShrinkageActivation, self).__init__()

    # implements the soft-thresholding function employed in ADMM
    def forward(self, x, lamda):
        return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - lamda)


# definition of the decoder
class DAD(nn.Module):
    def __init__(
        self,
        measurements=200,
        ambient=28 * 28,
        redundancy_multiplier=5,
        admm_iterations=10,
        lamda=0.0001,
        rho=1,
    ):
        super(DAD, self).__init__()
        self.lamda = lamda
        self.rho = rho
        self.redundancy_multiplier = redundancy_multiplier
        self.admm_iterations = admm_iterations
        self.measurements = measurements
        self.ambient = ambient
        self.activation = ShrinkageActivation()
        a = torch.randn(measurements, ambient)
        if _NORMALIZE is None:
            a = a
        elif _NORMALIZE == "sqrt_m":
            a = a / np.sqrt(self.measurements)
        elif _NORMALIZE == "orth":
            a = torch.nn.init.orthogonal_(a.t()).t().contiguous()
        self.register_buffer("a", a)
        phi = nn.Parameter(self._init_phi())
        self.register_parameter("phi", phi)

    def _init_phi(self):
        # initialization of the analysis operator
        init = torch.empty(self.ambient * self.redundancy_multiplier, self.ambient)
        init = torch.nn.init.kaiming_normal_(init)
        return init

    def extra_repr(self):
        return "(phi): Parameter({}, {})".format(*self.phi.shape)

    def measure_x(self, x):
        # Create measurements y=Ax+noise
        y = torch.einsum("ma,ba->bm", self.a, x)
        n = 1e-4 * torch.randn_like(y)
        y = y + n
        return y

    def multiplier(self, rho):
        # m = (A^T*A+Phi^T*Phi)^-1
        # Instead of calculating directly the inverse, we take the LU factorization of A^T*A+Phi^T*Phi
        ata = torch.mm(self.a.t(), self.a)
        ftf = torch.mm(self.phi.t(), self.phi)
        m = ata + rho * ftf
        m_lu, _ = m.lu()
        _, L, U = torch.lu_unpack(m_lu, _)
        Linv = torch.linalg.inv(L)
        Uinv = torch.linalg.inv(U)
        return Linv, Uinv

    def linear(self, x, u):
        # application of analysis operator Phi
        fx = torch.einsum("sa,ba->bs", self.phi, x)
        return fx + u  # (B, 3*784)

    def decode(self, y, min_x, max_x, u, z):
        rho = self.rho
        lamda = self.lamda
        Linv, Uinv = self.multiplier(rho)
        x0 = torch.einsum("am,bm->ba", self.a.t(), y)

        for _ in range(self.admm_iterations):
            x_L = torch.einsum(
                "aa,ba->ba", Linv, x0 + torch.einsum("as,bs->ba", rho * self.phi.t(), z - u)
            )
            x_hat = torch.einsum("aa,ba->ba", Uinv, x_L)
            fxu = self.linear(x_hat, u)
            z = self.activation(fxu, lamda / rho)
            u = u + fxu - z

        # truncate the reconstructed x_hat, so that it lies in the same values' interval as the original x
        return torch.clamp(x_hat, min=min_x, max=max_x)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)
        # measure x
        y = self.measure_x(x)
        # apply analysis operator Phi
        fx = torch.einsum("sa,ba->bs", self.phi, x)
        # create the dual variables z, u
        u = torch.zeros_like(fx)
        z = torch.zeros_like(fx)
        # pass y through the network-decoder to get the output x_hat
        x_hat = self.decode(y, min_x, max_x, u, z)
        return x_hat


def build_admm_dad():
    # Tiny menagerie-scale config: source default ambient=28*28 (MNIST), shrunk
    # measurements/redundancy_multiplier/admm_iterations for fast tracing.
    return DAD(
        measurements=20,
        ambient=28 * 28,
        redundancy_multiplier=2,
        admm_iterations=3,
        lamda=1e-4,
        rho=1,
    )


def example_input_admm_dad():
    # Source script feeds flattened MNIST images (B, 1, 28, 28); DAD.forward()
    # itself flattens via x.view(x.size(0), -1).
    torch.manual_seed(0)
    return (torch.randn(2, 1, 28, 28),)


MENAGERIE_ENTRIES = [
    (
        "ADMM-DAD",
        "build_admm_dad",
        "example_input_admm_dad",
        2022,
        "vendored-pytorch",
    ),
]
