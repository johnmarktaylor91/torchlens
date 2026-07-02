# SOURCE: vendored from ShlezingerLab/SubspaceNet @ main
# https://raw.githubusercontent.com/ShlezingerLab/SubspaceNet/main/src/models.py
# https://raw.githubusercontent.com/ShlezingerLab/SubspaceNet/main/src/utils.py
#
# SubspaceNet: model-based deep learning for DoA (direction-of-arrival) estimation.
# "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
# The network learns to map the tau-lag empirical autocorrelation tensor Rx_tau of
# shape [B, tau, 2N, N] to a surrogate covariance matrix via a CNN/deCNN encoder-decoder
# (conv1/conv2/conv3 -> deconv2/deconv3/deconv4, each followed by an anti-rectifier
# concat-ReLU nonlinearity), then feeds the resulting Hermitian/PSD surrogate covariance
# through a differentiable subspace method (Root-MUSIC or ESPRIT, implemented with
# eigendecomposition + a differentiable polynomial companion-matrix root-finder) to
# produce DoA angle predictions. Covers both cand_02341 and cand_02342 (identical
# repo/paper -- both queue rows resolve to this one SubspaceNet model).
#
# `SubspaceNet` and its helper functions (`root_music`, `esprit`, `gram_diagonal_overload`,
# `sum_of_diags_torch`, `find_roots_torch`) are transcribed verbatim from `src/models.py`
# and `src/utils.py`. No architectural changes were made -- every Conv2d/ConvTranspose2d
# layer and its arguments, the anti-rectifier concat-ReLU nonlinearity, and every
# eigendecomposition/matmul step of the Root-MUSIC subspace method are unchanged. Only
# the module-level `ModelGenerator`/training/plotting/simulation-config machinery (not
# part of the traced architecture) is dropped, and the global `device` is pinned to CPU
# for tracing.

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cpu")


def sum_of_diags_torch(matrix: torch.Tensor):
    """Calculates the sum of diagonals in a square matrix (Pytorch-oriented)."""
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[0] + 1, matrix.shape[0] - 1, 2 * matrix.shape[0] - 1, dtype=int
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


def find_roots_torch(coefficients: torch.Tensor):
    """Finds the roots of a polynomial defined by its coefficients (Pytorch-oriented)."""
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    roots = torch.linalg.eigvals(A)
    return roots


def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    """Multiply a matrix Kx with its Hermitian conjugate (gram matrix), and adds eps
    to the diagonal values of the matrix, ensuring a Hermitian and PSD matrix."""
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)

    Kx_list = []
    bs_kx = Kx
    for iter in range(batch_size):
        K = bs_kx[iter]
        Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)
        eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
        Rz = Kx_garm + eps_addition
        Kx_list.append(Rz)
    Kx_Out = torch.stack(Kx_list, dim=0)
    return Kx_Out


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Model-based Root-MUSIC algorithm (Pytorch, differentiable)."""
    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        diag_sum = sum_of_diags_torch(F)
        roots = find_roots_torch(diag_sum)
        roots_angels_all = torch.angle(roots)
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        roots = roots[sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        roots_angels = torch.angle(roots)
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    )


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Model-based Esprit algorithm (Pytorch, differentiable)."""
    doa_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        Us_upper, Us_lower = (
            Us[0 : R.shape[0] - 1],
            Us[1 : R.shape[0]],
        )
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)


class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation
    problem, over subspace methods."""

    def __init__(self, tau: int, M: int, diff_method: str = "root_music"):
        super(SubspaceNet, self).__init__()
        self.M = M
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNet.set_diff_method: Method {diff_method} is not defined for SubspaceNet"
            )

    def anti_rectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        Rx_real = Rx_View[:, : self.N, :]
        Rx_imag = Rx_View[:, self.N :, :]
        Kx_tag = torch.complex(Rx_real, Rx_imag)
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=self.batch_size)
        method_output = self.diff_method(Rz, self.M, self.batch_size)
        if isinstance(method_output, tuple):
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz


def build_subspacenet():
    torch.manual_seed(0)
    # Repo default is tau=8 (main.py .set_tau(8)); shrunk to tau=3, M=2, N=6 for
    # a tiny fast trace. N must be large enough that the conv/deconv stack (three
    # kernel-2 convs + three kernel-2 deconvs, net receptive-field-neutral) leaves
    # a positive spatial size at the bottleneck.
    model = SubspaceNet(tau=3, M=2, diff_method="root_music")
    model.eval()
    return model


def example_input_subspacenet():
    torch.manual_seed(0)
    # Rx_tau: [Batch, tau, 2N, N] empirical autocorrelation tensor, N=6, tau=3.
    return torch.randn(1, 3, 12, 6)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SubspaceNet", "build_subspacenet", "example_input_subspacenet", 2023, MENAGERIE_ZOO),
]
