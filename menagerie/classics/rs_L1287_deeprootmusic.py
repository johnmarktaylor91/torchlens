# SOURCE: vendored from DA-MUSIC/DR-MUSIC_ICASSP23 @ 6036df6b4fa319ce4dfcec4787d71f713927d7d7
# models.py (Deep_Root_Net, unmodified except stripping the module-level
# device auto-select and .to(device) calls, which only matter for GPU/CPU
# placement, not architecture).
"""Deep Root-MUSIC: data-driven, model-based DoA estimation (ICASSP 2023).

A CNN autoencoder regresses a Hermitian PSD "surrogate covariance" matrix
from raw sample-covariance input, which then feeds a differentiable
Root-MUSIC direction-of-arrival estimator (polynomial rooting via a
companion-matrix eigendecomposition).
"""

import numpy as np
import torch
import torch.nn as nn


class Deep_Root_Net(nn.Module):
    def __init__(self, tau, ActivationVal):
        self.tau = tau
        super(Deep_Root_Net, self).__init__()
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(ActivationVal)
        self.DropOut = nn.Dropout(0.2)

    def sum_of_diags(self, Matrix):
        coeff = []
        diag_index = torch.linspace(
            -Matrix.shape[0] + 1, Matrix.shape[0] - 1, (2 * Matrix.shape[0]) - 1, dtype=int
        )
        for idx in diag_index:
            coeff.append(torch.sum(torch.diagonal(Matrix, idx)))
        return torch.stack(coeff, dim=0)

    def find_roots(self, coeff):
        A_torch = torch.diag(torch.ones(len(coeff) - 2, dtype=coeff.dtype), -1)
        A_torch[0, :] = -coeff[1:] / coeff[0]
        roots = torch.linalg.eigvals(A_torch)
        return roots

    def Root_MUSIC(self, Rz, M):
        dist = 0.5
        f = 1
        DOA_list = []
        DOA_all_list = []
        Bs_Rz = Rz
        roots_to_return = None
        for iter in range(self.BATCH_SIZE):
            R = Bs_Rz[iter]
            eigenvalues, eigenvectors = torch.linalg.eig(R)  # EVD
            Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
            F = torch.matmul(Un, torch.t(torch.conj(Un)))  # info matrix
            coeff = self.sum_of_diags(F)  # sum of diagonals of F
            roots = self.find_roots(coeff)  # roots of coeff polynomial

            roots_angels_all = torch.angle(roots)
            DOA_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
            DOA_all_list.append(DOA_pred_all)
            roots_to_return = roots

            roots = roots[
                sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
            ]  # roots outside unit circle
            mask = (torch.abs(roots) - 1) < 0

            roots = roots[mask][:M]
            roots_angels = torch.angle(roots)
            DOA_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
            DOA_list.append(DOA_pred)
        return torch.stack(DOA_list, dim=0), torch.stack(DOA_all_list, dim=0), roots_to_return

    def Gramian_matrix(self, Kx, eps):
        """
        multiply a Matrix Kx with its Hermitian Conjecture,
        and adds eps to diagonal Value of the Matrix,
        In order to Ensure Hermit and PSD:
        Kx = (Kx)^H @ (Kx) + eps * I
        """
        Kx_list = []
        Bs_kx = Kx
        for iter in range(self.BATCH_SIZE):
            K = Bs_kx[iter]
            Kx_garm = torch.matmul(torch.t(torch.conj(K)), K)  # output size(NxN)
            eps_Unit_Mat = eps * torch.diag(torch.ones(Kx_garm.shape[0]))
            Rz = Kx_garm + eps_Unit_Mat  # output size(NxN)
            Kx_list.append(Rz)
        Kx_Out = torch.stack(Kx_list, dim=0)
        return Kx_Out

    def forward(self, New_Rx_tau, M):
        ## Input shape of signal X(t): [Batch size, N, T]
        self.N = New_Rx_tau.shape[-1]
        self.BATCH_SIZE = New_Rx_tau.shape[0]

        ## AutoEncoder Architecture
        x = self.conv1(New_Rx_tau)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)

        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        x = self.deconv3(x)
        x = self.LeakyReLU(x)
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))  # [Batch size, 2N, N]

        ## Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # [Batch size, N, N]
        Rx_imag = Rx_View[:, self.N :, :]  # [Batch size, N, N]
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # [Batch size, N, N]

        ## Apply Gramian transformation to ensure Hermitian and PSD matrix
        Rz = self.Gramian_matrix(Kx_tag, eps=1)  # [Batch size, N, N]

        ## Rest of Root MUSIC algorithm
        DOA, DOA_all, roots = self.Root_MUSIC(Rz, M)  # [Batch size, M]
        return DOA, DOA_all, roots, Rz


def build_deeprootmusic():
    return Deep_Root_Net(tau=8, ActivationVal=0.5)


def example_input_deeprootmusic():
    # main.py: tau=8, N=8 (sensors); input layout [Batch, tau, 2N, N]
    x = torch.randn(1, 8, 16, 8)
    M = 2  # number of sources -- a plain int hyperparameter, matches main.py
    return (x, M)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepRootMUSIC",
        "build_deeprootmusic",
        "example_input_deeprootmusic",
        2023,
        "vendored-pytorch",
    ),
]
