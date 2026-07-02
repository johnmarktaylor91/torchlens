# FAITHFUL PORT of https://github.com/hqyyqh888/DeepUnfolding_WMMSE @ main
#   (original: untyped procedural numpy/numpy.matrix script with module-level global
#   state -- `DeepUnfolding/train_main.py` `forward()`/`test_model.py` `forward_test()`
#   -- not a runnable nn.Module, and not torch)
#
# Ported functions: `train_main.py` / `test_model.py` `forward()` (the actual unrolled
# network's forward pass; `back_propagation()` is a hand-derived closed-form gradient
# used only to train the original numpy script and is NOT part of the network
# architecture, so it is not ported). Hu, Liu, Wen, Wymeersch, Jin, "Iterative
# Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser
# MIMO Systems" (IAIDNN), IEEE Trans. Wireless Commun., 2021 (arXiv:2006.08099).
#
# IAIDNN unrolls the classic WMMSE iterative precoding algorithm for a MU-MIMO downlink
# (User receivers, each with Nr antennas and d data streams, served by one Nt-antenna
# transmitter) into a fixed number of network "layers". Each layer performs the same
# 3-step WMMSE update (receive combiner U, MMSE weight W, transmit precoder V) as
# classic WMMSE, but every matrix pseudo-inverse in the update is REPLACED by a cheap
# diagonal-loading approximation (element-wise reciprocal of the matrix diagonal --
# `1/diag(A)` -- rather than a true `A^-1`), and the resulting per-layer/per-user
# update is affinely blended by *trainable* complex matrices (X, Y, Z, O per
# quantity/layer/user) learned to compensate for the approximation. This is the
# paper's core contribution: replacing all off-diagonal matrix inversion (the WMMSE
# bottleneck) with trainable diagonal approximations, i.e. "iterative-algorithm-
# induced deep unfolding".
#
# Faithful-port notes (no architectural changes, only numpy(complex matrix)->torch
# mechanical transcription):
#   - `np.matrix` complex ops (`*` = matmul, `.H` = conjugate transpose, `.I` = inverse)
#     -> `torch.matmul`/`.conj().transpose(-1,-2)`/`torch.linalg.inv`, applied per-user
#     via a leading (User,) batch dimension per layer (the original loops over `User`
#     imperatively; batching over that axis is the standard "python for-loop over
#     independent complex matrices" -> "torch batched linalg" transcription, not an
#     architectural change -- every matrix op and its operands are identical).
#   - `np.mat(np.diag(np.squeeze(1/np.diagonal(A))))` (diagonal-loading pseudo-inverse
#     approximation) -> `torch.diag_embed(1 / torch.diagonal(A, dim1=-2, dim2=-1))`,
#     batched the same way.
#   - The channel matrices H[k] (Nr x Nt per user) and the trainable X/Y/Z/O matrices
#     are complex64 tensors instead of numpy.matrix objects; all algebra is otherwise
#     identical, operation-for-operation, to `forward()`/`forward_test()`.
#   - `Y_U`/`Y_W`/`Y_V` are always trained with `scale_factor_Y = 0` in the real repo's
#     `train_main.py` (i.e. initialized to exactly zero and never updated -- see the
#     commented-out `G_*_Y_batch` gradient-accumulation lines); they are kept as
#     trainable zero-initialized parameters here for full architectural fidelity
#     (the *slot* is part of the paper's per-layer affine blend, even though the
#     reference hyperparameters zero it out).
#   - `V[0]` (layer-0 precoder) is initialized once per forward call from the batch's
#     channel matrices via the same zero-forcing pseudo-inverse formula
#     (`V0 = H_stacked^H (H_stacked H_stacked^H)^-1`) as the original script.
#
# Forward flow per batch element (see train_main.py `forward()`, User=number of
# receivers, Layer=number of unrolled layers, k indexes users):
#   V[0] = zero_forcing_precoder(H)                                    # real matrix inverse (unapproximated init)
#   for l in 1..Layer-2:
#       A[k] = sigma^2/Pt * sum_m tr(V[m]V[m]^H) I_Nr + sum_m H[k]V[m]V[m]^H H[k]^H
#       U[k] = (diagload(A[k]) X_U + A[k] Y_U + Z_U) H[k] V[k] + O_U
#       E[k] = I_d - U[k]^H H[k] V[k]
#       W[k] = diagload(E[k]) X_W + E[k] Y_W + Z_W
#       B    = sigma^2/Pt * sum_k tr(U[k]W[k]U[k]^H) I_Nt + sum_k H[k]^H U[k]W[k]U[k]^H H[k]
#       V[k] = (diagload(B) X_V + B Y_V + Z_V) H[k]^H U[k] W[k] + O_V
#   (final layer Layer-1 uses the exact WMMSE update for V, no diagonal-load/X/Y/Z/O)
#   return V[Layer-1]   # per-user Nt x d precoding matrices

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _diag_load_inv(A: torch.Tensor) -> torch.Tensor:
    """Port of `np.mat(np.diag(np.squeeze(1/np.diagonal(A))))`: a cheap diagonal-only
    approximation of the matrix inverse, batched over any leading dims."""
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    return torch.diag_embed(1.0 / diag)


class IAIDNNPrecoder(nn.Module):
    """Faithful port of the IAIDNN unrolled-WMMSE MU-MIMO precoding network
    (train_main.py/test_model.py `forward()`). Trainable per-layer, per-user complex
    matrices (X/Y/Z/O) blend a diagonal-loading matrix-inverse approximation to
    replace WMMSE's exact (expensive) matrix inversions."""

    def __init__(
        self,
        num_users: int,
        num_layers: int,
        num_tx_antennas: int,
        num_rx_antennas: int,
        num_streams: int,
        tx_power: float = 100.0,
        noise_sigma: float = 1.0,
        delta_noise: float = 0.0,
        scale_factor_x: float = 0.1,
        scale_factor_y: float = 0.0,
        scale_factor_z: float = 0.1,
        scale_factor_o: float = 0.1,
    ):
        super().__init__()
        self.User = num_users
        self.Layer = num_layers
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.d = num_streams
        self.Pt = tx_power
        self.sigma = noise_sigma
        self.delta_noise = delta_noise

        # trainable per-(layer, user) complex matrices, matching
        # `scale*(-1-1j + 2*rand + 2*rand*1j)`: uniform complex in
        # [-scale*(1+1j), scale*(1+1j)]. Layer 0 slots are unused
        # (the forward loop only reads layers 1..Layer-1) but kept for index parity
        # with the original script's `[Layer][User]`-indexed lists.
        self.X_U = nn.Parameter(
            scale_factor_x
            * self._rand_complex(num_layers, num_users, num_rx_antennas, num_rx_antennas)
        )
        self.X_W = nn.Parameter(
            scale_factor_x * self._rand_complex(num_layers, num_users, num_streams, num_streams)
        )
        self.X_V = nn.Parameter(
            scale_factor_x
            * self._rand_complex(num_layers, num_users, num_tx_antennas, num_tx_antennas)
        )

        self.Y_U = nn.Parameter(
            scale_factor_y
            * self._rand_complex(num_layers, num_users, num_rx_antennas, num_rx_antennas)
        )
        self.Y_W = nn.Parameter(
            scale_factor_y * self._rand_complex(num_layers, num_users, num_streams, num_streams)
        )
        self.Y_V = nn.Parameter(
            scale_factor_y
            * self._rand_complex(num_layers, num_users, num_tx_antennas, num_tx_antennas)
        )

        self.Z_U = nn.Parameter(
            scale_factor_z
            * self._rand_complex(num_layers, num_users, num_rx_antennas, num_rx_antennas)
        )
        self.Z_W = nn.Parameter(
            scale_factor_z * self._rand_complex(num_layers, num_users, num_streams, num_streams)
        )
        self.Z_V = nn.Parameter(
            scale_factor_z
            * self._rand_complex(num_layers, num_users, num_tx_antennas, num_tx_antennas)
        )

        self.O_U = nn.Parameter(
            scale_factor_o * self._rand_complex(num_layers, num_users, num_rx_antennas, num_streams)
        )
        self.O_V = nn.Parameter(
            scale_factor_o * self._rand_complex(num_layers, num_users, num_tx_antennas, num_streams)
        )

    @staticmethod
    def _rand_complex(*shape):
        real = torch.empty(*shape).uniform_(-1.0, 1.0)
        imag = torch.empty(*shape).uniform_(-1.0, 1.0)
        return torch.complex(real, imag)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (batch, User, Nr, Nt) complex channel matrices.
        Returns: (batch, User, Nt, d) complex precoding matrices V[Layer-1]."""
        batch, User, Nr, Nt = H.shape
        d = self.d
        Layer = self.Layer
        eye_nr = torch.eye(Nr, dtype=H.dtype, device=H.device)
        eye_nt = torch.eye(Nt, dtype=H.dtype, device=H.device)
        eye_d = torch.eye(d, dtype=H.dtype, device=H.device)
        Hh = H.conj().transpose(-2, -1)  # (batch, User, Nt, Nr)

        # ---- V[0]: zero-forcing precoder init (unapproximated pseudo-inverse) ----
        H_stack = H.reshape(batch, User * Nr, Nt)  # (batch, User*Nr, Nt)
        Hs_h = H_stack.conj().transpose(-2, -1)
        gram = torch.matmul(H_stack, Hs_h)  # (batch, User*Nr, User*Nr)
        VV = torch.matmul(Hs_h, torch.linalg.inv(gram))  # (batch, Nt, User*Nr)
        V1 = torch.stack(
            [VV[:, :, k * d : (k + 1) * d] for k in range(User)], dim=1
        )  # (batch, User, Nt, d)

        U = None
        W = None
        V = V1

        for layer_idx in range(1, Layer - 1):
            # A[k] = sigma^2/Pt * sum_m tr(V[m]V[m]^H) I_Nr + sum_m H[k]V[m]V[m]^H H[k]^H
            VVh = torch.matmul(V1, V1.conj().transpose(-2, -1))  # (batch, User, Nt, Nt)
            trace_sum = torch.diagonal(VVh, dim1=-2, dim2=-1).sum(-1).sum(-1)  # (batch,)
            # Build per-user A[k] by summing over m of H[k] V[m]V[m]^H H[k]^H
            A = (self.sigma**2 / self.Pt) * trace_sum.view(batch, 1, 1, 1) * eye_nr
            # sum_m H[k] V[m] V[m]^H H[k]^H : compute via H[k] (sum_m V[m]V[m]^H) H[k]^H
            sum_VVh = VVh.sum(dim=1, keepdim=True)  # (batch, 1, Nt, Nt)
            HkSumVVhHk = torch.matmul(torch.matmul(H, sum_VVh), Hh)  # (batch, User, Nr, Nr)
            A = A + HkSumVVhHk
            A = A + self.delta_noise * torch.ones_like(A)
            I_A = _diag_load_inv(A)

            U = (
                torch.matmul(I_A, self.X_U[layer_idx])
                + torch.matmul(A, self.Y_U[layer_idx])
                + self.Z_U[layer_idx]
            )
            U = torch.matmul(U, torch.matmul(H, V1)) + self.O_U[layer_idx]  # (batch, User, Nr, d)

            E = eye_d - torch.matmul(U.conj().transpose(-2, -1), torch.matmul(H, V1))
            E = E + self.delta_noise * torch.ones_like(E)
            I_E = _diag_load_inv(E)

            W = (
                torch.matmul(I_E, self.X_W[layer_idx])
                + torch.matmul(E, self.Y_W[layer_idx])
                + self.Z_W[layer_idx]
            )

            UWUh = torch.matmul(
                torch.matmul(U, W), U.conj().transpose(-2, -1)
            )  # (batch, User, Nr, Nr)
            trace_B = torch.diagonal(UWUh, dim1=-2, dim2=-1).sum(-1).sum(-1)  # (batch, User)
            trace_B_sum = trace_B.sum(-1)  # (batch,)
            B = (self.sigma**2 / self.Pt) * trace_B_sum.view(batch, 1, 1) * eye_nt
            HkhUWUhHk_sum = torch.matmul(torch.matmul(Hh, UWUh), H).sum(dim=1)  # (batch, Nt, Nt)
            B = B + HkhUWUhHk_sum
            B = B + self.delta_noise * torch.ones_like(B)
            I_B = _diag_load_inv(B)
            I_B = I_B.unsqueeze(1).expand(-1, User, -1, -1)
            B_exp = B.unsqueeze(1).expand(-1, User, -1, -1)

            V = (
                torch.matmul(I_B, self.X_V[layer_idx])
                + torch.matmul(B_exp, self.Y_V[layer_idx])
                + self.Z_V[layer_idx]
            )
            V = torch.matmul(V, torch.matmul(Hh, torch.matmul(U, W))) + self.O_V[layer_idx]

            V1 = V

        # ---- final layer (Layer-1): exact WMMSE update for U, W; then V ----
        layer_idx = Layer - 1
        VVh = torch.matmul(V1, V1.conj().transpose(-2, -1))
        trace_sum = torch.diagonal(VVh, dim1=-2, dim2=-1).sum(-1).sum(-1)
        sum_VVh_all = VVh.sum(dim=1, keepdim=True)
        A_last = (self.sigma**2 / self.Pt) * trace_sum.view(batch, 1, 1, 1) * eye_nr
        A_last = A_last + torch.matmul(torch.matmul(H, sum_VVh_all), Hh)
        A_last = A_last + self.delta_noise * torch.ones_like(A_last)
        I_A_last = _diag_load_inv(A_last)

        U = (
            torch.matmul(I_A_last, self.X_U[layer_idx])
            + torch.matmul(A_last, self.Y_U[layer_idx])
            + self.Z_U[layer_idx]
        )
        U = torch.matmul(U, torch.matmul(H, V1)) + self.O_U[layer_idx]

        E = eye_d - torch.matmul(U.conj().transpose(-2, -1), torch.matmul(H, V1))
        E = E + self.delta_noise * torch.ones_like(E)
        I_E = _diag_load_inv(E)
        W = (
            torch.matmul(I_E, self.X_W[layer_idx])
            + torch.matmul(E, self.Y_W[layer_idx])
            + self.Z_W[layer_idx]
        )

        UWUh = torch.matmul(torch.matmul(U, W), U.conj().transpose(-2, -1))
        C = (
            (self.sigma**2 / self.Pt)
            * torch.diagonal(UWUh, dim1=-2, dim2=-1).sum(-1).sum(-1).sum(-1).view(batch, 1, 1)
            * eye_nt
        )
        C = C + torch.matmul(torch.matmul(Hh, UWUh), H).sum(dim=1)
        C_inv = torch.linalg.inv(C).unsqueeze(1).expand(-1, User, -1, -1)

        V_final = torch.matmul(C_inv, torch.matmul(Hh, torch.matmul(U, W)))

        return V_final


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_iaidnn_wmmse():
    """IAIDNNPrecoder at tiny size for tracing. Architecture is unmodified from the
    real repo; only the size knobs (User/Layer/Nt/Nr/d) are shrunk from the paper's
    reference config (User=30, Layer=5, Nt=64, Nr=2, d=2) to keep tracing cheap."""
    model = IAIDNNPrecoder(
        num_users=2,
        num_layers=3,
        num_tx_antennas=4,
        num_rx_antennas=2,
        num_streams=2,
        tx_power=100.0,
        noise_sigma=1.0,
    )
    model.eval()
    return model


def example_input_iaidnn_wmmse():
    """Matches IAIDNNPrecoder.forward: (batch, User, Nr, Nt) complex MU-MIMO channel
    matrices."""
    real = torch.randn((1, 2, 2, 4))
    imag = torch.randn((1, 2, 2, 4))
    return torch.complex(real, imag)


MENAGERIE_ENTRIES = [
    ("IAIDNN", "build_iaidnn_wmmse", "example_input_iaidnn_wmmse", 2021, MENAGERIE_ZOO),
]
