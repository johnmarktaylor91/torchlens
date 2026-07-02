# FAITHFUL REIMPLEMENTATION from Zhang, Xu, Yu, Zhang, Chen, Yu, "ADL-MVDR: All
# Deep Learning MVDR Beamformer for Target Speech Separation" (ICASSP 2021,
# arXiv:2008.06994) -- no public code. The official repo
# (https://github.com/zzhang68/adlmvdr) is a demo-only page (audio/video/image
# assets, no model source), confirmed by directly listing its tree contents.
#
# The paper (Sec. 3.2, Eq. 7-8, and Fig. 1's "ADL-MVDR (replacing matrix
# inversion and PCA)" block) gives a complete, unambiguous architectural spec
# for the model's core novel contribution -- the two GRU-Net blocks that
# replace the matrix inversion and PCA steps of the classical MVDR solution:
#
#   v-hat(t,f)          = GRU-Net_v(  Phi_SS(t,f) )      -- Eq. 7 (steering vector)
#   Phi_NN^-1-hat(t,f)  = GRU-Net_NN( Phi_NN(t,f) )       -- Eq. 7 (inverse noise cov)
#   h(t,f) = Phi_NN^-1-hat(t,f) v-hat(t,f) /
#            ( v-hat^H(t,f) Phi_NN^-1-hat(t,f) v-hat(t,f) )               -- Eq. 8
#
# with the exact layer spec from the paper text (Sec. 4.2, last two
# paragraphs): "the v-hat(t,f) estimation network consists of two layers of
# GRU followed by another layer of fully connected (FC) neurons. The hidden
# size is set to 500 and 250 for the 2-layer GRU with tanh activation
# function, linear activation function is used for the FC layer with a hidden
# size of 30. As for Phi_NN^-1-hat(t,f) estimation, the corresponding GRU-Net
# features a similar structure, where each GRU layer contains 500 units with a
# 450-size FC layer." The real+imaginary parts of each complex covariance
# input are concatenated before being fed to the GRU-Net ("the real and
# imaginary parts of the complex-valued covariance matrix Phi are concatenated
# together as input to the GRU-Net", Sec 3.2) and the GRU-Net output is
# reshaped back into the real/imag parts of the complex-valued steering vector
# / inverse covariance (Fig. 1: "reshaped again as inputs for calculating MVDR
# weights").
#
# Out of scope of this reimplementation (per the ladder's family-not-variant
# discipline, and because the paper explicitly delegates them to prior/other
# work, not to a specified architecture of its own): the upstream cRF complex
# filter estimator ("a Conv-TasNet variant [9,10] is adopted as the front-end
# filter estimator", Sec 3.1) and the DOA/directional-feature extraction
# front-end (Fig. 1, left) -- these are cited external architectures (Conv-
# TasNet variant), not part of the ADL-MVDR novelty being captured here. The
# traced module below implements exactly the "ADL-MVDR (replacing matrix
# inversion and PCA)" block from Fig. 1: it consumes the frame-level speech
# and noise covariance-like statistics (Phi_SS, Phi_NN) and the noisy mixture
# spectrogram Y(t,f), and produces the frame-level beamformed output per
# Eq. (8)-(9).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class GRUNet(nn.Module):
    """GRU-Net from ADL-MVDR Sec 3.2 / Sec 4.2: N-layer GRU (tanh activation,
    the torch default for nn.GRU) followed by a linear (FC) output layer.
    Used both for the steering-vector estimator (v-hat) and the inverse
    noise-covariance estimator (Phi_NN^-1-hat), which "feature a similar
    structure" per the paper -- only the hidden/FC sizes differ.
    """

    def __init__(self, input_size, gru_hidden_sizes, fc_out_size):
        super().__init__()
        self.gru_layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in gru_hidden_sizes:
            self.gru_layers.append(nn.GRU(in_size, hidden_size, batch_first=True))
            in_size = hidden_size
        self.fc = nn.Linear(in_size, fc_out_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h = x
        for gru in self.gru_layers:
            h, _ = gru(h)
        return self.fc(h)


class ADLMVDRCore(nn.Module):
    """The ADL-MVDR core block (Fig. 1 red dashed box): replaces the matrix
    inversion + PCA of the classical MVDR solution (Eq. 4) with two GRU-Nets
    (Eq. 7), then computes frame-level MVDR beamforming weights (Eq. 8) and
    applies them to the noisy multi-channel mixture (Eq. 9).

    Inputs (matching the paper's frame-level, per-frequency-bin covariance
    statistics, real/imag concatenated per Sec 3.2):
        phi_ss: (batch, n_frames, 2*M*M) frame-level speech covariance matrix
            Phi_SS(t,f), real and imaginary parts flattened+concatenated over
            the M x M microphone-array covariance.
        phi_nn: (batch, n_frames, 2*M*M) frame-level noise covariance matrix
            Phi_NN(t,f), same layout.
        y_real, y_imag: (batch, n_frames, M) real/imag parts of the noisy
            multi-channel mixture Y(t,f) for the corresponding frame.

    Output:
        s_hat: (batch, n_frames) complex-valued frame-level beamformed
            (separated) speech estimate, per Eq. (9).
    """

    def __init__(
        self,
        num_mics=3,
        v_gru_hidden=(500, 250),
        v_fc_out=30,
        nn_gru_hidden=(500, 500),
        nn_fc_out=450,
        eps=1e-6,
    ):
        super().__init__()
        self.num_mics = num_mics
        cov_flat_dim = 2 * num_mics * num_mics

        # v-hat(t,f) estimation network: 2-layer GRU (500, 250) + FC(30).
        self.v_net = GRUNet(cov_flat_dim, v_gru_hidden, v_fc_out)
        # Phi_NN^-1-hat(t,f) estimation network: GRU layers (500 each) + FC(450).
        self.nn_net = GRUNet(cov_flat_dim, nn_gru_hidden, nn_fc_out)

        # Project the GRU-Net outputs (v_fc_out / nn_fc_out) back to the
        # complex-valued steering-vector (2*M) and inverse-covariance-matrix
        # (2*M*M) shapes ("reshaped again as inputs for calculating MVDR
        # weights", Fig. 1 caption).
        self.v_reshape = nn.Linear(v_fc_out, 2 * num_mics)
        self.nn_reshape = nn.Linear(nn_fc_out, 2 * num_mics * num_mics)
        self.eps = eps

    def forward(self, phi_ss, phi_nn, y_real, y_imag):
        batch, n_frames, _ = phi_ss.shape
        m = self.num_mics

        v_out = self.v_reshape(self.v_net(phi_ss))  # (batch, n_frames, 2M)
        v_real, v_imag = v_out[..., :m], v_out[..., m:]

        nninv_out = self.nn_reshape(self.nn_net(phi_nn))  # (batch, n_frames, 2*M*M)
        nninv_flat = nninv_out.view(batch, n_frames, 2, m, m)
        nninv_real, nninv_imag = nninv_flat[:, :, 0], nninv_flat[:, :, 1]

        # h(t,f) = Phi_NN^-1 v / (v^H Phi_NN^-1 v)  (Eq. 8), complex arithmetic
        # expanded into real/imag parts.
        v_r = v_real.unsqueeze(-1)  # (batch, n_frames, M, 1)
        v_i = v_imag.unsqueeze(-1)

        num_real = torch.matmul(nninv_real, v_r) - torch.matmul(nninv_imag, v_i)
        num_imag = torch.matmul(nninv_real, v_i) + torch.matmul(nninv_imag, v_r)
        num_real = num_real.squeeze(-1)  # (batch, n_frames, M)
        num_imag = num_imag.squeeze(-1)

        # v^H Phi_NN^-1 v (scalar per frame): v^H = conj(v)^T
        vh_num_real = (v_real * num_real + v_imag * num_imag).sum(dim=-1)
        vh_num_imag = (v_real * num_imag - v_imag * num_real).sum(dim=-1)
        denom_sq = vh_num_real**2 + vh_num_imag**2 + self.eps

        # h = num / denom (complex division by the scalar denom)
        h_real = (
            num_real * vh_num_real.unsqueeze(-1) + num_imag * vh_num_imag.unsqueeze(-1)
        ) / denom_sq.unsqueeze(-1)
        h_imag = (
            num_imag * vh_num_real.unsqueeze(-1) - num_real * vh_num_imag.unsqueeze(-1)
        ) / denom_sq.unsqueeze(-1)

        # s-hat(t,f) = h^H(t,f) Y(t,f)  (Eq. 9)
        s_real = (h_real * y_real + h_imag * y_imag).sum(dim=-1)
        s_imag = (h_real * y_imag - h_imag * y_real).sum(dim=-1)

        return torch.complex(s_real, s_imag)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the paper; sizing only).
# ---------------------------------------------------------------------------
_NUM_MICS = 3
_N_FRAMES = 5
_BATCH = 2


def build_adlmvdr():
    torch.manual_seed(0)
    model = ADLMVDRCore(
        num_mics=_NUM_MICS,
        v_gru_hidden=(500, 250),
        v_fc_out=30,
        nn_gru_hidden=(500, 500),
        nn_fc_out=450,
    )
    model.eval()
    return model


def example_input_adlmvdr():
    torch.manual_seed(0)
    cov_flat_dim = 2 * _NUM_MICS * _NUM_MICS
    phi_ss = torch.randn(_BATCH, _N_FRAMES, cov_flat_dim)
    phi_nn = torch.randn(_BATCH, _N_FRAMES, cov_flat_dim)
    y_real = torch.randn(_BATCH, _N_FRAMES, _NUM_MICS)
    y_imag = torch.randn(_BATCH, _N_FRAMES, _NUM_MICS)
    return (phi_ss, phi_nn, y_real, y_imag)


MENAGERIE_ENTRIES = [
    ("ADL-MVDR", "build_adlmvdr", "example_input_adlmvdr", 2021, MENAGERIE_ZOO),
]
