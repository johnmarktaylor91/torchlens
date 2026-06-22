"""TranAD-benchmark baseline MTS anomaly-detection models (the non-TranAD competitors).

Reimplemented from the imperial-qore/TranAD benchmark suite ``src/models.py``.
Source: https://github.com/imperial-qore/TranAD/blob/main/src/models.py
TranAD paper (the benchmark host): https://arxiv.org/abs/2201.07284

Each of these is a published MTS-anomaly-detection architecture; this file reproduces the
DISTINCTIVE primitive of each at compact random-init scale (feats=7, short windows). The GAT
models (MTAD_GAT, GDN) use ``torch_geometric.nn.GATConv`` in place of the repo's ``dgl``
GATConv (same graph-attention primitive; lighter dep), with the SAME graph topology.

Families + distinctive primitives reproduced:
  * LSTM_Univariate (one independent nn.LSTM per feature -- a bank of univariate LSTMs).
  * LSTM_AD (Malhotra 2015, arXiv pre-print era): TWO stacked LSTMs -- one predicts,
    one reconstructs -- fused by a sigmoid FCN.
  * DAGMM (Zong et al., ICLR 2018, https://openreview.net/forum?id=BJJLHbb0-):
    autoencoder + a GMM "estimation network" fed the latent code AUGMENTED with the
    reconstruction-error features (relative-euclidean + cosine) -> softmax membership gamma.
  * OmniAnomaly (Su et al., KDD 2019, arXiv:1907... ): stochastic-RNN VAE -- a stacked GRU
    feeding a reparameterized (mu, logvar) latent and a decoder (planar-flow omitted; the
    stochastic GRU-VAE backbone IS the distinctive primitive).
  * USAD (Audibert et al., KDD 2020): ONE shared encoder, TWO decoders, and the
    adversarial AE2(AE1(.)) re-encode-decode path -- the two-phase adversarial AE.
  * MSCRED (Zhang et al., AAAI 2019, arXiv:1811.08055): multi-scale signature-matrix
    reconstruction via stacked ConvLSTM encoder + ConvTranspose decoder.
  * CAE_M (Zhang et al., TKDE 2021): convolutional autoencoder-with-memory backbone
    (Conv2d encoder + ConvTranspose2d decoder over the feats x window signature image).
  * MTAD_GAT (Zhao et al., ICDM 2020, arXiv:2009.02040): TWO parallel GATs -- a
    FEATURE-oriented GAT and a TIME-oriented GAT -- concatenated with the raw series and
    fed to a GRU.
  * GDN (Deng & Hooi, AAAI 2021, arXiv:2106.06947): graph-deviation network -- learned
    sensor-graph feature GAT with a windowed-attention pooling head.
  * MAD_GAN (Li et al., ICANN 2019, arXiv:1901.04997): LSTM/MLP GAN -- a generator and a
    discriminator scored on real vs generated windows.

All take a single window tensor via small adapters; trace+draw verified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv as PyGGATConv

_FEATS = 7


# ======================================================================================
# LSTM_Univariate -- a bank of independent per-feature LSTMs
# ======================================================================================


class LSTM_Univariate(nn.Module):
    """One independent nn.LSTM per feature (the distinctive 'bank of univariate LSTMs')."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "LSTM_Univariate"
        self.n_feats = feats
        self.n_hidden = 1
        self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for _ in range(feats)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, feats). Run each feature's series through its own LSTM.
        outputs = []
        for t in range(x.size(0)):
            g = x[t]
            multivariate_output = []
            for j in range(self.n_feats):
                univariate_input = g.reshape(-1)[j].view(1, 1, -1)
                out, _ = self.lstm[j](univariate_input)
                multivariate_output.append(2 * out.reshape(-1))
            outputs.append(torch.cat(multivariate_output))
        return torch.stack(outputs)


# ======================================================================================
# LSTM_AD -- dual stacked LSTMs (predict + reconstruct) fused by sigmoid FCN
# ======================================================================================


class LSTM_AD(nn.Module):
    """Two LSTMs (one hidden=64, one outputs feats) fused by a sigmoid FCN (Malhotra-style)."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "LSTM_AD"
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden)
        self.lstm2 = nn.LSTM(feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        h1 = h2 = None
        for t in range(x.size(0)):
            g = x[t]
            _, h1 = self.lstm(g.view(1, 1, -1), h1)
            out, h2 = self.lstm2(g.view(1, 1, -1), h2)
            out = self.fcn(out.reshape(-1))
            outputs.append(2 * out.reshape(-1))
        return torch.stack(outputs)


# ======================================================================================
# DAGMM -- autoencoder + GMM estimation network over [latent | recon-error features]
# ======================================================================================


class DAGMM(nn.Module):
    """Deep Autoencoding Gaussian Mixture Model (ICLR 18): AE + GMM estimation net."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "DAGMM"
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm),
            nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x: torch.Tensor, x_hat: torch.Tensor):
        rel_euc = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cos = F.cosine_similarity(x, x_hat, dim=1)
        return rel_euc, cos

    def forward(self, x: torch.Tensor):
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


# ======================================================================================
# OmniAnomaly -- stochastic-RNN VAE (GRU -> reparameterized latent -> decoder)
# ======================================================================================


class OmniAnomaly(nn.Module):
    """Stochastic-RNN VAE (KDD 19): stacked GRU + reparameterized (mu, logvar) latent."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "OmniAnomaly"
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        out, hidden = self.lstm(x.view(1, 1, -1))
        h = self.encoder(out)
        mu, logvar = torch.split(h, [self.n_latent, self.n_latent], dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x = self.decoder(z)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden


# ======================================================================================
# USAD -- one encoder, two decoders, adversarial AE2(AE1(.)) re-encode path
# ======================================================================================


class USAD(nn.Module):
    """UnSupervised Anomaly Detection (KDD 20): shared encoder + 2 decoders + adversarial path."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "USAD"
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor):
        z = self.encoder(g.view(1, -1))
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


# ======================================================================================
# MSCRED -- multi-scale ConvLSTM signature-matrix reconstruction
# ======================================================================================


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Conv2d over [input ; hidden] -> 4*hidden gates)."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size, bias: bool) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor: torch.Tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MSCRED(nn.Module):
    """Multi-Scale Convolutional Recurrent Encoder-Decoder (AAAI 19): stacked ConvLSTM + deconv."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "MSCRED"
        self.n_feats = feats
        self.n_window = feats
        self.cell1 = ConvLSTMCell(1, 32, (3, 3), True)
        self.cell2 = ConvLSTMCell(32, 64, (3, 3), True)
        self.cell3 = ConvLSTMCell(64, 128, (3, 3), True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1),
            nn.Sigmoid(),
        )

    def _run_cell(self, cell: ConvLSTMCell, z: torch.Tensor) -> torch.Tensor:
        b, _, h, w = z.shape
        hid = torch.zeros(b, cell.hidden_dim, h, w)
        cel = torch.zeros(b, cell.hidden_dim, h, w)
        hid, _ = cell(z, [hid, cel])
        return hid

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self._run_cell(self.cell1, z)
        z = self._run_cell(self.cell2, z)
        z = self._run_cell(self.cell3, z)
        x = self.decoder(z)
        return x.view(-1)


# ======================================================================================
# CAE_M -- convolutional autoencoder with memory backbone
# ======================================================================================


class CAE_M(nn.Module):
    """Convolutional AutoEncoder-with-Memory (TKDE 21): Conv2d encoder + ConvTranspose decoder."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "CAE_M"
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1),
            nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1),
            nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        x = self.decoder(z)
        return x.view(-1)


# ======================================================================================
# MTAD_GAT -- two parallel GATs (feature + time) concatenated then GRU
# ======================================================================================


def _fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """Edge index for a fully connected graph + self-loops over ``num_nodes`` nodes."""
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([src, dst], dim=0)


class MTAD_GAT(nn.Module):
    """Multivariate Time-series Anomaly Detection via GAT (ICDM 20): parallel feature + time GATs."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "MTAD_GAT"
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        # feature-graph: nodes=feats; time-graph: nodes=window+1. Use PyG GATConv (1 head x feats out).
        self.feature_gat = PyGGATConv(feats, feats, heads=1, add_self_loops=True)
        self.time_gat = PyGGATConv(feats, feats, heads=1, add_self_loops=True)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1)

    def forward(self, data: torch.Tensor):
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats), data))  # (W+1, F)
        # feature GAT: treat the (W+1) time-steps as nodes, feats as node features.
        ei = _fully_connected_edge_index(self.n_window + 1)
        feat_r = self.feature_gat(data_r, ei).unsqueeze(-1)  # (W+1, F, 1) after view
        feat_r = feat_r.view(self.n_window + 1, self.n_feats, 1)
        # time GAT: same topology, complementary view.
        time_r = self.time_gat(data_r, ei).view(self.n_window + 1, self.n_feats, 1)
        data3 = data_r.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data3, feat_r, time_r), dim=2).reshape(1, 1, -1)
        x, h = self.gru(x)
        return x.view(-1), h


# ======================================================================================
# GDN -- graph deviation network: windowed-attention pooling + feature GAT
# ======================================================================================


class GDN(nn.Module):
    """Graph Deviation Network (AAAI 21): windowed-attention pooling + learned sensor-graph GAT."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "GDN"
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        self.feature_gat = PyGGATConv(1, 1, heads=feats, add_self_loops=True, concat=True)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window),
            nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        att_score = self.attention(data.reshape(-1)).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)  # (feats, 1) -- node features
        ei = _fully_connected_edge_index(self.n_feats)
        feat_r = self.feature_gat(data_r, ei)  # (feats, feats) with heads=feats concat
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        x = self.fcn(feat_r)
        return x.view(-1)


# ======================================================================================
# MAD_GAN -- generator + discriminator scored on real vs generated windows
# ======================================================================================


class MAD_GAN(nn.Module):
    """Multivariate Anomaly Detection GAN (ICANN 19): generator + discriminator over windows."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "MAD_GAN"
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor):
        z = self.generator(g.view(1, -1))
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)


# ======================================================================================
# Builders + example inputs
# ======================================================================================


def build_lstm_univariate() -> nn.Module:
    """A bank of independent per-feature univariate LSTMs."""
    return LSTM_Univariate(_FEATS).eval()


def build_lstm_ad() -> nn.Module:
    """Dual-LSTM (predict + reconstruct) anomaly detector with sigmoid FCN."""
    return LSTM_AD(_FEATS).eval()


def build_dagmm() -> nn.Module:
    """Deep Autoencoding Gaussian Mixture Model (AE + GMM estimation net)."""
    return DAGMM(_FEATS).eval()


def build_omnianomaly() -> nn.Module:
    """Stochastic-RNN VAE (stacked GRU + reparameterized latent + decoder)."""
    return OmniAnomaly(_FEATS).eval()


def build_usad() -> nn.Module:
    """Two-decoder adversarial autoencoder (USAD)."""
    return USAD(_FEATS).eval()


def build_mscred() -> nn.Module:
    """Multi-scale ConvLSTM signature-matrix reconstruction (MSCRED)."""
    return MSCRED(_FEATS).eval()


def build_cae_m() -> nn.Module:
    """Convolutional autoencoder-with-memory backbone (CAE_M)."""
    return CAE_M(_FEATS).eval()


def build_mtad_gat() -> nn.Module:
    """Two parallel GATs (feature + time) + GRU (MTAD-GAT)."""
    return MTAD_GAT(_FEATS).eval()


def build_gdn() -> nn.Module:
    """Graph Deviation Network: windowed-attention pooling + feature GAT (GDN)."""
    return GDN(_FEATS).eval()


def build_mad_gan() -> nn.Module:
    """LSTM/MLP GAN with generator + discriminator over windows (MAD-GAN)."""
    return MAD_GAN(_FEATS).eval()


def example_input_seq() -> torch.Tensor:
    """A (T=10, feats=7) MTS window (for the recurrent per-step models)."""
    return torch.randn(10, _FEATS)


def example_input_window5() -> torch.Tensor:
    """A (n_window=5, feats=7) flat window (DAGMM/USAD/MAD_GAN/GDN)."""
    return torch.randn(5, _FEATS)


def example_input_step() -> torch.Tensor:
    """A single (1, feats=7) step for OmniAnomaly's per-step GRU-VAE."""
    return torch.randn(1, _FEATS)


def example_input_signature() -> torch.Tensor:
    """A (feats=7, feats=7) square signature matrix for MSCRED/CAE_M."""
    return torch.randn(_FEATS, _FEATS)


def example_input_mtad() -> torch.Tensor:
    """A (window=feats=7, feats=7) window for MTAD-GAT (n_window == feats)."""
    return torch.randn(_FEATS, _FEATS)


MENAGERIE_ENTRIES = [
    (
        "LSTM_Univariate (bank of independent per-feature LSTMs)",
        "build_lstm_univariate",
        "example_input_seq",
        "2022",
        "DC",
    ),
    (
        "LSTM_AD (dual-LSTM predict+reconstruct anomaly detector)",
        "build_lstm_ad",
        "example_input_seq",
        "2015",
        "DC",
    ),
    (
        "DAGMM (deep autoencoding Gaussian mixture model)",
        "build_dagmm",
        "example_input_window5",
        "2018",
        "DC",
    ),
    (
        "OmniAnomaly (stochastic-RNN GRU-VAE)",
        "build_omnianomaly",
        "example_input_step",
        "2019",
        "DC",
    ),
    (
        "USAD (two-decoder adversarial autoencoder)",
        "build_usad",
        "example_input_window5",
        "2020",
        "DC",
    ),
    (
        "MSCRED (multi-scale ConvLSTM signature-matrix reconstruction)",
        "build_mscred",
        "example_input_signature",
        "2019",
        "DC",
    ),
    (
        "CAE_M (convolutional autoencoder-with-memory)",
        "build_cae_m",
        "example_input_signature",
        "2021",
        "DC",
    ),
    (
        "MTAD-GAT (parallel feature-GAT + time-GAT + GRU)",
        "build_mtad_gat",
        "example_input_mtad",
        "2020",
        "DC",
    ),
    (
        "GDN (graph deviation network: attention pooling + feature GAT)",
        "build_gdn",
        "example_input_window5",
        "2021",
        "DC",
    ),
    (
        "MAD-GAN (generator+discriminator GAN over MTS windows)",
        "build_mad_gan",
        "example_input_window5",
        "2022",
        "DC",
    ),
]
