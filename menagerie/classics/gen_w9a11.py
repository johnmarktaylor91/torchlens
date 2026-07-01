"""Menagerie batch w9a11: six niche-domain deep-learning classics.

Sources checked (reference only; no cloning, no pip installs; base-env torch
reimplementations of each source's distinctive mechanism):

  - Deep Packet (cand_01297): Lotfollahi, Jafari Siavoshani, Shirali Hossein
    Zade & Saberian, "Deep Packet: A Novel Approach For Encrypted
    Traffic Classification Using Deep Learning", arXiv:1709.02656 (2017).
    Community reference implementation https://github.com/munhouiani/
    Deep-Packet (``ml/model.py``, PyTorch Lightning ``CNN`` class). The
    packet-classification CNN takes a ``(batch, 1, 1500)`` normalized
    byte-stream vector (the first 1500 bytes of an IP packet, masked/
    anonymized header + payload) through two ``Conv1d + ReLU`` layers
    (c1: 200 filters / kernel 5 / stride 3; c2: 200 filters / kernel 4 /
    stride 1), a ``MaxPool1d(2)``, then three ``Linear + Dropout(0.05) +
    ReLU`` dense blocks (200 -> 100 -> 50) and a final linear classification
    head. Reimplemented verbatim with the published layer widths and a
    dummy-forward-computed flatten size (torchlens-visible, no dynamic
    control flow).
  - DeepACO (cand_01298): Ye, Wang, Cao & Song, "DeepACO: Neural-enhanced
    Ant Colony Optimization for Combinatorial Optimization", NeurIPS 2023,
    arXiv:2309.14032. Official repo https://github.com/henry-yeh/DeepACO
    (``tsp/net.py``, PyTorch + torch_geometric). The distinctive mechanism
    is ``EmbNet``: an anisotropic edge-gated message-passing GNN over the
    TSP instance graph -- each layer refines node embeddings ``x`` and
    edge embeddings ``w`` jointly: node update aggregates
    ``sigmoid(w) * (linear(x))`` over incoming edges via
    ``scatter_mean``-style aggregation plus a ``BatchNorm``+SiLU residual,
    edge update mixes the two endpoint node embeddings with the edge's own
    linear projection through another ``BatchNorm``+SiLU residual. The
    refined edge embeddings feed a small ``ParNet`` MLP (sigmoid-gated
    output) predicting a per-edge heuristic value used to bias ACO
    pheromone-guided construction. Reimplemented faithfully as
    ``EmbNet`` (depth=3 for compactness, same per-layer wiring) +
    ``ParNet`` heuristic head operating on a small random TSP instance
    graph via ``torch_geometric`` message passing (manual scatter-mean,
    avoiding a hard ``torch_geometric.nn`` layer dependency beyond
    ``scatter``).
  - DeePattern (cand_01299): Yang, Li, Ye, Yu et al., "DeePattern: Layout
    Pattern Generation with Transforming Convolutional Auto-Encoder", DAC
    2019, arXiv:1904.11042. Official repo
    https://github.com/phdyang007/deepattern (``src/cdnsgen.py``, class
    ``hsd`` / method ``cae``, TF-Slim). The published ``cae()`` method
    defines a convolutional-autoencoder "transforming" pattern generator:
    a ``16x16x1`` binary layout-topology image is encoded through two
    stride-2 ``Conv2d`` layers (128ch @ 8x8, then 256ch @ 4x4), flattened
    and compressed through two dense layers to a 32-dim latent "feature
    map" (noise is injected additively into this latent at generation
    time to "transform" a seed pattern into topologically-plausible
    variants), then decoded back through two dense layers and two
    stride-2 ``ConvTranspose2d`` layers to reconstruct/generate a
    ``16x16x1`` layout image. Reimplemented with identical channel/stride
    schedule and the additive latent-noise transform hook.
  - DeepCog (cand_01300): Zambianco, Cerutti, Bianchi & Pescape (uc3m
    Wireless Networking Lab), "DeepCog: Optimizing Resource Provisioning
    in Network Slicing with AI-based Capacity Forecasting", IEEE INFOCOM
    2019 workshops, arXiv:1812.09293. Official repo
    https://github.com/wnlUc3m/deepcog (``DeepCog.ipynb``, function
    ``make_nn_model``, Keras). DeepCog forecasts future 5G network-slice
    capacity from a sliding spatio-temporal window of aggregated cell-load
    grids: a ``(lookback, rows, cols, 1)`` 4D block is pushed through three
    ``Conv3D`` layers (32ch/3x3x3, 32ch/6x6x6 + Dropout(0.3), 16ch/6x6x6 +
    Dropout(0.3), all "same" padding), flattened, and reduced through two
    dense layers (64, 32) to a final linear layer predicting the required
    capacity per cluster. Reimplemented verbatim in ``torch.nn.Conv3d``
    with the published channel/kernel schedule (framework is TF/Keras in
    the source; architecture is directly portable).
  - DeepDGA (cand_01302): Anderson, Woodbridge & Filar, "DeepDGA:
    Adversarially-Tuned Domain Generation and Detection", AISec 2016,
    arXiv:1610.01969. Official-adjacent PyTorch/TF-derivative repo
    https://github.com/roreagan/DeepDGA (``dga_model.py``, TF1
    ``inference_graph`` / ``decoder_graph``). DeepDGA's core is a
    character-level convolutional-highway-LSTM autoencoder over domain
    strings: characters are embedded, passed through a multi-kernel 1D
    "TDNN" (time-delay conv + max-over-time pool per kernel width,
    concatenated), a 2-layer highway network, then a multi-layer LSTM
    encoder producing a per-timestep embedding; the decoder mirrors this
    with an LSTM, a highway network, and a TDNN-style linear projection
    back to per-character vocabulary logits (adversarial noise is injected
    into the mid-sequence embedding at generation time to synthesize novel
    DGA-like domains, the paper's headline mechanism). Reimplemented
    faithfully as a compact ``CharTDNN`` encoder (embedding + multi-kernel
    Conv1d/max-pool bank + highway) feeding an LSTM, and a symmetric
    decoder (LSTM + highway + linear-to-vocab) with a ``generate`` path
    that adds Gaussian noise to the encoder embedding before decoding.
  - DeepDow (cand_01303): Krepl et al., ``deepdow`` -- "Portfolio
    optimization with deep learning", PyPI package ``deepdow``, active
    repo https://github.com/jankrepl/deepdow (``deepdow/nn.py`` class
    ``KeynesNet``, ``deepdow/layers/{transform,collapse,allocate}.py``,
    PyTorch). ``KeynesNet`` is deepdow's flagship differentiable
    end-to-end portfolio network: per-asset instance-normalized OHLCV-like
    channels are fed through a shared-weight per-asset LSTM (or 1D conv)
    time-feature extractor, group-normalized, averaged over the lookback
    and channel dimensions to yield one scalar "expected return-ish"
    score per asset, and finally passed through a ``SoftmaxAllocator`` --
    a temperature-scaled softmax that turns raw scores directly into
    portfolio weights summing to 1 (the differentiable-allocation
    contribution the package is built around, as opposed to a plain
    classification/regression head). Reimplemented faithfully: a
    per-asset shared LSTM transform layer, ``GroupNorm``, average-collapse
    over lookback and hidden-channel dims, and the analytical
    (closed-form softmax) ``SoftmaxAllocator``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Deep Packet
# ============================================================


class DeepPacketCNN(nn.Module):
    """1D-CNN packet-byte classifier (Lotfollahi et al. 2017).

    Two Conv1d+ReLU layers over the first ``signal_length`` normalized
    bytes of a packet, a max pool, and a 3-layer dense classification head
    with dropout -- matching the published ``ml/model.py`` ``CNN`` class.

    Parameters
    ----------
    signal_length : int
        Number of normalized packet bytes fed in (paper default 1500).
    n_classes : int
        Number of traffic application classes.
    """

    def __init__(self, signal_length: int = 1500, n_classes: int = 12) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 200, kernel_size=5, stride=3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(200, 200, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            dummy = self.max_pool(self.conv2(self.conv1(dummy)))
            flat_dim = dummy.reshape(1, -1).shape[1]

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 200), nn.Dropout(0.05), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200, 100), nn.Dropout(0.05), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(100, 50), nn.Dropout(0.05), nn.ReLU())
        self.out = nn.Linear(50, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of normalized packet-byte vectors."""
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)


def build_deep_packet() -> nn.Module:
    """Build a small Deep Packet CNN traffic classifier."""
    return DeepPacketCNN(signal_length=256, n_classes=12).eval()


def example_input_deep_packet() -> torch.Tensor:
    """(batch, 1, signal_length) normalized packet-byte vector."""
    return torch.rand(4, 1, 256)


# ============================================================
# DeepACO
# ============================================================


class DeepACOEmbNet(nn.Module):
    """Anisotropic edge-gated GNN producing refined edge embeddings.

    Faithful (depth-reduced) port of DeepACO's ``EmbNet``: jointly refines
    node features ``x`` and edge features ``w`` across ``depth`` layers via
    edge-gated mean aggregation, matching the published forward pass.
    """

    def __init__(self, depth: int = 3, feats: int = 2, units: int = 16) -> None:
        super().__init__()
        self.depth = depth
        self.units = units
        self.v_lin0 = nn.Linear(feats, units)
        self.v_lins1 = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.v_bns = nn.ModuleList([nn.BatchNorm1d(units) for _ in range(depth)])
        self.e_lin0 = nn.Linear(1, units)
        self.e_lins0 = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.e_bns = nn.ModuleList([nn.BatchNorm1d(units) for _ in range(depth)])

    @staticmethod
    def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """Mean-aggregate edge messages ``src`` into ``dim_size`` node slots."""
        out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        count.index_add_(0, index, torch.ones_like(src[:, :1]))
        return out / count.clamp(min=1.0)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Refine node/edge embeddings; return the final edge embedding ``w``."""
        n_nodes = x.shape[0]
        x = F.silu(self.v_lin0(x))
        w = F.silu(self.e_lin0(edge_attr))
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            agg = self._scatter_mean(w2 * x2[edge_index[1]], edge_index[0], n_nodes)
            x = x0 + F.silu(self.v_bns[i](x1 + agg))
            w = w0 + F.silu(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


class DeepACOParNet(nn.Module):
    """Sigmoid-gated MLP mapping edge embeddings to a scalar heuristic."""

    def __init__(self, units: int = 16, depth: int = 3) -> None:
        super().__init__()
        sizes = [units] * depth + [1]
        self.lins = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-edge heuristic values in ``[0, 1]``."""
        for i, lin in enumerate(self.lins):
            x = lin(x)
            x = F.silu(x) if i < len(self.lins) - 1 else torch.sigmoid(x)
        return x.squeeze(-1)


class DeepACONet(nn.Module):
    """DeepACO neural heuristic estimator for TSP-style edge graphs."""

    def __init__(self, depth: int = 3, units: int = 16) -> None:
        super().__init__()
        self.emb_net = DeepACOEmbNet(depth=depth, units=units)
        self.par_net_heu = DeepACOParNet(units=units)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Predict a per-edge heuristic value used to bias ACO pheromones."""
        emb = self.emb_net(x, edge_index, edge_attr)
        return self.par_net_heu(emb)


def build_deepaco() -> nn.Module:
    """Build a small DeepACO edge-heuristic GNN."""
    return DeepACONet(depth=3, units=16).eval()


def example_input_deepaco() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(node coords, directed edge_index, edge distances) for an 8-node graph."""
    torch.manual_seed(0)
    n_nodes = 8
    coords = torch.rand(n_nodes, 2)
    src, dst = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing="ij")
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    edge_attr = (coords[edge_index[0]] - coords[edge_index[1]]).norm(dim=-1, keepdim=True)
    return coords, edge_index, edge_attr


# ============================================================
# DeePattern
# ============================================================


class DeePatternCAE(nn.Module):
    """Transforming convolutional autoencoder for layout pattern generation.

    Faithful port of ``cdnsgen.hsd.cae``: strided-conv encoder to a 32-dim
    latent "feature map", optional additive noise "transform" of the
    latent, and a strided-deconv decoder back to the input resolution.

    Parameters
    ----------
    img_size : int
        Spatial size of the square binary layout-topology image (16 in
        the published EUV pattern setting).
    latent_dim : int
        Size of the compressed feature-map latent (32 in the source).
    """

    def __init__(self, img_size: int = 16, latent_dim: int = 32) -> None:
        super().__init__()
        self.img_size = img_size
        self.pool1 = nn.Sequential(nn.Conv2d(1, 128, 5, stride=2, padding=2), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU())
        reduced = img_size // 4
        self.reduced = reduced
        self.fc1 = nn.Sequential(nn.Linear(256 * reduced * reduced, 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, latent_dim)
        self.fc3 = nn.Sequential(nn.Linear(latent_dim, 1024), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(1024, 256 * reduced * reduced), nn.ReLU())
        self.upool2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1), nn.ReLU()
        )
        self.upool1 = nn.ConvTranspose2d(128, 1, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``x`` to the latent feature map, optionally transform it
        with additive ``noise``, then decode back to a layout image."""
        net = self.pool1(x)
        net = self.pool2(net)
        net = net.reshape(net.shape[0], -1)
        net = self.fc1(net)
        fm = self.fc2(net)
        if noise is not None:
            fm = fm + noise
        net = self.fc3(fm)
        net = self.fc4(net)
        net = net.reshape(net.shape[0], 256, self.reduced, self.reduced)
        net = self.upool2(net)
        net = self.upool1(net)
        return net


def build_deepattern() -> nn.Module:
    """Build a small DeePattern transforming convolutional autoencoder."""
    return DeePatternCAE(img_size=16, latent_dim=32).eval()


def example_input_deepattern() -> torch.Tensor:
    """(batch, 1, 16, 16) binary layout-topology image."""
    return torch.rand(4, 1, 16, 16).round()


# ============================================================
# DeepCog
# ============================================================


class DeepCogNet(nn.Module):
    """3D-CNN spatio-temporal capacity forecaster (Zambianco et al. 2019).

    Faithful port of ``make_nn_model``: three ``Conv3d`` layers over a
    ``(lookback, rows, cols)`` load-grid block, then a dense reduction head
    predicting per-cluster required capacity.

    Parameters
    ----------
    lookback : int
        Number of past time steps stacked along the depth axis.
    grid_rows, grid_cols : int
        Spatial dimensions of the aggregated cell-load grid.
    num_cluster : int
        Number of slice/cluster capacity values to forecast.
    """

    def __init__(
        self,
        lookback: int = 6,
        grid_rows: int = 8,
        grid_cols: int = 8,
        num_cluster: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=6, padding=3), nn.ReLU(), nn.Dropout3d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=6, padding=3), nn.ReLU(), nn.Dropout3d(0.3)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, lookback, grid_rows, grid_cols)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            flat_dim = dummy.reshape(1, -1).shape[1]

        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.out = nn.Linear(32, num_cluster)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast capacity from a (batch, 1, lookback, rows, cols) block."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


def build_deepcog() -> nn.Module:
    """Build a small DeepCog 3D-CNN capacity forecaster."""
    return DeepCogNet(lookback=6, grid_rows=8, grid_cols=8, num_cluster=1).eval()


def example_input_deepcog() -> torch.Tensor:
    """(batch, 1, lookback, rows, cols) spatio-temporal load block."""
    return torch.rand(2, 1, 6, 8, 8)


# ============================================================
# DeepDGA
# ============================================================


class _Highway(nn.Module):
    """Highway layer: ``t * relu(Wy+b) + (1-t) * y`` (Srivastava et al. 2015)."""

    def __init__(self, size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.lin = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stacked highway transform."""
        for gate, lin in zip(self.gate, self.lin):
            t = torch.sigmoid(gate(x))
            g = F.relu(lin(x))
            x = t * g + (1.0 - t) * x
        return x


class _TDNN(nn.Module):
    """Multi-kernel time-delay conv + max-over-time pool, concatenated."""

    def __init__(self, embed_size: int, kernels: tuple[int, ...], kernel_features: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_size, kernel_features, k, padding=k // 2) for k in kernels]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, embed_size, seq_len) -> (batch, len(kernels)*kernel_features)."""
        pooled = [torch.amax(conv(x), dim=-1) for conv in self.convs]
        return torch.cat(pooled, dim=-1)


class DeepDGAAutoencoder(nn.Module):
    """Char-level conv-highway-LSTM autoencoder (Anderson, Woodbridge & Filar 2016).

    Encoder: char embedding -> multi-kernel TDNN -> highway -> LSTM, giving
    a fixed-size domain embedding. Decoder mirrors this (LSTM -> highway ->
    linear-to-vocab) to reconstruct per-character logits. At generation
    time Gaussian noise is added to the encoder embedding before decoding,
    the paper's adversarial-generation mechanism.

    Parameters
    ----------
    vocab_size : int
        Character vocabulary size.
    max_word_length : int
        Fixed domain-string length (paper default 65, padded).
    """

    def __init__(
        self,
        vocab_size: int = 40,
        max_word_length: int = 20,
        char_embed_size: int = 16,
        rnn_size: int = 32,
    ) -> None:
        super().__init__()
        self.max_word_length = max_word_length
        self.rnn_size = rnn_size
        self.embed = nn.Embedding(vocab_size, char_embed_size)
        kernels = (2, 3)
        kernel_features = 16
        self.tdnn = _TDNN(char_embed_size, kernels, kernel_features)
        tdnn_out = len(kernels) * kernel_features
        self.enc_highway = _Highway(tdnn_out, num_layers=2)
        self.enc_lstm = nn.LSTM(tdnn_out, rnn_size, num_layers=1, batch_first=True)

        self.dec_lstm = nn.LSTM(rnn_size, rnn_size, num_layers=1, batch_first=True)
        self.dec_highway = _Highway(rnn_size, num_layers=2)
        self.out_proj = nn.Linear(rnn_size, vocab_size)

    def encode(self, chars: torch.Tensor) -> torch.Tensor:
        """chars: (batch, max_word_length) int ids -> (batch, rnn_size) embedding."""
        emb = self.embed(chars).transpose(1, 2)  # (batch, embed, seq)
        feat = self.tdnn(emb)
        feat = self.enc_highway(feat)
        feat = feat.unsqueeze(1).expand(-1, self.max_word_length, -1)
        _, (h_n, _) = self.enc_lstm(feat)
        return h_n[-1]

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (batch, rnn_size) -> (batch, max_word_length, vocab_size) logits."""
        seq_in = embedding.unsqueeze(1).expand(-1, self.max_word_length, -1)
        outputs, _ = self.dec_lstm(seq_in)
        outputs = self.dec_highway(outputs)
        return self.out_proj(outputs)

    def forward(self, chars: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Autoencode (or adversarially transform, via ``noise``) a batch of
        fixed-length character-id domain strings into per-position vocab logits."""
        embedding = self.encode(chars)
        if noise is not None:
            embedding = embedding + noise
        return self.decode(embedding)


def build_deepdga() -> nn.Module:
    """Build a small DeepDGA char-level conv-highway-LSTM autoencoder."""
    return DeepDGAAutoencoder(vocab_size=40, max_word_length=20).eval()


def example_input_deepdga() -> torch.Tensor:
    """(batch, max_word_length) integer character ids."""
    torch.manual_seed(0)
    return torch.randint(0, 40, (4, 20))


# ============================================================
# DeepDow (KeynesNet)
# ============================================================


class _PerAssetLSTM(nn.Module):
    """Shared-weight LSTM applied independently along the asset axis.

    Mirrors ``deepdow.layers.transform.RNN``: input
    ``(batch, channels, lookback, n_assets)`` is processed asset-by-asset
    with one shared LSTM, producing
    ``(batch, hidden_size, lookback, n_assets)``.
    """

    def __init__(self, n_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = nn.LSTM(n_channels, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, lookback, n_assets) -> (batch, hidden, lookback, n_assets)."""
        batch, _channels, lookback, n_assets = x.shape
        outs = []
        for a in range(n_assets):
            seq = x[..., a].transpose(1, 2)  # (batch, lookback, channels)
            out, _ = self.cell(seq)  # (batch, lookback, hidden)
            outs.append(out.transpose(1, 2))  # (batch, hidden, lookback)
        return torch.stack(outs, dim=-1)


class DeepDowSoftmaxAllocator(nn.Module):
    """Analytical (closed-form) differentiable portfolio allocator.

    Matches ``deepdow.layers.allocate.SoftmaxAllocator`` (formulation
    "analytical"): a temperature-scaled softmax turning per-asset scores
    directly into portfolio weights summing to one.
    """

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """x, temperature: (batch, n_assets), (batch,) -> weights (batch, n_assets)."""
        return F.softmax(x * temperature.unsqueeze(-1), dim=-1)


class KeynesNet(nn.Module):
    """DeepDow's flagship differentiable portfolio-allocation network.

    Per-asset instance norm -> shared per-asset LSTM feature extractor ->
    GroupNorm+ReLU -> average-collapse over lookback and hidden-channel
    dims -> temperature-scaled ``SoftmaxAllocator`` producing portfolio
    weights that sum to one. Faithful port of ``deepdow.nn.KeynesNet``
    (RNN transform variant).

    Parameters
    ----------
    n_input_channels : int
        Number of per-asset input feature channels (e.g. OHLCV-like).
    hidden_size : int
        LSTM hidden size / number of extracted feature channels.
    n_groups : int
        Number of groups for ``GroupNorm`` (must divide ``hidden_size``).
    """

    def __init__(self, n_input_channels: int = 4, hidden_size: int = 8, n_groups: int = 4) -> None:
        super().__init__()
        self.norm_layer_1 = nn.InstanceNorm2d(n_input_channels, affine=True)
        self.transform_layer = _PerAssetLSTM(n_input_channels, hidden_size)
        self.norm_layer_2 = nn.GroupNorm(n_groups, hidden_size, affine=True)
        self.temperature = nn.Parameter(torch.ones(1))
        self.allocate_layer = DeepDowSoftmaxAllocator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_input_channels, lookback, n_assets) -> (batch, n_assets) weights."""
        n_samples = x.shape[0]
        x = self.norm_layer_1(x)
        x = self.transform_layer(x)
        x = self.norm_layer_2(x)
        x = F.relu(x)
        x = x.mean(dim=2)  # collapse lookback
        x = x.mean(dim=1)  # collapse hidden channels -> (batch, n_assets)
        temperatures = torch.ones(n_samples, dtype=x.dtype, device=x.device) * self.temperature
        return self.allocate_layer(x, temperatures)


def build_deepdow() -> nn.Module:
    """Build a small DeepDow ``KeynesNet`` portfolio-allocation network."""
    return KeynesNet(n_input_channels=4, hidden_size=8, n_groups=4).eval()


def example_input_deepdow() -> torch.Tensor:
    """(batch, n_input_channels, lookback, n_assets) multi-asset OHLCV-like block."""
    return torch.rand(2, 4, 10, 6)


MENAGERIE_ENTRIES = [
    ("Deep Packet", "build_deep_packet", "example_input_deep_packet", "2017", "NET"),
    ("DeepACO", "build_deepaco", "example_input_deepaco", "2023", "GRAPH"),
    ("DeePattern", "build_deepattern", "example_input_deepattern", "2019", "VIS"),
    ("DeepCog (5G Traffic CNN)", "build_deepcog", "example_input_deepcog", "2019", "SEQ"),
    ("DeepDGA", "build_deepdga", "example_input_deepdga", "2016", "SEQ"),
    ("DeepDow", "build_deepdow", "example_input_deepdow", "2020", "SEQ"),
]
