"""Compact faithful reimplementations for dependency-gated REIMPL3 batch 3.

Paper: Graphormer, Exphormer, GVP-GNN, HyperNetworks, Occupancy Networks, GIRAFFE.

The models in this module are deliberately small random-initialized PyTorch
reconstructions of install-heavy architectures.  Each keeps the audited
load-bearing primitive: graph structural attention biases, expander sparse graph
attention, scalar/vector geometric messages, gated linear recurrence, dynamic
hypernetwork weight generation, conditional-batch-normalized implicit occupancy
decoding, and compositional neural feature fields.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphormerSlim(nn.Module):
    """Graphormer with degree centrality and shortest-path attention bias."""

    def __init__(self, d_model: int = 32, num_heads: int = 4, num_classes: int = 3) -> None:
        """Initialize the compact Graphormer.

        Parameters
        ----------
        d_model:
            Hidden feature width.
        num_heads:
            Number of attention heads.
        num_classes:
            Number of graph-level output logits.
        """

        super().__init__()
        self.node_proj = nn.Linear(6, d_model)
        self.degree_emb = nn.Embedding(8, d_model)
        self.spatial_bias = nn.Embedding(8, num_heads)
        self.edge_bias = nn.Linear(2, num_heads, bias=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run graph-level classification.

        Parameters
        ----------
        batch:
            Dictionary with ``x``, ``adj``, ``dist``, and ``edge_attr`` tensors.

        Returns
        -------
        torch.Tensor
            Graph logits.
        """

        x = batch["x"]
        degree = batch["adj"].sum(dim=-1).long().clamp(max=7)
        h = self.node_proj(x) + self.degree_emb(degree)
        spatial = self.spatial_bias(batch["dist"].long().clamp(max=7))
        edge = self.edge_bias(batch["edge_attr"])
        bias = (spatial + edge).permute(0, 3, 1, 2)
        bsz, heads, nodes, _ = bias.shape
        attn_mask = bias.reshape(bsz * heads, nodes, nodes)
        attn_out = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        h = h + attn_out
        h = h + self.ffn(h)
        return self.head(h.mean(dim=1))


class ExphormerGPSVirtualNode(nn.Module):
    """GraphGPS-style local MPNN plus Exphormer sparse attention and virtual node."""

    def __init__(self, d_model: int = 32, num_heads: int = 4, num_classes: int = 3) -> None:
        """Initialize the compact Exphormer-GPS model."""

        super().__init__()
        self.node_proj = nn.Linear(5, d_model)
        self.local_msg = nn.Linear(d_model, d_model)
        self.virtual = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Classify a graph with local, expander, and virtual-node attention edges."""

        x = self.node_proj(batch["x"])
        adj = batch["adj"]
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local = torch.bmm(adj, self.local_msg(x)) / deg
        h = x + F.relu(local)
        bsz, nodes, _ = h.shape
        h = torch.cat((self.virtual.expand(bsz, -1, -1), h), dim=1)
        sparse = torch.zeros(bsz, nodes + 1, nodes + 1, device=h.device, dtype=torch.bool)
        sparse[:, 0, :] = True
        sparse[:, :, 0] = True
        sparse[:, 1:, 1:] = adj.bool() | batch["expander"].bool()
        mask = (~sparse).repeat_interleave(self.attn.num_heads, dim=0)
        h = self.norm(h + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])
        return self.head(h[:, 0])


class GeometricVectorPerceptron(nn.Module):
    """GVP layer mapping scalar features and 3D vector channels jointly."""

    def __init__(self, s_in: int, v_in: int, s_out: int, v_out: int) -> None:
        """Initialize scalar/vector projections."""

        super().__init__()
        self.scalar = nn.Linear(s_in + v_in, s_out)
        self.vector = nn.Linear(v_in, v_out, bias=False)
        self.gate = nn.Linear(s_out, v_out)

    def forward(
        self, scalars: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply an equivariant vector projection with invariant vector norms."""

        norms = vectors.norm(dim=-1)
        s = F.relu(self.scalar(torch.cat((scalars, norms), dim=-1)))
        v = self.vector(vectors.transpose(-2, -1)).transpose(-2, -1)
        return s, v * torch.sigmoid(self.gate(s)).unsqueeze(-1)


class GvpGnn(nn.Module):
    """Protein-structure GVP-GNN with scalar/vector message passing."""

    def __init__(self, scalar_dim: int = 16, vector_dim: int = 4) -> None:
        """Initialize the compact GVP-GNN."""

        super().__init__()
        self.scalar_in = nn.Linear(6, scalar_dim)
        self.vector_in = nn.Linear(1, vector_dim, bias=False)
        self.msg = GeometricVectorPerceptron(scalar_dim, vector_dim, scalar_dim, vector_dim)
        self.update = GeometricVectorPerceptron(
            scalar_dim * 2, vector_dim * 2, scalar_dim, vector_dim
        )
        self.vector_readout = nn.Linear(vector_dim, scalar_dim)
        self.out = nn.Linear(scalar_dim, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run scalar/vector graph message passing on 3D coordinates."""

        coords = batch["coords"]
        scalars = F.relu(self.scalar_in(batch["node"]))
        vectors = self.vector_in(coords.unsqueeze(-2).transpose(-2, -1)).transpose(-2, -1)
        adj = batch["adj"]
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        msg_s, msg_v = self.msg(scalars, vectors)
        agg_s = torch.bmm(adj, msg_s) / deg
        agg_v = torch.einsum("bij,bjvc->bivc", adj / deg, msg_v)
        scalars, vectors = self.update(
            torch.cat((scalars, agg_s), dim=-1), torch.cat((vectors, agg_v), dim=-2)
        )
        vector_summary = self.vector_readout(vectors.norm(dim=-1).mean(dim=1))
        return self.out(scalars.mean(dim=1) + vector_summary)


class RgLruBlock(nn.Module):
    """Real-gated linear recurrent unit used by Hawk."""

    def __init__(self, d_model: int) -> None:
        """Initialize input-dependent recurrence parameters."""

        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.recurrent_logit = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan an input-dependent gated linear recurrence over a sequence."""

        value, input_gate, forget_gate = self.in_proj(x).chunk(3, dim=-1)
        value = torch.tanh(value)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate + self.recurrent_logit)
        state = torch.zeros_like(value[:, 0])
        outputs = []
        for step in range(x.shape[1]):
            state = forget_gate[:, step] * state + input_gate[:, step] * value[:, step]
            outputs.append(state)
        return self.out_proj(torch.stack(outputs, dim=1))


class HawkLanguageModel(nn.Module):
    """Hawk language model alternating RG-LRU and gated MLP blocks."""

    def __init__(self, vocab: int = 128, d_model: int = 32, depth: int = 2) -> None:
        """Initialize the compact Hawk model."""

        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.rglru = nn.ModuleList([RgLruBlock(d_model) for _ in range(depth)])
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return token logits."""

        x = self.embed(ids)
        for rec, mlp in zip(self.rglru, self.mlp, strict=True):
            x = x + rec(x)
            x = x + mlp(x)
        return self.head(x)


class HyperLstm(nn.Module):
    """Dynamic HyperLSTM where a small LSTM generates main LSTM modulation."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, hyper_dim: int = 12) -> None:
        """Initialize the compact HyperLSTM."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.hyper_cell = nn.LSTMCell(input_dim + hidden_dim, hyper_dim)
        self.main = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)
        self.hyper_scale = nn.Linear(hyper_dim, hidden_dim * 4)
        self.hyper_bias = nn.Linear(hyper_dim, hidden_dim * 4)
        self.head = nn.Linear(hidden_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan the sequence with hypernetwork-modulated LSTM gates."""

        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_dim)
        c = x.new_zeros(batch, self.hidden_dim)
        hh = x.new_zeros(batch, self.hyper_cell.hidden_size)
        hc = x.new_zeros(batch, self.hyper_cell.hidden_size)
        outputs = []
        for step in range(x.shape[1]):
            hyper_in = torch.cat((x[:, step], h), dim=-1)
            hh, hc = self.hyper_cell(hyper_in, (hh, hc))
            gates = self.main(hyper_in) * (1.0 + torch.tanh(self.hyper_scale(hh)))
            gates = gates + self.hyper_bias(hh)
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return self.head(torch.stack(outputs, dim=1))


class HmlpHyperNetwork(nn.Module):
    """Fully connected hypernetwork that emits a target MLP's weights."""

    def __init__(self, cond_dim: int = 6, hidden_dim: int = 32, target_hidden: int = 10) -> None:
        """Initialize the HMLP."""

        super().__init__()
        self.target_hidden = target_hidden
        out_dim = 4 * target_hidden + target_hidden + target_hidden * 3 + 3
        self.hyper = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond_and_x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Generate target weights from a condition and apply them to inputs."""

        cond, x = cond_and_x
        weights = self.hyper(cond)
        h = self.target_hidden
        offset = 0
        w1 = weights[:, offset : offset + 4 * h].view(-1, 4, h)
        offset += 4 * h
        b1 = weights[:, offset : offset + h]
        offset += h
        w2 = weights[:, offset : offset + h * 3].view(-1, h, 3)
        offset += h * 3
        b2 = weights[:, offset : offset + 3]
        hidden = torch.tanh(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)
        return torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2


class ChunkedHmlpHyperNetwork(nn.Module):
    """Chunked MLP hypernetwork using learned chunk embeddings."""

    def __init__(self, cond_dim: int = 6, chunk_dim: int = 8, chunks: int = 4) -> None:
        """Initialize the chunked HMLP."""

        super().__init__()
        self.chunk_emb = nn.Parameter(torch.randn(chunks, chunk_dim) * 0.02)
        self.chunk_net = nn.Sequential(
            nn.Linear(cond_dim + chunk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.readout = nn.Linear(chunks * 16, 3)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Emit chunk-wise generated target parameters and summarize them."""

        batch = cond.shape[0]
        chunks = self.chunk_emb.unsqueeze(0).expand(batch, -1, -1)
        cond_exp = cond.unsqueeze(1).expand(-1, chunks.shape[1], -1)
        generated = self.chunk_net(torch.cat((cond_exp, chunks), dim=-1))
        return self.readout(generated.flatten(1))


class ConditionalBatchNorm1d(nn.Module):
    """Conditional batch normalization from Occupancy Networks."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        """Initialize conditional affine maps."""

        super().__init__()
        self.bn = nn.BatchNorm1d(channels, affine=False)
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Normalize point features and apply condition-dependent affine terms."""

        batch, points, channels = x.shape
        normed = self.bn(x.reshape(batch * points, channels)).view(batch, points, channels)
        return normed * (1.0 + self.gamma(cond).unsqueeze(1)) + self.beta(cond).unsqueeze(1)


class OccNetDecoder(nn.Module):
    """Occupancy Network decoder over continuous 3D query points."""

    def __init__(self, cond_dim: int = 12, hidden_dim: int = 32, use_cbn: bool = False) -> None:
        """Initialize an implicit occupancy decoder."""

        super().__init__()
        self.use_cbn = use_cbn
        self.fc_p = nn.Linear(3, hidden_dim)
        self.fc_c = nn.Linear(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.cbn = nn.ModuleList([ConditionalBatchNorm1d(hidden_dim, cond_dim) for _ in range(3)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return occupancy logits for query points conditioned on a latent code."""

        points, cond = inputs
        h = self.fc_p(points) + self.fc_c(cond).unsqueeze(1)
        for idx, block in enumerate(self.blocks):
            residual = h
            h = block(F.relu(h))
            if self.use_cbn:
                h = self.cbn[idx](h, cond)
            h = h + residual
        return self.out(F.relu(h)).squeeze(-1)


class GiraffeGenerator(nn.Module):
    """GIRAFFE-style compositional neural feature field generator."""

    def __init__(self, latent_dim: int = 8, feature_dim: int = 16) -> None:
        """Initialize the compact GIRAFFE renderer."""

        super().__init__()
        self.shape = nn.Linear(latent_dim + 3, feature_dim)
        self.appearance = nn.Linear(latent_dim, feature_dim)
        self.density = nn.Linear(feature_dim, 1)
        self.color = nn.Linear(feature_dim, 3)
        self.neural_renderer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Render a small image from object latents, transforms, and ray samples."""

        latents, transforms, points = inputs
        batch, objects, samples, _ = points.shape
        local_points = points - transforms[:, :, None, :]
        shape_in = torch.cat(
            (local_points, latents[:, :, None, :].expand(-1, -1, samples, -1)), dim=-1
        )
        feat = F.relu(self.shape(shape_in))
        feat = feat + self.appearance(latents).unsqueeze(2)
        sigma = F.softplus(self.density(feat))
        color = torch.sigmoid(self.color(feat))
        weights = torch.softmax(sigma.squeeze(-1), dim=1).unsqueeze(-1)
        composited = (weights * color).sum(dim=1)
        side = int(math.sqrt(samples))
        image = composited.view(batch, side, side, 3).permute(0, 3, 1, 2)
        return self.neural_renderer(image)


def graph_batch() -> dict[str, torch.Tensor]:
    """Create a small graph batch for graph transformer models."""

    adj = torch.tensor(
        [
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ]
        ],
        dtype=torch.float32,
    )
    dist = torch.tensor(
        [
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 1, 1, 2, 2],
                [1, 1, 0, 2, 1, 2],
                [2, 1, 2, 0, 1, 1],
                [2, 2, 1, 1, 0, 1],
                [3, 2, 2, 1, 1, 0],
            ]
        ]
    )
    edge_attr = torch.stack((adj, adj * 0.5), dim=-1)
    return {"x": torch.randn(1, 6, 6), "adj": adj, "dist": dist, "edge_attr": edge_attr}


def exphormer_batch() -> dict[str, torch.Tensor]:
    """Create a graph batch with fixed expander edges."""

    batch = graph_batch()
    expander = torch.tensor(
        [
            [
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
            ]
        ],
        dtype=torch.float32,
    )
    return {"x": torch.randn(1, 6, 5), "adj": batch["adj"], "expander": expander}


def gvp_batch() -> dict[str, torch.Tensor]:
    """Create a protein-like 3D graph batch."""

    return {
        "node": torch.randn(1, 7, 6),
        "coords": torch.randn(1, 7, 3),
        "adj": (torch.rand(1, 7, 7) > 0.45).float(),
    }


def token_ids() -> torch.Tensor:
    """Create a compact token sequence."""

    return torch.randint(0, 128, (1, 12))


def sequence_input() -> torch.Tensor:
    """Create a compact continuous sequence."""

    return torch.randn(1, 9, 8)


def hyper_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create condition and target-network input tensors."""

    return torch.randn(2, 6), torch.randn(2, 4)


def condition_input() -> torch.Tensor:
    """Create a hypernetwork condition tensor."""

    return torch.randn(2, 6)


def occ_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create query points and latent condition for occupancy decoders."""

    return torch.randn(1, 20, 3), torch.randn(1, 12)


def giraffe_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create latents, translations, and ray-sample points for GIRAFFE."""

    return torch.randn(1, 2, 8), torch.randn(1, 2, 3) * 0.2, torch.randn(1, 2, 16, 3)


def build_graphormer_slim() -> nn.Module:
    """Build Graphormer-Slim."""

    return GraphormerSlim()


def build_exphormer_gps_virtual_node() -> nn.Module:
    """Build Exphormer-GPS with virtual node."""

    return ExphormerGPSVirtualNode()


def build_gvp_gnn() -> nn.Module:
    """Build GVP-GNN."""

    return GvpGnn()


def build_hawk() -> nn.Module:
    """Build Hawk."""

    return HawkLanguageModel()


def build_hyper_lstm() -> nn.Module:
    """Build HyperLSTM."""

    return HyperLstm()


def build_hmlp() -> nn.Module:
    """Build hypnettorch-style HMLP."""

    return HmlpHyperNetwork()


def build_chunked_hmlp() -> nn.Module:
    """Build hypnettorch-style chunked HMLP."""

    return ChunkedHmlpHyperNetwork()


def build_occnet_decoder() -> nn.Module:
    """Build an Occupancy Network decoder."""

    return OccNetDecoder(use_cbn=False)


def build_occnet_decoder_cbatchnorm() -> nn.Module:
    """Build an Occupancy Network decoder with conditional batch norm."""

    return OccNetDecoder(use_cbn=True)


def build_giraffe_generator() -> nn.Module:
    """Build a compact GIRAFFE generator."""

    return GiraffeGenerator()


MENAGERIE_ENTRIES = [
    ("Graphormer-Slim", "build_graphormer_slim", "graph_batch", "2021", "DC"),
    (
        "Exphormer-GPS-VirtualNode",
        "build_exphormer_gps_virtual_node",
        "exphormer_batch",
        "2023",
        "DC",
    ),
    ("gvp_gnn", "build_gvp_gnn", "gvp_batch", "2021", "DC"),
    ("hawk_pytorch", "build_hawk", "token_ids", "2024", "DC"),
    ("HyperLSTM", "build_hyper_lstm", "sequence_input", "2016", "DC"),
    ("HMLP_hypnettorch", "build_hmlp", "hyper_input", "2020", "DC"),
    ("ChunkedHMLP", "build_chunked_hmlp", "condition_input", "2020", "DC"),
    ("OccNet_Decoder", "build_occnet_decoder", "occ_input", "2019", "DC"),
    ("OccNet_DecoderCBatchNorm", "build_occnet_decoder_cbatchnorm", "occ_input", "2019", "DC"),
    ("giraffe_generator", "build_giraffe_generator", "giraffe_input", "2021", "DC"),
]
