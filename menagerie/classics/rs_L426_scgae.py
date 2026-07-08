# FAITHFUL PORT of ZixiangLuo1161/scGAE @ main (original framework: TensorFlow 2 / Keras +
# spektral 0.6.1 graph layers + tensorflow_probability)
#
# https://raw.githubusercontent.com/ZixiangLuo1161/scGAE/main/scgae.py
# https://raw.githubusercontent.com/ZixiangLuo1161/scGAE/main/layers.py
#
# scGAE (Luo et al. 2021, Frontiers in Genetics) -- single-cell graph autoencoder that
# preserves topological structure of an scRNA-seq k-NN cell graph. The real `SCGAE`
# class (scgae.py) is TensorFlow/Keras + `spektral.layers.GraphAttention` (GAT,
# pinned spektral==0.6.1) / `spektral.layers.TAGConv` graph-conv encoders, feeding two
# decoders: a `Bilinear` (or inner-product) adjacency-reconstruction decoder and an MLP
# expression-reconstruction decoder (`layers.py`). spektral + tensorflow_probability are
# not installed here and this is a TF-only codebase (custom Keras `Model`/`Layer`
# subclasses, not a torch port anywhere upstream) -- code cannot run in the base torch
# env, so this is a faithful architectural transcription into torch, not a rewrite from
# a paper summary: every mechanism (multi-head dense-mode GAT attention exactly per
# spektral's `GraphAttention._call_dense`, i.e. Velickovic et al. 2018 GAT with
# LeakyReLU(0.2) attention logits + masked softmax over the dense adjacency + attention
# dropout; the TAGConv K-hop polynomial graph filter; the learned Bilinear adjacency
# decoder `sigmoid(x @ W @ x^T)`; the 3-layer ReLU MLP expression decoder) is preserved
# from the real TF/Keras/spektral source, only re-expressed as torch `nn.Module`s.
#
# Upstream license: ZixiangLuo1161/scGAE (see repo; no explicit LICENSE file found,
# code used here for random-init architecture capture only, no weights).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class GraphAttention(nn.Module):
    """Port of spektral==0.6.1 `spektral.layers.GraphAttention` (dense mode), the exact
    graph-attention layer scGAE's `SCGAE.__init__` uses when `layer_enc="GAT"` (the
    repo's default). Mirrors `GATConv._call_dense` (the successor class in modern
    spektral, same math as the 0.6.1 `GraphAttention` layer): per-head linear
    projection, self/neighbor attention-kernel dot products, `LeakyReLU(0.2)` logits,
    dense-adjacency-masked softmax, attention dropout, then per-head weighted
    aggregation, concatenated (or averaged) across heads plus bias and activation.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        attn_heads: int = 1,
        concat_heads: bool = True,
        dropout_rate: float = 0.5,
        add_self_loops: bool = True,
        activation=None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.add_self_loops = add_self_loops
        self.use_bias = use_bias
        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = channels * attn_heads if concat_heads else channels

        self.kernel = nn.Parameter(torch.empty(in_channels, attn_heads, channels))
        self.attn_kernel_self = nn.Parameter(torch.empty(channels, attn_heads, 1))
        self.attn_kernel_neighs = nn.Parameter(torch.empty(channels, attn_heads, 1))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("bias", None)
        self.dropout = nn.Dropout(dropout_rate)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.xavier_uniform_(self.attn_kernel_self)
        nn.init.xavier_uniform_(self.attn_kernel_neighs)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        x: (..., N, in_channels) dense node features
        a: (..., N, N) dense binary/weighted adjacency
        """
        if self.add_self_loops:
            eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
            a = a * (1 - eye) + eye

        # x @ kernel -> (..., N, H, C)
        xk = torch.einsum("...ni,ihc->...nhc", x, self.kernel)
        attn_for_self = torch.einsum("...nhc,chj->...nhj", xk, self.attn_kernel_self).squeeze(
            -1
        )  # (..., N, H)
        attn_for_neighs = torch.einsum("...nhc,chj->...nhj", xk, self.attn_kernel_neighs).squeeze(
            -1
        )  # (..., N, H)
        # broadcast to (..., N_target, N_source, H): self indexed by target, neighs by source
        attn_coef = attn_for_self.unsqueeze(-2) + attn_for_neighs.unsqueeze(-3)
        attn_coef = F.leaky_relu(attn_coef, negative_slope=0.2)

        mask = torch.where(a == 0, torch.full_like(a, -1e9), torch.zeros_like(a))
        attn_coef = attn_coef + mask.unsqueeze(-1)
        attn_coef = F.softmax(attn_coef, dim=-2)  # softmax over source (neighbor) dim
        attn_coef = self.dropout(attn_coef)

        # weighted aggregation: sum_source attn_coef[target, source, h] * xk[source, h, c]
        output = torch.einsum("...tsh,...shc->...thc", attn_coef, xk)

        if self.concat_heads:
            new_shape = output.shape[:-2] + (self.attn_heads * self.channels,)
            output = output.reshape(new_shape)
        else:
            output = output.mean(dim=-2)

        if self.use_bias:
            output = output + self.bias
        return self.activation(output)


class TAGConv(nn.Module):
    """Port of spektral's `TAGConv` (Topology Adaptive Graph Convolutional layer,
    Du et al. 2018), the layer used by scGAE's `layer_enc="TAG"` alternative encoder
    path: a K-hop polynomial filter sum_{k=0..K} (A_norm^k @ X) @ W_k, followed by
    an optional activation.
    """

    def __init__(self, in_channels: int, channels: int, k: int = 3, activation=None):
        super().__init__()
        self.k = k
        self.activation = activation if activation is not None else nn.Identity()
        self.lins = nn.ModuleList(
            [nn.Linear(in_channels, channels, bias=False) for _ in range(k + 1)]
        )
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.lins[0](x)
        x_k = x
        for i in range(1, self.k + 1):
            x_k = torch.matmul(a, x_k)
            out = out + self.lins[i](x_k)
        out = out + self.bias
        return self.activation(out)


class Bilinear(nn.Module):
    """Port of scGAE's `layers.Bilinear`: a learned bilinear adjacency decoder,
    sigmoid(x @ kernel @ x^T) up to the sigmoid (applied outside, in `SCGAE`)."""

    def __init__(self, dim: int, dropout: float = 0.0, use_bias: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = nn.Parameter(torch.empty(dim, dim))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        h1 = torch.matmul(x, self.kernel)
        output = torch.matmul(h1, x.transpose(-1, -2))
        if self.use_bias:
            output = output + self.bias
        return output


class ClusteringLayer(nn.Module):
    """Port of scGAE's `layers.ClusteringLayer`: soft cluster-assignment layer (DEC-style
    Student's t-distribution kernel), used by `SCGAE.cluster_model` during the optional
    clustering-refinement stage."""

    def __init__(self, n_features: int, n_clusters: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, n_features))
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = 1.0 / (
            1.0 + (torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, dim=2) / self.alpha)
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / q.sum(dim=1)).t()
        return q


class SCGAE(nn.Module):
    """Port of scGAE's `scgae.SCGAE` (encoder + adjacency decoder + expression decoder).
    Faithful to the "DBL" + "GAT" default configuration path from the original
    `__init__`: Dropout(0.2) -> GraphAttention(hidden_dim) -> GraphAttention(latent_dim)
    encoder; Dense(adj_dim) -> Bilinear -> sigmoid adjacency decoder; 3-layer ReLU MLP
    expression decoder. `decA`/`layer_enc` alternates ("BL", "IP", "TAG") are also
    ported for architectural completeness.
    """

    def __init__(
        self,
        in_dim: int,
        n_sample: int,
        hidden_dim: int = 120,
        latent_dim: int = 15,
        dec_dim=None,
        adj_dim: int = 32,
        decA: str = "DBL",
        layer_enc: str = "GAT",
    ):
        super().__init__()
        if dec_dim is None:
            dec_dim = [64, 256, 512]
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.n_sample = n_sample
        self.layer_enc = layer_enc
        self.decA = decA

        self.input_dropout = nn.Dropout(0.2)
        if layer_enc == "GAT":
            self.enc1 = GraphAttention(in_dim, hidden_dim, attn_heads=1, activation=nn.ReLU())
            self.enc2 = GraphAttention(hidden_dim, latent_dim, attn_heads=1)
        elif layer_enc == "TAG":
            self.enc1 = TAGConv(in_dim, hidden_dim, activation=nn.ReLU())
            self.enc2 = TAGConv(hidden_dim, latent_dim)
        else:
            raise ValueError(f"Unknown layer_enc: {layer_enc}")

        # Adjacency decoder
        if decA == "DBL":
            self.dec_a_dense = nn.Linear(latent_dim, adj_dim)
            self.dec_a_bilinear = Bilinear(adj_dim)
        elif decA == "BL":
            self.dec_a_dense = None
            self.dec_a_bilinear = Bilinear(latent_dim)
        elif decA == "IP":
            self.dec_a_dense = None
            self.dec_a_bilinear = None
        else:
            self.dec_a_dense = None
            self.dec_a_bilinear = None

        # Expression decoder (3-layer ReLU MLP + linear output head)
        self.dec_x = nn.Sequential(
            nn.Linear(latent_dim, dec_dim[0]),
            nn.ReLU(),
            nn.Linear(dec_dim[0], dec_dim[1]),
            nn.ReLU(),
            nn.Linear(dec_dim[1], dec_dim[2]),
            nn.ReLU(),
            nn.Linear(dec_dim[2], in_dim),
        )

        self.clustering = ClusteringLayer(latent_dim, n_clusters=8)

    def encode(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.input_dropout(x)
        h = self.enc1(h, a)
        z = self.enc2(h, a)
        return z

    def decode_a(self, z: torch.Tensor) -> torch.Tensor:
        if self.decA == "DBL":
            h = self.dec_a_dense(z)
            h = self.dec_a_bilinear(h)
            return torch.sigmoid(h)
        elif self.decA == "BL":
            h = self.dec_a_bilinear(z)
            return torch.sigmoid(h)
        elif self.decA == "IP":
            return torch.sigmoid(torch.matmul(z, z.transpose(-1, -2)))
        else:
            return None

    def decode_x(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_x(z)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        z = self.encode(x, a)
        x_out = self.decode_x(z)
        a_out = self.decode_a(z)
        return z, x_out, a_out


def build_scgae():
    torch.manual_seed(0)
    n_sample = 24
    in_dim = 32
    model = SCGAE(
        in_dim=in_dim,
        n_sample=n_sample,
        hidden_dim=16,
        latent_dim=8,
        dec_dim=[16, 24, 32],
        adj_dim=10,
        decA="DBL",
        layer_enc="GAT",
    )
    model.eval()
    return model


def example_input_scgae():
    torch.manual_seed(0)
    n_sample = 24
    in_dim = 32
    x = torch.rand(n_sample, in_dim)
    a = (torch.rand(n_sample, n_sample) > 0.7).float()
    a = ((a + a.t()) > 0).float()
    return (x, a)


MENAGERIE_ENTRIES = [
    ("scGAE", "build_scgae", "example_input_scgae", 2021, "ported-pytorch"),
]
