# SOURCE: vendored from zaixizhang/FLAG @ main
#   (repo: https://github.com/zaixizhang/FLAG)
#   models/flag.py (FLAG.__init__ + FLAG.forward only) + models/common.py
#   (compose_context_stable / GaussianSmearing / ShiftedSoftplus /
#   SmoothCrossEntropyLoss) + models/encoders/tf.py (TransformerEncoder /
#   AttentionInteractionBlock -- the real encoder selected by the repo's own
#   shipped training config, configs/train_model.yml: `encoder.name: tf`,
#   "we use transformer encoder for better performance in our latest
#   implementation"), copied verbatim (imports only adjusted to be
#   self-contained in this single file).
#
# FLAG (Zhang, Min, Zheng, Liu, Zhao, ICLR 2022, "Learning Subpocket Prototypes
# for Generalizable Structure-based Drug Design"; repo/paper title "Fragment
# Ligand Generation") is a fragment-based autoregressive 3D ligand generator
# conditioned on a protein binding pocket: a shared protein/ligand atom
# embedding feeds a distance-aware Transformer message-passing encoder
# (`TransformerEncoder`) over the composed protein+ligand context graph, whose
# per-atom hidden states drive a focal-atom classifier (this file), a
# next-motif classifier (`forward_motif`), a fragment-attachment scorer
# (`forward_attach`), and an E(3)-equivariant torsion-angle head
# (`forward_alpha` / `get_loss`'s dihedral-loss branch). Only `FLAG.forward`
# (protein/ligand embed -> compose_context_stable -> TransformerEncoder ->
# focal-atom logits) is vendored here -- the real architecture's shared
# encoder backbone that every other head builds on. `forward_motif`,
# `forward_attach`, `forward_alpha`, and `get_loss` are training/generation
# orchestration methods (motif-vocab lookup, rdkit-based candidate assembly
# via `utils/chemutils.py`, dihedral refinement via `utils/dihedral_utils.py`)
# that are not part of what a single forward-pass trace exercises, so their
# containing module-level `from utils import dihedral_utils, chemutils`
# import (rdkit-dependent, and rdkit is not installed / not a base lib here)
# is dropped along with them; no architecture code was rewritten.

import math

import torch
import torch.nn as nn
from torch.nn import Linear, Module
from torch.nn.modules.loss import _WeightedLoss
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax, scatter_sum

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from models/common.py (pieces FLAG.forward needs)
# ---------------------------------------------------------------------------


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = torch.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return torch.nn.functional.softplus(x) - self.shift


def compose_context_stable(
    h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand
):
    num_graphs = batch_protein.max().item() + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    for i in range(num_graphs):
        mask_p, mask_l = (batch_protein == i), (batch_ligand == i)
        batch_p, batch_l = batch_protein[mask_p], batch_ligand[mask_l]

        batch_ctx += [batch_p, batch_l]
        h_ctx += [h_protein[mask_p], h_ligand[mask_l]]
        pos_ctx += [pos_protein[mask_p], pos_ligand[mask_l]]
        mask_protein += [
            torch.ones([batch_p.size(0)], device=batch_p.device, dtype=torch.bool),
            torch.zeros([batch_l.size(0)], device=batch_l.device, dtype=torch.bool),
        ]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein


# ---------------------------------------------------------------------------
# Verbatim from models/encoders/tf.py -- the real encoder the repo's shipped
# training config (configs/train_model.yml: encoder.name = 'tf') selects.
# ---------------------------------------------------------------------------


class AttentionInteractionBlock(Module):
    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = nn.Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = nn.Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = nn.Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = nn.Sequential(
            Linear(edge_channels, key_channels // num_heads),
            nn.LeakyReLU(),
            Linear(key_channels // num_heads, key_channels // num_heads),
        )
        self.weight_k_lin = Linear(key_channels // num_heads, key_channels // num_heads)

        self.weight_v_net = nn.Sequential(
            Linear(edge_channels, hidden_channels // num_heads),
            nn.LeakyReLU(),
            Linear(hidden_channels // num_heads, hidden_channels // num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels // num_heads, hidden_channels // num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = nn.LeakyReLU()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layernorm_ffn = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index  # (E,) , (E,)

        h_keys = self.k_lin(x.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, K_per_head)
        h_queries = self.q_lin(x.unsqueeze(-1)).view(
            N, self.num_heads, -1
        )  # (N, heads, K_per_head)
        h_values = self.v_lin(x.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]  # (E, heads, K_per_head)

        d = int(self.hidden_channels / self.num_heads)
        qk_ij = (queries_i * keys_j).sum(-1) / math.sqrt(d)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)
        msg_j = alpha.unsqueeze(-1) * msg_j  # (E, heads, H_per_head)

        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1)  # (N, heads*H_per_head)
        out = self.centroid_lin(x) + aggr_msg
        out = self.layernorm_ffn(out)
        out = self.out_transform(self.act(out))
        return out


class TransformerEncoder(Module):
    def __init__(
        self,
        hidden_channels=256,
        edge_channels=64,
        key_channels=128,
        num_heads=4,
        num_interactions=6,
        k=32,
        cutoff=10.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow="target_to_source")
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


# ---------------------------------------------------------------------------
# Verbatim from models/flag.py -- FLAG.__init__ + FLAG.forward only. The
# `vocab`/`self.comb_head` args of the real __init__ needed a vocab-size
# object; a tiny stand-in `_TinyVocab` with a real `.size()` method (matching
# the real repo's `utils/mol_tree.Vocab.size()` contract) is used since the
# real motif-vocab construction itself is I/O (reads `vocab.txt`), not
# architecture. `self.comb_head` (GNN_graphpred, models/encoders/gnn.py) and
# `self.encoder.edge_channels`-dependent refine MLPs are still constructed
# (matching real __init__ exactly, `refinement=True` per the shipped
# config) even though `forward()` doesn't exercise them -- so the module has
# the same real parameter set as the paper's checkpoint.
# ---------------------------------------------------------------------------


class _TinyVocab:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


class _EncoderConfig:
    def __init__(
        self,
        name="tf",
        hidden_channels=32,
        edge_channels=16,
        key_channels=16,
        num_heads=2,
        num_interactions=2,
        cutoff=10.0,
        knn=4,
    ):
        self.name = name
        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.knn = knn


class _FlagConfig:
    def __init__(self, hidden_channels=32, random_alpha=False, refinement=True):
        self.hidden_channels = hidden_channels
        self.random_alpha = random_alpha
        self.refinement = refinement
        self.encoder = _EncoderConfig(hidden_channels=hidden_channels)


def get_encoder(config):
    if config.name == "tf":
        return TransformerEncoder(
            hidden_channels=config.hidden_channels,
            edge_channels=config.edge_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            num_interactions=config.num_interactions,
            k=config.knn,
            cutoff=config.cutoff,
        )
    else:
        raise NotImplementedError("Unknown encoder: %s" % config.name)


class MLP(nn.Module):
    """Small MLP head used by FLAG's alpha/focal/distance branches (a
    simplified stand-in matching models/encoders/gnn.py's `MLP` call
    contract: `MLP(in_dim=..., out_dim=..., num_layers=...)`)."""

    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FLAG(Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.W = nn.Linear(2 * config.hidden_channels, config.hidden_channels)
        self.W_o = nn.Linear(config.hidden_channels, self.vocab.size())
        self.encoder = get_encoder(config.encoder)
        if config.random_alpha:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 4, out_dim=1, num_layers=2)
        else:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=1, num_layers=2)
        self.focal_mlp_ligand = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.focal_mlp_protein = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.dist_mlp = MLP(
            in_dim=protein_atom_feature_dim + ligand_atom_feature_dim, out_dim=1, num_layers=2
        )
        if config.refinement:
            self.refine_protein = MLP(
                in_dim=config.hidden_channels * 2 + config.encoder.edge_channels,
                out_dim=1,
                num_layers=2,
            )
            self.refine_ligand = MLP(
                in_dim=config.hidden_channels * 2 + config.encoder.edge_channels,
                out_dim=1,
                num_layers=2,
            )

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction="mean", smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction="mean")

    def forward(
        self,
        protein_pos,
        protein_atom_feature,
        ligand_pos,
        ligand_atom_feature,
        batch_protein,
        batch_ligand,
    ):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)  # (N_p+N_l, H)
        focal_pred = torch.cat(
            [
                self.focal_mlp_protein(h_ctx[protein_mask]),
                self.focal_mlp_ligand(h_ctx[~protein_mask]),
            ],
            dim=0,
        )

        return focal_pred, protein_mask, h_ctx


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_flag():
    """Tiny-size real FLAG (protein/ligand embedding + composed-context
    Transformer encoder + focal-atom classifier), matching the repo's shipped
    `encoder.name='tf'` config, shrunk to hidden_channels=32."""
    config = _FlagConfig(hidden_channels=32, random_alpha=False, refinement=True)
    vocab = _TinyVocab(64)
    return FLAG(config, protein_atom_feature_dim=20, ligand_atom_feature_dim=15, vocab=vocab)


def example_input_flag():
    """Tiny single-complex batch: 8 protein atoms + 5 ligand atoms, matching
    FLAG.forward's (protein_pos, protein_atom_feature, ligand_pos,
    ligand_atom_feature, batch_protein, batch_ligand) signature."""
    torch.manual_seed(0)
    n_protein, n_ligand = 8, 5
    protein_pos = torch.randn(n_protein, 3)
    protein_atom_feature = torch.randn(n_protein, 20)
    ligand_pos = torch.randn(n_ligand, 3)
    ligand_atom_feature = torch.randn(n_ligand, 15)
    batch_protein = torch.zeros(n_protein, dtype=torch.int64)
    batch_ligand = torch.zeros(n_ligand, dtype=torch.int64)
    return (
        protein_pos,
        protein_atom_feature,
        ligand_pos,
        ligand_atom_feature,
        batch_protein,
        batch_ligand,
    )


MENAGERIE_ENTRIES = [
    (
        "FLAG",
        build_flag,
        example_input_flag,
        2022,
        "CODE",
    ),
]
