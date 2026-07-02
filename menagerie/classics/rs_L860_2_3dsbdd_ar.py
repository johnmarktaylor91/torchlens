# SOURCE: vendored from luost26/3D-Generative-SBDD @ main (2026-07-01)
# Files combined from the real repo (paths as in upstream):
#   models/maskfill.py            (MaskFillModel -- the autoregressive 3D structure-based drug
#                                   design model: atom-by-atom next-atom-type/position prediction)
#   models/common.py              (compose_context, GaussianSmearing, ShiftedSoftplus,
#                                   SmoothCrossEntropyLoss, MultiLayerPerceptron, and batch helpers)
#   models/encoders/schnet.py     (SchNetEncoder, CFConv, InteractionBlock -- the default context
#                                   encoder selected by config.encoder.name == 'schnet')
#   models/fields/classifier.py   (SpatialClassifier -- the default query field selected by
#                                   config.field.name == 'classifier')
#
# 3D-SBDD "AR" (autoregressive) is an atom-by-atom generative model for structure-based drug design
# (Luo et al., NeurIPS 2021, "A 3D Generative Model for Structure-Based Drug Design"): a protein
# pocket + partially-generated ligand context is encoded with an SE(3)-invariant SchNet-style message
# passing network, then a spatial classifier field scores candidate 3D query points for the next atom
# type/position via a k-NN local aggregation. Code is transcribed verbatim from the real repo; only
# import paths were flattened into this single file, and the `get_encoder`/`get_field` config-driven
# factories were inlined to their default ('schnet' / 'classifier') branches -- the other encoder
# branch (CFTransformerEncoder) is a config alternative for the same model class, not a separate
# architecture.
#
# Original license: MIT (per repo, LICENSE file).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, Module, ModuleList, Sequential
from torch.nn.modules.loss import _WeightedLoss
from torch_geometric.nn import MessagePassing, knn, radius_graph
from torch_scatter import scatter_add, scatter_mean

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/common.py
# ---------------------------------------------------------------------------
def split_tensor_by_batch(x, batch, num_graphs=None):
    if num_graphs is None:
        num_graphs = batch.max().item() + 1
    x_split = []
    for i in range(num_graphs):
        mask = batch == i
        x_split.append(x[mask])
    return x_split


def concat_tensors_to_batch(x_split):
    x = torch.cat(x_split, dim=0)
    batch = torch.repeat_interleave(
        torch.arange(len(x_split)), repeats=torch.LongTensor([s.size(0) for s in x_split])
    ).to(device=x.device)
    return x, batch


def split_tensor_to_segments(x, segsize):
    num_segs = math.ceil(x.size(0) / segsize)
    segs = []
    for i in range(num_segs):
        segs.append(x[i * segsize : (i + 1) * segsize])
    return segs


def split_tensor_by_lengths(x, lengths):
    segs = []
    for length in lengths:
        segs.append(x[:length])
        x = x[length:]
    return segs


def batch_intersection_mask(batch, batch_filter):
    batch_filter = batch_filter.unique()
    mask = (batch.view(-1, 1) == batch_filter.view(1, -1)).any(dim=1)
    return mask


class MultiLayerPerceptron(nn.Module):
    """Multi-layer Perceptron. Note there is no activation or dropout in the last layer."""

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


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
        lsm = F.log_softmax(inputs, -1)

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
        return F.softplus(x) - self.shift


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]

    return h_ctx, pos_ctx, batch_ctx


# ---------------------------------------------------------------------------
# models/encoders/schnet.py
# ---------------------------------------------------------------------------
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.cutoff is not None:
            C = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
            W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        edge_channels=64,
        cutoff=10.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels, num_filters, cutoff)
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h


# ---------------------------------------------------------------------------
# models/fields/classifier.py
# ---------------------------------------------------------------------------
class SpatialClassifier(Module):
    def __init__(self, num_classes, num_indicators, in_channels, num_filters, k=32, cutoff=10.0):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, num_filters)
        self.nn = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.classifier = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_classes),
        )
        self.property_pred = Sequential(
            Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_indicators),
        )
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_filters)
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos_query, pos_ctx, node_attr_ctx, batch_query, batch_ctx):
        assign_idx = knn(x=pos_ctx, y=pos_query, k=self.k, batch_x=batch_ctx, batch_y=batch_query)

        dist_ij = torch.norm(pos_query[assign_idx[0]] - pos_ctx[assign_idx[1]], p=2, dim=-1).view(
            -1, 1
        )
        node_attr_ctx_j = node_attr_ctx[assign_idx[1]]

        W = self.nn(self.distance_expansion(dist_ij))
        h = self.lin2(W * self.lin1(node_attr_ctx_j))

        C = 0.5 * (torch.cos(dist_ij * math.pi / self.cutoff) + 1.0)
        C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
        h = h * C.view(-1, 1)

        y = scatter_add(h, index=assign_idx[0], dim=0, dim_size=pos_query.size(0))

        y_cls = self.classifier(y)
        y_ind = self.property_pred(y)

        return y_cls, y_ind


# ---------------------------------------------------------------------------
# models/maskfill.py -- the AR (autoregressive) 3D-SBDD model itself
# ---------------------------------------------------------------------------
class MaskFillModel(Module):
    """Atom-by-atom autoregressive 3D structure-based drug design model (3D-SBDD "AR")."""

    def __init__(
        self,
        hidden_channels,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        num_classes,
        num_indicators,
        encoder_num_filters=64,
        encoder_num_interactions=3,
        encoder_edge_channels=32,
        encoder_cutoff=6.0,
        field_num_filters=64,
        field_k=8,
        field_cutoff=6.0,
    ):
        super().__init__()

        self.protein_atom_emb = Linear(protein_atom_feature_dim, hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, hidden_channels)

        self.encoder = SchNetEncoder(
            hidden_channels=hidden_channels,
            num_filters=encoder_num_filters,
            num_interactions=encoder_num_interactions,
            edge_channels=encoder_edge_channels,
            cutoff=encoder_cutoff,
        )
        self.field = SpatialClassifier(
            num_classes=num_classes,
            num_indicators=num_indicators,
            in_channels=self.encoder.out_channels,
            num_filters=field_num_filters,
            k=field_k,
            cutoff=field_cutoff,
        )

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction="mean", smoothing=0.1)

    def forward(
        self,
        pos_query,
        protein_pos,
        protein_atom_feature,
        ligand_pos,
        ligand_atom_feature,
        batch_query,
        batch_protein,
        batch_ligand,
    ):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)

        y_cls, y_ind = self.field(
            pos_query=pos_query,
            pos_ctx=pos_ctx,
            node_attr_ctx=h_ctx,
            batch_query=batch_query,
            batch_ctx=batch_ctx,
        )

        return y_cls, y_ind


# ---------------------------------------------------------------------------
# Menagerie build/example plumbing
# ---------------------------------------------------------------------------
def build_3dsbdd_ar():
    """Tiny random-init 3D-SBDD AR (MaskFillModel) with the default SchNet encoder + spatial classifier field."""
    torch.manual_seed(0)
    return MaskFillModel(
        hidden_channels=32,
        protein_atom_feature_dim=10,
        ligand_atom_feature_dim=8,
        num_classes=7,
        num_indicators=3,
        encoder_num_filters=16,
        encoder_num_interactions=2,
        encoder_edge_channels=12,
        encoder_cutoff=6.0,
        field_num_filters=16,
        field_k=4,
        field_cutoff=6.0,
    )


def example_input_3dsbdd_ar():
    """A tiny protein-pocket + partial-ligand context, plus a batch of query points (1 graph, 2 batches)."""
    torch.manual_seed(0)
    n_protein, n_ligand, n_query = 6, 4, 5

    protein_pos = torch.randn(n_protein, 3) * 3.0
    protein_atom_feature = torch.rand(n_protein, 10)
    batch_protein = torch.zeros(n_protein, dtype=torch.long)

    ligand_pos = torch.randn(n_ligand, 3) * 1.5
    ligand_atom_feature = torch.rand(n_ligand, 8)
    batch_ligand = torch.zeros(n_ligand, dtype=torch.long)

    pos_query = torch.randn(n_query, 3) * 2.0
    batch_query = torch.zeros(n_query, dtype=torch.long)

    return (
        pos_query,
        protein_pos,
        protein_atom_feature,
        ligand_pos,
        ligand_atom_feature,
        batch_query,
        batch_protein,
        batch_ligand,
    )


MENAGERIE_ENTRIES = [
    ("3D-SBDD AR (MaskFillModel)", build_3dsbdd_ar, example_input_3dsbdd_ar, 2021, "REAL"),
]
