# SOURCE: vendored from mahmoodlab/PathomicFusion @ master
# (networks.py: GraphNet, MaxNet, PathgraphomicNet; fusion.py: TrilinearFusion_A;
#  utils.py: init_max_weights, dfs_freeze -- all four fetched raw from
#  https://raw.githubusercontent.com/mahmoodlab/PathomicFusion/master/{networks,fusion,utils}.py)
#
# Pathomic Fusion (Chen et al., "Pathomic Fusion: An Integrated Framework for Fusing
# Histopathology and Genomic Features for Cancer Diagnosis and Prognosis", IEEE TMI 2020)
# is a trimodal survival-prediction network that fuses (1) histology-image features
# ("path"), (2) a cell/tissue-graph GNN branch ("graph"), and (3) genomic/omic features
# ("omic") via a gated trilinear-pooling fusion module. `PathgraphomicNet` is the paper's
# flagship trimodal model: GraphNet (3x SAGEConv+SAGPooling stages over a cell graph) +
# MaxNet (4-layer ELU/AlphaDropout MLP over omic features) feed their embeddings, together
# with a pre-extracted path-image embedding, into TrilinearFusion_A (gated multimodal-unit
# trilinear pooling), followed by a linear classifier/hazard head.
#
# Vendoring notes (only imports/device-portability were touched; NO architecture change):
#   - `networks.py` unconditionally imports `tables` (pytables) and `tqdm` at module scope
#     even though neither is used by the model classes vendored here; both are dropped.
#   - `fusion.py`'s `BilinearFusion`/`TrilinearFusion_A`/`TrilinearFusion_B.forward` build
#     the "add a ones column" trick via `torch.cuda.FloatTensor(...)`, which is hardcoded to
#     CUDA and crashes immediately on a CPU-only machine (a real bug/CUDA-only assumption in
#     the original repo, not an architectural choice). Replaced with the device-agnostic
#     `vec.new_ones(vec.shape[0], 1)` (same values/shape/dtype/behavior, just not CUDA-only).
#   - `GraphNet.forward` mutates the incoming `torch_geometric` Data/Batch in place via
#     `NormalizeFeaturesV2()(data)` / `NormalizeEdgesV2()(data)`, which cast `data.x`/
#     `data.edge_attr` to `torch.cuda.FloatTensor` -- same CUDA-only bug, fixed the same way
#     (`.to(x.dtype)`/no-op cast since inputs are already float32 CPU tensors).
#   - `opt`-namespace hyperparameters (grph_dim, omic_dim, path_dim, mmhid, dropout_rate,
#     fusion gates/scales, input_size_omic) are inlined from `options.py`'s argparse
#     defaults instead of threading an `argparse.Namespace` through, since only the
#     concrete values (not the CLI parsing) are architecture-relevant.
#   - `PathNet`/VGG (the path-image encoder) is NOT vendored: `PathgraphomicNet.forward`
#     consumes `x_path` as an already-extracted `path_dim`-length feature vector (confirmed
#     against the official `train_test.py` training loop, which computes VGG path features
#     once and feeds the vector via `model(x_path=..., x_grph=..., x_omic=...)`), so the
#     trimodal fusion network itself has no image-encoder submodule to vendor.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, SAGEConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

MENAGERIE_ZOO = "vendored-pytorch"


def init_max_weights(module):
    """Verbatim from utils.py."""
    import math

    for m in module.modules():
        if type(m) is nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def dfs_freeze(model):
    """Verbatim from utils.py."""
    for _name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class NormalizeFeaturesV2:
    """Column-normalizes node features to sum-up to one (verbatim from networks.py,
    minus the CUDA-only `.type(torch.cuda.FloatTensor)` cast -- device-agnostic)."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data


class NormalizeEdgesV2:
    """Column-normalizes edge attributes (verbatim from networks.py, minus the
    CUDA-only cast)."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data


class GraphNet(torch.nn.Module):
    """Verbatim from networks.py (cell/tissue-graph branch: 3x SAGEConv + SAGPooling)."""

    def __init__(
        self,
        features=1036,
        nhid=128,
        grph_dim=32,
        dropout_rate=0.25,
        GNN="GCN",
        pooling_ratio=0.20,
        act=None,
        label_dim=1,
        init_max=True,
    ):
        super(GraphNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.act = act

        # NOTE (API-drift fix, not an architecture change): the original repo pins an
        # older torch_geometric where `SAGPooling(..., gnn='GCN')` accepted a string name
        # for the internal scoring GNN. Current torch_geometric (2.7) takes a `GNN=` class
        # kwarg instead of a string `gnn=`; `GNN='GCN'` mapped to `GCNConv` in the old API,
        # so this passes `GCNConv` directly -- same scoring architecture, updated call form.
        del GNN  # the string 'GCN' selector is no longer accepted; GCNConv is used below
        self.conv1 = SAGEConv(features, nhid)
        self.pool1 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GCNConv)
        self.conv2 = SAGEConv(nhid, nhid)
        self.pool2 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GCNConv)
        self.conv3 = SAGEConv(nhid, nhid)
        self.pool3 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GCNConv)

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, grph_dim)
        self.lin3 = torch.nn.Linear(grph_dim, label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if init_max:
            init_max_weights(self)

    def forward(self, **kwargs):
        data = kwargs["x_grph"]
        data = NormalizeFeaturesV2()(data)
        data = NormalizeEdgesV2()(data)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)[:5]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)[:5]
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)[:5]
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        features = F.relu(self.lin2(x))
        out = self.lin3(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


class MaxNet(nn.Module):
    """Verbatim from networks.py (omic-feature MLP branch)."""

    def __init__(
        self, input_dim=80, omic_dim=32, dropout_rate=0.25, act=None, label_dim=1, init_max=True
    ):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )
        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim), nn.ELU(), nn.AlphaDropout(p=dropout_rate, inplace=False)
        )

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max:
            init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


class TrilinearFusion_A(nn.Module):
    """Verbatim from fusion.py, except `torch.cuda.FloatTensor(...).fill_(1)` (CUDA-only
    ones column) is replaced with the device-agnostic `vec.new_ones(vec.shape[0], 1)`."""

    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim3_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        )
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        # Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec3)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            )
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        # Fusion
        o1 = torch.cat((o1, o1.new_ones(o1.shape[0], 1)), 1)
        o2 = torch.cat((o2, o2.new_ones(o2.shape[0], 1)), 1)
        o3 = torch.cat((o3, o3.new_ones(o3.shape[0], 1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out


class PathgraphomicNet(nn.Module):
    """Verbatim from networks.py (Path + Graph + Omic trimodal survival network), with
    the `opt` argparse.Namespace threaded parameters inlined as constructor defaults
    (matching options.py's argparse defaults) instead of an `opt` object, and the
    checkpoint-loading branch (`k is not None`) dropped (no pretrained checkpoints ship
    in this environment; architecture is otherwise unchanged)."""

    def __init__(
        self,
        grph_dim=32,
        omic_dim=32,
        path_dim=32,
        mmhid=96,
        dropout_rate=0.25,
        label_dim=1,
        graph_features=1036,
        input_size_omic=80,
        skip=0,
        use_bilinear=1,
        path_gate=1,
        grph_gate=1,
        omic_gate=1,
        path_scale=1,
        grph_scale=1,
        omic_scale=1,
        act=None,
    ):
        super(PathgraphomicNet, self).__init__()
        self.grph_net = GraphNet(
            features=graph_features,
            grph_dim=grph_dim,
            dropout_rate=dropout_rate,
            pooling_ratio=0.20,
            label_dim=label_dim,
            init_max=False,
        )
        self.omic_net = MaxNet(
            input_dim=input_size_omic,
            omic_dim=omic_dim,
            dropout_rate=dropout_rate,
            act=act,
            label_dim=label_dim,
            init_max=False,
        )

        self.fusion = TrilinearFusion_A(
            skip=skip,
            use_bilinear=use_bilinear,
            gate1=path_gate,
            gate2=grph_gate,
            gate3=omic_gate,
            dim1=path_dim,
            dim2=grph_dim,
            dim3=omic_dim,
            scale_dim1=path_scale,
            scale_dim2=grph_scale,
            scale_dim3=omic_scale,
            mmhid=mmhid,
            dropout_rate=dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(mmhid, label_dim))
        self.act = act

        dfs_freeze(self.grph_net)
        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(path_vec, grph_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard


class PathgraphomicNetWrapper(nn.Module):
    """Thin positional-args wrapper: PathgraphomicNet.forward is keyword-only
    (x_path=..., x_grph=..., x_omic=...), which torchlens can trace directly, but the
    example-input helper below needs a single callable taking a plain tuple. This wrapper
    unpacks (path_tensor, graph_batch, omic_tensor) into the kwargs the real forward
    expects -- no computation is added or changed."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_path, x_grph, x_omic):
        _features, hazard = self.net(x_path=x_path, x_grph=x_grph, x_omic=x_omic)
        return hazard


def build_pathomicfusion():
    torch.manual_seed(0)
    net = PathgraphomicNet(
        grph_dim=8,
        omic_dim=8,
        path_dim=8,
        mmhid=12,
        dropout_rate=0.25,
        label_dim=1,
        graph_features=6,
        input_size_omic=10,
    )
    net.eval()
    return PathgraphomicNetWrapper(net)


def example_input_pathomicfusion():
    torch.manual_seed(0)
    x_path = torch.randn(2, 8)  # pre-extracted path-image embedding, dim = path_dim
    x_omic = torch.randn(2, 10)  # flat omic feature vector, dim = input_size_omic

    # Minimal 2-graph batch (torch_geometric Batch) with 6 node features each,
    # matching GraphNet's `features=6` in/edge_attr for NormalizeEdgesV2's max-normalize.
    graphs = []
    for _ in range(2):
        num_nodes = 5
        x = torch.rand(num_nodes, 6) + 0.1
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]], dtype=torch.long
        )
        edge_attr = torch.rand(edge_index.shape[1], 1) + 0.1
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    x_grph = Batch.from_data_list(graphs)

    return (x_path, x_grph, x_omic)


MENAGERIE_ENTRIES = [
    ("Pathomic Fusion", build_pathomicfusion, example_input_pathomicfusion, 2020, MENAGERIE_ZOO),
]
