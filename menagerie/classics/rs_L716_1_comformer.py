# SOURCE: vendored from divelab/AIRS @ main (OpenMat/ComFormer/comformer/models/comformer.py,
#         OpenMat/ComFormer/comformer/models/transformer.py,
#         OpenMat/ComFormer/comformer/models/utils.py)
#
# ComFormer (Yan et al., ICLR 2024, "Complete and Efficient Graph Transformers for Crystal
# Material Property Prediction") is a crystal-structure graph transformer, vendored here as
# both shipped variants: eComformer embeds atoms, expands interatomic-distance edges with an
# RBF basis, alternates ComformerConv message-passing attention layers with a single
# equivariant update layer (ComformerConvEqui, built from e3nn spherical-harmonic
# tensor-product convolutions over irreps of the interatomic-distance vectors), then
# mean-pools per-crystal node features through an MLP readout head. iComformer swaps the
# equivariant update for an invariant edge-update stage (ComformerConv_edge) that folds in
# RBF-expanded lengths/angles of each edge's 3 nearest-neighbor edges, otherwise sharing the
# same ComformerConv attention/readout stack. Vendored verbatim (architecture-relevant
# classes/functions only). Non-architectural changes: (1) `iComformerConfig`/`eComformerConfig`
# used pydantic v1 `BaseSettings`/`pydantic.typing.Literal`, both removed in the installed
# pydantic v2 -- these are hyperparameter containers only (no forward-pass behavior), so they
# are replaced with plain dataclasses carrying the identical field set/defaults; (2) the unused
# `from comformer.features import angle_emb_mp` import (dead import in the original file, not
# referenced by either forward pass) is dropped; (3) `ComformerConv.__init__` in the real repo
# calls `super(MatformerConv, self).__init__(...)` where `MatformerConv` is undefined anywhere
# in the file/package (a leftover from an internal Matformer-fork rename) -- this is a genuine
# NameError in the upstream code that makes ComformerConv (and therefore both eComformer and
# iComformer) uninstantiable as shipped; the trivially-intended `super(ComformerConv,
# self).__init__(...)` is used instead. This is a pure super()-target correction, not an
# architecture change -- every layer/mechanism is unchanged.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairTensor
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# --- comformer/models/utils.py (verbatim) ---
class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = torch.diff(self.centers).mean().item()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


# --- comformer/models/transformer.py (verbatim architecture classes) ---
class ComformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)

        self.lin_msg_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None
        )

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)

        return self.softplus(x[1] + self.bn(out))

    def message(
        self,
        query_i: Tensor,
        key_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        value_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels)
        )
        return out


class TensorProductConvLayer(torch.nn.Module):
    # from Torsional diffusion
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super(TensorProductConvLayer, self).__init__()
        from e3nn import o3

        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, tp.weight_numel),
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out


class ComformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 64,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        irrep_seq = [f"{ns}x0e", f"{ns}x0e + {nv}x1o + {nv}x2e", f"{ns}x0e"]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.skip_linear = nn.Linear(in_channels, out_channels)
        self.sh = "1x0e + 1x1o + 1x2e"
        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual,
        )
        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False,
        )
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(ns)
        self.node_linear_2 = nn.Linear(ns, out_channels)

    def forward(
        self,
        data,
        node_feature: Union[Tensor, PairTensor],
        edge_index,
        edge_feature: Union[Tensor, PairTensor],
        edge_nei_len: OptTensor = None,
    ):
        from e3nn import o3

        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(
            self.sh, edge_vec, normalize=True, normalization="component"
        )
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.node_linear_2(self.softplus(self.bn(node_feature))))
        node_feature += self.skip_linear(skip_connect)

        return node_feature


# --- comformer/models/transformer.py (verbatim ComformerConv_edge, used by iComformer) ---
class ComformerConv_edge(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lemb = nn.Embedding(num_embeddings=3, embedding_dim=32)
        self.embedding_dim = 32
        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge_len = nn.Linear(in_channels[0] + self.embedding_dim, in_channels[0])
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        edge: Union[Tensor, PairTensor],
        edge_nei_len: OptTensor = None,
        edge_nei_angle: OptTensor = None,
    ):
        # preprocess for edge of shape [num_edges, hidden_dim]
        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)

        key_y = torch.cat(
            (
                self.lin_key_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                self.lin_key_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                self.lin_key_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C),
            ),
            dim=1,
        )
        value_y = torch.cat(
            (
                self.lin_value_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                self.lin_value_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                self.lin_value_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C),
            ),
            dim=1,
        )

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        key = self.key_update(torch.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = (query_x * key) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(
                -1, 3, self.heads, self.out_channels
            )
        )

        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)

        return self.softplus(edge[1] + self.bn(out))


# --- comformer/models/comformer.py (verbatim eComformer architecture) ---
# Non-architectural change: pydantic v1 BaseSettings config replaced with a plain dataclass
# carrying the identical field set/defaults (config is a hyperparameter container only).
@dataclass
class eComformerConfig:
    conv_layers: int = 3
    edge_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False
    link: str = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False


class eComformer(nn.Module):  # eComFormer
    """att pyg implementation."""

    def __init__(self, config: eComformerConfig = eComformerConfig()):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(config.atom_input_features, config.node_features)
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(
                    in_channels=config.node_features,
                    out_channels=config.node_features,
                    heads=config.node_layer_head,
                    edge_dim=config.node_features,
                )
                for _ in range(config.conv_layers)
            ]
        )

        self.equi_update = ComformerConvEqui(
            in_channels=config.node_features,
            out_channels=config.node_features,
            edge_dim=config.node_features,
            use_second_order_repr=True,
        )

        self.fc = nn.Sequential(nn.Linear(config.node_features, config.fc_features), nn.SiLU())
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(config.fc_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, _, _ = data
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)

        return torch.squeeze(out)


# Non-architectural change: pydantic v1 BaseSettings config replaced with a plain dataclass
# carrying the identical field set/defaults (config is a hyperparameter container only).
@dataclass
class iComformerConfig:
    conv_layers: int = 4
    edge_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False
    link: str = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1))
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


class iComformer(nn.Module):  # iComFormer
    """att pyg implementation."""

    def __init__(self, config: iComformerConfig = iComformerConfig()):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(config.atom_input_features, config.node_features)
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            nn.Linear(config.triplet_input_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(
                    in_channels=config.node_features,
                    out_channels=config.node_features,
                    heads=config.node_layer_head,
                    edge_dim=config.node_features,
                )
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(
            in_channels=config.node_features,
            out_channels=config.node_features,
            heads=config.node_layer_head,
            edge_dim=config.node_features,
        )

        self.fc = nn.Sequential(nn.Linear(config.node_features, config.fc_features), nn.SiLU())
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(config.fc_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)  # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1)  # [num_edges, 3]
        edge_nei_angle = bond_cosine(
            data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1)
        )  # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)

        return torch.squeeze(out)


def build_comformer():
    torch.manual_seed(0)
    config = eComformerConfig(
        conv_layers=3,
        atom_input_features=16,
        edge_features=16,
        node_features=16,
        fc_features=16,
        output_features=1,
    )
    return eComformer(config)


def build_icomformer():
    torch.manual_seed(0)
    config = iComformerConfig(
        conv_layers=4,
        atom_input_features=16,
        edge_features=16,
        triplet_input_features=16,
        node_features=16,
        fc_features=16,
        output_features=1,
    )
    return iComformer(config)


def example_input_comformer():
    # eComformer.forward expects a 3-tuple `(data, ldata, lattice)` (the real train/eval loop
    # unpacks a PyG DataLoader batch this way -- ldata/lattice are unused by eComformer's
    # forward, only by iComformer). `data` is a single-crystal PyG graph: per-atom feature
    # vectors `x`, an edge index for the k-NN bond graph, per-edge displacement vectors
    # `edge_attr` (3D, consumed as a norm + as raw vectors for the e3nn spherical harmonics),
    # and a `batch` vector assigning every node to its crystal in the batch.
    torch.manual_seed(0)
    n_atoms, n_edges, atom_features = 6, 14, 16
    x = torch.randn(n_atoms, atom_features)
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    edge_attr = torch.randn(n_edges, 3) + 1.0  # nonzero displacement vectors
    batch = torch.zeros(n_atoms, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return ((data, None, None),)


def example_input_icomformer():
    # iComformer.forward expects the same 3-tuple `(data, ldata, lattice)` as eComformer, but
    # its `data` graph additionally carries `edge_nei`: for every edge, the displacement
    # vectors to its 3 nearest-neighbor edges (shape [num_edges, 3, 3]) -- built by
    # `PygGraph.atom_dgl_multigraph` in the real data pipeline (comformer/graphs.py) and
    # consumed for the invariant edge-length/edge-angle features in `edge_update_layer`.
    # ldata/lattice remain unused by iComformer's forward (only by the discarded line-graph
    # angle path).
    torch.manual_seed(0)
    n_atoms, n_edges, atom_features = 6, 14, 16
    x = torch.randn(n_atoms, atom_features)
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    edge_attr = torch.randn(n_edges, 3) + 1.0  # nonzero displacement vectors
    edge_nei = torch.randn(n_edges, 3, 3) + 1.0  # 3 neighbor displacement vectors per edge
    batch = torch.zeros(n_atoms, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_nei=edge_nei, batch=batch)
    return ((data, None, None),)


MENAGERIE_ENTRIES = [
    (
        "ComFormer_eComformer",
        "build_comformer",
        "example_input_comformer",
        2024,
        "vendored-pytorch",
    ),
    (
        "ComFormer_iComformer",
        "build_icomformer",
        "example_input_icomformer",
        2024,
        "vendored-pytorch",
    ),
]
