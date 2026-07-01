# SOURCE: vendored from ykiiiiii/GraDe_IF @ main
# Files: diffusion/gradeif.py (EGNN_NET only), diffusion/model/egnn_pytorch/egnn_pytorch.py,
#        diffusion/model/egnn_pytorch/egnn_pytorch_geometric.py (EGNN_Sparse only),
#        diffusion/model/egnn_pytorch/utils.py
# GraDe-IF: discrete diffusion model for protein inverse folding (graph-based). EGNN_NET is the
# real trainable denoiser network wrapped by GraDe_IF (the diffusion process itself is loss/
# noise-schedule scaffolding around this network, not a distinct architecture); we vendor and
# trace EGNN_NET directly, called on the underlying (node feats, edge feats, edge_index, batch)
# tensors of a torch_geometric graph plus a per-graph diffusion-timestep tensor, exactly like the
# real forward() call `self.model(noise_data, t)` in GraDe_IF.forward/diffusion_loss. Vendored
# verbatim aside from merging three files into one (relative imports flattened; unused
# GraDe_IF/Trainer/argparse/dataset scaffolding dropped -- non-NN scaffolding, not the
# architecture) and one torch_geometric API-compat shim: Inspector.distribute was renamed to
# Inspector.collect_param_data in torch_geometric>=2.5; EGNN_Sparse.propagate() (an override of
# MessagePassing.propagate needed for its custom coors/feats update interleaving) now falls back
# to the new name when the old one is absent, alongside the __check_input__/__collect__ shim the
# upstream code already carried for the same reason.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from einops import rearrange
from torch import einsum
from torch_geometric.nn import MessagePassing

"""============================================================================================="""
""" model/egnn_pytorch/egnn_pytorch.py (helpers used by EGNN_Sparse) """
"""============================================================================================="""


def exists(val):
    return val is not None


class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


"""============================================================================================="""
""" model/egnn_pytorch/utils.py """
"""============================================================================================="""


def get_node_feature_dims():
    """
    each node has 26 dim feature corrsponding to residual type, sasa, bfactor,dihedral, mu_r_norm

    update Apr 6th:
    remove bfactor as there is no bfactor in predicted structure
    """
    return [20, 1, 1, 4, 5]


def get_edge_feature_dims():
    """
    each node has 93 dim feature corrsponding to one hot sequence distance, interatomic distance, local frame orientation
    """
    return [65, 1, 15, 12]


class nodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_num=4):
        super().__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        if feature_num == 4:
            self.node_feature_dim = get_node_feature_dims()
        else:
            self.node_feature_dim = [20, 4, 5]

        for i, dim in enumerate(self.node_feature_dim):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.node_feature_dim)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count : feature_dim_count + self.node_feature_dim[i]]
            )
            feature_dim_count += self.node_feature_dim[i]
        return x_embedding


class edgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.edge_feature_dims = get_edge_feature_dims()
        for i, dim in enumerate(self.edge_feature_dims):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.edge_feature_dims)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count : feature_dim_count + self.edge_feature_dims[i]]
            )
            feature_dim_count += self.edge_feature_dims[i]
        return x_embedding


"""============================================================================================="""
""" model/egnn_pytorch/egnn_pytorch_geometric.py (EGNN_Sparse only) """
"""============================================================================================="""


class EGNN_Sparse(MessagePassing):
    """Different from the above since it separates the edge assignment
    from the computation (this allows for great reduction in time and
    computations when the graph is locally or sparse connected).
    * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        fourier_features=0,
        soft_edge=0,
        norm_feats=False,
        norm_coors=False,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_edge=False,
        update_coors=False,
        dropout=0.0,
        coor_weights_clamp_value=None,
        aggr="add",
        mlp_num=2,
        **kwargs,
    ):
        assert aggr in {"add", "sum", "max", "mean"}, "pool method must be a valid option"
        assert update_feats or update_coors, "you must update either features, coordinates, or both"
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.update_edge = update_edge
        self.coor_weights_clamp_value = None
        self.mlp_num = mlp_num
        self.edge_input_dim = edge_attr_dim
        self.message_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        if self.mlp_num > 2:
            self.edge_mlp = (
                nn.Sequential(
                    nn.Linear(self.edge_input_dim, self.edge_input_dim * 8),
                    self.dropout,
                    SiLU(),
                    nn.Linear(self.edge_input_dim * 8, self.edge_input_dim * 4),
                    self.dropout,
                    SiLU(),
                    nn.Linear(self.edge_input_dim * 4, self.edge_input_dim * 2),
                    self.dropout,
                    SiLU(),
                    nn.Linear(self.edge_input_dim * 2, m_dim),
                    SiLU(),
                )
                if update_feats
                else None
            )
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
                self.dropout,
                SiLU(),
                nn.Linear(self.edge_input_dim * 2, self.edge_input_dim),
                SiLU(),
            )
        self.message_mlp = nn.Sequential(
            nn.Linear(self.message_input_dim, self.message_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.message_input_dim * 2, m_dim),
            SiLU(),
        )
        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.edge_norm = (
            torch_geometric.nn.norm.LayerNorm(self.edge_input_dim) if self.update_edge else None
        )
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        if self.mlp_num > 2:
            self.node_mlp = (
                nn.Sequential(
                    nn.Linear(feats_dim + m_dim, feats_dim * 8),
                    self.dropout,
                    SiLU(),
                    nn.Linear(feats_dim * 8, feats_dim * 4),
                    self.dropout,
                    SiLU(),
                    nn.Linear(feats_dim * 4, feats_dim * 2),
                    self.dropout,
                    SiLU(),
                    nn.Linear(feats_dim * 2, feats_dim),
                )
                if update_feats
                else None
            )
        else:
            self.node_mlp = (
                nn.Sequential(
                    nn.Linear(feats_dim + m_dim, feats_dim * 2),
                    self.dropout,
                    SiLU(),
                    nn.Linear(feats_dim * 2, feats_dim),
                )
                if update_feats
                else None
            )

        # COORS
        self.coors_mlp = (
            nn.Sequential(
                nn.Linear(m_dim, m_dim * 4), self.dropout, SiLU(), nn.Linear(self.m_dim * 4, 1)
            )
            if update_coors
            else None
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None, angle_data=None, size=None):
        """Inputs:
        * x: (n_points, d) where d is pos_dims + feat_dims
        * edge_index: (2, n_edges)
        * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
        * batch: (n_points,) long tensor. specifies xloud belonging for each point
        * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
        * size: None
        """
        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        if self.update_edge:
            edge_batch = batch[edge_index[0]]
            edge_attr_feats = self.edge_mlp(edge_attr)
            edge_attr = self.edge_norm(self.dropout(edge_attr_feats) + edge_attr, edge_batch)

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            batch=batch,
        )
        if self.update_edge:
            return torch.cat([coors_out, hidden_out], dim=-1), edge_attr
        else:
            return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr):
        m_ij = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index, size=None, **kwargs):
        """The initial call to start propagating messages.
        Args:
        `edge_index` holds the indices of a general (sparse)
            assignment matrix of shape :obj:`[N, M]`.
        size (tuple, optional) if none, the size will be inferred
            and assumed to be quadratic.
        **kwargs: Any additional data which is needed to construct and
            aggregate messages, and to update node embeddings.
        """
        try:
            size = self.__check_input__(edge_index, size)
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        except AttributeError:
            size = self._check_input(edge_index, size)
            coll_dict = self._collect(self._user_args, edge_index, size, kwargs)

        # torch_geometric>=2.5 renamed Inspector.distribute -> Inspector.collect_param_data;
        # fall back for older installs (mirrors the __check_input__/__collect__ version-drift
        # shim already present in the upstream propagate() override above).
        _distribute = (
            getattr(self.inspector, "distribute", None) or self.inspector.collect_param_data
        )
        msg_kwargs = _distribute("message", coll_dict)
        aggr_kwargs = _distribute("aggregate", coll_dict)
        update_kwargs = _distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                pass

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = (
                self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            )
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out

        else:
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


"""============================================================================================="""
""" diffusion/gradeif.py (EGNN_NET only) """
"""============================================================================================="""


class EGNN_NET(torch.nn.Module):
    def __init__(
        self,
        input_feat_dim,
        hidden_channels,
        edge_attr_dim,
        dropout=0.0,
        n_layers=1,
        output_dim=20,
        embedding=False,
        embedding_dim=64,
        mlp_num=2,
        update_edge=True,
        embed_ss=-1,
        norm_feat=False,
    ):
        super().__init__()
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()

        self.embedding = embedding
        self.embed_ss = embed_ss
        self.n_layers = n_layers
        if embedding:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, embedding_dim)
            )

            self.ss_mlp = nn.Sequential(
                nn.Linear(8, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, embedding_dim)
            )
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, input_feat_dim)
            )

            self.ss_mlp = nn.Sequential(
                nn.Linear(8, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, input_feat_dim)
            )

        for i in range(n_layers):
            if embedding:
                layer = EGNN_Sparse(
                    embedding_dim,
                    m_dim=hidden_channels,
                    edge_attr_dim=embedding_dim,
                    dropout=dropout,
                    mlp_num=mlp_num,
                    update_edge=self.update_edge,
                    norm_feats=norm_feat,
                )
            else:
                layer = EGNN_Sparse(
                    input_feat_dim,
                    m_dim=hidden_channels,
                    edge_attr_dim=edge_attr_dim,
                    dropout=dropout,
                    mlp_num=mlp_num,
                    update_edge=self.update_edge,
                    norm_feats=norm_feat,
                )
            self.mpnn_layes.append(layer)

            if embedding:
                time_mlp_layer = nn.Sequential(
                    nn.SiLU(), nn.Linear(embedding_dim, (embedding_dim) * 2)
                )
                ff_layer = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Dropout(p=dropout),
                    nn.SiLU(),
                    torch_geometric.nn.norm.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                )
            else:
                time_mlp_layer = nn.Sequential(
                    nn.SiLU(), nn.Linear(input_feat_dim, (input_feat_dim) * 2)
                )
                ff_layer = nn.Sequential(
                    nn.Linear(input_feat_dim, input_feat_dim),
                    nn.Dropout(p=dropout),
                    nn.SiLU(),
                    torch_geometric.nn.norm.LayerNorm(input_feat_dim),
                    nn.Linear(input_feat_dim, input_feat_dim),
                )

            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)

        if embedding:
            self.node_embedding = nodeEncoder(embedding_dim)
            self.edge_embedding = edgeEncoder(embedding_dim)
            self.lin = nn.Linear(embedding_dim, output_dim)
        else:
            self.lin = nn.Linear(input_feat_dim, output_dim)

    def forward(self, data, time):
        # data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone, e.g. mu_r_norm, sasa, b factor and so on

        x, pos, extra_x, edge_index, edge_attr, ss, batch = (
            data.x,
            data.pos,
            data.extra_x,
            data.edge_index,
            data.edge_attr,
            data.ss,
            data.batch,
        )

        t = self.time_mlp(time)
        ss_embed = self.ss_mlp(ss)

        x = torch.cat([x, extra_x], dim=1)
        if self.embedding:
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)

        x = torch.cat([pos, x], dim=1)

        for i, layer in enumerate(self.mpnn_layes):
            if self.embed_ss == -2 and i == self.n_layers - 1:
                corr, feats = x[:, 0:3], x[:, 3:]
                feats = feats + ss_embed  # [N,hidden_dim]+[N,hidden_dim]
                x = torch.cat([corr, feats], dim=-1)

            if self.update_edge:
                h, edge_attr = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]
            else:
                h = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]

            corr, feats = h[:, 0:3], h[:, 3:]
            time_emb = self.time_mlp_list[i](t)  # [B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2, dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats * (scale + 1) + shift

            feats = self.ff_list[i](feats)

            x = torch.cat([corr, feats], dim=-1)

        corr, x = x[:, 0:3], x[:, 3:]

        if self.embed_ss == -1:
            x = x + ss_embed

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_gradeif():
    torch.manual_seed(0)
    return EGNN_NET(
        input_feat_dim=31,
        hidden_channels=16,
        edge_attr_dim=93,
        dropout=0.0,
        n_layers=1,
        output_dim=20,
        embedding=False,
        update_edge=True,
        embed_ss=-1,
        norm_feat=False,
    )


def example_input_gradeif():
    torch.manual_seed(0)
    n_nodes, n_edges, batch_size = 12, 30, 2
    data = torch_geometric.data.Data(
        x=torch.randn(n_nodes, 20),
        extra_x=torch.randn(n_nodes, 11),
        pos=torch.randn(n_nodes, 3),
        edge_index=torch.randint(0, n_nodes, (2, n_edges)),
        edge_attr=torch.randn(n_edges, 93),
        ss=torch.randn(n_nodes, 8),
        batch=torch.cat(
            [
                torch.zeros(n_nodes // 2, dtype=torch.long),
                torch.ones(n_nodes - n_nodes // 2, dtype=torch.long),
            ]
        ),
    )
    time = torch.rand(batch_size, 1)
    return (data, time)


MENAGERIE_ENTRIES = [
    ("GraDe-IF", "build_gradeif", "example_input_gradeif", 2023, "SOURCE_AVAILABLE"),
]
