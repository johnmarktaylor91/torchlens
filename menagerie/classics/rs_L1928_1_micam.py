# FAITHFUL PORT of MIRALab-USTC/AI4Sci-MiCaM @ main (original framework: PyTorch + torch_geometric)
# https://raw.githubusercontent.com/MIRALab-USTC/AI4Sci-MiCaM/main/src/model/nn.py
# https://raw.githubusercontent.com/MIRALab-USTC/AI4Sci-MiCaM/main/src/model/encoder.py
#
# "Learn to Extend Molecular Scaffolds with Motifs" (MiCaM), Geng et al., ICLR 2023.
# The full MiCaM VAE (src/model/MiCaM_VAE.py) trains an autoregressive motif-vocabulary
# decoder that requires a pretrained, on-disk motif-vocab graph batch (torch.load of
# model_params.vocab_processed_path) plus rdkit/guacamol/tensorboardX/networkx-coupled
# preprocessing (mol_graph.py, vocab.py, mydataclass.py) that isn't a base lib here and
# isn't a single forward-pass architecture -- it is data-dependent generation state. The
# GNN ENCODER (GIN_virtual + Atom_Embedding + Encoder, from model/nn.py + model/encoder.py)
# is the real, self-contained MiCaM graph-encoding architecture (message-passing GIN with a
# virtual "super node" summarizing the whole graph at every depth, GINEConv edge-aware
# convolutions, residual connections, mean/add pooling readout) and is transcribed here
# VERBATIM from the real repo code (only import paths trimmed to what's vendored below;
# no architectural change). `ATOM_FEATURES` sizes are given directly at construction time
# instead of being read from the repo's rdkit-derived global vocab tables.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

MENAGERIE_ZOO = "ported-pytorch"


# ---- model/nn.py (verbatim) ----
class MLP(nn.Module):
    def __init__(
        self,
        in_channels=None,
        hidden_channels=None,
        out_channels=None,
        num_layers=None,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = act

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = nn.BatchNorm1d(hidden_channels)
            else:
                norm = nn.Identity()
            self.norms.append(norm)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"


class GIN_virtual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        edge_dim,
        depth,
        dropout,
        virtual: bool = True,
        pooling: str = "mean",
        residual=True,
    ):
        super(GIN_virtual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_channels
        self.edge_dim = edge_dim
        self.depth = depth
        self.dropout = dropout
        self.residual = residual
        self.virtual = virtual
        self.pooling = pooling

        self.in_layer = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=3,
            act=nn.ReLU(inplace=True),
            dropout=dropout,
        )

        self.vitual_embed = nn.Embedding(1, hidden_channels)
        nn.init.constant_(self.vitual_embed.weight.data, 0)

        for i in range(depth):
            layer_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            layer = GINEConv(nn=layer_nn, train_eps=True, edge_dim=edge_dim)

            self.add_module(f"GNN_layer_{i}", layer)

            virtual_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            self.add_module(f"virtual_layer_{i}", virtual_layer)

        self.out_layer = MLP(
            in_channels=in_channels + hidden_channels * (depth + 1),
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            act=nn.ReLU(inplace=True),
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if self.virtual:
            v = torch.zeros(batch[-1].item() + 1, dtype=torch.long).to(x.device)
            virtual_embed = self.vitual_embed(v)

        x_list = [x, self.in_layer(x)]

        for i in range(self.depth):
            GNN_layer = getattr(self, f"GNN_layer_{i}")
            if self.virtual:
                virtual_layer = getattr(self, f"virtual_layer_{i}")
                x_list[-1] = x_list[-1] + virtual_embed[batch]

            x = GNN_layer(x_list[-1], edge_index, edge_attr)

            if self.residual:
                x = x + x_list[-1]

            x_list.append(x)

            if self.virtual:
                virtual_tmp = virtual_layer(global_add_pool(x_list[-1], batch) + virtual_embed)
                virtual_embed = virtual_embed + virtual_tmp if self.residual else virtual_tmp

        join_vecs = torch.cat(x_list, -1)
        nodes_reps = self.out_layer(join_vecs)

        if self.pooling == "mean":
            graph_reps = global_mean_pool(nodes_reps, batch)
        else:
            assert self.pooling == "add"
            graph_reps = global_add_pool(nodes_reps, batch)

        return nodes_reps, graph_reps


# ---- model/encoder.py (verbatim; ATOM_FEATURES sizes supplied explicitly) ----
class Atom_Embedding(nn.Module):
    def __init__(self, atom_embed_size, atom_feature_sizes):
        super().__init__()
        assert len(atom_embed_size) == len(atom_feature_sizes)
        self._n_features = len(atom_feature_sizes)
        for i in range(self._n_features):
            f_embed = nn.Embedding(atom_feature_sizes[i], atom_embed_size[i])
            self.add_module(f"f_embed_{i}", f_embed)

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        features = torch.split(atom_features, 1, dim=-1)
        features = [f.long().view([-1]) for f in features]
        return torch.cat([getattr(self, f"f_embed_{i}")(f) for i, f in enumerate(features)], dim=-1)

    def reset_parameters(self) -> None:
        for i in range(self._n_features):
            nn.init.xavier_normal_(getattr(self, f"f_embed_{i}").weight)


class Encoder(nn.Module):
    def __init__(self, atom_embedding, edge_embedding, GNN):
        super(Encoder, self).__init__()

        self.atom_embedding = atom_embedding
        self.edge_embedding = edge_embedding
        self.GNN = GNN

        self.hidden_size = GNN.out_channels

    def embed_graph(self, x, edge_attr):
        x = self.atom_embedding(x)
        edge_attr = self.edge_embedding(edge_attr.long().view([-1]))
        return x, edge_attr

    def forward(self, data: Data):
        x, edge_attr = self.embed_graph(data.x, data.edge_attr)
        nodes_reps, graph_reps = self.GNN(x, data.edge_index, edge_attr, data.batch)
        return nodes_reps, graph_reps


# ---- staging harness ----
# Real MiCaM defaults (src/arguments.py): atom_embed_size=[24]*5 -> sum=120 ("ATOM_FEATURES"
# has 5 vocabularies: symbol, aromatic, formal-charge, num-explicit-Hs, num-implicit-Hs),
# edge_embed_size=24, hidden_size=256, depth=6 (encoder), pooling="mean", virtual=True.
# Sizes are shrunk here for a fast trace-sized instance; the architecture is unchanged.
_ATOM_FEATURE_SIZES = [13, 2, 6, 5, 5]  # real vocab sizes from model/mol_graph.py Vocab lists


def build_micam_encoder():
    torch.manual_seed(0)
    atom_embed_size = [4, 2, 3, 3, 3]  # sums to 15
    edge_embed_size = 6
    hidden_size = 16
    depth = 2

    atom_embedding = nn.Sequential(
        Atom_Embedding(atom_embed_size, _ATOM_FEATURE_SIZES),
        nn.Dropout(0.0),
    )
    edge_embedding = nn.Sequential(
        nn.Embedding(4, edge_embed_size),
        nn.Dropout(0.0),
    )
    gnn = GIN_virtual(
        in_channels=sum(atom_embed_size),
        out_channels=hidden_size,
        hidden_channels=hidden_size,
        edge_dim=edge_embed_size,
        depth=depth,
        dropout=0.0,
        virtual=True,
        pooling="mean",
    )
    return Encoder(atom_embedding=atom_embedding, edge_embedding=edge_embedding, GNN=gnn)


def example_input_micam_encoder():
    torch.manual_seed(0)
    n_atoms_per_mol = [5, 7]
    n_atoms = sum(n_atoms_per_mol)

    x = torch.stack(
        [
            torch.randint(0, _ATOM_FEATURE_SIZES[0], (n_atoms,)),
            torch.randint(0, _ATOM_FEATURE_SIZES[1], (n_atoms,)),
            torch.randint(0, _ATOM_FEATURE_SIZES[2], (n_atoms,)),
            torch.randint(0, _ATOM_FEATURE_SIZES[3], (n_atoms,)),
            torch.randint(0, _ATOM_FEATURE_SIZES[4], (n_atoms,)),
        ],
        dim=-1,
    ).float()

    edges = []
    offset = 0
    for n in n_atoms_per_mol:
        for i in range(n - 1):
            edges.append((offset + i, offset + i + 1))
            edges.append((offset + i + 1, offset + i))
        offset += n
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.randint(0, 4, (edge_index.size(1),)).float().unsqueeze(-1)

    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_atoms_per_mol)]
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return (data,)


MENAGERIE_ENTRIES = [
    ("MiCaM_Encoder", "build_micam_encoder", "example_input_micam_encoder", 2023, "ported"),
]
