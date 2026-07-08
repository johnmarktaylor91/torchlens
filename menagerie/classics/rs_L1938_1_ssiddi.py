# SOURCE: vendored from kanz76/SSI-DDI @ 23e958f65f59f689edea8d68a38f944c6d18dc96
# https://raw.githubusercontent.com/kanz76/SSI-DDI/master/models.py
# https://raw.githubusercontent.com/kanz76/SSI-DDI/master/layers.py
#
# "SSI-DDI: substructure-substructure interactions for drug-drug interaction prediction"
# (Nyamabo et al., Briefings in Bioinformatics 2021). GAT-based substructure extraction with
# SAGPooling readout per drug, co-attention over per-block substructure representations, and
# a RESCAL bilinear scorer over relation-typed drug pairs. Vendored verbatim from models.py +
# layers.py (only relative imports collapsed into this single file; no architectural change).
# The real training pipeline builds `n_atom_feats` from an RDKit-derived one-hot atom-feature
# vocabulary (data_preprocessing.py); that vocabulary needs data/drug_smiles.csv on disk, so
# the staging harness below supplies a small explicit feature width instead of recomputing it.
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    LayerNorm,
    global_mean_pool,
    max_pool_neighbor_x,
    global_add_pool,
)

MENAGERIE_ZOO = "vendored-pytorch"


# ---- layers.py (verbatim) ----
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver  # noqa: F841 -- unused in the original source (kept verbatim)

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# ---- models.py (verbatim) ----
class SSI_DDI(nn.Module):
    def __init__(
        self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params
    ):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(
                n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim
            )
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)

            h_data = out1[0]
            t_data = out2[0]
            r_h = out1[1]
            r_t = out2[1]

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        return scores


class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(
            data.x, data.edge_index, batch=data.batch
        )
        global_graph_emb = global_add_pool(att_x, att_batch)

        return data, global_graph_emb


# ---- staging harness ----
# Real usage (train_script.py): SSI_DDI(n_atom_feats, n_atom_hid=64, kge_dim=64, rel_total=86,
# heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2]). `n_atom_feats` there is
# `TOTAL_ATOM_FEATS` computed from an RDKit one-hot vocabulary over data/drug_smiles.csv, which
# isn't available in this environment; a small explicit atom-feature width is substituted below
# (architecture unaffected -- it is just the GATConv input width).
_N_ATOM_FEATS = 12


def build_ssiddi():
    torch.manual_seed(0)
    # heads_out_feat_params[i] * blocks_params[i] == 8 for every block, so each block's
    # SAGPooling/global_add_pool readout emits width-8 vectors; kge_dim must match that
    # width since repr_h/repr_t stack those per-block readouts for the co-attention + RESCAL
    # scorer (real defaults: 4 blocks of [32,32,32,32] heads * [2,2,2,2] blocks -> kge_dim=64).
    return SSI_DDI(
        in_features=_N_ATOM_FEATS,
        hidd_dim=8,
        kge_dim=8,
        rel_total=6,
        heads_out_feat_params=[4, 4],
        blocks_params=[2, 2],
    )


def _tiny_mol_batch(n_mols_atoms):
    torch.manual_seed(0)
    graphs = []
    for n in n_mols_atoms:
        x = torch.randn(n, _N_ATOM_FEATS)
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        if not edges:
            edges = [(0, 0)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def example_input_ssiddi():
    torch.manual_seed(0)
    h_data = _tiny_mol_batch([5, 4])
    t_data = _tiny_mol_batch([6, 3])
    rels = torch.tensor([0, 3], dtype=torch.long)
    return ((h_data, t_data, rels),)


MENAGERIE_ENTRIES = [
    ("SSI-DDI", "build_ssiddi", "example_input_ssiddi", 2021, "vendored"),
]
