# SOURCE: vendored from lishuya17/MONN @ master
# https://raw.githubusercontent.com/lishuya17/MONN/master/src/CPI_model.py
#
# Tsubaki-style GNN + Weisfeiler-Lehman relabelling compound encoder, CNN protein-sequence
# encoder, pairwise-interaction attention module, and a DMA-GRU affinity-prediction module --
# "MONN: a Multi-Objective Neural Network for Predicting Pairwise Non-Covalent Interactions
# and Binding Affinities between Compounds and Proteins" (Li et al., 2020). `Net` is copied
# verbatim from src/CPI_model.py (only the training-time star-import `from pdbbind_utils
# import *` is dropped -- CPI_model.Net's forward pass never calls anything from that module,
# it is only used by the separate training script -- and two Python2/old-torch relics are
# fixed for Py3/torch2: `(kernel_size-1)/2` -> `(kernel_size-1)//2` for Conv1d padding
# [true division on an even kernel_size raised TypeError under Python 3's int-typed padding
# arg], and the removed `F.tanh` -> `torch.tanh`).
import torch
from torch import nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

elem_list = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
    "unknown",
]
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


class Net(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, init_word_features, params):
        super(Net, self).__init__()

        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features
        self.init_word_features = init_word_features
        """hyper part"""
        GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2 = (
            params
        )
        self.GNN_depth = GNN_depth
        self.inner_CNN_depth = inner_CNN_depth
        self.DMA_depth = DMA_depth
        self.k_head = k_head
        self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        """GraphConv Module"""
        self.vertex_embedding = nn.Linear(
            atom_fdim, self.hidden_size1
        )  # first transform vertex features into hidden representations

        # GWM parameters
        self.W_a_main = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]
                )
                for i in range(self.GNN_depth)
            ]
        )
        self.W_a_super = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]
                )
                for i in range(self.GNN_depth)
            ]
        )
        self.W_main = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]
                )
                for i in range(self.GNN_depth)
            ]
        )
        self.W_bmm = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(self.hidden_size1, 1) for i in range(self.k_head)])
                for i in range(self.GNN_depth)
            ]
        )

        self.W_super = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )
        self.W_main_to_super = nn.ModuleList(
            [
                nn.Linear(self.hidden_size1 * self.k_head, self.hidden_size1)
                for i in range(self.GNN_depth)
            ]
        )
        self.W_super_to_main = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )

        self.W_zm1 = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )
        self.W_zm2 = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )
        self.W_zs1 = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )
        self.W_zs2 = nn.ModuleList(
            [nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)]
        )
        self.GRU_main = nn.GRUCell(self.hidden_size1, self.hidden_size1)
        self.GRU_super = nn.GRUCell(self.hidden_size1, self.hidden_size1)

        # WLN parameters
        self.label_U2 = nn.ModuleList(
            [
                nn.Linear(self.hidden_size1 + bond_fdim, self.hidden_size1)
                for i in range(self.GNN_depth)
            ]
        )  # assume no edge feature transformation
        self.label_U1 = nn.ModuleList(
            [nn.Linear(self.hidden_size1 * 2, self.hidden_size1) for i in range(self.GNN_depth)]
        )

        """CNN-RNN Module"""
        # CNN parameters
        self.embed_seq = nn.Embedding(len(self.init_word_features), 20, padding_idx=0)
        self.embed_seq.weight = nn.Parameter(self.init_word_features)
        self.embed_seq.weight.requires_grad = False

        self.conv_first = nn.Conv1d(
            20, self.hidden_size1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2
        )
        self.conv_last = nn.Conv1d(
            self.hidden_size1,
            self.hidden_size1,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )

        self.plain_CNN = nn.ModuleList([])
        for i in range(self.inner_CNN_depth):
            self.plain_CNN.append(
                nn.Conv1d(
                    self.hidden_size1,
                    self.hidden_size1,
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) // 2,
                )
            )

        """Affinity Prediction Module"""
        self.super_final = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.c_final = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.p_final = nn.Linear(self.hidden_size1, self.hidden_size2)

        # DMA parameters
        self.mc0 = nn.Linear(hidden_size2, hidden_size2)
        self.mp0 = nn.Linear(hidden_size2, hidden_size2)

        self.mc1 = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )
        self.mp1 = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )

        self.hc0 = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )
        self.hp0 = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )
        self.hc1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])

        self.c_to_p_transform = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )
        self.p_to_c_transform = nn.ModuleList(
            [nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)]
        )

        self.GRU_dma = nn.GRUCell(self.hidden_size2, self.hidden_size2)
        # Output layer
        self.W_out = nn.Linear(self.hidden_size2 * self.hidden_size2 * 2, 1)

        """Pairwise Interaction Prediction Module"""
        self.pairwise_compound = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.pairwise_protein = nn.Linear(self.hidden_size1, self.hidden_size1)

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def GraphConv_module(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
        n_vertex = vertex_mask.size(1)

        # initial features
        vertex_initial = torch.index_select(self.init_atom_features, 0, vertex.view(-1))
        vertex_initial = vertex_initial.view(batch_size, -1, atom_fdim)
        edge_initial = torch.index_select(self.init_bond_features, 0, edge.view(-1))
        edge_initial = edge_initial.view(batch_size, -1, bond_fdim)

        vertex_feature = F.leaky_relu(self.vertex_embedding(vertex_initial), 0.1)
        super_feature = torch.sum(
            vertex_feature * vertex_mask.view(batch_size, -1, 1), dim=1, keepdim=True
        )

        for GWM_iter in range(self.GNN_depth):
            # prepare main node features
            for k in range(self.k_head):
                a_main = torch.tanh(self.W_a_main[GWM_iter][k](vertex_feature))
                a_super = torch.tanh(self.W_a_super[GWM_iter][k](super_feature))  # noqa: F841 (unused in real repo; kept verbatim -- registers W_a_super params but its output is never consumed)
                a = self.W_bmm[GWM_iter][k](a_main * super_feature)
                attn = self.mask_softmax(a.view(batch_size, -1), vertex_mask).view(
                    batch_size, -1, 1
                )
                k_main_to_super = torch.bmm(
                    attn.transpose(1, 2), self.W_main[GWM_iter][k](vertex_feature)
                )
                if k == 0:
                    m_main_to_super = k_main_to_super
                else:
                    m_main_to_super = torch.cat(
                        [m_main_to_super, k_main_to_super], dim=-1
                    )  # concat k-head
            main_to_super = torch.tanh(self.W_main_to_super[GWM_iter](m_main_to_super))
            main_self = self.wln_unit(
                batch_size,
                vertex_mask,
                vertex_feature,
                edge_initial,
                atom_adj,
                bond_adj,
                nbs_mask,
                GWM_iter,
            )

            super_to_main = torch.tanh(self.W_super_to_main[GWM_iter](super_feature))
            super_self = torch.tanh(self.W_super[GWM_iter](super_feature))
            # warp gate and GRU for update main node features, use main_self and super_to_main
            z_main = torch.sigmoid(
                self.W_zm1[GWM_iter](main_self) + self.W_zm2[GWM_iter](super_to_main)
            )
            hidden_main = (1 - z_main) * main_self + z_main * super_to_main
            vertex_feature = self.GRU_main(
                hidden_main.view(-1, self.hidden_size1), vertex_feature.view(-1, self.hidden_size1)
            )
            vertex_feature = vertex_feature.view(batch_size, n_vertex, self.hidden_size1)
            # warp gate and GRU for update super node features
            z_supper = torch.sigmoid(
                self.W_zs1[GWM_iter](super_self) + self.W_zs2[GWM_iter](main_to_super)
            )
            hidden_super = (1 - z_supper) * super_self + z_supper * main_to_super
            super_feature = self.GRU_super(
                hidden_super.view(batch_size, self.hidden_size1),
                super_feature.view(batch_size, self.hidden_size1),
            )
            super_feature = super_feature.view(batch_size, 1, self.hidden_size1)

        return vertex_feature, super_feature

    def wln_unit(
        self,
        batch_size,
        vertex_mask,
        vertex_features,
        edge_initial,
        atom_adj,
        bond_adj,
        nbs_mask,
        GNN_iter,
    ):
        n_vertex = vertex_mask.size(1)
        n_nbs = nbs_mask.size(2)

        vertex_mask = vertex_mask.view(batch_size, n_vertex, 1)
        nbs_mask = nbs_mask.view(batch_size, n_vertex, n_nbs, 1)

        vertex_nei = torch.index_select(
            vertex_features.view(-1, self.hidden_size1), 0, atom_adj
        ).view(batch_size, n_vertex, n_nbs, self.hidden_size1)
        edge_nei = torch.index_select(edge_initial.view(-1, bond_fdim), 0, bond_adj).view(
            batch_size, n_vertex, n_nbs, bond_fdim
        )

        # Weisfeiler Lehman relabelling
        l_nei = torch.cat((vertex_nei, edge_nei), -1)
        nei_label = F.leaky_relu(self.label_U2[GNN_iter](l_nei), 0.1)
        nei_label = torch.sum(nei_label * nbs_mask, dim=-2)
        new_label = torch.cat((vertex_features, nei_label), 2)
        new_label = self.label_U1[GNN_iter](new_label)
        vertex_features = F.leaky_relu(new_label, 0.1)

        return vertex_features

    def CNN_module(self, batch_size, seq_mask, sequence):
        ebd = self.embed_seq(sequence)
        ebd = ebd.transpose(1, 2)
        x = F.leaky_relu(self.conv_first(ebd), 0.1)

        for i in range(self.inner_CNN_depth):
            x = self.plain_CNN[i](x)
            x = F.leaky_relu(x, 0.1)

        x = F.leaky_relu(self.conv_last(x), 0.1)
        H = x.transpose(1, 2)

        return H

    def Pairwise_pred_module(self, batch_size, comp_feature, prot_feature, vertex_mask, seq_mask):
        pairwise_c_feature = F.leaky_relu(self.pairwise_compound(comp_feature), 0.1)
        pairwise_p_feature = F.leaky_relu(self.pairwise_protein(prot_feature), 0.1)
        pairwise_pred = torch.sigmoid(
            torch.matmul(pairwise_c_feature, pairwise_p_feature.transpose(1, 2))
        )
        pairwise_mask = torch.matmul(
            vertex_mask.view(batch_size, -1, 1), seq_mask.view(batch_size, 1, -1)
        )
        pairwise_pred = pairwise_pred * pairwise_mask

        return pairwise_pred

    def Affinity_pred_module(
        self,
        batch_size,
        comp_feature,
        prot_feature,
        super_feature,
        vertex_mask,
        seq_mask,
        pairwise_pred,
    ):
        comp_feature = F.leaky_relu(self.c_final(comp_feature), 0.1)
        prot_feature = F.leaky_relu(self.p_final(prot_feature), 0.1)
        super_feature = F.leaky_relu(self.super_final(super_feature.view(batch_size, -1)), 0.1)

        cf, pf = self.dma_gru(
            batch_size, comp_feature, vertex_mask, prot_feature, seq_mask, pairwise_pred
        )

        cf = torch.cat([cf.view(batch_size, -1), super_feature.view(batch_size, -1)], dim=1)
        kroneck = F.leaky_relu(
            torch.matmul(cf.view(batch_size, -1, 1), pf.view(batch_size, 1, -1)).view(
                batch_size, -1
            ),
            0.1,
        )

        affinity_pred = self.W_out(kroneck)
        return affinity_pred

    def dma_gru(self, batch_size, comp_feats, vertex_mask, prot_feats, seq_mask, pairwise_pred):
        vertex_mask = vertex_mask.view(batch_size, -1, 1)
        seq_mask = seq_mask.view(batch_size, -1, 1)

        c0 = torch.sum(comp_feats * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        p0 = torch.sum(prot_feats * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        m = c0 * p0
        for DMA_iter in range(self.DMA_depth):
            c_to_p = torch.matmul(
                pairwise_pred.transpose(1, 2),
                torch.tanh(self.c_to_p_transform[DMA_iter](comp_feats)),
            )  # batch * n_residue * hidden
            p_to_c = torch.matmul(
                pairwise_pred, torch.tanh(self.p_to_c_transform[DMA_iter](prot_feats))
            )  # batch * n_vertex * hidden

            c_tmp = (
                torch.tanh(self.hc0[DMA_iter](comp_feats))
                * torch.tanh(self.mc1[DMA_iter](m)).view(batch_size, 1, -1)
                * p_to_c
            )
            p_tmp = (
                torch.tanh(self.hp0[DMA_iter](prot_feats))
                * torch.tanh(self.mp1[DMA_iter](m)).view(batch_size, 1, -1)
                * c_to_p
            )

            c_att = self.mask_softmax(
                self.hc1[DMA_iter](c_tmp).view(batch_size, -1), vertex_mask.view(batch_size, -1)
            )
            p_att = self.mask_softmax(
                self.hp1[DMA_iter](p_tmp).view(batch_size, -1), seq_mask.view(batch_size, -1)
            )

            cf = torch.sum(comp_feats * c_att.view(batch_size, -1, 1), dim=1)
            pf = torch.sum(prot_feats * p_att.view(batch_size, -1, 1), dim=1)

            m = self.GRU_dma(m, cf * pf)

        return cf, pf

    def forward(self, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence):
        batch_size = vertex.size(0)

        atom_feature, super_feature = self.GraphConv_module(
            batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask
        )
        prot_feature = self.CNN_module(batch_size, seq_mask, sequence)

        pairwise_pred = self.Pairwise_pred_module(
            batch_size, atom_feature, prot_feature, vertex_mask, seq_mask
        )
        affinity_pred = self.Affinity_pred_module(
            batch_size,
            atom_feature,
            prot_feature,
            super_feature,
            vertex_mask,
            seq_mask,
            pairwise_pred,
        )

        return affinity_pred, pairwise_pred


def build_monn():
    torch.manual_seed(0)
    n_atom_types = 8
    n_bond_types = 5
    vocab_size = 26  # protein sequence vocabulary (residue tokens + padding)
    init_atom_features = torch.randn(n_atom_types, atom_fdim)
    init_bond_features = torch.randn(n_bond_types, bond_fdim)
    init_word_features = torch.randn(vocab_size, 20)
    params = (
        2,
        2,
        2,
        2,
        5,
        16,
        12,
    )  # GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2
    return Net(init_atom_features, init_bond_features, init_word_features, params)


def example_input_monn():
    torch.manual_seed(0)
    batch_size = 2
    n_vertex = 6
    n_nbs = max_nb
    seq_len = 10
    n_atom_types = 8
    n_bond_types = 5
    vocab_size = 26

    vertex_mask = torch.ones(batch_size, n_vertex)
    vertex = torch.randint(0, n_atom_types, (batch_size, n_vertex))
    edge = torch.randint(0, n_bond_types, (batch_size, n_vertex * n_nbs))
    atom_adj = torch.randint(0, batch_size * n_vertex, (batch_size * n_vertex * n_nbs,))
    bond_adj = torch.randint(0, batch_size * n_vertex * n_nbs, (batch_size * n_vertex * n_nbs,))
    nbs_mask = torch.ones(batch_size, n_vertex, n_nbs)
    seq_mask = torch.ones(batch_size, seq_len)
    sequence = torch.randint(0, vocab_size, (batch_size, seq_len))

    return (vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)


MENAGERIE_ENTRIES = [
    ("MONN", "build_monn", "example_input_monn", 2020, "vendored"),
]
