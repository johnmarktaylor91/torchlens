# SOURCE: vendored from kdmsit/crysgnn @ main (crysgnn/model.py: ConvLayer, CrysGNN)
#
# CrysGNN (Das et al., AAAI 2023, "CrysGNN: Distilling Pre-trained Knowledge to Enhance
# Property Prediction for Crystalline Materials") is a self-supervised pretraining encoder
# for crystal graphs. Atom features are embedded and passed through a stack of CGCNN-style
# gated ConvLayer message-passing blocks (concatenate atom + neighbor + bond features,
# gate with a sigmoid/softplus split, batch-norm, residual-add + softplus). The pretrained
# CrysGNN encoder then reconstructs, per crystal, an atom-adjacency probability (bilinear
# atom-pair scorer + log-softmax), the original atom features, and the crystal's space
# group (log-softmax classifier over 230 groups) as three self-supervised pretext tasks,
# alongside the mean-pooled crystal embedding. Vendored verbatim from `crysgnn/model.py`
# (only the commented-out, unused `CrystalGraphConvNet` class is dropped as dead code; the
# active `ConvLayer`/`CrysGNN` classes exposed by the module are untouched).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(
            N, M, self.atom_fea_len * 2
        )
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrysGNN(nn.Module):
    """
    Create a Deep GNN based Encoder Decoder Model for Crystalline Materials to learn
    representation in an self supervised way.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3):
        super(CrysGNN, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len, bias=False)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )

        self.fc_adj = nn.Bilinear(atom_fea_len, atom_fea_len, 6)
        self.fc1 = nn.Linear(6, 6)

        self.fc_atom_feature = nn.Linear(atom_fea_len, orig_atom_fea_len)

        self.fc_sg = nn.Linear(atom_fea_len, 230)  # 230

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, cuda_flag):
        # Encoder Part (Crystal Graph Convolution Encoder )
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        atom_emb = []

        bt_atom_fea = [atom_fea[idx_map] for idx_map in crystal_atom_idx]

        edge_prob_list = []
        atom_feature_list = []
        sg_pred_list = []
        crys_fea_list = []
        for i in range(len(bt_atom_fea)):
            atom_fea = bt_atom_fea[i]
            atom_fea = F.normalize(atom_fea, dim=1, p=2)
            atom_emb.append(atom_fea)
            z_G = torch.mean(atom_fea, dim=0, keepdim=True)
            crys_fea_list.append(z_G)
            N = atom_fea.shape[0]
            dim = atom_fea.shape[1]

            # Repeat feature N times : (N,N,dim)
            atom_nbr_fea = atom_fea.repeat(N, 1, 1)
            atom_nbr_fea = atom_nbr_fea.contiguous().view(-1, dim)

            # Expand N times : (N,N,dim)
            atom_adj_fea = torch.unsqueeze(atom_fea, 1).expand(N, N, dim)
            atom_adj_fea = atom_adj_fea.contiguous().view(-1, dim)

            # Bilinear Layer : Adjacency List Reconstruction
            edge_p = self.fc_adj(atom_adj_fea, atom_nbr_fea)
            edge_p = self.fc1(edge_p)
            edge_p = F.log_softmax(edge_p, dim=1)
            edge_prob_list.append(edge_p)

            # Atom Feature Reconstruction
            atom_feature_list.append(self.fc_atom_feature(atom_fea))

            # Space group Reconstruct
            sg_pred = F.log_softmax(self.fc_sg(z_G), dim=1)
            sg_pred_list.append(sg_pred)

        atom_feature_list = torch.cat(atom_feature_list, dim=0)
        sg_pred_list = torch.cat(sg_pred_list, dim=0)
        edge_prob_list = torch.cat(edge_prob_list, dim=0)

        crys_fea_list = torch.cat(crys_fea_list, dim=0)
        return edge_prob_list, atom_feature_list, sg_pred_list, crys_fea_list, atom_emb


def build_crysgnn():
    torch.manual_seed(0)
    orig_atom_fea_len = 16
    nbr_fea_len = 8
    return CrysGNN(orig_atom_fea_len, nbr_fea_len, atom_fea_len=16, n_conv=2)


def example_input_crysgnn():
    # Real usage (crysgnn/data.py collate_pool) batches several crystals into flat
    # concatenated tensors: `atom_fea` (total_atoms, orig_atom_fea_len), `nbr_fea`
    # (total_atoms, M, nbr_fea_len) per-atom neighbor bond features, `nbr_fea_idx`
    # (total_atoms, M) LongTensor of neighbor indices into the flat atom_fea tensor, and
    # `crystal_atom_idx` a list of per-crystal LongTensor index maps into the flat batch.
    torch.manual_seed(0)
    orig_atom_fea_len = 16
    nbr_fea_len = 8
    M = 4  # neighbors per atom
    n_atoms_per_crystal = [5, 4]
    total_atoms = sum(n_atoms_per_crystal)

    atom_fea = torch.randn(total_atoms, orig_atom_fea_len)
    nbr_fea = torch.randn(total_atoms, M, nbr_fea_len)
    nbr_fea_idx = torch.randint(0, total_atoms, (total_atoms, M))

    crystal_atom_idx = []
    base = 0
    for n in n_atoms_per_crystal:
        crystal_atom_idx.append(torch.arange(base, base + n, dtype=torch.long))
        base += n

    return (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, False)


MENAGERIE_ENTRIES = [
    ("CrysGNN", "build_crysgnn", "example_input_crysgnn", 2023, "vendored-pytorch"),
]
