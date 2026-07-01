# SOURCE: vendored from wengong-jin/abdockgen @ 7f54d38af2aefda7752e2433e708436b80c9307c
# (ICML 2022, "Antibody-Antigen Docking and Design via Hierarchical Equivariant
# Refinement"). Files vendored verbatim (architecture unmodified): bindgen/data.py
# (ALPHABET/ATOM_TYPES/RES_ATOM14 constants only -- the PDB-parsing dataset class is
# dropped, as it needs Bio.PDB and is irrelevant to the model), bindgen/utils.py,
# bindgen/nnutils.py, bindgen/protein_features.py, bindgen/encoder.py, bindgen/dock.py
# (RefineDocker -- the CDR-loop docking model; the sequence-design decoder in
# generate.py was not vendored as RefineDocker is fully self-contained without it).
#
# Two classes of mechanical, non-architectural fixes were applied on top of the
# verbatim logic:
#   1. Intra-package imports (`from bindgen.xxx import *`) were collapsed since this
#      staging module inlines every file into one namespace.
#   2. Several tensors were constructed with a hardcoded `.cuda()` (the original repo
#      assumed a GPU-only environment). Each was changed to build on the same device
#      as a nearby tensor already in scope (`device=X.device`) instead of forcing CUDA,
#      so the model runs on CPU too. No shapes, dtypes, or numerics were changed.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt  # noqa: F401 (only used by commented-out debug code upstream)

MENAGERIE_ZOO = "vendored-pytorch"

#########################################################################
# --- bindgen/data.py (constants only, verbatim) ---
#########################################################################

RESTYPE_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

ALPHABET = [
    "#",
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
ATOM_TYPES = [
    "",
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
RES_ATOM14 = [
    [""] * 14,
    ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
]


#########################################################################
# --- bindgen/utils.py (verbatim except .cuda() -> device-agnostic) ---
#########################################################################

ReturnType = namedtuple(
    "ReturnType",
    ("loss", "nll", "ppl", "bind_X", "handle"),
    defaults=(None, None, None, None, None),
)


def kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1, 2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1, 2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return A_aligned, R, t


# X: [B, N, 4, 3], R: [B, 3, 3], t: [B, 3]
def rigid_transform(X, R, t):
    B, N, L = X.size(0), X.size(1), X.size(2)
    X = X.reshape(B, N * L, 3)
    X = torch.bmm(R, X.transpose(1, 2)).transpose(1, 2) + t
    return X.view(B, N, L, 3)


# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd(A, B, mask):
    A_aligned, _, _ = kabsch(A, B)
    rmsd = ((A_aligned - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()


# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd_no_align(A, B, mask):
    rmsd = ((A - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()


def eig_coord(X, mask):
    D, mask_2D = self_square_dist(X, torch.ones_like(mask))
    return eig_coord_from_dist(D)


def eig_coord_from_dist(D):
    M = (D[:, :1, :] + D[:, :, :1] - D) / 2
    L, V = torch.linalg.eigh(M)
    L = torch.diag_embed(L)
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, :, -3:].detach()


def inner_square_dist(X, mask):
    L = mask.size(2)
    dX = X.unsqueeze(2) - X.unsqueeze(3)  # [B,N,1,L,3] - [B,N,L,1,3]
    mask_2D = mask.unsqueeze(2) * mask.unsqueeze(3)
    mask_2D = mask_2D * (1 - torch.eye(L)[None, None, :, :]).to(mask_2D)
    D = torch.sum(dX**2, dim=-1)
    return D * mask_2D, mask_2D


def self_square_dist(X, mask):
    X = X[:, :, 1]
    dX = X.unsqueeze(1) - X.unsqueeze(2)  # [B, 1, N, 3] - [B, N, 1, 3]
    D = torch.sum(dX**2, dim=-1)
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, 1, N] x [B, N, 1]
    mask_2D = mask_2D * (1 - torch.eye(mask.size(1))[None, :, :]).to(mask_2D)
    return D, mask_2D


def cross_square_dist(X, Y, xmask, ymask):
    X, Y = X[:, :, 1], Y[:, :, 1]
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, N, 1, 3] - [B, 1, M, 3]
    D = torch.sum(dxy**2, dim=-1)
    mask_2D = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, N, 1] x [B, 1, M]
    return D, mask_2D


def full_square_dist(X, Y, XA, YA, contact=False, remove_diag=False):
    B, N, M, L = X.size(0), X.size(1), Y.size(1), Y.size(2)
    X = X.view(B, N * L, 3)
    Y = Y.view(B, M * L, 3)
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, NL, 1, 3] - [B, 1, ML, 3]
    D = torch.sum(dxy**2, dim=-1)
    D = D.view(B, N, L, M, L)
    D = D.transpose(2, 3).reshape(B, N, M, L * L)

    xmask = XA.clamp(max=1).float().view(B, N * L)
    ymask = YA.clamp(max=1).float().view(B, M * L)
    mask = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, NL, 1] x [B, 1, ML]
    mask = mask.view(B, N, L, M, L)
    mask = mask.transpose(2, 3).reshape(B, N, M, L * L)
    if remove_diag:
        mask = mask * (1 - torch.eye(N)[None, :, :, None]).to(mask)

    if contact:
        D = D + 1e6 * (1 - mask)
        return D.amin(dim=-1), mask.amax(dim=-1)
    else:
        return D, mask


""" Quaternion functions """


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (1e-4 + (quaternions * quaternions).sum(-1))
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(rot):
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


""" Graph functions """


def autoregressive_mask(E_idx):
    N_nodes = E_idx.size(1)
    ii = torch.arange(N_nodes, device=E_idx.device)
    ii = ii.view((1, -1, 1))
    mask = E_idx - ii < 0
    return mask.float()


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


#########################################################################
# --- bindgen/nnutils.py (verbatim except .cuda() -> device-agnostic) ---
#########################################################################


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)

        return gain * (x - mu) / (sigma + self.epsilon) + bias


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.Identity()  # Normalize(num_hidden)
        self.W = nn.Sequential(
            nn.Linear(num_hidden + num_in, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, h_V, h_E, mask_attend):
        # h_V: [B, N, H]; h_E: [B, N, K, H]
        # mask_attend: [B, N, K]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)  # [B, N, K, H]
        h_message = self.W(h_EV) * mask_attend.unsqueeze(-1)
        dh = torch.mean(h_message, dim=-2)
        h_V = self.norm(h_V + self.dropout(dh))
        return h_V


class PosEmbedding(nn.Module):
    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    # E_idx: [B, N]
    def forward(self, E_idx):
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = E_idx.unsqueeze(-1) * frequency.view((1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class AAEmbedding(nn.Module):
    def __init__(self):
        super(AAEmbedding, self).__init__()
        self.hydropathy = {
            "#": 0,
            "I": 4.5,
            "V": 4.2,
            "L": 3.8,
            "F": 2.8,
            "C": 2.5,
            "M": 1.9,
            "A": 1.8,
            "W": -0.9,
            "G": -0.4,
            "T": -0.7,
            "S": -0.8,
            "Y": -1.3,
            "P": -1.6,
            "H": -3.2,
            "N": -3.5,
            "D": -3.5,
            "Q": -3.5,
            "E": -3.5,
            "K": -3.9,
            "R": -4.5,
        }
        self.volume = {
            "#": 0,
            "G": 60.1,
            "A": 88.6,
            "S": 89.0,
            "C": 108.5,
            "D": 111.1,
            "P": 112.7,
            "N": 114.1,
            "T": 116.1,
            "E": 138.4,
            "V": 140.0,
            "Q": 143.8,
            "H": 153.2,
            "M": 162.9,
            "I": 166.7,
            "L": 166.7,
            "K": 168.6,
            "R": 173.4,
            "F": 189.9,
            "Y": 193.6,
            "W": 227.8,
        }
        self.charge = {
            **{"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1},
            **{x: 0 for x in "ABCFGIJLMNOPQSTUVWXYZ#"},
        }
        self.polarity = {**{x: 1 for x in "RNDQEHKSTY"}, **{x: 0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x: 1 for x in "DENQHSTY"}, **{x: 0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x: 1 for x in "RKWNQHSTY"}, **{x: 0 for x in "DEACGILMFPV#"}}
        self.embedding = torch.tensor(
            [
                [
                    self.hydropathy[aa],
                    self.volume[aa] / 100,
                    self.charge[aa],
                    self.polarity[aa],
                    self.acceptor[aa],
                    self.donor[aa],
                ]
                for aa in ALPHABET
            ]
        )

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view(1, 1, -1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-(((D_expand - D_mu) / stride) ** 2))

    def transform(self, aa_vecs):
        return torch.cat(
            [
                self.to_rbf(aa_vecs[:, :, 0], -4.5, 4.5, 0.1),
                self.to_rbf(aa_vecs[:, :, 1], 0, 2.2, 0.1),
                self.to_rbf(aa_vecs[:, :, 2], -1.0, 1.0, 0.25),
                torch.sigmoid(aa_vecs[:, :, 3:] * 6 - 3),
            ],
            dim=-1,
        )

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        B, N = x.size(0), x.size(1)
        aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs

    def soft_forward(self, x):
        B, N = x.size(0), x.size(1)
        aa_vecs = torch.matmul(x.reshape(B * N, -1), self.embedding).view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return rbf_vecs


class ABModel(nn.Module):
    def __init__(self, args):
        super(ABModel, self).__init__()
        self.k_neighbors = args.k_neighbors
        self.hidden_size = args.hidden_size
        self.embedding = AAEmbedding()
        self.features = ProteinFeatures(
            top_k=args.k_neighbors,
            num_rbf=args.num_rbf,
            features_type="full",
            direction="bidirectional",
        )
        self.W_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.huber_loss = nn.SmoothL1Loss(reduction="none")

    def select_target(self, tgt_X, tgt_h, tgt_A, tgt_pos):
        max_len = max([len(pos) for pos in tgt_pos])
        xlist = [tgt_X[i, pos] for i, pos in enumerate(tgt_pos)]
        hlist = [tgt_h[i, pos] for i, pos in enumerate(tgt_pos)]
        alist = [tgt_A[i, pos] for i, pos in enumerate(tgt_pos)]
        tgt_X = [F.pad(x, (0, 0, 0, 0, 0, max_len - len(x))) for x in xlist]
        tgt_h = [F.pad(h, (0, 0, 0, max_len - len(h))) for h in hlist]
        tgt_A = [F.pad(a, (0, 0, 0, max_len - len(a))) for a in alist]
        return torch.stack(tgt_X, dim=0), torch.stack(tgt_h, dim=0), torch.stack(tgt_A, dim=0)


#########################################################################
# --- bindgen/protein_features.py (verbatim except .cuda() -> device-agnostic) ---
#########################################################################


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2, 1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        # i-j
        N_batch = E_idx.size(0)  # noqa: F841 (kept verbatim from upstream source)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)  # noqa: F841 (kept verbatim from upstream source)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        features_type="backbone",
        direction="forward",
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.direction = direction

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            "atom": (0, num_positional_embeddings + num_rbf),
            "backbone": (6, num_positional_embeddings + num_rbf + 7),
        }

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)

    def _dist(self, X, mask, eps=1e-6):
        """Pairwise euclidean distances"""
        N = X.size(1)
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        if self.direction == "bidirectional":
            mask_2D = mask_2D - torch.eye(N, device=X.device).unsqueeze(0)  # remove self
            mask_2D = mask_2D.clamp(min=0)
        elif self.direction == "forward":
            nmask = torch.arange(X.size(1), device=X.device)
            nmask = nmask.view(1, -1, 1) > nmask.view(1, 1, -1)
            mask_2D = nmask.float() * mask_2D  # [B, N, N]
        else:
            raise ValueError("invalid direction", direction)  # noqa: F821 (kept verbatim; unreachable upstream bug -- should be self.direction, never hit since direction is always 'bidirectional' or 'forward' in this codebase)

        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (not including self)
        D_adjust = D + (1.0 - mask_2D) * 10000
        top_k = min(self.top_k, N)
        D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        # Debug plot KNN
        # print(E_idx[:10,:10])
        # D_simple = mask_2D * torch.zeros(D.size()).scatter(-1, E_idx, torch.ones_like(knn_D))
        # print(D_simple)
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # D_simple = D.data.numpy()[0,:,:]
        # plt.imshow(D_simple, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('D_knn.pdf')
        # exit(0)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _quaternions(self, R):
        """Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1))
        )
        _R = lambda i, j: R[:, :, :, i, j]  # noqa: E731 (kept verbatim from upstream source)
        signs = torch.sign(
            torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
        )
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        """Contacts"""
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1e-3):
        """Hydrogen bonds and contact map"""
        X_atoms = dict(zip(["N", "CA", "C", "O"], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms["C_prev"] = F.pad(X_atoms["C"][:, 1:, :], (0, 0, 0, 1), "constant", 0)
        X_atoms["H"] = X_atoms["N"] + F.normalize(
            F.normalize(X_atoms["N"] - X_atoms["C_prev"], -1)
            + F.normalize(X_atoms["N"] - X_atoms["CA"], -1),
            -1,
        )

        def _distance(X_a, X_b):
            return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1.0 / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
            _inv_distance(X_atoms["O"], X_atoms["N"])
            + _inv_distance(X_atoms["C"], X_atoms["H"])
            - _inv_distance(X_atoms["O"], X_atoms["H"])
            - _inv_distance(X_atoms["C"], X_atoms["N"])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    def _AD_features(self, X, eps=1e-6):
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack(
            (torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2
        )
        return F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)  # noqa: F841 (kept verbatim from upstream source)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)  # noqa: E741 (kept verbatim from upstream source)
        O = O.view(list(O.shape[:2]) + [9])  # noqa: E741
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)  # noqa: E741

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])  # noqa: E741
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)
        return torch.cat((dU, Q), dim=-1)

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        D = F.pad(D, (3, 0), "constant", 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, mask):
        """Featurize coordinates as an attributed graph"""
        if self.features_type == "backbone":
            X_ca = X[:, :, 1, :]
            D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)
            RBF = self._rbf(D_neighbors)
            E_positional = self.embeddings(E_idx)
            O_features = self._orientations_coarse(X_ca, E_idx)
            E = torch.cat((E_positional, RBF, O_features), -1)
            V = self._dihedrals(X)

        elif self.features_type == "atom":
            D_neighbors, E_idx, mask_neighbors = self._dist(X, mask)
            RBF = self._rbf(D_neighbors)
            E_positional = self.embeddings(E_idx)
            E = torch.cat((E_positional, RBF), -1)
            V = None

        return V, E, E_idx


#########################################################################
# --- bindgen/encoder.py (verbatim) ---
#########################################################################


class EGNNEncoder(nn.Module):
    def __init__(self, args, node_hdim=0, features_type="backbone", update_X=True):
        super(EGNNEncoder, self).__init__()
        self.update_X = update_X
        self.features_type = features_type
        self.features = ProteinFeatures(
            top_k=args.k_neighbors,
            num_rbf=args.num_rbf,
            features_type=features_type,
            direction="bidirectional",
        )
        self.node_in, self.edge_in = self.features.feature_dimensions[features_type]
        self.node_in += node_hdim

        self.W_v = nn.Linear(self.node_in, args.hidden_size)
        self.W_e = nn.Linear(self.edge_in, args.hidden_size)
        self.layers = nn.ModuleList(
            [
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
            ]
        )
        if self.update_X:
            self.W_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 14))

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # [backbone] X: [B,N,L,3], V/S: [B,N,H], A: [B,N,L]
    # [atom] X: [B,N*L,3], V/S: [B,N*L,H], A: [B,N*L]
    def forward(self, X, V, S, A):
        mask = A.clamp(max=1).float()
        vmask = mask[:, :, 1] if self.features_type == "backbone" else mask
        _, E, E_idx = self.features(X, vmask)

        h = self.W_v(V)  # [B, N, H]
        h_e = self.W_e(E)  # [B, N, K, H]
        nei_s = gather_nodes(S, E_idx)  # [B, N, K, H]
        emask = gather_nodes(vmask[..., None], E_idx).squeeze(-1)

        # message passing
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=emask)  # [B, N, H]
            h = h * vmask.unsqueeze(-1)  # [B, N, H]

        if self.update_X and self.features_type == "backbone":
            ca_mask = mask[:, :, 1]  # [B, N]
            mij = self.W_x(h).unsqueeze(2) + self.U_x(h).unsqueeze(1)  # [B,N,N,H]
            xij = X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N,L,3]
            xij = xij * self.T_x(mij).unsqueeze(-1)  # [B,N,N,L,3]
            f = torch.sum(xij * ca_mask[:, None, :, None, None], dim=2)  # [B,N,N,L,3] * [B,1,N,1,1]
            f = f / (1e-6 + ca_mask.sum(dim=1)[:, None, None, None])  # [B,N,L,3] / [B,1,1,1]
            X = X + f.clamp(min=-20.0, max=20.0)

        return h, X * mask[..., None]


class HierEGNNEncoder(nn.Module):
    def __init__(self, args, update_X=True, backbone_CA_only=True):
        super(HierEGNNEncoder, self).__init__()
        self.update_X = update_X
        self.backbone_CA_only = backbone_CA_only
        self.clash_step = args.clash_step
        self.residue_mpn = EGNNEncoder(
            args,
            features_type="backbone",
            node_hdim=args.hidden_size,
            update_X=False,
        )
        self.atom_mpn = EGNNEncoder(
            args,
            features_type="atom",
            node_hdim=args.hidden_size,
            update_X=False,
        )
        if self.update_X:
            # backbone coord update
            self.W_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_x = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 4))
            # side chain coord update
            self.W_a = nn.Linear(args.hidden_size, args.hidden_size)
            self.U_a = nn.Linear(args.hidden_size, args.hidden_size)
            self.T_a = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 1))

        self.embedding = nn.Embedding(len(ATOM_TYPES), args.hidden_size)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # X: [B,N,L,3], V: [B,N,6], S: [B,N,H], A: [B,N,L]
    def forward(self, X, V, S, A):
        B, N, L = X.size()[:3]
        X_atom = X.view(B, N * L, 3)
        mask = A.clamp(max=1).float()

        # atom message passing
        h_atom = self.embedding(A).view(B, N * L, -1)
        h_atom, _ = self.atom_mpn(X_atom, h_atom, h_atom, A.view(B, -1))
        h_atom = h_atom.view(B, N, L, -1)
        h_atom = h_atom * mask[..., None]
        h_A = h_atom.sum(dim=-2) / (1e-6 + mask.sum(dim=-1)[..., None])

        # residue message passing
        h_V = torch.cat([V, h_A], dim=-1)
        h_res, _ = self.residue_mpn(X, h_V, S, A)

        if self.update_X:
            # backbone update
            bb_mask = mask[:, :, :4]  # [B, N, 4]
            X_bb = X[:, :, :4]  # backbone atoms
            mij = self.W_x(h_res).unsqueeze(2) + self.U_x(h_res).unsqueeze(1)  # [B,N,N,H]
            xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
            dij = xij.norm(dim=-1)  # [B,N,N,4]
            fij = torch.maximum(self.T_x(mij), 3.8 - dij)  # break term [B,N,N,4]
            xij = xij * fij.unsqueeze(-1)
            f_res = torch.sum(
                xij * bb_mask[:, None, :, :, None], dim=2
            )  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
            f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[..., None])  # [B,N,4,3]
            X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # Clash correction
            for _ in range(self.clash_step):
                xij = X_bb.unsqueeze(2) - X_bb.unsqueeze(1)  # [B,N,N,4,3]
                dij = xij.norm(dim=-1)  # [B,N,N,4]
                fij = F.relu(3.8 - dij)  # repulsion term [B,N,N,4]
                xij = xij * fij.unsqueeze(-1)
                f_res = torch.sum(
                    xij * bb_mask[:, None, :, :, None], dim=2
                )  # [B,N,N,4,3] * [B,1,N,4,1] -> [B,N,4,3]
                f_res = f_res / (1e-6 + bb_mask.sum(dim=1, keepdims=True)[..., None])  # [B,N,4,3]
                X_bb = X_bb + f_res.clamp(min=-20.0, max=20.0)

            # side chain update
            mij = self.W_a(h_atom).unsqueeze(3) + self.U_a(h_atom).unsqueeze(
                2
            )  # [B,N,L,1,H] + [B,N,1,L,H]
            xij = X.unsqueeze(3) - X.unsqueeze(2)  # [B,N,L,1,3] - [B,N,1,L,3]
            dij = xij.norm(dim=-1)  # [B,N,L,L]
            fij = torch.maximum(self.T_a(mij).squeeze(-1), 1.5 - dij)  # break term [B,N,L,L]
            xij = xij * fij.unsqueeze(-1)  # [B,N,L,L,3]
            f_atom = torch.sum(
                xij * mask[:, :, None, :, None], dim=3
            )  # [B,N,L,L,3] * [B,N,1,L,1] -> [B,N,L,3]
            X_sc = X + 0.1 * f_atom

            if self.backbone_CA_only:
                X = torch.cat((X_sc[:, :, :1], X_bb[:, :, 1:2], X_sc[:, :, 2:]), dim=2)
            else:
                X = torch.cat((X_bb[:, :, :4], X_sc[:, :, 4:]), dim=2)

        return h_res, X * mask[..., None]


#########################################################################
# --- bindgen/dock.py (verbatim) ---
#########################################################################


class RefineDocker(ABModel):
    def __init__(self, args):
        super(RefineDocker, self).__init__(args)
        self.rstep = args.rstep
        self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.target_mpn = EGNNEncoder(args, update_X=False)
        self.hierarchical = args.hierarchical
        if args.hierarchical:
            self.struct_mpn = HierEGNNEncoder(args)
        else:
            self.struct_mpn = EGNNEncoder(args)

        self.W_x0 = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.U_x0 = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def struct_loss(self, bind_X, tgt_X, true_V, true_R, true_D, inter_D, true_C):
        # dihedral loss
        bind_V = self.features._dihedrals(bind_X)
        vloss = self.mse_loss(bind_V, true_V).sum(dim=-1)
        # local loss
        rdist = bind_X.unsqueeze(2) - bind_X.unsqueeze(3)
        rdist = torch.sum(rdist**2, dim=-1)
        rloss = self.huber_loss(rdist, true_R) + 10 * F.relu(1.5 - rdist)
        # full loss
        cdist, _ = full_square_dist(
            bind_X, tgt_X, torch.ones_like(bind_X)[..., 0], torch.ones_like(tgt_X)[..., 0]
        )
        closs = self.huber_loss(cdist, true_C) + 10 * F.relu(1.5 - cdist)
        # alpha carbon
        bind_X, tgt_X = bind_X[:, :, 1], tgt_X[:, :, 1]
        # CDR self distance
        dist = bind_X.unsqueeze(1) - bind_X.unsqueeze(2)
        dist = torch.sum(dist**2, dim=-1)
        dloss = self.huber_loss(dist, true_D) + 10 * F.relu(14.4 - dist)
        # inter distance
        idist = bind_X.unsqueeze(2) - tgt_X.unsqueeze(1)
        idist = torch.sum(idist**2, dim=-1)
        iloss = self.huber_loss(idist, inter_D) + 10 * F.relu(14.4 - idist)
        return dloss, vloss, rloss, iloss, closs

    def forward(self, binder, target, surface):
        true_X, true_S, true_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_surface, tgt_surface = surface

        # Encode target
        tgt_S = self.embedding(tgt_S)
        tgt_V = self.features._dihedrals(tgt_X)
        tgt_h, _ = self.target_mpn(tgt_X, tgt_V, self.U_i(tgt_S), tgt_A)
        _, tgt_S, _ = self.select_target(tgt_X, tgt_S, tgt_A, tgt_surface)
        tgt_X, tgt_h, tgt_A = self.select_target(tgt_X, tgt_h, tgt_A, tgt_surface)
        tgt_V = self.features._dihedrals(tgt_X)

        B, N, M = true_S.size(0), true_S.size(1), tgt_X.size(1)  # noqa: F841 (B, M kept verbatim from upstream source)
        true_mask = true_A[:, :, 1].clamp(max=1).float()
        tgt_mask = tgt_A[:, :, 1].clamp(max=1).float()

        tgt_mean = (tgt_X[:, :, 1] * tgt_mask[..., None]).sum(dim=1) / tgt_mask[..., None].sum(
            dim=1
        ).clamp(min=1e-4)
        bind_X = tgt_mean[:, None, None, :] + torch.rand_like(true_X)
        init_loss = 0

        # Refine
        dloss = vloss = rloss = iloss = closs = 0
        for t in range(self.rstep):
            # Interpolated label
            ratio = (t + 1) / self.rstep
            label_X = true_X * ratio + bind_X.detach() * (1 - ratio)
            true_V = self.features._dihedrals(label_X)
            true_R, rmask_2D = inner_square_dist(label_X, true_A.clamp(max=1).float())
            true_D, mask_2D = self_square_dist(label_X, true_mask)
            true_C, cmask_2D = full_square_dist(label_X, tgt_X, true_A, tgt_A)
            inter_D, imask_2D = cross_square_dist(label_X, tgt_X, true_mask, tgt_mask)

            bind_V = self.features._dihedrals(bind_X)
            V = torch.cat([bind_V, tgt_V], dim=1).detach()
            X = torch.cat([bind_X, tgt_X], dim=1).detach()
            A = torch.cat([true_A, tgt_A], dim=1).detach()
            S = torch.cat([self.embedding(true_S), tgt_S], dim=1).detach()

            h_S = self.W_i(S)
            h, X = self.struct_mpn(X, V, h_S, A)
            bind_X = X[:, :N]

            dloss_t, vloss_t, rloss_t, iloss_t, closs_t = self.struct_loss(
                bind_X, tgt_X, true_V, true_R, true_D, inter_D, true_C
            )
            vloss = vloss + vloss_t * true_mask
            dloss = dloss + dloss_t * mask_2D
            iloss = iloss + iloss_t * imask_2D
            rloss = rloss + rloss_t * rmask_2D
            closs = closs + closs_t * cmask_2D

        dloss = torch.sum(dloss) / mask_2D.sum()
        iloss = torch.sum(iloss) / imask_2D.sum()
        vloss = torch.sum(vloss) / true_mask.sum()
        if self.hierarchical:
            rloss = torch.sum(rloss) / rmask_2D.sum()
        else:
            rloss = torch.sum(rloss[:, :, :4, :4]) / rmask_2D[:, :, :4, :4].sum()

        loss = init_loss + dloss + iloss + vloss + rloss
        return ReturnType(loss=loss, bind_X=bind_X.detach(), handle=(tgt_X, tgt_A))


#########################################################################
# --- staging harness ---
#########################################################################


def _hern_args():
    """Tiny hyperparameter namespace matching dock_train.py's argparse defaults,
    scaled down for a fast trace."""
    import argparse

    return argparse.Namespace(
        hierarchical=False,
        hidden_size=16,
        k_neighbors=4,
        num_rbf=4,
        dropout=0.0,
        depth=1,
        rstep=2,
        clash_step=0,
    )


def build_hern():
    """Tiny RefineDocker (HERN docking model) for tracing."""
    torch.manual_seed(0)
    return RefineDocker(_hern_args())


def example_input_hern():
    """Builds a small synthetic antibody(binder)-antigen(target) docking batch.

    Shapes follow bindgen/data.py's `completize`/`make_batch` convention:
    X: [B, N, 14, 3] (atom14 coordinates), S: [B, N] (residue-type indices into
    ALPHABET), A: [B, N, 14] (atom-type indices into ATOM_TYPES; 0 = absent atom).
    """
    torch.manual_seed(0)
    B, N_bind, N_tgt = 1, 6, 8

    def make_chain(n_res):
        X = torch.randn(B, n_res, 14, 3)
        S = torch.randint(1, len(ALPHABET), (B, n_res))
        # First 4 backbone atoms (N, CA, C, O) + CB present for every residue;
        # remaining side-chain slots absent (atom index 0), matching RES_ATOM14 shape.
        A = torch.zeros(B, n_res, 14, dtype=torch.long)
        A[:, :, 0] = ATOM_TYPES.index("N")
        A[:, :, 1] = ATOM_TYPES.index("CA")
        A[:, :, 2] = ATOM_TYPES.index("C")
        A[:, :, 3] = ATOM_TYPES.index("O")
        A[:, :, 4] = ATOM_TYPES.index("CB")
        return X, S, A

    true_X, true_S, true_A = make_chain(N_bind)
    tgt_X, tgt_S, tgt_A = make_chain(N_tgt)

    binder = (true_X, true_S, true_A, None)
    target = (tgt_X, tgt_S, tgt_A, None)
    # `surface` is a pair of per-batch-element index lists selecting which target
    # residues are near the binding interface; here we just use all target residues.
    bind_surface = [list(range(N_bind))]
    tgt_surface = [list(range(N_tgt))]
    surface = (bind_surface, tgt_surface)

    return (binder, target, surface)


MENAGERIE_ENTRIES = [
    ("HERN", "build_hern", "example_input_hern", 2022, "vendored-pytorch"),
]
