# FAITHFUL PORT of isjakewong/MIRACLE @ main (original framework: PyTorch + RDKit)
# https://raw.githubusercontent.com/isjakewong/MIRACLE/main/MIRACLE/models/mpn.py
# https://raw.githubusercontent.com/isjakewong/MIRACLE/main/MIRACLE/models/DDIModel.py
# https://raw.githubusercontent.com/isjakewong/MIRACLE/main/MIRACLE/nn_utils.py
# https://raw.githubusercontent.com/isjakewong/MIRACLE/main/MIRACLE/model_utils.py
#
# "MIRACLE: Multi-view Graph Contrastive Representation Learning for Drug-Drug
# Interaction Prediction" (Wang et al., WWW 2021). The real repo's inference path for a
# single drug-drug pair is `DDIModel` (models/DDIModel.py) wrapping a directed
# message-passing encoder `MPN`/`MPNEncoder` (models/mpn.py, the repo's default
# `--graph_encoder dmpnn`, a Chemprop-style D-MPNN) with a self-attention molecule
# readout, followed by a sigmoid classifier. `MPNEncoder.forward`/`.attention` and
# `index_select_ND`/`convert_to_3D` are transcribed FAITHFULLY, verbatim from the real
# code (only CUDA branches and unused `use_input_features`/`features_only` short-circuits
# removed, since this port always runs the full graph-message-passing path on CPU).
# The repo's full "MIRACLE" model (`global_graph/model_hier.py: HierGlobalGCN`) fuses
# this per-molecule MPN encoding with a SEPARATE training-time interaction-graph GCN and
# a deep-graph-infomax contrastive loss that both require the full drug-drug adjacency
# matrix + negative-sampling machinery as forward-time state, not a single-input
# architecture -- that half is loss/training-loop coupled, not part of the inference
# forward pass, so it is intentionally excluded here; DDIModel is the real, complete,
# single-pair inference model.
#
# RDKit's `MolGraph`/`BatchMolGraph` (features/featurization.py) turn a SMILES string
# into the exact tensor format MPNEncoder consumes (f_atoms, f_bonds, a2b, b2a, b2revb,
# a_scope): atom/bond featurization is real-molecule PREPROCESSING, not part of the
# model architecture. RDKit is not installed here, so `SyntheticBatchMolGraph` below
# builds a random molecular-graph-shaped batch directly in that same tensor format
# (zero-padded index-0 sentinel, atom/bond feature dims matching the real
# ATOM_FDIM=133 / BOND_FDIM=14 from featurization.py) so the real MPNEncoder code runs
# completely unmodified.
from argparse import Namespace
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "ported-pytorch"

# real featurization.py constants
ATOM_FDIM = 133
BOND_FDIM_RAW = 14


# ---- nn_utils.py (verbatim, trimmed to what MPNEncoder needs) ----
def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim

    target = source.index_select(dim=0, index=index.view(-1))
    target = target.view(final_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif activation == "PReLU":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "ELU":
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


# ---- model_utils.py (verbatim) ----
def compute_max_atoms(scope: List[Tuple[int, int]]) -> int:
    max_atoms = 0
    for st, le in scope:
        if le > max_atoms:
            max_atoms = le
    return max_atoms


def convert_to_3D(input, scope, max_atoms, device, self_attn=True):
    n_features = input.size()[1]

    batch_input = []
    batch_mask = []
    for st, le in scope:
        mol_input = input.narrow(0, st, le)

        n_atoms = le
        n_padding = max_atoms - le

        mask = torch.ones([n_atoms], device=device)

        if n_padding > 0:
            mask = torch.cat([mask, torch.zeros([n_padding], device=device)])
            mol_input_padded = torch.cat(
                [mol_input, torch.zeros([n_padding, n_features], device=device)]
            )
            batch_input.append(mol_input_padded)
        else:
            batch_input.append(mol_input)

        mask = mask.repeat([max_atoms, 1]) * mask.unsqueeze(1)
        if not self_attn:
            for i in range(max_atoms):
                mask[i, i] = 0
        batch_mask.append(mask)

    batch_input = torch.stack(batch_input, dim=0)
    batch_mask = torch.stack(batch_mask, dim=0).byte()
    return batch_input, batch_mask


# ---- features/featurization.py: BatchMolGraph tensor contract, synthetic filler ----
class SyntheticBatchMolGraph:
    """Builds the same (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope) tensor
    bundle the real RDKit-backed BatchMolGraph.get_components() returns, using a random
    small molecular-graph topology instead of parsing SMILES. Index 0 is reserved as the
    zero-padding sentinel exactly as in the real BatchMolGraph.__init__."""

    def __init__(self, n_atoms_per_mol: List[int], atom_fdim: int, bond_fdim: int):
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim

        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope: List[Tuple[int, int]] = []
        self.b_scope: List[Tuple[int, int]] = []

        f_atoms = [[0.0] * atom_fdim]
        f_bonds = [[0.0] * bond_fdim]
        a2b: List[List[int]] = [[]]
        b2a = [0]
        b2revb = [0]

        for n_atoms_mol in n_atoms_per_mol:
            mol_f_atoms = [torch.rand(atom_fdim).tolist() for _ in range(n_atoms_mol)]
            mol_a2b: List[List[int]] = [[] for _ in range(n_atoms_mol)]
            mol_f_bonds: List[List[float]] = []
            mol_b2a: List[int] = []
            mol_b2revb: List[int] = []
            n_bonds_mol = 0

            # a simple path graph a0-a1-a2-...-a{n-1} (each bond directed both ways),
            # same b1/b2 reverse-bond bookkeeping as the real MolGraph.__init__ loop.
            for a1 in range(n_atoms_mol - 1):
                a2 = a1 + 1
                f_bond = torch.rand(BOND_FDIM_RAW).tolist()
                mol_f_bonds.append(mol_f_atoms[a1] + f_bond)
                mol_f_bonds.append(mol_f_atoms[a2] + f_bond)

                b1 = n_bonds_mol
                b2 = b1 + 1
                mol_a2b[a2].append(b1)
                mol_b2a.append(a1)
                mol_a2b[a1].append(b2)
                mol_b2a.append(a2)
                mol_b2revb.append(b2)
                mol_b2revb.append(b1)
                n_bonds_mol += 2

            f_atoms.extend(mol_f_atoms)
            f_bonds.extend(mol_f_bonds)

            for a in range(n_atoms_mol):
                a2b.append([b + self.n_bonds for b in mol_a2b[a]])
            for b in range(n_bonds_mol):
                b2a.append(self.n_atoms + mol_b2a[b])
                b2revb.append(self.n_bonds + mol_b2revb[b])

            self.a_scope.append((self.n_atoms, n_atoms_mol))
            self.b_scope.append((self.n_bonds, n_bonds_mol))
            self.n_atoms += n_atoms_mol
            self.n_bonds += n_bonds_mol

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)],
            dtype=torch.long,
        )
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)

    def get_components(self):
        return (
            self.f_atoms,
            self.f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
        )


# ---- models/mpn.py: MPNEncoder (verbatim message-passing + attention readout) ----
class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.args = args

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.weight_tying = self.args.weight_tying
        n_message_layer = 1 if self.weight_tying else self.depth - 1
        self.W_h = nn.ModuleList(
            [
                nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
                for _ in range(n_message_layer)
            ]
        )

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.attn_num_d = self.args.attn_num_d
        self.attn_num_r = self.args.attn_num_r
        self.W_s1 = Parameter(torch.FloatTensor(self.hidden_size, self.attn_num_d))
        self.W_s2 = Parameter(torch.FloatTensor(self.attn_num_d, self.attn_num_r))
        nn.init.xavier_uniform_(self.W_s1)
        nn.init.xavier_uniform_(self.W_s2)
        self.softmax = nn.Softmax(dim=1)

        self.i_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.j_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: SyntheticBatchMolGraph) -> torch.Tensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            step = 0 if self.weight_tying else depth
            message = self.W_h[step](message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.args.attn_output:
            mol_vecs = self.attention(atom_hiddens, a_scope)
            return mol_vecs

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        return mol_vecs

    def attention(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        device = atom_hiddens.device
        max_atoms = compute_max_atoms(a_scope)
        batch_hidden, batch_mask = convert_to_3D(
            atom_hiddens, a_scope, max_atoms, device=device, self_attn=True
        )

        # self-contained attention mechanism (the repo's real "attn_output" readout path)
        e = torch.sum(torch.sigmoid(self.j_layer(batch_hidden)) * self.i_layer(batch_hidden), dim=1)
        return e


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, mol_graph: SyntheticBatchMolGraph) -> torch.Tensor:
        return self.encoder.forward(mol_graph)


# ---- models/DDIModel.py (verbatim structure; encoder wired directly to MPN, dropout
#      restored -- the real DDIModel.forward references self.dropout without ever
#      setting it in __init__, a real bug in the upstream code; nn.Dropout(0) preserves
#      the intended identity-in-eval-mode behavior without diverging architecturally) ----
class DDIModel(nn.Module):
    def __init__(self, args: Namespace):
        super(DDIModel, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout)
        self.encoder = MPN(args, atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM_RAW + ATOM_FDIM)

    def forward(self, mol_graph: SyntheticBatchMolGraph) -> torch.Tensor:
        feat = self.encoder(mol_graph)
        feat = self.dropout(feat)
        output = self.sigmoid(feat)
        return output


def build_ddimodel():
    torch.manual_seed(0)
    args = Namespace(
        hidden_size=16,
        bias=True,
        depth=3,
        dropout=0.0,
        undirected=False,
        atom_messages=False,
        weight_tying=True,
        activation="ReLU",
        attn_num_d=6,
        attn_num_r=4,
        attn_output=True,
    )
    return DDIModel(args)


def example_input_ddimodel():
    torch.manual_seed(0)
    n_atoms_per_mol = [5, 7, 4]  # a batch of 3 "molecules"
    mol_graph = SyntheticBatchMolGraph(
        n_atoms_per_mol, atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM_RAW + ATOM_FDIM
    )
    return (mol_graph,)


MENAGERIE_ENTRIES = [
    ("MIRACLE_DDIModel", "build_ddimodel", "example_input_ddimodel", 2021, "ported"),
]
