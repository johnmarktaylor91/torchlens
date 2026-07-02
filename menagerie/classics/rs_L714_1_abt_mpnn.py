# SOURCE: vendored from https://github.com/LCY02/ABT-MPNN @ master (repo has no tags;
# HEAD as of the fetch date)
#
# ABT-MPNN (Chen, Zeng, Ye, Wang, Zhang. 2023, "Atom-Bond Transformer-based Message-Passing
# Neural Network for Molecular Property Prediction") -- a chemprop fork that adds a
# bond-level self-attention block (Transformer or Fastformer) inside each message-passing
# step, and an atom-level self-attention readout block that can bias attention with
# adjacency / distance / Coulomb matrices. Vendored verbatim (architecture-relevant classes
# only) from the repo's own files:
#   https://raw.githubusercontent.com/LCY02/ABT-MPNN/master/chemprop/models/model.py
#   https://raw.githubusercontent.com/LCY02/ABT-MPNN/master/chemprop/models/mpn.py
#   https://raw.githubusercontent.com/LCY02/ABT-MPNN/master/chemprop/models/attention.py
#   https://raw.githubusercontent.com/LCY02/ABT-MPNN/master/chemprop/features/featurization.py
#   https://raw.githubusercontent.com/LCY02/ABT-MPNN/master/chemprop/nn_utils.py
#
# What is kept: MoleculeModel (encoder + FFN head), MPN/MPNEncoder (the directed
# bond-message-passing loop with `atom_messages=False`), MultiBondAttention (the real
# Transformer-style bond self-attention block used inside each message-passing step when
# `bond_attention=True`), MultiAtomAttention (the real atom-level readout self-attention
# block with adjacency/distance/Coulomb-biased attention when `atom_attention=True`),
# SublayerConnection, BatchMolGraph/MolGraph's tensor-construction logic (`get_components`,
# `a2b`/`b2a`/`b2revb`/`a_scope`/`b_scope` bookkeeping), `index_select_ND`,
# `get_activation_function`, `initialize_weights` -- every mechanism in the real trainable
# network, transcribed unmodified.
#
# What is dropped (infra plumbing / non-architectural, not part of the forward-pass
# computation graph): `chemprop.args.TrainArgs` (a `tap.Tap`-based CLI-argument dataclass
# with dozens of training/CV/hyperopt-only fields) is replaced with a plain `_NS` namespace
# holding only the attributes the vendored modules actually read at `__init__`/`forward`
# time (`hidden_size`, `bias`, `depth`, `dropout`, `activation`, `atom_messages`,
# `undirected`, `aggregation`, `aggregation_norm`, `bond_attention`, `bond_fast_attention`,
# `atom_attention`, `num_heads`, `f_scale`, `adjacency`, `distance`, `coulomb`,
# `normalize_matrices`, `device`, `features_only`, `use_input_features`,
# `overwrite_default_atom_features`, `overwrite_default_bond_features`, `mpn_shared`,
# `number_of_molecules`, `dataset_type`, `num_tasks`, `multiclass_num_classes`,
# `ffn_num_layers`, `ffn_hidden_size`, `features_size`); `MolGraph.__init__`'s RDKit-based
# SMILES-to-atom/bond-feature extraction (`Chem.MolFromSmiles`, `atom_features`,
# `bond_features`) is bypassed because `rdkit` is not an installed base lib here -- instead
# a small synthetic-graph builder constructs the exact same tensors
# (`f_atoms`/`f_bonds`/`a2b`/`b2a`/`b2revb`/`a_scope`/`b_scope`, with the real
# `ATOM_FDIM`/`BOND_FDIM` feature widths) that `BatchMolGraph.__init__` and
# `MPNEncoder.forward` consume, so `MPNEncoder`/`MultiBondAttention`/`MultiAtomAttention`
# run their real, unmodified code paths on realistic random molecular-graph connectivity.
# `chemprop.attention_visualization.visualize_atom_attention` (a matplotlib-only debug
# plotting call gated by `viz_dir is not None`, never invoked here) and the
# `nn_utils.py` module's unrelated `seaborn`/`pandas`/`matplotlib`/`tqdm`-importing plotting
# helpers are omitted; only the pure-torch `index_select_ND`, `get_activation_function`,
# `initialize_weights` functions are carried over verbatim.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from functools import reduce
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from chemprop/nn_utils.py (pure-torch helpers only; the rest of that file
# imports seaborn/pandas/matplotlib/rdkit for unrelated plotting utilities)
# ---------------------------------------------------------------------------
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


def initialize_weights(model: nn.Module) -> None:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


# ---------------------------------------------------------------------------
# from chemprop/features/featurization.py (feature-width constants +
# BatchMolGraph's tensor-construction bookkeeping; MolGraph's RDKit-based
# per-atom/per-bond feature extraction is bypassed -- see header note)
# ---------------------------------------------------------------------------
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": list(range(5)),
}
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


class SyntheticMolGraph:
    """Synthetic stand-in for `chemprop.features.MolGraph` that carries the exact same
    fields (`n_atoms`, `n_bonds`, `f_atoms`, `f_bonds`, `a2b`, `b2a`, `b2revb`, `f_adj`,
    `f_dist`, `f_clb`) `BatchMolGraph.__init__` reads, built directly from random
    atom/bond feature vectors of the real feature widths instead of RDKit-parsed SMILES."""

    def __init__(self, n_atoms: int, generator: torch.Generator):
        self.smiles = f"<synthetic-{n_atoms}-atom-mol>"
        self.n_atoms = n_atoms
        self.n_bonds = 0
        self.f_atoms = [torch.rand(ATOM_FDIM, generator=generator).tolist() for _ in range(n_atoms)]
        self.f_bonds: List[List[float]] = []
        self.a2b: List[List[int]] = [[] for _ in range(n_atoms)]
        self.b2a: List[int] = []
        self.b2revb: List[int] = []

        # simple ring/chain connectivity so every atom has >= 1 bond
        for a1 in range(n_atoms):
            a2 = (a1 + 1) % n_atoms
            if a2 <= a1:
                continue
            fbond = torch.rand(BOND_FDIM, generator=generator).tolist()
            self.f_bonds.append(self.f_atoms[a1] + fbond)
            self.f_bonds.append(self.f_atoms[a2] + fbond)

            b1 = self.n_bonds
            b2 = b1 + 1
            self.a2b[a2].append(b1)
            self.b2a.append(a1)
            self.a2b[a1].append(b2)
            self.b2a.append(a2)
            self.b2revb.append(b2)
            self.b2revb.append(b1)
            self.n_bonds += 2

        self.f_adj = None
        self.f_dist = None
        self.f_clb = None


class BatchMolGraph:
    """Vendored verbatim from chemprop.features.featurization.BatchMolGraph."""

    def __init__(self, mol_graphs: List[SyntheticMolGraph]):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope: List[Tuple[int, int]] = []
        self.b_scope: List[Tuple[int, int]] = []

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        f_adj = [[]]
        f_dist = [[]]
        f_clb = [[]]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_adj.append(mol_graph.f_adj)
            f_dist.append(mol_graph.f_dist)
            f_clb.append(mol_graph.f_clb)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_adj = f_adj
        self.f_clb = f_clb
        self.f_dist = f_dist
        self.a2b = torch.LongTensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)]
        )
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None

    def get_components(self, atom_messages: bool = False):
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages) :]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_a2a(self) -> torch.LongTensor:
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
        return self.a2a


# ---------------------------------------------------------------------------
# from chemprop/models/attention.py (verbatim)
# ---------------------------------------------------------------------------
class MultiBondAttention(nn.Module):
    """Bond-level self-attention block (Transformer) in the message-passing phase."""

    def __init__(self, args):
        super(MultiBondAttention, self).__init__()
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.num_heads = args.num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size**-0.5

        self.W_b_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(self, message, b_scope):
        bond_vecs = []
        for i, (b_start, b_size) in enumerate(b_scope):
            if i == 0:
                bond_vecs.append(self.cached_zero_vector)

            cur_bond_message = message.narrow(0, b_start, b_size)
            cur_bond_message_size = cur_bond_message.size()
            b_q = self.W_b_q(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.att_size
            )
            b_k = self.W_b_k(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.att_size
            )
            b_v = self.W_b_v(cur_bond_message).view(
                cur_bond_message_size[0], self.num_heads, self.att_size
            )
            b_q = b_q.transpose(0, 1)
            b_k = b_k.transpose(0, 1).transpose(1, 2)
            b_v = b_v.transpose(0, 1)

            att_b_w = torch.matmul(b_q, b_k)
            att_b_w = F.softmax(att_b_w * self.scale_factor, dim=2)
            att_b_h = torch.matmul(att_b_w, b_v)
            att_b_h = self.act_func(att_b_h)
            att_b_h = self.dropout_layer(att_b_h)

            att_b_h = att_b_h.transpose(0, 1).contiguous()
            att_b_h = att_b_h.view(cur_bond_message_size[0], self.num_heads * self.att_size)
            att_b_h = self.W_b_o(att_b_h)
            assert att_b_h.size() == cur_bond_message_size

            att_b_h = att_b_h.unsqueeze(dim=0)
            att_b_h = self.norm(att_b_h)
            att_b_h = (att_b_h).squeeze(dim=0)

            bond_vecs.extend(att_b_h)

        bond_vecs = torch.stack(bond_vecs, dim=0)
        return bond_vecs


class MultiAtomAttention(nn.Module):
    """Atom-level self-attention block (Transformer) in the readout phase."""

    def __init__(self, args):
        super(MultiAtomAttention, self).__init__()
        self.atom_attention = args.atom_attention
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.normalize_matrices = args.normalize_matrices
        self.distance = args.distance
        self.adjacency = args.adjacency
        self.coulomb = args.coulomb
        self.device = args.device
        self.num_heads = args.num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size**-0.5
        self.f_scale = args.f_scale

        self.W_a_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(self, cur_hiddens, i, f_adj, f_dist, f_clb, viz_dir=None):
        cur_hiddens_size = cur_hiddens.size()

        a_q = self.W_a_q(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_k = self.W_a_k(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_v = self.W_a_v(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_q = a_q.transpose(0, 1)
        a_k = a_k.transpose(0, 1).transpose(1, 2)
        a_v = a_v.transpose(0, 1)

        att_a_w = torch.matmul(a_q, a_k)

        if self.adjacency:
            mol_adj = torch.Tensor(f_adj[i]).to(self.device)
            att_a_w[0] = att_a_w[0] + self.f_scale * mol_adj
            att_a_w[1] = att_a_w[1] + self.f_scale * mol_adj

        if self.distance:
            mol_dist = torch.Tensor(f_dist[i]).to(self.device)
            if self.normalize_matrices:
                mol_dist = F.softmax(mol_dist, dim=1)
            att_a_w[2] = att_a_w[2] + self.f_scale * mol_dist
            att_a_w[3] = att_a_w[3] + self.f_scale * mol_dist

        if self.coulomb:
            mol_clb = torch.Tensor(f_clb[i]).to(self.device)
            if self.normalize_matrices:
                mol_clb = F.softmax(mol_clb, dim=1)
            att_a_w[4] = att_a_w[4] + self.f_scale * mol_clb
            att_a_w[5] = att_a_w[5] + self.f_scale * mol_clb

        att_a_w = F.softmax(att_a_w * self.scale_factor, dim=2)
        att_a_h = torch.matmul(att_a_w, a_v)
        att_a_h = self.act_func(att_a_h)
        att_a_h = self.dropout_layer(att_a_h)

        att_a_h = att_a_h.transpose(0, 1).contiguous()
        att_a_h = att_a_h.view(cur_hiddens_size[0], self.num_heads * self.att_size)
        att_a_h = self.W_a_o(att_a_h)
        assert att_a_h.size() == cur_hiddens_size

        att_a_h = att_a_h.unsqueeze(dim=0)
        att_a_h = self.norm(att_a_h)

        mol_vec = (att_a_h).squeeze(dim=0)

        return mol_vec, torch.mean(att_a_w, axis=0)


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""

    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, original, attention):
        return original + self.dropout(attention)


# ---------------------------------------------------------------------------
# from chemprop/models/mpn.py (verbatim; `mol2graph`'s RDKit-only SMILES
# branch dropped, see header note -- forward() always receives a
# pre-built BatchMolGraph here)
# ---------------------------------------------------------------------------
class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout

        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        self.bond_attention = args.bond_attention
        self.bond_fast_attention = args.bond_fast_attention
        self.atom_attention = args.atom_attention
        self.distance = args.distance
        self.adjacency = args.adjacency
        self.coulomb = args.coulomb

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.bond_attention:
            self.bond_attention_block = MultiBondAttention(args)
            self.bond_residual = SublayerConnection(dropout=self.dropout)

        if self.bond_fast_attention:
            # NOTE: the real repo's Fastformer bond-attention (MultiBondFastAttention) is
            # architecturally distinct from MultiBondAttention; this staging module ships
            # with `bond_fast_attention=False` in its tiny config (see build_abt_mpnn), so
            # that branch is not exercised (bond_attention=True path used instead).
            raise NotImplementedError(
                "bond_fast_attention path not vendored in this staging build; "
                "use bond_attention=True instead"
            )

        if self.atom_attention:
            self.atom_attention_block = MultiAtomAttention(args)
            self.atom_residual = SublayerConnection(dropout=self.dropout)

    def forward(
        self,
        mol_graph: BatchMolGraph,
        mol_adj_batch=None,
        mol_dist_batch=None,
        mol_clb_batch=None,
        viz_dir: str = None,
    ) -> torch.FloatTensor:
        f_adj = mol_adj_batch
        f_dist = mol_dist_batch
        f_clb = mol_clb_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages
        )
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device),
            f_bonds.to(self.device),
            a2b.to(self.device),
            b2a.to(self.device),
            b2revb.to(self.device),
        )

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)

        message = self.act_func(input)

        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message
                if self.bond_attention or self.bond_fast_attention:
                    att_message = self.bond_attention_block(message, b_scope)
                    message = self.bond_residual(message, att_message)

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b

        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)

        a_input = torch.cat([f_atoms, a_message], dim=1)

        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                if self.atom_attention:
                    mol_vec, att_a_w = self.atom_attention_block(
                        cur_hiddens, i, f_adj, f_dist, f_clb, viz_dir
                    )
                    mol_vec = self.atom_residual(cur_hiddens, mol_vec)
                else:
                    mol_vec = cur_hiddens

                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs


class MPN(nn.Module):
    """A wrapper around MPNEncoder which featurizes input as needed."""

    def __init__(self, args, atom_fdim: int = None, bond_fdim: int = None):
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList(
                [MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules
            )
        else:
            self.encoder = nn.ModuleList(
                [
                    MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                    for _ in range(args.number_of_molecules)
                ]
            )

    def forward(
        self,
        batch: List[BatchMolGraph],
        features_batch=None,
        mol_adj_batch=None,
        mol_dist_batch=None,
        mol_clb_batch=None,
    ) -> torch.FloatTensor:
        # NOTE: real forward() re-featurizes `batch` via `mol2graph` (RDKit) if it isn't
        # already a list of BatchMolGraph; this staging build always calls with pre-built
        # BatchMolGraphs (see example_input_abt_mpnn), so that RDKit-only branch is unused.
        encodings = [
            enc(ba, mol_adj_batch, mol_dist_batch, mol_clb_batch)
            for enc, ba in zip(self.encoder, batch)
        ]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
            output = torch.cat([output, features_batch], dim=1)

        return output


# ---------------------------------------------------------------------------
# from chemprop/models/model.py (verbatim)
# ---------------------------------------------------------------------------
class MoleculeModel(nn.Module):
    """A model which contains a message passing network followed by feed-forward layers."""

    def __init__(self, args, featurizer: bool = False):
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.featurizer = featurizer

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args) -> None:
        self.encoder = MPN(args)

    def create_ffn(self, args) -> None:
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend(
                    [activation, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)]
                )
            ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, self.output_size)])

        self.ffn = nn.Sequential(*ffn)

    def forward(
        self,
        batch,
        features_batch=None,
        mol_adj_batch=None,
        mol_dist_batch=None,
        mol_clb_batch=None,
    ) -> torch.FloatTensor:
        if self.featurizer:
            return self.ffn[:-1](
                self.encoder(batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch)
            )

        output = self.ffn(
            self.encoder(batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch)
        )

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
class _NS:
    """Minimal attribute namespace equivalent to the subset of
    `chemprop.args.TrainArgs` the vendored modules actually read (see header note)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _tiny_abt_mpnn_args():
    return _NS(
        hidden_size=32,
        bias=False,
        depth=3,
        dropout=0.0,
        activation="ReLU",
        atom_messages=False,
        undirected=False,
        aggregation="mean",
        aggregation_norm=100,
        bond_attention=True,
        bond_fast_attention=False,
        atom_attention=True,
        num_heads=4,
        f_scale=0.45,
        adjacency=False,
        distance=False,
        coulomb=False,
        normalize_matrices=False,
        device=torch.device("cpu"),
        features_only=False,
        use_input_features=False,
        overwrite_default_atom_features=False,
        overwrite_default_bond_features=False,
        mpn_shared=False,
        number_of_molecules=1,
        dataset_type="regression",
        num_tasks=1,
        multiclass_num_classes=3,
        ffn_num_layers=2,
        ffn_hidden_size=32,
        features_size=0,
    )


def build_abt_mpnn():
    return MoleculeModel(_tiny_abt_mpnn_args())


def example_input_abt_mpnn():
    # MoleculeModel.forward(self, batch, features_batch=None, ...) takes `batch` as its
    # single required positional arg, where `batch` is itself a list of length
    # `number_of_molecules` (=1 here) of BatchMolGraph (one BatchMolGraph per "molecule
    # slot", each holding the whole minibatch for that slot). Wrapping in an extra list
    # here (`[[batch_graph]]`) is deliberate: `tl.trace`/`normalize_input_args` treats a
    # bare list `input_args` as "one element per positional arg" (issue #43 disambiguation)
    # since `MoleculeModel.forward` does not take exactly 1 named positional param; the
    # outer list is the *args wrapper, and its one element (`[batch_graph]`) is the actual
    # `batch` value forward() receives.
    generator = torch.Generator().manual_seed(0)
    mol_graphs = [
        SyntheticMolGraph(n_atoms=6, generator=generator),
        SyntheticMolGraph(n_atoms=5, generator=generator),
    ]
    batch_graph = BatchMolGraph(mol_graphs)
    return [[batch_graph]]


MENAGERIE_ENTRIES = [
    ("ABT-MPNN", "build_abt_mpnn", "example_input_abt_mpnn", 2023, "vendored-pytorch"),
]
