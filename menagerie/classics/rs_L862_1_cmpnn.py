# SOURCE: vendored from https://github.com/SY575/CMPNN @ master (b647df22ec8fde81785c5a86138ac1efd9ccf9c1)
# (chemprop/models/mpn.py, chemprop/models/model.py, chemprop/nn_utils.py -- the real
# CMPNN "Communicative Message Passing Neural Network" architecture, unmodified. The
# original repo's featurization (chemprop/features/featurization.py) builds the
# `BatchMolGraph` input tensors from RDKit-parsed SMILES; RDKit is not installed in
# this environment, so this staging module builds a structurally-equivalent
# `BatchMolGraph`-shaped input directly out of synthetic tensors (same fields, same
# dtypes, same zero-padding convention as the real `BatchMolGraph.__init__`) and feeds
# it to the real `MPN(..., graph_input=True)` encoder path, which is exactly the code
# path the original repo uses when molecule graphs are pre-built (skips `mol2graph`).
# The nn.Module architecture below (MPNEncoder, BatchGRU, MPN, MoleculeModel) is
# untouched from the source.
from argparse import Namespace
from typing import List, Tuple, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Feature dimensions (chemprop/features/featurization.py, RDKit-free constants)
# ---------------------------------------------------------------------------
ATOM_FDIM = 133
BOND_FDIM = 14


def get_atom_fdim(args: Namespace) -> int:
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    return BOND_FDIM


# ---------------------------------------------------------------------------
# chemprop/nn_utils.py (verbatim, minus the MoleculeDataset-only helper)
# ---------------------------------------------------------------------------
def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(
        dim=0, index=index.view(-1)
    )  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target[index == 0] = 0
    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    """
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


def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


# ---------------------------------------------------------------------------
# chemprop/models/mpn.py (verbatim architecture: MPNEncoder, BatchGRU, MPN)
# ---------------------------------------------------------------------------
class BatchMolGraph:
    """
    A structurally-equivalent stand-in for chemprop's real `BatchMolGraph`. The real
    class (chemprop/features/featurization.py) is built by parsing SMILES with RDKit;
    this constructor instead takes already-built synthetic tensors following the exact
    same shapes/padding convention (index 0 reserved as the zero-padding row) so the
    real `MPNEncoder.forward` / `get_components()` contract is unchanged.
    """

    def __init__(self, f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope):
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.a2b = a2b
        self.b2a = b2a
        self.b2revb = b2revb
        self.a_scope = a_scope
        self.b_scope = b_scope
        self.bonds = None

    def get_components(self):
        return (
            self.f_atoms,
            self.f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.bonds,
        )


class MPNEncoder(nn.Module):
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
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f"W_h_{depth}"] = nn.Linear(
                w_h_input_size_bond, self.hidden_size, bias=self.bias
            )

        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

    def forward(self, mol_graph, features_batch=None) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                f_atoms.cuda(),
                f_bonds.cuda(),
                a2b.cuda(),
                b2a.cuda(),
                b2revb.cuda(),
            )

        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f"W_h_{depth}"](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs  # B x H


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(
            -1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size)
        )

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(
                cur_message
            )
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat(
            [
                torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                cur_message_unpadding,
            ],
            0,
        )
        return message


class MPN(nn.Module):
    def __init__(
        self,
        args: Namespace,
        atom_fdim: int = None,
        bond_fdim: int = None,
        graph_input: bool = False,
    ):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = (
            bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        )
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(
        self, batch: Union[List[str], "BatchMolGraph"], features_batch=None
    ) -> torch.FloatTensor:
        # graph_input=True is the real repo's own code path for pre-built molecule
        # graphs (chemprop/models/mpn.py MPN.forward): it skips `mol2graph` (the only
        # RDKit-dependent call in this module) and passes `batch` straight to the
        # encoder, which only ever calls `batch.get_components()`.
        output = self.encoder.forward(batch, features_batch)
        return output


# ---------------------------------------------------------------------------
# chemprop/models/model.py (verbatim: MoleculeModel, build_model)
# ---------------------------------------------------------------------------
class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        self.encoder = MPN(args, graph_input=True)

    def create_ffn(self, args: Namespace):
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, args.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend(
                    [
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ]
                )
            ffn.extend(
                [
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.output_size),
                ]
            )

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == "multiclass":
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(
        classification=args.dataset_type == "classification",
        multiclass=args.dataset_type == "multiclass",
    )
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model


# ---------------------------------------------------------------------------
# Menagerie staging glue: tiny CMPNN + synthetic BatchMolGraph builder
# ---------------------------------------------------------------------------
def _cmpnn_args() -> Namespace:
    args = Namespace()
    args.hidden_size = 32
    args.bias = False
    args.depth = 3
    args.dropout = 0.0
    args.undirected = False
    args.atom_messages = False
    args.features_only = False
    args.use_input_features = False
    args.cuda = False
    args.activation = "ReLU"
    args.dataset_type = "classification"
    args.num_tasks = 1
    args.ffn_num_layers = 2
    args.ffn_hidden_size = 32
    args.no_cache = True
    return args


def _synthetic_batch_mol_graph(atom_fdim: int, bond_fdim_combined: int) -> BatchMolGraph:
    """
    Builds two tiny "molecules" worth of zero-padded synthetic graph tensors,
    matching the exact shapes/padding convention `BatchMolGraph.__init__` produces
    (index 0 reserved as the zero row for both atoms and bonds).
    """
    torch.manual_seed(0)
    mol_atom_counts = [4, 3]  # two toy molecules: 4 atoms, 3 atoms
    n_atoms = 1  # index 0 is zero-padding
    n_bonds = 1
    f_atoms = [torch.zeros(atom_fdim)]
    f_bonds = [torch.zeros(bond_fdim_combined)]
    a2b_raw: List[List[int]] = [[]]
    b2a = [0]
    b2revb = [0]
    a_scope: List[Tuple[int, int]] = []
    b_scope: List[Tuple[int, int]] = []

    for n_mol_atoms in mol_atom_counts:
        mol_a2b = [[] for _ in range(n_mol_atoms)]
        mol_start_bond = n_bonds
        # fully connect a simple ring/path so every atom has >=1 bond
        for a1 in range(n_mol_atoms):
            a2 = (a1 + 1) % n_mol_atoms
            if a2 <= a1:
                continue
            fb = torch.rand(bond_fdim_combined)
            f_bonds.append(fb)
            f_bonds.append(fb)
            b1 = n_bonds
            b2 = b1 + 1
            mol_a2b[a2].append(b1)
            b2a.append(a1 + n_atoms)
            mol_a2b[a1].append(b2)
            b2a.append(a2 + n_atoms)
            b2revb.append(b2)
            b2revb.append(b1)
            n_bonds += 2

        for a in range(n_mol_atoms):
            f_atoms.append(torch.rand(atom_fdim))
            a2b_raw.append([b + 0 for b in mol_a2b[a]])
        a_scope.append((n_atoms, n_mol_atoms))
        b_scope.append((mol_start_bond, n_bonds - mol_start_bond))
        n_atoms += n_mol_atoms

    max_num_bonds = max(1, max(len(bonds) for bonds in a2b_raw))
    f_atoms_t = torch.stack(f_atoms, dim=0)
    f_bonds_t = torch.stack(f_bonds, dim=0)
    a2b_t = torch.LongTensor(
        [row[:max_num_bonds] + [0] * (max_num_bonds - len(row)) for row in a2b_raw]
    )
    b2a_t = torch.LongTensor(b2a)
    b2revb_t = torch.LongTensor(b2revb)

    return BatchMolGraph(f_atoms_t, f_bonds_t, a2b_t, b2a_t, b2revb_t, a_scope, b_scope)


def build_cmpnn():
    args = _cmpnn_args()
    return build_model(args)


def example_input_cmpnn():
    args = _cmpnn_args()
    bond_fdim_combined = get_bond_fdim(args) + (not args.atom_messages) * get_atom_fdim(args)
    graph = _synthetic_batch_mol_graph(get_atom_fdim(args), bond_fdim_combined)
    return (graph,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CMPNN", build_cmpnn, example_input_cmpnn, 2020, "SOURCE_AVAILABLE"),
]
