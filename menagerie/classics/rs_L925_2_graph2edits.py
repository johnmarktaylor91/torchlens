# SOURCE: vendored from Jamson-Zhong/Graph2Edits @ b12e97391c378e46e79e617d80017e95b21bba65
# (models/encoder.py::MPNEncoder, MultiHeadAttention, FeedForward, Global_Attention;
#  models/model_utils.py::index_select_ND, creat_edits_feats, unbatch_feats; the
#  Graph2Edits.compute_edit_scores forward path from models/graph2edits.py -- module
#  bodies only, trimmed of everything not on the random-init forward-pass call path)
#
# Graph2Edits ("An End-to-End Step-Wise, Equivariant Graph Generative Model for
# Reaction Prediction and Molecule Generation" / Zhong et al., graph-edit-sequence
# retrosynthesis model) predicts a single-shot set of (atom-edit, bond-edit,
# graph-level stop) scores for a product molecule graph via a D-MPNN encoder
# (`MPNEncoder`) with an optional Transformer-style global self-attention refinement
# (`Global_Attention`), followed by 3 linear MLP heads (atom_linear, bond_linear,
# graph_linear) that mirror `Graph2Edits.compute_edit_scores` from `models/graph2edits.py`.
#
# The real repo's `Graph2Edits` wrapper class additionally imports `rdkit` (for
# `apply_edit_to_mol`/SMILES parsing used only by the `predict()` inference-time
# molecule-editing loop) and drives `MolGraph`/`get_batch_graphs`
# (`utils/rxn_graphs.py`, `utils/collate_fn.py`) to featurize real molecules with
# rdkit -- none of that is architecture, it is data preprocessing. This staging
# module vendors the real `nn.Module` classes verbatim (MPNEncoder's D-MPNN
# message-passing recurrence, the multi-head global-attention block, and the three
# scoring heads exactly as `compute_edit_scores` wires them) and drives them with a
# synthetic directed-bond-graph batch built by hand in the exact tensor layout
# `utils/collate_fn.get_batch_graphs` produces (index 0 reserved as padding for
# both atoms and bonds; each undirected bond expands into 2 directed bond records;
# `a2b[a]` lists incoming directed-bond indices for atom `a`, padded per batch to
# the max in/degree) -- so no rdkit/graph-featurization dependency is needed to
# exercise the real architecture end to end.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    Parameters
    ----------
    source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    final_size = index_size + suffix_dim

    # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.reshape(-1))
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = target.view(final_size)

    return target


def creat_edits_feats(atom_feats, atom_scope):
    a_feats = []
    masks = []

    for idx, (st_a, le_a) in enumerate(atom_scope):
        feats = atom_feats[st_a : st_a + le_a]
        mask = torch.ones(feats.size(0), dtype=torch.uint8)
        a_feats.append(feats)
        masks.append(mask)

    a_feats = nn.utils.rnn.pad_sequence(a_feats, batch_first=True, padding_value=0)
    masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

    return a_feats, masks


def unbatch_feats(feats, atom_scope):
    atom_feats = []

    for idx, (st_a, le_a) in enumerate(atom_scope):
        atom_feats.append(feats[idx][:le_a])

    a_feats = torch.cat(atom_feats, dim=0)

    pad_tensor = torch.zeros(1, a_feats.size(1), device=a_feats.device)
    return torch.cat((pad_tensor, a_feats), dim=0)


class MPNEncoder(nn.Module):
    """Class: 'MPNEncoder' is a message passing neural network for encoding molecules."""

    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int,
        depth: int,
        dropout: float = 0.15,
        atom_message: bool = False,
    ):
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
        atom_message: 'D-MPNN' or 'MPNN', centers messages on bonds or atoms.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.atom_message = atom_message

        # Input
        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)

        # Update message
        if self.atom_message:
            self.w_h = nn.Linear(self.bond_fdim + self.hidden_size, self.hidden_size)

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Output
        self.W_o = nn.Sequential(
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU()
        )

    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        # Input
        if self.atom_message:
            a2a = b2a[a2b]  # num_atoms x max_num_bonds
            f_bonds = f_bonds[:, -self.bond_fdim :]
            input = self.w_i(f_atoms)  # num_atoms x hidden
        else:
            input = self.w_i(f_bonds)  # num_bonds x hidden

        # Message passing
        message = input
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding

        for depth in range(self.depth - 1):
            if self.atom_message:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2a)
                # num_atoms x max_num_bonds x bond_fdim
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                # num_atoms x max_num_bonds x hidden + bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                # num_atoms x hidden + bond_fdim
                message = nei_message.sum(dim=1)
                message = self.w_h(message)  # num_bonds x hidden
            else:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.gru(input, message)  # num_bonds x hidden_size
            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden

        if self.atom_message:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2a)
        else:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden

        if mask is None:
            mask = torch.ones(atom_hiddens.size(0), 1, device=f_atoms.device)
            mask[0, 0] = 0  # first node is padding

        return atom_hiddens * mask


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores[~mask.bool()] = float(-9e15)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class Global_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):
        super(Global_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x


class Graph2EditsScorer(nn.Module):
    """Faithful re-wiring of `Graph2Edits.compute_edit_scores` (models/graph2edits.py)
    for a single forward step: encoder -> optional global attention -> 3 scoring
    heads (atom/bond/graph edits), matching `_build_layers` + `compute_edit_scores`
    verbatim (rxn/atom/bond vocab bookkeeping and the rdkit-driven `predict()`
    inference loop are dropped; they are not part of the trainable architecture)."""

    def __init__(self, config: dict, atom_outdim: int, bond_outdim: int) -> None:
        super().__init__()
        self.config = config
        self.atom_outdim = atom_outdim
        self.bond_outdim = bond_outdim
        self._build_layers()

    def _build_layers(self) -> None:
        config = self.config
        self.encoder = MPNEncoder(
            atom_fdim=config["n_atom_feat"],
            bond_fdim=config["n_bond_feat"],
            hidden_size=config["mpn_size"],
            depth=config["depth"],
            dropout=config["dropout_mpn"],
            atom_message=config["atom_message"],
        )

        self.W_vv = nn.Linear(config["mpn_size"], config["mpn_size"], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config["mpn_size"], config["mpn_size"], bias=False)

        if config["use_attn"]:
            self.attn = Global_Attention(d_model=config["mpn_size"], heads=config["n_heads"])

        self.atom_linear = nn.Sequential(
            nn.Linear(config["mpn_size"], config["mlp_size"]),
            nn.ReLU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], self.atom_outdim),
        )
        self.bond_linear = nn.Sequential(
            nn.Linear(config["mpn_size"] * 2, config["mlp_size"]),
            nn.ReLU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], self.bond_outdim),
        )

        self.graph_linear = nn.Sequential(
            nn.Linear(config["mpn_size"], config["mlp_size"]),
            nn.ReLU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], 1),
        )

    def forward(
        self, prod_tensors: Tuple[torch.Tensor], prod_scopes: Tuple[List]
    ) -> Tuple[torch.Tensor]:
        atom_scope, bond_scope = prod_scopes
        n_atoms = prod_tensors[0].size(0)
        prev_atom_hiddens = torch.zeros(
            n_atoms, self.config["mpn_size"], device=prod_tensors[0].device
        )

        a_feats = self.encoder(prod_tensors, mask=None)
        if self.config["use_attn"]:
            feats, mask = creat_edits_feats(a_feats, atom_scope)
            attention_score, feats = self.attn(feats, mask)
            a_feats = unbatch_feats(feats, atom_scope)

        atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))

        node_feats = atom_feats.clone()
        bond_starts = index_select_ND(atom_feats, index=prod_tensors[-1][:, 0])
        bond_ends = index_select_ND(atom_feats, index=prod_tensors[-1][:, 1])
        bond_feats = torch.cat([bond_starts, bond_ends], dim=1)

        graph_vecs = torch.stack([atom_feats[st : st + le].sum(dim=0) for st, le in atom_scope])

        atom_outs = self.atom_linear(node_feats)
        bond_outs = self.bond_linear(bond_feats)
        graph_outs = self.graph_linear(graph_vecs)

        edit_scores = [
            torch.cat(
                [
                    bond_outs[st_b : st_b + le_b].flatten(),
                    atom_outs[st_a : st_a + le_a].flatten(),
                    graph_outs[idx],
                ],
                dim=-1,
            )
            for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))
        ]

        return edit_scores


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
ATOM_FDIM = 83  # utils/mol_features.ATOM_FDIM (use_rxn_class=False path)
BOND_FDIM = 12  # utils/mol_features.BOND_FDIM


def build_graph2edits():
    config = {
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,  # atom_message=False: bond feat = atom+bond concat
        "mpn_size": 16,
        "mlp_size": 20,
        "depth": 3,
        "dropout_mpn": 0.0,
        "dropout_mlp": 0.0,
        "atom_message": False,
        "use_attn": True,
        "n_heads": 4,
    }
    atom_outdim = 5  # toy atom-edit vocab size
    bond_outdim = 6  # toy bond-edit vocab size
    return Graph2EditsScorer(config, atom_outdim=atom_outdim, bond_outdim=bond_outdim)


def _toy_mol_graph(n_atoms: int, bonds: List[Tuple[int, int]], atom_fdim: int, bond_fdim: int):
    """Builds one molecule's directed-bond-graph tensors, mirroring
    `utils/rxn_graphs.MolGraph._build_graph` (each undirected bond -> 2 directed
    bond records; f_bonds[b] = concat(f_atoms[origin_atom], bond_features))."""
    f_atoms = [torch.randn(atom_fdim) for _ in range(n_atoms)]
    f_bonds: List[torch.Tensor] = []
    a2b: List[List[int]] = [[] for _ in range(n_atoms)]
    b2a: List[int] = []
    b2revb: List[int] = []
    undirected_b2a: List[Tuple[int, int]] = []
    n_bonds = 0
    for a1, a2 in bonds:
        f_bond = torch.randn(bond_fdim)
        f_bonds.append(torch.cat([f_atoms[a1], f_bond]))
        f_bonds.append(torch.cat([f_atoms[a2], f_bond]))
        b1, b2 = n_bonds, n_bonds + 1
        a2b[a2].append(b1)  # b1 = a1 --> a2
        b2a.append(a1)
        a2b[a1].append(b2)  # b2 = a2 --> a1
        b2a.append(a2)
        b2revb.append(b2)
        b2revb.append(b1)
        n_bonds += 2
        undirected_b2a.append((a1, a2))
    return f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a, n_atoms, n_bonds


def example_input_graph2edits():
    torch.manual_seed(0)
    atom_fdim, bond_fdim = ATOM_FDIM, BOND_FDIM

    # Two toy "molecules": a 4-atom ring and a 3-atom chain.
    mol_specs = [
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        (3, [(0, 1), (1, 2)]),
    ]

    n_atoms_total = 1  # index 0 reserved for padding
    n_bonds_total = 1
    f_atoms = [torch.zeros(atom_fdim)]
    f_bonds = [torch.zeros(atom_fdim + bond_fdim)]
    a2b: List[List[int]] = [[]]
    b2a: List[int] = [0]
    b2revb: List[int] = [0]
    undirected_b2a: List[List[int]] = [[0, 0]]
    atom_scope = []
    bond_scope = []

    for n_atoms, bonds in mol_specs:
        (m_f_atoms, m_f_bonds, m_a2b, m_b2a, m_b2revb, m_undirected_b2a, m_n_atoms, m_n_bonds) = (
            _toy_mol_graph(n_atoms, bonds, atom_fdim, bond_fdim)
        )

        f_atoms.extend(m_f_atoms)
        f_bonds.extend(m_f_bonds)
        for a in range(m_n_atoms):
            a2b.append([b + n_bonds_total for b in m_a2b[a]])
        for b in range(m_n_bonds):
            b2a.append(n_atoms_total + m_b2a[b])
            b2revb.append(n_bonds_total + m_b2revb[b])
        for a1, a2 in m_undirected_b2a:
            undirected_b2a.append([a1 + n_atoms_total, a2 + n_atoms_total])

        atom_scope.append((n_atoms_total, m_n_atoms))
        bond_scope.append((len(undirected_b2a) - len(m_undirected_b2a), len(m_undirected_b2a)))
        n_atoms_total += m_n_atoms
        n_bonds_total += m_n_bonds

    max_deg = max(len(x) for x in a2b)
    a2b_padded = torch.zeros(len(a2b), max_deg, dtype=torch.long)
    for i, row in enumerate(a2b):
        a2b_padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)

    f_atoms = torch.stack(f_atoms)
    f_bonds = torch.stack(f_bonds)
    b2a = torch.tensor(b2a, dtype=torch.long)
    b2revb = torch.tensor(b2revb, dtype=torch.long)
    undirected_b2a = torch.tensor(undirected_b2a, dtype=torch.long)

    prod_tensors = (f_atoms, f_bonds, a2b_padded, b2a, b2revb, undirected_b2a)
    prod_scopes = (atom_scope, bond_scope)
    return prod_tensors, prod_scopes


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Graph2Edits", build_graph2edits, example_input_graph2edits, 2023, "SOURCE_AVAILABLE"),
]
