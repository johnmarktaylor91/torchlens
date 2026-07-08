# FAITHFUL PORT of CDDLeiden/DrugEx @ master (original framework: PyTorch, but the real
# class cannot be imported in a base torch env -- see rationale below)
#   drugex/training/generators/graph_transformer.py :: Block, AtomLayer, GraphTransformer
#   drugex/training/generators/utils.py :: PositionwiseFeedForward, SublayerConnection,
#                                           PositionalEncoding, tri_mask
#
# DrugEx v3's atom-by-atom autoregressive Graph Transformer for scaffold-constrained
# de novo molecule generation, per Liu et al. / van Westen group,
# "DrugEx v3: scaffold-constrained drug design with graph transformer-based
# reinforcement learning" (J. Cheminform. 2023). The model reads a linearized
# molecular-graph token stream (4 fields per node/edge event: atom type, current
# position, previous position, bond order) through a stack of self-attention blocks
# and decodes atom / bond / attachment-locus distributions with a GRUCell-based head.
#
# `graph_transformer.py` and `utils.py` themselves import ONLY torch (confirmed:
# no rdkit/gensim/etc in these two files). The real GraphTransformer class is NOT
# vendorable via a plain rung-2 copy, however, because its base classes
# (`FragGenerator` -> `Generator` in `drugex/training/generators/interfaces.py`)
# `from rdkit import Chem` at MODULE import time, and its constructor requires a
# `voc_trg` vocabulary object (`VocGraph` in `drugex/data/corpus/vocabulary.py`)
# that is itself built by iterating rdkit `Chem.Atom(...).GetAtomicNum()` calls.
# rdkit is not an installed base lib here and is not a per-rung-2-rules
# ad-hoc-installable dependency, so this is a faithful port: the SAME nn.Module
# architecture classes below (Block, AtomLayer, GraphTransformer.__init__/forward)
# are transcribed verbatim from the real repo file, with the `FragGenerator`
# base-class plumbing (rdkit-free training loop bookkeeping, GPU attach helpers)
# dropped, and the `voc_trg` vocabulary replaced by a minimal, rdkit-free stand-in
# carrying only the plain-Python metadata (`max_len`, `n_frags`, `size`, `tk2ix`,
# `masks`) the ARCHITECTURE code actually reads -- no atom-valence semantics or
# molecule chemistry is reimplemented, only a lookup-table shape stub.
#
# Ported here: the `forward(src, is_train=True)` teacher-forced loss branch, which is
# a single deterministic forward pass over a pre-built token tensor (matches how the
# real trainNet() calls the model). The `is_train=False` autoregressive sampling
# branch (atom-by-atom multinomial rollout, `sample()`, `trainNet`/`validateNet`) is
# real-code infrastructure around the architecture, not the architecture itself, and
# is intentionally not ported here.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


# ---- drugex/training/generators/utils.py (verbatim, torch-only) ----


def tri_mask(seq, diag=1):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
    masks = masks.bool().to(seq.device)
    return masks


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        y = self.w_1(x).relu()
        y = self.w_2(y)
        return y


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm"""

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(x)
        y = self.dropout(y)
        y = self.norm(x + y)
        return y


class PositionalEncoding(nn.Module):
    """Positional encoding for graph transformer"""

    def __init__(self, d_model: int, max_len=100, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        bsize, sqlen = x.size()
        y = x.reshape(bsize * sqlen)
        code = self.pe[y, :].view(bsize, sqlen, -1)
        return code


# ---- drugex/training/generators/graph_transformer.py (verbatim architecture) ----


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model) for _ in range(2)])

    def forward(self, x, key_mask=None, attn_mask=None):
        x = self.connector[0](x, lambda x: self.attn(x, x, x, key_mask, attn_mask=attn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class AtomLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_inner=1024, n_layer=12):
        super(AtomLayer, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.blocks = nn.ModuleList(
            [Block(self.d_model, self.n_head, d_inner=d_inner) for _ in range(self.n_layer)]
        )

    def forward(self, x: torch.Tensor, key_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x, key_mask=key_mask, attn_mask=attn_mask)
        return x


class _VocGraphStub:
    """Minimal rdkit-free stand-in for drugex.data.corpus.vocabulary.VocGraph,
    carrying only the plain lookup-table fields GraphTransformer's ARCHITECTURE code
    reads (max_len, n_frags, size, tk2ix['*'|'GO'], masks). No atom-valence/chemistry
    semantics reimplemented -- this is metadata shape only, not a functional stand-in
    for molecule validity checking."""

    def __init__(self, size=40, max_len=24, n_frags_field=4):
        self.size = size
        self.max_len = max_len
        self.n_frags = n_frags_field
        self.tk2ix = {"EOS": 0, "GO": 1, "*": size - 1}
        # masks[i] = max bond-valence for token i (0 for EOS/GO), matching VocGraph.masks semantics
        self.masks = torch.randint(0, 4, (size,)).long()
        self.masks[0] = 0
        self.masks[1] = 0


class GraphTransformer(nn.Module):
    """Graph Transformer for molecule generation from fragments, per
    graph_transformer.py::GraphTransformer.__init__/forward (FragGenerator/Generator
    base-class training/GPU-attach plumbing dropped; architecture unchanged)."""

    def __init__(
        self, voc_trg, d_emb=512, d_model=512, n_head=8, d_inner=1024, n_layer=12, pad_idx=0
    ):
        super(GraphTransformer, self).__init__()
        self.mol_type = "graph"
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_grows = voc_trg.max_len - voc_trg.n_frags - 1
        self.n_frags = voc_trg.n_frags + 1
        self.d_emb = d_emb
        self.emb_word = nn.Embedding(voc_trg.size * 4, self.d_emb, padding_idx=pad_idx)
        self.emb_atom = nn.Embedding(voc_trg.size, self.d_emb, padding_idx=pad_idx)
        self.emb_loci = nn.Embedding(self.n_grows, self.d_emb)
        self.emb_site = PositionalEncoding(self.d_emb, max_len=self.n_grows * self.n_grows)
        self.attn = AtomLayer(d_model=d_model, n_head=n_head, d_inner=d_inner, n_layer=n_layer)

        self.rnn = nn.GRUCell(self.d_model, self.d_model)
        self.prj_atom = nn.Linear(d_emb, self.voc_trg.size)
        self.prj_bond = nn.Linear(d_model, 4)
        self.prj_loci = nn.Linear(d_model, self.n_grows)
        self.init_states()

        self.model_name = "GraphTransformer"

    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, is_train=True):
        """src: `torch.Tensor` of shape (batch, seq_len, 4) -- linearized molecular
        graph event stream, matching the real forward()'s documented input. Only the
        `is_train=True` teacher-forced branch is ported (see module docstring)."""
        assert is_train, "only the is_train=True teacher-forced branch is ported"
        src, trg = src[:, :-1, :], src[:, 1:, :]
        batch, sqlen, _ = src.shape
        triu = tri_mask(src[:, :, 0])

        # dec - atom environment
        emb = self.emb_word(src[:, :, 3] + src[:, :, 0] * 4)
        emb += self.emb_site(src[:, :, 1] * self.n_grows + src[:, :, 2])
        dec = self.attn(emb.transpose(0, 1), attn_mask=triu)

        dec = dec.transpose(0, 1).reshape(batch * sqlen, -1)
        out_atom = self.prj_atom(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
        out_atom = out_atom.gather(2, trg[:, :, 0].unsqueeze(2))

        atom = self.emb_atom(trg[:, :, 0]).reshape(batch * sqlen, -1)
        dec = self.rnn(atom, dec)
        out_bond = self.prj_bond(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
        out_bond = out_bond.gather(2, trg[:, :, 3].unsqueeze(2))

        word = self.emb_word(trg[:, :, 3] + trg[:, :, 0] * 4)
        word = word.reshape(batch * sqlen, -1)
        dec = self.rnn(word, dec)
        out_prev = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
        out_prev = out_prev.gather(2, trg[:, :, 2].unsqueeze(2))

        curr = self.emb_loci(trg[:, :, 2]).reshape(batch * sqlen, -1)
        dec = self.rnn(curr, dec)
        out_curr = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
        out_curr = out_curr.gather(2, trg[:, :, 1].unsqueeze(2))

        return out_atom, out_curr, out_prev, out_bond


def build_drugex_graph_transformer():
    torch.manual_seed(0)
    voc_trg = _VocGraphStub(size=40, max_len=24, n_frags_field=4)
    return GraphTransformer(voc_trg, d_emb=64, d_model=64, n_head=4, d_inner=128, n_layer=2).eval()


def example_input_drugex_graph_transformer():
    torch.manual_seed(0)
    batch, seq_len, n_grows = 2, 20, 19  # seq_len = max_len - 1 -> valid growing-phase indices
    src = torch.zeros(batch, seq_len, 4, dtype=torch.long)
    src[:, :, 0] = torch.randint(0, 38, (batch, seq_len))  # atom token id (< voc size - 1)
    src[:, :, 1] = torch.randint(0, n_grows, (batch, seq_len))  # current locus
    src[:, :, 2] = torch.randint(0, n_grows, (batch, seq_len))  # previous locus
    src[:, :, 3] = torch.randint(0, 4, (batch, seq_len))  # bond order
    return (src,)


MENAGERIE_ENTRIES = [
    (
        "DrugEx-GraphTransformer",
        build_drugex_graph_transformer,
        example_input_drugex_graph_transformer,
        2023,
        "ported-pytorch",
    ),
]
