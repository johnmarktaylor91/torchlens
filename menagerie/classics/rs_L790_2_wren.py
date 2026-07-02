# SOURCE: vendored from CompRhys/aviary @ main (aviary/wren/model.py, aviary/networks.py,
# aviary/segments.py, aviary/scatter.py)
# https://github.com/CompRhys/aviary
"""Wren: a Wyckoff-position-based coordinate-free message-passing network for
crystal property prediction from the ``aviary`` package.

``Wren``/``DescriptorNetwork`` (the message-passing graph architecture: element
embedding + Wyckoff-site embedding -> ``MessageLayer`` weighted-attention
message passing over element/site pairs -> multi-head ``WeightedAttentionPooling``
crystal-level readout -> residual trunk/output networks) is transcribed
unmodified from ``aviary/wren/model.py``. ``MessageLayer``/``WeightedAttentionPooling``
(``aviary/segments.py``), ``SimpleNetwork``/``ResidualNetwork``
(``aviary/networks.py``), and ``scatter_reduce`` (``aviary/scatter.py``) are
transcribed unmodified.

Two non-architectural substitutions were made so the model can be constructed
without the optional ``pymatgen``/``wandb`` dependencies (data-loading and
experiment-tracking concerns, not part of the architecture):

* ``BaseModelClass`` (``aviary/core.py``) is a thin ``nn.Module`` subclass that
  only stores bookkeeping attributes (``task_dict``, ``robust``, ``device``,
  ...) in its ``__init__`` -- it adds no architecture. It is replaced here by
  plain ``nn.Module`` with the same bookkeeping inlined.
* ``get_element_embedding``/``get_sym_embedding`` (``aviary/utils.py``) just
  build a plain ``nn.Embedding`` and copy in pretrained feature-table weights
  loaded from a JSON file on disk. The embedding *architecture* is exactly
  ``nn.Embedding(num_rows, feature_dim)``; only the file-loading and weight
  initialization are skipped here (random init instead of the pretrained
  matscholar200/bra-alg-off tables), matching the real vocab sizes
  (103 elements x 200-dim matscholar200; 1732 Wyckoff-site rows x 444-dim
  bra-alg-off, per the JSON tables shipped in the repo).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


def scatter_reduce(src, index, dim=-1, dim_size=None, reduce="sum"):
    """Performs a scatter-reduce operation on the input tensor."""
    if dim_size is None:
        dim_size = index.max().item() + 1

    shape = list(src.shape)
    shape[dim] = dim_size

    if index.dim() != src.dim():
        if index.dim() != 1:
            raise RuntimeError(
                "Index tensor must be 1D or have the same number of dimensions "
                f"as src tensor. {index.shape=} != {src.shape=}"
            )
        repeat_shape = [1] * src.dim()
        repeat_shape[dim] = src.size(dim)
        index = index.view(-1, *[1] * (src.dim() - 1)).expand_as(src)

    if reduce in ("sum", "mean"):
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out = out.scatter_add(dim, index, src)
        if reduce == "mean":
            count = torch.zeros(shape, dtype=src.dtype, device=src.device)
            count = count.scatter_add(dim, index, torch.ones_like(src))
            out = out / (count + (count == 0).float())
    elif reduce in ("amax", "max"):
        out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
        out = torch.max(out, out.scatter(dim, index, src))
    elif reduce in ("amin", "min"):
        out = torch.full(shape, float("inf"), dtype=src.dtype, device=src.device)
        out = torch.min(out, out.scatter(dim, index, src))
    elif reduce == "prod":
        out = torch.ones(shape, dtype=src.dtype, device=src.device)
        out = out.scatter(dim, index, src, reduce="multiply")
    else:
        raise ValueError(f"Unsupported reduction method: {reduce}")

    return out


class SimpleNetwork(nn.Module):
    """Simple Feed Forward Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1))
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        for fc, bn, act in zip(self.fcs, self.bns, self.acts, strict=False):
            x = act(bn(fc(x)))

        return self.fc_out(x)


class ResidualNetwork(nn.Module):
    """Feed forward Residual Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        dims = [input_dim, *list(hidden_layer_dims)]

        self.fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )

        if batch_norm:
            self.bns = nn.ModuleList(nn.BatchNorm1d(dims[idx + 1]) for idx in range(len(dims) - 1))
        else:
            self.bns = nn.ModuleList(nn.Identity() for _ in range(len(dims) - 1))

        self.res_fcs = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1], bias=False)
            if (dims[idx] != dims[idx + 1])
            else nn.Identity()
            for idx in range(len(dims) - 1)
        )
        self.acts = nn.ModuleList(activation() for _ in range(len(dims) - 1))

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts, strict=False):
            x = act(bn(fc(x))) + res_fc(x)

        return self.fc_out(x)


class WeightedAttentionPooling(nn.Module):
    """Weighted softmax attention layer."""

    def __init__(self, gate_nn: nn.Module, message_nn: nn.Module) -> None:
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor, index: Tensor, weights: Tensor) -> Tensor:
        gate = self.gate_nn(x)

        gate -= scatter_reduce(gate, index, dim=0, reduce="amax")[index]
        gate = (weights**self.pow) * gate.exp()
        gate /= scatter_reduce(gate, index, dim=0, reduce="sum")[index] + 1e-10

        x = self.message_nn(x)
        return scatter_reduce(gate * x, index, dim=0, reduce="sum")


class MessageLayer(nn.Module):
    """MessageLayer to propagate information between nodes in graph."""

    def __init__(
        self,
        msg_fea_len: int,
        num_msg_heads: int,
        msg_gate_layers: Sequence[int],
        msg_net_layers: Sequence[int],
    ) -> None:
        super().__init__()

        self.pooling = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork(2 * msg_fea_len, 1, msg_gate_layers),
                message_nn=SimpleNetwork(2 * msg_fea_len, msg_fea_len, msg_net_layers),
            )
            for _ in range(num_msg_heads)
        )

    def forward(
        self,
        node_weights: Tensor,
        node_prev_features: Tensor,
        self_idx: LongTensor,
        neighbor_idx: LongTensor,
    ) -> Tensor:
        node_nbr_weights = node_weights[neighbor_idx, :]
        msg_nbr_fea = node_prev_features[neighbor_idx, :]
        msg_self_fea = node_prev_features[self_idx, :]
        message = torch.cat([msg_self_fea, msg_nbr_fea], dim=1)

        head_features = []
        for attn_head in self.pooling:
            out_msg = attn_head(message, index=self_idx, weights=node_nbr_weights)
            head_features.append(out_msg)

        node_update = torch.stack(head_features).mean(dim=0)

        return node_update + node_prev_features


class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the Roost model."""

    def __init__(
        self,
        elem_emb_len: int,
        sym_emb_len: int,
        elem_fea_len: int = 32,
        sym_fea_len: int = 32,
        n_graph: int = 3,
        elem_heads: int = 1,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 1,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
    ):
        super().__init__()

        self.elem_embed = nn.Linear(elem_emb_len, elem_fea_len)
        self.sym_embed = nn.Linear(sym_emb_len + 1, sym_fea_len)

        fea_len = elem_fea_len + sym_fea_len

        self.graphs = nn.ModuleList(
            MessageLayer(
                msg_fea_len=fea_len,
                num_msg_heads=elem_heads,
                msg_gate_layers=elem_gate,
                msg_net_layers=elem_msg,
            )
            for _ in range(n_graph)
        )

        self.cry_pool = nn.ModuleList(
            WeightedAttentionPooling(
                gate_nn=SimpleNetwork(fea_len, 1, cry_gate),
                message_nn=SimpleNetwork(fea_len, fea_len, cry_msg),
            )
            for _ in range(cry_heads)
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> Tensor:
        elem_fea = self.elem_embed(elem_fea)
        sym_fea = self.sym_embed(torch.cat([sym_fea, elem_weights], dim=1))

        elem_fea = torch.cat([elem_fea, sym_fea], dim=1)

        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_idx, nbr_idx)

        head_fea = [
            attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            for attnhead in self.cry_pool
        ]

        return scatter_reduce(
            torch.mean(torch.stack(head_fea), dim=0), aug_cry_idx, dim=0, reduce="mean"
        )


class Wren(nn.Module):
    """The Wren model is comprised of a fully connected network and message
    passing graph layers, operating on Wyckoff-position (coordinate-free)
    crystal-structure descriptors."""

    def __init__(
        self,
        robust: bool,
        n_targets: Sequence[int],
        elem_emb_len: int = 200,
        sym_emb_len: int = 444,
        elem_fea_len: int = 32,
        sym_fea_len: int = 32,
        n_graph: int = 3,
        elem_heads: int = 1,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 3,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
        trunk_hidden: Sequence[int] = (1024, 512),
        out_hidden: Sequence[int] = (256, 128, 64),
        n_elements: int = 104,
        n_wyckoff_rows: int = 1732,
    ) -> None:
        super().__init__()
        self.robust = robust

        # Real element/Wyckoff embeddings are `nn.Embedding` tables populated
        # from pretrained matscholar200 (103 elements, 200-dim) / bra-alg-off
        # (1732 rows, 444-dim) JSON feature files on disk; the architecture
        # is the plain `nn.Embedding` itself, initialized randomly here.
        self.elem_embedding = nn.Embedding(n_elements, elem_emb_len)
        self.sym_embedding = nn.Embedding(n_wyckoff_rows, sym_emb_len)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "sym_emb_len": sym_emb_len,
            "sym_fea_len": sym_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
        }

        self.material_nn = DescriptorNetwork(**desc_dict)

        if self.robust:
            n_targets = [2 * n for n in n_targets]

        self.trunk_nn = ResidualNetwork(elem_fea_len + sym_fea_len, out_hidden[0], trunk_hidden)

        self.output_nns = nn.ModuleList(
            ResidualNetwork(out_hidden[0], n, out_hidden[1:]) for n in n_targets
        )

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        sym_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        aug_cry_idx: LongTensor,
    ) -> tuple[Tensor, ...]:
        elem_fea = self.elem_embedding(elem_fea)
        sym_fea = self.sym_embedding(sym_fea)
        crys_fea = self.material_nn(
            elem_weights,
            elem_fea,
            sym_fea,
            self_idx,
            nbr_idx,
            cry_elem_idx,
            aug_cry_idx,
        )

        crys_fea = F.relu(self.trunk_nn(crys_fea))

        return tuple(output_nn(crys_fea) for output_nn in self.output_nns)


def build_wren() -> Wren:
    """Tiny random-init Wren model (shrunk feature/hidden dims, n_targets=(1,))."""
    torch.manual_seed(0)
    return Wren(
        robust=False,
        n_targets=(1,),
        elem_emb_len=200,
        sym_emb_len=444,
        elem_fea_len=16,
        sym_fea_len=16,
        n_graph=2,
        elem_heads=1,
        elem_gate=(32,),
        elem_msg=(32,),
        cry_heads=2,
        cry_gate=(32,),
        cry_msg=(32,),
        trunk_hidden=(64, 32),
        out_hidden=(32, 16),
    ).eval()


def example_input_wren() -> tuple[
    Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor
]:
    """A tiny synthetic 2-crystal batch: crystal 0 has 3 elements/Wyckoff sites,
    crystal 1 has 2; each crystal contributes one Wyckoff-augmentation copy
    (``aug_cry_idx`` == ``cry_elem_idx``'s crystal grouping), and the
    self/neighbor index pairs form the fully-connected element-pair graph
    used by Roost/Wren message passing (every element attends to every
    other element, including itself, within its own crystal)."""
    torch.manual_seed(0)
    # Two crystals: crystal 0 with 3 sites, crystal 1 with 2 sites -> 5 total.
    n_sites_per_crystal = [3, 2]
    n_sites = sum(n_sites_per_crystal)

    elem_weights = torch.rand(n_sites, 1)
    elem_fea = torch.randint(1, 100, (n_sites,), dtype=torch.long)
    sym_fea = torch.randint(0, 1732, (n_sites,), dtype=torch.long)

    self_idx: list[int] = []
    nbr_idx: list[int] = []
    cry_elem_idx: list[int] = []
    offset = 0
    for cry_i, n in enumerate(n_sites_per_crystal):
        for i in range(n):
            for j in range(n):
                self_idx.append(offset + i)
                nbr_idx.append(offset + j)
            cry_elem_idx.append(cry_i)
        offset += n

    aug_cry_idx = list(range(len(n_sites_per_crystal)))

    return (
        elem_weights,
        elem_fea,
        sym_fea,
        torch.tensor(self_idx, dtype=torch.long),
        torch.tensor(nbr_idx, dtype=torch.long),
        torch.tensor(cry_elem_idx, dtype=torch.long),
        torch.tensor(aug_cry_idx, dtype=torch.long),
    )


MENAGERIE_ENTRIES = [
    ("Wren materials", "build_wren", "example_input_wren", 2023, MENAGERIE_ZOO),
]
