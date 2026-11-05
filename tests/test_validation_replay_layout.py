"""Forward replay feeds each parent value in the memory layout the captured call saw.

Saved outs are compact copies, so a parent that reached its child as a strided view
(one half of a ``chunk``, a column slice) used to be replayed from a row-major
tensor. Kernels choose their path by layout: float32 ``gelu`` runs the oneDNN
kernel only on a contiguous input, a ``Linear`` over a strided column runs
``matmul`` plus ``add`` instead of a fused ``addmm``, and BLAS ``gemv`` sums a
strided vector in another order. The replay then missed the saved out by a
rounding step and validation failed. Validation compares exactly as before; the
replay is what changed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.intervention.types import CapturedArgTemplate, ParentRef
from torchlens.validation._replay_storage_context import relay_to_captured_layout


class _GEGLU(nn.Module):
    """Gated GELU feed-forward: ``gelu`` reads the second half of a ``chunk``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 2048)
        with torch.no_grad():
            self.proj.weight.normal_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, split in two halves and gate one by ``gelu`` of the other."""

        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class _ColumnEmbedding(nn.Module):
    """``Linear(1, k)`` over one strided feature column, as time-series embedders do."""

    def __init__(self) -> None:
        super().__init__()
        self.to_emb = nn.Linear(1, 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed feature column 5 of every step."""

        return self.to_emb(x[:, :, 5].unsqueeze(2))


class _StridedVectorSolve(nn.Module):
    """A float64 ``matrix @ y[:, 0]`` whose terms cancel, as a least-squares fit does."""

    def __init__(self) -> None:
        super().__init__()
        self.basis = nn.Parameter(torch.randn(4, 37, dtype=torch.float64))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Project the first column of ``xy`` on the basis rows."""

        return self.basis @ xy[:, 0]


def _orthogonal_columns(model: _StridedVectorSolve) -> torch.Tensor:
    """Return an ``(37, 2)`` input whose first column is orthogonal to the basis rows."""

    basis = model.basis.detach()
    column = torch.randn(37, dtype=torch.float64)
    column = column - basis.T @ torch.linalg.lstsq(basis.T, column).solution
    return torch.stack((column, torch.randn(37, dtype=torch.float64)), dim=1)


def test_gelu_on_chunk_half_validates() -> None:
    torch.manual_seed(0)
    model = _GEGLU()
    x = torch.randn(1, 256, 16)
    assert tl.validate(model, x, scope="forward", validate_metadata=True)


def test_linear_on_strided_column_validates() -> None:
    torch.manual_seed(0)
    model = _ColumnEmbedding()
    x = torch.randn(2, 65, 6)
    assert tl.validate(model, x, scope="forward", validate_metadata=True)


def test_matrix_times_strided_vector_validates() -> None:
    torch.manual_seed(0)
    model = _StridedVectorSolve()
    xy = _orthogonal_columns(model)
    assert tl.validate(model, xy, scope="forward", validate_metadata=True)


class _Layer:
    """Stand-in op carrying only an argument template."""

    def __init__(self, template: CapturedArgTemplate | None) -> None:
        self.args_template = template


def _strided_template() -> tuple[CapturedArgTemplate, torch.Tensor]:
    """Return a template recording a chunk-half layout for args[0] and kwargs['other']."""

    view = torch.randn(3, 8).chunk(2, dim=-1)[1]
    layout = (tuple(view.shape), tuple(view.stride()), view.storage_offset())
    ref = ParentRef("chunk_1_1", _capture_layout=layout)
    return CapturedArgTemplate(args=(ref, (ref,)), kwargs=(("other", ref),)), view


def test_relay_places_value_in_recorded_layout() -> None:
    template, view = _strided_template()
    layer = _Layer(template)
    value = view.contiguous()
    for arg_type, key in (("args", 0), ("args", (1, 0)), ("kwargs", "other")):
        relayed = relay_to_captured_layout(layer, arg_type, key, value)
        assert relayed.stride() == view.stride()
        assert relayed.storage_offset() == view.storage_offset()
        assert torch.equal(relayed, value)


def test_relay_keeps_grad_flags() -> None:
    template, view = _strided_template()
    layer = _Layer(template)
    leaf = view.contiguous().requires_grad_(True)
    relayed_leaf = relay_to_captured_layout(layer, "args", 0, leaf)
    assert relayed_leaf.requires_grad and relayed_leaf.is_leaf
    assert relayed_leaf.stride() == view.stride()
    non_leaf = leaf * 1
    relayed = relay_to_captured_layout(layer, "args", 0, non_leaf)
    assert relayed.requires_grad and not relayed.is_leaf
    assert relayed.stride() == view.stride()
    assert torch.equal(relayed.detach(), non_leaf.detach())
    relayed.add_(1.0)  # an in-place op legal on the captured non-leaf stays legal


def test_relay_leaves_unrecorded_slots_alone() -> None:
    template, view = _strided_template()
    value = view.contiguous()
    assert relay_to_captured_layout(_Layer(None), "args", 0, value) is value
    assert relay_to_captured_layout(_Layer(template), "args", 5, value) is value
    assert relay_to_captured_layout(_Layer(template), "kwargs", "missing", value) is value
    reshaped = torch.randn(4, 4)
    assert relay_to_captured_layout(_Layer(template), "args", 0, reshaped) is reshaped
