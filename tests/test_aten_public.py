"""Public ATen operation recording: ``CaptureOptions(aten_ops=True)`` and its accessors.

The switch is the only arm. ``Trace.has_aten_ops`` reports whether rows exist,
``Trace.aten_ops`` / ``Op.aten_ops`` serve them, ``Trace.aten_gaps`` discloses
interiors the recorder could not observe, and every accessor refuses typed
(``AtenOpsUnavailableError``, code ``aten_ops_unavailable``) when the capture
recorded none. Internals (row shapes, FKs, mutation lineage) stay in
``tests/test_aten_profile.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.classics_corpus._loader import corpus_entries
from torchlens.validation.invariants import check_metadata_invariants

_ATEN = tl.options.CaptureOptions(aten_ops=True)


class _AttentionNormModel(nn.Module):
    """layer_norm, then scaled dot-product attention, then gelu."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, attend the sequence to itself, and activate."""

        h = self.norm(x)
        attended = torch.nn.functional.scaled_dot_product_attention(h, h, h)
        return torch.nn.functional.gelu(attended)


class _TwoOpModel(nn.Module):
    """Add then relu."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run add followed by relu."""

        return torch.relu(x + 1)


def _row_keys(trace: Any) -> list[tuple[str, str, int]]:
    """Return the (operator, overload, sequence) key of every recorded row."""

    return [(row.operator, row.overload, row.sequence) for row in trace.aten_ops]


def _assert_refuses(read: Any) -> None:
    """Assert that reading an accessor raises the typed refusal naming the switch."""

    with pytest.raises(tl.errors.AtenOpsUnavailableError) as excinfo:
        read()
    assert excinfo.value.fields["code"] == "aten_ops_unavailable"
    assert "CaptureOptions(aten_ops=True)" in str(excinfo.value)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_op_aten.py, _data_classes/_trace_aten.py, _data_classes/_trace_rehydrate.py ....
def test_save_and_load_keep_rows_and_has_aten_ops(tmp_path: Path) -> None:
    """A plain save/load keeps every row key and the per-op partition."""

    torch.manual_seed(0)
    trace = tl.trace(_AttentionNormModel(), torch.randn(2, 3, 8), capture=_ATEN)
    path = tmp_path / "aten.tlspec"
    trace.save(path)
    loaded = tl.load(path)

    assert loaded.has_aten_ops is True
    assert _row_keys(loaded) == _row_keys(trace)
    assert len(loaded.aten_gaps) == len(trace.aten_gaps)
    for op in trace.ops:
        assert [row.sequence for row in loaded[op.layer_label].aten_ops] == [
            row.sequence for row in op.aten_ops
        ]
    assert check_metadata_invariants(trace)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_op_aten.py, _data_classes/_trace_aten.py, _errors.py.
def test_loaded_artifact_saved_without_the_switch_refuses(tmp_path: Path) -> None:
    """An artifact saved from an off capture carries no profile and refuses typed."""

    trace = tl.trace(_TwoOpModel(), torch.ones(2))
    path = tmp_path / "plain.tlspec"
    trace.save(path)
    loaded = tl.load(path)

    assert loaded.has_aten_ops is False
    _assert_refuses(lambda: loaded.aten_ops)
    _assert_refuses(lambda: loaded.ops[0].aten_ops)


@pytest.mark.smoke
def test_backward_rows_join_trace_aten_ops_in_order() -> None:
    """Backward rows append after the forward rows and keep their phase."""

    torch.manual_seed(0)
    trace = tl.trace(
        _TwoOpModel(),
        torch.ones(2, requires_grad=True),
        capture=tl.options.CaptureOptions(
            backward_ready=True, aten_ops=True, save_mode="reference"
        ),
    )
    forward_count = len(trace.aten_ops)
    trace.backward(trace.output_ops[0].out.sum())

    rows = trace.aten_ops
    assert len(rows) > forward_count
    assert all(row.capture_phase == "forward" for row in rows[:forward_count])
    assert all(row.capture_phase == "backward" for row in rows[forward_count:])
    assert check_metadata_invariants(trace)


@pytest.mark.rare
@pytest.mark.backend_jax
def test_non_torch_backend_records_nothing_and_refuses() -> None:
    """A non-torch backend ignores the switch: no profile, typed refusal, invariants hold."""

    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def model(x: Any) -> Any:
        """Square and sum."""

        return jnp.sum(x * x)

    trace = tl.trace(model, jnp.ones((3,)), backend="jax", capture=_ATEN)
    assert trace.has_aten_ops is False
    _assert_refuses(lambda: trace.aten_ops)
    assert check_metadata_invariants(trace)


_SMOKE_ENTRY = next(
    entry
    for entry in corpus_entries("smoke")
    if entry.id == "Avalanche multi-task simple MLP compact"
)
