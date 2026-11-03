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
from tests.classics_corpus._loader import build_entry, corpus_entries
from torchlens.backends.torch import _aten_capture
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


class _LinearModel(nn.Module):
    """One linear layer, for the intervention case."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer."""

        return self.linear(x)


def _row_keys(trace: Any) -> list[tuple[str, str, int]]:
    """Return the (operator, overload, sequence) key of every recorded row."""

    return [(row.operator, row.overload, row.sequence) for row in trace.aten_ops]


def _assert_refuses(read: Any) -> None:
    """Assert that reading an accessor raises the typed refusal naming the switch."""

    with pytest.raises(tl.errors.AtenOpsUnavailableError) as excinfo:
        read()
    assert excinfo.value.fields["code"] == "aten_ops_unavailable"
    assert "CaptureOptions(aten_ops=True)" in str(excinfo.value)


def test_switch_records_rows_and_each_op_serves_its_own() -> None:
    """``aten_ops=True`` records rows; ``Op.aten_ops`` partitions them by owner."""

    torch.manual_seed(0)
    trace = tl.trace(_AttentionNormModel(), torch.randn(2, 3, 8), capture=_ATEN)

    assert trace.has_aten_ops is True
    rows = trace.aten_ops
    assert isinstance(rows, tuple) and rows
    assert [row.sequence for row in rows] == sorted(row.sequence for row in rows)
    operators = {row.operator for row in rows}
    assert "native_layer_norm" in operators
    assert "gelu" in operators
    attention_rows = [
        row
        for op in trace.ops
        if "scaleddotproductattention" in op.layer_label
        for row in op.aten_ops
    ]
    assert attention_rows, "the attention op owns no ATen rows"
    assert trace.aten_gaps == ()

    owned = [row for op in trace.ops for row in op.aten_ops]
    assert owned
    assert len(owned) == len({id(row) for row in owned}), "a row has two owners"
    assert all(row.owner_status == "forward_op" for row in owned)
    layer_norm_ops = [op for op in trace.ops if op.func_name == "layer_norm"]
    assert layer_norm_ops
    assert {row.operator for row in layer_norm_ops[0].aten_ops} >= {"native_layer_norm"}
    assert check_metadata_invariants(trace)


def test_switch_off_refuses_typed_and_reports_no_rows() -> None:
    """The default capture records nothing and every accessor refuses typed."""

    trace = tl.trace(_TwoOpModel(), torch.ones(2))

    assert trace.has_aten_ops is False
    _assert_refuses(lambda: trace.aten_ops)
    _assert_refuses(lambda: trace.aten_gaps)
    _assert_refuses(lambda: trace.ops[0].aten_ops)
    assert trace._primitive_op_profile is None


def test_switch_off_installs_no_dispatch_path_and_on_counts_one_call_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic overhead counter: zero recorder calls off, one per row on."""

    calls: list[int] = []
    real_prepare = _aten_capture._prepare_aten_call

    def counting_prepare(*args: object, **kwargs: object) -> object:
        """Count every entry into the recorder's dispatch path."""

        calls.append(1)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(_aten_capture, "_prepare_aten_call", counting_prepare)
    torch.manual_seed(0)
    off = tl.trace(_TwoOpModel(), torch.ones(2))
    assert calls == []
    assert off.has_aten_ops is False

    on = tl.trace(_TwoOpModel(), torch.ones(2), capture=_ATEN)
    assert on.has_aten_ops is True
    assert len(calls) == len(on.aten_ops)
    assert len(on.aten_ops) >= 2


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


def test_loaded_artifact_saved_without_the_switch_refuses(tmp_path: Path) -> None:
    """An artifact saved from an off capture carries no profile and refuses typed."""

    trace = tl.trace(_TwoOpModel(), torch.ones(2))
    path = tmp_path / "plain.tlspec"
    trace.save(path)
    loaded = tl.load(path)

    assert loaded.has_aten_ops is False
    _assert_refuses(lambda: loaded.aten_ops)
    _assert_refuses(lambda: loaded.ops[0].aten_ops)


def test_refresh_inherits_the_switch_and_fast_run_refuses() -> None:
    """``run()`` refreshes keep the recording, the fast live tier included."""

    torch.manual_seed(0)
    model = _TwoOpModel()
    trace = tl.trace(model, torch.ones(2), capture=_ATEN)
    before = _row_keys(trace)

    refreshed = trace.run(model, torch.ones(2) * 2)
    assert refreshed.has_aten_ops is True
    assert _row_keys(refreshed) == before

    # The fast live tier refreshes the SAME live Trace in place (its result
    # trace is this trace), so the capture's recording stays, held to the
    # recorded path by the fast static guard.
    fast = trace.run(inputs=torch.ones(2) * 3, fast=True)
    assert fast.trace is trace
    assert fast.trace.has_aten_ops is True
    assert _row_keys(fast.trace) == before


def test_intervention_capture_with_the_switch_passes_invariants() -> None:
    """An in-flight intervention beside the recorder keeps every invariant."""

    torch.manual_seed(0)
    direction = torch.randn(4)
    trace = tl.trace(
        _LinearModel(),
        torch.randn(2, 4),
        intervene=tl.when(tl.func("linear"), tl.steer(direction, dim=1)),
        capture=_ATEN,
    )

    assert trace.has_aten_ops is True
    assert isinstance(trace.aten_gaps, tuple)
    assert check_metadata_invariants(trace)


@pytest.mark.smoke
def test_backward_rows_join_trace_aten_ops_in_order() -> None:
    """Backward rows append after the forward rows and keep their phase."""

    torch.manual_seed(0)
    trace = tl.trace(
        _TwoOpModel(),
        torch.ones(2, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, aten_ops=True),
        save_mode="reference",
    )
    forward_count = len(trace.aten_ops)
    trace.log_backward(trace.output_ops[0].out.sum())

    rows = trace.aten_ops
    assert len(rows) > forward_count
    assert all(row.capture_phase == "forward" for row in rows[:forward_count])
    assert all(row.capture_phase == "backward" for row in rows[forward_count:])
    assert check_metadata_invariants(trace)


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


def test_classics_smoke_model_records_and_validates() -> None:
    """One classics smoke model captured with the switch replays and validates."""

    model, inputs = build_entry(_SMOKE_ENTRY)
    trace = tl.trace(
        model,
        inputs,
        # save_args keeps the replay arguments; without it the per-op replay
        # reports the whole model as a region it cannot replay, switch or not.
        capture=tl.options.CaptureOptions(aten_ops=True, layers_to_save="all", save_args=True),
    )
    assert trace.has_aten_ops is True
    assert trace.aten_ops
    with torch.no_grad():
        expected = model(*inputs)
    assert trace.validate_forward_pass(expected, validate_metadata=True) is True
    assert all(isinstance(op.aten_ops, tuple) for op in trace.ops)
