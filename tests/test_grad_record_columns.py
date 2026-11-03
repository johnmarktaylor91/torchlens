"""Persisted gradient records cost opcodes per op, not per record (columnar ``_grad_records``).

A portable save pickles each op's ``GradientRecord`` list as one columnar table, so the
``metadata.pkl`` opcode count no longer grows with ops times backward passes (a Mamba-130m trace
with ~96 Jacobian passes passed the loader's opcode ceiling under the per-record encoding).
"""

from __future__ import annotations

import pickle
import pickletools
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


def _mlp() -> nn.Module:
    """Return a seeded three-layer MLP."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))


def _trace_with_passes(num_passes: int) -> tl.Trace:
    """Capture the MLP with saved grads and run ``num_passes`` backward passes over it."""

    x = torch.randn(1, 8, requires_grad=True)
    trace = tl.trace(_mlp(), x, save="all", capture=CaptureOptions(save_grads="all"))
    out = trace[trace.output_layers[0]].out
    for _ in range(num_passes):
        torch.autograd.grad(out.sum(), x, retain_graph=True)
    return trace


def _metadata_opcodes(trace: tl.Trace, path: Path) -> int:
    """Save ``trace`` portable at ``path`` and count the opcodes of its ``metadata.pkl``."""

    trace.save(path, level="portable")
    return sum(1 for _ in pickletools.genops((path / "metadata.pkl").read_bytes()))


def _records(op: object) -> list:
    """Return an op's gradient records (an absent list counts as empty)."""

    return list(getattr(op, "_grad_records", None) or [])


def _record_opcodes(trace: tl.Trace, directory: Path) -> tuple[int, int]:
    """Return the opcodes the gradient records cost in ``metadata.pkl``, and their count.

    The trace is saved with its records and again after clearing them; every other
    per-pass fact (backward passes, grad_fn calls) is in both saves, so the difference is
    the encoding of the records alone.
    """

    ops = list(trace)
    num_records = sum(len(_records(op)) for op in ops)
    with_records = _metadata_opcodes(trace, directory / "with.tlspec")
    for op in ops:
        op._clear_gradient_records()
    without_records = _metadata_opcodes(trace, directory / "without.tlspec")
    return with_records - without_records, num_records


def test_grad_record_opcodes_scale_by_op_not_by_record(tmp_path: Path) -> None:
    """Thirty-eight more backward passes add bytes to the records, not opcodes per record."""

    few, many = _trace_with_passes(2), _trace_with_passes(40)
    num_ops = len(list(many))
    (tmp_path / "few").mkdir()
    (tmp_path / "many").mkdir()
    few_cost, few_records = _record_opcodes(few, tmp_path / "few")
    many_cost, many_records = _record_opcodes(many, tmp_path / "many")
    assert many_records - few_records >= 38 * 4, "the passes must reach the ops"
    assert many_cost - few_cost < 4 * num_ops, (
        f"gradient-record opcodes grew {many_cost - few_cost} ({few_cost} -> {many_cost}) for "
        f"{many_records - few_records} more records over {num_ops} ops"
    )


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_grad_records_round_trip(tmp_path: Path, lazy: bool) -> None:
    """Every record's facts and saved gradient survive a portable save and load."""

    trace = _trace_with_passes(3)
    path = tmp_path / "trace.tlspec"
    trace.save(path, level="portable")
    loaded = tl.load(path, lazy=lazy)
    live_ops, loaded_ops = list(trace), list(loaded)
    assert [op.label for op in loaded_ops] == [op.label for op in live_ops]
    saved_grads = 0
    for live_op, loaded_op in zip(live_ops, loaded_ops, strict=True):
        live_records, loaded_records = _records(live_op), _records(loaded_op)
        assert len(loaded_records) == len(live_records), live_op.label
        for live, record in zip(live_records, loaded_records, strict=True):
            assert record.owner is loaded_op
            assert (record.ordinal, record.backward_pass_number) == (
                live.ordinal,
                live.backward_pass_number,
            )
            assert (record.shape, record.dtype, record.timestamp) == (
                live.shape,
                live.dtype,
                live.timestamp,
            )
            assert int(record.memory) == int(live.memory)
            assert type(record.memory) is type(live.memory)
            if live.grad is not None:
                saved_grads += 1
                assert torch.equal(record.grad, live.grad), live_op.label
        for live in live_records:
            pass_index = live.backward_pass_number
            assert loaded_op.grads.for_pass(pass_index).ordinal == live.ordinal
    assert saved_grads > 0, "the round trip must cover saved gradient payloads"


def test_op_without_records_keeps_the_empty_list(tmp_path: Path) -> None:
    """An op no backward pass reached persists ``[]``, byte-identical to before."""

    from torchlens._io._grad_record_columns import (
        pack_gradient_records,
        reduce_with_grad_record_columns,
    )

    trace = tl.trace(_mlp(), torch.randn(1, 8), save="all")
    op = list(trace)[0]
    assert _records(op) == []
    assert pack_gradient_records([]) is None
    assert reduce_with_grad_record_columns(op, type(op), pickle.HIGHEST_PROTOCOL) is (
        NotImplemented
    )
    path = tmp_path / "plain.tlspec"
    trace.save(path, level="portable")
    assert b"GradientRecordColumns" not in (path / "metadata.pkl").read_bytes()
    assert all(_records(loaded_op) == [] for loaded_op in tl.load(path))
