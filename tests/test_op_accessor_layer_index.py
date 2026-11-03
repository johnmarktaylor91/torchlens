"""Bare-layer-label Op reads are served from the alias index, not a row scan.

``GradFn.op`` resolves its bare layer label through ``trace.ops.resolve_all`` once
per op on every backward pass. When that read scanned every row, a Jacobian fit
on Mamba-130m (about 24k ops, 96 passes per prompt) took six times longer than
the same fit with an indexed read. The gate is deterministic: after one warming
read, the row list refuses iteration and the reads must still answer.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _NoScan(list):
    """A row list that fails the test the moment anything iterates it."""

    def __iter__(self) -> Iterator:  # pragma: no cover - the failure path
        raise AssertionError("resolve_all scanned the op rows instead of using the index")


class _Twice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.relu(self.lin(x)))


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    x = torch.randn(2, 4, requires_grad=True)
    return tl.trace(_Twice().eval(), x, save="all", capture=CaptureOptions(replayable=True))


def test_resolve_all_bare_layer_label_reads_the_index() -> None:
    trace = _trace()
    ops = trace.ops
    multi_pass = next(
        op for op in trace._layer_list if trace._layer_num_calls.get(op.layer_label, 0) > 1
    )
    single_pass = next(
        op for op in trace._layer_list if trace._layer_num_calls.get(op.layer_label, 0) == 1
    )
    expected_multi = ops.resolve_all(multi_pass.layer_label)
    expected_single = ops.resolve_all(single_pass.layer_label)
    assert len(expected_multi) == 2 and expected_single == [single_pass]

    ops._list = _NoScan(ops._list)
    assert ops.resolve_all(multi_pass.layer_label) == expected_multi
    assert ops.resolve_all(multi_pass.short_layer_label) == expected_multi
    assert ops.resolve_all(single_pass.layer_label) == expected_single
    assert ops.resolve_all(f"{multi_pass.layer_label}:last") == [expected_multi[-1]]
    assert ops.resolve_all("no_such_layer_9") == []


def test_resolve_all_falls_back_to_the_scan_when_the_index_is_unavailable() -> None:
    trace = _trace()
    ops = trace.ops
    multi_pass = next(
        op for op in trace._layer_list if trace._layer_num_calls.get(op.layer_label, 0) > 1
    )
    expected = ops.resolve_all(multi_pass.layer_label)
    ops._alias_indexes = None
    ops._alias_index_unavailable = True
    assert ops.resolve_all(multi_pass.layer_label) == expected


def test_grad_fn_op_reads_do_not_scan_the_rows() -> None:
    trace = _trace()
    out = trace[trace.output_layers[0]].out
    source = next(op for op in trace._layer_list if not op.is_input).out
    torch.autograd.grad(out.sum(), source, retain_graph=True)
    grad_fns = [gf for gf in trace.autograd.grad_fns if gf.op_label is not None]
    if not grad_fns:
        pytest.skip("no GradFn carries an op label in this capture")
    resolved = [gf.op for gf in grad_fns]
    trace.ops._list = _NoScan(trace.ops._list)
    assert [gf.op for gf in grad_fns] == resolved
