"""The receptive- and projective-field walks handle models deeper than the recursion limit.

The per-unit walks map an index-set state one graph hop at a time. They used to recurse
once per hop, so a deep enough chain (a few hundred convolutions) raised ``RecursionError``
from the public ``.at()`` query and from ``trace.validate_forward_pass``, whose metadata
invariants run the same walks. The walks now keep pending hops on an explicit stack.

Closed form for the deep model: a chain of ``k`` stride-1, padding-1 convolutions of kernel
size 3 widens a unit's receptive field by 2 per layer, so the centre unit of a length
``2k + 101`` signal sees exactly ``2k + 1`` input positions, ``[c - k, c + k]``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _forward_query, _query
from torchlens.receptive_field._errors import ReceptiveFieldError

DEPTH = 600
LENGTH = 2 * DEPTH + 101
CENTER = LENGTH // 2


class _DeepConvChain(nn.Module):
    """``DEPTH`` stride-1 kernel-3 padding-1 one-channel convolutions in a row.

    Each kernel is ``[0.25, 0.5, 0.25]``, so activations stay on the input's scale at any depth
    and the forward replay compares ordinary finite values.
    """

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(1, 1, 3, padding=1, bias=False) for _ in range(depth))
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.tensor([[[0.25, 0.5, 0.25]]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Residual(nn.Module):
    """Two conv branches merged by addition, then a pool: a merge the memo collapses."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(2, 2, 3, padding=1)
        self.b = nn.Conv2d(2, 2, 5, padding=2)
        self.c = nn.Conv2d(2, 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a(x)
        y = y + self.b(x)
        y = torch.relu(self.c(y) + self.c(x))
        return nn.functional.max_pool2d(y, 2)


class _Concat(nn.Module):
    """A channel concatenation of a dilated and a strided branch."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(1, 2, 3, padding=2, dilation=2)
        self.b = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.a(x), self.b(x)], dim=1)
        return nn.functional.avg_pool2d(self.b(x) * y[:, :2], 3, stride=1, padding=1)


class _Diamond(nn.Module):
    """Repeated diamonds: each level forks and rejoins, so path counts double per level."""

    def __init__(self, levels: int) -> None:
        super().__init__()
        self.left = nn.ModuleList(nn.Conv1d(1, 1, 3, padding=1) for _ in range(levels))
        self.right = nn.ModuleList(nn.Conv1d(1, 1, 5, padding=2) for _ in range(levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for left, right in zip(self.left, self.right, strict=True):
            x = left(x) + right(x)
        return x


def _final_conv(trace: tl.Trace) -> tl.Op:
    return [op for op in trace.layer_list if op.func_name == "conv1d"][-1]


def test_deep_chain_receptive_field_matches_closed_form() -> None:
    """The public ``.at()`` query on a 600-deep chain returns ``2k + 1`` positions."""

    torch.manual_seed(0)
    trace = tl.trace(_DeepConvChain(DEPTH), torch.randn(1, 1, LENGTH))
    box = _final_conv(trace).receptive_field.at((CENTER,))
    axis = box.axes[-1]
    assert (axis.index_start, axis.index_stop) == (CENTER - DEPTH, CENTER + DEPTH + 1)
    assert axis.index_stop - axis.index_start == 2 * DEPTH + 1
    assert box.exact


def test_deep_chain_projective_field_matches_closed_form() -> None:
    """The projective walk on the same chain: input unit ``c`` reaches ``[c - k, c + k]``."""

    torch.manual_seed(0)
    trace = tl.trace(_DeepConvChain(DEPTH), torch.randn(1, 1, LENGTH))
    source = trace.input_ops[0]
    box = source.projective_field.at((CENTER,), target=_final_conv(trace))
    axis = box.axes[-1]
    assert (axis.index_start, axis.index_stop) == (CENTER - DEPTH, CENTER + DEPTH + 1)


@pytest.mark.slow
def test_deep_chain_validate_forward_pass_completes() -> None:
    """``validate_forward_pass`` runs the receptive-field invariants without RecursionError.

    Slow (about three minutes): the receptive-field metadata invariants query every op's
    unit box, and on a 600-deep chain that is the cost of the check, not of the walk method.
    """

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        passed = tl.validation.validate_forward_pass(
            _DeepConvChain(DEPTH), torch.randn(1, 1, LENGTH), seed=0
        )
    failure = tl.validation.last_validation_failure()
    assert passed is True, failure.summary() if failure is not None else "no failure record"


def _recursive_reference(
    body: Callable[[tl.Op, _query._AxisSets, bool], _query._WalkStep],
    op: tl.Op,
    sets: _query._AxisSets,
    exact: bool,
    memo: dict[_query._WalkKey, tuple[_query._TerminalState, ...]],
) -> tuple[_query._TerminalState, ...]:
    """The former recursive driver, run over the same hop bodies."""

    key = (op.label, sets, exact)
    cached = memo.get(key)
    if cached is not None:
        return cached
    step = body(op, sets, exact)
    sent: tuple[_query._TerminalState, ...] | None = None
    while True:
        try:
            request = next(step) if sent is None else step.send(sent)
        except StopIteration as finished:
            memo[key] = finished.value
            return finished.value
        sent = _recursive_reference(body, *request, memo)


def _shallow_models() -> list[tuple[nn.Module, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        (_Residual(), torch.randn(1, 2, 9, 9)),
        (_Concat(), torch.randn(1, 1, 8, 8)),
        (_Diamond(6), torch.randn(1, 1, 24)),
        (_DeepConvChain(8), torch.randn(1, 1, 30)),
    ]


@pytest.mark.parametrize("index", range(4))
def test_explicit_stack_walk_equals_recursive_walk(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every receptive and projective box query matches the recursion: same terminals in the
    same order and the same memo keys in the same insertion order."""

    model, x = _shallow_models()[index]
    trace = tl.trace(model, x)
    calls: list[int] = []
    iterative = _query._run_memoized_walk

    def compare(body, op, sets, exact, memo):  # type: ignore[no-untyped-def]
        reference_memo = dict(memo)
        expected = _recursive_reference(body, op, sets, exact, reference_memo)
        actual = iterative(body, op, sets, exact, memo)
        assert actual == expected
        assert list(memo.items()) == list(reference_memo.items())
        calls.append(len(memo))
        return actual

    monkeypatch.setattr(_query, "_run_memoized_walk", compare)
    monkeypatch.setattr(_forward_query, "_run_memoized_walk", compare)
    for op in trace.layer_list:
        if op.func_name not in {"conv1d", "conv2d", "max_pool2d", "avg_pool2d", "__add__"}:
            continue
        center = tuple(int(size) // 2 for size in op.shape[2:])
        try:
            op.receptive_field.at(center)
        except ReceptiveFieldError:
            continue  # no per-unit geometry for this op; the walk never runs
    for target in trace.output_ops:
        source = trace.input_ops[0]
        center = tuple(int(size) // 2 for size in source.shape[2:])
        try:
            source.projective_field.at(center, target=target)
        except ReceptiveFieldError:
            continue  # no per-unit projective geometry to this target; the walk never runs
    assert len(calls) >= 3
    assert any(size > 1 for size in calls)
