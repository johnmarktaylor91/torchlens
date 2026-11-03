"""``backward()`` called inside forward is a recorded, faithfully validated boundary op.

The call becomes one ``autogradbackward`` op per leaf gradient the engine pass writes
(``transform_kind="autograd.backward"``, ``is_transform=True``), parented on the recorded
roots and, for a leaf that already held a grad, on that previous grad as the in-place
receiver. The engine's interior dispatches (including the accumulation ``add_`` of a
retained-graph second pass) belong to the innermost op, so the dispatch census balances,
and a later op reading ``param.grad`` has a producer. Validation re-derives each gradient
by replaying the recorded subgraph from the leaf to the roots and adding the previous
grad, never by exemption: a tampered saved gradient or a forged func fails.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.validation.core as validation_core
from torchlens._state import pause_logging
from torchlens.validation import last_validation_failure


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward"))


def _backward_ops(trace: Any) -> list[Any]:
    return [op for op in trace.ops if op.func_name == "autogradbackward"]


def _engine_flags(op: Any) -> dict[str, Any]:
    return {k: v for k, v in op.transform_config.items() if not k.startswith("_tl")}


@contextmanager
def _tampered_validation(tamper: Callable[[Any], None]) -> Iterator[list[Any]]:
    """Run ``tamper`` on the validator's own capture before its checks run."""

    seen: list[Any] = []
    original = validation_core.validate_saved_outs

    def tampering(trace: Any, *args: Any, **kwargs: Any) -> Any:
        tamper(trace)
        seen.append(trace)
        return original(trace, *args, **kwargs)

    validation_core.validate_saved_outs = tampering
    try:
        yield seen
    finally:
        validation_core.validate_saved_outs = original


class _GradReader(nn.Module):
    """Runs one backward inside forward and consumes the written parameter grad."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def _reset(self) -> None:
        for param in self.fc1.parameters():
            param.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._reset()
        loss = torch.tanh(self.fc1(x.detach())).sum()
        loss.backward()
        grad = self.fc1.weight.grad
        return self.fc2(x) * grad.mean()


class _RetainTwice(_GradReader):
    """Two backward passes through a retained graph: the second accumulates in place."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._reset()
        loss = torch.tanh(self.fc1(x.detach())).sum()
        loss.backward(retain_graph=True)
        loss.backward()
        grad = self.fc1.weight.grad
        return self.fc2(x) * grad.mean()


class _RestrictedInputs(_GradReader):
    """``inputs=`` restricts the pass to one leaf; only that leaf gets an op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._reset()
        loss = torch.tanh(self.fc1(x.detach())).sum()
        torch.autograd.backward(loss, inputs=[self.fc1.bias])
        return self.fc2(x) * self.fc1.bias.grad.sum()


class _UnseenBackward(nn.Module):
    """A backward whose roots the capture never saw runs unrecorded, as before."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with pause_logging():
            leaf = torch.randn(4, requires_grad=True)
            (leaf * 2).sum().backward()
        return self.fc(x)


@pytest.mark.smoke
def test_backward_in_forward_is_a_recorded_boundary_op() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_GradReader(), torch.rand(2, 4))
    ops = _backward_ops(trace)
    assert len(ops) == 2  # fc1.weight and fc1.bias
    for op in ops:
        assert op.is_transform is True
        assert op.transform_kind == "autograd.backward"
        assert not _engine_flags(op).get("retain_graph")
        assert op.is_inplace is False
        assert len(op.parents) == 1 and op.parents[0].startswith("sum")
        assert not op.unattributed_tensor_args
    mean_ops = [op for op in trace.ops if op.func_name == "mean"]
    assert len(mean_ops) == 1
    assert set(mean_ops[0].parents) <= {op.label for op in ops}
    assert not mean_ops[0].unattributed_tensor_args
    assert [op.label for op in trace.transforms] == [op.label for op in ops]
    selected = tl.trace(
        _GradReader(), torch.rand(2, 4), save=tl.func_transform("autograd.backward")
    )
    assert len(selected.transforms) == 2


def test_second_backward_through_a_retained_graph_accumulates_in_place() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_RetainTwice(), torch.rand(2, 4))
    ops = _backward_ops(trace)
    assert len(ops) == 4
    first, second = ops[:2], ops[2:]
    first_labels = {op.label for op in first}
    for op in first:
        assert op.is_inplace is False
        assert _engine_flags(op).get("retain_graph") is True
    for op in second:
        assert op.is_inplace is True
        assert not _engine_flags(op).get("retain_graph")
        assert len(op.parents) == 2
        assert len(set(op.parents) & first_labels) == 1
    mean_ops = [op for op in trace.ops if op.func_name == "mean"]
    assert set(mean_ops[0].parents) <= {op.label for op in second}


def test_inputs_restriction_records_only_the_written_leaf() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_RestrictedInputs(), torch.rand(2, 4))
    ops = _backward_ops(trace)
    assert len(ops) == 1
    assert ops[0].shape == (4,)


def test_backward_on_unseen_roots_is_not_recorded() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_UnseenBackward(), torch.rand(2, 4))
    assert _backward_ops(trace) == []


@pytest.mark.parametrize(
    "model_factory",
    [_GradReader, _RetainTwice, _RestrictedInputs],
    ids=["single_pass", "retain_twice", "restricted_inputs"],
)
def test_recorded_backward_validates_by_subgraph_replay(
    model_factory: Callable[[], nn.Module],
) -> None:
    results: list[tuple[str, Any]] = []

    def check(trace: Any) -> None:
        for op in _backward_ops(trace):
            plain = validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                trace, op.label
            )
            results.append((op.label, plain))
            for parent in op.parents:
                perturbed = (
                    validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                        trace, op.label, perturb=True, layers_to_perturb=[parent]
                    )
                )
                results.append((f"{op.label}<-{parent}", perturbed))

    torch.manual_seed(0)
    with _tampered_validation(check):
        verdict = _validate(model_factory(), torch.randn(2, 4))
    assert verdict is True, last_validation_failure()
    assert last_validation_failure() is None
    assert results
    assert {label: r.decision for label, r in results} == {
        label: "validated" for label, _ in results
    }


def test_tampered_saved_backward_gradient_fails_validation() -> None:
    results: list[Any] = []

    def tamper(trace: Any) -> None:
        op = _backward_ops(trace)[-1]
        op._slot("out").add_(0.5)
        results.append(
            validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                trace, op.label
            )
        )

    torch.manual_seed(0)
    with _tampered_validation(tamper):
        assert _validate(_RetainTwice(), torch.randn(2, 4)) is False
    assert len(results) == 1
    assert results[0].decision == "failed" and results[0].reason == "replay_mismatch"


def test_forged_backward_func_fails_replay() -> None:
    def forge(trace: Any) -> None:
        def autogradbackward(accumulate_into: Any, tensors: Any, *_a: Any, **_k: Any) -> Any:
            return torch.zeros(4, 4)

        op = _backward_ops(trace)[0]
        op.func = autogradbackward
        assert op.transform_kind == "autograd.backward"

    torch.manual_seed(0)
    with _tampered_validation(forge):
        assert _validate(_GradReader(), torch.randn(2, 4)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "forward_replay"
    assert failure.op_label is not None and failure.op_label.startswith("autogradbackward")
