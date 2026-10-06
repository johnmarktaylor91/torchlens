"""``torch.autograd.grad`` called inside forward is a recorded, faithfully validated boundary op.

The call becomes one ``autogradgrad`` op per returned gradient
(``transform_kind="autograd.grad"``, ``is_transform=True``), parented on the recorded
tensors among its ``outputs``/``inputs``/``grad_outputs``. Its saved ``outputs`` are
detached snapshots, so validation cannot re-call ``torch.autograd.grad`` on them; it
replays the recorded forward subgraph from fresh leaves holding the saved ``inputs``
values to the ``outputs`` roots, calls ``torch.autograd.grad`` on that replay with the
recorded ``grad_outputs`` and flags, and compares to the saved gradients. These tests
pin both sides: recorded grads validate (MAML, intermediate inputs, ``grad_outputs``,
nested second-order grads), and a forged func, forged parents, a tampered saved
gradient, or an unrecorded tensor in the same model all fail.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from example_models import MAMLInnerLoop
from torch import nn

import torchlens as tl
import torchlens.validation.core as validation_core
from torchlens.validation import last_validation_failure

_GLOBAL_OFFSET = torch.randn(3)


def _validate(model: nn.Module, *inputs: torch.Tensor) -> bool:
    """Validate forward, silencing the capture's own warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, inputs if len(inputs) > 1 else inputs[0], scope="forward"))


def _grad_ops(trace: Any) -> list[Any]:
    return [op for op in trace.ops if op.func_name == "autogradgrad"]


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


class _TwoLosses(nn.Module):
    """Differentiates one of two scalar losses with respect to its parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 3))
        self.v = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x @ self.w) + self.v
        first = (h**2).sum()
        second = (h**3).sum()
        grads = torch.autograd.grad(first, [self.w, self.v], create_graph=True)
        return h * second + grads[0].sum() + grads[1].sum()


class _IntermediateInput(nn.Module):
    """Grad of a non-scalar output with respect to an intermediate, with grad_outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        y = torch.sin(h) * h
        weights = torch.linspace(0.5, 1.5, y.shape[-1]).expand_as(y)
        (dh,) = torch.autograd.grad(y, h, grad_outputs=weights, retain_graph=True)
        return y + dh


class _SecondOrder(nn.Module):
    """A grad whose subgraph contains another recorded grad (grad of grad)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 3) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = torch.tanh(x @ self.w).pow(2).sum()
        (g,) = torch.autograd.grad(loss, [self.w], create_graph=True)
        penalty = (g**2).sum()
        (g2,) = torch.autograd.grad(penalty, [self.w], create_graph=True)
        return x @ (self.w - 0.1 * g2)


class _TwoCallsDifferentArity(nn.Module):
    """Two grad calls with different numbers of inputs (no name-cached ArgSpec)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(3))
        self.b = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = ((x * self.a).sin() * self.b).sum()
        (ga,) = torch.autograd.grad(loss, [self.a], create_graph=True)
        h = x * ga
        loss2 = (h.cos() * self.b).sum()
        gh, gb = torch.autograd.grad(loss2, [h, self.b], create_graph=True)
        return gh + gb + h


class _GradWithGlobal(nn.Module):
    """A recorded grad in a model that also reads an unrecorded module-global tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x * self.w).pow(2).sum()
        (g,) = torch.autograd.grad(loss, [self.w], create_graph=True)
        return (self.w - g) * x + _GLOBAL_OFFSET


def test_autograd_grad_in_forward_is_a_recorded_boundary_op() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(MAMLInnerLoop(), torch.rand(4, 8))
    assert not [w for w in caught if "no graph/source provenance" in str(w.message)]
    grad_ops = _grad_ops(trace)
    assert len(grad_ops) == 4
    for index, op in enumerate(grad_ops):
        assert op.is_transform is True
        assert op.transform_kind == "autograd.grad"
        engine_flags = {k: v for k, v in op.transform_config.items() if not k.startswith("_tl")}
        assert engine_flags == {"create_graph": True}
        assert op.multi_output_index == index
        assert len(op.parents) == 1 and op.parents[0].startswith("mean")
        assert not op.unattributed_tensor_args
    transform_labels = [op.label for op in trace.transforms]
    assert transform_labels == [op.label for op in grad_ops]
    selected = tl.trace(MAMLInnerLoop(), torch.rand(4, 8), save=tl.func_transform("autograd.grad"))
    assert [op.label for op in selected.transforms] == transform_labels


def test_autograd_grad_recorder_runs_once_and_holds_no_live_call() -> None:
    torch.manual_seed(0)
    trace = tl.trace(MAMLInnerLoop(), torch.rand(4, 8))
    recorder = _grad_ops(trace)[0].func
    with pytest.raises(RuntimeError, match="runs once"):
        recorder()


@pytest.mark.parametrize(
    ("model_factory", "shape"),
    [
        (MAMLInnerLoop, (4, 8)),
        (_TwoLosses, (2, 3)),
        (_IntermediateInput, (2, 4)),
    ],
    ids=["maml", "two_losses", "intermediate_input_grad_outputs"],
)
def test_recorded_autograd_grad_validates_by_subgraph_replay(
    model_factory: Callable[[], nn.Module], shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    assert _validate(model_factory(), torch.randn(*shape)) is True


@pytest.mark.parametrize(
    ("model_factory", "shape"),
    [(_SecondOrder, (2, 3)), (_TwoCallsDifferentArity, (3,))],
    ids=["second_order", "two_arities"],
)
def test_nested_and_dependent_input_grads_replay_faithfully(
    model_factory: Callable[[], nn.Module], shape: tuple[int, ...]
) -> None:
    """A grad inside another grad's subgraph, and an input that depends on another input.

    The end-to-end validation of these captures stops earlier, on a backward-graph
    metadata invariant for higher-order grad_fn handles that fails on the base too,
    so the replay and perturbation checks are pinned on the validator's own capture.
    """

    results: list[tuple[str, Any]] = []

    def check(trace: Any) -> None:
        for op in _grad_ops(trace):
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
        _validate(model_factory(), torch.randn(*shape))
    assert results
    assert {label: r.decision for label, r in results} == {
        label: "validated" for label, _ in results
    }


def test_forged_autograd_grad_func_fails_replay() -> None:
    def forge(trace: Any) -> None:
        def autogradgrad(outputs: Any, inputs: Any, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
            return tuple(torch.zeros_like(tensor) for tensor in inputs)

        op = _grad_ops(trace)[0]
        op.func = autogradgrad
        assert op.transform_kind == "autograd.grad"

    torch.manual_seed(0)
    with _tampered_validation(forge):
        assert _validate(_TwoLosses(), torch.randn(2, 3)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check == "forward_replay"
    assert failure.op_label is not None and failure.op_label.startswith("autogradgrad")


def test_forged_autograd_grad_parent_fails_replay() -> None:
    results: list[Any] = []

    def forge(trace: Any) -> None:
        sums = [op for op in trace.ops if op.func_name == "sum"]
        for op in _grad_ops(trace):
            true_root = op.parents[0]
            forged = next(s for s in sums if s.label.split(":")[0] != true_root)
            forged_label = forged.label.split(":")[0]
            op.parents = (forged_label,)
            op.parent_arg_positions = {"args": {0: forged_label}, "kwargs": {}}
            op.saved_args = (forged.out.detach().clone(), *tuple(op.saved_args)[1:])
            results.append(
                validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                    trace, op.label
                )
            )

    torch.manual_seed(0)
    with _tampered_validation(forge):
        assert _validate(_TwoLosses(), torch.randn(2, 3)) is False
    assert results and all(r.decision == "failed" for r in results)
    assert {r.reason for r in results} == {"replay_mismatch"}


def test_tampered_saved_gradient_fails_validation() -> None:
    results: list[Any] = []

    def tamper(trace: Any) -> None:
        op = _grad_ops(trace)[0]
        op._slot("out").add_(0.5)
        results.append(
            validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                trace, op.label
            )
        )

    torch.manual_seed(0)
    with _tampered_validation(tamper):
        assert _validate(_TwoLosses(), torch.randn(2, 3)) is False
    assert len(results) == 1
    assert results[0].decision == "failed" and results[0].reason == "replay_mismatch"


def test_unrecorded_tensor_beside_a_recorded_grad_still_fails_source_provenance() -> None:
    torch.manual_seed(0)
    assert _validate(_GradWithGlobal(), torch.randn(3)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check == "source_provenance"
    assert list(failure.extra.get("reasons", [])) == ["unattributed_tensor_args"]
    assert failure.op_label is not None and failure.op_label.startswith("add")
