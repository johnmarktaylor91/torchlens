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
from torchlens.validation.invariants import check_metadata_invariants

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


# Imported by tests/test_valmem3_saved_arg_offload.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_autograd_grad_boundary.py, campaign mutant kills.
def test_autograd_grad_recorder_runs_once_and_holds_no_live_call() -> None:
    torch.manual_seed(0)
    trace = tl.trace(MAMLInnerLoop(), torch.rand(4, 8))
    recorder = _grad_ops(trace)[0].func
    with pytest.raises(RuntimeError, match="runs once"):
        recorder()


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
@pytest.mark.parametrize(
    ("model_factory", "shape"),
    [(MAMLInnerLoop, (4, 8)), (_IntermediateInput, (2, 4))],
    ids=["maml", "intermediate_input_grad_outputs"],
)
def test_recorded_autograd_grad_validates_by_subgraph_replay(
    model_factory: Callable[[], nn.Module], shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    assert _validate(model_factory(), torch.randn(*shape)) is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_autograd_grad_replay.py.
@pytest.mark.parametrize(
    ("model_factory", "shape"),
    [(_SecondOrder, (2, 3)), (_TwoCallsDifferentArity, (3,))],
    ids=["second_order", "two_arities"],
)
def test_nested_and_dependent_input_grads_replay_faithfully(
    model_factory: Callable[[], nn.Module], shape: tuple[int, ...]
) -> None:
    """A grad inside another grad's subgraph, and an input that depends on another input.

    The replay and perturbation checks are pinned on the validator's own capture, and
    the end-to-end verdict must be True: a second-order grad inside forward used to fail
    the backward-graph metadata invariants (a forward op applied to the first gradient
    was inferred into order 2 while the nodes its differentiation created kept order 2
    instead of 3), and the recorded failure was never asserted here.
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
        verdict = _validate(model_factory(), torch.randn(*shape))
    assert verdict is True, last_validation_failure()
    assert last_validation_failure() is None
    assert results
    assert {label: r.decision for label, r in results} == {
        label: "validated" for label, _ in results
    }


class _GradOfInputsLinearInEach(nn.Module):
    """Energy linear in each of two forward inputs, differentiated against both.

    Mirrors an energy/force/dipole head: ``force = W^T ef`` does not read ``cart``
    and ``dipole = lin(cart)`` does not read ``ef``, so each gradient's own input
    can influence it only through the derivative, never through its value.
    """

    def __init__(self, zero_head: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        if zero_head:
            nn.init.zeros_(self.lin.weight)

    def forward(self, cart: torch.Tensor, ef: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cart.requires_grad = True
        ef.requires_grad = True
        energy = (self.lin(cart) * ef).sum()
        force, dipole = torch.autograd.grad(energy, (cart, ef), create_graph=True)
        return energy, dipole, -force


def _perturbation_decisions(model: nn.Module, *inputs: torch.Tensor) -> dict[str, Any]:
    """Return each autograd.grad op's perturbation decision per parent edge."""

    results: dict[str, Any] = {}

    def check(trace: Any) -> None:
        for op in _grad_ops(trace):
            for parent in op.parents:
                results[f"{op.label}<-{parent}"] = (
                    validation_core._check_whether_func_on_saved_parents_yields_saved_tensor(
                        trace, op.label, perturb=True, layers_to_perturb=[parent]
                    )
                )

    with _tampered_validation(check):
        _validate(model, *inputs)
    return results


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_autograd_grad_replay.py, validation/_degenerate_forward.py.
def test_identically_zero_grad_still_fails_its_input_perturbation() -> None:
    """A zero weight leaves every gradient zero; the input edge still fails, recorded."""

    torch.manual_seed(0)
    model = _GradOfInputsLinearInEach(zero_head=True)
    decisions = _perturbation_decisions(model, torch.randn(2, 3), torch.randn(2, 3))
    failed = {edge: r.reason for edge, r in decisions.items() if r.failed}
    assert failed and set(failed.values()) == {"perturbation_insensitive"}
    torch.manual_seed(0)
    assert _validate(model, torch.randn(2, 3), torch.randn(2, 3)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.op_label is not None and failure.op_label.startswith("autogradgrad")


@pytest.mark.parametrize(
    ("model_cls", "shape", "paired_funcs"),
    [
        (_SecondOrder, (2, 3), {"matmul", "pow", "sum"}),
        (_TwoLosses, (2, 3), {"matmul", "tanh", "add", "pow", "sum"}),
        (_IntermediateInput, (2, 4), {"linear", "sin", "mul"}),
    ],
    ids=["second-order", "two-losses", "intermediate-input"],
)
def test_grad_fns_of_a_backward_inside_forward_pair_with_their_captured_ops(
    model_cls: type[nn.Module], shape: tuple[int, ...], paired_funcs: set[str]
) -> None:
    """A backward pass run inside forward pairs each GradFn with the op that records it.

    The walk of such a pass runs before any layer exists, so the pairing must not
    depend on walk time: every GradFn whose node a captured op records names that op,
    exactly as a backward run after the trace would.
    """

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model_cls(), torch.randn(*shape))
    try:
        grad_fns = {grad_fn.grad_fn_object_id: grad_fn for grad_fn in trace.autograd.grad_fns}
        recording_ops = [
            op
            for op in trace._layer_list
            if op.grad_fn_object_id in grad_fns and getattr(op, "is_compute_op", False)
        ]
        assert recording_ops
        for op in recording_ops:
            grad_fn = grad_fns[op.grad_fn_object_id]
            assert grad_fn.has_op, grad_fn.label
            assert grad_fn.op is not None and grad_fn.op_label == op.layer_label
        paired = [grad_fn for grad_fn in grad_fns.values() if grad_fn.has_op]
        # Operator calls record their dunder name (``x @ w`` is ``__matmul__``).
        assert {grad_fn.op.func_name.strip("_") for grad_fn in paired} >= paired_funcs
        check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_second_order_inside_forward_keeps_the_orders_of_the_same_math_after_the_trace() -> None:
    """Pairing the penalty's ops does not change derivative orders: pass 2 stays order 2.

    ``penalty = (g ** 2).sum()`` is captured inside forward, so its nodes pair with
    their ops; they still feed the first gradient's order-2 nodes, so they sit at
    order 2 and the nodes the second grad creates from them at order 3.
    """

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_SecondOrder(), torch.randn(2, 3))
    try:
        assert [p.order for p in trace.autograd.backward_passes] == [1, 2]
        grad_fns = list(trace.autograd.grad_fns)
        ops = list(trace.ops)
        first_grad, second_grad = [
            index for index, op in enumerate(ops) if op.transform_kind == "autograd.grad"
        ]
        # ``penalty = (g ** 2).sum()`` runs between the two grads: ``__pow__`` then ``sum``.
        penalty_ops = {op.func_name.strip("_"): op for op in ops[first_grad + 1 : second_grad]}
        assert {"pow", "sum"} <= set(penalty_ops), sorted(penalty_ops)
        penalty_ids = {penalty_ops["pow"].grad_fn_object_id, penalty_ops["sum"].grad_fn_object_id}
        penalty_nodes = [g for g in grad_fns if g.grad_fn_object_id in penalty_ids]
        assert len(penalty_nodes) == 2 and all(g.has_op for g in penalty_nodes)
        assert {grad_fn.order for grad_fn in penalty_nodes} == {2}
        creations = [g for g in grad_fns if g.creator_object_id in penalty_ids]
        assert creations and {g.order for g in creations} == {3}
    finally:
        trace.cleanup()
