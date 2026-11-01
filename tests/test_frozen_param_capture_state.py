"""Capture leaves frozen parameters frozen, and their provenance does not depend on autograd.

Capture used to force ``requires_grad=True`` on every floating parameter outside
``backward_ready`` so intermediate tensors carried ``grad_fn`` metadata. That changed
the program torch ran (attention picks a different kernel for gradient-tracking
weights), so the captured forward differed from eager. Capture now never flips the
flag: parameter reads are resolved by identity at the op boundary, so parameter
edges, ``Param`` fields, saved traces, backward capture and validation hold for
frozen weights exactly as for trainable ones. These tests use a partly frozen model.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure

FROZEN = ("frozen.weight", "frozen.bias", "scale")
TRAINABLE = ("head.weight", "head.bias")


class _PartlyFrozen(nn.Module):
    """A frozen linear and a frozen scale feeding a trainable head."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 5)
        self.scale = nn.Parameter(torch.full((5,), 0.5), requires_grad=False)
        self.head = nn.Linear(5, 3)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.frozen(x)) * self.scale
        return self.head(hidden)


def _model_and_input(seed: int = 0) -> tuple[_PartlyFrozen, torch.Tensor]:
    torch.manual_seed(seed)
    return _PartlyFrozen(), torch.randn(2, 4)


def _flags_inside_forward(model: nn.Module, run: Any) -> dict[str, bool]:
    """Run ``run()`` and return each parameter's ``requires_grad`` as the forward saw it."""

    seen: dict[str, bool] = {}

    def record(module: nn.Module, args: Any) -> None:
        seen.update({name: bool(p.requires_grad) for name, p in module.named_parameters()})

    handle = model.register_forward_pre_hook(record)
    try:
        run()
    finally:
        handle.remove()
    return seen


def _param_state(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            bool(param.requires_grad),
            param.is_leaf,
            param._version,
            None if param.grad is None else param.grad.clone(),
        )
        for name, param in model.named_parameters()
    }


def _assert_same_state(before: dict[str, tuple[Any, ...]], after: dict[str, tuple[Any, ...]]):
    assert before.keys() == after.keys()
    for name, (flag, leaf, version, grad) in before.items():
        a_flag, a_leaf, a_version, a_grad = after[name]
        assert (a_flag, a_leaf, a_version) == (flag, leaf, version), name
        if grad is None:
            assert a_grad is None, name
        else:
            assert a_grad is not None and torch.equal(a_grad, grad), name


def _ops_reading(trace: tl.Trace, address: str) -> list[Any]:
    return [op for op in trace.ops if address in [p.address for p in op.params]]


@pytest.mark.smoke
@pytest.mark.parametrize("backward_ready", [False, True])
@pytest.mark.parametrize("grad_mode", [True, False])
def test_capture_never_flips_requires_grad_and_leaves_model_state_identical(
    backward_ready: bool, grad_mode: bool
) -> None:
    model, x = _model_and_input()
    before = _param_state(model)

    def capture() -> None:
        with torch.set_grad_enabled(grad_mode):
            tl.trace(model, x, capture=CaptureOptions(backward_ready=backward_ready))

    seen = _flags_inside_forward(model, capture)
    assert seen == {name: name in TRAINABLE for name in (*FROZEN, *TRAINABLE)}
    _assert_same_state(before, _param_state(model))


def test_state_is_identical_when_parameters_already_hold_gradients() -> None:
    model, x = _model_and_input()
    model(x).sum().backward()
    assert model.head.weight.grad is not None and model.frozen.weight.grad is None
    before = _param_state(model)
    tl.trace(model, x)
    _assert_same_state(before, _param_state(model))


def test_frozen_parameter_edges_and_param_fields() -> None:
    model, x = _model_and_input()
    trace = tl.trace(model, x)

    for address in (*FROZEN, *TRAINABLE):
        readers = _ops_reading(trace, address)
        assert len(readers) == 1, (address, [op.label for op in readers])
        assert trace.params[address].num_calls == 1
    assert {p.address for p in trace.params.frozen} == set(FROZEN)
    assert {p.address for p in trace.params.trainable} == set(TRAINABLE)
    for address in FROZEN:
        assert trace.params[address].is_trainable is False
    for address in TRAINABLE:
        assert trace.params[address].is_trainable is True

    frozen_linear = _ops_reading(trace, "frozen.weight")[0]
    assert frozen_linear.func_name == "linear"
    assert "frozen.bias" in [p.address for p in frozen_linear.params]
    scale_mul = _ops_reading(trace, "scale")[0]
    head_linear = _ops_reading(trace, "head.weight")[0]
    assert scale_mul.label in [trace[parent].label for parent in head_linear.parents]


def test_ops_on_frozen_weights_carry_the_eager_autograd_state() -> None:
    """With a plain input, ops before the trainable head have no autograd node, as in eager."""

    model, x = _model_and_input()
    eager_hidden = torch.relu(model.frozen(x)) * model.scale
    assert eager_hidden.grad_fn is None
    trace = tl.trace(model, x)

    assert _ops_reading(trace, "frozen.weight")[0].grad_fn_class_name is None
    assert _ops_reading(trace, "scale")[0].grad_fn_class_name is None
    assert _ops_reading(trace, "head.weight")[0].grad_fn_class_name is not None


def test_saved_and_loaded_trace_keeps_frozen_parameter_fields(tmp_path: Path) -> None:
    model, x = _model_and_input()
    trace = tl.trace(model, x)
    bundle = tmp_path / "partly_frozen.tl"
    trace.save(bundle)
    restored = tl.load(bundle)

    for address in (*FROZEN, *TRAINABLE):
        assert restored.params[address].is_trainable is trace.params[address].trainable
        assert restored.params[address].num_calls == 1
        original_readers = [op.label for op in _ops_reading(trace, address)]
        assert [op.label for op in _ops_reading(restored, address)] == original_readers


def test_backward_capture_on_a_partly_frozen_model() -> None:
    model, x = _model_and_input()
    eager = copy.deepcopy(model)
    eager(x).sum().backward()

    trace = tl.trace(model, x, capture=CaptureOptions(backward_ready=True, save_grads="all"))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    for address in TRAINABLE:
        assert trace.params[address].has_grad is True, address
    for address in FROZEN:
        assert trace.params[address].has_grad is False, address
    for name, param in model.named_parameters():
        eager_grad = dict(eager.named_parameters())[name].grad
        if eager_grad is None:
            assert param.grad is None, name
        else:
            assert param.grad is not None and torch.allclose(param.grad, eager_grad), name
    head_linear = _ops_reading(trace, "head.weight")[0]
    assert any(isinstance(record.grad, torch.Tensor) for record in head_linear.grads)


@pytest.mark.parametrize("scope", ["forward", "backward"])
def test_validation_passes_on_a_partly_frozen_model(scope: str) -> None:
    model, x = _model_and_input()
    before = _param_state(model)
    assert tl.validate(model, x, scope=scope), last_validation_failure()
    _assert_same_state(before, _param_state(model))


def test_fully_frozen_model_captures_and_validates_without_autograd() -> None:
    model, x = _model_and_input()
    model.requires_grad_(False)
    before = _param_state(model)
    trace = tl.trace(model, x)
    assert all(op.grad_fn_class_name is None for op in trace.ops)
    assert len(trace.params.frozen) == len(FROZEN) + len(TRAINABLE)
    _assert_same_state(before, _param_state(model))
    assert tl.validate(model, x, scope="forward"), last_validation_failure()


def test_gradient_analysis_scope_builds_a_path_through_frozen_weights() -> None:
    """Only an analysis that backpropagates through its own capture flips frozen flags."""

    from torchlens.backends.torch.model_prep import frozen_parameter_gradients
    from torchlens.mechinterp._eap import _capture

    model, x = _model_and_input()
    model.requires_grad_(False)
    before = _param_state(model)

    plain = tl.trace(model, x)
    assert _ops_reading(plain, "head.weight")[0].grad_fn_class_name is None

    def scoped_capture() -> None:
        with frozen_parameter_gradients():
            captured.append(tl.trace(model, x))

    captured: list[tl.Trace] = []
    assert all(_flags_inside_forward(model, scoped_capture).values())
    scoped = captured[0]
    assert _ops_reading(scoped, "frozen.weight")[0].grad_fn_class_name is not None
    assert all(scoped.params[name].is_trainable is False for name in (*FROZEN, *TRAINABLE))
    _assert_same_state(before, _param_state(model))

    analysis = _capture(model, x)
    assert _ops_reading(analysis, "head.weight")[0].grad_fn_class_name is not None
    _assert_same_state(before, _param_state(model))

    plain_flags = _flags_inside_forward(model, lambda: tl.trace(model, x))
    assert not any(plain_flags.values())
