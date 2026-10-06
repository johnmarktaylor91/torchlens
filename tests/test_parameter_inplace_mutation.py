"""In-place ops on a prepared ``nn.Parameter`` inside ``forward`` are captured.

MIX-HIC's ``with torch.no_grad(): self.temp.clamp_(0.001, 0.5)`` on
``self.temp = nn.Parameter(...)`` used to vanish from the graph: the op returns the
Parameter itself, and Parameter outputs were never logged, so the completeness
tripwire failed (``3 dispatched vs 2 captured``). The op is now logged with the
Parameter as its parameter input, and every later read of the Parameter in the pass
consumes the op's output.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import param_mutation, wrappers


class _ParamMutator(nn.Module):
    """Mutate a scalar Parameter in place under ``no_grad``, then divide by it."""

    def __init__(self, op: str, init: float = 0.9) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.temp = nn.Parameter(init * torch.ones([]))
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.op == "clamp_":
                self.temp.clamp_(0.001, 0.5)
            elif self.op == "copy_":
                self.temp.copy_(torch.tensor(0.25))
            elif self.op == "mul_":
                self.temp.mul_(0.5)
            elif self.op == "add_":
                self.temp.add_(0.125)
            elif self.op == "twice":
                self.temp.clamp_(0.001, 0.5)
                self.temp.mul_(0.5)
        return self.lin(x) / self.temp


class _ReadBeforeAndAfter(nn.Module):
    """Read a Parameter, mutate it in place, then read it again."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((4,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x * self.scale
        with torch.no_grad():
            self.scale.mul_(3.0)
        after = x * self.scale
        return before + after


class _NoMutation(nn.Module):
    """Same Parameter reads with no mutation: the plain-capture control."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.temp = nn.Parameter(0.9 * torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) / self.temp


def _ops_by_type(trace: Any, op_type: str) -> list[Any]:
    return [layer for layer in trace.layers if layer.type == op_type]


def _parent_labels(trace: Any, layer: Any) -> list[str]:
    return [trace[parent].label for parent in layer.parents]


def _eager_output_and_state(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    eager = copy.deepcopy(model)
    with torch.no_grad():
        out = eager(x)
    return out, {k: v.detach().clone() for k, v in eager.state_dict().items()}


@pytest.mark.parametrize("op", ["clamp_", "copy_", "mul_", "add_"])
def test_parameter_inplace_op_is_captured_and_consumed(op: str) -> None:
    torch.manual_seed(0)
    model = _ParamMutator(op)
    x = torch.randn(3, 4)
    eager_out, eager_state = _eager_output_and_state(model, x)

    trace = tl.trace(copy.deepcopy(model), x)
    op_type = op.rstrip("_")
    mutations = _ops_by_type(trace, op_type)
    assert len(mutations) == 1, [layer.label for layer in trace.layers]
    mutation = mutations[0]
    assert [p.address for p in mutation.params] == ["temp"]
    truediv = _ops_by_type(trace, "truediv")[0]
    # The later read consumes the mutation op's output, not the Parameter source.
    assert mutation.label in _parent_labels(trace, truediv)
    assert "temp" not in [p.address for p in truediv.params]
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)

    traced_model = copy.deepcopy(model)
    tl.trace(traced_model, x)
    for key, value in traced_model.state_dict().items():
        assert torch.equal(value, eager_state[key]), key

    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


def test_parameter_mutated_twice_chains_both_ops() -> None:
    torch.manual_seed(0)
    model = _ParamMutator("twice")
    x = torch.randn(3, 4)
    eager_out, _ = _eager_output_and_state(model, x)
    trace = tl.trace(copy.deepcopy(model), x)
    clamp = _ops_by_type(trace, "clamp")[0]
    mul = _ops_by_type(trace, "mul")[0]
    truediv = _ops_by_type(trace, "truediv")[0]
    assert [p.address for p in clamp.params] == ["temp"]
    assert _parent_labels(trace, mul) == [clamp.label]
    assert mul.params == [] or not list(mul.params)
    assert mul.label in _parent_labels(trace, truediv)
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


@pytest.mark.smoke
def test_parameter_read_before_and_after_mutation() -> None:
    model = _ReadBeforeAndAfter()
    x = torch.randn(2, 4)
    eager_out, eager_state = _eager_output_and_state(model, x)
    traced_model = copy.deepcopy(model)
    trace = tl.trace(traced_model, x)
    muls = _ops_by_type(trace, "mul")
    assert len(muls) == 3, [layer.label for layer in trace.layers]
    before, mutation, after = muls
    # The before-read sees the old value through its parameter edge ...
    assert [p.address for p in before.params] == ["scale"]
    assert torch.allclose(before.out, x * 2.0)
    assert [p.address for p in mutation.params] == ["scale"]
    # ... and the after-read consumes the mutation op's output (the new value).
    assert mutation.label in _parent_labels(trace, after)
    assert not list(after.params)
    assert torch.allclose(after.out, x * 6.0)
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)
    assert torch.equal(traced_model.scale.detach(), eager_state["scale"])
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


def test_validate_restores_mutated_parameter_like_a_buffer() -> None:
    """``tl.validate`` leaves model state as it found it (the buffer contract)."""

    model = _ParamMutator("mul_")
    before = model.temp.detach().clone()
    assert tl.validate(model, torch.randn(3, 4), scope="forward") is True
    assert torch.equal(model.temp.detach(), before)


def test_mutation_label_does_not_leak_onto_the_parameter() -> None:
    model = _ParamMutator("clamp_")
    tl.trace(model, torch.randn(3, 4))
    assert getattr(model.temp, "_tl", None) is None
    # A second capture of the same model starts from a clean parameter edge.
    trace = tl.trace(model, torch.randn(3, 4))
    clamp = _ops_by_type(trace, "clamp")[0]
    assert [p.address for p in clamp.params] == ["temp"]


def test_plain_capture_graph_unchanged_without_mutation() -> None:
    trace = tl.trace(_NoMutation(), torch.randn(3, 4))
    assert [layer.type for layer in trace.layers] == ["input", "linear", "truediv", "output"]
    truediv = _ops_by_type(trace, "truediv")[0]
    assert [p.address for p in truediv.params] == ["temp"]
    assert trace.num_ops == 2


@pytest.mark.smoke
def test_validation_still_fails_when_the_parameter_mutation_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completeness tripwire bites if capture drops the Parameter mutation again."""

    original = wrappers._parameter_mutation_output_for_logging

    def _drop_prepared_parameter_mutations(
        trace: Any, value: Any, *, source: Any, was_inplace: bool, is_storage_rebind: bool = False
    ) -> Any:
        if (
            was_inplace
            and isinstance(source, nn.Parameter)
            and not param_mutation.is_unregistered_parameter(trace, source)
        ):
            return value
        return original(
            trace,
            value,
            source=source,
            was_inplace=was_inplace,
            is_storage_rebind=is_storage_rebind,
        )

    monkeypatch.setattr(
        wrappers, "_parameter_mutation_output_for_logging", _drop_prepared_parameter_mutations
    )
    with pytest.warns(Warning):
        assert tl.validate(_ParamMutator("clamp_"), torch.randn(3, 4), scope="forward") is False
    # The failure is the completeness tripwire on the dropped op, not an unrelated check.
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert "bfs_completeness" in failure.summary()
    assert "dispatched vs" in failure.summary()


class _AddThenScale(nn.Module):
    """Shift a Parameter in place under ``no_grad``, then scale by it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.full((4,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.scale.add_(1.0)
        return self.lin(x) * self.scale


def test_intervention_on_the_parameter_mutation_op_applies() -> None:
    """A live intervention on the mutation op lands in the Parameter; the op stays."""

    torch.manual_seed(0)
    model = _AddThenScale()
    x = torch.randn(3, 4)
    trace = tl.trace(model, x, intervene=tl.when(tl.func("add"), tl.zero_ablate()))
    adds = _ops_by_type(trace, "add")
    assert len(adds) == 1, [layer.label for layer in trace.layers]
    mul = _ops_by_type(trace, "mul")[0]
    assert adds[0].label in _parent_labels(trace, mul)
    assert "scale" not in [p.address for p in mul.params]
    # Eager semantics with the hook: the in-place op's result (zeroed) is the
    # Parameter's new value, so every later read sees zeros.
    assert torch.equal(model.scale.detach(), torch.zeros(4))
    assert torch.equal(trace[trace.output_layers[0]].out, torch.zeros(3, 4))


def test_intervention_elsewhere_keeps_the_parameter_mutation_unhooked() -> None:
    """Narrowness: hooking another op leaves the mutation op and its value alone."""

    torch.manual_seed(0)
    model = _AddThenScale()
    x = torch.randn(3, 4)
    eager_lin = model.lin(x).detach()
    trace = tl.trace(model, x, intervene=tl.when(tl.func("linear"), tl.zero_ablate()))
    add = _ops_by_type(trace, "add")[0]
    mul = _ops_by_type(trace, "mul")[0]
    assert add.label in _parent_labels(trace, mul)
    assert torch.equal(model.scale.detach(), torch.full((4,), 3.0))
    assert not torch.equal(eager_lin, torch.zeros(3, 4))
    assert torch.equal(trace[trace.output_layers[0]].out, torch.zeros(3, 4))


class _FrozenMutator(nn.Module):
    """Mutate a frozen (``requires_grad=False``) Parameter in place with grad mode on."""

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.temp = nn.Parameter(0.9 * torch.ones([]), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.temp.mul_(0.5)
        return self.lin(x) / self.temp


def test_frozen_parameter_mutated_without_no_grad_captures_like_eager() -> None:
    torch.manual_seed(0)
    model = _FrozenMutator()
    x = torch.randn(3, 4)
    eager = copy.deepcopy(model)
    eager_out = eager(x).detach()

    traced_model = copy.deepcopy(model)
    trace = tl.trace(traced_model, x)
    mul = _ops_by_type(trace, "mul")[0]
    assert [p.address for p in mul.params] == ["temp"]
    truediv = _ops_by_type(trace, "truediv")[0]
    assert mul.label in _parent_labels(trace, truediv)
    assert "temp" not in [p.address for p in truediv.params]
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)
    assert torch.equal(traced_model.temp.detach(), eager.temp.detach())
    assert traced_model.temp.requires_grad is False

    validated = copy.deepcopy(model)
    assert tl.validate(validated, x, scope="forward") is True
    assert torch.equal(validated.temp.detach(), model.temp.detach())


def test_trainable_parameter_mutated_without_no_grad_still_raises_like_eager() -> None:
    """Narrowness: only frozen Parameters run untracked; eager refuses this one too."""

    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
        _FrozenMutator(requires_grad=True)(x)
    with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
        tl.trace(_FrozenMutator(requires_grad=True), x)


class _FrozenGradOperandMutator(nn.Module):
    """Mutate a frozen Parameter in place with a grad-requiring operand."""

    def __init__(self, spelling: str) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.temp = nn.Parameter(0.9 * torch.ones([]), requires_grad=False)
        self.spelling = spelling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        operand = self.lin.weight.sum()
        if self.spelling == "keyword":
            self.temp.add_(other=operand)
        else:
            self.temp.add_(operand)
        return x / self.temp


@pytest.mark.parametrize("spelling", ["positional", "keyword"])
def test_frozen_parameter_with_grad_operand_is_never_run_untracked(spelling: str) -> None:
    """A grad-requiring operand in any spelling keeps the call tracked.

    Eager makes the frozen Parameter a non-leaf here. Capture's forced
    ``requires_grad`` makes autograd refuse the same call, and that refusal must
    surface in every spelling: running the keyword spelling under ``no_grad``
    would silently drop the Parameter's autograd history.
    """

    x = torch.randn(3, 4)
    eager = _FrozenGradOperandMutator(spelling)
    eager(x)
    assert eager.temp.grad_fn is not None
    with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
        tl.trace(_FrozenGradOperandMutator(spelling), x)


def test_operand_scan_reaches_nested_containers() -> None:
    """The frozen-receiver operand scan descends into tuples, lists and dicts."""

    grad = torch.ones(2, requires_grad=True)
    plain = torch.ones(2)
    operand = (plain, [{"values": grad}], 3)
    found = list(param_mutation._iter_operand_tensors(operand))
    assert len(found) == 2
    assert found[0] is plain and found[1] is grad


class _LazyThenMutate(nn.Module):
    """A lazy layer materialized in forward beside an initialized, mutated Parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.lazy = nn.LazyLinear(4)
        self.temp = nn.Parameter(0.9 * torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        return self.lazy(x) / self.temp


def test_lazy_materialization_is_not_a_parameter_mutation() -> None:
    """Lazy init writes stay out of the graph; an initialized Parameter's write is logged."""

    trace = tl.trace(_LazyThenMutate(), torch.randn(3, 4))
    assert sorted(layer.layer_type for layer in trace.layer_list) == [
        "clamp",
        "input",
        "linear",
        "output",
        "truediv",
    ]
    assert trace.num_ops == 3
    clamp = _ops_by_type(trace, "clamp")[0]
    assert [p.address for p in clamp.params] == ["temp"]
    assert clamp.label in _parent_labels(trace, _ops_by_type(trace, "truediv")[0])
