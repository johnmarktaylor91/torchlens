"""Entity-property tests for receptive-field views."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import FieldPolicy
from torchlens.intervention.errors import MultiOutputModuleError
from torchlens.receptive_field import ReceptiveFieldStatus, ReceptiveFieldView, _rules
from torchlens.receptive_field._errors import (
    AmbiguousCallError,
    AmbiguousInputError,
    AmbiguousPassError,
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
)
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global receptive-field registry after every test."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _register_rules() -> None:
    """Register the exact convolution and merge rules used by these tests."""

    @_rules.register_rf_rule("conv2d", replace="conv2d" in _rules._RF_RULES)
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit captured two-dimensional convolution geometry."""

        kernel = context.cfg("kernel_size")
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        assert isinstance(kernel, tuple)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1, 1)),
            padding=context.cfg("padding", (0, 0)),
            dilation=context.cfg("dilation", (1, 1)),
        )

    @_rules.register_rf_rule("add", "relu", replace="add" in _rules._RF_RULES)
    def passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve aligned elementwise geometry and merge input ancestry."""

        return context.passthrough()


def _conv_trace() -> tuple[object, object]:
    """Capture a small convolution and return its trace and target operation."""

    _register_rules()
    trace = tl.trace(nn.Sequential(nn.Conv2d(2, 4, 3)), torch.randn(2, 2, 8, 8))
    target = next(op for op in trace.layer_list if op.func_name == "conv2d")
    return trace, target


class _TwoInputAdd(nn.Module):
    """Merge two distinct model inputs at one operation."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Add both model inputs elementwise."""

        return left + right


class _RepeatedLinear(nn.Module):
    """Invoke one shared module twice to create pass and call ambiguity."""

    def __init__(self) -> None:
        """Create the shared linear module."""

        super().__init__()
        self.shared = nn.Linear(3, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same module in two consecutive calls."""

        return self.shared(self.shared(inputs))


class _Pair(nn.Module):
    """Return two tensor outputs from one module call."""

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce two distinct output operations."""

        return inputs + 1, inputs * 2


class _UsesPair(nn.Module):
    """Consume both outputs from a multi-output child call."""

    def __init__(self) -> None:
        """Create the child module."""

        super().__init__()
        self.pair = _Pair()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Merge the child module's pair of outputs."""

        left, right = self.pair(inputs)
        return left + right


def test_op_view_is_cached_and_deleter_recomputes() -> None:
    """Keep repeated property access identity-stable until explicit deletion."""

    trace, target = _conv_trace()
    first = target.receptive_field

    assert isinstance(first, ReceptiveFieldView)
    assert target.receptive_field is first
    assert trace[target.layer_label].receptive_field is first

    del target.receptive_field
    second = target.receptive_field
    assert second is not first
    assert target.receptive_field is second


def test_view_mapping_passthrough_at_and_center_unit() -> None:
    """Expose mapping lookup and single-input convenience queries exactly."""

    trace, target = _conv_trace()
    view = target.receptive_field
    input_op = next(op for op in trace.layer_list if op.is_input)
    role = str(input_op.io_role)
    descriptor = view.per_input[role]

    assert view[role] is descriptor
    assert view[input_op] is descriptor
    assert view.status is ReceptiveFieldStatus.EXACT
    assert view.axes is descriptor.axes
    assert view.size == (3, 3)
    assert view.jump == descriptor.jump
    assert view.center0 == descriptor.center0
    assert view.layout is descriptor.layout
    assert view.at((2, 2), input=input_op).unit == (2, 2)
    assert view.at("center", input=role).unit == (3, 3)

    with pytest.raises(ReceptiveFieldError, match="batch_index is required"):
        view.center_unit()
    assert view.center_unit(batch_index=1) == (1, 2, 3, 3)


def test_landed_gradient_is_wired_and_check_is_typed_unavailable() -> None:
    """Expose the landed gradient core and retain the typed T9 wiring point."""

    _register_rules()
    inputs = torch.randn(1, 2, 8, 8, requires_grad=True)
    trace = tl.trace(
        nn.Conv2d(2, 4, 3),
        inputs,
        backward_ready=True,
        save_mode="reference",
    )
    target = next(op for op in trace.layer_list if op.func_name == "conv2d")
    input_op = next(op for op in trace.layer_list if op.is_input)
    view = target.receptive_field

    gradient = view.gradient((0, 0, 2, 2), input=input_op)
    assert gradient.op_label == target.label

    with pytest.raises(ReceptiveFieldUnavailableError, match="Task T9"):
        view.check((0, 0, 2, 2))


def test_multi_input_passthrough_names_roles_and_explicit_selection_works() -> None:
    """Reject ambiguous convenience fields while preserving explicit lookup."""

    _register_rules()
    trace = tl.trace(_TwoInputAdd(), (torch.randn(1, 3), torch.randn(1, 3)))
    target = next(op for op in trace.layer_list if op.func_name == "__add__")
    view = target.receptive_field
    roles = tuple(view.per_input)

    assert len(roles) == 2
    with pytest.raises(AmbiguousInputError) as caught:
        _ = view.status
    assert all(role in str(caught.value) for role in roles)
    assert view[roles[0]].io_role == roles[0]


def test_layer_and_module_ambiguities_are_typed_and_name_choices() -> None:
    """Raise domain errors for recurrent layers and repeated module calls."""

    _register_rules()
    trace = tl.trace(_RepeatedLinear(), torch.randn(2, 3))
    layer = next(layer for layer in trace.layers if layer.num_passes == 2)
    module = trace.modules["shared"]

    with pytest.raises(AmbiguousPassError) as pass_error:
        _ = layer.receptive_field
    assert "layer.ops[" in str(pass_error.value)

    with pytest.raises(AmbiguousCallError) as call_error:
        _ = module.receptive_field
    assert "module.calls[" in str(call_error.value)


def test_module_call_and_module_delegate_for_single_output_single_call() -> None:
    """Delegate unambiguous module entities to their boundary output operation."""

    trace, _ = _conv_trace()
    module = trace.modules["0"]
    call = module.calls[0]
    output = trace.ops[call.output_ops[0]]

    assert call.receptive_field is output.receptive_field
    assert module.receptive_field is output.receptive_field


def test_multi_output_module_call_reuses_existing_error_and_lists_candidates() -> None:
    """Reject multi-output calls with the established intervention error type."""

    trace = tl.trace(_UsesPair(), torch.randn(2, 3))
    call = trace.modules["pair"].calls[0]

    assert len(call.output_ops) == 2
    with pytest.raises(MultiOutputModuleError) as caught:
        _ = call.receptive_field
    assert all(label in str(caught.value) for label in call.output_ops)


def test_tlspec_round_trip_drops_and_recomputes_view(tmp_path: Path) -> None:
    """Keep the view runtime-only and reconstruct it after portable loading."""

    trace, target = _conv_trace()
    original = target.receptive_field
    path = tmp_path / "rf-view.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    loaded_target = loaded.ops[target.label]
    assert loaded_target._slot("_receptive_field_cache") is None
    recomputed = loaded_target.receptive_field
    assert isinstance(recomputed, ReceptiveFieldView)
    assert recomputed is loaded_target.receptive_field
    assert recomputed is not original
    assert type(loaded_target).PORTABLE_STATE_SPEC["_receptive_field_cache"] is FieldPolicy.DROP


def test_registry_epoch_invalidation_refreshes_all_cached_views() -> None:
    """Refresh every populated operation slot when registration advances the epoch."""

    trace, target = _conv_trace()
    first = target.receptive_field
    input_op = next(op for op in trace.layer_list if op.is_input)
    first_input = input_op.receptive_field

    @_rules.register_rf_rule("rf_entity_epoch_probe")
    def epoch_probe(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Advance the registry epoch with an unrelated passthrough rule."""

        return context.passthrough()

    second = target.receptive_field
    assert second is not first
    assert second._solution is not first._solution
    assert input_op.receptive_field is not first_input
    assert input_op.receptive_field._solution is second._solution
