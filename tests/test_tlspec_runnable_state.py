"""Stage 4 sparse runnable state binding and initialization tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable_state import prepare_runnable_state
from torchlens.errors import StateBindingError
from torchlens.options import CaptureOptions
from torchlens.runnable import RunnableErrorCode, StateSlotRole, StateSource, TensorSlotRole


class StatefulRunnableModel(nn.Module):
    """Small stateful model whose forward count detects accidental execution."""

    def __init__(self) -> None:
        """Initialize parameter, normalization, buffer, and execution counter state."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run one stateful graph and record invocation count."""

        self.forward_calls += 1
        return self.linear(value) * self.scale


@pytest.fixture(scope="module")
def runnable_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, torch.Tensor]]:
    """Create one sparse runnable artifact and its independent user state mapping."""

    model = StatefulRunnableModel().eval()
    trace = tl.trace(
        model,
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    assert model.forward_calls == 1
    path = tmp_path_factory.mktemp("runnable-state") / "state.tlspec"
    trace.save(path, level="runnable")
    return path, {name: value.detach().clone() for name, value in model.state_dict().items()}


def _with_rebuilt_state_metadata_witnesses(descriptor: Any) -> Any:
    """Regenerate the totalized state_metadata witnesses from the (synthetic) bindings.

    r71 E1 totalizes state-metadata emission over the declared state-name universe
    and the staging belt requires witness/binding agreement, so a test that
    synthesizes slot tables in memory must keep the witness stream coherent (an
    incoherent descriptor is exactly what the belt refuses).
    """

    from torchlens._io.runnable import STATE_METADATA_FACT_SITE_PREFIX, _encode_literal
    from torchlens.runnable import ControlWitness, ControlWitnessKind

    witnesses = [
        witness
        for witness in descriptor.control_witnesses
        if not (
            witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
            and witness.site_label.startswith(STATE_METADATA_FACT_SITE_PREFIX)
        )
    ]
    facts: dict[str, tuple[bool, bool]] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is not None:
            facts.setdefault(
                binding.state_dict_name,
                (binding.captured_requires_grad, binding.captured_grad_fn),
            )
    for name in sorted(facts):
        requires_grad, grad_fn_present = facts[name]
        observed = _encode_literal(
            {
                "state_metadata": True,
                "state": name,
                "facts": {"grad_fn": grad_fn_present, "requires_grad": requires_grad},
            }
        )
        order = len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                order=order,
                call_id=None,
                site_label=f"{STATE_METADATA_FACT_SITE_PREFIX}{name}",
                observed_value=observed,
            )
        )
    return replace(descriptor, control_witnesses=tuple(witnesses))


def _load(artifact: tuple[Path, dict[str, torch.Tensor]]) -> tl.Trace:
    """Load a fresh sparse Trace for an isolated state lifecycle test."""

    trace = tl.load(artifact[0])
    assert isinstance(trace, tl.Trace)
    return trace


def _codes(error: StateBindingError) -> set[RunnableErrorCode]:
    """Return machine-readable diagnostic codes from a binding exception."""

    diagnostics = error.fields["diagnostics"]
    assert isinstance(diagnostics, tuple)
    return {diagnostic.code for diagnostic in diagnostics}


@pytest.mark.smoke
def test_load_state_dict_maps_names_to_slots_and_does_not_execute(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """Stage canonical parameter/buffer values without invoking a recorded callable."""

    trace = _load(runnable_artifact)
    state = runnable_artifact[1]
    descriptor_before = trace.runnable_descriptor
    calls_before = dict(trace.__dict__["_runnable_callables_by_call_id"])
    for call_id in calls_before:
        trace.__dict__["_runnable_callables_by_call_id"][call_id] = _execution_forbidden

    trace.load_state_dict(state)
    prepared = prepare_runnable_state(trace, seed=19)

    assert prepared.state_source is StateSource.USER_STATE_DICT
    assert trace.readiness is not None
    assert trace.readiness.state_sources_available[0] is StateSource.USER_STATE_DICT
    assert prepared.random_filled_slot_ids == ()
    assert trace.runnable_descriptor is descriptor_before
    for slot in descriptor_before.tensor_slots:
        if slot.state_binding is None:
            assert slot.slot_id not in prepared.slot_values
            continue
        expected = state[slot.state_binding.state_dict_name]
        assert torch.equal(prepared.slot_values[slot.slot_id], expected)


def _execution_forbidden(*args: Any, **kwargs: Any) -> None:
    """Fail if state-only Stage 4 accidentally invokes a graph callable."""

    raise AssertionError(f"Stage 4 executed a graph callable with {args!r}, {kwargs!r}")


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing", RunnableErrorCode.STATE_MISSING_KEY),
        ("unexpected", RunnableErrorCode.STATE_UNEXPECTED_KEY),
        ("shape", RunnableErrorCode.STATE_SHAPE_MISMATCH),
        ("dtype", RunnableErrorCode.STATE_DTYPE_MISMATCH),
    ],
)
def test_load_state_dict_strict_mapping_failures(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
    mutation: str,
    expected_code: RunnableErrorCode,
) -> None:
    """Reject every strict key and tensor-contract failure category."""

    trace = _load(runnable_artifact)
    state = {name: value.clone() for name, value in runnable_artifact[1].items()}
    target = "linear.weight"
    if mutation == "missing":
        del state[target]
    elif mutation == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
    elif mutation == "shape":
        state[target] = torch.ones(1, dtype=state[target].dtype)
    else:
        state[target] = state[target].to(torch.float64)

    with pytest.raises(StateBindingError) as caught:
        trace.load_state_dict(state)

    assert expected_code in _codes(caught.value)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("module_path", "wrong", RunnableErrorCode.STATE_MODULE_PATH_MISMATCH),
        ("semantic_role", StateSlotRole.BIAS, RunnableErrorCode.STATE_ROLE_MISMATCH),
    ],
)
def test_load_state_dict_verifies_module_path_and_semantic_role(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
    field: str,
    replacement: str | StateSlotRole,
    expected_code: RunnableErrorCode,
) -> None:
    """Verify recorded ownership and role only after canonical name mapping."""

    trace = _load(runnable_artifact)
    descriptor = trace.runnable_descriptor
    assert descriptor is not None
    slots = list(descriptor.tensor_slots)
    index = next(
        index
        for index, slot in enumerate(slots)
        if slot.state_binding is not None and slot.state_binding.state_dict_name == "linear.weight"
    )
    binding = slots[index].state_binding
    assert binding is not None
    slots[index] = replace(slots[index], state_binding=replace(binding, **{field: replacement}))
    trace.__dict__["_runnable_descriptor"] = replace(descriptor, tensor_slots=tuple(slots))

    with pytest.raises(StateBindingError) as caught:
        trace.load_state_dict(runnable_artifact[1])

    assert expected_code in _codes(caught.value)


def test_alias_groups_reject_conflicts_and_share_one_staged_value(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """Require coherent alias entries and stage one shared tensor allocation."""

    trace = _load(runnable_artifact)
    descriptor = trace.runnable_descriptor
    assert descriptor is not None
    original = next(
        slot
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None and slot.state_binding.state_dict_name == "linear.weight"
    )
    binding = original.state_binding
    assert binding is not None
    first = replace(original, state_binding=replace(binding, alias_group="tied:linear"))
    second = replace(
        original,
        slot_id="state:linear.weight_alias",
        state_binding=replace(
            binding,
            state_dict_name="linear.weight_alias",
            alias_group="tied:linear",
        ),
    )
    trace.__dict__["_runnable_descriptor"] = _with_rebuilt_state_metadata_witnesses(
        replace(
            descriptor,
            tensor_slots=tuple(slot for slot in descriptor.tensor_slots if slot is not original)
            + (first, second),
        )
    )
    state = {name: value.clone() for name, value in runnable_artifact[1].items()}
    state["linear.weight_alias"] = state["linear.weight"].clone()
    trace.load_state_dict(state)
    prepared = prepare_runnable_state(trace)
    assert prepared.slot_values[first.slot_id] is prepared.slot_values[second.slot_id]

    state["linear.weight_alias"] = state["linear.weight_alias"] + 1
    with pytest.raises(StateBindingError) as caught:
        trace.load_state_dict(state)
    assert RunnableErrorCode.STATE_ALIAS_CONFLICT in _codes(caught.value)


def test_failed_restage_preserves_prior_binding_and_success_replaces_atomically(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """Keep the old staged mapping on failure and replace it only after full validation."""

    trace = _load(runnable_artifact)
    first = {name: value.clone() for name, value in runnable_artifact[1].items()}
    trace.load_state_dict(first)
    first_prepared = prepare_runnable_state(trace)
    invalid = dict(first)
    del invalid["linear.bias"]
    with pytest.raises(StateBindingError):
        trace.load_state_dict(invalid)
    after_failure = prepare_runnable_state(trace)
    assert torch.equal(
        after_failure.slot_values["state:linear.weight"],
        first_prepared.slot_values["state:linear.weight"],
    )

    second = {name: value + 2 for name, value in first.items()}
    trace.load_state_dict(second)
    second_prepared = prepare_runnable_state(trace)
    assert torch.equal(second_prepared.slot_values["state:linear.weight"], second["linear.weight"])


@pytest.mark.smoke
def test_n1a_initializes_every_role_deterministically_and_names_every_slot(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """Allocate parameters/buffers per N1-a with isolated seeded determinism."""

    trace = _load(runnable_artifact)
    descriptor = trace.runnable_descriptor
    assert descriptor is not None
    template = next(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.PARAMETER
    )
    roles = tuple(StateSlotRole)
    slots = []
    for index, role in enumerate(roles):
        is_counter = role is StateSlotRole.COUNTER
        binding = template.state_binding
        assert binding is not None
        slots.append(
            replace(
                template,
                slot_id=f"state:test:{role.value}",
                role=(
                    TensorSlotRole.PARAMETER
                    if role
                    in {
                        StateSlotRole.WEIGHT,
                        StateSlotRole.BIAS,
                        StateSlotRole.NORM_SCALE,
                        StateSlotRole.NORM_OFFSET,
                    }
                    else TensorSlotRole.BUFFER
                ),
                shape=(4,) if role is StateSlotRole.WEIGHT else (2,),
                dtype="torch.int64" if is_counter else "torch.float32",
                rank=1,
                state_binding=replace(
                    binding,
                    state_dict_name=f"test.{role.value}",
                    semantic_role=role,
                    trainable=role
                    in {
                        StateSlotRole.WEIGHT,
                        StateSlotRole.BIAS,
                        StateSlotRole.NORM_SCALE,
                        StateSlotRole.NORM_OFFSET,
                    },
                    # r71 E1: the totalized declared fact must stay coherent with the
                    # synthetic role (an int64 counter cannot require grad).
                    captured_requires_grad=role
                    in {
                        StateSlotRole.WEIGHT,
                        StateSlotRole.BIAS,
                        StateSlotRole.NORM_SCALE,
                        StateSlotRole.NORM_OFFSET,
                    },
                    alias_group=None,
                ),
            )
        )
    tied = replace(
        slots[0],
        slot_id="state:test:tied_weight",
        state_binding=replace(slots[0].state_binding, alias_group="tied:test"),
    )
    slots[0] = replace(
        slots[0], state_binding=replace(slots[0].state_binding, alias_group="tied:test")
    )
    trace.__dict__["_runnable_descriptor"] = _with_rebuilt_state_metadata_witnesses(
        replace(descriptor, tensor_slots=tuple(slots) + (tied,))
    )
    global_rng_before = torch.random.get_rng_state().clone()

    first = prepare_runnable_state(trace, seed=1234)
    second = prepare_runnable_state(trace, seed=1234)

    assert first.state_source is StateSource.RANDOM_INITIALIZATION
    assert first.initializer_policy_version == "torchlens_role_init_v2"
    assert first.seed == 1234
    assert set(first.random_filled_slot_ids) == set(first.slot_values)
    assert torch.equal(global_rng_before, torch.random.get_rng_state())
    assert all(
        torch.equal(first.slot_values[key], second.slot_values[key]) for key in first.slot_values
    )
    assert first.slot_values[slots[0].slot_id] is first.slot_values[tied.slot_id]
    assert torch.count_nonzero(first.slot_values["state:test:bias"]) == 0
    assert torch.count_nonzero(first.slot_values["state:test:norm_offset"]) == 0
    assert torch.count_nonzero(first.slot_values["state:test:running_mean"]) == 0
    assert torch.count_nonzero(first.slot_values["state:test:counter"]) == 0
    assert torch.count_nonzero(first.slot_values["state:test:generic_buffer"]) == 0
    assert torch.all(first.slot_values["state:test:norm_scale"] == 1)
    assert torch.all(first.slot_values["state:test:running_var"] == 1)
    assert torch.count_nonzero(first.slot_values["state:test:weight"]) > 0


def test_user_state_overrides_embedded_hook_and_embedded_precedes_random(
    runnable_artifact: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """Exercise the Stage 7 embedded-state hook and effective source precedence."""

    trace = _load(runnable_artifact)
    embedded = {name: value + 1 for name, value in runnable_artifact[1].items()}
    trace.__dict__["_runnable_embedded_state"] = embedded
    prepared_embedded = prepare_runnable_state(trace, seed=7)
    assert prepared_embedded.state_source is StateSource.EMBEDDED_CAPTURE_STATE

    user = {name: value + 2 for name, value in runnable_artifact[1].items()}
    trace.load_state_dict(user)
    prepared_user = prepare_runnable_state(trace, seed=7)
    assert prepared_user.state_source is StateSource.USER_STATE_DICT
    assert torch.equal(prepared_user.slot_values["state:linear.weight"], user["linear.weight"])
