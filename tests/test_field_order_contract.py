"""Static FIELD_ORDER / FieldPolicy / portable-state contract tests."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from torchlens import constants
from torchlens._io import FieldPolicy
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace


@dataclass(frozen=True)
class FieldOrderCase:
    """Static contract data for one public field-order surface."""

    name: str
    cls: type[Any]
    field_order: tuple[str, ...]
    portable_only_fields: frozenset[str] = frozenset()
    dropped_display_fields: frozenset[str] = frozenset()


FIELD_ORDER_CASES: tuple[FieldOrderCase, ...] = (
    FieldOrderCase(
        "Trace",
        Trace,
        tuple(constants.MODEL_LOG_FIELD_ORDER),
        # _grad_fn_param_refs, _phase_timings, and _replay_arg_version_data_complete
        # were promoted into MODEL_LOG_FIELD_ORDER (cert10), so only the two
        # remaining portable-only fields stay documented here.
        portable_only_fields=frozenset(
            {
                "_buffer_initial_values",
                "ops_with_params",
            }
        ),
        dropped_display_fields=frozenset(
            {
                "_code_context_cache",
                "_intervention_spec",
                "_last_hook_handle_ids",
                "_out_dedup_mode",
                "_out_hash_cache",
                "_out_identity_cache",
                "_output_transform",
                "_runnable_descriptor",
                "_runnable_embedded_state",
                "_runnable_first_mismatch",
                "_runnable_path_faithfulness",
                "_runnable_poisoned",
                "_runnable_readiness",
                "_runnable_staged_user_state",
                "_source_model_ref",
                "_transform",
                "_visualizer_dir",
                "_warned_direct_write",
                "_warned_mutate_in_place",
                "activation_transform",
                "backward_ready",
                "capture_guard_passes",
                "capture_owner_thread_id",
                "capture_owner_thread_qualified",
                "capture_thread_activity_detected",
                "capture_thread_count_end",
                "capture_thread_count_start",
                "capture_verification_reason",
                "capture_verified",
                "completeness_decompositions",
                "completeness_diagnostics",
                "completeness_witness_accounted_count",
                "completeness_witness_callback_ns",
                "completeness_witness_event_count",
                "completeness_witness_expected_opaque_count",
                "completeness_witness_mode",
                "completeness_witness_unaccounted_count",
                "completeness_witness_verified",
                "detached_patch_epoch",
                "detached_patch_policy",
                "escape_detector_backward_coverage",
                "escape_detector_callback_ns",
                "escape_detector_event_count",
                "escape_detector_mode",
                "escape_detector_verified",
                "escape_diagnostics",
                "facet_registry_snapshot",
                "grad_transform",
                "last_run",
                "layer_visualizers",
                "module_filter",
                "num_modules",
                "parent_run",
                "replay_frontier",
            }
        ),
    ),
    FieldOrderCase(
        "Op",
        Op,
        tuple(constants.OP_LOG_FIELD_ORDER),
        portable_only_fields=frozenset({"_grad_records", "_is_in_conditional_body"}),
        dropped_display_fields=frozenset(
            {
                "_construction_done",
                "activation_transform",
                "arg_expressions",
                "args_template",
                "func",
                "grad_fn",
                "grad_fn_handle",
                "input_activations",
                "input_dtypes",
                "input_memory",
                "input_ops",
                "input_shapes",
                # "interventions" became a KEEP (portable) ordered field in
                # cert10, so it is no longer a documented DROP display field.
                "is_internally_initialized",
                "kwargs_template",
                "num_inputs",
                "raw_label",
                "source_trace",
            }
        ),
    ),
    FieldOrderCase(
        "Layer",
        Layer,
        tuple(constants.LAYER_LOG_FIELD_ORDER),
        # The buffer_*/io_role/is_atomic_module/output_of_* fields were promoted
        # into LAYER_LOG_FIELD_ORDER (cert10); only the private conditional-body
        # flag remains portable-only.
        portable_only_fields=frozenset({"_is_in_conditional_body"}),
        dropped_display_fields=frozenset(
            {"activation_transform", "func", "grad_fn", "grad_fn_handle"}
        ),
    ),
    FieldOrderCase(
        "Module",
        Module,
        tuple(constants.MODULE_LOG_FIELD_ORDER),
        dropped_display_fields=frozenset({"cls"}),
    ),
    FieldOrderCase(
        "ModuleCall",
        ModuleCall,
        tuple(constants.MODULE_PASS_LOG_FIELD_ORDER),
        dropped_display_fields=frozenset(
            {"cls", "forward_args_template", "forward_kwargs_template"}
        ),
    ),
    FieldOrderCase(
        "Param",
        Param,
        tuple(constants.PARAM_LOG_FIELD_ORDER),
        # _derived_grad_record_path, all_addresses, and all_module_addresses were
        # promoted into PARAM_LOG_FIELD_ORDER (cert10).
        portable_only_fields=frozenset(
            {
                "_derived_grad_payload",
                "_grad_dtype",
                "_grad_memory",
                "_grad_shape",
                "_has_grad",
            }
        ),
    ),
    FieldOrderCase(
        "Buffer",
        Buffer,
        tuple(constants.BUFFER_LOG_FIELD_ORDER),
        # module_address and versions were promoted into BUFFER_LOG_FIELD_ORDER
        # (cert10).
        portable_only_fields=frozenset({"_initial_value"}),
    ),
    FieldOrderCase("GradFn", GradFn, tuple(constants.GRAD_FN_LOG_FIELD_ORDER)),
    FieldOrderCase(
        "GradFnCall",
        GradFnCall,
        tuple(constants.GRAD_FN_PASS_LOG_FIELD_ORDER),
        portable_only_fields=frozenset({"label"}),
    ),
    FieldOrderCase(
        "FuncCallLocation",
        FuncCallLocation,
        tuple(constants.FUNC_CALL_LOCATION_FIELD_ORDER),
        portable_only_fields=frozenset(
            {
                "_call_line",
                "_code_context",
                "_code_context_labeled",
                "_func_docstring",
                "_func_signature",
                "_num_context_lines",
                "_num_context_lines_requested",
                "_source_context",
                "_source_loaded",
            }
        ),
    ),
)


def test_field_order_names_are_unique_and_backed() -> None:
    """Every FIELD_ORDER name maps to a field, property, or portable spec entry."""

    for case in FIELD_ORDER_CASES:
        duplicates = _duplicates(case.field_order)
        assert not duplicates, f"{case.name} FIELD_ORDER duplicates: {duplicates}"

        portable_fields = set(case.cls.PORTABLE_STATE_SPEC)
        class_fields = set(dir(case.cls))
        missing_backing = [
            field
            for field in case.field_order
            if field not in portable_fields and field not in class_fields
        ]
        assert not missing_backing, f"{case.name} FIELD_ORDER has unbacked names: {missing_backing}"


def test_portable_state_display_fields_are_in_field_order_or_documented() -> None:
    """Portable non-DROP display fields must be ordered or explicitly portable-only."""

    for case in FIELD_ORDER_CASES:
        portable_display_fields = {
            field
            for field, policy in case.cls.PORTABLE_STATE_SPEC.items()
            if policy not in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}
        }
        missing_from_order = portable_display_fields - set(case.field_order)
        assert missing_from_order == case.portable_only_fields, (
            f"{case.name} portable/display drift: {sorted(missing_from_order)}"
        )


def test_field_order_drop_policy_fields_are_documented() -> None:
    """FIELD_ORDER may include DROP fields, but each one must be intentional."""

    for case in FIELD_ORDER_CASES:
        dropped_in_order = {
            field
            for field in case.field_order
            if case.cls.PORTABLE_STATE_SPEC.get(field)
            in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}
        }
        assert dropped_in_order == case.dropped_display_fields, (
            f"{case.name} undocumented DROP/WEAKREF_STRIP FIELD_ORDER fields: "
            f"{sorted(dropped_in_order)}"
        )


def test_known_glossary_aliases_have_backed_field_order_names() -> None:
    """Back-compat aliases are reconciled to current FIELD_ORDER names."""

    assert "activation_memory" in constants.OP_LOG_FIELD_ORDER
    assert "activation_memory" in constants.LAYER_LOG_FIELD_ORDER
    assert "activation_memory" in constants.BUFFER_LOG_FIELD_ORDER
    assert "memory" not in constants.OP_LOG_FIELD_ORDER
    assert "memory" not in constants.LAYER_LOG_FIELD_ORDER

    assert "buffer_overwrite_index" in constants.BUFFER_LOG_FIELD_ORDER
    assert "buffer_pass" in constants.BUFFER_LOG_FIELD_ORDER
    assert hasattr(Buffer, "buffer_overwrite_index")
    assert hasattr(Buffer, "buffer_pass")

    assert "has_op" in constants.GRAD_FN_LOG_FIELD_ORDER
    assert "is_intervening" not in constants.GRAD_FN_LOG_FIELD_ORDER
    assert "has_op" in GradFn.PORTABLE_STATE_SPEC


def _duplicates(values: Collection[str]) -> list[str]:
    """Return duplicate strings in first-repeated order.

    Parameters
    ----------
    values:
        Strings to scan.

    Returns
    -------
    list[str]
        Duplicate values, each reported once.
    """

    seen: set[str] = set()
    duplicated: list[str] = []
    duplicated_set: set[str] = set()
    for value in values:
        if value in seen and value not in duplicated_set:
            duplicated.append(value)
            duplicated_set.add(value)
        seen.add(value)
    return duplicated
