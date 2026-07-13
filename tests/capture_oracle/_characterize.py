"""Capture characterization and measurement helpers for the Stage-0 oracle."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib
import random
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import Recording
from torchlens.ir import CaptureEvents
from torchlens.ir.events import OpEvent

from ._models import build_model_case

_SEED = 20260710
_MEASUREMENT_REPETITIONS = 3
_GROUND_TRUTH_EXCLUDED_POPULATION_PATHS = frozenset({"function.func_id"})

_PREDICATE_WART_PREFIXES = (
    "parents[",
    "parent_arg_positions",
    "_edge_uses",
    "params",
    "parent_params",
    "function.code_context",
    "function.func_duration",
    "function.flops_forward",
    "function.flops_backward",
    "function.func_rng_states",
    "function.func_autocast_state",
    "function.arg_names",
    "function.func_non_tensor_args",
    "function.is_inplace",
    "function.func_config",
    "backend_semantics.autograd_memory",
    "backend_semantics.num_autograd_tensors",
    "backend_semantics.bytes_delta_at_call",
    "backend_semantics.bytes_peak_at_call",
    "equivalence_class",
    "has_internal_source_ancestor",
    "internal_source_ancestors",
    "input_ancestors",
    "root_ancestors",
)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One model/configuration point in the characterization matrix."""

    name: str
    model_axis: str
    config: str
    expected_forward_invocations: int
    features: tuple[str, ...] = ()


CASES: tuple[CaseSpec, ...] = (
    CaseSpec("plain_cnn__exhaustive", "plain_cnn", "exhaustive", 1),
    CaseSpec("plain_cnn__predicate_live", "plain_cnn", "predicate_live", 1),
    CaseSpec("plain_cnn__record", "plain_cnn", "record", 1),
    CaseSpec("plain_cnn__two_pass_negative", "plain_cnn", "two_pass_negative", 1),
    CaseSpec("plain_cnn__mixed_selector", "plain_cnn", "mixed_selector", 1),
    CaseSpec("plain_cnn__lookback", "plain_cnn", "lookback_trace", 1, ("lookback",)),
    CaseSpec("plain_cnn__intervene_trace", "plain_cnn", "intervene_trace", 1, ("intervene",)),
    CaseSpec("plain_cnn__intervene_record", "plain_cnn", "intervene_record", 1, ("intervene",)),
    CaseSpec("plain_cnn__halt_trace", "plain_cnn", "halt_trace", 1, ("halt",)),
    CaseSpec("plain_cnn__halt_record", "plain_cnn", "halt_record", 1, ("halt",)),
    CaseSpec("plain_cnn__backward_trace", "plain_cnn", "backward_trace", 1, ("backward",)),
    CaseSpec("plain_cnn__backward_record", "plain_cnn", "backward_record", 1, ("backward",)),
    CaseSpec("plain_cnn__disk_exhaustive", "plain_cnn", "disk_exhaustive", 1, ("disk",)),
    CaseSpec("plain_cnn__disk_predicate", "plain_cnn", "disk_predicate", 1, ("disk",)),
    CaseSpec("plain_cnn__disk_record", "plain_cnn", "disk_record", 1, ("disk",)),
    CaseSpec("train_batchnorm__exhaustive", "train_batchnorm", "exhaustive", 1),
    CaseSpec("train_batchnorm__predicate_live", "train_batchnorm", "predicate_live", 1),
    CaseSpec(
        "train_batchnorm__two_pass_negative",
        "train_batchnorm",
        "two_pass_negative",
        1,
    ),
    CaseSpec("recurrent__exhaustive", "recurrent", "exhaustive", 1),
    CaseSpec("recurrent__record", "recurrent", "record", 1),
    CaseSpec("recurrent__two_pass_negative", "recurrent", "two_pass_negative", 1),
    CaseSpec("conditional__exhaustive", "conditional", "exhaustive", 1),
    CaseSpec("conditional__predicate_live", "conditional", "predicate_live", 1),
    CaseSpec("in_place__exhaustive", "in_place", "exhaustive", 1),
    CaseSpec("in_place__predicate_live", "in_place", "predicate_live", 1),
    CaseSpec("mutating_pre_hook__exhaustive", "mutating_pre_hook", "exhaustive", 1),
    CaseSpec(
        "mutating_pre_hook__two_pass_negative",
        "mutating_pre_hook",
        "two_pass_negative",
        1,
    ),
    CaseSpec("tiny_transformer__exhaustive", "tiny_transformer", "exhaustive", 1),
    CaseSpec("tiny_transformer__record", "tiny_transformer", "record", 1),
    CaseSpec(
        "failing_conditional__trace",
        "failing_conditional",
        "failed_trace",
        1,
        ("failed",),
    ),
    CaseSpec(
        "failing_conditional__record",
        "failing_conditional",
        "failed_record",
        1,
        ("failed",),
    ),
)

CASE_BY_NAME = {case.name: case for case in CASES}


@dataclass(slots=True)
class _Instrumentation:
    """Test-only observers installed around one capture call."""

    producer_modes: list[str]
    event_snapshots: list[list[dict[str, Any]]]
    restore_callbacks: list[Callable[[], None]]

    def restore(self) -> None:
        """Restore every monkeypatched observer target."""

        for callback in reversed(self.restore_callbacks):
            callback()


def _digest_chunks(data: bytes) -> list[str]:
    """Return a scanner-safe SHA256 digest split into short chunks.

    Parameters
    ----------
    data:
        Bytes to fingerprint.

    Returns
    -------
    list[str]
        Eight-character digest chunks.
    """

    digest = hashlib.sha256(data).hexdigest()
    return [digest[index : index + 8] for index in range(0, len(digest), 8)]


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    """Return a stable metadata and value fingerprint for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to fingerprint.

    Returns
    -------
    dict[str, Any]
        JSON-compatible tensor fingerprint.
    """

    detached = tensor.detach().cpu().contiguous()
    byte_view = detached.reshape(-1).view(torch.uint8)
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "sha256_chunks": _digest_chunks(byte_view.numpy().tobytes()),
    }


def _rng_fingerprint() -> list[str]:
    """Return a fingerprint of the current CPU RNG state."""

    return _digest_chunks(torch.get_rng_state().numpy().tobytes())


def _population_state(value: Any) -> str:
    """Classify a captured field as populated, unknown, or defaulted.

    Parameters
    ----------
    value:
        Captured field value.

    Returns
    -------
    str
        Stable population category.
    """

    if value is None:
        return "unknown"
    if value is False:
        return "defaulted_false"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return "defaulted_zero"
    if isinstance(value, (tuple, list, dict, set, frozenset)) and not value:
        return "defaulted_empty"
    return "populated"


def _flatten_nested_population(value: Any, prefix: str, result: dict[str, str]) -> None:
    """Flatten selected nested dataclass population into dotted paths.

    Parameters
    ----------
    value:
        Value to flatten.
    prefix:
        Current field path.
    result:
        Destination population mapping.
    """

    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            child = getattr(value, field_info.name)
            _flatten_nested_population(child, f"{prefix}.{field_info.name}", result)
        return
    if (
        isinstance(value, (tuple, list))
        and value
        and all(is_dataclass(item) and not isinstance(item, type) for item in value)
    ):
        for index, item in enumerate(value):
            _flatten_nested_population(item, f"{prefix}[{index}]", result)
        return
    result[prefix] = _population_state(value)


def _event_population(event: OpEvent) -> dict[str, str]:
    """Return flattened population states for every top-level OpEvent field.

    Parameters
    ----------
    event:
        Operation event to inspect.

    Returns
    -------
    dict[str, str]
        Field path to population category.
    """

    nested_fields = {
        "function",
        "output",
        "templates",
        "parents",
        "module_stack",
        "backend_semantics",
        "policy",
    }
    result: dict[str, str] = {}
    for field_info in fields(event):
        value = getattr(event, field_info.name)
        if field_info.name in nested_fields:
            _flatten_nested_population(value, field_info.name, result)
        else:
            result[field_info.name] = _population_state(value)
    for path in _GROUND_TRUTH_EXCLUDED_POPULATION_PATHS:
        result.pop(path, None)
    return dict(sorted(result.items()))


def _event_identity(event: OpEvent) -> dict[str, Any]:
    """Project faithful operation identity, topology, and output facts.

    Parameters
    ----------
    event:
        Operation event to project.

    Returns
    -------
    dict[str, Any]
        Stable event identity projection.
    """

    tensor_ref = event.output.tensor
    return {
        "label_raw": event.label_raw,
        "kind": event.kind,
        "layer_type": event.layer_type,
        "raw_index": event.raw_index,
        "type_index": event.type_index,
        "step_index": event.step_index,
        "pass_index": event.pass_index,
        "func_name": event.function.func_name,
        "parent_labels_raw": [edge.parent_label_raw for edge in event.parents],
        "shape": None if tensor_ref.shape is None else list(tensor_ref.shape),
        "dtype": None if tensor_ref.dtype is None else str(tensor_ref.dtype).removeprefix("torch."),
        "has_saved_activation": event.output.has_saved_activation,
        "predicate_matched": event.predicate_matched,
        "is_output_parent": event.is_output_parent,
        "is_bottom_level": event.is_bottom_level,
        "is_scalar_bool": event.is_scalar_bool,
        "bool_value": event.bool_value,
        "intervention_fired": event.intervention_fired,
        "intervention_replaced": event.intervention_replaced,
    }


def _is_predicate_wart_path(path: str) -> bool:
    """Return whether a population path is a documented lossy-producer wart.

    Parameters
    ----------
    path:
        Flattened event field path.

    Returns
    -------
    bool
        Whether the path is expected to change during unification.
    """

    if path.startswith("parents["):
        return path.endswith(".arg_position") or path.endswith(".edge_use")
    return path.startswith(_PREDICATE_WART_PREFIXES[1:])


def _project_event(event: OpEvent) -> dict[str, Any]:
    """Project one event before postprocess drains its source buffer.

    Parameters
    ----------
    event:
        Operation event to project.

    Returns
    -------
    dict[str, Any]
        Identity and field-population projection.
    """

    population = _event_population(event)
    return {
        "identity": _event_identity(event),
        "population": population,
        "predicate_wart_population": {
            path: state for path, state in population.items() if _is_predicate_wart_path(path)
        },
        "predicate_wart_values": {
            "params_count": len(event.params),
            "parent_params_count": len(event.parent_params),
            "parent_arg_positions": [edge.arg_position for edge in event.parents],
            "parent_edge_uses": [edge.edge_use for edge in event.parents],
            "is_inplace": event.function.is_inplace,
        },
    }


def _snapshot_events(events: CaptureEvents) -> list[dict[str, Any]]:
    """Snapshot operation events without retaining live tensor references.

    Parameters
    ----------
    events:
        Capture event buffer about to be materialized.

    Returns
    -------
    list[dict[str, Any]]
        Raw-order operation event projections.
    """

    return [_project_event(event) for event in events.op_events if event.kind == "op"]


def _install_instrumentation() -> _Instrumentation:
    """Install read-only test observers around producer selection and materialization.

    Returns
    -------
    _Instrumentation
        Observer state and restoration callbacks.
    """

    producer_modes: list[str] = []
    event_snapshots: list[list[dict[str, Any]]] = []
    restore_callbacks: list[Callable[[], None]] = []

    ops_module = importlib.import_module("torchlens.backends.torch.ops")
    original_set_policy = ops_module.set_capture_producer_policy

    def observing_set_policy(trace: Any, mode: str) -> None:
        """Record a producer selection and delegate unchanged."""

        if not producer_modes or producer_modes[-1] != mode:
            producer_modes.append(mode)
        original_set_policy(trace, mode)

    ops_module.set_capture_producer_policy = observing_set_policy

    def restore_policy() -> None:
        """Restore the producer-policy function."""

        ops_module.set_capture_producer_policy = original_set_policy

    restore_callbacks.append(restore_policy)

    postprocess_module = importlib.import_module("torchlens.postprocess")
    materialize_module = importlib.import_module("torchlens.postprocess._materialize")
    original_public_materialize = postprocess_module.materialize_from_events
    original_direct_materialize = materialize_module.materialize_from_events

    def observing_materialize(trace: Any, events: CaptureEvents) -> None:
        """Snapshot immutable field population and delegate unchanged."""

        event_snapshots.append(_snapshot_events(events))
        original_direct_materialize(trace, events)

    postprocess_module.materialize_from_events = observing_materialize
    materialize_module.materialize_from_events = observing_materialize

    def restore_materialize() -> None:
        """Restore both materialization references."""

        postprocess_module.materialize_from_events = original_public_materialize
        materialize_module.materialize_from_events = original_direct_materialize

    restore_callbacks.append(restore_materialize)
    return _Instrumentation(producer_modes, event_snapshots, restore_callbacks)


def _buffer_snapshot(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Snapshot registered buffer values and mutation versions.

    Parameters
    ----------
    model:
        Model whose buffers are observed.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-buffer fingerprints, versions, and scalar values.
    """

    result: dict[str, dict[str, Any]] = {}
    for name, buffer in model.named_buffers():
        scalar_value: int | float | None = None
        if buffer.numel() == 1:
            scalar_value = buffer.detach().cpu().item()
        result[name] = {
            "fingerprint": _tensor_fingerprint(buffer),
            "version": int(buffer._version),
            "scalar_value": scalar_value,
        }
    return result


def _buffer_changes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Describe buffer mutations between two snapshots.

    Parameters
    ----------
    before, after:
        Buffer snapshots surrounding a capture.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-buffer mutation facts.
    """

    changes: dict[str, dict[str, Any]] = {}
    for name in sorted(before.keys() | after.keys()):
        old = before.get(name)
        new = after.get(name)
        if old is None or new is None:
            changes[name] = {"present_before": old is not None, "present_after": new is not None}
            continue
        old_scalar = old["scalar_value"]
        new_scalar = new["scalar_value"]
        numeric_delta = None
        if old_scalar is not None and new_scalar is not None:
            numeric_delta = new_scalar - old_scalar
        changes[name] = {
            "value_changed": old["fingerprint"] != new["fingerprint"],
            "version_delta": new["version"] - old["version"],
            "numeric_delta": numeric_delta,
            "before": old["fingerprint"],
            "after": new["fingerprint"],
        }
    return changes


def _save_all_ops(ctx: Any) -> bool:
    """Select all operation contexts for predicate capture."""

    return getattr(ctx, "kind", None) == "op"


def _halt_on_relu(ctx: Any) -> bool:
    """Halt immediately after the first ReLU operation."""

    return getattr(ctx, "kind", None) == "op" and getattr(ctx, "func_name", None) == "relu"


def _run_capture(
    case: CaseSpec,
    model: nn.Module,
    input_tensor: torch.Tensor,
    disk_path: Path,
) -> tuple[Any, BaseException | None, Any | None]:
    """Execute one configured public capture surface.

    Parameters
    ----------
    case:
        Matrix case specification.
    model:
        Fresh model.
    input_tensor:
        Fresh model input.
    disk_path:
        Temporary bundle destination for disk cases.

    Returns
    -------
    tuple[Any, BaseException | None, Any | None]
        Capture product, optional exception, and optional model output.
    """

    config = case.config
    if config == "exhaustive":
        return tl.trace(model, input_tensor, layers_to_save="all", random_seed=_SEED), None, None
    if config == "predicate_live":
        return tl.trace(model, input_tensor, save=_save_all_ops, random_seed=_SEED), None, None
    if config == "record":
        recording = tl.record(
            model,
            input_tensor,
            save=_save_all_ops,
            include_source_events=True,
            random_seed=_SEED,
        )
        return recording, None, None
    if config == "two_pass_negative":
        return tl.trace(model, input_tensor, layers_to_save=[-1], random_seed=_SEED), None, None
    if config == "mixed_selector":
        return tl.trace(model, input_tensor, layers_to_save=[1, -1], random_seed=_SEED), None, None
    if config == "lookback_trace":
        selector = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
        trace = tl.trace(
            model,
            input_tensor,
            save=selector,
            lookback=4,
            lookback_payload_policy="detached_raw",
            random_seed=_SEED,
        )
        return trace, None, None
    if config == "intervene_trace":
        trace = tl.trace(
            model,
            input_tensor,
            save=_save_all_ops,
            intervene=tl.when(tl.func("relu"), tl.add(0.0)),
            random_seed=_SEED,
        )
        return trace, None, None
    if config == "intervene_record":
        recording = tl.record(
            model,
            input_tensor,
            save=_save_all_ops,
            intervene=tl.when(tl.func("relu"), tl.add(0.0)),
            include_source_events=True,
            random_seed=_SEED,
        )
        return recording, None, None
    if config == "halt_trace":
        trace = tl.trace(
            model,
            input_tensor,
            save=_save_all_ops,
            halt=_halt_on_relu,
            random_seed=_SEED,
        )
        return trace, None, None
    if config == "halt_record":
        recording = tl.record(
            model,
            input_tensor,
            save=_save_all_ops,
            halt=_halt_on_relu,
            include_source_events=True,
            random_seed=_SEED,
        )
        return recording, None, None
    if config == "backward_trace":
        trace = tl.trace(
            model,
            input_tensor.requires_grad_(True),
            layers_to_save="all",
            save_grads="all",
            backward_ready=True,
            random_seed=_SEED,
        )
        output_op = trace[trace.output_layers[0]]
        if not getattr(output_op, "has_saved_activation", False):
            raise AssertionError("backward trace output activation was not saved")
        trace.log_backward(output_op.out.sum())
        return trace, None, None
    if config == "backward_record":
        output, recording = cast(
            tuple[torch.Tensor, Recording],
            tl.record(
                model,
                input_tensor.requires_grad_(True),
                save=_save_all_ops,
                save_grads=True,
                backward_ready=True,
                include_source_events=True,
                return_output=True,
                random_seed=_SEED,
            ),
        )
        recording.log_backward(output.sum(), save_grads=True)
        return recording, None, output
    if config == "disk_exhaustive":
        trace = tl.trace(
            model,
            input_tensor,
            layers_to_save="all",
            storage=tl.to_disk(disk_path, retain_in_memory=True),
            random_seed=_SEED,
        )
        return trace, None, None
    if config == "disk_predicate":
        trace = tl.trace(
            model,
            input_tensor,
            save=_save_all_ops,
            storage=tl.to_disk(disk_path, retain_in_memory=True),
            random_seed=_SEED,
        )
        return trace, None, None
    if config == "disk_record":
        recording = tl.record(
            model,
            input_tensor,
            save=_save_all_ops,
            storage=tl.to_disk(disk_path, retain_in_memory=True),
            include_source_events=True,
            random_seed=_SEED,
        )
        return recording, None, None
    if config == "failed_record":
        recording = tl.record(
            model,
            input_tensor,
            save=_save_all_ops,
            include_source_events=True,
            on_forward_error="return_partial",
            random_seed=_SEED,
        )
        return recording, RuntimeError("capture-oracle intentional forward failure"), None
    if config == "failed_trace":
        try:
            tl.trace(model, input_tensor, layers_to_save="all", random_seed=_SEED)
        except RuntimeError as exc:
            return tl.partial.from_failed_capture(exc), exc, None
        raise AssertionError("intentional failing trace unexpectedly completed")
    raise KeyError(f"unknown capture-oracle config: {config}")


def _events_from_product(product: Any) -> list[dict[str, Any]]:
    """Project retained operation events from a capture product.

    Parameters
    ----------
    product:
        Trace, Recording, or PartialTrace capture product.

    Returns
    -------
    list[dict[str, Any]]
        Per-event projections.
    """

    events = getattr(product, "op_events", ())
    return [_project_event(event) for event in events if event.kind == "op"]


def _final_op_projection(product: Any) -> list[dict[str, Any]]:
    """Project final user-facing operation topology.

    Parameters
    ----------
    product:
        Trace, Recording, or PartialTrace capture product.

    Returns
    -------
    list[dict[str, Any]]
        Stable final operation rows.
    """

    trace = getattr(product, "trace", product)
    layers: Iterable[Any]
    if hasattr(product, "raw_layers"):
        layers = product.raw_layers
    elif hasattr(trace, "layer_list"):
        layers = trace.layer_list
    else:
        return []
    rows: list[dict[str, Any]] = []
    for op in layers:
        rows.append(
            {
                "label": getattr(op, "layer_label", getattr(op, "_label_raw", None)),
                "label_raw": getattr(op, "_label_raw", None),
                "layer_type": getattr(op, "layer_type", None),
                "func_name": getattr(op, "func_name", None),
                "parents": list(getattr(op, "parents", ()) or ()),
                "children": list(getattr(op, "children", ()) or ()),
                "shape": None if getattr(op, "shape", None) is None else list(getattr(op, "shape")),
                "dtype": None
                if getattr(op, "dtype", None) is None
                else str(getattr(op, "dtype")).removeprefix("torch."),
                "has_saved_activation": bool(getattr(op, "has_saved_activation", False)),
                "is_output": bool(getattr(op, "is_output", False)),
                "is_inplace": bool(getattr(op, "is_inplace", False)),
            }
        )
    return rows


def _payload_projection(product: Any) -> list[dict[str, Any]]:
    """Project selected activation payload labels, metadata, and values.

    Parameters
    ----------
    product:
        Trace, Recording, or PartialTrace capture product.

    Returns
    -------
    list[dict[str, Any]]
        Ordered selected-payload projection.
    """

    payloads: list[dict[str, Any]] = []
    if isinstance(product, Recording):
        for record in product.records:
            if record.ctx.kind != "op" or record.ram_payload is None:
                continue
            payloads.append(
                {
                    "label_raw": record.ctx.raw_label or record.ctx.label,
                    "pass_index": record.ctx.pass_index,
                    "tensor": _tensor_fingerprint(record.ram_payload),
                }
            )
        return payloads
    trace = getattr(product, "trace", product)
    layers = (
        product.raw_layers if hasattr(product, "raw_layers") else getattr(trace, "layer_list", ())
    )
    for op in layers:
        if not getattr(op, "has_saved_activation", False):
            continue
        out = op.out
        if not isinstance(out, torch.Tensor):
            continue
        payloads.append(
            {
                "label_raw": getattr(op, "_label_raw", None),
                "pass_index": getattr(op, "pass_index", None),
                "tensor": _tensor_fingerprint(out),
            }
        )
    return payloads


def _gradient_projection(product: Any) -> dict[str, Any]:
    """Project backward event and saved-gradient facts.

    Parameters
    ----------
    product:
        Trace or Recording capture product.

    Returns
    -------
    dict[str, Any]
        Backward sidecar and payload summary.
    """

    backward_events = getattr(product, "backward_events", ())
    event_types = [type(event).__name__ for event in backward_events]
    if isinstance(product, Recording):
        payloads = [
            {
                "label": record.ctx.effective_label,
                "kind": record.ctx.grad_kind,
                "tensor": _tensor_fingerprint(record.ram_payload),
            }
            for record in product.grad_records
            if record.ram_payload is not None
        ]
    else:
        payloads = []
        for op in getattr(product, "layer_list", ()):
            for grad in getattr(op, "grads", ()):
                if isinstance(grad, torch.Tensor):
                    payloads.append(
                        {
                            "label": getattr(op, "layer_label", None),
                            "tensor": _tensor_fingerprint(grad),
                        }
                    )
    return {"event_types": event_types, "saved_payloads": payloads}


def _outcome_projection(product: Any, exception: BaseException | None) -> dict[str, Any]:
    """Project complete, halted, or failed terminal state and partial fields.

    Parameters
    ----------
    product:
        Capture product.
    exception:
        Original forward exception, if any.

    Returns
    -------
    dict[str, Any]
        Stable terminal outcome projection.
    """

    if isinstance(product, Recording):
        return {
            "status": product.status,
            "failed": product.failed,
            "halted": product.halted,
            "halt_reason": product.halt_reason,
            "n_ops_completed": product.n_ops_completed,
            "last_successful_op_label": product.last_successful_op_label,
            "last_event_label": product.last_event_label,
            "last_event_func": product.last_event_func,
            "error_type": None
            if not product.failed or exception is None
            else type(exception).__name__,
            "error_message": None if not product.failed or exception is None else str(exception),
        }
    if hasattr(product, "raw_layers"):
        return {
            "status": "failed",
            "failed": True,
            "halted": False,
            "partial_raw_layer_count": len(product.raw_layers),
            "error_type": None if exception is None else type(exception).__name__,
            "error_message": None if exception is None else str(exception),
        }
    if exception is not None:
        return {
            "status": "failed",
            "failed": True,
            "halted": False,
            "partial_product": product is not None,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        }
    halted = bool(getattr(product, "halted", False))
    return {
        "status": "halted" if halted else "complete",
        "failed": False,
        "halted": halted,
        "halt_reason": getattr(product, "halt_reason", None),
        "halt_frontier": getattr(product, "halt_frontier", None),
        "output_layers": list(getattr(product, "output_layers", ())),
    }


def _disk_projection(disk_path: Path) -> dict[str, Any]:
    """Project temporary disk-storage artifacts without machine-specific paths.

    Parameters
    ----------
    disk_path:
        Configured bundle path.

    Returns
    -------
    dict[str, Any]
        Existence and relative file-name projection.
    """

    if not disk_path.exists():
        return {"created": False, "files": []}
    if disk_path.is_file():
        return {"created": True, "files": [disk_path.name]}
    return {
        "created": True,
        "files": sorted(
            str(path.relative_to(disk_path)) for path in disk_path.rglob("*") if path.is_file()
        ),
    }


def _split_event_population(
    events: list[dict[str, Any]],
    predicate_producer_ran: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split faithful event facts from explicitly documented predicate warts.

    Parameters
    ----------
    events:
        Full event projections.
    predicate_producer_ran:
        Whether the lossy predicate producer handled this case.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Ground-truth event projections and wart-only observations.
    """

    ground_truth: list[dict[str, Any]] = []
    wart_observations: list[dict[str, Any]] = []
    for event in events:
        population = event["population"]
        wart_population = event["predicate_wart_population"] if predicate_producer_ran else {}
        ground_truth.append(
            {
                "identity": event["identity"],
                "population": {
                    path: state for path, state in population.items() if path not in wart_population
                },
            }
        )
        if predicate_producer_ran:
            wart_observations.append(
                {
                    "label_raw": event["identity"]["label_raw"],
                    "population": wart_population,
                    "values": event["predicate_wart_values"],
                }
            )
    return ground_truth, wart_observations


def _case_uses_predicate_producer(case: CaseSpec) -> bool:
    """Return whether current main routes a case through the predicate producer.

    Parameters
    ----------
    case:
        Matrix case specification.

    Returns
    -------
    bool
        Whether the Stage-0 baseline uses the predicate producer.
    """

    return case.config in {
        "record",
        "intervene_record",
        "halt_record",
        "backward_record",
        "disk_record",
        "failed_record",
    }


def _case_has_stateful_two_pass_outcome_wart(case: CaseSpec) -> bool:
    """Return whether the case owns the stateful two-pass failure carve-out.

    Parameters
    ----------
    case:
        Matrix case specification.

    Returns
    -------
    bool
        Whether the current two-pass train-BatchNorm outcome is expected to change.
    """

    return case.config == "two_pass_negative" and case.model_axis == "train_batchnorm"


def _partial_product_from_exception(exception: Exception) -> Any | None:
    """Recover a capture product attached to an unexpected capture exception.

    Parameters
    ----------
    exception:
        Exception raised by the configured capture call.

    Returns
    -------
    Any | None
        Attached partial recording or trace, when TorchLens exposed one.
    """

    partial_recording = getattr(exception, "partial_recording", None)
    if partial_recording is not None:
        return partial_recording
    if getattr(exception, "partial_log", None) is None:
        return None
    try:
        return tl.partial.from_failed_capture(exception)
    except Exception:
        return None


def _capture_once(case: CaseSpec) -> tuple[dict[str, Any], dict[str, float | int | None]]:
    """Capture one fresh model and return semantics plus tracking measurements.

    Parameters
    ----------
    case:
        Matrix case specification.

    Returns
    -------
    tuple[dict[str, Any], dict[str, float | int | None]]
        Stable semantic characterization and non-absolute measurements.
    """

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    model, input_tensor = build_model_case(case.model_axis)
    buffer_before = _buffer_snapshot(model)
    input_before = _tensor_fingerprint(input_tensor)
    input_version_before = int(input_tensor._version)
    rng_before = _rng_fingerprint()
    instrumentation = _install_instrumentation()

    with tempfile.TemporaryDirectory(prefix="torchlens-capture-oracle-") as temp_dir:
        disk_path = Path(temp_dir) / "capture.tlspec"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracemalloc.start()
        started = time.perf_counter()
        product: Any | None = None
        exception: BaseException | None = None
        try:
            try:
                product, exception, _output = _run_capture(case, model, input_tensor, disk_path)
            except Exception as capture_exception:
                exception = capture_exception
                product = _partial_product_from_exception(capture_exception)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            instrumentation.restore()

        events = instrumentation.event_snapshots[-1] if instrumentation.event_snapshots else []
        if not events:
            events = _events_from_product(product)
        predicate_ran = "predicate" in instrumentation.producer_modes
        predicate_case = _case_uses_predicate_producer(case)
        ground_truth_events, predicate_warts = _split_event_population(events, predicate_case)

        buffer_after = _buffer_snapshot(model)
        side_effects = {
            "forward_invocations": int(getattr(model, "forward_invocations", 0)),
            "pre_hook_invocations": int(getattr(model, "pre_hook_invocations", 0)),
            "rng_draw_count": int(getattr(model, "rng_draw_count", 0)),
            "rng_state_before": rng_before,
            "rng_state_after": _rng_fingerprint(),
            "buffer_changes": _buffer_changes(buffer_before, buffer_after),
            "buffer_mutation_count": sum(
                int(change.get("version_delta", 0))
                for change in _buffer_changes(buffer_before, buffer_after).values()
            ),
            "input_value_before": input_before,
            "input_value_after": _tensor_fingerprint(input_tensor),
            "input_version_delta": int(input_tensor._version) - input_version_before,
            "input_value_changed": input_before != _tensor_fingerprint(input_tensor),
        }
        final_ops = _final_op_projection(product)
        stable_side_effects: dict[str, Any] = {
            "inplace_op_count": sum(row["is_inplace"] for row in final_ops),
        }
        expected_to_change: dict[str, dict[str, Any]] = {
            "producer_path": {
                "reason": "Internal producer identities are migration mechanics, not model facts.",
                "current": instrumentation.producer_modes,
            }
        }
        if case.config == "two_pass_negative":
            expected_to_change["two_pass_double_execution"] = {
                "reason": (
                    "Structure-dependent selectors currently replay the user forward and double "
                    "observable side effects; exactly-once migration must reduce these totals."
                ),
                "current": side_effects,
            }
        else:
            stable_side_effects.update(side_effects)
        if predicate_case:
            expected_to_change["predicate_lossy_event_fields"] = {
                "reason": (
                    "The predicate producer currently emits empty parameter and argument-edge "
                    "metadata plus false/zero/empty richness defaults instead of explicit "
                    "unavailable completeness."
                ),
                "current": predicate_warts,
            }

        if predicate_case and not predicate_ran:
            expected_to_change["predicate_lossy_event_fields"]["legacy_producer_removed"] = True

        outcome = _outcome_projection(product, exception)
        if _case_has_stateful_two_pass_outcome_wart(case):
            expected_to_change["stateful_two_pass_outcome"] = {
                "reason": (
                    "Two-pass double-execution on a stateful (train-mode BatchNorm) model "
                    "diverges on the recorded second pass; exactly-once migration removes the "
                    "second pass and this case should become a clean single-pass success -- a "
                    "later stage that makes it succeed is fixing the wart, not regressing."
                ),
                "current": outcome,
            }

        ground_truth = {
            "events": ground_truth_events,
            "final_ops": final_ops,
            "selected_payloads": _payload_projection(product),
            "gradients": _gradient_projection(product),
            "side_effects": stable_side_effects,
            "disk_storage": _disk_projection(disk_path),
        }
        if not _case_has_stateful_two_pass_outcome_wart(case):
            ground_truth["outcome"] = outcome

        semantics = {
            "schema_version": 1,
            "case": {
                "name": case.name,
                "model_axis": case.model_axis,
                "config": case.config,
                "features": list(case.features),
            },
            "ground_truth": ground_truth,
            "expected_to_change": expected_to_change,
        }
        cuda_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        tracking: dict[str, float | int | None] = {
            "wall_time_ms": elapsed_ms,
            "python_peak_memory_bytes": int(peak_memory),
            "cuda_peak_memory_bytes": cuda_peak,
        }
        return semantics, tracking


def characterize_case(case_name: str) -> dict[str, Any]:
    """Generate a deterministic multi-sample characterization record.

    Parameters
    ----------
    case_name:
        Matrix case identifier.

    Returns
    -------
    dict[str, Any]
        Golden-ready characterization record.

    Raises
    ------
    KeyError
        If the case identifier is unknown.
    AssertionError
        If repeated semantic captures are not deterministic.
    """

    case = CASE_BY_NAME[case_name]
    semantic_samples: list[dict[str, Any]] = []
    tracking_samples: list[dict[str, float | int | None]] = []
    for _ in range(_MEASUREMENT_REPETITIONS):
        semantics, tracking = _capture_once(case)
        semantic_samples.append(semantics)
        tracking_samples.append(tracking)
    first = semantic_samples[0]
    for sample in semantic_samples[1:]:
        if sample != first:
            raise AssertionError(f"non-deterministic semantic characterization for {case_name}")

    result = dict(first)
    result["tracking"] = {
        "sample_count": _MEASUREMENT_REPETITIONS,
        "wall_time_median_ms": statistics.median(
            float(sample["wall_time_ms"]) for sample in tracking_samples
        ),
        "python_peak_memory_median_bytes": int(
            statistics.median(
                int(sample["python_peak_memory_bytes"]) for sample in tracking_samples
            )
        ),
        "cuda_peak_memory_median_bytes": None
        if tracking_samples[0]["cuda_peak_memory_bytes"] is None
        else int(
            statistics.median(
                int(cast(int, sample["cuda_peak_memory_bytes"])) for sample in tracking_samples
            )
        ),
        "torch_version": torch.__version__,
    }
    return result
