"""Torch implementation of the capture backend Protocol."""

from __future__ import annotations

import dataclasses
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Mapping, cast

import torch

from ... import _state
from ...data_classes.internal_types import FuncExecutionContext
from ...ir.events import OpEvent, TraceBuildState
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...utils.introspection import get_vars_of_type_from_obj
from ...utils.rng import log_current_autocast_state, log_current_rng_states
from ...utils.tensor_utils import safe_copy
from . import _tl
from .aliasing import detect_torch_alias_contract
from .buffer_writes import uninstall_buffer_write_tracker
from .model_prep import _cleanup_model_session, _ensure_model_prepared, _prepare_model_session
from .ops import (
    _get_autograd_saved_stats_for_tensor,
    _walk_output_tensors_with_paths,
    log_function_output_tensors,
)
from .sources import log_source_tensor as _log_source_tensor
from .wrappers import unwrap_torch, wrap_torch

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


class TorchBackend:
    """Adapter from the backend-neutral capture Protocol to TorchLens' torch path."""

    name = "torch"
    supports_backward_capture = True

    def wrap(self, value: object) -> object:
        """Install torch wrappers and return ``value`` unchanged."""
        wrap_torch()
        return value

    def unwrap(self, value: object) -> object:
        """Remove torch wrappers and return ``value`` unchanged."""
        unwrap_torch()
        return value

    def is_wrapped(self, value: object) -> bool:
        """Return whether torch wrappers are currently installed."""
        return _state._is_decorated

    def start_session(self, options: object) -> object:
        """Return the existing options object as the M2 session token."""
        return options

    def prepare_model(self, session: object, model: object) -> object:
        """Apply one-time and per-session model preparation."""
        self.prepare_model_once(model)
        self.prepare_model_session(session, model)
        return model

    def prepare_model_once(self, model: object) -> object:
        """Apply one-time torch model preparation."""
        _ensure_model_prepared(cast(torch.nn.Module, model))
        return model

    def prepare_model_session(self, session: object, model: object) -> object:
        """Apply per-session torch model preparation."""
        optimizer = getattr(session, "_optimizer", None)
        _prepare_model_session(cast(Any, session), cast(torch.nn.Module, model), optimizer)
        return model

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session torch metadata."""
        model: object
        input_tensors: object
        if isinstance(prepared_model, tuple) and len(prepared_model) == 2:
            model, input_tensors = prepared_model
        else:
            model, input_tensors = prepared_model, None
        uninstall_buffer_write_tracker(cast("Trace", session))
        _cleanup_model_session(cast(torch.nn.Module, model), input_tensors)

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Return the existing torch logging context manager."""
        return _state.active_logging(cast("Trace", session))

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return the existing torch pause-logging context manager."""
        return _state.pause_logging()

    def snapshot_rng(self, session: object) -> object:
        """Capture the current torch RNG state."""
        return log_current_rng_states(torch_only=True)

    def snapshot_autocast(self, session: object) -> object:
        """Capture the current torch autocast state."""
        return log_current_autocast_state()

    def log_source_tensor(
        self,
        session: object,
        tensor: object,
        source: str,
        extra_address: str | None = None,
    ) -> None:
        """Log a torch source tensor in the active capture session.

        Parameters
        ----------
        session:
            Active trace.
        tensor:
            Torch tensor to log as a source.
        source:
            Source role, such as ``"input"`` or ``"buffer"``.
        extra_address:
            Optional input or buffer address.

        Returns
        -------
        None
            The trace is updated in place.
        """

        _log_source_tensor(
            cast("Trace", session),
            cast(torch.Tensor, tensor),
            source,
            extra_address,
        )

    def extract_and_mark_outputs(
        self,
        session: object,
        outputs: object,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Extract torch output tensors, deduplicate them, and mark output parents.

        Parameters
        ----------
        session:
            Active trace.
        outputs:
            Raw model output object returned by the captured forward pass.

        Returns
        -------
        tuple[list[torch.Tensor], list[str]]
            Output tensors and their display addresses.
        """

        self_trace = cast("Trace", session)
        if getattr(self_trace, "intervention_ready", False):
            output_tensors_w_addresses_all = [
                (tensor, _container_path_to_address(path), None)
                for tensor, path, container_spec in _walk_output_tensors_with_paths(outputs)
            ]
            output_specs_by_raw_label = {}
            for tensor, path, container_spec in _walk_output_tensors_with_paths(outputs):
                _label_raw = _tl.get_tensor_label(tensor)
                if _label_raw is not None:
                    output_specs_by_raw_label[_label_raw] = (
                        path,
                        container_spec,
                    )
            setattr(self_trace, "_output_container_specs_by_raw_label", output_specs_by_raw_label)
        else:
            output_tensors_w_addresses_all = get_vars_of_type_from_obj(
                outputs,
                torch.Tensor,
                search_depth=5,
                return_addresses=True,
                allow_repeats=True,
            )
        # Remove duplicate addresses (same tensor at multiple output positions).
        addresses_seen = set()
        output_tensors_w_addresses = []
        for entry in output_tensors_w_addresses_all:
            if entry[1] in addresses_seen:
                continue
            output_tensors_w_addresses.append(entry)
            addresses_seen.add(entry[1])

        output_tensors = [t for t, _, _ in output_tensors_w_addresses]
        output_tensor_addresses = [addr for _, addr, _ in output_tensors_w_addresses]

        for t in output_tensors:
            # Only record output_layers during exhaustive pass; fast pass reuses the list.
            # Defensive: user-injected output tensors (raw register_forward_hook
            # returning a fresh tensor, intervention API replacements that don't
            # propagate metadata, etc.) lack _tl labels. Skip them rather than
            # crashing - they aren't in our graph but the experiment can continue.
            _label_raw = _tl.get_tensor_label(t)
            if _label_raw is None:
                continue
            if self_trace.capture_mode in {"exhaustive", "predicate"}:
                self_trace.output_layers.append(_label_raw)
                event = self_trace.capture_events.op_event_by_label_raw.get(_label_raw)
                if event is not None:
                    updated_event = dataclasses.replace(event, is_output_parent=True)
                    self_trace.capture_events.op_event_by_label_raw[_label_raw] = updated_event
                    for index, existing_event in enumerate(self_trace.capture_events.op_events):
                        if existing_event.label_raw == _label_raw:
                            self_trace.capture_events.op_events[index] = updated_event
                            break

        return output_tensors, output_tensor_addresses

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build a minimal record context for Protocol callers."""
        tensor = output if isinstance(output, torch.Tensor) else None
        return RecordContext(
            kind="op",
            label=reserved.label,
            raw_label=reserved.label_raw,
            pass_index=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_event_input.func_name,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=func_event_input.module_stack,
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=tuple(tensor.shape) if tensor is not None else None,
            dtype=DtypeRef.from_value(tensor.dtype) if tensor is not None else None,
            tensor_device=DeviceRef.from_value(tensor.device) if tensor is not None else None,
            tensor_requires_grad=tensor.requires_grad if tensor is not None else None,
            output_index=None,
            is_bottom_level_func=func_event_input.is_bottom_level_func,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=func_event_input.func_call_id,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=tensor.dtype == torch.bool and tensor.dim() == 0
            if tensor is not None
            else None,
            bool_value=bool(tensor.item())
            if tensor is not None and tensor.dtype == torch.bool and tensor.dim() == 0
            else None,
        )

    def detect_in_place_isolation_required(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> bool:
        """Return whether the output is the first positional input object."""
        return len(func_event_input.args) > 0 and id(output) == id(func_event_input.args[0])

    def detect_backend_semantics(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> BackendSemantics:
        """Return torch autograd and mutation semantics for one output."""
        grad_fn_handle = output.grad_fn if isinstance(output, torch.Tensor) else None
        saved_memory, saved_count = (
            _get_autograd_saved_stats_for_tensor(output)
            if isinstance(output, torch.Tensor)
            else (None, None)
        )
        return detect_torch_alias_contract(
            func_event_input,
            backend_grad_handle=grad_fn_handle,
            grad_fn_class_name=type(grad_fn_handle).__name__
            if grad_fn_handle is not None
            else None,
            autograd_memory=saved_memory,
            num_autograd_tensors=saved_count,
            bytes_delta_at_call=0,
            bytes_peak_at_call=0,
        )

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for a torch tensor without deferred materialization."""
        if not isinstance(value, torch.Tensor):
            return TensorRef("", None, None, None, None, None, payload, None, None)
        with self.pause_logging(session):
            memory = value.nelement() * value.element_size()
        return TensorRef(
            label_raw=_tl.get_tensor_label(value) or "",
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            requires_grad=value.requires_grad,
            memory=memory,
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the TorchLens raw tensor label on a torch tensor."""
        if isinstance(value, torch.Tensor):
            _tl.set_tensor_label(value, label)

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is a torch tensor."""
        return isinstance(value, torch.Tensor)

    def is_parameter(self, value: object) -> bool:
        """Return whether ``value`` is a torch parameter."""
        return isinstance(value, torch.nn.Parameter)

    def mark_same_object_candidates(
        self,
        session: object,
        func_event_input: FunctionEventInput,
    ) -> object:
        """Mark the first labeled positional input as a same-object candidate."""
        if not func_event_input.args:
            return {}
        first_arg = func_event_input.args[0]
        if isinstance(first_arg, torch.Tensor) and _tl.get_tensor_label(first_arg) is not None:
            return {id(first_arg): first_arg}
        return {}

    def isolate_same_object_returns(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        raw_output: object,
        premarked_inputs: object,
    ) -> object:
        """Clone a raw output that is the same object as the marked first input."""
        marked = cast(Mapping[int, object], premarked_inputs)
        if id(raw_output) in marked and isinstance(raw_output, torch.Tensor):
            return safe_copy(raw_output)
        return raw_output

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Apply live hooks through the torch intervention runtime."""
        if not isinstance(value, torch.Tensor):
            return value, ()
        from ...intervention.runtime import _apply_live_hooks

        return _apply_live_hooks(value, site=site.site)

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Copy a torch value with logging paused."""
        return safe_copy(value, detach_tensor=not policy.save_grad, save_mode=policy.save_mode)

    def copy_replacement_metadata(self, session: object, src: object, dst: object) -> None:
        """Copy TorchLens replacement metadata between tensors."""
        _tl.copy_replacement_meta(src, dst)

    def emit_function_outputs(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        isolated_output: object,
        output_sites: tuple[object, ...],
        reserved_block: tuple[ReservedLabel, ...],
    ) -> tuple[OpEvent, ...]:
        """Delegate output logging to the existing raw-layer-dict writer."""
        exec_ctx = FuncExecutionContext(
            time_elapsed=0.0,
            rng_states={},
            autocast_state={},
        )
        log_function_output_tensors(
            cast(Any, session),
            cast(Any, func_event_input.func),
            func_event_input.func_name,
            func_event_input.args,
            dict(func_event_input.kwargs),
            func_event_input.arg_copies or (),
            dict(func_event_input.kwarg_copies or {}),
            isolated_output,
            exec_ctx,
            func_event_input.is_bottom_level_func,
            func_event_input.func_call_id,
        )
        return ()

    def finalize_forward_session(self, session: object, trace_state: TraceBuildState) -> None:
        """Finalize torch backend session state after forward materialization."""
        return None


def _container_path_to_address(path: tuple[Any, ...]) -> str:
    """Convert an output path tuple to TorchLens' display address string.

    Parameters
    ----------
    path:
        Path components from path-aware output traversal.

    Returns
    -------
    str
        Dot-separated output address suffix.
    """

    parts: list[str] = []
    for component in path:
        if hasattr(component, "index"):
            parts.append(str(component.index))
        elif hasattr(component, "key"):
            parts.append(str(component.key))
        elif hasattr(component, "name"):
            parts.append(str(component.name))
        else:
            parts.append(str(component))
    return ".".join(parts)


__all__ = ["TorchBackend"]
