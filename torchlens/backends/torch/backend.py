"""Torch implementation of the capture backend Protocol."""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state
from ..._capture.session import capture_session_for
from ..._data_classes.internal_types import FuncExecutionContext
from ..._errors import CaptureGapWarning, CaptureNotActiveError
from ..._ir.events import OpEvent
from ..._ir.intervention import FireResult, FunctionEventInput
from ..._ir.predicate import RecordContext
from ..._ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ..._ir.semantics import BackendSemantics, CapturePolicy
from ..._utils import _tensor_label as _tl
from ..._utils._cuda_probe import _is_cuda_available
from ..._utils._exception_notes import annotate_guarded
from ..._utils.arg_handling import normalize_input_args, safe_copy_input_tree
from ..._utils.input_leaves import flatten_input_tensor_addresses
from ..._utils.rng import log_current_rng_states, set_random_seed, set_rng_from_saved_states
from ..._utils.tensor_utils import safe_copy
from ...errors._base import transparent_contextmanager
from ...recording._types import ModuleStackFrame
from . import _ops_capture_records as _ops_capture_records_mod
from ._completeness_finalize import capture_completeness_witness
from ._completeness_origins import internal_scalar_read
from ._completeness_storage import capture_scalar_escape_warning
from ._held_refs_capture import rebind_held_torch_refs, restore_held_torch_refs
from ._ops_autograd import _get_autograd_saved_stats_for_tensor
from ._ops_emission import log_function_output_tensors
from .aliasing import detect_torch_alias_contract
from .buffer_writes import uninstall_buffer_write_tracker
from .escape_detection import capture_escape_guard
from .model_prep import (
    _cleanup_model_session,
    _ensure_model_prepared,
    _prepare_model_session,
)
from .module_stack import pop_frame, push_existing_frame
from .sources import (
    _collect_input_tensor_leaves,
    _move_input_tensors_to_device,
    log_source_tensor as _log_source_tensor,
)
from .structure_only_belt import structure_only_escape_belt
from .wrappers import unwrap_torch, wrap_torch

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
from torchlens.backends.torch._failed_session import (
    cleanup_failed_forward_session as _cleanup_failed_forward_session_body,
    cleanup_stopped_forward_session as _cleanup_stopped_forward_session_body,
)
from torchlens.backends.torch._outputs import (
    _OUTPUT_ATTRIBUTION_REMEDY as _OUTPUT_ATTRIBUTION_REMEDY,
    _assign_nested_input_value as _assign_nested_input_value,
    _dedupe_output_addresses as _dedupe_output_addresses,
    _is_direct_registered_buffer_output as _is_direct_registered_buffer_output,
    _model_input_output_label as _model_input_output_label,
    _promote_layers_to_save_output_parent as _promote_layers_to_save_output_parent,
    _register_model_output_container_snapshot as _register_model_output_container_snapshot,
    _same_tensor_storage_identity as _same_tensor_storage_identity,
    _tensor_memory_bytes as _tensor_memory_bytes,
    _write_output_parent_blob as _write_output_parent_blob,
    extract_and_mark_outputs as _extract_and_mark_outputs_body,
    finalize_forward_session as _finalize_forward_session_body,
    output_attribution_remedy as output_attribution_remedy,
)


def _record_input_semantics_gaps(
    session: object,
    gaps: list[str] | tuple[str, ...],
    *,
    reason: str,
) -> None:
    """Record and warn about input-boundary semantics that cannot be verified.

    Parameters
    ----------
    session:
        Active capture session receiving private witness diagnostics.
    gaps:
        Human-readable input paths or copy failures.
    reason:
        Stable internal reason identifying the failed input-boundary proof.

    Returns
    -------
    None
        Appends fail-closed witness diagnostics and emits one visible warning.
    """

    if not gaps:
        return
    reports = session.__dict__.setdefault("completeness_diagnostics", [])
    for gap in gaps:
        reports.append(
            {
                "operator": None,
                "reason": reason,
                "input_path": gap,
                "scope": "input_boundary",
                "enforced": False,
            }
        )
    warnings.warn(
        "TorchLens cannot verify the captured model-input semantics "
        f"({reason}); affected path(s): {', '.join(gaps)}. The completeness "
        "verdict is ceilinged at trace.provenance.verified=False.",
        CaptureGapWarning,
        stacklevel=3,
    )


def _get_input_arg_names(model: torch.nn.Module, input_args: list[Any]) -> list[str]:
    """Extract parameter names from the model's forward() signature for the given input args.

    Parameters
    ----------
    model:
        Model whose ``forward`` signature should be inspected.
    input_args:
        Normalized positional inputs.

    Returns
    -------
    list[str]
        Forward parameter names aligned to ``input_args``.

    Notes
    -----
    Inspects the forward method's argspec, strips 'self', and generates synthetic
    names for any *args overflow positions.
    """
    spec = inspect.getfullargspec(model.forward)
    input_arg_names = list(spec.args)
    if "self" in input_arg_names:
        input_arg_names.remove("self")
    input_arg_names = input_arg_names[0 : len(input_args)]
    # Handle *args: generate synthetic names for uncovered positions
    if len(input_arg_names) < len(input_args) and spec.varargs is not None:
        for i in range(len(input_arg_names), len(input_args)):
            input_arg_names.append(f"{spec.varargs}_{i}")
    return input_arg_names


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
        """Return the existing options object as the session token."""
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
        # Last, while wrapped: point module-held pristine torch functions at the
        # wrappers so this capture needs no rescue forward; undone at cleanup.
        rebind_held_torch_refs(session, cast(torch.nn.Module, model))
        return model

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session torch metadata."""
        model: object
        input_tensors: object
        input_objects: object
        if isinstance(prepared_model, tuple) and len(prepared_model) == 3:
            model, input_tensors, input_objects = prepared_model
        elif isinstance(prepared_model, tuple) and len(prepared_model) == 2:
            model, input_tensors = prepared_model
            input_objects = None
        else:
            model, input_tensors, input_objects = prepared_model, None, None

        def cleanup_action() -> None:
            """Run the legacy model teardown at its historical call site."""

            try:
                restore_held_torch_refs(session)
            finally:
                uninstall_buffer_write_tracker(cast("Trace", session))
                _cleanup_model_session(
                    cast("Trace", session),
                    cast(torch.nn.Module, model),
                    input_tensors,
                    input_objects,
                )

        capture_session = capture_session_for(session)
        if capture_session is None:
            cleanup_action()
            return
        capture_session.run_cleanup("model_session", cleanup_action)

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Compose owner-thread/detector guard with the logging context."""

        @transparent_contextmanager
        def guarded_logging() -> Iterator[None]:
            """Enter detector state before enabling wrapper logging."""

            trace = cast("Trace", session)
            with capture_escape_guard(trace), capture_completeness_witness(trace):
                # Layer-1 mode belt: escalated device-neutral escape
                # refusals for structure_only=True sessions; a no-op context
                # on the default path (zero-diff). The plain scalar-escape
                # warning belt hands off to it in-mode (its module notes why).
                with structure_only_escape_belt(trace), capture_scalar_escape_warning(trace):
                    # Plane-W completion authority: a no-op
                    # for unarmed captures; for armed captures it installs the
                    # capture-scoped funcol wait interposition and settles
                    # every funcol boundary's completion evidence on exit.
                    from .funcol import distributed_recording_session

                    with distributed_recording_session(trace), _state.active_logging(trace):
                        # R54 wrapped-epoch check: model prep wrapped torch
                        # BEFORE admission, so a concurrent unwrap_torch()
                        # completing in between (it now holds the admission
                        # lock through teardown) leaves this capture admitted
                        # into an UNWRAPPED process -- the forward would run
                        # with zero logging and return a silently empty Trace.
                        # Refuse loudly instead.
                        if not _state._is_decorated:
                            raise CaptureNotActiveError(
                                "torch wrappers were removed between model "
                                "preparation and capture admission (a "
                                "concurrent unwrap_torch() call)",
                                code="wrappers_removed_before_capture",
                                remedy=(
                                    "do not call unwrap_torch() concurrently "
                                    "with capture entry; re-run tl.trace -- "
                                    "the next capture re-installs the wrappers"
                                ),
                            )
                        yield

        return guarded_logging()

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return the existing torch pause-logging context manager."""
        return _state.pause_logging()

    def setup_inputs_and_device(
        self,
        session: object,
        model: object,
        inputs: object,
        input_kwargs: dict[Any, Any] | None,
    ) -> tuple[list[Any], dict[Any, Any], list[str], object]:
        """Normalize inputs, detect model device, copy args, and extract input arg names.

        Parameters
        ----------
        session:
            Active capture session. Unused for the torch implementation.
        model:
            Torch module being captured.
        inputs:
            Caller-provided positional inputs.
        input_kwargs:
            Caller-provided keyword inputs, or ``None``.

        Returns
        -------
        tuple[list[Any], dict[Any, Any], list[str], object]
            Copied positional inputs, copied keyword inputs, positional input names,
            and the selected model device.

        Notes
        -----
        This is the single place where user-provided inputs are transformed into
        the canonical internal form:
          1. Unwrap DataParallel to get the underlying module.
          2. ``normalize_input_args``: resolve the tuple-vs-single-arg ambiguity
             by inspecting the model's forward() signature.
          3. ``safe_copy_args/kwargs``: clone tensors so in-place device moves
             (in ``_fetch_label_move_input_tensors``) don't mutate the caller's data.
          4. Detect model device from first param or buffer (for auto-moving
             inputs). A model with neither pins no device: ``model_device`` is
             ``None`` and inputs are never moved.
        """
        torch_model = cast(torch.nn.Module, model)
        if isinstance(torch_model, torch.nn.DataParallel):
            torch_model = torch_model.module

        # Resolve ambiguity: is [tensor_a, tensor_b] two args or one list-arg?
        # normalize_input_args checks the model's forward() signature to decide.
        inputs = normalize_input_args(inputs, torch_model)

        if not input_kwargs:
            input_kwargs = {}

        # Detect device from first param or buffer. A model with NO parameters and
        # NO buffers pins no device: eager execution runs each op on its operands'
        # devices, so the inputs must stay exactly where the caller put them
        # (``None`` = no move). The historical ``"cpu"`` fallback silently dragged
        # CUDA inputs to the CPU and computed the whole forward there -- first seen
        # on real H200 hardware when the CUPTI correlation matrix came back empty
        # because the "CUDA" capture had launched zero kernels.
        # Meta state never pins a device either: offload-hooked models
        # (accelerate device_map / cpu/disk offload) hold meta
        # params between forwards, the hooks place inputs on the real
        # execution device themselves, and moving inputs to meta poisons a
        # forward that runs fine unlogged ("Cannot copy out of meta tensor").
        # The first NON-meta param/buffer (mixed dispatch keeps some modules
        # materialized) still pins; an all-meta model pins no device.
        first_param = next((p for p in torch_model.parameters() if p.device.type != "meta"), None)
        first_buffer = next((b for b in torch_model.buffers() if b.device.type != "meta"), None)
        if first_param is not None:
            model_device: object | None = first_param.device
        elif first_buffer is not None:
            model_device = first_buffer.device
        else:
            model_device = None

        # Copy args and kwargs as ONE graph so repeated tensor identity, shared
        # storage, view geometry, strides, and offsets survive caller protection.
        inputs, input_kwargs, input_copy_gaps = safe_copy_input_tree(
            inputs,
            input_kwargs,
            require_distinct_tensor_sites=bool(getattr(session, "is_replayable", False)),
        )
        _record_input_semantics_gaps(
            session,
            input_copy_gaps,
            reason="input_copy_semantics_unverifiable",
        )
        input_arg_names = _get_input_arg_names(torch_model, inputs)

        return inputs, input_kwargs, input_arg_names, model_device

    def _fetch_label_move_input_tensors(
        self,
        session: object,
        inputs: list[Any],
        input_arg_names: list[str],
        input_kwargs: dict[Any, Any],
        model_device: object,
    ) -> tuple[list[Any], list[str]]:
        """Extract all tensors from input args/kwargs, move to model device, and build addresses.

        Parameters
        ----------
        session:
            Active capture session. Unused for the torch implementation.
        inputs:
            Copied positional inputs that may be mutated for internal device moves.
        input_arg_names:
            Forward signature names for positional inputs.
        input_kwargs:
            Copied keyword inputs that may be mutated for internal device moves.
        model_device:
            Device selected by :meth:`setup_inputs_and_device`, or ``None``
            for a device-less (parameter- and buffer-free) model whose inputs
            must stay on their own devices.

        Returns
        -------
        tuple[list[Any], list[str]]
            Input tensor leaves and their source-address labels.

        Notes
        -----
        Handles nested structures (lists, tuples, dicts) up to
        ``INPUT_SEARCH_DEPTH_LIMIT`` with cycle-safe traversal. Reaching that explicit
        limit records the unresolved path and fails the completeness witness closed.
        Each tensor gets a hierarchical address string like ``"input.x"`` or
        ``"input.x.0.nested"`` that is stored as its ``io_role``.
        """
        input_arg_tensors, input_kwarg_tensors, traversal_gaps = _collect_input_tensor_leaves(
            inputs, input_arg_names, input_kwargs
        )
        _record_input_semantics_gaps(
            session,
            traversal_gaps,
            reason="input_traversal_depth_exceeded",
        )
        _move_input_tensors_to_device(
            inputs,
            input_kwargs,
            input_arg_tensors,
            input_kwarg_tensors,
            model_device,
            _assign_nested_input_value,
        )
        return flatten_input_tensor_addresses(
            input_arg_tensors, input_kwarg_tensors, input_arg_names, input_kwargs
        )

    def snapshot_rng(self, session: object) -> object:
        """Capture the current torch RNG state."""
        return log_current_rng_states(torch_only=True)

    def seed_rng(self, session: object, seed: int) -> None:
        """Seed torch, Python, NumPy, and CUDA RNG engines.

        Parameters
        ----------
        session:
            Active trace session, unused by torch RNG seeding.
        seed:
            Integer seed value.

        Returns
        -------
        None
            Process-local RNG engines are seeded in place.
        """

        del session
        set_random_seed(seed)

    def set_capture_producer_policy(self, session: object, capture_mode: object) -> None:
        """Install torch producer policy metadata on the active trace.

        Parameters
        ----------
        session:
            Active trace session.
        capture_mode:
            Capture mode name.

        Returns
        -------
        None
            Torch producer policy metadata is updated in place.
        """

        _ops_capture_records_mod.set_capture_producer_policy(
            cast("Trace", session), cast(Any, capture_mode)
        )

    def restore_rng(self, session: object, rng_state: object) -> None:
        """Restore a previously captured torch RNG state.

        Parameters
        ----------
        session:
            Active trace session, unused by torch RNG restoration.
        rng_state:
            Opaque RNG snapshot returned by :meth:`snapshot_rng`.

        Returns
        -------
        None
            The process-local torch RNG state is restored in place.
        """

        del session
        set_rng_from_saved_states(cast(dict[str, Any], rng_state))

    def inference_context(self, session: Any) -> AbstractContextManager[None]:
        """Return the torch inference-only context for this session.

        Parameters
        ----------
        session:
            Active trace session whose ``inference_only`` flag controls the context.

        Returns
        -------
        AbstractContextManager[None]
            ``torch.no_grad()`` when requested, otherwise a null context.
        """

        return torch.no_grad() if session._options_inference_only else contextlib.nullcontext()

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

    def push_existing_module_frame(
        self,
        session: object,
        module_stack: list[Any],
        frame: object,
    ) -> None:
        """Push an existing torch module-stack frame."""
        del session
        push_existing_frame(
            cast(list[ModuleStackFrame], module_stack),
            cast(ModuleStackFrame, frame),
        )

    def pop_module_frame(
        self,
        session: object,
        module_stack: list[Any],
        frame: object,
    ) -> None:
        """Pop and validate the current torch module-stack frame."""
        del session
        pop_frame(
            cast(list[ModuleStackFrame], module_stack),
            cast(ModuleStackFrame, frame),
        )

    extract_and_mark_outputs = _extract_and_mark_outputs_body

    @staticmethod
    def _scalar_bool_value(tensor: torch.Tensor | None) -> bool | None:
        """Read a scalar-bool output's value as a capture-internal read, else ``None``."""
        if tensor is None or tensor.dtype != torch.bool or tensor.ndim != 0:
            return None
        with internal_scalar_read():
            return bool(tensor.item())

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
            pass_number=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_event_input.func_name,
            address=None,
            module_type=None,
            module_pass_number=None,
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
            is_scalar_bool=tensor.dtype == torch.bool and tensor.ndim == 0
            if tensor is not None
            else None,
            bool_value=self._scalar_bool_value(tensor),
        )

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
            bytes_delta_at_call=None,
            bytes_peak_at_call=None,
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

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Apply live hooks through the torch intervention runtime."""
        if not isinstance(value, torch.Tensor):
            return value, ()
        from ...intervention._runtime import _apply_live_hooks

        return _apply_live_hooks(value, site=site.site)

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Copy a torch value with logging paused."""
        return safe_copy(value, detach_tensor=not policy.save_grad, save_mode=policy.save_mode)

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

    finalize_forward_session = _finalize_forward_session_body

    cleanup_stopped_forward_session = _cleanup_stopped_forward_session_body

    @staticmethod
    def _warn_without_masking(
        exc: BaseException,
        message: str,
        category: type[Warning],
        stacklevel: int,
    ) -> None:
        """Emit one failure-path advisory without ever masking ``exc``.

        A warnings-as-error filter raises the advisory AT THE WARN SITE;
        letting that escape the failed-capture cleanup would REPLACE the
        error the capture door raises -- and with it the ``.partial_trace``
        recovery the advisory's own text advertises. The
        advisory degrades to a note on that error instead; it stays the one
        that propagates.
        """

        try:
            warnings.warn(message, category, stacklevel=stacklevel + 1)
        except Exception as advisory_error:
            with contextlib.suppress(Exception):
                annotate_guarded(
                    exc,
                    "TorchLens advisory suppressed (a warnings filter raised it "
                    f"as {type(advisory_error).__name__}): {message}",
                )

    cleanup_failed_forward_session = _cleanup_failed_forward_session_body

    def cleanup_forward_memory(self, session: object) -> None:
        """Release torch transient forward-memory caches.

        Parameters
        ----------
        session:
            Active trace session, consulted for the capture-touched-CUDA
            predicate before clearing the allocator cache.

        Returns
        -------
        None
            CUDA allocator cache is cleared when this capture touched CUDA.
        """

        # an unconditional empty_cache stalled EVERY capture teardown
        # on CUDA hosts (synchronizes the device and drops the warm allocator
        # arena) even for pure-CPU captures. Gate on the capture-touched-CUDA
        # predicate, matching the postprocess executor site.
        from ..._utils.tensor_utils import capture_touched_cuda

        if _is_cuda_available() and capture_touched_cuda(session):
            torch.cuda.empty_cache()


__all__ = ["TorchBackend"]
