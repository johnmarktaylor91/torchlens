"""Torch implementation of the capture backend Protocol."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from ..._io import BlobRef as PortableBlobRef
from ...capture.session import capture_session_for
from ...data_classes.internal_types import FuncExecutionContext
from ...fastlog.types import CaptureSpec, ModuleStackFrame, StorageIntent
from ...ir.container import ContainerSpec, OutputPathComponent
from ...ir.container_registry import ContainerLeafOccurrence, ModelSite, Phase, Role
from ...ir.events import OpEvent, OutputRef
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.op_record import amend_output_parent_promotion
from ...ir.predicate import RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...ir.workspaces import RawGraphWorkspace
from ...utils.arg_handling import (
    INPUT_WAS_PARAMETER_ATTR,
    normalize_input_args,
    safe_copy_input_tree,
)
from ...utils.introspection import (
    INPUT_SEARCH_DEPTH_LIMIT,
    get_vars_of_type_from_obj,
    nested_assign,
)
from ...utils.rng import log_current_rng_states, set_random_seed, set_rng_from_saved_states
from ...utils.tensor_utils import _is_cuda_available, safe_copy
from . import _tl
from ._held_refs_capture import rebind_held_torch_refs, restore_held_torch_refs
from .aliasing import detect_torch_alias_contract
from .buffer_writes import reconcile_buffer_writes, uninstall_buffer_write_tracker
from .completeness_witness import capture_completeness_witness, capture_scalar_escape_warning
from .escape_detection import capture_escape_guard
from .model_prep import (
    _cleanup_model_session,
    _ensure_model_prepared,
    _prepare_model_session,
)
from .module_stack import pop_frame, push_existing_frame
from .ops import (
    _get_autograd_saved_stats_for_tensor,
    _walk_output_tensors_with_paths,
    log_function_output_tensors,
    runnable_output_losslessness,
)
from .sources import log_source_tensor as _log_source_tensor
from .structure_only_belt import structure_only_escape_belt
from .wrappers import unwrap_torch, wrap_torch

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


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
        "verdict is ceilinged at capture_verified=False.",
        TorchLensCaptureGapWarning,
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


def _tensor_memory_bytes(tensor: torch.Tensor) -> int:
    """Return the byte size of a tensor payload.

    Parameters
    ----------
    tensor:
        Tensor whose payload memory should be measured.

    Returns
    -------
    int
        Number of bytes occupied by the tensor storage view.
    """

    return int(tensor.nelement() * tensor.element_size())


def _write_output_parent_blob(
    trace: Trace,
    label_raw: str,
    payload: torch.Tensor | None,
    kind: str,
) -> PortableBlobRef | None:
    """Write a promoted output-parent payload to the active bundle writer.

    Parameters
    ----------
    trace:
        Trace that may own an output bundle writer.
    label_raw:
        Raw label for the output-parent operation.
    payload:
        Tensor payload to persist, if any.
    kind:
        Bundle payload kind, such as ``"out"`` or ``"transformed_out"``.

    Returns
    -------
    PortableBlobRef | None
        Blob reference for the persisted payload, or ``None`` when no payload
        was written.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None or payload is None:
        return None
    blob_id = writer.next_blob_id()
    writer.submit_blob(blob_id, payload, kind=kind, label=label_raw)
    return PortableBlobRef(blob_id=blob_id, kind=kind)


def _promote_layers_to_save_output_parent(
    trace: Trace,
    event: OpEvent,
    tensor: torch.Tensor,
) -> tuple[OutputRef, CapturePolicy, bool, object]:
    """Resolve the ``output_parent_promotion`` amendment values for one output.

    Parameters
    ----------
    trace:
        Predicate-mode trace whose selective ``layers_to_save`` request must
        preserve the output-parent retention rule.
    event:
        Existing operation event for the tensor returned by the model.
    tensor:
        Live model-output tensor corresponding to ``event``.

    Returns
    -------
    tuple[OutputRef, CapturePolicy, bool, object]
        ``(output, policy, predicate_matched, capture_spec)`` for the caller's
        ``amend_output_parent_promotion`` — the event's current values when no
        payload retention is required, otherwise the saved-payload rebinds.
        The computed record context is deliberately NOT returned: the family
        schema preserves the documented legacy quirk of dropping it.
    """

    save_request = getattr(trace, "_rerun_save_request", None) or {}
    if (
        not save_request.get("retain_output_parents_for_layers_to_save", False)
        or getattr(trace, "_predicate_save_options", None) is None
        or event.output.has_saved_activation
    ):
        return event.output, event.policy, event.predicate_matched, event.capture_spec

    from ...capture.projections import _record_context_from_event
    from ...fastlog._storage_resolver import _resolve_storage

    ctx = dataclasses.replace(_record_context_from_event(event), is_output_parent=True)
    options = trace._predicate_save_options
    streaming = options.streaming
    intent = StorageIntent(
        in_ram=streaming is None or streaming.bundle_path is None or streaming.retain_in_memory,
        on_disk=streaming is not None and streaming.bundle_path is not None,
    )
    output_device = getattr(trace, "output_device", None)
    if output_device == "same":
        output_device = None
    spec = CaptureSpec(
        save_out=True,
        save_metadata=True,
        keep_grad=False,
        device=output_device,
        save_mode=cast(Any, getattr(trace, "save_mode", "copy")),
    )
    # Output-parent promotion retains real payloads and must be visible to
    # the save-budget accountant like every other RAM retention: this path
    # kept ram/transformed payloads with no admit/commit, silently
    # undercounting on every selective capture whose outputs were not
    # selected (grind-r6 b5 R34-N2). Admit BEFORE the copy allocates;
    # disk-only routes stay exempt like the other predicate disk saves.
    budget = getattr(trace, "_save_budget_accountant", None)
    reservation = None
    if budget is not None and intent.in_ram:
        target_device = torch.device(output_device) if output_device is not None else tensor.device
        reservation = budget.admit(
            str(event.label_raw),
            target_device,
            int(tensor.nelement() * tensor.element_size()),
            site="primary",
        )
    ram_payload, disk_payload, transformed_ram_payload, transformed_disk_payload = _resolve_storage(
        tensor,
        spec,
        intent,
        activation_transform=getattr(trace, "activation_transform", None),
        save_raw_activations=getattr(trace, "save_raw_activations", True),
        ctx=ctx,
        kind="activation",
    )
    if budget is not None and reservation is not None:
        budget.commit(reservation, (ram_payload, transformed_ram_payload))
    raw_blob_ref = _write_output_parent_blob(trace, event.label_raw, disk_payload, "out")
    transformed_blob_ref = _write_output_parent_blob(
        trace,
        event.label_raw,
        transformed_disk_payload,
        "transformed_out",
    )
    tensor_ref = dataclasses.replace(
        event.output.tensor,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=tensor.requires_grad,
        memory=_tensor_memory_bytes(tensor),
        payload=ram_payload,
        blob_ref=cast(Any, raw_blob_ref),
        backend_handle_id=str(id(tensor)),
    )
    transformed_ref = event.output.transformed_tensor
    if transformed_ram_payload is not None:
        transformed_ref = TensorRef(
            label_raw=event.label_raw,
            shape=tuple(transformed_ram_payload.shape),
            dtype=str(transformed_ram_payload.dtype),
            device=str(transformed_ram_payload.device),
            requires_grad=transformed_ram_payload.requires_grad,
            memory=_tensor_memory_bytes(transformed_ram_payload),
            payload=transformed_ram_payload,
            blob_ref=cast(Any, transformed_blob_ref),
            backend_handle_id=str(id(transformed_ram_payload)),
        )
    elif transformed_disk_payload is not None and transformed_ref is not None:
        transformed_ref = dataclasses.replace(
            transformed_ref,
            blob_ref=cast(Any, transformed_blob_ref),
        )
    output_ref = dataclasses.replace(
        event.output,
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=True,
    )
    policy = dataclasses.replace(event.policy, save_payload=True)
    return output_ref, policy, True, spec


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

        @contextlib.contextmanager
        def guarded_logging() -> Iterator[None]:
            """Enter detector state before enabling wrapper logging."""

            trace = cast("Trace", session)
            with capture_escape_guard(trace), capture_completeness_witness(trace):
                # L7a Layer-1 mode belt: escalated device-neutral escape
                # refusals for structure_only=True sessions; a no-op context
                # on the default path (zero-diff). The plain scalar-escape
                # warning belt hands off to it in-mode (its module notes why).
                with structure_only_escape_belt(trace), capture_scalar_escape_warning(trace):
                    # Plane-W completion authority (merge-ranks C2): a no-op
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
                            from ..._errors import CaptureContextError

                            raise CaptureContextError(
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
        input_args: object,
        input_kwargs: dict[Any, Any] | None,
    ) -> tuple[list[Any], dict[Any, Any], list[str], object]:
        """Normalize inputs, detect model device, copy args, and extract input arg names.

        Parameters
        ----------
        session:
            Active capture session. Unused for the torch implementation.
        model:
            Torch module being captured.
        input_args:
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
             (in ``fetch_label_move_input_tensors``) don't mutate the caller's data.
          4. Detect model device from first param or buffer (for auto-moving
             inputs). A model with neither pins no device: ``model_device`` is
             ``None`` and inputs are never moved.
        """
        torch_model = cast(torch.nn.Module, model)
        if isinstance(torch_model, torch.nn.DataParallel):
            torch_model = torch_model.module

        # Resolve ambiguity: is [tensor_a, tensor_b] two args or one list-arg?
        # normalize_input_args checks the model's forward() signature to decide.
        input_args = normalize_input_args(input_args, torch_model)

        if not input_kwargs:
            input_kwargs = {}

        # Detect device from first param or buffer. A model with NO parameters
        # and NO buffers pins no device: eager execution runs each op on its
        # operands' devices, so the inputs must stay exactly where the caller
        # put them (``None`` = no move). The historical ``"cpu"`` fallback
        # silently dragged CUDA inputs to the CPU and computed the whole
        # forward there -- first observed on real H200 hardware when the
        # CUPTI correlation matrix came back empty because the "CUDA" capture
        # had launched zero kernels.
        # Meta state never pins a device either: offload-hooked models
        # (accelerate device_map / cpu/disk offload, lane F37) hold meta
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
        input_args, input_kwargs, input_copy_gaps = safe_copy_input_tree(
            input_args,
            input_kwargs,
            require_distinct_tensor_sites=bool(getattr(session, "intervention_ready", False)),
        )
        _record_input_semantics_gaps(
            session,
            input_copy_gaps,
            reason="input_copy_semantics_unverifiable",
        )
        input_arg_names = _get_input_arg_names(torch_model, input_args)

        return input_args, input_kwargs, input_arg_names, model_device

    def fetch_label_move_input_tensors(
        self,
        session: object,
        input_args: list[Any],
        input_arg_names: list[str],
        input_kwargs: dict[Any, Any],
        model_device: object,
    ) -> tuple[list[Any], list[str]]:
        """Extract all tensors from input args/kwargs, move to model device, and build addresses.

        Parameters
        ----------
        session:
            Active capture session. Unused for the torch implementation.
        input_args:
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
        input_arg_tensors = []
        input_kwarg_tensors = []
        traversal_gaps: list[str] = []
        for arg_index, arg in enumerate(input_args):
            unresolved: list[str] = []
            input_arg_tensors.append(
                get_vars_of_type_from_obj(
                    arg,
                    torch.Tensor,
                    search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                    return_addresses=True,
                    depth_exceeded_paths=unresolved,
                )
            )
            arg_name = input_arg_names[arg_index]
            traversal_gaps.extend(
                f"input.{arg_name}{f'.{path}' if path else ''}" for path in unresolved
            )
        for key, kwarg in input_kwargs.items():
            unresolved = []
            input_kwarg_tensors.append(
                get_vars_of_type_from_obj(
                    kwarg,
                    torch.Tensor,
                    search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                    return_addresses=True,
                    depth_exceeded_paths=unresolved,
                )
            )
            traversal_gaps.extend(f"input.{key}{f'.{path}' if path else ''}" for path in unresolved)
        _record_input_semantics_gaps(
            session,
            traversal_gaps,
            reason="input_traversal_depth_exceeded",
        )
        # Move each tensor to model device.  Plain tuples must be temporarily
        # converted to lists for item assignment, then converted back to
        # preserve type.  This roundtrip only applies to *exact* ``tuple``
        # instances: they are the only ones addressed positionally
        # (``("ind", i)``) by ``get_vars_of_type_from_obj``, which treats
        # tuple *subclasses* (e.g. a NamedTuple-based GNN batch container) as
        # plain attribute-bearing objects instead, addressed via
        # ``("attr", name)``.  Applying the list roundtrip to a subclass would
        # silently discard its identity and break downstream named-field
        # access (``batch.edge_features``); ``_assign_nested_input_value``
        # already knows how to mutate those in place via ``attr`` addressing.
        moved_tensors_by_id: dict[int, torch.Tensor] = {}
        for arg_idx, arg in enumerate(input_args):
            was_tuple = type(arg) is tuple
            if was_tuple:
                input_args[arg_idx] = list(arg)
            for tensor_idx, (tensor, addr, addr_full) in enumerate(input_arg_tensors[arg_idx]):
                moved_tensor = moved_tensors_by_id.get(id(tensor))
                if moved_tensor is None:
                    # ``model_device is None`` = the model pins no device
                    # (no parameters or buffers); inputs stay on their own
                    # devices, matching eager semantics.
                    moved_tensor = tensor if model_device is None else tensor.to(model_device)
                    moved_tensors_by_id[id(tensor)] = moved_tensor
                if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                    setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
                input_arg_tensors[arg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
                if not addr_full:
                    input_args[arg_idx] = moved_tensor
                else:
                    input_args[arg_idx] = _assign_nested_input_value(
                        input_args[arg_idx], addr_full, moved_tensor
                    )
            if was_tuple and isinstance(input_args[arg_idx], list):
                input_args[arg_idx] = tuple(input_args[arg_idx])

        for kwarg_idx, (key, _val) in enumerate(input_kwargs.items()):
            for tensor_idx, (tensor, addr, addr_full) in enumerate(input_kwarg_tensors[kwarg_idx]):
                moved_tensor = moved_tensors_by_id.get(id(tensor))
                if moved_tensor is None:
                    moved_tensor = tensor if model_device is None else tensor.to(model_device)
                    moved_tensors_by_id[id(tensor)] = moved_tensor
                if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                    setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
                input_kwarg_tensors[kwarg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
                if not addr_full:
                    input_kwargs[key] = moved_tensor
                else:
                    input_kwargs[key] = _assign_nested_input_value(
                        input_kwargs[key], addr_full, moved_tensor
                    )

        # Build flat lists of (tensor, address) for both positional and keyword args.
        # Address format: "input.<argname>" or "input.<argname>.<nested_path>"
        input_tensors = []
        input_tensor_addresses = []
        seen_tensor_ids: set[int] = set()
        for arg_idx, arg_tensors in enumerate(input_arg_tensors):
            for tensor, addr, addr_full in arg_tensors:
                if id(tensor) in seen_tensor_ids:
                    continue
                seen_tensor_ids.add(id(tensor))
                input_tensors.append(tensor)
                tensor_addr = f"input.{input_arg_names[arg_idx]}"
                if addr != "":
                    tensor_addr += f".{addr}"
                input_tensor_addresses.append(tensor_addr)

        for arg_idx, kwarg_tensors in enumerate(input_kwarg_tensors):
            for tensor, addr, addr_full in kwarg_tensors:
                if id(tensor) in seen_tensor_ids:
                    continue
                seen_tensor_ids.add(id(tensor))
                input_tensors.append(tensor)
                tensor_addr = f"input.{list(input_kwargs.keys())[arg_idx]}"
                if addr != "":
                    tensor_addr += f".{addr}"
                input_tensor_addresses.append(tensor_addr)

        return input_tensors, input_tensor_addresses

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

        from .ops import set_capture_producer_policy

        set_capture_producer_policy(cast("Trace", session), cast(Any, capture_mode))

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

    def inference_context(self, session: object) -> AbstractContextManager[None]:
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

        return (
            torch.no_grad()
            if getattr(session, "inference_only", False)
            else contextlib.nullcontext()
        )

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
        output_entries = list(_walk_output_tensors_with_paths(outputs))
        # r35 I1 (subsumes r33 R32-B1): stamp the POSITIVE model-output losslessness
        # proof -- exact root kind, recursively supported children, encodable literal
        # leaves, and a tensor-leaf/typed-path bijection (duplicate paths and any BFS
        # fallback break the proof). The runnable producer refuses any save whose
        # output is not PROVED lossless (refuse-unless-proved), closing every lossy
        # cardinality/depth: bare one-tensor sets, nested sets, opaque tensor
        # holders, set subclasses, and multi-tensor collapses alike. Ordinary
        # analysis capture is unaffected by the stamp.
        self_trace._runnable.output_losslessness = runnable_output_losslessness(
            outputs, output_entries
        )
        # The per-op container_spec is only user-facing metadata when explicitly
        # opted into via capture_container_structure (or implied by
        # intervention_ready); with the default OFF it must stay None on output
        # layers. The container *path*, however, is always preserved so
        # forward-replay validation can slice multi-output containers back to the
        # right leaf, and the FINAL-output ContainerSpec is always registered as
        # the model-output snapshot (below): the live ``run()`` provider rebuilds
        # the exact tuple/dict/ModelOutput the model returned from that one
        # snapshot, so a default capture settles VERIFIED instead of the
        # remedy-only ``container_contract_unrecorded`` (W051-HONESTY H2).
        persist_container_spec = getattr(self_trace, "intervention_ready", False) or getattr(
            self_trace, "_capture_container_structure", False
        )
        if output_entries:
            output_tensors_w_addresses_all = [
                (tensor, _container_path_to_address(path), None)
                for tensor, path, _container_spec in output_entries
            ]
            output_specs_by_raw_label = {}
            for tensor, path, container_spec in output_entries:
                _label_raw = _tl.get_tensor_label(tensor)
                if _label_raw is not None:
                    output_specs_by_raw_label[_label_raw] = (
                        path,
                        container_spec if persist_container_spec else None,
                    )
            setattr(self_trace, "_output_container_specs_by_raw_label", output_specs_by_raw_label)
        else:
            output_tensors_w_addresses_all = []
        if output_entries:
            # Every capture (one spec per capture): the model-output snapshot is
            # the contract the live provider reconstructs from.
            _register_model_output_container_snapshot(self_trace, outputs, output_entries)
        # (container_path is stored above for validation replay even when the spec is None)
        if not output_entries:
            output_tensors_w_addresses_all = get_vars_of_type_from_obj(
                outputs,
                torch.Tensor,
                search_depth=5,
                return_addresses=True,
                allow_repeats=True,
            )
        output_tensors, output_tensor_addresses = _dedupe_output_addresses(
            output_tensors_w_addresses_all
        )

        attributable_output_tensors: list[torch.Tensor] = []
        attributable_output_tensor_addresses: list[str] = []
        for t, output_address in zip(output_tensors, output_tensor_addresses):
            _label_raw = _tl.get_tensor_label(t)
            if _label_raw is None:
                if _is_direct_registered_buffer_output(self_trace, t):
                    # Late-logged in postprocess so untouched buffer outputs get
                    # real source nodes without leaking labels onto model state.
                    attributable_output_tensors.append(t)
                    attributable_output_tensor_addresses.append(output_address)
                    continue
                _label_raw = _model_input_output_label(self_trace, t)
            if _label_raw is None:
                if getattr(self_trace, "_raw_transform_escape_detected", False):
                    continue
                from ..._errors import OutputAttributionError

                try:
                    shape_text = str(tuple(t.shape))
                except RuntimeError:
                    # Nested tensors raise from ``.shape``; the refusal must
                    # stay typed instead of crashing on its own message (R65).
                    shape_text = "<unavailable: nested>" if t.is_nested else "<unavailable>"
                if getattr(t, "is_nested", False):
                    # An unlabeled NESTED output is an unsupported tensor
                    # variant constructed inside forward (protocol-invisible
                    # constructors like torch.nested.nested_tensor are never
                    # logged), NOT a pre-bound-function escape -- the escape
                    # remedy can never fix it (R65: typed-misdiagnosis
                    # successor of the round-4 raw crash).
                    raise OutputAttributionError(
                        "TorchLens could not attribute a model output tensor to any "
                        f"traced op (output address {output_address!r}, "
                        f"shape={shape_text}, dtype={t.dtype}): the output is a NESTED "
                        "tensor constructed inside forward(), an unsupported tensor "
                        "variant TorchLens cannot log. Remedy: build the nested tensor "
                        "outside the traced region, or pad to a dense tensor before "
                        "the ops you want captured.",
                        code="output_unsupported_tensor_variant",
                        remedy=(
                            "build the nested tensor outside the traced region, or "
                            "pad to a dense tensor before the ops you want captured"
                        ),
                        output_address=output_address,
                    )
                raise OutputAttributionError(
                    "TorchLens could not attribute a model output tensor to any traced op "
                    f"(output address {output_address!r}, "
                    f"shape={shape_text}, dtype={t.dtype}). This may indicate an opaque "
                    "execution boundary or a pre-bound torch function that escaped wrapping. "
                    "Use ordinary torch module attributes during forward, or bind/import torch "
                    "functions after TorchLens has wrapped torch.",
                    code="output_attribution_failed",
                    remedy=(
                        "use ordinary torch module attributes during forward, or "
                        "bind/import torch functions after TorchLens has wrapped torch"
                    ),
                    output_address=output_address,
                )
            attributable_output_tensors.append(t)
            attributable_output_tensor_addresses.append(output_address)
            if self_trace.capture_mode in {"exhaustive", "predicate"}:
                self_trace.output_layers.append(_label_raw)
                event = self_trace.capture_events.op_event_by_label_raw.get(_label_raw)
                if event is not None:
                    promoted_output, promoted_policy, promoted_matched, promoted_spec = (
                        _promote_layers_to_save_output_parent(
                            self_trace,
                            event,
                            t,
                        )
                    )
                    self_trace.capture_events.append_amendment(
                        amend_output_parent_promotion(
                            event.seq,
                            _label_raw,
                            is_output_parent=True,
                            output=promoted_output,
                            policy=promoted_policy,
                            predicate_matched=promoted_matched,
                            capture_spec=promoted_spec,
                        )
                    )

        return attributable_output_tensors, attributable_output_tensor_addresses

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

    def finalize_forward_session(
        self,
        session: object,
        trace_state: RawGraphWorkspace,
    ) -> None:
        """Run torch post-forward reconciliation before output extraction."""
        reconcile_buffer_writes(cast("Trace", session), trace_state)

    def cleanup_halted_forward_session(self, session: object, prepared_model: object) -> None:
        """Clean up torch metadata after a halted forward capture."""
        self.cleanup_model_session(session, prepared_model)
        # F5: the raw-graph workspace may already be gone when the halt fired
        # around a postprocess-tail boundary; a missing workspace must degrade
        # gracefully instead of masking the in-flight signal with
        # AttributeError.
        raw_graph_ws = getattr(session, "__dict__", {}).get("_raw_graph_ws")
        if raw_graph_ws is None:
            return
        raw_layer_dict = raw_graph_ws.raw_layer_dict
        for label in list(raw_layer_dict.keys()):
            entry = raw_layer_dict.get(label)
            if entry is not None and hasattr(entry, "out") and entry.out is not None:
                _tl.clear_meta(entry.out)

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
        user's real forward exception -- and with it the ``exc.partial_log``
        recovery the advisory's own text advertises (b3-opus-R07-1). The
        advisory degrades to an exception note instead; the user's exception
        stays the one that propagates.
        """

        try:
            warnings.warn(message, category, stacklevel=stacklevel + 1)
        except Exception as advisory_error:
            with contextlib.suppress(Exception):
                exc.add_note(
                    "TorchLens advisory suppressed (a warnings filter raised it "
                    f"as {type(advisory_error).__name__}): {message}"
                )

    def cleanup_failed_forward_session(
        self,
        session: object,
        prepared_model: object,
        exc: Exception,
    ) -> None:
        """Attach partial trace diagnostics and clean up failed torch capture state."""
        # active_logging's __exit__ already turned off the toggle.
        # Clean up model session state and strip TorchLens metadata from any
        # partially-constructed tensor entries to avoid stale references (#110).
        from ...partial import PartialTrace, _register_failed_capture

        # Stamp the failed forward's ACTUAL buffer-write record (value-changing
        # journal events) on the exception while the journal is still live —
        # ``cleanup_model_session`` below clears ``capture_events``, and the
        # rescue driver needs this record to refuse a double-forward re-run
        # after an output-attribution failure (R16-2 for the failed-primary
        # trigger). Only an exhaustive session arms the tracker, so only there
        # is an empty journal proof of "no writes".
        if getattr(session, "capture_mode", None) == "exhaustive":
            events = getattr(getattr(session, "capture_events", None), "buffer_write_events", None)
            if events is not None:
                with contextlib.suppress(Exception):
                    exc._torchlens_actual_buffer_writes = tuple(  # type: ignore[attr-defined]
                        str(getattr(event, "address", None) or "?")
                        for event in events
                        if getattr(event, "value_changed", None) is not False
                    )

        if getattr(session, "capture_mode", None) == "predicate":
            from ...ir import CaptureEvents

            events = getattr(session, "capture_events", None)
            if events is not None:
                # Snapshot the failing pass through the sanctioned merge path:
                # concat clones and re-stamps every mergeable lane under the
                # declared merge law, keeping the snapshot's counter coherent
                # with its seq values (a direct lane splice preserved source
                # seqs that collided with the re-stamped op lane).
                failed_fastlog_events = CaptureEvents()
                failed_fastlog_events.concat(events)
                setattr(session, "_failed_fastlog_capture_events", failed_fastlog_events)
        # Partial diagnostics are BEST EFFORT; the model-session teardown is not.
        # The arms below catch ``Exception``, so an interruption (KeyboardInterrupt,
        # SystemExit) raised while building or attaching the partial trace used to
        # escape straight past ``cleanup_model_session`` -- leaving the user's model
        # with TorchLens-forced ``requires_grad``, ``tl_*`` metadata and an
        # installed buffer tracker. The ``finally`` guarantees teardown without
        # swallowing the interruption.
        try:
            try:
                partial_log = PartialTrace.from_trace(cast("Trace", session), exc)
            except Exception as construction_error:
                self._warn_without_masking(
                    exc,
                    "TorchLens could not construct partial-trace recovery after the "
                    f"forward failed: {type(construction_error).__name__}: "
                    f"{construction_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                with contextlib.suppress(Exception):
                    exc.add_note(
                        "TorchLens partial-trace construction also failed: "
                        f"{type(construction_error).__name__}: {construction_error}"
                    )
            else:
                try:
                    exc.partial_log = partial_log  # type: ignore[attr-defined]
                    # B8-45: the SUCCESS path must tell the user the recovery
                    # exists too -- only the two attachment-FAILURE arms did.
                    with contextlib.suppress(Exception):
                        exc.add_note(
                            "TorchLens attached partial capture diagnostics: inspect "
                            "exc.partial_log, or recover it with "
                            "torchlens.partial.from_failed_capture(exception)."
                        )
                except Exception as attachment_error:
                    _register_failed_capture(exc, partial_log)
                    self._warn_without_masking(
                        exc,
                        "The forward exception rejected TorchLens partial_log attachment; "
                        "recovery remains available through "
                        "torchlens.partial.from_failed_capture(exception). "
                        f"Attachment error: {type(attachment_error).__name__}: "
                        f"{attachment_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    with contextlib.suppress(Exception):
                        exc.add_note(
                            "TorchLens retained partial capture recovery in its bounded "
                            "exception-identity registry; call "
                            "torchlens.partial.from_failed_capture(exception)."
                        )
        finally:
            self.cleanup_model_session(session, prepared_model)
        # F5: a postprocess-tail failure (step 18-20 or the relation freeze)
        # arrives here AFTER the transient-state seam popped the raw-graph
        # workspace. The unguarded read used to double-fault with
        # AttributeError, MASKING the original exception (it survived only as
        # __context__) and losing partial diagnostics; degrade gracefully so
        # the original failure propagates.
        raw_graph_ws = getattr(session, "__dict__", {}).get("_raw_graph_ws")
        if raw_graph_ws is not None:
            raw_layer_dict = raw_graph_ws.raw_layer_dict
            for label in list(raw_layer_dict.keys()):
                entry = raw_layer_dict.get(label)
                if entry is not None and hasattr(entry, "out") and entry.out is not None:
                    _tl.clear_meta(entry.out)
        # B8-35/B8-44: the historical unconditional stdout banner ("Feature
        # extraction failed; returning model and environment to normal") was
        # factually false on rescue-recovered captures and supported
        # return_partial flows, corrupted machine-readable stdout, and was
        # unfilterable. One accurate ROUTED warning replaces it, naming what
        # actually happened and where the diagnostics live; stacklevel targets
        # the user's tl.trace call through the driver frames. The dedicated
        # category (a RuntimeWarning subclass, so user filters keep matching)
        # lets the rescue driver defer it while a rescue re-run can still
        # swallow this failure — a successful rescue drops the advisory
        # instead of pointing users at an exception they never receive.
        from .rescue import CaptureAttemptFailedWarning

        # B8-44 follow-up (quickstart memo 4.6): the restoration sentence is
        # scoped to what teardown actually restores -- TorchLens
        # instrumentation and the torch environment. Model state mutated by
        # the partially-executed forward (a materialized lazy module, updated
        # BatchNorm running stats) is NOT rolled back, so the historical
        # "the model ... restored" claim was false exactly when a forward
        # materialized a module and then failed.
        self._warn_without_masking(
            exc,
            "TorchLens capture attempt failed "
            f"({type(exc).__name__}); TorchLens instrumentation is removed "
            "from the model and the torch environment restored (state the "
            "partial forward already mutated -- e.g. norm running statistics, "
            "materialized lazy modules, updated buffers -- is not rolled "
            "back). Partial diagnostics ride the exception "
            "(exc.partial_log / torchlens.partial.from_failed_capture).",
            CaptureAttemptFailedWarning,
            stacklevel=4,
        )

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

        # R16-4b: an unconditional empty_cache() stalled EVERY capture teardown
        # on CUDA hosts (synchronizes the device and drops the warm allocator
        # arena) even for pure-CPU captures. Gate on the capture-touched-CUDA
        # predicate, matching the postprocess executor site.
        from ...utils.tensor_utils import capture_touched_cuda

        if _is_cuda_available() and capture_touched_cuda(session):
            torch.cuda.empty_cache()


def _dedupe_output_addresses(
    entries: list[tuple[torch.Tensor, str, Any]],
) -> tuple[list[torch.Tensor], list[str]]:
    """Drop entries repeating an already-seen structural output address.

    Parameters
    ----------
    entries:
        ``(tensor, address, container_spec)`` output entries in walk order.

    Returns
    -------
    tuple[list[torch.Tensor], list[str]]
        First-occurrence output tensors and their display addresses.
    """

    addresses_seen: set[str] = set()
    deduped = []
    for entry in entries:
        if entry[1] in addresses_seen:
            continue
        deduped.append(entry)
        addresses_seen.add(entry[1])
    return [t for t, _, _ in deduped], [addr for _, addr, _ in deduped]


def _register_model_output_container_snapshot(
    trace: Trace,
    output: object,
    output_entries: list[
        tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
    ],
) -> None:
    """Register the final model-output container snapshot when present.

    Parameters
    ----------
    trace:
        Active trace.
    output:
        Raw model output object.
    output_entries:
        Path-aware tensor entries from the existing output walker.
    """

    spec = next((container_spec for _, _, container_spec in output_entries if container_spec), None)
    if spec is None:
        return
    # An opaque model-output container (custom Mapping, unsafe defaultdict, unknown
    # dict subclass, or an unrepresentable non-tensor leaf) is recorded but marked
    # NON-reconstructable so producer preflight refuses to advertise it runnable and
    # a live run reports UNVERIFIABLE -- never a silent bare-tensor/plain-dict.
    reconstructable = spec.kind != "opaque"
    occurrences: list[ContainerLeafOccurrence] = []
    for occ_index, (tensor, path, _container_spec) in enumerate(output_entries):
        producer_label = _tl.get_tensor_label(tensor)
        occurrences.append(
            ContainerLeafOccurrence(
                path=path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=occ_index,
            )
        )
    registry = trace._wrapper_runtime_ws.container_registry
    registry.register_snapshot(
        output,
        site=ModelSite(model_ref="self:1", position="return"),
        role=Role.MODEL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
        spec=spec,
        leaf_occurrences=tuple(occurrences),
        reconstructable=reconstructable,
    )
    registry.register_snapshot(
        output,
        site=ModelSite(model_ref="self:1", position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
        spec=spec,
        leaf_occurrences=tuple(occurrences),
        reconstructable=reconstructable,
    )


def _is_direct_registered_buffer_output(trace: Trace, tensor: torch.Tensor) -> bool:
    """Return whether an unlabeled output is a registered source-model buffer.

    Parameters
    ----------
    trace:
        Active trace that may hold a weak reference to the source model.
    tensor:
        Unlabeled output tensor returned by ``forward``.

    Returns
    -------
    bool
        True when ``tensor`` is exactly one of the source model's registered
        buffers. Such tensors are late-logged during postprocess; other
        unlabeled outputs fail loud at the output boundary.
    """

    model_ref = getattr(trace, "_source_model_ref", None)
    model = model_ref() if model_ref is not None else None
    if model is None:
        return False
    return any(tensor is buffer for _address, buffer in model.named_buffers())


def _model_input_output_label(trace: Trace, tensor: torch.Tensor) -> str | None:
    """Return the input-source label when an unlabeled output is a model input.

    Parameters
    ----------
    trace:
        Active trace carrying the tensors that were explicitly marked as model
        inputs for this capture.
    tensor:
        Unlabeled model output tensor.

    Returns
    -------
    str | None
        Raw input label for ``tensor`` when structural tensor identity/storage
        proves it is one of the marked model inputs; otherwise ``None``.
    """

    input_tensors = getattr(trace, "_output_attribution_input_tensors", ())
    input_labels = tuple(getattr(trace, "input_layers", ()))
    for index, input_tensor in enumerate(input_tensors):
        if not isinstance(input_tensor, torch.Tensor):
            continue
        if not _same_tensor_storage_identity(tensor, input_tensor):
            continue
        live_label = _tl.get_tensor_label(input_tensor)
        if live_label is not None:
            return live_label
        if index < len(input_labels):
            return str(input_labels[index])
    return None


def _same_tensor_storage_identity(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors are the same object or identical storage view.

    Parameters
    ----------
    left:
        Candidate output tensor.
    right:
        Marked input tensor.

    Returns
    -------
    bool
        True when the tensors are the same Python object, or when their storage
        pointer, offset, shape, stride, dtype, and device all match.
    """

    if left is right:
        return True
    if left.dtype != right.dtype or left.device != right.device:
        return False
    try:
        if tuple(left.shape) != tuple(right.shape) or tuple(left.stride()) != tuple(right.stride()):
            return False
        if left.storage_offset() != right.storage_offset():
            return False
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except RuntimeError:
        # Nested tensors raise from shape/stride/storage reads; a tensor whose
        # layout metadata is unreadable cannot be structurally proven to be a
        # marked input, so identity attribution conservatively says no (R65).
        return False


def _assign_nested_input_value(
    obj: Any,
    addr: list[tuple[Any, Any]],
    value: Any,
) -> Any:
    """Assign ``value`` inside nested input containers, rebuilding tuples.

    Parameters
    ----------
    obj:
        Root object to update.
    addr:
        Address path from ``get_vars_of_type_from_obj``.
    value:
        Replacement tensor.

    Returns
    -------
    Any
        Updated root object.
    """

    if not addr:
        return value
    entry_type, entry_val = addr[0]
    rest = addr[1:]
    if entry_type == "ind":
        if isinstance(obj, tuple):
            items = list(obj)
            items[entry_val] = _assign_nested_input_value(items[entry_val], rest, value)
            obj_type = type(obj)
            if hasattr(obj_type, "_fields"):
                return obj_type(*items)
            return obj_type(items)
        if isinstance(obj, list):
            obj[entry_val] = _assign_nested_input_value(obj[entry_val], rest, value)
            return obj
        if isinstance(obj, dict):
            obj[entry_val] = _assign_nested_input_value(obj[entry_val], rest, value)
            return obj
    if entry_type == "attr":
        child = getattr(obj, entry_val)
        new_child = _assign_nested_input_value(child, rest, value)
        try:
            setattr(obj, entry_val, new_child)
        except AttributeError:
            # Immutable attribute — e.g. a real (non-property) NamedTuple
            # field on a NamedTuple subclass such as a GNN batch container.
            # ``_replace`` returns a new instance with that field swapped in.
            obj_fields = getattr(type(obj), "_fields", None)
            if hasattr(obj, "_replace") and obj_fields is not None and entry_val in obj_fields:
                return obj._replace(**{entry_val: new_child})
            # Otherwise this is a read-only *derived* property (e.g. a
            # convenience accessor that returns a value already stored in a
            # mutable nested container). The tensor's real backing storage is
            # reached and moved to device through its own container address;
            # there is nothing else to update at this alias.
        return obj
    nested_assign(obj, addr, value)
    return obj


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
