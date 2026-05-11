"""Backend Protocol for unified TorchLens capture."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol

from ..ir.events import OpEvent, TraceBuildState
from ..ir.intervention import FireResult, FunctionEventInput
from ..ir.predicate import RecordContext
from ..ir.refs import ReservedLabel, TensorRef
from ..ir.semantics import BackendSemantics, CapturePolicy


class CaptureBackend(Protocol):
    """Protocol implemented by backend-specific capture adapters."""

    name: str
    supports_backward_capture: bool

    def wrap(self, value: object) -> object:
        """Return a wrapped backend callable or object."""
        ...

    def unwrap(self, value: object) -> object:
        """Return the original backend callable or object."""
        ...

    def is_wrapped(self, value: object) -> bool:
        """Return whether a backend callable or object is wrapped."""
        ...

    def start_session(self, options: object) -> object:
        """Start a backend capture session."""
        ...

    def prepare_model(self, session: object, model: object) -> object:
        """Prepare a model for a backend capture session."""
        ...

    def prepare_model_once(self, model: object) -> object:
        """Apply one-time backend model preparation."""
        ...

    def prepare_model_session(self, session: object, model: object) -> object:
        """Apply per-session backend model preparation."""
        ...

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session backend model preparation."""
        ...

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that enables backend logging."""
        ...

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that pauses backend logging."""
        ...

    def snapshot_rng(self, session: object) -> object:
        """Capture backend RNG state for the current session."""
        ...

    def snapshot_autocast(self, session: object) -> object:
        """Capture backend autocast state for the current session."""
        ...

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one output."""
        ...

    def detect_in_place_isolation_required(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> bool:
        """Return whether the output requires in-place isolation."""
        ...

    def detect_backend_semantics(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> BackendSemantics:
        """Detect backend-specific scalar semantics for one output."""
        ...

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for a tensor-like value without materializing payload."""
        ...

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the backend tensor label used for parent tracking."""
        ...

    def is_tensor(self, value: object) -> bool:
        """Return whether a value is a backend tensor."""
        ...

    def is_parameter(self, value: object) -> bool:
        """Return whether a value is a backend parameter."""
        ...

    def mark_same_object_candidates(
        self,
        session: object,
        func_event_input: FunctionEventInput,
    ) -> object:
        """Mark input objects that may be returned by identity."""
        ...

    def isolate_same_object_returns(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        raw_output: object,
        premarked_inputs: object,
    ) -> object:
        """Clone raw outputs that alias premarked inputs."""
        ...

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: object,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Apply matching live intervention hooks for one output site."""
        ...

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Materialize a backend-safe payload copy under the given policy."""
        ...

    def copy_replacement_metadata(self, session: object, src: object, dst: object) -> None:
        """Copy backend metadata from an intervention source to replacement."""
        ...

    def emit_function_outputs(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        isolated_output: object,
        output_sites: tuple[object, ...],
        reserved_block: tuple[ReservedLabel, ...],
    ) -> tuple[OpEvent, ...]:
        """Emit per-output operation events for an already-invoked function."""
        ...

    def finalize_forward_session(self, session: object, trace_state: TraceBuildState) -> None:
        """Finalize backend session state after forward postprocess materialization."""
        ...


__all__ = ["CaptureBackend"]
