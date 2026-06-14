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

    def setup_inputs_and_device(
        self,
        session: object,
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None,
    ) -> tuple[list[Any], dict[Any, Any], list[str], object]:
        """Normalize inputs, copy caller-owned values, and detect the model device."""
        ...

    def fetch_label_move_input_tensors(
        self,
        session: object,
        input_args: list[Any],
        input_arg_names: list[str],
        input_kwargs: dict[Any, Any],
        model_device: object,
    ) -> tuple[list[Any], list[str]]:
        """Extract input tensors, move them to the model device, and build labels."""
        ...

    def snapshot_rng(self, session: object) -> object:
        """Capture backend RNG state for the current session."""
        ...

    def snapshot_autocast(self, session: object) -> object:
        """Capture backend autocast state for the current session."""
        ...

    def log_source_tensor(
        self,
        session: object,
        tensor: object,
        source: str,
        extra_address: str | None = None,
    ) -> None:
        """Log a backend source tensor in the active capture session.

        Parameters
        ----------
        session:
            Active backend capture session.
        tensor:
            Backend tensor value to log as a source.
        source:
            Source role, such as ``"input"`` or ``"buffer"``.
        extra_address:
            Optional backend-specific input or buffer address.

        Returns
        -------
        None
            The active capture session is updated in place.
        """
        ...

    def push_existing_module_frame(
        self,
        session: object,
        module_stack: list[Any],
        frame: object,
    ) -> None:
        """Push an existing backend module-stack frame.

        Parameters
        ----------
        session:
            Active backend capture session.
        module_stack:
            Mutable module-stack list owned by the active recording state.
        frame:
            Existing backend frame to push.

        Returns
        -------
        None
            The module stack is updated in place.
        """
        ...

    def pop_module_frame(
        self,
        session: object,
        module_stack: list[Any],
        frame: object,
    ) -> None:
        """Pop and validate the current backend module-stack frame.

        Parameters
        ----------
        session:
            Active backend capture session.
        module_stack:
            Mutable module-stack list owned by the active recording state.
        frame:
            Expected top frame to pop.

        Returns
        -------
        None
            The module stack is updated in place.
        """
        ...

    def extract_and_mark_outputs(
        self,
        session: object,
        outputs: object,
    ) -> tuple[list[Any], list[str]]:
        """Extract final backend output tensors and mark output-parent events.

        Parameters
        ----------
        session:
            Active backend capture session.
        outputs:
            Raw model output object returned by the captured forward pass.

        Returns
        -------
        tuple[list[Any], list[str]]
            Output tensor leaves and their display addresses.
        """
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
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Run live intervention hooks against the candidate output tensor.

        Returns the possibly mutated tensor and the ``FireResult`` tuple.
        """
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

    def finalize_forward_session(
        self,
        session: object,
        trace_state: TraceBuildState | None = None,
    ) -> None:
        """Finalize backend state after forward logging and before output extraction.

        Parameters
        ----------
        session:
            Active backend capture session.
        trace_state:
            Optional transient trace build state for backends that materialize
            deferred payloads from event state at this seam.

        Returns
        -------
        None
            Backend-owned forward-session state is reconciled in place.

        Notes
        -----
        This hook must run after the forward pass exits backend logging and
        before final model outputs are extracted or marked. Torch registered
        buffer reconciliation depends on this timing because buffer writes must
        be validated before output-parent extraction observes final graph state.
        """
        ...

    def cleanup_halted_forward_session(self, session: object, prepared_model: object) -> None:
        """Clean up backend state after a halted forward capture.

        Parameters
        ----------
        session:
            Active backend capture session.
        prepared_model:
            Backend-prepared model object, or a backend-specific cleanup tuple.

        Returns
        -------
        None
            Session metadata is cleared in place.
        """
        ...

    def cleanup_failed_forward_session(
        self,
        session: object,
        prepared_model: object,
        exc: Exception,
    ) -> None:
        """Clean up backend state after a failed forward capture.

        Parameters
        ----------
        session:
            Active backend capture session.
        prepared_model:
            Backend-prepared model object, or a backend-specific cleanup tuple.
        exc:
            Exception raised by the forward capture.

        Returns
        -------
        None
            Session metadata is cleared in place and exception diagnostics may
            be attached to ``exc``.
        """
        ...


__all__ = ["CaptureBackend"]
