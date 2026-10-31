"""Narrow ownership seam between ``Trace`` and sparse runnable internals."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from torchlens._runnable import ReadinessReport, RunnableDiagnostic

    from ._runnable import (
        ArchivedActivation,
        DivergencePolicy,
        PathFaithfulness,
        RunResult,
        SparseRunDescriptor,
    )


RUNNABLE_TRACE_PUBLIC_MEMBERS = frozenset(
    {
        "readiness",
        "archived_activations",
        "load_state_dict",
        "run",
    }
)
"""Complete public ``Trace`` surface owned by the runnable subsystem."""


@dataclass(slots=True)
class RunnableTraceState:
    """All private sparse-runnable state owned by one ``Trace``.

    The container is session-only and is always dropped by portable state
    handling. Its fields preserve the pre-seam defaults exactly; runtime code
    must use this object instead of adding runnable-prefixed attributes to
    ``Trace``.
    """

    descriptor: SparseRunDescriptor | None = None
    readiness: ReadinessReport | None = None
    staged_user_state: Mapping[str, torch.Tensor] | None = None
    embedded_state: Mapping[str, torch.Tensor] | None = None
    capture_state: Mapping[str, torch.Tensor] | None = None
    embedded_nonpersistent_buffers: Mapping[str, torch.Tensor] | None = None
    archived_activations: Mapping[str, ArchivedActivation] | None = None
    path_faithfulness: PathFaithfulness | None = None
    first_mismatch: RunnableDiagnostic | None = None
    poisoned: bool = False
    callables_by_call_id: dict[str, Any] | None = None
    host_rng_consumed: bool | None = None
    capture_ambient: Mapping[str, Any] | None = None
    state_alias_topology: Any = None
    capture_state_signatures: Mapping[str, Any] | None = None
    persistent_buffer_universe: Mapping[str, Any] | None = None
    host_rng_unreplayable: bool | None = None
    host_rng_channels: tuple[Any, ...] | None = None
    host_rng_replayable_reads: tuple[Any, ...] | None = None
    rng_monitor_uncertain: bool | None = None
    rng_monitor_uncertain_detail: tuple[str, ...] | None = None
    output_losslessness: Mapping[str, Any] | None = None
    input_nontensor_leaves: tuple[Any, ...] | None = None
    input_structure: tuple[Any, ...] | None = None
    input_tensor_sites: Mapping[int, Any] | None = None
    input_metadata_reads: dict[Any, Any] = field(default_factory=dict)
    input_label_layouts: Mapping[str, Any] | None = None
    module_training_modes: Mapping[str, bool] | None = None
    #: Session-scoped STATE-COMPROMISED latch (L4 5.4, never persisted): set when
    #: a declared-state restore failed AFTER execution, leaving the LIVE MODEL's
    #: state unknown. Deliberately NOT the poison bit -- the trace's recorded
    #: path facts are not a lie; only live/fast re-execution is refused.
    state_compromised: Mapping[str, Any] | None = None

    def pickle_safe_copy(self) -> RunnableTraceState:
        """Return a shallow copy with mapping-proxy bindings made picklable.

        Returns
        -------
        RunnableTraceState
            Independent state container whose immutable tensor bindings use
            ordinary dictionaries without cloning tensor values.
        """

        return replace(
            self,
            staged_user_state=_plain_mapping(self.staged_user_state),
            embedded_state=_plain_mapping(self.embedded_state),
            capture_state=_plain_mapping(self.capture_state),
            embedded_nonpersistent_buffers=_plain_mapping(self.embedded_nonpersistent_buffers),
        )


def _plain_mapping(value: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    """Return a plain-dict view of one optional tensor binding.

    Parameters
    ----------
    value:
        Optional immutable or mutable state mapping.

    Returns
    -------
    dict[str, torch.Tensor] | None
        Plain mapping retaining tensor identity, or ``None``.
    """

    return None if value is None else dict(value)


def runnable_trace_state(trace: Any) -> RunnableTraceState:
    """Return the single runnable state container owned by ``trace``.

    Parameters
    ----------
    trace:
        Trace-like object with dict-backed internal state.

    Returns
    -------
    RunnableTraceState
        Existing state, or a newly installed default container.
    """

    state = trace.__dict__.get("_runnable")
    if not isinstance(state, RunnableTraceState):
        state = RunnableTraceState()
        trace.__dict__["_runnable"] = state
    return state


def normalize_runnable_trace_state(mapping: MutableMapping[str, Any]) -> RunnableTraceState:
    """Normalize pickle state onto the one runnable field.

    Parameters
    ----------
    mapping:
        Mutable Trace state being restored.

    Returns
    -------
    RunnableTraceState
        Normalized container installed under ``mapping["_runnable"]``.
    """

    current = mapping.get("_runnable")
    state = current if isinstance(current, RunnableTraceState) else RunnableTraceState()
    mapping["_runnable"] = state
    return state


class RunnableCoordinator(Protocol):
    """Four-verb boundary implemented by sparse runnable orchestration."""

    def produce(self, trace: Any, options: Mapping[str, Any]) -> SparseRunDescriptor:
        """Produce one semantic descriptor from a completed live Trace."""

        ...

    def decode(self, raw: Mapping[str, Any]) -> SparseRunDescriptor:
        """Decode transport data into one typed sparse descriptor."""

        ...

    def prepare(
        self,
        trace: Any,
        descriptor: SparseRunDescriptor,
        payloads: Mapping[str, Any],
    ) -> ReadinessReport:
        """Resolve and bind a descriptor without executing it."""

        ...

    def execute(
        self,
        trace: Any,
        inputs: Any,
        *,
        seed: int | None,
        policy: DivergencePolicy,
    ) -> RunResult:
        """Execute and settle one transactional runnable provider call."""

        ...


__all__ = [
    "RUNNABLE_TRACE_PUBLIC_MEMBERS",
    "RunnableCoordinator",
    "RunnableTraceState",
    "normalize_runnable_trace_state",
    "runnable_trace_state",
]
