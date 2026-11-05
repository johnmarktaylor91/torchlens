"""Error classes for the frozen sparse runnable ``.tlspec`` contract."""

from __future__ import annotations

from ._base import CompatibilityError, ConfigurationError, TorchLensError, ValidationError


class RunnableTLSPECError(TorchLensError):
    """Base class for sparse runnable ``.tlspec`` failures.

    Examples
    --------
    >>> issubclass(tl.errors.PoisonedRunError, RunnableTLSPECError)
    True
    >>> issubclass(RunnableTLSPECError, tl.errors.TorchLensError)
    True
    """


class RunnablePreflightError(RunnableTLSPECError, ConfigurationError, ValueError):
    """Whole-graph producer preflight rejected a runnable claim.

    Examples
    --------
    >>> issubclass(RunnablePreflightError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(RunnablePreflightError, ValueError)
    True
    """


class SparseCorePayloadError(RunnableTLSPECError, ValidationError, AssertionError):
    """A value-free sparse core carried a tensor or tensor-blob payload.

    Raised by the sparse-core tripwire
    (:func:`torchlens._io.runnable.assert_sparse_core_has_no_tensor_payload`)
    with ``fields["code"] == "sparse_core_payload_invalid"``. ``AssertionError``
    stays in the MRO deliberately: the tripwire's historical raise class was a
    bare ``AssertionError``, so existing ``except AssertionError`` callers keep
    working while new callers branch on the stable code.

    Examples
    --------
    >>> issubclass(SparseCorePayloadError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(SparseCorePayloadError, AssertionError)
    True
    """


class RunCapabilityUnavailableError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """A Trace has no runnable provider or supported backend adapter.

    Examples
    --------
    >>> issubclass(RunCapabilityUnavailableError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(RunCapabilityUnavailableError, RuntimeError)
    True
    """


class ReattachError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """Atomic callable reattachment failed with a complete readiness report.

    Examples
    --------
    >>> issubclass(ReattachError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(ReattachError, RuntimeError)
    True
    """


class StateBindingError(RunnableTLSPECError, ConfigurationError, ValueError):
    """A state mapping failed strict name, role, tensor, or alias validation.

    Examples
    --------
    >>> issubclass(StateBindingError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(StateBindingError, ValueError)
    True
    """


class RunPreconditionError(RunnableTLSPECError, ConfigurationError, ValueError):
    """Runtime inputs or sparse call construction violated a frozen contract.

    Examples
    --------
    >>> issubclass(RunPreconditionError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(RunPreconditionError, ValueError)
    True
    """


class BufferSinkRoutingError(RunnableTLSPECError, ConfigurationError, ValueError):
    """The live refresh projector refused mutable or unproven buffer-sink routing.

    Raised with ``fields["code"] == "buffer_sink_routing_unsupported"`` (a
    ``RunnableErrorCode`` member; provisional spelling, documented-unstable)
    by all four typed buffer-sink arms of the mode-aware refresh projector
    (D18): a train-mode buffer WRITER (``buffer_value_changed=True``),
    unproven write evidence (``None``, fail closed), a recorded mode claim
    contradicting the write evidence, and the refresh write tripwire (the
    refreshed rerun's own journal recorded a value-changing buffer write on
    the newly-allowed no-write path). ``ValueError`` stays in the MRO and the
    message keeps the pinned "computational graph changed" term, so every
    historical ``except ValueError`` caller and pinned match survive.

    Examples
    --------
    >>> issubclass(BufferSinkRoutingError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(BufferSinkRoutingError, ValueError)
    True
    """


class RuntimeSignatureDriftError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """A resolved native callable rejected the frozen recipe during execution.

    Examples
    --------
    >>> issubclass(RuntimeSignatureDriftError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(RuntimeSignatureDriftError, RuntimeError)
    True
    """


class PathDivergenceError(RunnableTLSPECError, ValidationError, RuntimeError, ValueError):
    """A structure, shape, mutation, or control witness contradicted the path.

    ``ValueError`` lineage (W051-HONESTY H3): the historical ``except ValueError``
    callers and the pinned ``match="computational graph changed"`` on a refresh
    graph change keep holding while the RAISE policy surfaces the TYPED error
    (structured ``fields``) instead of the projector's bare ``ValueError``.

    Examples
    --------
    >>> issubclass(PathDivergenceError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(PathDivergenceError, RuntimeError)
    True
    """


class ReplayValueMismatchError(RunnableTLSPECError, ValidationError, RuntimeError):
    """A recomputed selected activation differed from its archived bytes.

    Examples
    --------
    >>> issubclass(ReplayValueMismatchError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(ReplayValueMismatchError, RuntimeError)
    True
    """


class PoisonedRunError(RunnableTLSPECError, ValidationError, RuntimeError):
    """A downstream faithful-result consumer refused a poisoned run.

    Examples
    --------
    >>> issubclass(PoisonedRunError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(PoisonedRunError, RuntimeError)
    True
    """


class CollectiveBoundaryReplayError(RunnableTLSPECError, ValidationError, RuntimeError):
    """Runnable save or forward replay refused a collective-crossing trace.

    Examples
    --------
    >>> issubclass(CollectiveBoundaryReplayError, tl.errors.RunnableTLSPECError)
    True
    >>> issubclass(CollectiveBoundaryReplayError, RuntimeError)
    True
    """


__all__ = [
    "BufferSinkRoutingError",
    "CollectiveBoundaryReplayError",
    "ReplayValueMismatchError",
    "PathDivergenceError",
    "PoisonedRunError",
    "ReattachError",
    "RunCapabilityUnavailableError",
    "RunPreconditionError",
    "RunnablePreflightError",
    "RunnableTLSPECError",
    "RuntimeSignatureDriftError",
    "SparseCorePayloadError",
    "StateBindingError",
]
