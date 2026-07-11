"""Error classes for the frozen sparse runnable ``.tlspec`` contract."""

from __future__ import annotations

from ._base import CompatibilityError, ConfigurationError, TorchLensError, ValidationError


class RunnableTLSPECError(TorchLensError):
    """Base class for sparse runnable ``.tlspec`` failures."""


class RunnablePreflightError(RunnableTLSPECError, ConfigurationError, ValueError):
    """Whole-graph producer preflight rejected a runnable claim."""


class RunCapabilityUnavailableError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """A Trace has no runnable provider or supported backend adapter."""


class ReattachError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """Atomic callable reattachment failed with a complete readiness report."""


class StateBindingError(RunnableTLSPECError, ConfigurationError, ValueError):
    """A state mapping failed strict name, role, tensor, or alias validation."""


class RunPreconditionError(RunnableTLSPECError, ConfigurationError, ValueError):
    """Runtime inputs or sparse call construction violated a frozen contract."""


class RuntimeSignatureDriftError(RunnableTLSPECError, CompatibilityError, RuntimeError):
    """A resolved native callable rejected the frozen recipe during execution."""


class PathDivergenceError(RunnableTLSPECError, ValidationError, RuntimeError):
    """A structure, shape, mutation, or control witness contradicted the path."""


class NumericAttestationError(RunnableTLSPECError, ValidationError, RuntimeError):
    """A recomputed selected activation differed from its archived bytes."""


class PoisonedRunError(RunnableTLSPECError, ValidationError, RuntimeError):
    """A downstream faithful-result consumer refused a poisoned run."""


__all__ = [
    "NumericAttestationError",
    "PathDivergenceError",
    "PoisonedRunError",
    "ReattachError",
    "RunCapabilityUnavailableError",
    "RunPreconditionError",
    "RunnablePreflightError",
    "RunnableTLSPECError",
    "RuntimeSignatureDriftError",
    "StateBindingError",
]
