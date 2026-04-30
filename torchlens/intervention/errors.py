"""Error ownership for the planned TorchLens intervention API."""

from typing import NoReturn


def _not_implemented(name: str, phase: str) -> NoReturn:
    """Raise a deterministic placeholder error for future intervention work.

    Parameters
    ----------
    name:
        Public API name that is not implemented yet.
    phase:
        Future implementation phase expected to provide the behavior.

    Raises
    ------
    NotImplementedError
        Always raised with the provided API name and phase.
    """

    raise NotImplementedError(f"torchlens.{name} is reserved for intervention API {phase}.")


class TorchLensInterventionError(RuntimeError):
    """Base class for future TorchLens intervention errors."""


class ReplayPreconditionError(TorchLensInterventionError):
    """Raised when replay cannot satisfy its future execution preconditions."""


class EngineDispatchError(TorchLensInterventionError):
    """Raised when future intervention engine dispatch arguments are invalid."""


class SiteResolutionError(TorchLensInterventionError):
    """Raised when future selector resolution cannot identify requested sites."""


__all__ = [
    "EngineDispatchError",
    "ReplayPreconditionError",
    "SiteResolutionError",
    "TorchLensInterventionError",
]
