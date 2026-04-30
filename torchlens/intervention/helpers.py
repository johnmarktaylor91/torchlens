"""Built-in helper constructors for future TorchLens interventions."""

from collections.abc import Callable
from typing import Any

from .errors import _not_implemented


def zero_ablate() -> Any:
    """Create a future zero-ablation helper.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 3 implements helpers.
    """

    return _not_implemented("zero_ablate", "Phase 3")


def mean_ablate(*, over: str = "self") -> Any:
    """Create a future mean-ablation helper.

    Parameters
    ----------
    over:
        Future locality scope for the mean.
    """

    return _not_implemented("mean_ablate", "Phase 3")


def resample_ablate(*, from_: Any) -> Any:
    """Create a future resample-ablation helper.

    Parameters
    ----------
    from_:
        Source activations or log for future resampling.
    """

    return _not_implemented("resample_ablate", "Phase 3")


def steer(
    direction: Any,
    *,
    coef: float = 1.0,
    positions: Any | None = None,
    tensor_filter: Any | None = None,
) -> Any:
    """Create a future steering helper.

    Parameters
    ----------
    direction:
        Direction tensor or compatible future object.
    coef:
        Scaling coefficient for the direction.
    positions:
        Optional residual-stream position shorthand.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("steer", "Phase 3")


def scale(factor: float, *, tensor_filter: Any | None = None) -> Any:
    """Create a future scaling helper.

    Parameters
    ----------
    factor:
        Multiplicative factor.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("scale", "Phase 3")


def clamp(
    *,
    min: float | None = None,
    max: float | None = None,
    tensor_filter: Any | None = None,
) -> Any:
    """Create a future clamping helper.

    Parameters
    ----------
    min:
        Optional lower bound.
    max:
        Optional upper bound.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("clamp", "Phase 3")


def noise(std: float, *, seed: int | None = None, tensor_filter: Any | None = None) -> Any:
    """Create a future noise helper.

    Parameters
    ----------
    std:
        Standard deviation for future noise.
    seed:
        Optional deterministic seed.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("noise", "Phase 3")


def project_onto(subspace: Any, *, tensor_filter: Any | None = None) -> Any:
    """Create a future projection-onto helper.

    Parameters
    ----------
    subspace:
        Future subspace representation.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("project_onto", "Phase 3")


def project_off(direction: Any, *, tensor_filter: Any | None = None) -> Any:
    """Create a future projection-off helper.

    Parameters
    ----------
    direction:
        Direction to remove in a future implementation.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("project_off", "Phase 3")


def swap_with(value: Any, *, tensor_filter: Any | None = None) -> Any:
    """Create a future activation-swap helper.

    Parameters
    ----------
    value:
        Replacement value for a future intervention.
    tensor_filter:
        Optional future tensor slice specification.
    """

    return _not_implemented("swap_with", "Phase 3")


def splice_module(module: Any, *, input: str = "activation", output: str = "activation") -> Any:
    """Create a future module-splicing helper.

    Parameters
    ----------
    module:
        Module to call as a future black-box helper.
    input:
        Future input routing policy.
    output:
        Future output routing policy.
    """

    return _not_implemented("splice_module", "Phase 3")


def bwd_hook(fn: Callable[..., Any]) -> Any:
    """Create a future backward hook helper.

    Parameters
    ----------
    fn:
        Callable to wrap for future tensor backward hooks.
    """

    return _not_implemented("bwd_hook", "Phase 3")


def gradient_zero() -> Any:
    """Create a future gradient-zero helper.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 3 implements backward helpers.
    """

    return _not_implemented("gradient_zero", "Phase 3")


def gradient_scale(factor: float) -> Any:
    """Create a future gradient-scale helper.

    Parameters
    ----------
    factor:
        Multiplicative gradient factor.
    """

    return _not_implemented("gradient_scale", "Phase 3")


__all__ = [
    "bwd_hook",
    "clamp",
    "gradient_scale",
    "gradient_zero",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "resample_ablate",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
]
