"""Attribution result and error base types.

This module is the import root of the attribution package: every attribution
submodule may import from here without creating a cycle. ``AttributionError``
and ``AttributionResult`` remain re-exported through ``_core`` and the package
``__init__`` for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from torch import Tensor

from torchlens.errors._base import ConfigurationError, TorchLensWarning

AttributionValueTree: TypeAlias = Tensor | tuple[Any, ...] | list[Any] | dict[str, Any]


class AttributionError(ConfigurationError, ValueError):
    """Error raised for unsupported or invalid attribution requests.

    Historically a bare ``ValueError`` subclass; the ``TorchLensError`` base is
    additive (``except ValueError`` still catches it) and supplies the shared
    structured-payload contract: new raise sites pass ``code=`` and end their
    message with a ``Remedy:`` sentence per the S-17 refusal-site contract.
    """


class AttributionWarning(TorchLensWarning):
    """Coded disclosure warning for attribution methods and metrics (S-18)."""


def _summarize_attribution_tree(value: Any) -> str:
    """Return shape/dtype summaries for attribution values.

    Parameters
    ----------
    value
        Attribution value tree whose tensors should be summarized.

    Returns
    -------
    str
        Compact human-readable structure summary.
    """

    if isinstance(value, Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)!r}, "
            f"dtype={value.dtype}, device={value.device.type!r})"
        )
    if isinstance(value, tuple):
        return "(" + ", ".join(_summarize_attribution_tree(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_summarize_attribution_tree(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{key!r}: {_summarize_attribution_tree(item)}" for key, item in value.items()
        )
        return "{" + items + "}"
    return repr(value)


@dataclass(frozen=True)
class AttributionResult:
    """Container for provisional input-attribution results.

    Parameters
    ----------
    method
        Name of the attribution method that produced this result.
    values
        Attribution values. Single attributed-leaf calls return a bare tensor for
        v1 compatibility. Multi-leaf calls return a nested structure with
        attribution tensors in attributed positions and ``None`` in non-attributed
        leaf positions.
    target_repr
        Compact representation of the scalarization target.
    extra
        Method-specific metadata. This schema is intentionally minimal for v1.
    """

    method: str
    values: AttributionValueTree
    target_repr: str
    extra: dict[str, Any]

    def __repr__(self) -> str:
        """Return a compact representation without dumping attribution tensors."""

        return (
            "AttributionResult("
            f"method={self.method!r}, "
            f"values={_summarize_attribution_tree(self.values)}, "
            f"target_repr={self.target_repr!r}, "
            f"extra_keys={sorted(self.extra.keys())!r})"
        )


__all__ = [
    "AttributionError",
    "AttributionWarning",
    "AttributionResult",
    "AttributionValueTree",
]
