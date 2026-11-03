"""Attribution result and error base types.

This module is the import root of the attribution package: every attribution
submodule may import from here without creating a cycle. It also holds the
``draw(color_by=)`` painter for attribution results (``Attribution._color_by_source``
calls it here, so this module imports no other attribution module). ``AttributionError``
and ``AttributionResult`` remain re-exported through ``_core`` and the package
``__init__`` for compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

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
class Attribution:
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

    # Read by ``draw(color_by=)`` so the drawing code needs no import of this package.
    _color_by_kind: ClassVar[str] = "attribution"

    def _color_by_source(self, trace: Any, source: Any) -> Any:
        """Return the ``draw(color_by=)`` value source for ``source`` (this result or a collection)."""

        return _attribution_painter(trace, source, "abs_sum")

    def __repr__(self) -> str:
        """Return a compact representation without dumping attribution tensors."""

        return (
            "AttributionResult("
            f"method={self.method!r}, "
            f"values={_summarize_attribution_tree(self.values)}, "
            f"target_repr={self.target_repr!r}, "
            f"extra_keys={sorted(self.extra.keys())!r})"
        )


_OverlayReduce: TypeAlias = Literal["abs_sum", "abs_mean", "sum", "max"]

_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."

_REDUCERS: dict[str, Callable[[Tensor], float]] = {
    "abs_sum": lambda t: float(t.detach().abs().sum().item()),
    "abs_mean": lambda t: float(t.detach().abs().mean().item()),
    "sum": lambda t: float(t.detach().sum().item()),
    "max": lambda t: float(t.detach().max().item()),
}


def _reduce_value(value: Any, reduce: str) -> float:
    """Reduce one attribution value to a finite color scalar.

    Parameters
    ----------
    value
        Attribution tensor or plain real scalar.
    reduce
        Closed-vocabulary reduction applied to tensor values.

    Returns
    -------
    float
        Scalar magnitude for the color channel.

    Raises
    ------
    AttributionError
        If ``value`` is neither a tensor nor a real scalar.
    """

    if isinstance(value, Tensor):
        return _REDUCERS[reduce](value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttributionError(
            "overlay values must be attribution tensors or real scalars; "
            f"received {type(value).__name__}"
        )
    return float(value)


def _overlay_refusal(message: str, *, code: str) -> AttributionError:
    """Build one coded overlay refusal ending in its docs pointer."""

    return AttributionError(f"{message} {_DOCS_POINTER}", code=code)


def _entry_items(source: Any) -> list[tuple[str, Any]]:
    """Normalize ``source`` into ``(layer key, raw value)`` pairs.

    Parameters
    ----------
    source
        One layer-scoped :class:`AttributionResult`, an iterable of them, or a
        mapping from module name / layer label to result, tensor, or scalar.

    Returns
    -------
    list[tuple[str, Any]]
        Key/value pairs prior to reduction.

    Raises
    ------
    AttributionError
        If a result carries no ``extra["layer"]`` key to anchor it, or the
        source kind is unsupported.
    """

    if isinstance(source, Attribution):
        source = (source,)
    if isinstance(source, Mapping):
        items: list[tuple[str, Any]] = []
        for key, value in source.items():
            items.append((key, value.values if isinstance(value, Attribution) else value))
        return items
    if isinstance(source, Iterable):
        items = []
        for result in source:
            if not isinstance(result, Attribution):
                raise AttributionError(
                    "overlay iterables must contain AttributionResult entries; "
                    f"received {type(result).__name__}"
                )
            layer = result.extra.get("layer")
            if not isinstance(layer, str):
                raise AttributionError(
                    f"{result.method!r} result carries no extra['layer'] anchor; "
                    "input attributions color no graph node -- pass a mapping "
                    "{module_name_or_layer_label: value} instead"
                )
            items.append((layer, result.values))
        return items
    raise AttributionError(
        "overlay source must be an AttributionResult, an iterable of them, or "
        f"a mapping; received {type(source).__name__}"
    )


def _classify_entries(
    trace: Any,
    source: Any,
    reduce: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve overlay entries into module-address and layer-label value maps.

    Parameters
    ----------
    trace
        Trace whose module outputs and layer labels anchor the keys.
    source
        Overlay source accepted by :func:`overlay`.
    reduce
        Validated closed-vocabulary reduction name.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        Module-address-keyed and layer-label-keyed scalar maps.

    Raises
    ------
    AttributionError
        If a key matches neither a module output nor a layer label.
    """

    known_modules: set[str] = set()
    known_labels: set[str] = set()
    for label in trace.layer_labels:
        known_labels.add(label)
        known_modules.update(trace[label].output_of_modules)

    module_values: dict[str, float] = {}
    label_values: dict[str, float] = {}
    for key, raw in _entry_items(source):
        value = _reduce_value(raw, reduce)
        if key in known_modules:
            module_values[key] = value
        elif key in known_labels:
            label_values[key] = value
        else:
            raise AttributionError(
                f"overlay key {key!r} matches no module output and no layer "
                "label on this trace; use a name from model.named_modules() "
                "or a label from trace.layer_labels"
            )
    return module_values, label_values


def _attribution_painter(trace: Any, source: Any, reduce: str) -> Callable[[Any], float | None]:
    """Build the ``color_by`` callable painting ``source``'s reduced values onto ``trace``'s nodes.

    Nodes that are the output of an attributed module, or match an
    attributed layer label, return that value; every other node returns
    ``None``. ``source`` is a layer-scoped :class:`Attribution`, an iterable
    of them, or a mapping (see :func:`_entry_items`).
    """

    module_values, label_values = _classify_entries(trace, source, reduce)

    def attribution(node: Any) -> float | None:
        """Return the attributed magnitude for one rendered node."""

        if getattr(node, "is_input", False) or getattr(node, "is_output", False):
            return None
        label = getattr(node, "layer_label", None)
        if label in label_values:
            return label_values[label]
        for address in getattr(node, "output_of_modules", ()) or ():
            if address in module_values:
                return module_values[address]
        return None

    return attribution


__all__ = [
    "AttributionError",
    "AttributionWarning",
    "Attribution",
    "AttributionValueTree",
]
