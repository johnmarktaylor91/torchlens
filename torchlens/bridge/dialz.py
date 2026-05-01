"""dialz bridge helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._utils import tensor_layers


def analyze(
    log: Any,
    *,
    sites: Iterable[Any] | None = None,
    analyzer: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a dialz analyzer over saved TorchLens activations.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    sites:
        Optional sites to include. Defaults to all non-input tensor layers.
    analyzer:
        Optional analyzer callable or object exposing ``analyze``.
    **kwargs:
        Additional keyword arguments forwarded to the analyzer.

    Returns
    -------
    dict[str, Any]
        Contract payload containing labels, activations, and downstream result.

    Raises
    ------
    ImportError
        If dialz is unavailable.
    RuntimeError
        If no supported analyzer is exposed.
    """

    try:
        import dialz as dialz_module
    except ImportError as exc:
        raise ImportError(
            "dialz bridge requires the `dialz` extra: install torchlens[dialz]."
        ) from exc

    layers = tensor_layers(log, sites)
    labels = [str(getattr(layer, "layer_label", "layer")) for layer in layers]
    activations = [getattr(layer, "activation") for layer in layers]
    runner = _resolve_analyzer(dialz_module, analyzer)
    result = _call_analyzer(runner, activations=activations, labels=labels, **kwargs)
    return {
        "schema": "torchlens.dialz.v1",
        "labels": labels,
        "activations": activations,
        "result": result,
    }


def _resolve_analyzer(module: Any, analyzer: Any | None) -> Any:
    """Return a dialz analyzer callable.

    Parameters
    ----------
    module:
        Imported ``dialz`` module.
    analyzer:
        Optional explicit analyzer.

    Returns
    -------
    Any
        Analyzer callable or object.

    Raises
    ------
    RuntimeError
        If no analyzer is available.
    """

    if analyzer is not None:
        return analyzer
    for attr_name in ("analyze", "Analyzer", "Dialz"):
        candidate = getattr(module, attr_name, None)
        if callable(candidate):
            return candidate
    raise RuntimeError("Installed dialz does not expose analyze, Analyzer, or Dialz.")


def _call_analyzer(runner: Any, *, activations: list[Any], labels: list[str], **kwargs: Any) -> Any:
    """Call a dialz analyzer in function or object form.

    Parameters
    ----------
    runner:
        Analyzer callable or object exposing ``analyze``.
    activations:
        Saved activations.
    labels:
        TorchLens layer labels.
    **kwargs:
        Additional analyzer keyword arguments.

    Returns
    -------
    Any
        Downstream analyzer result.
    """

    if hasattr(runner, "analyze"):
        return runner.analyze(activations, labels=labels, **kwargs)
    return runner(activations, labels=labels, **kwargs)


__all__ = ["analyze"]
