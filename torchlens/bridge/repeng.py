"""repeng bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import out_at


def control_vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    vector_factory: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a repeng control vector from saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    positive_site:
        Site containing positive outs.
    negative_site:
        Optional site containing negative outs.
    vector_factory:
        Optional downstream factory or class.
    **kwargs:
        Additional keyword arguments forwarded to the downstream factory.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the downstream control vector.

    Raises
    ------
    ImportError
        If repeng is unavailable.
    RuntimeError
        If no supported factory is exposed.
    """

    try:
        import repeng as repeng_module
    except ImportError as exc:
        raise ImportError(
            "repeng bridge requires the `repeng` extra: install torchlens[repeng]."
        ) from exc

    positive = out_at(log, positive_site)
    negative = None if negative_site is None else out_at(log, negative_site)
    factory = _resolve_factory(repeng_module, vector_factory)
    result = _call_factory(factory, positive=positive, negative=negative, **kwargs)
    return {
        "schema": "torchlens.repeng.v1",
        "control_vector": result,
        "positive": positive,
        "negative": negative,
    }


def _resolve_factory(module: Any, vector_factory: Any | None) -> Any:
    """Return a repeng control-vector factory.

    Parameters
    ----------
    module:
        Imported ``repeng`` module.
    vector_factory:
        Optional explicit factory.

    Returns
    -------
    Any
        Callable factory or classmethod.

    Raises
    ------
    RuntimeError
        If no factory is available.
    """

    if vector_factory is not None:
        return vector_factory
    control_vector_cls = getattr(module, "ControlVector", None)
    train = getattr(control_vector_cls, "train", None)
    if callable(train):
        return train
    if callable(control_vector_cls):
        return control_vector_cls
    raise RuntimeError("Installed repeng does not expose ControlVector or ControlVector.train.")


def _call_factory(factory: Any, *, positive: Any, negative: Any | None, **kwargs: Any) -> Any:
    """Call a repeng vector factory with the normalized out pair.

    Parameters
    ----------
    factory:
        Factory callable.
    positive:
        Positive outs.
    negative:
        Optional negative outs.
    **kwargs:
        Additional factory keyword arguments.

    Returns
    -------
    Any
        Downstream vector object.
    """

    return factory(positive, negative, **kwargs)


__all__ = ["control_vector"]
