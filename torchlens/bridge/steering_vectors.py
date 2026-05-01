"""steering-vectors bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import activation_at


def vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    trainer: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train or build a steering vector from saved TorchLens activations.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    positive_site:
        Site containing positive-class activations.
    negative_site:
        Optional site containing negative-class activations.
    trainer:
        Optional callable trainer. Defaults to the installed package's
        ``train_steering_vector`` function.
    **kwargs:
        Additional keyword arguments forwarded to the trainer.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the downstream steering vector.

    Raises
    ------
    ImportError
        If steering-vectors is unavailable.
    RuntimeError
        If the installed package does not expose a supported trainer.
    """

    try:
        import steering_vectors as steering_module
    except ImportError as exc:
        raise ImportError(
            "steering-vectors bridge requires the `steering` extra: install torchlens[steering]."
        ) from exc

    positive = activation_at(log, positive_site)
    negative = None if negative_site is None else activation_at(log, negative_site)
    train = _resolve_trainer(steering_module, trainer)
    result = train(positive, negative, **kwargs)
    return {
        "schema": "torchlens.steering_vectors.v1",
        "vector": result,
        "positive": positive,
        "negative": negative,
    }


def _resolve_trainer(module: Any, trainer: Any | None) -> Any:
    """Return a steering-vector trainer callable.

    Parameters
    ----------
    module:
        Imported ``steering_vectors`` module.
    trainer:
        Optional explicit trainer.

    Returns
    -------
    Any
        Callable trainer.

    Raises
    ------
    RuntimeError
        If no trainer is available.
    """

    if trainer is not None:
        return trainer
    candidate = getattr(module, "train_steering_vector", None)
    if callable(candidate):
        return candidate
    vector_cls = getattr(module, "SteeringVector", None)
    class_train = getattr(vector_cls, "train", None)
    if callable(class_train):
        return class_train
    raise RuntimeError("Installed steering_vectors does not expose a supported trainer.")


__all__ = ["vector"]
