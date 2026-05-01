"""Brain-Score bridge helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._utils import tensor_layers


def per_layer(
    log: Any,
    benchmark: Any,
    *,
    sites: Iterable[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a Brain-Score-style benchmark independently for saved layers.

    This offline adapter accepts a callable benchmark fixture so tests can run
    without network access or GUI resources.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` containing saved activations.
    benchmark:
        Callable benchmark accepting ``(activation, layer=..., **kwargs)``.
    sites:
        Optional iterable of layer labels/selectors to score. Defaults to all
        saved tensor layers except input placeholders.
    **kwargs:
        Additional benchmark keyword arguments.

    Returns
    -------
    dict[str, Any]
        Mapping from TorchLens layer label to benchmark score.

    Raises
    ------
    TypeError
        If ``benchmark`` is not callable.
    """

    if not callable(benchmark):
        raise TypeError("Brain-Score bridge currently requires a callable offline benchmark.")

    scores: dict[str, Any] = {}
    for layer in tensor_layers(log, sites):
        activation = getattr(layer, "activation")
        label = str(getattr(layer, "layer_label", getattr(layer, "layer_label_no_pass", "layer")))
        # TODO: connect the real Brain-Score Benchmark API once the offline
        # vendored fixtures are available in the launch extras matrix.
        scores[label] = benchmark(activation, layer=label, **kwargs)
    return scores


__all__ = ["per_layer"]
