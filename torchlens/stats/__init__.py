"""Streaming statistics for out aggregation."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

import torch
from torch import nn


class StreamingStat(Protocol):
    """Protocol implemented by all streaming statistic accumulators."""

    name: str | None

    def update(self, value: Any) -> None:
        """Update the accumulator with one batch value."""

    def result(self) -> Any:
        """Return the finalized statistic value."""


def _as_float_tensor(value: Any) -> torch.Tensor:
    """Return ``value`` as a detached CPU float tensor.

    Parameters
    ----------
    value:
        Tensor-like value.

    Returns
    -------
    torch.Tensor
        Flattened CPU float tensor.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return torch.as_tensor(value, dtype=torch.float64).reshape(-1)


class Mean:
    """Running mean accumulator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._count = 0
        self._mean: torch.Tensor | None = None

    def update(self, value: Any) -> None:
        """Update the running mean.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        tensor = _as_float_tensor(value)
        if tensor.numel() == 0:
            return
        batch_mean = tensor.mean()
        batch_count = int(tensor.numel())
        if self._mean is None:
            self._mean = batch_mean
            self._count = batch_count
            return
        total = self._count + batch_count
        self._mean = self._mean + (batch_mean - self._mean) * (batch_count / total)
        self._count = total

    def result(self) -> float:
        """Return the finalized mean.

        Returns
        -------
        float
            Running mean, or NaN when no values were seen.
        """

        if self._mean is None:
            return math.nan
        return float(self._mean.item())


class Norm:
    """Running mean of per-update tensor norms."""

    def __init__(self, p: float = 2.0, name: str | None = None) -> None:
        """Initialize the norm accumulator.

        Parameters
        ----------
        p:
            Norm order passed to ``torch.linalg.vector_norm``.
        name:
            Optional metric name.
        """

        self.name = name
        self.p = float(p)
        self._mean = Mean()

    def update(self, value: Any) -> None:
        """Update the running norm mean."""

        tensor = _as_float_tensor(value)
        if tensor.numel() == 0:
            return
        self._mean.update(torch.linalg.vector_norm(tensor, ord=self.p))

    def result(self) -> float:
        """Return the finalized mean norm."""

        return self._mean.result()


class Quantile:
    """Reservoir-sampling running quantile estimator."""

    def __init__(
        self,
        quantiles: Iterable[float] = (0.5, 0.95, 0.99),
        name: str | None = None,
        reservoir_size: int = 8192,
    ) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        quantiles:
            Quantiles in ``[0, 1]`` to estimate.
        name:
            Optional metric name.
        reservoir_size:
            Maximum sampled values retained in memory.
        """

        self.name = name
        self.quantiles = tuple(float(q) for q in quantiles)
        self.reservoir_size = int(reservoir_size)
        self._seen = 0
        self._reservoir: list[float] = []

    def update(self, value: Any) -> None:
        """Update the reservoir.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        for item in _as_float_tensor(value).tolist():
            self._seen += 1
            if len(self._reservoir) < self.reservoir_size:
                self._reservoir.append(float(item))
                continue
            replacement = random.randint(0, self._seen - 1)
            if replacement < self.reservoir_size:
                self._reservoir[replacement] = float(item)

    def result(self) -> dict[float, float]:
        """Return finalized quantile estimates.

        Returns
        -------
        dict[float, float]
            Mapping from requested quantile to estimated value.
        """

        if not self._reservoir:
            return {q: math.nan for q in self.quantiles}
        tensor = torch.tensor(self._reservoir, dtype=torch.float64)
        return {q: float(torch.quantile(tensor, q).item()) for q in self.quantiles}


class TopK:
    """Streaming top-k value tracker."""

    def __init__(self, k: int = 10, name: str | None = None) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        k:
            Number of largest scalar values to retain.
        name:
            Optional metric name.
        """

        self.name = name
        self.k = int(k)
        self._heap: list[float] = []

    def update(self, value: Any) -> None:
        """Update the tracked top-k values.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        if self.k <= 0:
            return
        for item in _as_float_tensor(value).tolist():
            scalar = float(item)
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, scalar)
            elif scalar > self._heap[0]:
                heapq.heapreplace(self._heap, scalar)

    def result(self) -> list[float]:
        """Return top values in descending order.

        Returns
        -------
        list[float]
            Retained top-k values.
        """

        return sorted(self._heap, reverse=True)


class Covariance:
    """Running covariance matrix accumulator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._count = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def update(self, value: Any) -> None:
        """Update covariance from one batch.

        Parameters
        ----------
        value:
            Tensor-like batch. One-dimensional inputs are treated as one row.
        """

        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.reshape(tensor.shape[0], -1)
        for row in tensor:
            self._count += 1
            if self._mean is None:
                self._mean = torch.zeros_like(row)
                self._m2 = torch.zeros((row.numel(), row.numel()), dtype=torch.float64)
            assert self._m2 is not None
            delta = row - self._mean
            self._mean = self._mean + delta / self._count
            self._m2 = self._m2 + torch.outer(delta, row - self._mean)

    def result(self) -> torch.Tensor:
        """Return the finalized covariance matrix.

        Returns
        -------
        torch.Tensor
            Covariance matrix.
        """

        if self._m2 is None:
            return torch.empty((0, 0), dtype=torch.float64)
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)


class PCA:
    """Simple incremental PCA backed by running covariance."""

    def __init__(self, n_components: int, name: str | None = None) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        n_components:
            Number of principal components to return.
        name:
            Optional metric name.
        """

        self.name = name
        self.n_components = int(n_components)
        self._covariance = Covariance(name=name)

    def update(self, value: Any) -> None:
        """Update the PCA estimator.

        Parameters
        ----------
        value:
            Tensor-like batch.
        """

        self._covariance.update(value)

    def result(self) -> dict[str, torch.Tensor]:
        """Return components and explained variances.

        Returns
        -------
        dict[str, torch.Tensor]
            ``components`` and ``explained_variance`` tensors.
        """

        cov = self._covariance.result()
        if cov.numel() == 0:
            return {
                "components": torch.empty((0, 0), dtype=torch.float64),
                "explained_variance": torch.empty((0,), dtype=torch.float64),
            }
        values, vectors = torch.linalg.eigh(cov)
        order = torch.argsort(values, descending=True)[: self.n_components]
        return {"components": vectors[:, order].T, "explained_variance": values[order]}


class Aggregator:
    """Combine multiple streaming accumulators in one update pass."""

    def __init__(self, *stats: StreamingStat, name: str | None = None) -> None:
        """Initialize the combined aggregator.

        Parameters
        ----------
        *stats:
            Streaming statistic instances.
        name:
            Optional metric name.
        """

        self.name = name
        self.stats = tuple(stats)

    def update(self, value: Any) -> None:
        """Update each child statistic.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        for stat in self.stats:
            stat.update(value)

    def result(self) -> dict[str, Any]:
        """Return each child statistic result.

        Returns
        -------
        dict[str, Any]
            Mapping from child names/classes to finalized results.
        """

        results: dict[str, Any] = {}
        for index, stat in enumerate(self.stats):
            key = stat.name or type(stat).__name__
            if key in results:
                key = f"{key}_{index}"
            results[key] = stat.result()
        return results


def _metric_value_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one metric input value from a Trace.

    Parameters
    ----------
    log:
        Captured model log.
    metric_name:
        Layer selector or ``"output"``.

    Returns
    -------
    Any
        Tensor-like value for the metric.
    """

    if metric_name == "output" and log.output_layers:
        return log[log.output_layers[-1]].out
    try:
        return log[metric_name].out
    except Exception:
        matches = [
            layer
            for layer in log.layer_list
            if metric_name in str(layer.layer_label) and layer.has_saved_activation
        ]
        if not matches:
            raise KeyError(f"No saved out matched metric {metric_name!r}.")
        return matches[0].out


def _metric_grad_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one gradient metric input value from a Trace."""

    if metric_name == "output" and log.output_layers:
        return log[log.output_layers[-1]].grad
    try:
        value = log[metric_name].grad
    except Exception:
        matches = [
            layer
            for layer in log.layer_list
            if metric_name in str(layer.layer_label) and layer.has_grad
        ]
        if not matches:
            raise KeyError(f"No saved grad matched metric {metric_name!r}.")
        value = matches[0].grad
    if value is None:
        raise KeyError(f"No saved grad matched metric {metric_name!r}.")
    return value


def _split_batch_for_loss(batch: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return model input and extra loss arguments from a dataloader batch."""

    if isinstance(batch, tuple) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    if isinstance(batch, list) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    return batch, ()


def aggregate(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    target: str = "out",
    loss_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Stream outs through metric accumulators.

    Parameters
    ----------
    model:
        Model to capture.
    dataloader:
        Iterable of model inputs.
    metrics:
        Mapping from layer selector to streaming statistic.
    target:
        ``"out"`` for activation statistics or ``"grad"`` for gradient
        statistics.
    loss_fn:
        Callable used to build a loss from ``(output, *batch_tail)`` when
        ``target="grad"``.

    Returns
    -------
    dict[str, Any]
        Finalized metric results.
    """

    from .. import trace

    if target not in {"out", "grad"}:
        raise ValueError("target must be 'out' or 'grad'")
    if target == "grad" and loss_fn is None:
        raise TypeError("aggregate(target='grad') requires loss_fn=")
    grad_loss_fn = loss_fn

    layers = [name for name in metrics if name != "output"]
    capture_layers: str | list[str] = layers if layers else "all"
    for batch in dataloader:
        model_input, loss_args = _split_batch_for_loss(batch)
        log = trace(
            model,
            model_input,
            layers_to_save=capture_layers,
            gradients_to_save=capture_layers if target == "grad" else None,
        )
        try:
            if target == "grad":
                if grad_loss_fn is None:
                    raise TypeError("aggregate(target='grad') requires loss_fn=")
                loss = grad_loss_fn(_metric_value_from_log(log, "output"), *loss_args)
                log.log_backward(loss)
            for metric_name, stat in metrics.items():
                value = (
                    _metric_grad_from_log(log, metric_name)
                    if target == "grad"
                    else _metric_value_from_log(log, metric_name)
                )
                stat.update(value)
        finally:
            log.cleanup()
    return {name: stat.result() for name, stat in metrics.items()}


__all__ = [
    "Aggregator",
    "Covariance",
    "Mean",
    "Norm",
    "PCA",
    "Quantile",
    "StreamingStat",
    "TopK",
    "aggregate",
]
