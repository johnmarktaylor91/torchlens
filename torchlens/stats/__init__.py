"""Streaming statistics for activation aggregation."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterable, Mapping
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
    """Resolve one metric input value from a ModelLog.

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
        return log[log.output_layers[-1]].activation
    try:
        return log[metric_name].activation
    except Exception:
        matches = [
            layer
            for layer in log.layer_list
            if metric_name in str(layer.layer_label) and layer.has_saved_activations
        ]
        if not matches:
            raise KeyError(f"No saved activation matched metric {metric_name!r}.")
        return matches[0].activation


def aggregate(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
) -> dict[str, Any]:
    """Stream activations through metric accumulators.

    Parameters
    ----------
    model:
        Model to capture.
    dataloader:
        Iterable of model inputs.
    metrics:
        Mapping from layer selector to streaming statistic.

    Returns
    -------
    dict[str, Any]
        Finalized metric results.
    """

    from .. import log_forward_pass

    layers = [name for name in metrics if name != "output"]
    capture_layers: str | list[str] = layers if layers else "all"
    for batch in dataloader:
        log = log_forward_pass(model, batch, layers_to_save=capture_layers)
        try:
            for metric_name, stat in metrics.items():
                stat.update(_metric_value_from_log(log, metric_name))
        finally:
            log.cleanup()
    return {name: stat.result() for name, stat in metrics.items()}


__all__ = [
    "Aggregator",
    "Covariance",
    "Mean",
    "PCA",
    "Quantile",
    "StreamingStat",
    "TopK",
    "aggregate",
]
