"""Streaming statistic kernels (stats/ promotion, C01 item 18)."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterable
from typing import Any, Protocol, cast

import torch

__tl_layer__ = "L5"


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
    """Running mean of per-update tensor norms.

    Each ``update()`` reduces the provided batch to one scalar norm before the
    running mean is updated. This is intentionally different from a global norm
    over all elements seen across all updates, so regrouping the same values
    into different update batches can change the reported result.
    """

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
    """Reservoir-sampling running quantile estimator.

    The estimate is exact while at most ``reservoir_size`` values have been
    seen; beyond that it is a uniform subsample, whose quantile standard
    error is ~``sqrt(q * (1 - q) / reservoir_size)`` in RANK space (about
    +/-0.55 percentile points at the median with the default reservoir).

    Sampling uses a PRIVATE ``random.Random(seed)`` stream: it never draws
    from (or perturbs) the process-global ``random`` module, which capture
    replay snapshots and restores as a protected engine, and two runs with
    the same seed and the same update stream produce identical estimates.
    """

    def __init__(
        self,
        quantiles: Iterable[float] = (0.5, 0.95, 0.99),
        name: str | None = None,
        reservoir_size: int = 8192,
        seed: int | None = 0,
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
        seed:
            Seed for the private sampling stream. The default ``0`` makes
            estimates reproducible run-to-run; pass ``None`` for a
            fresh-entropy stream (still decoupled from the global engine).
        """

        self.name = name
        self.quantiles = tuple(float(q) for q in quantiles)
        self.reservoir_size = int(reservoir_size)
        self.seed = seed
        self._rng = random.Random(seed)
        self._seen = 0
        self._reservoir: list[float] = []

    def update(self, value: Any) -> None:
        """Update the reservoir.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        items = _as_float_tensor(value).tolist()
        start = 0
        free = self.reservoir_size - len(self._reservoir)
        if free > 0:
            # Fill phase: the first reservoir_size values are all retained,
            # so bulk-extend instead of looping (algorithm-R equivalent).
            start = min(free, len(items))
            self._reservoir.extend(float(item) for item in items[:start])
            self._seen += start
        for item in items[start:]:
            self._seen += 1
            replacement = self._rng.randint(0, self._seen - 1)
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
            return dict.fromkeys(self.quantiles, math.nan)
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

        NaN values are ignored: they have no order, so they can neither rank
        among the top k nor displace a real value. The batch is reduced with
        ``torch.topk`` and merged with the retained values (identical
        semantics to the former per-element Python loop at a fraction of the
        cost -- the loop was ~50x slower than ``torch.topk`` on real batches).

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        if self.k <= 0:
            return
        tensor = _as_float_tensor(value)
        tensor = tensor[~tensor.isnan()]
        if tensor.numel() == 0:
            return
        if tensor.numel() > self.k:
            tensor = torch.topk(tensor, self.k).values
        if self._heap:
            merged = torch.cat([torch.tensor(self._heap, dtype=torch.float64), tensor])
        else:
            merged = tensor
        if merged.numel() > self.k:
            merged = torch.topk(merged, self.k).values
        self._heap = [float(item) for item in merged.tolist()]
        heapq.heapify(self._heap)

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

    def __init__(
        self,
        name: str | None = None,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        device:
            Compute/state device (F20 D-25; default: historical CPU).
        dtype:
            Floating compute dtype (default: historical ``float64``).
        """

        self.name = name
        self._device, self._dtype = _resolve_placement(device, dtype)
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

        tensor = torch.as_tensor(value).detach().to(device=self._device, dtype=self._dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.reshape(tensor.shape[0], -1)
        if self._mean is not None and tensor.shape[1] != self._mean.numel():
            raise ValueError("Covariance feature dimensions cannot change across updates.")
        rows = tensor.shape[0]
        if rows == 0:
            return
        if self._mean is None:
            self._mean = torch.zeros(tensor.shape[1], dtype=self._dtype, device=self._device)
            self._m2 = torch.zeros(
                (tensor.shape[1], tensor.shape[1]), dtype=self._dtype, device=self._device
            )
        assert self._m2 is not None
        # Chan et al. batch combine: one FxF update per BATCH with in-place
        # accumulation (the per-row Welford loop allocated a fresh FxF outer
        # product per ROW, ~33 MB of churn for modest feature widths).
        batch_mean = tensor.mean(dim=0)
        centered = tensor - batch_mean
        total = self._count + rows
        delta = batch_mean - self._mean
        self._m2 += centered.T @ centered
        self._m2 += torch.outer(delta, delta) * (self._count * rows / total)
        self._mean += delta * (rows / total)
        self._count = total

    def result(self) -> torch.Tensor:
        """Return the finalized covariance matrix.

        Returns
        -------
        torch.Tensor
            Covariance matrix.
        """

        if self._m2 is None:
            return torch.empty((0, 0), dtype=self._dtype, device=self._device)
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)


def _resolve_placement(device: Any, dtype: Any) -> tuple[torch.device, torch.dtype]:
    """Resolve the accumulator compute placement (F20, brainpipe D-25).

    The ONE policy seam for accumulator ``device=``/``dtype=`` keywords, so
    later accumulators cannot fork conventions: ``None`` means the historical
    CPU ``float64`` contract (bit-identical default), an explicit device
    keeps state and computation there, and only floating dtypes are lawful.

    Parameters
    ----------
    device:
        Target device or ``None`` for the historical CPU placement.
    dtype:
        Target floating dtype or ``None`` for the historical ``float64``.

    Returns
    -------
    tuple[torch.device, torch.dtype]
        Resolved placement.

    Raises
    ------
    ValueError
        If ``dtype`` is not a floating dtype.
    """

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    resolved_dtype = dtype if dtype is not None else torch.float64
    if not isinstance(resolved_dtype, torch.dtype) or not resolved_dtype.is_floating_point:
        raise ValueError(f"Accumulator dtype must be a floating torch.dtype; got {dtype!r}.")
    return resolved_device, resolved_dtype


def _as_feature_matrix(
    value: Any,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``value`` as a detached feature matrix on the given placement.

    Parameters
    ----------
    value:
        Tensor-like batch. One-dimensional inputs are treated as one row.
    device:
        Target device (default: the historical CPU placement).
    dtype:
        Target dtype (default: the historical ``float64``).

    Returns
    -------
    torch.Tensor
        A two-dimensional ``(n_rows, n_features)`` tensor.
    """

    tensor = (
        torch.as_tensor(value).detach().to(device=device or "cpu", dtype=dtype or torch.float64)
    )
    if tensor.ndim == 0:
        raise ValueError("A feature batch must have at least one dimension.")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(tensor.shape[0], -1)


class CrossCovariance:
    """Running cross-covariance matrix accumulator.

    The accumulator retains only feature-sized running means and the cross
    second moment. Inputs are converted to detached tensors on the resolved
    placement (default: the historical CPU ``float64``).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        device:
            Compute/state device (F20 D-25; default: historical CPU).
        dtype:
            Floating compute dtype (default: historical ``float64``).
        """

        self.name = name
        self._device, self._dtype = _resolve_placement(device, dtype)
        self._count = 0
        self._mean_a: torch.Tensor | None = None
        self._mean_b: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def update(self, a: Any, b: Any) -> None:
        """Update cross-covariance from one paired batch.

        Parameters
        ----------
        a:
            First tensor-like batch, with rows as observations.
        b:
            Second tensor-like batch, with rows as observations.

        Raises
        ------
        ValueError
            If the batches have different row counts or a feature dimension
            changes across updates.
        """

        matrix_a = _as_feature_matrix(a, self._device, self._dtype)
        matrix_b = _as_feature_matrix(b, self._device, self._dtype)
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                "CrossCovariance requires matched row counts; "
                f"got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )
        if self._mean_a is not None and matrix_a.shape[1] != self._mean_a.numel():
            raise ValueError("CrossCovariance feature dimensions cannot change across updates.")
        if self._mean_b is not None and matrix_b.shape[1] != self._mean_b.numel():
            raise ValueError("CrossCovariance feature dimensions cannot change across updates.")
        rows = matrix_a.shape[0]
        if rows == 0:
            return
        if self._mean_a is None or self._mean_b is None:
            self._mean_a = torch.zeros(matrix_a.shape[1], dtype=self._dtype, device=self._device)
            self._mean_b = torch.zeros(matrix_b.shape[1], dtype=self._dtype, device=self._device)
            self._m2 = torch.zeros(
                (matrix_a.shape[1], matrix_b.shape[1]), dtype=self._dtype, device=self._device
            )
        assert self._m2 is not None
        # Same batch combine as Covariance.update: one (d_a, d_b) update per
        # BATCH instead of one fresh outer product per row.
        batch_mean_a = matrix_a.mean(dim=0)
        batch_mean_b = matrix_b.mean(dim=0)
        centered_a = matrix_a - batch_mean_a
        centered_b = matrix_b - batch_mean_b
        total = self._count + rows
        delta_a = batch_mean_a - self._mean_a
        delta_b = batch_mean_b - self._mean_b
        self._m2 += centered_a.T @ centered_b
        self._m2 += torch.outer(delta_a, delta_b) * (self._count * rows / total)
        self._mean_a += delta_a * (rows / total)
        self._mean_b += delta_b * (rows / total)
        self._count = total

    def result(self) -> torch.Tensor:
        """Return the finalized sample cross-covariance matrix.

        Returns
        -------
        torch.Tensor
            Cross-covariance with shape ``(d_a, d_b)``. Fewer than two rows
            produce a zero matrix of the established feature shape.
        """

        if self._m2 is None:
            return torch.empty((0, 0), dtype=self._dtype, device=self._device)
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)


class CKA:
    r"""Streaming linear centered kernel alignment accumulator.

    Linear CKA is

    .. math::

        \operatorname{CKA}(A, B) =
        \frac{\lVert C_{AB}\rVert_F^2}
        {\lVert C_{AA}\rVert_F\,\lVert C_{BB}\rVert_F}.

    Only feature-sized covariance terms are retained. If either input has
    zero variance, ``result()`` returns NaN because the alignment denominator
    is zero. This follows the linear CKA formulation of Kornblith et al. (2019).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        device:
            Compute/state device (F20 brainpipe D-25; e.g. ``"cuda"``).
            ``None`` keeps the historical CPU placement bit-identical.
        dtype:
            Floating compute dtype (default: historical ``float64``).
        """

        self.name = name
        self._device, self._dtype = _resolve_placement(device, dtype)
        self._cross = CrossCovariance(device=device, dtype=dtype)
        self._covariance_a = Covariance(device=device, dtype=dtype)
        self._covariance_b = Covariance(device=device, dtype=dtype)

    def update(self, a: Any, b: Any) -> None:
        """Update linear CKA from one paired batch.

        Parameters
        ----------
        a:
            First tensor-like batch, with rows as observations.
        b:
            Second tensor-like batch, with rows as observations.
        """

        matrix_a = _as_feature_matrix(a)
        matrix_b = _as_feature_matrix(b)
        self._cross.update(matrix_a, matrix_b)
        self._covariance_a.update(matrix_a)
        self._covariance_b.update(matrix_b)

    def result(self) -> float:
        """Return the finalized linear CKA value.

        Returns
        -------
        float
            Linear CKA, or NaN when either representation has zero variance.
        """

        cross = self._cross.result()
        covariance_a = self._covariance_a.result()
        covariance_b = self._covariance_b.result()
        numerator = torch.linalg.matrix_norm(cross, ord="fro").square()
        denominator = torch.linalg.matrix_norm(covariance_a, ord="fro") * torch.linalg.matrix_norm(
            covariance_b, ord="fro"
        )
        if denominator.item() == 0.0:
            return math.nan
        return float((numerator / denominator).item())


def cka(a: Any, b: Any, *, device: Any = None, dtype: Any = None) -> float:
    r"""Compute one-shot linear centered kernel alignment.

    Linear CKA is

    .. math::

        \operatorname{CKA}(A, B) =
        \frac{\lVert C_{AB}\rVert_F^2}
        {\lVert C_{AA}\rVert_F\,\lVert C_{BB}\rVert_F}.

    This is the linear CKA measure described by Kornblith et al. (2019).
    Inputs are treated as ``(n_observations, n_features)`` matrices; the
    default computes on CPU ``float64`` (bit-identical to the historical
    behavior). A zero-variance input produces NaN.

    Parameters
    ----------
    a:
        First tensor-like representation.
    b:
        Second tensor-like representation with the same row count.
    device:
        Compute device (F20 brainpipe D-25; e.g. ``"cuda"`` keeps the Gram
        work on-device). ``None`` keeps the historical CPU placement.
    dtype:
        Floating compute dtype (default: historical ``float64``).

    Returns
    -------
    float
        Linear CKA value, or NaN for a degenerate zero-variance input.
    """

    resolved_device, resolved_dtype = _resolve_placement(device, dtype)
    matrix_a = _as_feature_matrix(a, resolved_device, resolved_dtype)
    matrix_b = _as_feature_matrix(b, resolved_device, resolved_dtype)
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"CKA requires matched row counts; got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
        )

    centered_a = matrix_a - matrix_a.mean(dim=0, keepdim=True)
    centered_b = matrix_b - matrix_b.mean(dim=0, keepdim=True)
    gram_a = centered_a @ centered_a.T
    gram_b = centered_b @ centered_b.T
    denominator = torch.linalg.matrix_norm(gram_a, ord="fro") * torch.linalg.matrix_norm(
        gram_b, ord="fro"
    )
    if denominator.item() == 0.0:
        return math.nan
    numerator = torch.sum(gram_a * gram_b)
    return float((numerator / denominator).item())


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

    def _solve(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigensolve the running covariance with deterministic sign canonicalization.

        Each component row is flipped so its largest-magnitude entry is
        positive (first occurrence on exact ties, per ``torch.argmax``), so
        two identical fits produce IDENTICAL arrays — an eigensolver's sign
        choice is otherwise arbitrary (transforms memo P6).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(components, explained_variance)`` — ``(k, d)`` rows and the
            descending eigenvalues.
        """

        cov = self._covariance.result()
        if cov.numel() == 0:
            return (
                torch.empty((0, 0), dtype=torch.float64),
                torch.empty((0,), dtype=torch.float64),
            )
        values, vectors = torch.linalg.eigh(cov)
        order = torch.argsort(values, descending=True)[: self.n_components]
        components = vectors[:, order].T.contiguous()
        anchor = components.abs().argmax(dim=1, keepdim=True)
        flip = torch.where(torch.gather(components, 1, anchor) < 0, -1.0, 1.0)
        return components * flip, values[order]

    def result(self) -> dict[str, Any]:
        """Return components, variances, and the P6 additive fit facts.

        Returns
        -------
        dict[str, Any]
            ``components`` (sign-canonicalized) and ``explained_variance``
            as before, plus ``mean`` (the feature mean the fit centered
            on), ``n_samples``, ``n_features``, and ``digest`` (the
            ``sha256:`` content digest of the fitted arrays — transforms
            memo P6).
        """

        from ._fitted import fitted_digest

        components, explained = self._solve()
        mean = self._covariance._mean
        mean = torch.empty((0,), dtype=torch.float64) if mean is None else mean.detach().clone()
        return {
            "components": components,
            "explained_variance": explained,
            "mean": mean,
            "n_samples": self._covariance._count,
            "n_features": int(mean.numel()),
            "digest": fitted_digest(components, mean, explained),
        }

    def fitted(self, fit_scope: str = "unspecified") -> Any:
        """Freeze the fit into a persistable :class:`~torchlens.stats.FittedPCA`.

        Parameters
        ----------
        fit_scope:
            Recorded disclosure of what the fit consumed (fitting across
            held-out stimuli leaks analysis information; say so here).

        Returns
        -------
        FittedPCA
            The frozen payload ``tl.stats.save_fitted`` persists and
            ``tl.transforms.pca_apply`` consumes.

        Raises
        ------
        FittedArtifactError
            ``pca_fitted_unavailable`` when fewer than two rows were seen —
            there is no covariance to solve, and fabricating a basis would
            be a silent wrong number.
        """

        from ._fitted import FittedArtifactError, FittedPCA, fitted_digest

        if self._covariance._count < 2:
            raise FittedArtifactError(
                f"PCA saw {self._covariance._count} row(s); a fit needs at "
                "least two rows before fitted() has anything to freeze.",
                code="pca_fitted_unavailable",
                remedy="update() the estimator with the fitting batches first",
                n_samples=self._covariance._count,
            )
        components, explained = self._solve()
        # count >= 2 (checked above) implies the running mean exists.
        mean = cast(torch.Tensor, self._covariance._mean).detach().clone()
        return FittedPCA(
            components=components,
            mean=mean,
            explained_variance=explained,
            n_samples=self._covariance._count,
            n_features=int(mean.numel()),
            fit_scope=str(fit_scope),
            digest=fitted_digest(components, mean, explained),
        )
