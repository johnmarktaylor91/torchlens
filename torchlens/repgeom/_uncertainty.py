"""Bootstrap uncertainty for TorchLens's own representational scores.

Every interval here is a percentile bootstrap over STIMULI: the stimulus
set is resampled with replacement, the score is recomputed on each
resample, and the interval is read off the resampled scores. It describes
how much TorchLens's descriptive score moves when the stimuli change. It
is not an inferential test: no p-values, no subject or noise-ceiling
inference (rsatoolbox.inference owns those), and no claim of parity with
rsatoolbox's corrected bootstraps. The point score is always the
unchanged estimator on all stimuli.

RDM resamples drop the entries that pair a stimulus with a copy of itself
(the duplicated-diagonal artefact a resampled RDM would otherwise carry),
as rsatoolbox's stimulus bootstrap does. CKA resamples reindex the two
stimulus Gram matrices, which equals recomputing linear CKA on the
resampled rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._geometry import _as_numpy_array, _kendall_tau_a, _validate_finite

__tl_layer__ = "L5"

#: Basis string recorded on every stimulus-bootstrap interval.
STIMULUS_BOOTSTRAP_BASIS = "bootstrap over stimuli, percentile interval"

_COMPARE_METHODS = ("pearson", "spearman", "kendall")


@dataclass(frozen=True)
class ScoreInterval:
    """A point score with its stimulus-bootstrap interval beside it.

    Attributes
    ----------
    score:
        The point score on all stimuli (the unchanged estimator).
    low, high:
        Percentile interval bounds over the resampled scores.
    confidence:
        Nominal coverage of ``[low, high]`` (for example ``0.95``).
    n_resamples:
        Resamples drawn.
    n_valid:
        Resamples whose score was finite; the interval uses only these.
    seed:
        Seed of the ``numpy.random.default_rng`` that drew the resamples.
    basis:
        How the interval was made, in words.
    """

    score: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    n_valid: int
    seed: int
    basis: str = STIMULUS_BOOTSTRAP_BASIS

    @property
    def width(self) -> float:
        """Return ``high - low``."""

        return self.high - self.low

    def to_dict(self) -> dict[str, Any]:
        """Return the interval as a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Every field plus ``width``.
        """

        return {
            "score": self.score,
            "low": self.low,
            "high": self.high,
            "width": self.width,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "n_valid": self.n_valid,
            "seed": self.seed,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class BootstrapSettings:
    """Validated stimulus-bootstrap settings shared by every interval.

    Attributes
    ----------
    n_resamples:
        Resamples to draw (at least 2).
    seed:
        Seed for ``numpy.random.default_rng``.
    confidence:
        Nominal coverage, strictly between 0 and 1.

    Raises
    ------
    ValueError
        On an invalid setting.
    """

    n_resamples: int = 1000
    seed: int = 0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate the settings."""

        n_resamples, seed = self.n_resamples, self.seed
        if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
            raise ValueError(f"n_resamples must be an integer >= 2; got {n_resamples!r}.")
        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError(
                f"confidence must lie strictly between 0 and 1; got {self.confidence!r}."
            )
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer; got {seed!r}.")


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    n_stimuli: int,
    score: float,
    settings: BootstrapSettings,
) -> ScoreInterval:
    """Resample stimulus indices and summarise ``statistic`` as an interval.

    Parameters
    ----------
    statistic:
        Function of a resampled index vector (length ``n_stimuli``, drawn
        with replacement) returning the score on that resample.
    n_stimuli:
        Number of stimuli.
    score:
        The point score on all stimuli, recorded unchanged.
    settings:
        Resample count, seed and confidence.

    Returns
    -------
    ScoreInterval
        The interval; ``low``/``high`` are NaN when fewer than half the
        resamples gave a finite score.
    """

    rng = np.random.default_rng(int(settings.seed))
    indices = rng.integers(0, n_stimuli, size=(settings.n_resamples, n_stimuli))
    values = np.array([statistic(row) for row in indices], dtype=np.float64)
    finite = values[np.isfinite(values)]
    tail = (1.0 - float(settings.confidence)) / 2.0
    # A resample with fewer than two distinct stimuli is constant; its
    # statistic is NaN and it is counted out of n_valid.
    if finite.size * 2 < settings.n_resamples:
        low = high = math.nan
    else:
        low, high = (float(v) for v in np.quantile(finite, [tail, 1.0 - tail]))
    return ScoreInterval(
        score=float(score),
        low=low,
        high=high,
        confidence=float(settings.confidence),
        n_resamples=int(settings.n_resamples),
        n_valid=int(finite.size),
        seed=int(settings.seed),
    )


def _average_ranks_vector(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, vectorised (equal to ``_average_ranks``)."""

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts_mask = np.concatenate(([True], ordered[1:] != ordered[:-1]))
    starts = np.flatnonzero(starts_mask)
    ends = np.append(starts[1:], values.size) - 1
    group = np.cumsum(starts_mask) - 1
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ((starts + ends) / 2.0 + 1.0)[group]
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson r with an exact 1.0 for identical inputs.

    The denominator is ``sqrt(ss_a * ss_b)`` rather than a product of norms,
    so ``a == b`` gives ``s / sqrt(s * s) == 1.0`` exactly.
    """

    centered_a = a - a.mean()
    centered_b = b - b.mean()
    ss_a = float(np.dot(centered_a, centered_a))
    ss_b = float(np.dot(centered_b, centered_b))
    if _is_constant(ss_a, a) or _is_constant(ss_b, b):
        return math.nan
    return float(np.dot(centered_a, centered_b) / math.sqrt(ss_a * ss_b))


#: Relative size, on squared scale, below which a centred vector or Gram
#: counts as constant: mean subtraction on a constant input leaves
#: rounding residue of about machine epsilon times its magnitude, and a
#: correlation of that residue is noise, not a score.
_CONSTANT_RELATIVE_TOLERANCE = 1e-24


def _is_constant(centred_sum_of_squares: float, raw: np.ndarray) -> bool:
    """Return whether a centred sum of squares is rounding residue of ``raw``."""

    scale = float(np.sum(np.square(raw)))
    return centred_sum_of_squares <= _CONSTANT_RELATIVE_TOLERANCE * scale


def _weighted_tau_a(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """Return Kendall tau-a over distinct entries carrying multiplicities.

    A resample repeats some stimulus pairs; copies of one pair are the same
    observation, so they are never compared with each other (comparing them
    would add artificial ties that pull tau-a below 1 even for identical
    RDMs). Every other comparison counts ``w_p * w_q`` times. With all
    weights 1 this is plain tau-a.
    """

    sign_a = np.sign(a[:, None] - a[None, :])
    sign_b = np.sign(b[:, None] - b[None, :])
    upper = np.triu_indices(a.size, k=1)
    pair_weights = (weights[:, None] * weights[None, :])[upper]
    total = float(np.sum(pair_weights))
    if total == 0.0:
        return math.nan
    return float(np.sum(pair_weights * sign_a[upper] * sign_b[upper])) / total


def _vector_compare(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Compare two aligned dissimilarity vectors with ``method``."""

    if a.size < 2:
        return math.nan
    if method == "spearman":
        return _pearson(_average_ranks_vector(a), _average_ranks_vector(b))
    if method == "kendall":
        return _kendall_tau_a(a, b)
    return _pearson(a, b)


def _square_rdm(value: Any, name: str) -> np.ndarray:
    """Return a finite square float64 RDM (condensed input is expanded)."""

    array = _as_numpy_array(value).astype(np.float64)
    _validate_finite(array, name)
    if array.ndim == 1:
        n = int(round((1.0 + math.sqrt(1.0 + 8.0 * array.size)) / 2.0))
        if n * (n - 1) // 2 != array.size:
            raise ValueError(
                f"{name} is a 1-D vector of length {array.size}, which is not a condensed "
                "upper triangle of any square RDM."
            )
        square = np.zeros((n, n), dtype=np.float64)
        upper = np.triu_indices(n, k=1)
        square[upper] = array
        square[(upper[1], upper[0])] = array
        return square
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} must be a square RDM or a condensed upper-triangle vector; "
            f"got shape {array.shape}."
        )
    return array


def rdm_compare_interval(  # noqa: PLR0913 -- the spec'd row-1 door: two RDMs, the rdm_compare method, and the three bootstrap knobs, documented one-to-one
    rdm_a: Any,
    rdm_b: Any,
    method: str = "spearman",
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ScoreInterval:
    """Return :func:`rdm_compare` with a stimulus-bootstrap interval beside it.

    DOCUMENTED-UNSTABLE. The point ``score`` is exactly
    ``tl.repgeom.rdm_compare(rdm_a, rdm_b, method)``. Each resample draws
    stimuli with replacement, takes the strict upper triangle of both
    resampled RDMs, drops the entries that pair a stimulus with a copy of
    itself, and recomputes the comparison. The interval describes the
    score's sensitivity to the stimulus sample; it is not an inferential
    test (rsatoolbox.inference owns those).

    Parameters
    ----------
    rdm_a, rdm_b:
        Square RDMs (or condensed upper triangles) over the same stimuli in
        the same order.
    method:
        ``"spearman"`` (default), ``"pearson"``, or ``"kendall"`` (tau-a;
        quadratic in the number of pairs per resample). For kendall, the
        copies of one stimulus pair that a resample repeats are never
        compared with each other (they are one observation); every other
        comparison is weighted by the copies' multiplicities.
    n_resamples:
        Bootstrap resamples (at least 2).
    seed:
        Seed for ``numpy.random.default_rng``; the same seed gives the same
        interval.
    confidence:
        Nominal coverage of the percentile interval.

    Returns
    -------
    ScoreInterval
        Point score, interval, and the settings that made it.

    Raises
    ------
    ValueError
        On RDMs of different sizes, fewer than 3 stimuli, an unknown
        method, or invalid bootstrap settings.
    """

    from ._geometry import rdm_compare

    settings = BootstrapSettings(n_resamples=n_resamples, seed=seed, confidence=confidence)
    square_a = _square_rdm(rdm_a, "rdm_a")
    square_b = _square_rdm(rdm_b, "rdm_b")
    if square_a.shape != square_b.shape:
        raise ValueError(
            f"rdm_compare_interval requires RDMs over the same stimuli; got shapes "
            f"{square_a.shape} and {square_b.shape}."
        )
    if method not in _COMPARE_METHODS:
        raise ValueError(
            f"Unknown rdm_compare_interval method: {method!r} (use 'pearson', 'spearman', "
            "or 'kendall')."
        )
    score = rdm_compare(square_a, square_b, method=method)
    n = square_a.shape[0]
    rows, cols = np.triu_indices(n, k=1)

    def statistic(index: np.ndarray) -> float:
        """Compare the resampled upper triangles, self-pairs dropped."""

        left = index[rows]
        right = index[cols]
        keep = left != right
        left, right = left[keep], right[keep]
        if method == "kendall":
            codes, weights = np.unique(
                np.minimum(left, right) * n + np.maximum(left, right), return_counts=True
            )
            first, second = np.divmod(codes, n)
            return _weighted_tau_a(
                square_a[first, second], square_b[first, second], weights.astype(np.float64)
            )
        return _vector_compare(square_a[left, right], square_b[left, right], method)

    return bootstrap_interval(statistic, n, score, settings)


def _gram_cka(gram_a: np.ndarray, gram_b: np.ndarray) -> float:
    """Return linear CKA from two (uncentred-row) stimulus Gram matrices.

    The Grams are double-centred here, so a reindexed (resampled) Gram is
    re-centred over the resample, which equals linear CKA on the
    resampled, re-centred rows.
    """

    centered_a = gram_a - gram_a.mean(axis=0, keepdims=True)
    centered_a = centered_a - centered_a.mean(axis=1, keepdims=True)
    centered_b = gram_b - gram_b.mean(axis=0, keepdims=True)
    centered_b = centered_b - centered_b.mean(axis=1, keepdims=True)
    cross = float(np.sum(centered_a * centered_b))
    self_a = float(np.sum(centered_a * centered_a))
    self_b = float(np.sum(centered_b * centered_b))
    if _is_constant(self_a, gram_a) or _is_constant(self_b, gram_b):
        return math.nan
    return cross / math.sqrt(self_a * self_b)


def _as_matrix(value: Any, name: str) -> np.ndarray:
    """Return a finite ``[n_stimuli, n_features]`` float64 matrix."""

    array = _as_numpy_array(value)
    _validate_finite(array, name)
    if array.ndim < 1:
        raise ValueError(f"{name} needs a stimulus axis; got a scalar.")
    return array.reshape(array.shape[0], -1)


def cka_interval(
    a: Any,
    b: Any,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ScoreInterval:
    """Return linear CKA with a stimulus-bootstrap interval beside it.

    DOCUMENTED-UNSTABLE. The point ``score`` is linear CKA with
    the biased Gretton HSIC, the variant ``tl.stats.cka`` computes
    (similarity-repository ``measure/*/cka-kernel=linear-hsic=gretton-score``),
    evaluated through stimulus Gram matrices so each resample costs
    ``O(n^2)`` whatever the feature count. Each resample draws stimuli
    with replacement and recomputes CKA on the resampled rows. The
    interval describes the score's sensitivity to the stimulus sample; it
    is not a permutation or subject test. The biased HSIC estimator rises
    when a resample repeats rows (fewer distinct stimuli), so at small
    stimulus counts the interval can sit above the point score.

    Parameters
    ----------
    a, b:
        Representations with the same first (stimulus) axis; trailing axes
        are flattened.
    n_resamples, seed, confidence:
        Bootstrap settings (see :func:`rdm_compare_interval`).

    Returns
    -------
    ScoreInterval
        Point score, interval, and the settings that made it.

    Raises
    ------
    ValueError
        On mismatched stimulus counts, fewer than 3 stimuli, non-finite
        input, or invalid bootstrap settings.
    """

    settings = BootstrapSettings(n_resamples=n_resamples, seed=seed, confidence=confidence)
    matrix_a = _as_matrix(a, "a")
    matrix_b = _as_matrix(b, "b")
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"cka_interval requires the same stimuli in both inputs; got "
            f"{matrix_a.shape[0]} and {matrix_b.shape[0]} rows."
        )
    n = matrix_a.shape[0]
    if n < 3:
        raise ValueError(f"cka_interval needs at least 3 stimuli; got {n}.")
    gram_a = matrix_a @ matrix_a.T
    gram_b = matrix_b @ matrix_b.T

    def statistic(index: np.ndarray) -> float:
        """Recompute CKA on the resampled stimuli."""

        return _gram_cka(gram_a[np.ix_(index, index)], gram_b[np.ix_(index, index)])

    return bootstrap_interval(statistic, n, _gram_cka(gram_a, gram_b), settings)
