"""One call: RSA, cross-validated ridge and linear CKA on one extraction.

``neuro.compare(source, target, analyses=("rsa", "ridge", "cka"))`` reads
each site's saved activations ONCE (it never re-runs the model) and scores
every site against one target with each requested analysis. All three
scores are DESCRIPTIVE: they say how alike the site and the target are on
these stimuli. Inference (p-values, model comparison, noise ceilings),
variance partitioning, voxel maps and stacked fits stay with rsatoolbox,
himalaya, scikit-learn and Net2Brain.

The ridge score's procedure, exactly: stimuli are split into ``n_folds``
folds by a seeded permutation; on each training fold the features and the
target are centred by the training means, the ridge penalty is chosen from
``ridge_alphas`` by the exact leave-one-out error of the training fold
(one penalty for all target columns, smallest mean squared error, first on
ties), and the held-out fold is predicted. The score is the mean over
target columns of the Pearson correlation between the out-of-fold
predictions and the target (columns with a constant target or prediction
are left out of the mean). Spellings are DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__tl_layer__ = "L5"

#: Analyses the one call knows, in report order.
ANALYSES = ("rsa", "ridge", "cka")

#: Default ridge penalties: absolute, applied to centred (unscaled) features.
DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = tuple(float(10.0**k) for k in range(-2, 9))

_RSA_METHODS = ("pearson", "spearman", "kendall")


@dataclass(frozen=True)
class Comparison:
    """Every requested analysis scored per site against one target.

    Attributes
    ----------
    sites:
        Site keys, in sweep order.
    analyses:
        Analyses run, in report order.
    scores:
        ``scores[analysis][site]`` point score.
    intervals:
        ``intervals[analysis][site]`` stimulus-bootstrap
        :class:`~torchlens.repgeom.ScoreInterval` when ``n_resamples`` was
        set; empty otherwise.
    settings:
        The settings that produced the scores (metric, methods, penalties,
        folds, seed, resamples, stimulus count, target shape).
    ridge_alphas_chosen:
        ``ridge_alphas_chosen[site]``: the penalty each fold chose.
    similarity_repository_ids:
        Catalogue identifier per analysis (``None`` when the catalogue has
        no exact match; ridge has none).
    """

    sites: tuple[str, ...]
    analyses: tuple[str, ...]
    scores: dict[str, dict[str, float]]
    intervals: dict[str, dict[str, Any]]
    settings: dict[str, Any]
    ridge_alphas_chosen: dict[str, list[float]] = field(default_factory=dict)
    similarity_repository_ids: dict[str, str | None] = field(default_factory=dict)

    def table(self) -> str:
        """Return a plain-text table: one row per site, one column per analysis.

        Returns
        -------
        str
            Scores, with ``[low, high]`` beside each when intervals exist.
        """

        width = max([len("site"), *(len(site) for site in self.sites)])
        cell = 26 if self.intervals else 10
        lines = ["site".ljust(width) + "".join(f"  {name:>{cell}}" for name in self.analyses)]
        for site in self.sites:
            cells = "".join(f"  {self._cell(name, site):>{cell}}" for name in self.analyses)
            lines.append(site.ljust(width) + cells)
        return "\n".join(lines)

    def _cell(self, analysis: str, site: str) -> str:
        """Format one score cell."""

        value = self.scores[analysis][site]
        text = "nan" if math.isnan(value) else f"{value:.4f}"
        interval = self.intervals.get(analysis, {}).get(site)
        if interval is None:
            return text
        return f"{text} [{interval.low:.4f}, {interval.high:.4f}]"

    def disagreement(self) -> Any:
        """Return where the analyses rank the sites differently.

        Returns
        -------
        torchlens.repgeom.MetricDisagreement
            Per-analysis ranks and the pairs of sites ordered oppositely
            (intervals included when present).
        """

        from ..repgeom import metric_disagreement

        source = {
            name: dict(self.intervals[name]) if name in self.intervals else dict(self.scores[name])
            for name in self.analyses
        }
        return metric_disagreement(source)

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as plain data.

        Returns
        -------
        dict[str, Any]
            Every field, intervals as dictionaries.
        """

        return {
            "sites": list(self.sites),
            "analyses": list(self.analyses),
            "scores": {name: dict(values) for name, values in self.scores.items()},
            "intervals": {
                name: {site: interval.to_dict() for site, interval in values.items()}
                for name, values in self.intervals.items()
            },
            "settings": dict(self.settings),
            "ridge_alphas_chosen": {site: list(v) for site, v in self.ridge_alphas_chosen.items()},
            "similarity_repository_ids": dict(self.similarity_repository_ids),
        }


def _pearson_columns(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Pearson r per column (NaN where either column is constant)."""

    centred_p = predicted - predicted.mean(axis=0)
    centred_t = target - target.mean(axis=0)
    ss_p, ss_t = np.sum(centred_p**2, axis=0), np.sum(centred_t**2, axis=0)
    # Rounding residue of a constant column is not variance (relative 1e-24
    # on the squared scale, as tl.repgeom's bootstrap correlations use).
    constant = (ss_p <= 1e-24 * np.sum(predicted**2, axis=0)) | (
        ss_t <= 1e-24 * np.sum(target**2, axis=0)
    )
    safe = np.where(constant, 1.0, np.sqrt(ss_p * ss_t))
    return np.where(constant, np.nan, np.sum(centred_p * centred_t, axis=0) / safe)


def _mean_column_r(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return the mean Pearson r over target columns, constant columns left out."""

    values = _pearson_columns(predicted, target)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def loo_errors(features: np.ndarray, target: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """Return the exact leave-one-out mean squared error for each penalty.

    ``features``/``target`` are one training fold, uncentred. With an
    unpenalised intercept the ridge hat matrix is
    ``H = U diag(s^2 / (s^2 + alpha)) U^T + 11^T / n`` over the centred
    features, and the leave-one-out residual is ``(y - H y) / (1 - h_ii)``;
    the error is the mean over rows and target columns of its square.

    Parameters
    ----------
    features, target:
        One training fold.
    alphas:
        Penalties.

    Returns
    -------
    np.ndarray
        One error per penalty (``inf``/NaN where a leverage reaches 1).
    """

    n = features.shape[0]
    centred_x = features - features.mean(axis=0)
    centred_y = target - target.mean(axis=0)
    u, s, _vt = np.linalg.svd(centred_x, full_matrices=False)
    projected = u.T @ centred_y
    errors = []
    for alpha in alphas:
        shrink = s**2 / (s**2 + alpha)
        fitted = u @ (shrink[:, None] * projected)
        leverage = (u**2) @ shrink + 1.0 / n
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = (centred_y - fitted) / (1.0 - leverage)[:, None]
            errors.append(float(np.mean(residual**2)))
    return np.asarray(errors)


def _loo_alpha(features: np.ndarray, target: np.ndarray, alphas: Sequence[float]) -> float:
    """Return the penalty with the smallest exact leave-one-out error (first on ties).

    Raises
    ------
    ValueError
        If no penalty gives a finite leave-one-out error.
    """

    errors = loo_errors(features, target, alphas)
    if not np.any(np.isfinite(errors)):
        raise ValueError(
            "neuro.compare ridge: no penalty in ridge_alphas gives a finite leave-one-out "
            "error on a training fold; use larger penalties."
        )
    return float(alphas[int(np.nanargmin(np.where(np.isfinite(errors), errors, np.nan)))])


def _ridge_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float
) -> np.ndarray:
    """Fit ridge with an unpenalised intercept and predict ``test_x``."""

    mean_x, mean_y = train_x.mean(axis=0), train_y.mean(axis=0)
    u, s, vt = np.linalg.svd(train_x - mean_x, full_matrices=False)
    weights = vt.T @ ((s / (s**2 + alpha))[:, None] * (u.T @ (train_y - mean_y)))
    return (test_x - mean_x) @ weights + mean_y


def ridge_folds(n_stimuli: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return the held-out index arrays of the seeded k-fold split.

    Parameters
    ----------
    n_stimuli, n_folds, seed:
        Stimulus count, fold count, permutation seed.

    Returns
    -------
    list[np.ndarray]
        One sorted index array per fold.
    """

    order = np.random.default_rng(int(seed)).permutation(n_stimuli)
    return [np.sort(chunk) for chunk in np.array_split(order, n_folds)]


def cross_validated_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alphas: Sequence[float],
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Return out-of-fold ridge predictions and the penalty each fold chose.

    Parameters
    ----------
    features:
        ``[n_stimuli, n_features]`` float64 design matrix.
    target:
        ``[n_stimuli, n_columns]`` float64 target.
    alphas, n_folds, seed:
        Penalty grid, fold count, split seed (see the module docstring).

    Returns
    -------
    tuple[np.ndarray, list[float]]
        Predictions for every stimulus from the fold that held it out, and
        the chosen penalty per fold.
    """

    predictions = np.empty_like(target)
    chosen: list[float] = []
    for held_out in ridge_folds(features.shape[0], n_folds, seed):
        train = np.setdiff1d(np.arange(features.shape[0]), held_out)
        alpha = _loo_alpha(features[train], target[train], alphas)
        chosen.append(alpha)
        predictions[held_out] = _ridge_fit_predict(
            features[train], target[train], features[held_out], alpha
        )
    return predictions, chosen


def _as_rows(value: Any, name: str) -> np.ndarray:
    """Return a finite ``[n, features]`` float64 matrix."""

    from ..repgeom._geometry import _as_numpy_array, _validate_finite

    array = _as_numpy_array(value)
    if array.ndim < 1:
        raise ValueError(f"neuro.compare {name} needs a stimulus axis; got a scalar.")
    matrix = array.reshape(array.shape[0], -1)
    _validate_finite(matrix, f"neuro.compare {name}")
    return matrix


def _mapping_matrices(source: Mapping[Any, Any], sites: Any) -> OrderedDict[str, np.ndarray]:
    """Read the requested keys of a ``{site: array}`` mapping once each."""

    selected = list(source) if sites is None else list(sites)
    missing = [site for site in selected if site not in source]
    if missing:
        raise ValueError(f"neuro.compare: sites {missing} are not keys of the source mapping.")
    return OrderedDict((str(site), _as_rows(source[site], f"site {site!r}")) for site in selected)


def _site_matrices(source: Any, sites: Any) -> OrderedDict[str, np.ndarray]:
    """Read each site's activations once, keyed by site."""

    if isinstance(source, Mapping):
        return _mapping_matrices(source, sites)
    from ._handoff import _source_kind, enumerate_extraction_sites, enumerate_trace_sites

    if _source_kind(source) == "trace":
        payloads, _ledger = enumerate_trace_sites(source, sites, verb="neuro.compare")
    else:
        payloads, _ledger, _loaded = enumerate_extraction_sites(source, sites, verb="neuro.compare")
    return OrderedDict(
        (payload.key, _as_rows(payload.matrix, f"site {payload.key!r}")) for payload in payloads
    )


def _check_settings(
    analyses: Sequence[str], rsa_method: str, alphas: Sequence[float], n_folds: int
) -> None:
    """Validate the analysis list, RSA method, penalties and fold count."""

    unknown = [name for name in analyses if name not in ANALYSES]
    if unknown or not analyses:
        raise ValueError(
            f"neuro.compare analyses must be a non-empty subset of {ANALYSES}; got "
            f"{tuple(analyses)!r}."
        )
    if rsa_method not in _RSA_METHODS:
        raise ValueError(
            f"neuro.compare rsa_method must be one of {_RSA_METHODS}; got {rsa_method!r}. "
            "RDM comparison beyond these (whitened cosine, rho-a, ...) is "
            "rsatoolbox.rdm.compare's vocabulary; feed it RDMs from neuro.rdms()."
        )
    if not alphas or any(not (math.isfinite(a) and a > 0) for a in alphas):
        raise ValueError(f"neuro.compare ridge_alphas must be positive and finite; got {alphas!r}.")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"neuro.compare n_folds must be an integer >= 2; got {n_folds!r}.")


def _analysis_runners(
    target: np.ndarray, settings: Mapping[str, Any], bootstrap: Any
) -> dict[str, Callable[[np.ndarray], tuple[Any, list[float] | None]]]:
    """Build one scorer per analysis over a site matrix (target work done once)."""

    from ..repgeom import rdm
    from ..repgeom._geometry import rdm_compare
    from ..repgeom._uncertainty import (
        _gram_cka,
        bootstrap_interval,
        rdm_compare_interval,
    )

    target_rdm = rdm(target, settings["rdm_metric"])
    target_gram = target @ target.T

    def rsa(matrix: np.ndarray) -> tuple[Any, None]:
        """Score one site by RSA against the target RDM, with its interval if asked."""

        site_rdm = rdm(matrix, settings["rdm_metric"])
        if bootstrap is None:
            return rdm_compare(site_rdm, target_rdm, method=settings["rsa_method"]), None
        return rdm_compare_interval(
            site_rdm,
            target_rdm,
            settings["rsa_method"],
            n_resamples=bootstrap.n_resamples,
            seed=bootstrap.seed,
            confidence=bootstrap.confidence,
        ), None

    def cka(matrix: np.ndarray) -> tuple[Any, None]:
        """Score one site by linear CKA against the target, with its interval if asked."""

        gram = matrix @ matrix.T
        point = _gram_cka(gram, target_gram)
        if bootstrap is None:
            return point, None

        def statistic(index: np.ndarray) -> float:
            """Return linear CKA on the resampled stimuli ``index``."""

            sub = np.ix_(index, index)
            return _gram_cka(gram[sub], target_gram[sub])

        return bootstrap_interval(statistic, matrix.shape[0], point, bootstrap), None

    def ridge(matrix: np.ndarray) -> tuple[Any, list[float]]:
        """Score one site by cross-validated ridge; also return each fold's penalty."""

        predicted, chosen = cross_validated_ridge(
            matrix,
            target,
            alphas=settings["ridge_alphas"],
            n_folds=settings["n_folds"],
            seed=settings["seed"],
        )
        point = _mean_column_r(predicted, target)
        if bootstrap is None:
            return point, chosen

        def statistic(index: np.ndarray) -> float:
            """Return the mean per-column r on the resampled stimuli ``index``."""

            return _mean_column_r(predicted[index], target[index])

        return bootstrap_interval(statistic, matrix.shape[0], point, bootstrap), chosen

    return {"rsa": rsa, "ridge": ridge, "cka": cka}


def _repository_ids(
    analyses: Sequence[str], rdm_metric: str, rsa_method: str
) -> dict[str, str | None]:
    """Return the similarity-repository identifier per analysis."""

    from ..repgeom._similarity_ids import LINEAR_CKA_ID, RSA_IDS

    known = {"rsa": RSA_IDS.get((rdm_metric, rsa_method)), "cka": LINEAR_CKA_ID, "ridge": None}
    return {name: known[name] for name in analyses}


def compare(  # noqa: PLR0913 -- the one-call door: source, target, the analysis list, and each analysis's documented knob
    source: Any,
    target: Any,
    *,
    analyses: Sequence[str] = ANALYSES,
    sites: Any = None,
    rdm_metric: str = "correlation",
    rsa_method: str = "spearman",
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    num_folds: int = 5,
    n_resamples: int | None = None,
    seed: int = 0,
    confidence: float = 0.95,
) -> Comparison:
    """Score every site against one target with RSA, ridge and CKA in one call.

    DOCUMENTED-UNSTABLE. Each site's activations are read
    once from what ``source`` already holds and reused by every analysis;
    the model is never re-run. All scores are descriptive similarities on
    these stimuli, not tests.

    - ``"rsa"``: ``tl.repgeom.rdm_compare(rdm(site, rdm_metric),
      rdm(target, rdm_metric), rsa_method)``.
    - ``"ridge"``: the cross-validated ridge score of the module docstring
      (seeded ``n_folds`` split, penalty per fold by exact leave-one-out on
      the training fold, mean Pearson r over target columns of the
      out-of-fold predictions).
    - ``"cka"``: linear CKA with the biased Gretton HSIC (the variant
      ``tl.stats.cka`` computes), through stimulus Gram matrices.

    Variance partitioning, voxel maps, stacked fits and inference stay
    with himalaya, scikit-learn, Net2Brain and rsatoolbox.

    Parameters
    ----------
    source:
        A Trace with saved activations, a ``LoadedExtraction`` or
        extraction directory, or a mapping ``{site: [n_stimuli, ...]}``.
    target:
        ``[n_stimuli, ...]`` responses (brain recordings, another model's
        features) for the same stimuli in the same order.
    analyses:
        Subset of ``("rsa", "ridge", "cka")``, in the order to report.
    sites:
        Sites to score (default: every stimulus-indexed saved site, or every
        mapping key).
    rdm_metric:
        RDM metric for RSA (``tl.repgeom.rdm``'s names).
    rsa_method:
        ``"spearman"``, ``"pearson"`` or ``"kendall"`` (tau-a).
    ridge_alphas:
        Penalty grid (absolute, on centred features).
    n_folds:
        Cross-validation folds for ridge (each training fold needs at least
        3 stimuli).
    n_resamples:
        When set, every score gets a stimulus-bootstrap
        :class:`~torchlens.repgeom.ScoreInterval` (the ridge interval
        resamples the evaluation of the out-of-fold predictions; it does
        not refit).
    seed:
        Seed for the fold split and the bootstrap.
    confidence:
        Nominal coverage of the intervals.

    Returns
    -------
    Comparison
        Scores, intervals, settings, chosen penalties, ``table()``,
        ``disagreement()`` and ``to_dict()``.

    Raises
    ------
    ValueError
        On an unknown analysis or method, invalid settings, a site whose
        stimulus count differs from the target's, or too few stimuli.
    """

    from ..repgeom._uncertainty import BootstrapSettings

    analyses = tuple(analyses)
    alphas = tuple(float(a) for a in ridge_alphas)
    _check_settings(analyses, rsa_method, alphas, num_folds)
    bootstrap = (
        None
        if n_resamples is None
        else BootstrapSettings(n_resamples=n_resamples, seed=seed, confidence=confidence)
    )
    target_matrix = _as_rows(target, "target")
    n_stimuli = target_matrix.shape[0]
    if n_stimuli < max(3, num_folds + 2):
        raise ValueError(
            f"neuro.compare needs at least max(3, n_folds + 2) = {max(3, num_folds + 2)} "
            f"stimuli; the target has {n_stimuli}."
        )
    matrices = _site_matrices(source, sites)
    for key, matrix in matrices.items():
        if matrix.shape[0] != n_stimuli:
            raise ValueError(
                f"neuro.compare site {key!r} has {matrix.shape[0]} stimuli; the target has "
                f"{n_stimuli}. Both must cover the same stimuli in the same order."
            )
    settings: dict[str, Any] = {
        "rdm_metric": rdm_metric,
        "rsa_method": rsa_method,
        "ridge_alphas": alphas,
        "n_folds": num_folds,
        "seed": int(seed),
        "n_resamples": n_resamples,
        "confidence": float(confidence),
        "n_stimuli": n_stimuli,
        "target_shape": tuple(target_matrix.shape),
        "basis": "descriptive similarity on these stimuli; not an inferential test",
    }
    runners = _analysis_runners(target_matrix, settings, bootstrap)
    return _run(matrices, analyses, runners, settings)


def _run(
    matrices: Mapping[str, np.ndarray],
    analyses: tuple[str, ...],
    runners: Mapping[str, Callable[[np.ndarray], tuple[Any, list[float] | None]]],
    settings: dict[str, Any],
) -> Comparison:
    """Run every analysis on every site matrix and assemble the report."""

    from ..repgeom import ScoreInterval

    scores: dict[str, dict[str, float]] = {name: {} for name in analyses}
    intervals: dict[str, dict[str, Any]] = {}
    chosen: dict[str, list[float]] = {}
    for key, matrix in matrices.items():
        for name in analyses:
            value, alphas = runners[name](matrix)
            if isinstance(value, ScoreInterval):
                intervals.setdefault(name, {})[key] = value
                value = value.score
            scores[name][key] = float(value)
            if alphas is not None:
                chosen[key] = alphas
    return Comparison(
        sites=tuple(matrices),
        analyses=analyses,
        scores=scores,
        intervals=intervals,
        settings=settings,
        ridge_alphas_chosen=chosen,
        similarity_repository_ids=_repository_ids(
            analyses, settings["rdm_metric"], settings["rsa_method"]
        ),
    )
