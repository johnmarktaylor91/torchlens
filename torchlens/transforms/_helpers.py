"""SRP sizing + fidelity helpers (memo B7): honest advisors, never transforms.

``srp_dims_for`` is the DENSE-JL L2 sizing heuristic — the sklearn
``johnson_lindenstrauss_min_dim`` practice with its scope relabeled: it is a
poor guide for correlation-distance RDMs, and it says so on the returned
record. ``srp_fidelity_probe`` measures L2-RDM and corr-RDM fidelity per
candidate width on the USER'S OWN pilot batch, because the measured spread
BETWEEN corpora (0.28 vs 0.91 at the same site and k) is larger than the
spread across k — the only honest advisor is a probe on your own stimuli.
The probe's report is a storable, provenance-bearing object carrying the
T-C12 scope fields, so pasted numbers come with their scope for free.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._context import TransformContext
from ._errors import TransformContractError
from ._spec import canonical_json
from ._srp import srp
from ._srp_hash import ALGORITHM_VERSION

__tl_layer__ = "L4"

__all__ = ["ProbeReport", "SrpSizing", "srp_dims_for", "srp_fidelity_probe"]


@dataclass(frozen=True)
class SrpSizing:
    """An explicit sized width with its formula, version, and assumptions.

    Converts to ``int`` (feeds ``srp(n_components=...)`` directly); the
    record fields make an ``"auto"`` spelling lower to an explicit recorded
    k instead of an unexplained number.

    Attributes
    ----------
    n_components:
        The sized output width.
    formula:
        Formula id (``"dense_jl_l2_heuristic"``).
    version:
        Formula version.
    n_samples:
        Row count the bound was sized for.
    eps:
        Requested maximum pairwise-L2 distortion.
    note:
        The scope label: a DENSE-JL Euclidean heuristic, a poor guide for
        correlation-distance RDMs.
    """

    n_components: int
    formula: str
    version: int
    n_samples: int
    eps: float
    note: str

    def __int__(self) -> int:
        """Return the sized width."""

        return self.n_components

    def __index__(self) -> int:
        """Index conversion so the sizing feeds ``srp()`` directly."""

        return self.n_components

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-portable sizing record.

        Returns
        -------
        dict[str, Any]
            All sizing fields, canonical-JSON-portable.
        """

        return {
            "n_components": self.n_components,
            "formula": self.formula,
            "version": self.version,
            "n_samples": self.n_samples,
            "eps": self.eps,
            "note": self.note,
        }


def srp_dims_for(n_samples: int, eps: float = 0.1) -> SrpSizing:
    """Size an SRP width with the dense-JL L2 heuristic (honestly labeled).

    ``k = ceil(4 ln(n) / (eps^2/2 - eps^3/3))`` — the classical dense-JL
    bound the sklearn ``johnson_lindenstrauss_min_dim`` helper ships. The
    implemented sparse constructions record ``empirical_only``, so this is
    a HEURISTIC here, and it is L2-scoped: run ``srp_fidelity_probe`` on
    your own pilot batch before committing a harvest whose analysis is a
    correlation-distance RDM.

    Parameters
    ----------
    n_samples:
        Number of rows the projection must preserve pairwise (``>= 2``).
    eps:
        Maximum pairwise-L2 distortion target, in ``(0, 1)``.

    Returns
    -------
    SrpSizing
        The sized width with formula/version/assumptions recorded.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise TransformContractError(
            f"srp_dims_for(n_samples={n_samples!r}) needs an int >= 2 (the bound "
            "is about preserving pairwise distances).",
            code="transform_params_invalid",
            remedy="pass the row count of the dataset being projected",
            n_samples=n_samples,
        )
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not (0.0 < eps < 1.0):
        raise TransformContractError(
            f"srp_dims_for(eps={eps!r}) needs a distortion target in (0, 1).",
            code="transform_params_invalid",
            remedy="pass eps in (0, 1), e.g. 0.1",
            eps=eps,
        )
    eps = float(eps)
    denominator = eps**2 / 2.0 - eps**3 / 3.0
    k = math.ceil(4.0 * math.log(n_samples) / denominator)
    return SrpSizing(
        n_components=k,
        formula="dense_jl_l2_heuristic",
        version=1,
        n_samples=n_samples,
        eps=eps,
        note=(
            "dense-JL Euclidean heuristic (sklearn johnson_lindenstrauss_min_dim "
            "practice); the implemented sparse constructions record "
            "empirical_only, and this bound is a poor guide for "
            "correlation-distance RDMs -- probe your own pilot batch"
        ),
    )


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties averaged) of a 1-D float64 tensor.

    Parameters
    ----------
    values:
        1-D tensor.

    Returns
    -------
    torch.Tensor
        float64 ranks, 1-based, ties averaged.
    """

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    counts = counts.double()
    first = torch.cumsum(counts, 0) - counts + 1.0
    average = first + (counts - 1.0) / 2.0
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = average[inverse]
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors in float64."""

    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator == 0.0:
        return float("nan")
    return float((a @ b) / denominator)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-D tensors (float64, ties averaged).

    Parameters
    ----------
    a:
        First value vector.
    b:
        Second value vector (same length).

    Returns
    -------
    float
        Spearman rho.
    """

    return _pearson(_rankdata(a.double()), _rankdata(b.double()))


def _condensed_l2_rdm(features: torch.Tensor) -> torch.Tensor:
    """Condensed (upper-triangle) pairwise-L2 RDM of an (n, d) matrix."""

    distances = torch.cdist(features.double(), features.double())
    n = features.shape[0]
    rows, cols = torch.triu_indices(n, n, offset=1)
    return distances[rows, cols]


def _condensed_corr_rdm(features: torch.Tensor, *, context: str) -> torch.Tensor:
    """Condensed correlation-distance RDM of an (n, d) matrix.

    Parameters
    ----------
    features:
        (n, d) feature matrix.
    context:
        Caller name for refusal text.

    Raises
    ------
    TransformContractError
        ``transform_params_invalid`` when a row has zero variance —
        correlation distance is undefined there and fabricating a value
        would be a silent wrong number.
    """

    centered = features.double() - features.double().mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1)
    if bool((norms == 0).any().item()):
        bad = torch.nonzero(norms == 0).flatten().tolist()
        raise TransformContractError(
            f"{context}: rows {bad} have zero variance, so their correlation "
            "distance is undefined; fabricating a value would be a silent "
            "wrong number.",
            code="transform_params_invalid",
            remedy="drop constant rows (or add informative features) before probing",
            rows=bad,
        )
    normalized = centered / norms.unsqueeze(1)
    similarity = normalized @ normalized.T
    n = features.shape[0]
    rows, cols = torch.triu_indices(n, n, offset=1)
    return 1.0 - similarity[rows, cols]


@dataclass(frozen=True)
class ProbeReport:
    """Fidelity-probe result: rows per candidate width + T-C12 scope fields.

    Attributes
    ----------
    rows:
        One mapping per candidate width: ``n_components``,
        ``l2_rdm_spearman``, ``corr_rdm_spearman``.
    n_rows:
        Pilot-batch row count.
    input_extent:
        Flattened per-row feature width ``D``.
    seed:
        Base seed used for every probe projection.
    construction:
        SRP construction probed.
    algorithm_version:
        Construction algorithm version.
    corpus:
        Scope disclosure — always the user's own pilot batch.
    distance_claim:
        The construction's earned claim (``"empirical_only"``).
    """

    rows: tuple[dict[str, Any], ...]
    n_rows: int
    input_extent: int
    seed: int
    construction: str
    algorithm_version: int
    corpus: str
    distance_claim: str

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-portable probe record (scope travels with numbers).

        Returns
        -------
        dict[str, Any]
            The full report as a canonical-JSON-portable mapping.
        """

        return {
            "schema": "tl_srp_fidelity_probe_v1",
            "rows": list(self.rows),
            "n_rows": self.n_rows,
            "input_extent": self.input_extent,
            "seed": self.seed,
            "construction": self.construction,
            "algorithm_version": self.algorithm_version,
            "corpus": self.corpus,
            "distance_claim": self.distance_claim,
        }

    def canonical_json(self) -> str:
        """Return the canonical JSON of :meth:`to_record`."""

        return canonical_json(self.to_record())

    def __str__(self) -> str:
        """Small aligned table with the scope line attached."""

        lines = [
            f"SRP fidelity probe on YOUR pilot batch (n={self.n_rows}, "
            f"D={self.input_extent}, seed={self.seed}, {self.construction} "
            f"v{self.algorithm_version}, distance_claim={self.distance_claim}):",
            f"{'k':>8}  {'L2-RDM rho':>10}  {'corr-RDM rho':>12}",
        ]
        for row in self.rows:
            lines.append(
                f"{row['n_components']:>8}  {row['l2_rdm_spearman']:>10.3f}  "
                f"{row['corr_rdm_spearman']:>12.3f}"
            )
        lines.append("Numbers are scoped to THIS batch; other stimuli will differ (T-C12).")
        return "\n".join(lines)


def srp_fidelity_probe(
    features: torch.Tensor,
    n_components: Sequence[int],
    seed: int = 0,
    construction: str = "very_sparse_fixed",
) -> ProbeReport:
    """Measure L2-RDM and corr-RDM fidelity per candidate width on a pilot batch.

    Parameters
    ----------
    features:
        Pilot activations, ``(n, ...)`` with the stimulus axis leading
        (flattened per row); ``n >= 4`` so the RDMs have enough pairs.
    n_components:
        Candidate output widths to probe.
    seed:
        Base seed for the probe projections.
    construction:
        SRP construction to probe.

    Returns
    -------
    ProbeReport
        Per-width fidelity rows with the T-C12 scope fields attached.
    """

    if not isinstance(features, torch.Tensor) or features.dim() < 2 or features.shape[0] < 4:
        raise TransformContractError(
            "srp_fidelity_probe needs a (n >= 4, ...) float tensor of pilot "
            "activations; RDM fidelity over fewer rows has almost no pairs.",
            code="transform_params_invalid",
            remedy="pass a pilot batch of at least 4 stimuli",
        )
    if not features.is_floating_point():
        raise TransformContractError(
            f"srp_fidelity_probe got dtype {features.dtype}; probe float activations.",
            code="transform_params_invalid",
            remedy="cast the pilot batch to a float dtype",
        )
    widths = list(n_components)
    if not widths:
        raise TransformContractError(
            "srp_fidelity_probe needs at least one candidate n_components.",
            code="transform_params_invalid",
            remedy="pass candidate widths, e.g. (256, 1024, 4096)",
        )
    flat = features.reshape(features.shape[0], -1)
    extent = int(flat.shape[1])
    full_l2 = _condensed_l2_rdm(flat)
    full_corr = _condensed_corr_rdm(flat, context="srp_fidelity_probe")
    rows: list[dict[str, Any]] = []
    for k in widths:
        spec = srp(k, seed=seed, construction=construction)
        projected = spec.apply(flat, TransformContext(site_label=f"probe:k={k}"))
        rows.append(
            {
                "n_components": int(k),
                "l2_rdm_spearman": spearman(full_l2, _condensed_l2_rdm(projected)),
                "corr_rdm_spearman": spearman(
                    full_corr,
                    _condensed_corr_rdm(projected, context="srp_fidelity_probe"),
                ),
            }
        )
    return ProbeReport(
        rows=tuple(rows),
        n_rows=int(flat.shape[0]),
        input_extent=extent,
        seed=seed,
        construction=construction,
        algorithm_version=ALGORITHM_VERSION,
        corpus="user_pilot_batch",
        distance_claim="empirical_only",
    )
