"""``torchlens.neuro.rdms``: RDM handoff with provenance that cannot lie (F22).

Memo 4.2 (item 7): ONE name, two type-distinct modes, one shared condensed
converter. Source mode (the taught path) computes through the canonical
``tl.repgeom`` distance arithmetic and records the metric it actually used --
when the function computes, the label cannot lie. Matrix mode (the escape
hatch) converts already-computed square matrices only and REQUIRES the
declared dissimilarity measure, because a bare matrix has forgotten its
metric. The condensed extraction is ``square[np.triu_indices(n, k=1)]`` --
measured to match rsatoolbox's own convention exactly at 0.1.5 and 0.3.2,
so no scipy dependency.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._handoff import (
    PRESENTATION_INDEX_KEY,
    NeuroHandoffError,
    SiteOutcome,
    SitePayload,
    _source_kind,
    base_descriptors,
    check_pool,
    enumerate_extraction_sites,
    enumerate_trace_sites,
    require_rsatoolbox,
    widen_for_handoff,
)

#: Sentinel distinguishing "metric not passed" from an explicit value, so
#: matrix mode can refuse even an explicit ``metric="correlation"`` (the
#: source-mode default) instead of silently ignoring a mixed-mode argument.
_UNSET: Any = object()

#: The convention each computed metric records in words (memo D9):
#: ``dissimilarity_measure`` keeps the plain metric word (rsatoolbox's own
#: field; downstream code switches on it) and ``measure_convention`` carries
#: the formula, telling users exactly how to reproduce their number.
_MEASURE_CONVENTIONS: dict[str, str] = {
    "correlation": (
        "1 - Pearson correlation across features per stimulus pair (rows "
        "centered); matches rsatoolbox's 'correlation' exactly"
    ),
    "euclidean": (
        "unsquared euclidean distance over flattened features; rsatoolbox's "
        "'euclidean' equals this value squared and divided by n_features"
    ),
    "cosine": (
        "1 - cosine similarity over flattened features; rsatoolbox has no cosine RDM method"
    ),
    "gaussian": (
        "1 - exp(-D / (2 * mean(D))) with D the squared euclidean distances "
        "(thingsvision's bandwidth convention)"
    ),
}


def rdms(  # noqa: PLR0913 -- memo-specified public door: one name, two type-distinct modes; the knob union IS the spec'd surface
    source: Any,
    sites: Any = None,
    *,
    metric: Any = _UNSET,
    dissimilarity_measure: str | None = None,
    pool: Any = None,
    obs: Any = None,
    pattern_descriptors: Any = None,
    chunked: Any = None,
) -> Any:
    """Convert stimulus-indexed sites (or precomputed matrices) into RDMs.

    Two type-distinct modes share one converter and reject mixed arguments:

    - SOURCE MODE (the taught path): ``source`` is a TorchLens ``Trace``, a
      ``LoadedExtraction``, or an extraction-directory path. Site selection
      runs through the core stimulus-indexed gate and distance arithmetic
      delegates to ``tl.repgeom``; ``dissimilarity_measure`` is set from the
      metric ACTUALLY used (``measure_source="computed"``).
    - MATRIX MODE (the escape hatch): ``source`` is a mapping of site key to
      already-computed square dissimilarity matrix; ``metric=`` is invalid
      and ``dissimilarity_measure=`` is REQUIRED
      (``measure_source="declared"``). Pattern descriptors are NEVER
      invented from stimulus order -- this function cannot know a foreign
      matrix's row order (a matrix that came back from rsatoolbox's
      ``calc_rdm(descriptor=...)`` is in SORTED order, not presentation
      order).

    Parameters
    ----------
    source:
        Trace | LoadedExtraction | extraction-directory path (source mode),
        or ``Mapping[str, square_matrix]`` (matrix mode).
    sites:
        Source mode only: ``None`` sweeps every stimulus-indexed saved site;
        an explicit lookup refuses ineligible sites actionably.
    metric:
        Source mode only. Defaults to ``"correlation"`` -- field canon for
        the RSA audience, and the one metric measured to match rsatoolbox
        exactly at 1.000000 on real features. This default deliberately
        diverges from ``tl.repgeom.rdm`` (whose default is ``"euclidean"``);
        both docstrings cross-reference the divergence. The vocabulary is
        repgeom's: ``"euclidean"``, ``"cosine"``, ``"correlation"``,
        ``"gaussian"``.
    dissimilarity_measure:
        Matrix mode only, REQUIRED there: the declared measure of the
        precomputed matrices (a bare matrix has forgotten its metric).
    pool:
        Source mode only. Launch vocabulary: ``None`` -> disclosed
        flattening, recorded per-site as ``pool="flatten"``.
    obs:
        Source mode only: mapping of descriptor name to per-stimulus
        sequence. With one row per stimulus, observation descriptors ARE
        pattern descriptors after the RDM handoff, so validated columns land
        in ``pattern_descriptors``. Length-checked before rsatoolbox sees
        them. NOTE: unlike ``rsatoolbox.rdm.calc_rdm(descriptor=...)``, this
        function never averages repeated conditions -- each stimulus row is
        one pattern; average upstream (or hand ``neuro.datasets`` output to
        ``calc_rdm``) when averaging is what you want.
    pattern_descriptors:
        Mapping of per-pattern columns, validated to the pattern count.
        In matrix mode this is the ONLY identity door; absent, rows carry a
        positional index disclosed as positional.
    chunked:
        Reserved execution seam (NSD-scale streaming; memo section 8).
        Launch is eager: any non-``None`` value refuses typed.

    Returns
    -------
    Any
        ``rsatoolbox.rdm.RDMs`` with one RDM per site. ``rdm_descriptors``
        carry the per-site identity story (site, requested lookup, site_key,
        layer label, pool, dtype, casts) plus ``measure_source`` and
        ``measure_convention``; the sweep ledger is attached as the
        session-only ``tl_ledger`` attribute (a tuple of per-site outcomes).

    Raises
    ------
    ImportError
        When rsatoolbox is not installed.
    NeuroHandoffError
        Typed refusals: mixed-mode arguments
        (``neuro_rdms_mode_conflict``), missing declared measure
        (``neuro_rdms_measure_required``), invalid matrices
        (``neuro_matrix_invalid``), inconsistent site row counts
        (``neuro_sites_inconsistent_rows``), reserved ``chunked=``
        (``neuro_rdms_chunked_unavailable``), and the shared source/pool/
        obs refusals.
    ValueError
        Source-mode core refusals, identical in class to calling
        ``tl.repgeom.rdm_evolution`` directly (the pinned anti-divergence
        invariant): explicitly requested ineligible or unsaved sites,
        unsupported metrics, and metric preconditions.
    """

    rsatoolbox = require_rsatoolbox()
    if chunked is not None:
        raise NeuroHandoffError(
            f"chunked={chunked!r} is not available: the chunked execution "
            "seam is reserved for NSD-scale streaming RDMs and launch is "
            "eager.",
            code="neuro_rdms_chunked_unavailable",
            remedy="drop chunked= (eager execution) for now",
        )

    if isinstance(source, Mapping):
        return _matrix_mode(
            rsatoolbox,
            source,
            sites=sites,
            metric=metric,
            dissimilarity_measure=dissimilarity_measure,
            pool=pool,
            obs=obs,
            pattern_descriptors=pattern_descriptors,
        )
    return _source_mode(
        rsatoolbox,
        source,
        sites=sites,
        metric=metric,
        dissimilarity_measure=dissimilarity_measure,
        pool=pool,
        obs=obs,
        pattern_descriptors=pattern_descriptors,
    )


def _condensed(square: np.ndarray) -> np.ndarray:
    """Extract the condensed upper triangle (the ONE shared converter).

    ``square[np.triu_indices(n, k=1)]`` is measured to match rsatoolbox's
    own condensed ordering exactly at 0.1.5 and 0.3.2.

    Parameters
    ----------
    square:
        Square dissimilarity matrix.

    Returns
    -------
    np.ndarray
        The ``n * (n - 1) / 2`` upper-triangle values in rsatoolbox order.
    """

    return square[np.triu_indices(square.shape[0], k=1)]


def _validate_pattern_descriptors(
    pattern_descriptors: Any, n_patterns: int
) -> dict[str, np.ndarray]:
    """Validate user per-pattern descriptor columns.

    Parameters
    ----------
    pattern_descriptors:
        Mapping of descriptor name to per-pattern sequence, or ``None``.
    n_patterns:
        Pattern count every column must match.

    Returns
    -------
    dict[str, np.ndarray]
        Validated columns (empty when ``None``).

    Raises
    ------
    NeuroHandoffError
        ``neuro_obs_descriptor_invalid`` on a non-mapping or a mis-lengthed
        column.
    """

    if pattern_descriptors is None:
        return {}
    if not hasattr(pattern_descriptors, "items"):
        raise NeuroHandoffError(
            f"pattern_descriptors= must be a mapping of descriptor name to "
            f"per-pattern sequence; got {type(pattern_descriptors).__name__}.",
            code="neuro_obs_descriptor_invalid",
            remedy="pass e.g. pattern_descriptors={'condition': [...one entry per pattern...]}",
        )
    validated: dict[str, np.ndarray] = {}
    for key, column in pattern_descriptors.items():
        name = str(key)
        values = list(column)
        if len(values) != n_patterns:
            raise NeuroHandoffError(
                f"pattern descriptor {name!r} has {len(values)} entries for "
                f"{n_patterns} patterns; a misaligned condition vector is "
                "plausible, scientifically wrong, and invisible downstream.",
                code="neuro_obs_descriptor_invalid",
                remedy="pass exactly one entry per pattern, in row order",
                descriptor=name,
                n_entries=len(values),
                n_patterns=n_patterns,
            )
        validated[name] = np.asarray(values)
    return validated


def _source_mode(  # noqa: PLR0913 -- mirrors the rdms() door's knob set verbatim (one pass-through worker)
    rsatoolbox: Any,
    source: Any,
    *,
    sites: Any,
    metric: Any,
    dissimilarity_measure: str | None,
    pool: Any,
    obs: Any,
    pattern_descriptors: Any,
) -> Any:
    """Compute RDMs from a Trace or extraction artifact (memo 4.2).

    Parameters
    ----------
    rsatoolbox:
        The imported rsatoolbox package.
    source:
        Trace, ``LoadedExtraction``, or extraction-directory path.
    sites, metric, dissimilarity_measure, pool, obs, pattern_descriptors:
        See :func:`rdms`.

    Returns
    -------
    Any
        ``rsatoolbox.rdm.RDMs``.

    Raises
    ------
    NeuroHandoffError
        ``neuro_rdms_mode_conflict`` when ``dissimilarity_measure=`` is
        passed (source mode computes; the label cannot be hand-typed).
    ValueError
        Core site-selection and metric refusals (anti-divergence with
        ``tl.repgeom``).
    """

    from ..repgeom._geometry import activation_distance_matrix

    if dissimilarity_measure is not None:
        raise NeuroHandoffError(
            f"dissimilarity_measure={dissimilarity_measure!r} is invalid in "
            "source mode: torchlens computes the RDMs here, so the measure "
            "is recorded from the metric actually used -- a hand-typed label "
            "restates something ambiguous (two live euclidean conventions "
            "exist in this ecosystem).",
            code="neuro_rdms_mode_conflict",
            remedy=(
                "pass metric= to choose what is computed, or pass a mapping "
                "of precomputed matrices to use matrix mode"
            ),
        )
    resolved_metric = "correlation" if metric is _UNSET or metric is None else str(metric)
    check_pool(pool)
    kind = _source_kind(source)

    trace = source if kind == "trace" else None
    if kind == "trace":
        payloads, ledger = enumerate_trace_sites(source, sites, verb="neuro.rdms")
    else:
        payloads, ledger, _loaded = enumerate_extraction_sites(source, sites, verb="neuro.rdms")

    if not payloads:
        raise ValueError(
            "neuro.rdms found no eligible stimulus-indexed saved sites; "
            "capture with save= covering the layers to compare, or check "
            "the returned ledger reasons on a datasets() sweep."
        )
    row_counts = {int(payload.matrix.shape[0]) for payload in payloads}
    if len(row_counts) > 1:
        raise NeuroHandoffError(
            f"the selected sites disagree on stimulus-row counts "
            f"{sorted(row_counts)}; one RDMs stack must share one pattern "
            "set.",
            code="neuro_sites_inconsistent_rows",
            remedy="select sites from one capture with one stimulus set",
            row_counts=sorted(row_counts),
        )
    n_rows = row_counts.pop()
    if n_rows < 2:
        raise ValueError(f"neuro.rdms has too few stimuli: got {n_rows}, need at least 2.")

    condensed_rows: list[np.ndarray] = []
    computed: list[SitePayload] = []
    casts: list[str] = []
    for payload in payloads:
        matrix, cast_record = widen_for_handoff(payload.matrix)
        try:
            square = activation_distance_matrix(matrix, metric=resolved_metric)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(f"neuro.rdms failed for site {payload.key!r}: {exc}") from exc
        condensed_rows.append(_condensed(square))
        computed.append(payload)
        casts.append(cast_record or "")

    patterns, identity = _source_pattern_descriptors(computed, n_rows, obs, pattern_descriptors)
    rdm_descriptors = _source_rdm_descriptors(computed, casts, source_kind=kind, trace=trace)

    sample = base_descriptors(computed[0], source_kind=kind, trace=trace)
    descriptors: dict[str, Any] = {
        "source": "torchlens",
        "source_kind": kind,
        "measure_source": "computed",
        "measure_convention": _MEASURE_CONVENTIONS.get(
            resolved_metric, f"computed by tl.repgeom metric {resolved_metric!r}"
        ),
        "pattern_identity": identity,
        "tl_version": sample["tl_version"],
        "rsatoolbox_version": sample["rsatoolbox_version"],
    }
    if "model" in sample:
        descriptors["model"] = sample["model"]
    if "tl_intervened" in sample:
        descriptors["tl_intervened"] = sample["tl_intervened"]

    result = rsatoolbox.rdm.RDMs(
        dissimilarities=np.stack(condensed_rows),
        dissimilarity_measure=resolved_metric,
        descriptors=descriptors,
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=patterns,
    )
    result.tl_ledger = tuple(ledger)
    return result


def _source_pattern_descriptors(
    computed: list[SitePayload],
    n_rows: int,
    obs: Any,
    pattern_descriptors: Any,
) -> tuple[dict[str, np.ndarray], str]:
    """Build the shared pattern-descriptor block for a source-mode stack.

    Pattern identity: recorded artifact ids when every site carries the
    same recording, else a positional index disclosed as synthetic. The
    presentation-index descriptor is ALWAYS written (memo D10).

    Parameters
    ----------
    computed:
        Payloads in RDM-stack order.
    n_rows:
        Shared stimulus-row count.
    obs:
        User observation-descriptor mapping (one row per stimulus, so
        observation descriptors ARE pattern descriptors after the handoff).
    pattern_descriptors:
        User per-pattern columns.

    Returns
    -------
    tuple[dict[str, np.ndarray], str]
        The pattern-descriptor mapping and the identity provenance token.
    """

    from ._handoff import validate_obs

    id_sets = {tuple(payload.row_ids) for payload in computed if payload.row_ids is not None}
    if len(id_sets) == 1 and all(payload.row_ids is not None for payload in computed):
        ids = [str(item) for item in id_sets.pop()]
        identity = "recorded"
    else:
        ids = [str(index) for index in range(n_rows)]
        identity = "synthetic_positional"
    patterns: dict[str, np.ndarray] = {
        "stimulus_id": np.asarray(ids),
        PRESENTATION_INDEX_KEY: np.arange(n_rows),
    }
    for name, column in validate_obs(obs, n_rows, site="<all sites>").items():
        patterns[name] = np.asarray(column)
    patterns.update(_validate_pattern_descriptors(pattern_descriptors, n_rows))
    return patterns, identity


def _source_rdm_descriptors(
    computed: list[SitePayload],
    casts: list[str],
    *,
    source_kind: str,
    trace: Any | None,
) -> dict[str, np.ndarray]:
    """Assemble the per-RDM identity columns for a source-mode stack.

    Parameters
    ----------
    computed:
        Payloads in RDM-stack order.
    casts:
        Per-site handoff-cast records (empty string = no cast).
    source_kind:
        ``"trace"`` or ``"extraction"``.
    trace:
        Source trace on the in-memory route, else ``None``.

    Returns
    -------
    dict[str, np.ndarray]
        Column name to per-RDM array.
    """

    columns: dict[str, list[Any]] = {
        "site": [],
        "requested": [],
        "site_key": [],
        "layer_label": [],
        "pool": [],
        "source_dtype": [],
        "handoff_cast": [],
        "pass_index": [],
    }
    for payload, cast_record in zip(computed, casts, strict=True):
        per_site = base_descriptors(payload, source_kind=source_kind, trace=trace)
        columns["site"].append(payload.key)
        columns["requested"].append(payload.requested)
        columns["site_key"].append(payload.site_key)
        columns["layer_label"].append(payload.layer_label)
        columns["pool"].append(per_site["pool"])
        columns["source_dtype"].append(payload.source_dtype)
        columns["handoff_cast"].append(cast_record)
        columns["pass_index"].append(payload.pass_index if payload.pass_index is not None else 0)
    return {name: np.asarray(column) for name, column in columns.items()}


def _matrix_mode(  # noqa: PLR0913 -- mirrors the rdms() door's knob set verbatim (one pass-through worker)
    rsatoolbox: Any,
    source: Mapping[str, Any],
    *,
    sites: Any,
    metric: Any,
    dissimilarity_measure: str | None,
    pool: Any,
    obs: Any,
    pattern_descriptors: Any,
) -> Any:
    """Convert precomputed square matrices into RDMs (memo 4.2 escape hatch).

    Parameters
    ----------
    rsatoolbox:
        The imported rsatoolbox package.
    source:
        Mapping of site key to square dissimilarity matrix.
    sites, metric, dissimilarity_measure, pool, obs, pattern_descriptors:
        See :func:`rdms`.

    Returns
    -------
    Any
        ``rsatoolbox.rdm.RDMs`` with ``measure_source="declared"``.

    Raises
    ------
    NeuroHandoffError
        Mixed-mode arguments (``neuro_rdms_mode_conflict``), a missing
        declared measure (``neuro_rdms_measure_required``), or invalid
        matrices (``neuro_matrix_invalid``).
    """

    conflicts = [
        name
        for name, value in (
            ("metric", None if metric is _UNSET else metric),
            ("sites", sites),
            ("pool", pool),
            ("obs", obs),
        )
        if value is not None or (name == "metric" and metric is not _UNSET)
    ]
    if conflicts:
        raise NeuroHandoffError(
            f"matrix mode (a mapping of precomputed matrices) rejects "
            f"{', '.join(sorted(set(conflicts)))}=: nothing is computed or "
            "selected here, only converted.",
            code="neuro_rdms_mode_conflict",
            remedy=(
                "drop the source-mode arguments, or pass a Trace/"
                "LoadedExtraction/path to compute in source mode"
            ),
            arguments=sorted(set(conflicts)),
        )
    if not dissimilarity_measure or not str(dissimilarity_measure).strip():
        raise NeuroHandoffError(
            "matrix mode REQUIRES dissimilarity_measure=: a bare matrix has "
            "forgotten its metric, and rsatoolbox comparison code switches "
            "on this field.",
            code="neuro_rdms_measure_required",
            remedy=(
                "declare the measure the matrices were computed with, e.g. "
                "dissimilarity_measure='correlation'"
            ),
        )

    if not source:
        raise NeuroHandoffError(
            "matrix mode received an empty mapping: there is nothing to convert.",
            code="neuro_matrix_invalid",
            remedy="pass at least one site key -> square matrix entry",
        )
    keys = [str(key) for key in source]
    if len(set(keys)) != len(keys):
        raise NeuroHandoffError(
            "matrix-mode keys collide after string conversion; RDM "
            "descriptors must identify each matrix uniquely.",
            code="neuro_matrix_invalid",
            remedy="use unique string site keys",
        )

    arrays: list[np.ndarray] = []
    for key, value in source.items():
        arrays.append(_validated_matrix(str(key), value, arrays))

    return _build_matrix_rdms(
        rsatoolbox,
        keys=keys,
        arrays=arrays,
        dissimilarity_measure=str(dissimilarity_measure),
        pattern_descriptors=pattern_descriptors,
    )


def _validated_matrix(key: str, value: Any, earlier: list[np.ndarray]) -> np.ndarray:
    """Validate ONE matrix-mode entry (the full refusal ladder, memo 4.2).

    Parameters
    ----------
    key:
        Site key of the entry (named in every refusal).
    value:
        The caller's matrix-like value.
    earlier:
        Previously validated arrays (shape-agreement evidence).

    Returns
    -------
    np.ndarray
        The validated float64 square matrix.

    Raises
    ------
    NeuroHandoffError
        ``neuro_matrix_invalid`` for non-square, too-few-pattern,
        pattern-count-disagreeing, non-finite, asymmetric, or
        non-zero-diagonal input.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise NeuroHandoffError(
            f"matrix {key!r} has shape {tuple(array.shape)}; matrix mode "
            "converts square dissimilarity matrices only.",
            code="neuro_matrix_invalid",
            remedy="pass n x n square matrices",
            site=key,
            shape=list(array.shape),
        )
    if array.shape[0] < 2:
        raise NeuroHandoffError(
            f"matrix {key!r} has {array.shape[0]} pattern(s); an RDM needs at least 2.",
            code="neuro_matrix_invalid",
            remedy="pass matrices over at least 2 patterns",
            site=key,
        )
    if earlier and int(array.shape[0]) != int(earlier[0].shape[0]):
        raise NeuroHandoffError(
            f"matrix {key!r} has {array.shape[0]} patterns but earlier "
            f"matrices have {int(earlier[0].shape[0])}; one RDMs stack "
            "must share one pattern set.",
            code="neuro_matrix_invalid",
            remedy="convert matrices over one shared stimulus set together",
            site=key,
        )
    if not np.isfinite(array).all():
        raise NeuroHandoffError(
            f"matrix {key!r} contains non-finite values; the finite-value "
            "policy refuses rather than propagating NaN/Inf into "
            "rsatoolbox comparisons.",
            code="neuro_matrix_invalid",
            remedy="clean or mask the non-finite entries before conversion",
            site=key,
        )
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    tolerance = 1e-10 * max(1.0, scale)
    if float(np.max(np.abs(array - array.T))) > tolerance:
        raise NeuroHandoffError(
            f"matrix {key!r} is not symmetric within tolerance; a dissimilarity matrix must be.",
            code="neuro_matrix_invalid",
            remedy="symmetrize the matrix or fix the upstream computation",
            site=key,
        )
    if float(np.max(np.abs(np.diagonal(array)))) > tolerance:
        raise NeuroHandoffError(
            f"matrix {key!r} has a non-zero diagonal; self-dissimilarity must be zero.",
            code="neuro_matrix_invalid",
            remedy="zero the diagonal or fix the upstream computation",
            site=key,
        )
    return array


def _build_matrix_rdms(
    rsatoolbox: Any,
    *,
    keys: list[str],
    arrays: list[np.ndarray],
    dissimilarity_measure: str,
    pattern_descriptors: Any,
) -> Any:
    """Assemble the matrix-mode RDMs stack from validated inputs.

    NEVER invents pattern identity: a foreign matrix's row order is
    unknowable (rsatoolbox's ``calc_rdm(descriptor=...)`` returns SORTED
    rows). Either the caller supplies descriptors or rows carry a
    positional index disclosed as positional.

    Parameters
    ----------
    rsatoolbox:
        The imported rsatoolbox package.
    keys:
        Validated unique site keys, in insertion order.
    arrays:
        Validated square matrices, aligned with ``keys``
        (their shared leading axis is the pattern count).
    dissimilarity_measure:
        The caller's declared measure.
    pattern_descriptors:
        User per-pattern columns, or ``None``.

    Returns
    -------
    Any
        ``rsatoolbox.rdm.RDMs`` with ``measure_source="declared"``.
    """

    n_patterns = int(arrays[0].shape[0])
    patterns = _validate_pattern_descriptors(pattern_descriptors, n_patterns)
    pattern_identity = "user_supplied" if patterns else "positional"
    if not patterns:
        patterns = {"tl_pattern_index": np.arange(n_patterns)}

    import torchlens

    from ._handoff import rsatoolbox_version

    result = rsatoolbox.rdm.RDMs(
        dissimilarities=np.stack([_condensed(array) for array in arrays]),
        dissimilarity_measure=str(dissimilarity_measure),
        descriptors={
            "source": "torchlens",
            "source_kind": "matrices",
            "measure_source": "declared",
            "measure_convention": (
                "declared by the caller (measure_source='declared'); "
                "torchlens did not compute these values"
            ),
            "pattern_identity": pattern_identity,
            "tl_version": str(getattr(torchlens, "__version__", "unknown")),
            "rsatoolbox_version": rsatoolbox_version(),
        },
        rdm_descriptors={"site": np.asarray(keys)},
        pattern_descriptors=patterns,
    )
    result.tl_ledger = tuple(SiteOutcome(key, "computed") for key in keys)
    return result


__all__ = ["rdms"]
