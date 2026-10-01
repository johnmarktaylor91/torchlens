"""The sink-neutral emission view over the C06 history records (memo 3.1-3.5).

One conversion layer turns explorer-owned completed records
(:class:`~torchlens.observability.StepBlockRecord` +
:class:`~torchlens.observability.ObservationRecord`) into immutable,
tensor-free emission points every sink serializes from. Nothing here touches
a live tensor, rescans a payload, or invents a second step axis.

Namespace grammar (memo 3.4; structure panel-decided, spellings
DOCUMENTED-UNSTABLE):

- ``<family>/<statistic>/<leaf>`` for scalars,
- ``<family>/hist/<leaf>`` for histograms,
- ``<family>/<name>/...`` when ``name=`` inserts the multi-model component,
- ``torchlens/run/<health_key>`` for run health,
- ``torchlens/meta/<kind>`` for the manifest,
- ``torchlens/check/<check_name>`` for check outcomes (0 pass/1 warn/2 fail).

Family comes FIRST because both major dashboards group on the first slash
component only (measured); statistic-before-leaf keeps the cross-layer scan
property (``gradients/norm/*`` sorts adjacent).

Tag safety is an ASSERTION, never a rewrite (D4 majority rule): torch 2.13
ships no sanitizer, older torch silently rewrote -- so enforcement is ours,
and a tag that would need sanitizing refuses and names the site.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from ..observability import (
    HISTORY_SCHEMA_VERSION,
    CommittedBlock,
    HistogramResult,
    RunRecord,
    SiteRecord,
    SpineResult,
)
from ._errors import TagGrammarError

__tl_layer__ = "L8"

#: Emission-view schema version (rides the manifest and every JSONL row).
EMISSION_SCHEMA_VERSION = 1

#: The closed tag families (memo 3.4). ``torchlens`` is the reserved
#: run-health/meta/check root, deliberately NOT a data family.
FAMILIES = (
    "parameters",
    "gradients",
    "activations",
    "activation_grads",
    "updates",
    "nonfinite",
)

#: C06 stream -> tag family.
STREAM_FAMILIES = {
    "param": "parameters",
    "param_grad": "gradients",
    "activation": "activations",
    "activation_grad": "activation_grads",
    "param_delta": "updates",
}

#: Scalar statistics derived from the always-on spine, in emission order.
#: Every histogram's summary series ships as these first-class scalar tags
#: because scalars are the ONLY cross-sink-portable form (memo 3.3).
SPINE_STATISTICS = (
    "mean",
    "std",
    "norm",
    "min",
    "max",
    "count",
    "zero_fraction",
)

#: Versioned tag-safety check (D4). Printable ASCII only; the URL-hazard and
#: markdown-table characters that break dashboard routing refuse. A bump to
#: this version is a documented behavior change, never silent.
TAG_SAFETY_CHECK_VERSION = 1

_TAG_FORBIDDEN = set("?#%\"'\\`|<>")


def assert_tag_safe(tag: str, *, site: str | None = None) -> str:
    """Assert one full tag is dashboard-safe; refuse-and-name otherwise.

    The measured basis: 447/447 real site keys across two real checkpoints
    are safe, so this branch fires only on exotic keys -- and when it fires,
    a silent rewrite would break the pasteable-leaf property (the leaf must
    paste back into ``trace[...]`` / ``named_modules()`` verbatim), so the
    only honest behavior is a typed refusal naming the site (D4 majority;
    Sol's reversible-escape minority is recorded, not implemented).
    """

    bad = sorted({ch for ch in tag if ord(ch) < 0x20 or ord(ch) > 0x7E or ch in _TAG_FORBIDDEN})
    components = tag.split("/")
    if not bad and all(part not in ("", ".", "..") for part in components):
        return tag
    raise TagGrammarError(
        f"Tag {tag!r} fails the versioned tag-safety check "
        f"(v{TAG_SAFETY_CHECK_VERSION}): "
        + (
            f"characters {bad!r} are not dashboard-safe"
            if bad
            else "empty or dot-only path components break dashboard grouping"
        )
        + ". TorchLens never silently rewrites a tag: a rewritten leaf would "
        "no longer paste back into trace[...] or named_modules(), which is "
        "the property these tags exist to keep.",
        code="tracker_tag_unsafe",
        tag=tag,
        site=site or "",
        check_version=TAG_SAFETY_CHECK_VERSION,
        remedy=(
            "Rename the offending module/parameter, or pass name=/namespace= "
            "to re-root your series under a safe component."
        ),
    )


@dataclass(frozen=True)
class ScalarPoint:
    """One scalar emission: ``tag`` at ``step`` with ``value``."""

    tag: str
    step: int
    value: float


@dataclass(frozen=True)
class HistogramPoint:
    """One histogram emission: exact counts on explicit edges.

    ``edges`` has ``len(counts) + 1`` entries; bin ``i`` covers
    ``[edges[i], edges[i+1])``. The signed-log2 grid is NON-uniform, which
    is why relay paths refuse histogram series (G6): a relay reconstructs
    outer edges by linear extrapolation, exact only for uniform bins.
    ``summary`` carries the spine fields TB's histogram proto wants
    (min/max/num/sum/sum_squares); every relay destroys them (measured), so
    the same numbers ALSO ship as first-class scalar tags.
    """

    tag: str
    step: int
    counts: tuple[int, ...]
    edges: tuple[float, ...]
    summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TextPoint:
    """One text emission (manifests, disclosures)."""

    tag: str
    step: int
    text: str


@dataclass(frozen=True)
class StepEmission:
    """Everything one committed step block emits, in emission order."""

    step: int
    scalars: tuple[ScalarPoint, ...]
    histograms: tuple[HistogramPoint, ...]
    texts: tuple[TextPoint, ...] = ()


class TagGrammar:
    """The tag renderer: one function over one internal identity (D1).

    ``name=`` inserts the multi-model component after the family;
    ``namespace=`` re-roots every tag (the collision remedy the refusal
    names). Both are validated once at construction.
    """

    def __init__(self, *, name: str | None = None, namespace: str | None = None) -> None:
        """Validate and freeze the optional grammar components."""

        for label, value in (("name", name), ("namespace", namespace)):
            if value is not None:
                if "/" in value or not value:
                    raise TagGrammarError(
                        f"{label}={value!r} must be a single non-empty tag "
                        "component: it becomes one path segment of every "
                        "emitted tag, and a slash would silently change which "
                        "dashboard section owns the series.",
                        code="tracker_tag_unsafe",
                        tag=value or "",
                        check_version=TAG_SAFETY_CHECK_VERSION,
                        remedy=f"Pass a slash-free non-empty {label}.",
                    )
                assert_tag_safe(value, site=f"{label}=")
        self.name = name
        self.namespace = namespace

    def _root(self, family: str) -> str:
        """Family (plus optional namespace/name components) prefix."""

        parts = []
        if self.namespace is not None:
            parts.append(self.namespace)
        parts.append(family)
        if self.name is not None and family in FAMILIES:
            parts.append(self.name)
        return "/".join(parts)

    def data(self, family: str, statistic: str, leaf: str) -> str:
        """Render one data tag: ``<family>/<statistic>/<leaf>``."""

        return assert_tag_safe(f"{self._root(family)}/{statistic}/{leaf}", site=leaf)

    def run_health(self, key: str) -> str:
        """Render one run-health tag: ``torchlens/run/<key>``."""

        return assert_tag_safe(f"{self._root('torchlens')}/run/{key}", site=key)

    def meta(self, kind: str) -> str:
        """Render one meta tag: ``torchlens/meta/<kind>``."""

        return assert_tag_safe(f"{self._root('torchlens')}/meta/{kind}", site=kind)

    def check(self, check_name: str) -> str:
        """Render one check-outcome tag: ``torchlens/check/<name>``."""

        return assert_tag_safe(f"{self._root('torchlens')}/check/{check_name}", site=check_name)


def spine_scalars(spine: SpineResult) -> dict[str, float]:
    """Derive the portable scalar statistics from one spine record.

    ``std`` comes from the stable Chan moments (``m2 / count_finite``);
    ``norm`` is the L2 norm from ``sum_squares``. Absent floats (empty
    finite population) are simply omitted -- missing is never zero (D6).
    """

    out: dict[str, float] = {"count": float(spine.count_finite)}
    if spine.count_finite > 0:
        total = float(spine.count_total)
        out["zero_fraction"] = spine.count_zero / total if total else 0.0
        if spine.mean is not None:
            out["mean"] = spine.mean
        if spine.m2 is not None and spine.count_finite > 1:
            out["std"] = math.sqrt(max(spine.m2, 0.0) / spine.count_finite)
        if spine.sum_squares is not None:
            out["norm"] = math.sqrt(max(spine.sum_squares, 0.0))
        if spine.finite_min is not None:
            out["min"] = spine.finite_min
        if spine.finite_max is not None:
            out["max"] = spine.finite_max
    return out


def nonfinite_scalars(spine: SpineResult) -> dict[str, float]:
    """The nonfinite-family series: NaN / +-inf counts, exact integers."""

    return {
        "nan": float(spine.count_nan),
        "posinf": float(spine.count_posinf),
        "neginf": float(spine.count_neginf),
    }


def histogram_points(sketch: HistogramResult) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Render one signed-log2 sketch as explicit ``(counts, edges)``.

    Layout: negative bins (descending magnitude), one CENTER band covering
    ``(-2**lo_exp, +2**lo_exp)`` that holds exact zeros AND both per-side
    underflow counts (their magnitudes genuinely lie inside that interval --
    honest placement, not folding), then positive bins. Per-side OVERFLOW
    and nonfinite counts are deliberately NOT drawn into edge bins (that
    would lie about their magnitude); they ride the scalar series
    (``nonfinite/*`` and the histogram summary) instead.
    """

    magnitude_edges = sketch.descriptor.bucket_edges()
    neg_edges = [-edge for edge in reversed(magnitude_edges)]
    pos_edges = list(magnitude_edges)
    edges = tuple(neg_edges + pos_edges)
    center = (
        sketch.specials.get("zero", 0)
        + sketch.specials.get("pos_underflow", 0)
        + sketch.specials.get("neg_underflow", 0)
    )
    counts = tuple(reversed(sketch.neg_counts)) + (center,) + tuple(sketch.pos_counts)
    return (counts, edges)


def _histogram_summary(spine: SpineResult) -> dict[str, float]:
    """The TB histogram-proto summary fields, from the paired spine."""

    summary: dict[str, float] = {"num": float(spine.count_finite)}
    if spine.finite_min is not None:
        summary["min"] = spine.finite_min
    if spine.finite_max is not None:
        summary["max"] = spine.finite_max
    if spine.sum is not None:
        summary["sum"] = spine.sum
    if spine.sum_squares is not None:
        summary["sum_squares"] = spine.sum_squares
    return summary


def emission_from_block(
    block: CommittedBlock,
    sites: dict[str, SiteRecord],
    grammar: TagGrammar,
) -> StepEmission:
    """Convert one committed C06 block into its full emission view.

    Every observed observation yields its spine scalar series; gradient
    streams additionally stamp nothing here -- the scale basis already rides
    the record (D7) and the run-health series carries the scale itself.
    Non-observed presences yield NO points: missing is never zero, and the
    close report (not a fabricated series) is where skips are disclosed.
    """

    scalars: list[ScalarPoint] = []
    histograms: list[HistogramPoint] = []
    step = block.block.global_step
    for observation in block.observations:
        if observation.presence != "observed" or observation.spine is None:
            continue
        site = sites.get(observation.site_id)
        leaf = site.display_label if site is not None else observation.site_id
        family = STREAM_FAMILIES[observation.stream]
        for statistic, value in spine_scalars(observation.spine).items():
            scalars.append(ScalarPoint(grammar.data(family, statistic, leaf), step, value))
        nonfinite = nonfinite_scalars(observation.spine)
        if any(nonfinite.values()):
            for statistic, value in nonfinite.items():
                scalars.append(ScalarPoint(grammar.data("nonfinite", statistic, leaf), step, value))
        if observation.sketch is not None:
            counts, edges = histogram_points(observation.sketch)
            histograms.append(
                HistogramPoint(
                    tag=grammar.data(family, "hist", leaf),
                    step=step,
                    counts=counts,
                    edges=edges,
                    summary=_histogram_summary(observation.spine),
                )
            )
    truth = block.block
    if truth.scale is not None:
        scalars.append(ScalarPoint(grammar.run_health("amp_scale"), step, float(truth.scale)))
    scalars.append(ScalarPoint(grammar.run_health("last_step"), step, float(step)))
    return StepEmission(
        step=step,
        scalars=tuple(scalars),
        histograms=tuple(histograms),
    )


def architecture_fingerprint(sites: dict[str, SiteRecord]) -> str:
    """Hash the site census so cross-run comparability is checkable (3.5).

    Site labels are deterministic across runs of one architecture but DO
    shift downstream of an inserted/removed op; two runs whose fingerprints
    differ must not be overlaid as if their series named the same sites.
    """

    rows = sorted(
        (site.site_id, site.kind, site.module_path or "", site.param_name or "")
        for site in sites.values()
    )
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=True).encode("ascii"))
    return digest.hexdigest()[:16]


def build_manifest(
    run: RunRecord,
    sites: dict[str, SiteRecord],
    grammar: TagGrammar,
    *,
    grains: dict[str, str] | None = None,
    extra: dict[str, object] | None = None,
) -> TextPoint:
    """Build the versioned series manifest (one ``add_text``, memo 3.5).

    Maps every series leaf to its site identity (module path / param name /
    shape / dtype / grain / structural-key crosslink), and carries the
    architecture fingerprint, histogram descriptor (the versioned edge
    policy: C06 owns it; an edge-policy change starts a NEW series), and the
    emission schema version.
    """

    descriptor = run.descriptor
    payload: dict[str, object] = {
        "kind": "torchlens.trackers.manifest",
        "version": EMISSION_SCHEMA_VERSION,
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": run.run_id,
        "segment_id": run.segment_id,
        "architecture_fingerprint": architecture_fingerprint(sites),
        "edge_policy": {
            "base": descriptor.base,
            "bins_per_octave": descriptor.bins_per_octave,
            "lo_exp": descriptor.lo_exp,
            "hi_exp": descriptor.hi_exp,
            "signed": descriptor.signed,
        },
        "tag_safety_check_version": TAG_SAFETY_CHECK_VERSION,
        "series": [
            {
                "leaf": site.display_label,
                "site_id": site.site_id,
                "grain": (grains or {}).get(site.site_id, site.kind),
                "module_path": site.module_path,
                "param_name": site.param_name,
                "shape": list(site.shape) if site.shape is not None else None,
                "dtype": site.dtype,
                "op_site_key": site.structural_key,
            }
            for site in sites.values()
        ],
    }
    if extra:
        payload.update(extra)
    return TextPoint(
        tag=grammar.meta("manifest"),
        step=0,
        text=json.dumps(payload, ensure_ascii=True, sort_keys=True),
    )


__all__ = [
    "EMISSION_SCHEMA_VERSION",
    "FAMILIES",
    "SPINE_STATISTICS",
    "STREAM_FAMILIES",
    "TAG_SAFETY_CHECK_VERSION",
    "HistogramPoint",
    "ScalarPoint",
    "StepEmission",
    "TagGrammar",
    "TextPoint",
    "architecture_fingerprint",
    "assert_tag_safe",
    "build_manifest",
    "emission_from_block",
    "histogram_points",
    "nonfinite_scalars",
    "spine_scalars",
]
