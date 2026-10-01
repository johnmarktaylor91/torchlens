"""The ONE per-step history artifact schema (explorer D4-D7; lane C06).

Four normalized layers (D4):

- :class:`RunRecord` -- schema version, run/segment identity, resume
  ancestry, fingerprints, package versions, selector plan, histogram
  descriptor, reduction dtypes, cadences, budgets, device, rank/world.
- :class:`SiteRecord` -- stable STRUCTURAL identity; labels are display
  metadata only (D5). Topology drift ends a series and starts a new one --
  disclosure, never re-binding.
- :class:`StepBlockRecord` -- the atomic training coordinate. A block
  commits only when every scheduled observation has an outcome (D6).
- :class:`ObservationRecord` -- (step, site, stream, phase) plus presence,
  spine, sketch, and disclosures. Missing is never zero: a non-observed
  presence token may not carry payloads.

Integer counts merge EXACTLY; floating fields merge with the documented
stable pairwise algorithm and are never called exact (D8's language rule).
Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ._errors import HistorySchemaError
from ._kernels import HistogramDescriptor, HistogramResult, SpineResult, StatKernelError

__tl_layer__ = "L5"

#: Version of this record schema; recorded in every run manifest.
HISTORY_SCHEMA_VERSION = 1

#: The five v1 streams (D7), with lifecycle truth.
STREAMS = (
    "activation",
    "activation_grad",
    "param",
    "param_grad",
    "param_delta",
)

#: Closed presence vocabulary (D6); missing is never zero.
PRESENCE = (
    "observed",
    "not_scheduled",
    "site_absent",
    "unsupported",
    "capture_failed",
    "budget_dropped",
)

#: Closed phase vocabulary. param-grads finalize after accumulation and AMP
#: unscale, before clipping, by default (D7); pre-clip and post-clip are
#: distinct phases; the param-delta boundary is the optimizer step.
PHASES = (
    "forward",
    "backward",
    "pre_clip",
    "post_clip",
    "pre_step",
    "post_step",
    "step",
)

#: Gradient scale truth (D7): unknown never masquerades as either.
GRAD_SCALE_PROVENANCE = ("scaled", "unscaled", "unknown")

#: Step-boundary provenance (D17): the implicit optimizer-hook spelling
#: records unknown for what it cannot determine; explicit is authoritative.
STEP_PROVENANCE = ("explicit", "implicit")

#: Optimizer outcome for one StepBlock: a skipped GradScaler step records
#: ``skipped`` and NO fake update (D7/D11).
OPTIMIZER_STATUS = ("applied", "skipped", "unknown")

#: Tri-state disclosure token for unscale/clip facts the collector cannot
#: prove (implicit mode never guesses).
TRI_STATE = ("yes", "no", "unknown")


def _require_token(field_name: str, value: str, vocabulary: tuple[str, ...]) -> str:
    """Validate one closed-vocabulary token; refuse typed on violation."""

    if value not in vocabulary:
        raise HistorySchemaError(
            f"{field_name}={value!r} is not in the closed vocabulary "
            f"{vocabulary}. History records never carry invented tokens: every "
            "consumer (merge, renderers, sinks) branches on these values.",
            code="history_vocab_invalid",
            field=field_name,
            value=value,
            vocabulary=vocabulary,
            remedy=f"Use one of {vocabulary}.",
        )
    return value


@dataclass(frozen=True)
class RunRecord:
    """Run-level identity and configuration (D4).

    ``segment_index`` / ``parent_segment_id`` carry resume ancestry: a
    restart appends a NEW segment -- the boundary is preserved so a merged
    view can never present a plausible-but-false continuous history.
    """

    run_id: str
    segment_id: str
    segment_index: int = 0
    parent_segment_id: str | None = None
    schema_version: int = HISTORY_SCHEMA_VERSION
    model_fingerprint: str | None = None
    package_versions: Mapping[str, str] = field(default_factory=dict)
    selector_plan: str | None = None
    descriptor: HistogramDescriptor = field(default_factory=HistogramDescriptor)
    reduction_dtypes: Mapping[str, str] = field(default_factory=dict)
    cadences: Mapping[str, int] = field(default_factory=dict)
    budgets: Mapping[str, int] = field(default_factory=dict)
    device: str = "cpu"
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        """Validate identity and cadence geometry."""

        if not self.run_id or not self.segment_id:
            raise HistorySchemaError(
                "RunRecord needs non-empty run_id and segment_id; they key every "
                "site, block, and observation in the artifact.",
                code="history_vocab_invalid",
                field="run_id/segment_id",
                remedy="Pass non-empty run and segment ids.",
            )
        for stream, cadence in self.cadences.items():
            _require_token("cadences key", stream, STREAMS)
            if not isinstance(cadence, int) or cadence < 1:
                raise HistorySchemaError(
                    f"cadence for stream {stream!r} must be a positive int, got "
                    f"{cadence!r}. Cadence is per-STREAM (D21).",
                    code="history_vocab_invalid",
                    field="cadences",
                    value=cadence,
                    remedy="Use integer cadences >= 1 per stream.",
                )


@dataclass(frozen=True)
class SiteRecord:
    """Stable structural site identity (D5); labels are display metadata.

    ``site_id`` is the artifact-local structural key. ``structural_key``
    carries the L1 ``site_key`` spelling when the site was derived from a
    TorchLens capture (label unification consumed); it is nullable because
    Route-A module sites exist without a capture.
    """

    site_id: str
    kind: str
    display_label: str
    module_path: str | None = None
    param_name: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    numel: int | None = None
    structural_key: str | None = None
    first_step: int | None = None
    ended_step: int | None = None
    ended_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the closed site kinds."""

        _require_token("kind", self.kind, ("module", "param"))

    def ended(self, step: int, reason: str) -> SiteRecord:
        """Return a copy marking this series ended at ``step`` (D5 drift).

        Drift ends a series and starts a new one -- disclosure, not refusal
        and not re-binding.
        """

        return replace(self, ended_step=step, ended_reason=reason)


@dataclass(frozen=True)
class StepBlockRecord:
    """The atomic training coordinate (D4/D6/D17)."""

    segment_id: str
    global_step: int
    provenance: str
    optimizer_status: str = "unknown"
    scale: float | None = None
    unscaled: str = "unknown"
    clipped: str = "unknown"
    micro_batches: int | None = None
    train_mode: bool | None = None

    def __post_init__(self) -> None:
        """Validate the closed step vocabularies."""

        _require_token("provenance", self.provenance, STEP_PROVENANCE)
        _require_token("optimizer_status", self.optimizer_status, OPTIMIZER_STATUS)
        _require_token("unscaled", self.unscaled, TRI_STATE)
        _require_token("clipped", self.clipped, TRI_STATE)


@dataclass(frozen=True)
class ObservationRecord:
    """(step, site, stream, phase) plus presence, payloads, disclosures.

    ``presence != 'observed'`` carries NO payloads: a missing value must
    never read as zero (D6). Disclosures ride ``grad_scale`` (mandatory for
    gradient streams), ``reduction_dtype``, ``estimated`` (the D11 update
    channel), and ``sample_size``.
    """

    global_step: int
    site_id: str
    stream: str
    phase: str
    presence: str
    spine: SpineResult | None = None
    sketch: HistogramResult | None = None
    grad_scale: str | None = None
    estimated: bool = False
    sample_size: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate vocabulary, payload-presence coherence, and grad truth."""

        _require_token("stream", self.stream, STREAMS)
        _require_token("phase", self.phase, PHASES)
        _require_token("presence", self.presence, PRESENCE)
        if self.grad_scale is not None:
            _require_token("grad_scale", self.grad_scale, GRAD_SCALE_PROVENANCE)
        if self.presence == "observed":
            if self.spine is None:
                raise HistorySchemaError(
                    "An 'observed' observation must carry a spine payload; the "
                    "spine is always-on (D8).",
                    code="history_vocab_invalid",
                    field="spine",
                    remedy="Attach the SpineResult, or use a non-observed presence token.",
                )
            if self.stream in ("activation_grad", "param_grad") and self.grad_scale is None:
                raise HistorySchemaError(
                    "Gradient-stream observations must state grad_scale "
                    "('scaled' | 'unscaled' | 'unknown'); unknown never "
                    "masquerades (D7), and without the stamp a dashboard cannot "
                    "label its own y-axis.",
                    code="history_vocab_invalid",
                    field="grad_scale",
                    remedy="Stamp grad_scale explicitly; 'unknown' is a legal value.",
                )
        elif self.spine is not None or self.sketch is not None:
            raise HistorySchemaError(
                f"presence={self.presence!r} may not carry payloads: a "
                "non-observed observation with numbers would let missing read "
                "as data (D6).",
                code="history_vocab_invalid",
                field="presence",
                remedy="Drop the payloads or mark the observation 'observed'.",
            )

    @property
    def identity(self) -> tuple[int, str, str, str]:
        """The merge identity: (step, site, stream, phase)."""

        return (self.global_step, self.site_id, self.stream, self.phase)


def merge_spine_results(a: SpineResult, b: SpineResult) -> SpineResult:
    """Merge two finalized spines; integer counts EXACT, floats stable.

    Moments merge with Chan's parallel algorithm; sums add pairwise. The
    result's floating fields are documented approximate across merges (D8).
    """

    if a.count_finite == 0:
        base_floats = b
    elif b.count_finite == 0:
        base_floats = a
    else:
        base_floats = None
    n_a, n_b = a.count_finite, b.count_finite
    total_finite = n_a + n_b
    if base_floats is not None:
        mean, m2 = base_floats.mean, base_floats.m2
        finite_min, finite_max = base_floats.finite_min, base_floats.finite_max
        finite_absmax = base_floats.finite_absmax
        sum_, sum_squares, sum_abs = (
            base_floats.sum,
            base_floats.sum_squares,
            base_floats.sum_abs,
        )
    else:
        delta = (b.mean or 0.0) - (a.mean or 0.0)
        mean = (a.mean or 0.0) + delta * (n_b / total_finite)
        m2 = (a.m2 or 0.0) + (b.m2 or 0.0) + delta * delta * (n_a * n_b / total_finite)
        finite_min = min(a.finite_min, b.finite_min)  # type: ignore[type-var]
        finite_max = max(a.finite_max, b.finite_max)  # type: ignore[type-var]
        finite_absmax = max(a.finite_absmax, b.finite_absmax)  # type: ignore[type-var]
        sum_ = (a.sum or 0.0) + (b.sum or 0.0)
        sum_squares = (a.sum_squares or 0.0) + (b.sum_squares or 0.0)
        sum_abs = (a.sum_abs or 0.0) + (b.sum_abs or 0.0)
    reduction_dtype = (
        "float64" if "float64" in (a.reduction_dtype, b.reduction_dtype) else a.reduction_dtype
    )
    return SpineResult(
        count_total=a.count_total + b.count_total,
        count_finite=total_finite,
        count_zero=a.count_zero + b.count_zero,
        count_negative=a.count_negative + b.count_negative,
        count_nan=a.count_nan + b.count_nan,
        count_posinf=a.count_posinf + b.count_posinf,
        count_neginf=a.count_neginf + b.count_neginf,
        finite_min=finite_min,
        finite_max=finite_max,
        finite_absmax=finite_absmax,
        sum=sum_,
        sum_squares=sum_squares,
        sum_abs=sum_abs,
        mean=mean,
        m2=m2,
        reduction_dtype=reduction_dtype,
    )


def merge_histogram_results(a: HistogramResult, b: HistogramResult) -> HistogramResult:
    """Merge two finalized histograms; integer adds, EXACT; no rebinning."""

    if a.descriptor != b.descriptor:
        raise StatKernelError(
            "Cannot merge histograms across unequal grid descriptors "
            f"({a.descriptor!r} vs {b.descriptor!r}); rebinning is banned (D9).",
            code="sketch_descriptor_mismatch",
            mine=repr(a.descriptor),
            theirs=repr(b.descriptor),
            remedy="Re-accumulate both populations on one shared descriptor.",
        )
    specials = {key: a.specials.get(key, 0) + b.specials.get(key, 0) for key in a.specials}
    for key, value in b.specials.items():
        if key not in specials:
            specials[key] = value
    return HistogramResult(
        descriptor=a.descriptor,
        pos_counts=tuple(x + y for x, y in zip(a.pos_counts, b.pos_counts, strict=True)),
        neg_counts=tuple(x + y for x, y in zip(a.neg_counts, b.neg_counts, strict=True)),
        specials=specials,
    )


def merge_observations(a: ObservationRecord, b: ObservationRecord) -> ObservationRecord:
    """Merge two observations of the SAME identity (the rank-merge unit).

    Identity is (step, site, stream, phase); anything else refuses typed.
    Presence merges conservatively: observed+observed merges payloads
    exactly on integer counts; observed+anything-else keeps the observed
    side and records nothing invented; two non-observed tokens must agree.
    """

    if a.identity != b.identity:
        raise HistorySchemaError(
            f"Cannot merge observations with different identities {a.identity} "
            f"vs {b.identity}; merge is within-a-series only (D12): across "
            "steps, micro-batches, ranks, runs -- never across sites.",
            code="history_merge_incompatible",
            mine=a.identity,
            theirs=b.identity,
            remedy="Merge only observations sharing (step, site, stream, phase).",
        )
    if a.presence == "observed" and b.presence == "observed":
        if (a.grad_scale or b.grad_scale) and a.grad_scale != b.grad_scale:
            raise HistorySchemaError(
                f"Cannot merge observations with conflicting grad_scale stamps "
                f"({a.grad_scale!r} vs {b.grad_scale!r}); a merged gradient "
                "number with a mixed scale basis is unlabelable (D7).",
                code="history_merge_incompatible",
                mine=a.grad_scale,
                theirs=b.grad_scale,
                remedy="Unscale (or keep scaled) consistently across ranks before merging.",
            )
        spine = merge_spine_results(a.spine, b.spine)  # type: ignore[arg-type]
        sketch = None
        if a.sketch is not None and b.sketch is not None:
            sketch = merge_histogram_results(a.sketch, b.sketch)
        elif a.sketch is not None or b.sketch is not None:
            raise HistorySchemaError(
                "Cannot merge an observation with a sketch into one without: "
                "the merged sketch would silently cover only part of the "
                "population it claims.",
                code="history_merge_incompatible",
                remedy="Schedule the sketch tier identically across ranks.",
            )
        return replace(
            a,
            spine=spine,
            sketch=sketch,
            estimated=a.estimated or b.estimated,
            sample_size=(
                None
                if a.sample_size is None or b.sample_size is None
                else a.sample_size + b.sample_size
            ),
        )
    if a.presence == "observed":
        return a
    if b.presence == "observed":
        return b
    if a.presence != b.presence:
        raise HistorySchemaError(
            f"Cannot merge conflicting non-observed presences {a.presence!r} vs "
            f"{b.presence!r}; the merged token would erase one side's reason.",
            code="history_merge_incompatible",
            mine=a.presence,
            theirs=b.presence,
            remedy="Keep rank-local rows separate when their outcomes differ.",
        )
    return a


def validate_step_order(
    prior_step: int | None,
    step: int,
    *,
    same_segment: bool,
) -> None:
    """Refuse duplicate/decreasing explicit steps within one segment (D6).

    A resume declares a NEW segment; within a segment the step axis is
    strictly increasing.
    """

    if same_segment and prior_step is not None and step <= prior_step:
        raise HistorySchemaError(
            f"global_step {step} is not greater than the previous step "
            f"{prior_step} in the same segment. Duplicate or decreasing steps "
            "refuse without a declared new segment (D6): a silently reused "
            "coordinate corrupts every series that shares the axis.",
            code="history_step_regression",
            prior_step=prior_step,
            step=step,
            remedy=(
                "Pass strictly increasing global steps, or declare a new "
                "segment (resume) to start a new axis span."
            ),
        )


__all__ = [
    "GRAD_SCALE_PROVENANCE",
    "HISTORY_SCHEMA_VERSION",
    "OPTIMIZER_STATUS",
    "PHASES",
    "PRESENCE",
    "STEP_PROVENANCE",
    "STREAMS",
    "TRI_STATE",
    "ObservationRecord",
    "RunRecord",
    "SiteRecord",
    "StepBlockRecord",
    "merge_histogram_results",
    "merge_observations",
    "merge_spine_results",
    "validate_step_order",
]
