"""Stochastic/population edit verbs and the derived-seed law (F02, edits memo).

The stochastic donor family causal scrubbing needs, built on ONE primitive
(edits memo D1): :func:`sample_from` returns an immutable donor-sampling PLAN;
``tl.patch_from(plan)`` applies it; every other stochastic verb is a named
lowering onto the same substrate, and every mean is a deterministic reduction
over the same population noun -- never a draw in disguise, never a fake seed.
Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.

THE HONEST DRAW LAW (D4-D7), the panel's centerpiece. Both shipped behaviors
were measured defective: an unseeded stochastic edit is silently re-rolled by
later unrelated edits, and a seeded one draws the IDENTICAL sample at every
pass, step, and site. The fix is one stateless mechanism:

- ``seed=`` is a REQUIRED int or the literal ``"auto"``; ``None`` never
  reaches a hook (D4).
- ``"auto"`` canonicalizes ONCE per capture into
  ``base_seed = H(trace.random_seed, rule_identity, population_identity)``
  (D5 with the DIS-4 adjudication: the identity component is PERSISTED helper
  identity, never a transient construction ordinal) -- never the ambient
  global RNG, and recorded.
- Every logical firing derives its own seed statelessly (D6)::

      derived_seed = H(base_seed, population_identity, donor_group_id,
                       logical_firing_coordinate, target_row, draw_index,
                       composed_leaf_path)
      logical_firing_coordinate = (site*, pass_index, generation_step)
          * omitted for a shared donor group -- that omission IS donor sharing

- ``donor_group_id`` is itself a CONTENT DIGEST of the plan's declared inputs
  (population identity, seed spelling, share_draw, agreement condition),
  never a transient construction artifact (the D5 anti-transient rule scaled
  down to the group key): rerunning the same declared experiment -- fresh
  process, fresh objects, same declarations -- reproduces every draw, which
  is what the stateless law MEANS. Identical declarations are the same
  experiment by extensionality; deliberate cross-clause donor sharing stays
  plan OBJECT reuse (D8), and the selection-batch normalizer disambiguates
  distinct equal-content plan objects inside one transaction with
  deterministic clause-order suffixes (:func:`assign_batch_donor_groups`),
  so accidental kwargs-coincidence sharing within a batch cannot happen.

- ``share_draw=`` is a named, defaulted, RECORDED choice (D7):
  ``"per_firing"`` (default -- each firing is a distinct node in the
  treeified scrub graph), ``"per_rule"`` (one draw for every firing --
  today's accidental behavior, now a choice), ``"per_group"`` (donor sharing
  across sites via one plan identity). The shipped ``noise`` /
  ``scramble_elements`` keep their frozen ``per_rule`` behavior forever.

Every firing routes its audit through C03's ONE FireRecord builder: realized
draws land as ``ledger_notes`` (lifted into ``FireRecord.determinism_note`` by
``build_fire_record``) plus a structured session-side record served by
:func:`sampling_records`. No new persisted field is introduced (the C07 field
census is frozen; a structured ``sampling`` FireRecord field would need its
own adjudicated ``sprint/field_intent.tsv`` row first).

Axis law (D24/D25): axes are NEVER inferred from rank. The ladder is explicit
``axis=`` > facet/recipe-declared semantic role > registered model/site
metadata > typed ``axis_semantics_unknown`` refusal. No axis-role evidence
source is registered in v1, so named tokens (``"batch"``/``"position"``/
``"feature"``) refuse with the ladder in the message until the positions lane
lands curated axis facts.

Backend law (D36): the family executes on torch only. Construction, repr,
equality, and save-identity work everywhere; execution on a preview backend
refuses at that backend's helper-resolution preflight (helper name not in the
backend's supported table), BEFORE any forward or mutation.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch

from .._errors import InvalidArgumentError
from ..errors._base import TorchLensWarning
from .hooks import HookContext
from .population import OneDatum, PerRowDatums, Reference
from .types import HelperSpec

__all__ = [
    "SamplingPlan",
    "mean_fill",
    "mean_from",
    "permute_batch",
    "resample_rows_from",
    "sample_from",
    "sampling_records",
    "set_direction_mean",
]

_SHARE_DRAW_TOKENS = ("per_firing", "per_rule", "per_group")

#: Closed named-axis tokens of the ``over=`` vocabulary (D17). They resolve
#: through the axis ladder and refuse until axis-role evidence exists.
_NAMED_AXIS_TOKENS = ("batch", "position", "feature")


# ----------------------------------------------------------------------
# hashing + seed law
# ----------------------------------------------------------------------
def _hash_seed(*parts: Any) -> int:
    """Derive a stable 63-bit seed from string-rendered parts (sha256)."""

    payload = "\x1f".join(repr(part) for part in parts)
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _mint_donor_group_id(
    population_identity: str,
    seed: int | str,
    share_draw: str,
    matching: Any,
    agree_on: Callable[[Any], Any] | None,
) -> str:
    """Mint the plan's donor group key as a CONTENT DIGEST of declared inputs.

    Deterministic by design (D5's anti-transient rule applied to the group
    key): the same declared experiment reconstructed in a fresh process mints
    the same key, so the derived-seed law reproduces every draw on a rerun.
    ``agree_on`` callables have no stable content hash, so they contribute a
    presence marker only; the realized eligible class already reflects the
    key function at fire time. Distinct equal-content plan objects inside ONE
    selection-batch transaction are disambiguated by
    :func:`assign_batch_donor_groups` (D8), never at construction.
    """

    payload = "\x1f".join(
        (
            "donor_group_v1",
            population_identity,
            repr(seed),
            share_draw,
            repr(matching),
            "keyed" if agree_on is not None else "unkeyed",
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _require_seed(seed: Any, verb: str) -> int | str:
    """Enforce the D4 seed law at construction: an int or the literal ``"auto"``.

    Parameters
    ----------
    seed:
        User-supplied seed spelling.
    verb:
        Verb name for the refusal message.

    Returns
    -------
    int | str
        The validated seed spelling.

    Raises
    ------
    InvalidArgumentError
        ``sampling_seed_required`` -- the user must acknowledge stochasticity
        at the call site; ``None`` never reaches a hook.
    """

    if isinstance(seed, bool) or not (isinstance(seed, int) or seed == "auto"):
        raise InvalidArgumentError(
            f"{verb} requires seed=<int> or seed='auto'; received {seed!r}. An "
            "implicit global-RNG draw is silently re-rolled by later unrelated "
            "edits, so unacknowledged stochasticity never reaches a hook",
            code="sampling_seed_required",
            remedy="pass an explicit integer seed, or seed='auto' to derive one "
            "from the capture's recorded trace.random_seed",
            argument="seed",
        )
    return seed


def _base_seed(
    seed: int | str,
    *,
    rule_identity: str,
    population_identity: str,
    hook: HookContext,
    verb: str,
) -> int:
    """Resolve the base seed for one firing (D5, DIS-4 spelling).

    An explicit int is its own base. ``"auto"`` derives from the capture's
    recorded ``trace.random_seed`` (threaded through ``run_ctx`` by the
    live/replay doors), the helper's persisted identity, and the population
    identity -- never the ambient RNG, never a transient construction ordinal.
    """

    if isinstance(seed, int):
        return seed
    trace_seed = hook.run_ctx.get("trace_random_seed")
    if trace_seed is None:
        raise InvalidArgumentError(
            f"{verb} seed='auto' needs the capture's recorded trace.random_seed, "
            "and this door's run context does not carry one",
            code="sampling_trace_seed_unavailable",
            remedy="pass an explicit integer seed=, or run the edit on a door "
            "that carries the capture seed (live capture or replay on a Trace)",
            argument="seed",
        )
    return _hash_seed("auto_base", trace_seed, rule_identity, population_identity)


def _firing_coordinate(hook: HookContext, share_draw: str) -> tuple[Any, ...]:
    """Build the LOGICAL firing coordinate (D6) for one hook fire.

    ``per_firing``: ``(site, pass_index, generation_step)`` where ``site`` is
    the L1 site key when minted, else the pass-qualified label. ``per_group``
    omits the site component -- that omission IS donor sharing across sites.
    ``per_rule`` is the empty coordinate (one draw for every firing).
    The coordinate is logical, never a physical fire counter, so re-firing is
    idempotent and checkpoint recompute reuses the same draw.
    """

    if share_draw == "per_rule":
        return ()
    layer_log = hook.layer_log or {}
    pass_index = layer_log.get("pass_index")
    generation_step = hook.run_ctx.get("generation_step")
    if share_draw == "per_group":
        return (pass_index, generation_step)
    site = layer_log.get("site_key") or layer_log.get("label") or layer_log.get("layer_label")
    return (site, pass_index, generation_step)


def _derived_seed(  # noqa: PLR0913 -- the D6 formula's seven NAMED components ARE the spec; packing them would hide the law
    base_seed: int,
    *,
    population_identity: str,
    donor_group_id: str,
    coordinate: tuple[Any, ...],
    target_row: int | None = None,
    draw_index: int = 0,
    leaf_path: tuple[int, ...] = (),
) -> int:
    """Derive one firing's stateless seed (the D6 centerpiece formula)."""

    return _hash_seed(
        base_seed,
        population_identity,
        donor_group_id,
        coordinate,
        target_row,
        draw_index,
        leaf_path,
    )


def _leaf_path(hook: HookContext) -> tuple[int, ...]:
    """Read the composed-leaf path threaded by ``tl.compose`` (D6/D20)."""

    path = hook.ctx.get("compose_leaf_path", ())
    return tuple(path) if isinstance(path, (tuple, list)) else ()


def _record_sampling(hook: HookContext, verb: str, record: dict[str, Any]) -> None:
    """Land one realized-draw record on the session and the note stream.

    The structured dict rides ``run_ctx["sampling_records"]`` (session-time;
    served by :func:`sampling_records`); the compact rendering is enqueued as
    a ``ledger_notes`` entry so C03's ONE FireRecord builder lifts it into the
    persisted ``FireRecord.determinism_note`` -- no parallel audit path.
    """

    hook.run_ctx.setdefault("sampling_records", []).append(dict(record))
    draw = record.get("donor_ids", record.get("permutation"))
    draw_repr = repr(list(draw)[:16]) if isinstance(draw, (list, tuple)) else repr(draw)
    note = (
        f"sampling[{verb}] base_seed={record.get('base_seed')} "
        f"derived_seed={record.get('derived_seed')} share={record.get('share_draw')} "
        f"coordinate={record.get('logical_firing_coordinate')!r} "
        f"eligible={record.get('eligible_count')}/{record.get('population_count')} "
        f"draw={draw_repr} draw_digest={record.get('realized_draw_digest')} "
        f"population={record.get('population_identity')}"
    )
    hook.run_ctx.setdefault("ledger_notes", []).append(note)
    state_history = hook.run_ctx.get("state_history")
    if isinstance(state_history, list):
        state_history.append(note)


def _draw_digest(indices: Sequence[int]) -> str:
    """Hash the realized index vector (cheap; never the tensor payload)."""

    return hashlib.sha256(",".join(str(i) for i in indices).encode()).hexdigest()[:16]


def sampling_records(trace: Any) -> tuple[dict[str, Any], ...]:
    """Return the session-time realized-draw records of a trace's last run.

    Session-only by design: the persisted carriers are the FireRecord's
    ``determinism_note`` (via the ONE builder's note lift) and the helper's
    KEEP-persisted identity kwargs; the structured per-draw record family is
    a declared C07-amendment follow-up, never smuggled through free kwargs.
    """

    run_ctx = getattr(trace, "last_run", None)
    if not isinstance(run_ctx, dict):
        return ()
    return tuple(run_ctx.get("sampling_records", ()))


# ----------------------------------------------------------------------
# axis ladder (D24/D25) + the over= vocabulary (D17)
# ----------------------------------------------------------------------
def _resolve_over_axes(
    over: Any, ndim: int | None, *, verb: str
) -> Literal["all"] | tuple[int, ...]:
    """Resolve one ``over=`` spelling through the closed vocabulary + ladder.

    Parameters
    ----------
    over:
        ``"all"``, ``"self"`` (warns; maps to ``"all"``), an explicit int
        axis, a tuple of ints, or a named token (refuses until axis-role
        evidence exists).
    ndim:
        Concrete rank for range validation; ``None`` runs the vocabulary
        check only (construction time, before any value exists).
    verb:
        Verb name for refusal messages.

    Returns
    -------
    "all" | tuple[int, ...]
        The canonical reduction spelling (axes normalized when ``ndim`` is
        known).
    """

    if over == "self":
        warnings.warn(
            TorchLensWarning(
                f"{verb}(over='self') is the legacy spelling of the explicit "
                "global-scalar mean; it maps to over='all'. Remedy: spell "
                "over='all' directly",
                code="mean_over_self_alias",
            ),
            stacklevel=3,
        )
        return "all"
    if over == "all":
        return "all"
    if isinstance(over, str) and over in _NAMED_AXIS_TOKENS:
        raise InvalidArgumentError(
            f"{verb}(over={over!r}) names a semantic axis, and no axis-role "
            "evidence is registered for this site: the axis ladder is explicit "
            "axis integers > facet/recipe-declared semantic role > registered "
            "model/site metadata > this refusal. An axis is never inferred "
            "from rank (on stock nn.TransformerEncoderLayer axis 0 is TIME)",
            code="axis_semantics_unknown",
            remedy="pass the explicit integer axis (or tuple of axes) for this site's geometry",
            argument="over",
        )
    axes: tuple[Any, ...]
    if isinstance(over, bool):
        axes = (over,)  # falls through to the type refusal below
    elif isinstance(over, int):
        axes = (over,)
    elif isinstance(over, (tuple, list)) and over:
        axes = tuple(over)
    else:
        axes = ()
    if not axes or not all(isinstance(axis, int) and not isinstance(axis, bool) for axis in axes):
        supported = "'all', an int axis, a tuple of int axes, " + ", ".join(
            repr(token) for token in _NAMED_AXIS_TOKENS
        )
        raise InvalidArgumentError(
            f"{verb}(over={over!r}) is outside the closed over= vocabulary "
            f"({supported}); an accepted-but-ignored token is the audit-only-"
            "label defect this gate exists to kill",
            code="intervention_over_invalid",
            remedy="pass over='all' for the explicit global scalar, or explicit integer axes",
            argument="over",
        )
    if ndim is None:
        return tuple(int(axis) for axis in axes)
    return _normalize_over_axes(axes, ndim, verb)


def _normalize_over_axes(axes: tuple[Any, ...], ndim: int, verb: str) -> tuple[int, ...]:
    """Range-check and canonicalize explicit ``over=`` axes against a rank."""

    normalized: list[int] = []
    for axis in axes:
        if not -ndim <= axis < ndim:
            raise InvalidArgumentError(
                f"{verb} axis {axis} is out of range for rank {ndim}",
                code="sampling_geometry_mismatch",
                remedy=f"pass axes in [-{ndim}, {ndim - 1}]",
                argument="over",
            )
        normalized.append(axis % ndim)
    return tuple(dict.fromkeys(normalized))


def _require_explicit_axis(axis: Any, verb: str) -> int:
    """Enforce the D24/D25 axis law at construction: explicit int or refuse."""

    if isinstance(axis, int) and not isinstance(axis, bool):
        return axis
    raise InvalidArgumentError(
        f"{verb} refuses without a derived-or-explicit batch axis (received "
        f"{axis!r}): on stock PyTorch (nn.TransformerEncoderLayer, "
        "batch_first=False by default) axis 0 is TIME, so any positional "
        "default is a silent wrong-axis edit. No axis-role evidence source is "
        "registered in v1, so the ladder ends here",
        code="axis_semantics_unknown",
        remedy=f"pass the explicit batch axis, e.g. {verb.split('(')[0]}(..., axis=0) "
        "for batch-first activations or axis=1 for (T, B, E) layouts",
        argument="axis",
    )


def _normalize_fire_axis(axis: int, out: torch.Tensor, verb: str) -> int:
    """Validate + normalize an explicit axis against the fire-time rank."""

    if not -out.ndim <= axis < out.ndim:
        raise InvalidArgumentError(
            f"{verb} axis {axis} is out of range for the site output of rank "
            f"{out.ndim} (shape {tuple(out.shape)!r})",
            code="sampling_geometry_mismatch",
            remedy=f"pass an axis in [-{out.ndim}, {out.ndim - 1}] for this site",
            argument="axis",
        )
    return axis % out.ndim


# ----------------------------------------------------------------------
# the sampling PLAN (D1, D8) and its patch application
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingPlan:
    """Immutable donor-sampling plan (the ONE stochastic primitive's noun).

    Lazy by definition, eagerly inspectable: the plan carries the population,
    the agreement condition, the seed spelling, and the recorded draw-sharing
    policy; donor CHOICE happens per logical firing under the derived-seed
    law. ``donor_group_id`` is a CONTENT DIGEST of the declared inputs, so
    reconstructing the same declaration reproduces the same draws (the
    stateless law's whole point); deliberate donor sharing across clauses is
    plan OBJECT reuse (D8), and the selection-batch normalizer
    (:func:`assign_batch_donor_groups`) suffixes distinct equal-content plan
    objects inside one transaction so kwargs coincidence never shares a group.
    """

    population: Reference
    agree_on: Callable[[Any], Any] | None
    matching: Any
    seed: int | str
    share_draw: str
    strict: bool
    donor_group_id: str

    @property
    def rule_identity(self) -> str:
        """Persisted identity string entering the auto-seed derivation (DIS-4)."""

        return (
            f"sample_from|{self.population.population_identity}|"
            f"{self.share_draw}|{self.donor_group_id}"
        )

    def identity_kwargs(self) -> dict[str, Any]:
        """JSON-scalar identity facts persisted on the helper spec (D14).

        Identity travels, payloads never: the population payload itself rides
        the closure only (an ``opaque_audit`` helper), while these scalars
        survive save/load so a saved artifact still names the experiment.
        """

        return {
            "seed": self.seed,
            "share_draw": self.share_draw,
            "donor_group_id": self.donor_group_id,
            "population_identity": self.population.population_identity,
            "population_count": len(self.population),
            "population_digest_kind": self.population.digest_kind,
            "population_origin": self.population.origin,
        }


def _matching_value(matching: Any, verb: str, *, allow_per_row: bool) -> Any:
    """Unwrap a ``matching=`` carrier (OneDatum default for bare values).

    Bare lists/tuples refuse as ambiguous: they could mean ONE sequence-valued
    datum or per-row datums, and guessing recreates the false-friend class.
    """

    if isinstance(matching, PerRowDatums):
        if not allow_per_row:
            raise InvalidArgumentError(
                f"{verb} applies ONE coherent donor per event, so matching= "
                "takes ONE subject datum; per-row donors are the different "
                "verb (resample_rows_from)",
                code="population_matching_invalid",
                remedy="pass matching=OneDatum(value), or switch to "
                "resample_rows_from for per-row draws",
                argument="matching",
            )
        return matching
    if isinstance(matching, OneDatum):
        return matching.value
    if isinstance(matching, (list, tuple)):
        raise InvalidArgumentError(
            f"{verb} received a bare {type(matching).__name__} for matching=; a "
            "sequence is ambiguous between ONE sequence-valued datum and "
            "per-row datums, and a guessed reading is a silent wrong experiment",
            code="population_matching_invalid",
            remedy="wrap it explicitly: OneDatum([...]) for one datum, "
            "PerRowDatums([...]) for one datum per batch row",
            argument="matching",
        )
    return matching


def _check_agreement_pair(agree_on: Any, matching: Any, verb: str) -> None:
    """Refuse half an agreement condition (both-or-neither; edits memo s4)."""

    if (agree_on is None) != (matching is None):
        missing = "matching" if matching is None else "agree_on"
        raise InvalidArgumentError(
            f"{verb} takes agree_on= and matching= together: the key function "
            f"and the subject datum are two halves of one agreement condition "
            f"({missing}= is missing)",
            code="population_matching_invalid",
            remedy="pass both agree_on= and matching=, or neither",
            argument=missing,
        )


def _disclose_class_size(
    eligible_count: int, *, strict: bool, verb: str, hook: HookContext | None
) -> None:
    """Apply the D15 class-size-one rule: disclose by default, refuse strict.

    A conditioned "resample" over a class of one is a fixed patch wearing a
    sampler's name -- a reviewer must be able to SEE that.
    """

    if eligible_count > 1:
        return
    if strict:
        raise InvalidArgumentError(
            f"{verb} resolved an agreement class of size 1 under strict=True: a "
            "single-donor 'sample' is a deterministic patch wearing a sampler's "
            "name",
            code="sampling_agreement_class_too_small",
            remedy="widen the agreement condition, add population members, or "
            "drop strict= to proceed with the disclosed single-donor patch",
            argument="strict",
        )
    warnings.warn(
        TorchLensWarning(
            f"{verb} resolved an agreement class of size 1: every draw returns "
            "the same donor, so this firing is deterministic. Remedy: widen the "
            "agreement condition or add population members (strict=True refuses "
            "instead)",
            code="sampling_agreement_class_singleton",
        ),
        stacklevel=2,
    )


def sample_from(  # noqa: PLR0913 -- spec'd public signature (edits memo D1/D7/D15): the six declared inputs ARE the plan
    population: Reference,
    *,
    agree_on: Callable[[Any], Any] | None = None,
    matching: Any = None,
    seed: int | str,
    share_draw: str = "per_firing",
    strict: bool = False,
) -> SamplingPlan:
    """Build the immutable donor-sampling plan (the ONE stochastic primitive).

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.

    Parameters
    ----------
    population:
        The prepared population noun (:func:`torchlens.intervention.population.reference`).
    agree_on:
        Optional datum -> key callable; donors are drawn only from the
        agreement class matching the subject's key (causal scrubbing's
        agreement-conditioned resample).
    matching:
        The subject's datum: a bare value or ``OneDatum`` for whole-event
        donors (``PerRowDatums`` belongs to ``resample_rows_from``).
    seed:
        REQUIRED int or ``"auto"`` (the D4 seed law).
    share_draw:
        Recorded draw-sharing policy: ``"per_firing"`` (default),
        ``"per_rule"``, or ``"per_group"`` (D7).
    strict:
        Refuse (rather than disclose) agreement classes of size 1 (D15).

    Returns
    -------
    SamplingPlan
        The immutable plan; apply it with ``tl.patch_from(plan)``.
    """

    if not isinstance(population, Reference):
        raise InvalidArgumentError(
            f"sample_from takes the population noun, received {type(population).__name__}",
            code="population_source_invalid",
            remedy='build the population first: ref = reference(members, origin="...", data=...)',
            argument="population",
        )
    _check_agreement_pair(agree_on, matching, "sample_from")
    validated_seed = _require_seed(seed, "sample_from")
    if share_draw not in _SHARE_DRAW_TOKENS:
        raise InvalidArgumentError(
            f"sample_from share_draw={share_draw!r} is outside the closed "
            f"vocabulary {_SHARE_DRAW_TOKENS!r}",
            code="sampling_share_draw_invalid",
            remedy="pass share_draw='per_firing' (independent reproducible "
            "draws), 'per_rule' (one draw everywhere), or 'per_group' (donor "
            "sharing across sites)",
            argument="share_draw",
        )
    matching_value = (
        _matching_value(matching, "sample_from", allow_per_row=False)
        if matching is not None
        else None
    )
    return SamplingPlan(
        population=population,
        agree_on=agree_on,
        matching=matching_value,
        seed=validated_seed,
        share_draw=share_draw,
        strict=strict,
        donor_group_id=_mint_donor_group_id(
            population.population_identity, validated_seed, share_draw, matching_value, agree_on
        ),
    )


def _resolve_trace_donor(member: Any, hook: HookContext, out: torch.Tensor) -> torch.Tensor:
    """Resolve one trace member's donor value at the CURRENT site, pass-qualified.

    Reuses the shipped pass-qualified donor resolution (A04: a bare label on a
    multi-pass donor refuses typed; the historical bare lookup silently
    returned the LAST pass).
    """

    from .helpers import _resolve_patch_donor_site

    site_label, source_site = _resolve_patch_donor_site(member, hook.layer_log)
    value = source_site.out
    if not isinstance(value, torch.Tensor):
        raise InvalidArgumentError(
            f"population trace member holds no tensor value at {site_label!r}",
            code="sampling_geometry_mismatch",
            remedy="capture the donor traces with the site saved (save= covering it)",
            argument="population",
        )
    return value


def _prove_event_geometry(
    donor: torch.Tensor, out: torch.Tensor, *, verb: str, axis: int | None = None
) -> None:
    """Prove the D9 event geometry: whole-site match, or row match minus axis.

    Anything else refuses printing BOTH shapes and the axis proof -- there is
    no elementwise fallback (that is ``scramble_elements``, and keeping the
    line bright is the point).
    """

    if axis is None:
        if tuple(donor.shape) == tuple(out.shape):
            return
        raise InvalidArgumentError(
            f"{verb} whole-site donor shape {tuple(donor.shape)!r} does not "
            f"match the target shape {tuple(out.shape)!r}; a donor event must "
            "match the site exactly (no elementwise fallback -- that is "
            "scramble_elements, deliberately a different verb)",
            code="sampling_geometry_mismatch",
            remedy="build the population from values captured at this site "
            "geometry, or use resample_rows_from for row-shaped donors",
            argument="population",
        )
    row_shape = tuple(dim for index, dim in enumerate(out.shape) if index != axis)
    if tuple(donor.shape) == row_shape:
        return
    raise InvalidArgumentError(
        f"{verb} row donor shape {tuple(donor.shape)!r} does not match the "
        f"target row shape {row_shape!r} (target {tuple(out.shape)!r} minus "
        f"axis {axis}, explicit)",
        code="sampling_geometry_mismatch",
        remedy="stack donors row-shaped for this site, or fix axis= to the site's real batch axis",
        argument="population",
    )


def plan_patch_helper(plan: SamplingPlan) -> HelperSpec:
    """Lower ``tl.patch_from(plan)`` onto the sampling substrate.

    One coherent donor per logical firing event (D1): the derived-seed law
    picks one eligible member; tensor members must match the site geometry
    exactly (D9); trace members resolve pass-qualified at the firing site.
    Identity persists, payloads never (D14: ``opaque_audit``).
    """

    from .helpers import _helper_spec

    ref = plan.population

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime donor-patching hook for this plan."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Replace the site value with one sampled coherent donor."""

            eligible = ref.eligible_indices(plan.agree_on, plan.matching)
            _disclose_class_size(
                len(eligible), strict=plan.strict, verb="patch_from(plan)", hook=hook
            )
            base = _base_seed(
                plan.seed,
                rule_identity=plan.rule_identity,
                population_identity=ref.population_identity,
                hook=hook,
                verb="patch_from(plan)",
            )
            coordinate = _firing_coordinate(hook, plan.share_draw)
            derived = _derived_seed(
                base,
                population_identity=ref.population_identity,
                donor_group_id=plan.donor_group_id,
                coordinate=coordinate,
                leaf_path=_leaf_path(hook),
            )
            generator = torch.Generator()
            generator.manual_seed(derived)
            position = int(torch.randint(len(eligible), (1,), generator=generator).item())
            member_index = eligible[position]
            member = ref.members[member_index]
            if ref.kind == "trace":
                donor = _resolve_trace_donor(member, hook, out)
            else:
                donor = member
            _prove_event_geometry(donor, out, verb="patch_from(plan)")
            _record_sampling(
                hook,
                "patch_from",
                {
                    "stochastic": True,
                    "population_identity": ref.population_identity,
                    "digest_kind": ref.digest_kind,
                    "population_count": len(ref),
                    "eligible_count": len(eligible),
                    "agreement_class_size": len(eligible),
                    "base_seed": base,
                    "derived_seed": derived,
                    "donor_group_id": plan.donor_group_id,
                    "share_draw": plan.share_draw,
                    "logical_firing_coordinate": coordinate,
                    "donor_ids": [member_index],
                    "realized_draw_digest": _draw_digest([member_index]),
                    "replacement_mode": "whole_event",
                    "donor_grad": "detached",
                    "recomputation": bool(hook.run_ctx.get("recomputation", False)),
                },
            )
            # External donors are detached snapshots by design (D37): target
            # upstream gradients cut, downstream preserved, both facts recorded.
            return donor.detach().to(device=out.device, dtype=out.dtype).clone()

        return _hook

    spec = _helper_spec(
        "patch_from",
        kwargs={"plan": "sampling_plan", **plan.identity_kwargs()},
        factory=factory,
        portability="opaque_audit",
        batch_independent=True,
        metadata={"stochastic": True, "seeded": True},
    )
    # Session-only handle for the D8 batch normalizer (object-identity
    # grouping needs the live plan; the _tl_frozen_cache precedent). Never
    # a dataclass field: it must not persist, compare, or repr.
    object.__setattr__(spec, "_tl_sampling_plan", plan)
    return spec


def assign_batch_donor_groups(
    pairs: Sequence[tuple[Any, Any]],
) -> list[tuple[Any, Any]]:
    """Normalize donor groups across one selection-batch transaction (D8).

    Donor sharing is a deliberate act: reusing ONE plan object across
    clauses keeps one ``donor_group_id`` (all its clauses share draws under
    ``per_group``). Separately constructed plans never share -- even with
    identical visible arguments: distinct plan OBJECTS whose content digests
    collide inside this transaction get deterministic clause-order suffixes
    (``<digest>#1``, ``#2``, ...; the first keeps the bare digest), so the
    disambiguation itself reproduces on a rerun of the same declared batch.
    Composed leaves need no entry here: ``composed_leaf_path`` already enters
    the derived seed and separates leaf streams.

    Parameters
    ----------
    pairs:
        The validated ``(selection, edit)`` clauses of one batch ``do()``.

    Returns
    -------
    list[tuple[Any, Any]]
        The clauses with duplicate-content plans re-lowered onto suffixed
        donor groups; non-sampling edits pass through untouched.
    """

    replacement_by_object: dict[int, SamplingPlan | None] = {}
    digest_order: dict[str, list[int]] = {}
    normalized: list[tuple[Any, Any]] = []
    for selection, edit in pairs:
        plan = getattr(edit, "_tl_sampling_plan", None)
        if plan is None:
            normalized.append((selection, edit))
            continue
        key = id(plan)
        if key not in replacement_by_object:
            order = digest_order.setdefault(plan.donor_group_id, [])
            ordinal = len(order)
            order.append(key)
            replacement_by_object[key] = (
                replace(plan, donor_group_id=f"{plan.donor_group_id}#{ordinal}")
                if ordinal
                else None
            )
        replaced = replacement_by_object[key]
        normalized.append((selection, edit if replaced is None else plan_patch_helper(replaced)))
    return normalized


# ----------------------------------------------------------------------
# the named lowerings: permute_batch / resample_rows_from
# ----------------------------------------------------------------------
def permute_batch(*, seed: int | str, axis: int | None = None) -> HelperSpec:
    """Create the self-population batch-row exchange (a draw WITHOUT replacement).

    D24: REFUSES without a derived-or-explicit axis (a positional default is a
    silent wrong-axis edit on ``batch_first=False`` stacks); batch extent 1
    refuses; identity permutations are legal and DISCLOSED, never silently
    excluded; gradients are graph-connected and permuted (an ``index_select``,
    never a detach); append-incompatible by derived flag.

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.
    """

    validated_seed = _require_seed(seed, "permute_batch")
    explicit_axis = _require_explicit_axis(axis, "permute_batch")

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime batch-permutation hook."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Exchange whole rows of the batch axis under the derived seed."""

            fire_axis = _normalize_fire_axis(explicit_axis, out, "permute_batch")
            extent = out.shape[fire_axis]
            if extent < 2:
                raise InvalidArgumentError(
                    f"permute_batch needs a batch extent >= 2 on axis "
                    f"{fire_axis}; this site's output has extent {extent} "
                    f"(shape {tuple(out.shape)!r}) -- permuting one row is a "
                    "guaranteed no-op wearing an intervention's name",
                    code="permute_batch_extent_invalid",
                    remedy="run a batch with >= 2 examples, or drop the edit",
                    argument="axis",
                )
            self_identity = (
                f"self:{tuple(out.shape)!r}:{out.dtype}:"
                f"{hook.layer_log.get('label') if hook.layer_log else None}"
            )
            base = _base_seed(
                validated_seed,
                rule_identity=f"permute_batch|axis={explicit_axis}",
                population_identity=self_identity,
                hook=hook,
                verb="permute_batch",
            )
            coordinate = _firing_coordinate(hook, "per_firing")
            derived = _derived_seed(
                base,
                population_identity=self_identity,
                donor_group_id="self",
                coordinate=coordinate,
                leaf_path=_leaf_path(hook),
            )
            generator = torch.Generator()
            generator.manual_seed(derived)
            permutation = torch.randperm(extent, generator=generator)
            indices = [int(i) for i in permutation]
            record = {
                "stochastic": True,
                "population_identity": self_identity,
                "digest_kind": "address",
                "population_count": extent,
                "eligible_count": extent,
                "agreement_class_size": extent,
                "base_seed": base,
                "derived_seed": derived,
                "donor_group_id": "self",
                "share_draw": "per_firing",
                "logical_firing_coordinate": coordinate,
                "permutation": indices,
                "realized_draw_digest": _draw_digest(indices),
                "replacement_mode": "row_exchange",
                "donor_grad": "graph_connected_permuted",
                "recomputation": bool(hook.run_ctx.get("recomputation", False)),
                "axis": fire_axis,
                "axis_proof": "explicit",
            }
            if indices == list(range(extent)):
                # Identity permutations are legal and DISCLOSED (D24): a
                # derangement default would silently exclude legal draws.
                record["identity_permutation"] = True
            _record_sampling(hook, "permute_batch", record)
            return out.index_select(fire_axis, permutation.to(out.device))

        return _hook

    from .helpers import _helper_spec

    return _helper_spec(
        "permute_batch",
        kwargs={"seed": validated_seed, "axis": explicit_axis},
        factory=factory,
        batch_independent=False,
        compatible_with_append=False,
        metadata={
            "stochastic": True,
            "seeded": True,
            "row_coherent": True,
            "batch_axis": explicit_axis,
            "batch_coherent": True,
        },
    )


def resample_rows_from(  # noqa: PLR0913 -- spec'd public signature (edits memo D16/D24): the six declared inputs ARE the plan
    source: Reference,
    *,
    seed: int | str,
    axis: int | None = None,
    group_by: Callable[[Any], Any] | None = None,
    matching: Any = None,
    strict: bool = False,
) -> HelperSpec:
    """Create whole-row coherent donor sampling WITH replacement.

    THE field-standard resampling ablation (never the elementwise scramble --
    that is ``scramble_elements``): each subject row is replaced by one
    coherent donor row drawn from the population, independently per row under
    the derived-seed law (D6's ``target_row`` component).

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.

    Parameters
    ----------
    source:
        The population noun; members are DONOR ROWS (build it from a stacked
        tensor -- members are axis-0 slices by the noun's documented contract).
    seed:
        REQUIRED int or ``"auto"``.
    axis:
        The subject value's batch axis, explicit (the D24/D25 axis law).
    group_by:
        Optional datum -> key callable over population datums.
    matching:
        Subject datums: ``PerRowDatums`` (one per batch row) or ``OneDatum``.
    strict:
        Refuse agreement classes of size 1 (D15).
    """

    if not isinstance(source, Reference):
        raise InvalidArgumentError(
            f"resample_rows_from takes the population noun, received {type(source).__name__}",
            code="population_source_invalid",
            remedy='build the donor pool first: reference(row_stack, origin="...")',
            argument="source",
        )
    if source.kind != "tensor":
        raise InvalidArgumentError(
            "resample_rows_from needs tensor row donors; trace-backed "
            "populations resolve whole-site values, not rows",
            code="population_source_invalid",
            remedy="build the population from row tensors "
            "(e.g. reference(stacked_rows, origin=...))",
            argument="source",
        )
    validated_seed = _require_seed(seed, "resample_rows_from")
    explicit_axis = _require_explicit_axis(axis, "resample_rows_from")
    _check_agreement_pair(group_by, matching, "resample_rows_from")
    matching_carrier = (
        _matching_value(matching, "resample_rows_from", allow_per_row=True)
        if matching is not None
        else None
    )
    # Content-digest group key (rerun-reproducible; the verb is per_firing
    # only, so cross-site draw sharing cannot arise and no batch
    # normalization entry is needed).
    donor_group_id = _mint_donor_group_id(
        source.population_identity, validated_seed, "per_firing", matching_carrier, group_by
    )
    rule_identity = f"resample_rows_from|{source.population_identity}|{donor_group_id}"

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime per-row donor-sampling hook."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Replace every subject row with one sampled coherent donor row."""

            fire_axis = _normalize_fire_axis(explicit_axis, out, "resample_rows_from")
            extent = out.shape[fire_axis]
            _prove_event_geometry(source.members[0], out, verb="resample_rows_from", axis=fire_axis)
            if isinstance(matching_carrier, PerRowDatums):
                if len(matching_carrier.values) != extent:
                    raise InvalidArgumentError(
                        f"resample_rows_from matching= carries "
                        f"{len(matching_carrier.values)} per-row datums for a "
                        f"batch of {extent} rows (axis {fire_axis})",
                        code="population_datum_count_mismatch",
                        remedy="pass exactly one datum per subject batch row",
                        argument="matching",
                    )
                row_datums: list[Any] = list(matching_carrier.values)
            else:
                row_datums = [matching_carrier] * extent
            base = _base_seed(
                validated_seed,
                rule_identity=rule_identity,
                population_identity=source.population_identity,
                hook=hook,
                verb="resample_rows_from",
            )
            coordinate = _firing_coordinate(hook, "per_firing")
            eligible_cache: dict[int, tuple[int, ...]] = {}
            donor_ids: list[int] = []
            rows: list[torch.Tensor] = []
            for row in range(extent):
                cache_key = row if group_by is None else hash(repr(group_by(row_datums[row])))
                eligible = eligible_cache.get(cache_key)
                if eligible is None:
                    eligible = source.eligible_indices(group_by, row_datums[row])
                    eligible_cache[cache_key] = eligible
                    _disclose_class_size(
                        len(eligible), strict=strict, verb="resample_rows_from", hook=hook
                    )
                derived = _derived_seed(
                    base,
                    population_identity=source.population_identity,
                    donor_group_id=donor_group_id,
                    coordinate=coordinate,
                    target_row=row,
                    leaf_path=_leaf_path(hook),
                )
                generator = torch.Generator()
                generator.manual_seed(derived)
                position = int(torch.randint(len(eligible), (1,), generator=generator).item())
                member_index = eligible[position]
                donor_ids.append(member_index)
                rows.append(
                    source.members[member_index].detach().to(device=out.device, dtype=out.dtype)
                )
            _record_sampling(
                hook,
                "resample_rows_from",
                {
                    "stochastic": True,
                    "population_identity": source.population_identity,
                    "digest_kind": source.digest_kind,
                    "population_count": len(source),
                    "eligible_count": len(set(donor_ids)),
                    "agreement_class_size": min(
                        (len(rows_) for rows_ in eligible_cache.values()), default=0
                    ),
                    "base_seed": base,
                    "derived_seed": None,
                    "donor_group_id": donor_group_id,
                    "share_draw": "per_firing",
                    "logical_firing_coordinate": coordinate,
                    "donor_ids": donor_ids,
                    "realized_draw_digest": _draw_digest(donor_ids),
                    "replacement_mode": "per_row",
                    "donor_grad": "detached",
                    "recomputation": bool(hook.run_ctx.get("recomputation", False)),
                    "axis": fire_axis,
                    "axis_proof": "explicit",
                },
            )
            return torch.stack(rows, dim=fire_axis)

        return _hook

    from .helpers import _helper_spec

    return _helper_spec(
        "resample_rows_from",
        kwargs={
            "seed": validated_seed,
            "axis": explicit_axis,
            "donor_group_id": donor_group_id,
            "population_identity": source.population_identity,
            "population_count": len(source),
        },
        factory=factory,
        portability="opaque_audit",
        batch_independent=False,
        compatible_with_append=False,
        metadata={
            "stochastic": True,
            "seeded": True,
            "row_coherent": True,
            "batch_axis": explicit_axis,
        },
    )


# ----------------------------------------------------------------------
# deterministic reductions as edits (D3): mean_from / mean_fill
# ----------------------------------------------------------------------
def mean_from(
    ref: Reference,
    *,
    group_by: Callable[[Any], Any] | None = None,
    matching: Any = None,
) -> HelperSpec:
    """Create the deterministic elementwise evidence-set mean fill.

    A mean is a reduction, never a draw (D3): the fill tensor is computed
    eagerly at construction (populations are prepared before hooks install,
    D10), the record says ``stochastic=False``, and no seed is ever invented.

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.
    """

    _check_agreement_pair(group_by, matching, "mean_from")
    fill = ref.mean(group_by=group_by, matching=matching)

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime evidence-mean fill hook."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Fill the site with the prepared evidence-set mean."""

            _prove_event_geometry(fill, out, verb="mean_from")
            _record_sampling(
                hook,
                "mean_from",
                {
                    "stochastic": False,
                    "population_identity": ref.population_identity,
                    "digest_kind": ref.digest_kind,
                    "population_count": len(ref),
                    "eligible_count": None,
                    "base_seed": None,
                    "derived_seed": None,
                    "donor_group_id": None,
                    "share_draw": None,
                    "logical_firing_coordinate": (),
                    "realized_draw_digest": None,
                    "replacement_mode": "evidence_mean",
                    "donor_grad": "detached",
                    "recomputation": bool(hook.run_ctx.get("recomputation", False)),
                },
            )
            return fill.detach().to(device=out.device, dtype=out.dtype).clone()

        return _hook

    from .helpers import _helper_spec

    return _helper_spec(
        "mean_from",
        kwargs={
            "population_identity": ref.population_identity,
            "population_count": len(ref),
        },
        factory=factory,
        portability="opaque_audit",
        batch_independent=True,
        metadata={"stochastic": False},
    )


def mean_fill(
    ref: Reference | None = None,
    *,
    over: Any,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create the FIXED axis-aware mean fill (the ``mean_ablate`` successor).

    ``over=`` is REQUIRED and CLOSED (D17): ``"all"`` is the explicit spelling
    of the old global scalar (and carries the batch-coupled flag when reading
    the fire-time value); explicit int/tuple axes reduce with ``keepdim``
    broadcast; ``"self"`` warns and maps to ``"all"``; named tokens refuse
    ``axis_semantics_unknown`` until axis-role evidence exists (the ladder is
    never rank-guessing). ``batch_independent`` is DERIVED, never tabled
    (D18): ``True`` only when the fill provably never reads the traced batch
    (an external ``ref=``), ``False`` otherwise.

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.
    """

    canonical_over = _resolve_over_axes(over, None, verb="mean_fill")
    prepared_fill: torch.Tensor | None = None
    if ref is not None:
        if not isinstance(ref, Reference):
            raise InvalidArgumentError(
                f"mean_fill ref= takes the population noun, received {type(ref).__name__}",
                code="population_source_invalid",
                remedy='build the population first: reference(members, origin="...")',
                argument="ref",
            )
        prepared_fill = ref.mean(over if over != "self" else "all")

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime axis-aware mean-fill hook."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Fill the site from the configured mean (population or self)."""

            if prepared_fill is not None:
                fill = prepared_fill.detach().to(device=out.device, dtype=out.dtype)
                try:
                    expanded = fill.expand(out.shape) if fill.shape != out.shape else fill
                except RuntimeError:
                    raise InvalidArgumentError(
                        f"mean_fill population mean of shape "
                        f"{tuple(fill.shape)!r} does not broadcast to the site "
                        f"shape {tuple(out.shape)!r}",
                        code="sampling_geometry_mismatch",
                        remedy="build the population at this site's geometry, or "
                        "reduce over axes that broadcast back",
                        argument="ref",
                    ) from None
                return expanded.clone()
            if canonical_over == "all":
                return torch.zeros_like(out) + out.mean()
            # canonical_over is not "all"/"self" on this branch, so the
            # resolved spelling is always an axes tuple.
            axes = cast(
                "tuple[int, ...]",
                _resolve_over_axes(canonical_over, out.ndim, verb="mean_fill"),
            )
            return torch.zeros_like(out) + out.mean(dim=list(axes), keepdim=True)

        return _hook

    from .helpers import _helper_spec

    # D18: derived, never tabled. An external ref provably never reads the
    # traced batch; every self-sourced reduction reads the fire-time value
    # (and no batch-axis proof source exists in v1), so it fails closed.
    batch_independent = ref is not None
    over_kwarg = "all" if canonical_over == "all" else canonical_over
    return _helper_spec(
        "mean_fill",
        kwargs={
            "over": over_kwarg,
            "force_shape_change": force_shape_change,
            **(
                {
                    "population_identity": ref.population_identity,
                    "population_count": len(ref),
                }
                if ref is not None
                else {}
            ),
        },
        factory=factory,
        portability="builtin" if ref is None else "opaque_audit",
        batch_independent=batch_independent,
        compatible_with_append=not force_shape_change and batch_independent,
        metadata={
            "stochastic": False,
            **({} if ref is not None else {"batch_coherent": True}),
        },
    )


# ----------------------------------------------------------------------
# directional dataset mean (D12 of the roster): set_direction_mean
# ----------------------------------------------------------------------
def set_direction_mean(
    direction: torch.Tensor,
    ref: Reference,
    *,
    feature_axis: int,
) -> HelperSpec:
    """Create the directional mean-setting edit.

    ``out - proj_v(out) + mean(coef(ref)) * v_hat``: the site's component
    along ``direction`` is replaced by the population's MEAN coefficient
    along the same direction -- a composition of the shipped project/steer
    semantics with the D3 reducer (the mean coefficient is computed eagerly
    at construction, in float32, and enters the hook as a detached constant).
    Gradients flow through the orthogonal component only (D37).

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.

    Parameters
    ----------
    direction:
        1-D direction vector ``v`` (finite, nonzero norm).
    ref:
        Tensor-membered population supplying the coefficient evidence.
    feature_axis:
        Explicit axis of the site value that ``direction`` lives on (the
        D24/D25 axis law: never guessed from rank).
    """

    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise InvalidArgumentError(
            "set_direction_mean direction= must be a 1-D tensor",
            code="direction_vector_invalid",
            remedy="pass the direction as a 1-D vector over the feature axis",
            argument="direction",
        )
    if not bool(torch.isfinite(direction).all()) or float(direction.norm()) == 0.0:
        raise InvalidArgumentError(
            "set_direction_mean direction= must be finite with nonzero norm",
            code="direction_vector_invalid",
            remedy="pass a finite, nonzero direction vector",
            argument="direction",
        )
    explicit_axis = _require_explicit_axis(feature_axis, "set_direction_mean")
    host_direction = direction.detach().to("cpu", torch.float32)
    unit = host_direction / host_direction.norm()
    members = ref._tensor_members("set_direction_mean")
    extent = int(unit.shape[0])
    coefs: list[torch.Tensor] = []
    for index, member in enumerate(members):
        axis = explicit_axis % member.ndim if -member.ndim <= explicit_axis < member.ndim else None
        if axis is None or member.shape[axis] != extent:
            raise InvalidArgumentError(
                f"set_direction_mean population member {index} has shape "
                f"{tuple(member.shape)!r}, which does not carry the direction's "
                f"extent {extent} on feature_axis={explicit_axis}",
                code="sampling_geometry_mismatch",
                remedy="align feature_axis= with the members' feature axis, or "
                "rebuild the population at the direction's geometry",
                argument="feature_axis",
            )
        shape = [1] * member.ndim
        shape[axis] = extent
        aligned = unit.reshape(shape)
        coefs.append((member.detach().to(torch.float32) * aligned).sum(dim=axis))
    mean_coef = float(torch.stack([coef.mean() for coef in coefs]).mean())

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime directional mean-setting hook."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Replace the direction component with the population mean."""

            fire_axis = _normalize_fire_axis(explicit_axis, out, "set_direction_mean")
            if out.shape[fire_axis] != extent:
                raise InvalidArgumentError(
                    f"set_direction_mean direction extent {extent} does not "
                    f"match the site's feature_axis={fire_axis} extent "
                    f"{out.shape[fire_axis]} (shape {tuple(out.shape)!r})",
                    code="sampling_geometry_mismatch",
                    remedy="target sites whose feature axis carries the direction's extent",
                    argument="feature_axis",
                )
            shape = [1] * out.ndim
            shape[fire_axis] = extent
            aligned = unit.to(device=out.device, dtype=out.dtype).reshape(shape)
            projection = (out * aligned).sum(dim=fire_axis, keepdim=True) * aligned
            _record_sampling(
                hook,
                "set_direction_mean",
                {
                    "stochastic": False,
                    "population_identity": ref.population_identity,
                    "digest_kind": ref.digest_kind,
                    "population_count": len(ref),
                    "base_seed": None,
                    "derived_seed": None,
                    "donor_group_id": None,
                    "share_draw": None,
                    "logical_firing_coordinate": (),
                    "realized_draw_digest": None,
                    "replacement_mode": "direction_mean",
                    "donor_grad": "orthogonal_component_only",
                    "recomputation": bool(hook.run_ctx.get("recomputation", False)),
                    "mean_coefficient": mean_coef,
                },
            )
            # Gradients through the orthogonal component only (D37): the
            # projection subtraction stays graph-connected; the added mean
            # term is a detached constant.
            return out - projection + mean_coef * aligned

        return _hook

    from .helpers import _helper_spec

    direction_digest = hashlib.sha256(unit.contiguous().numpy().tobytes()).hexdigest()[:16]
    return _helper_spec(
        "set_direction_mean",
        kwargs={
            "feature_axis": explicit_axis,
            "direction_digest": direction_digest,
            "population_identity": ref.population_identity,
            "population_count": len(ref),
            "mean_coefficient": mean_coef,
        },
        factory=factory,
        portability="opaque_audit",
        batch_independent=True,
        metadata={"stochastic": False},
    )
