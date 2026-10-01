"""Lens resolution: evidence checks, refusal taxonomy, disclosure assembly.

The one front door that turns (trace, lens row, explicit user kwargs, skin)
into concrete ``Trace.draw`` parameters, enforcing the themes memo's
honesty terms:

- Defaults-not-overrides with the full precedence chain (C05's
  ``resolve_lens_request`` merge; ``preset_spec_fn`` composes BEFORE the
  user's ``node_spec_fn``, which keeps the last word).
- HEADLINE / SECONDARY semantics (N2): missing headline evidence refuses
  typed naming the capture remedy; a missing SECONDARY degrades with a
  coded warning AND a rendered notice. Silence is never an option.
- View-aware source-family resolution (N16) with the zero-coverage refusal
  and the mandatory rolled aggregation line.
- The visible-detail budget resolver (N10) for rows that compact.
- Refusal taxonomy (memo section 2 item 5): overview/blueprint never
  refuse; speed/memory/compute/dims refuse on missing EVIDENCE;
  transformer/sequence refuse on factual absence of SUBJECT. Every refusal
  names the remedy.
- Disclosure is a RENDERED artifact: the assembled disclosure lines ship as
  a bottom graph caption, never a tooltip and never only metadata.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any

from ..._errors import InvalidArgumentError
from ...errors._base import TorchLensWarning
from ..theme_registry import LensPreset, get_lens, resolve_lens_request
from ._budget import BudgetResolution, resolve_budget
from ._families import ResolvedSource, resolve_source_family, scalar_or_none, source_coverage
from ._filter import CompiledDisplayFilter, DisplayFilter, compile_display_filter
from ._nonfinite import NonfiniteChannel, derive_nonfinite_channel, nonfinite_spec_fn
from ._roster import LENS_SUBJECTS, PERF_FAMILIES

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "LENS_DETAIL_CEILING",
    "LEGEND_DEPENDENT_LENSES",
    "LensResolution",
    "draw_with_lens",
    "resolve_lens",
]

#: Above this op count the collapse="none" rows (debug, dims) refuse typed
#: rather than render an illegible wall (candidate value: the existing
#: optimizer ceiling constant; the single R2 tuning session may move it).
LENS_DETAIL_CEILING = 2000

#: Lenses the legibility gate ruled legend-DEPENDENT: naive readers could
#: not identify the channel without the legend, so explicit
#: ``show_legend=False`` refuses typed citing the gate result. EMPTY until
#: the battery rules a lens in (the mechanism ships entry-dark; D03 runs
#: the battery).
LEGEND_DEPENDENT_LENSES: set[str] = set()

#: Request-field name -> Trace.draw kwarg name, where they differ.
_DRAW_KWARG_SPELLINGS = {
    "theme": "vis_theme",
    "graph_overrides": "vis_graph_overrides",
    "edge_overrides": "vis_edge_overrides",
    "grad_edge_overrides": "vis_grad_edge_overrides",
    "module_overrides": "vis_module_overrides",
    "engine": "vis_node_placement",
    "intervention_mode": "vis_intervention_mode",
    "show_cone": "vis_show_cone",
}

#: Attention role rows: module-address suffix -> per-layer fact wording.
#: These are per-LAYER facts (the module tree's own names plus op types),
#: never trace classification -- a hybrid gets rows exactly where the facts
#: are.
_ATTENTION_ROLE_SUFFIXES = {
    "q_proj": "attention role: query projection",
    "k_proj": "attention role: key projection",
    "v_proj": "attention role: value projection",
    "out_proj": "attention role: output projection",
    "o_proj": "attention role: output projection",
    "qkv": "attention role: fused q/k/v projection",
}

#: Op types that are themselves attention facts.
_ATTENTION_OP_TYPES = frozenset(
    {
        "scaled_dot_product_attention",
        "scaleddotproductattention",
        "sdpa",
        "multi_head_attention_forward",
        "multiheadattentionforward",
    }
)


@dataclass(frozen=True)
class LensResolution:
    """One resolved lens draw: parameters plus mandatory disclosure.

    Attributes
    ----------
    lens:
        The resolved registry row.
    draw_kwargs:
        Concrete ``Trace.draw`` keyword arguments (draw spellings).
    disclosure:
        The rendered disclosure lines (also embedded in the caption).
    source:
        The N16-resolved performance source, when the row declares one.
    budget:
        The N10 budget resolution, when the row compacts.
    nonfinite:
        The derived six-state channel (debug row only).
    display_filter:
        The compiled display filter, when the caller passed one.
    """

    lens: LensPreset
    draw_kwargs: dict[str, Any]
    disclosure: tuple[str, ...]
    source: ResolvedSource | None = None
    budget: BudgetResolution | None = None
    nonfinite: NonfiniteChannel | None = None
    display_filter: CompiledDisplayFilter | None = None
    notices: tuple[str, ...] = dataclass_field(default=())


def _has_attention_subject(trace: Trace) -> bool:
    """Return whether the trace factually contains attention structure."""

    try:
        if next(iter(trace.attention_blocks()), None) is not None:
            return True
    except (AttributeError, TypeError):
        pass
    for op in trace.ops:
        if str(getattr(op, "layer_type", "")) in _ATTENTION_OP_TYPES:
            return True
        for address in getattr(op, "modules", ()) or ():
            tail = str(address).split(":", 1)[0].rsplit(".", 1)[-1].lower()
            if tail in _ATTENTION_ROLE_SUFFIXES:
                return True
    return False


def _has_multipass_subject(trace: Trace) -> bool:
    """Return whether any layer runs more than one pass."""

    return any(getattr(op, "num_passes", 1) > 1 for op in trace.ops)


def _check_subject(trace: Trace, lens: LensPreset) -> None:
    """Refuse typed on factual absence of the lens subject (memo item 5)."""

    subject = LENS_SUBJECTS.get(lens.name)
    if subject is None:
        return
    if subject == "attention_structure" and not _has_attention_subject(trace):
        raise InvalidArgumentError(
            "no attention structure detected in this trace; the transformer "
            "lens never draws an ordinary graph under a domain name",
            code="lens_attention_subject_absent",
            remedy="use theme='overview' for a domain-neutral picture",
            argument="lens",
        )
    if subject == "multi_pass" and not _has_multipass_subject(trace):
        raise InvalidArgumentError(
            "this trace has a single pass; nothing to sequence",
            code="lens_single_pass_no_sequence",
            remedy="use theme='overview' for a domain-neutral picture",
            argument="lens",
        )


def _effective_view(lens: LensPreset, user_kwargs: dict[str, Any]) -> str:
    """Return the effective render view: user > lens member > bare default."""

    if "vis_mode" in user_kwargs:
        return str(user_kwargs["vis_mode"])
    member = lens.members.get("vis_mode")
    if member is not None:
        return str(member)
    return "unrolled"


def _resolve_perf_source(
    trace: Trace, lens: LensPreset, view: str
) -> tuple[ResolvedSource, dict[str, float]]:
    """Resolve the row's source family; refuse on missing/zero evidence."""

    family_token = PERF_FAMILIES[lens.name]
    source = resolve_source_family(family_token, view)
    coverage = source_coverage(trace, source.member)
    if coverage.encoded == 0:
        family = source.family
        if coverage.total == 0 or _no_member_anywhere(trace, family_token):
            code = "lens_headline_evidence_missing"
        else:
            code = "lens_source_zero_coverage"
        raise InvalidArgumentError(
            f"the {lens.name!r} lens needs {family.name} evidence "
            f"({source.member}) and this capture has none: 0 of "
            f"{coverage.total} ops carry it -- the picture would answer a "
            "different question under a populated legend",
            code=code,
            remedy=family.capture_remedy,
            argument="lens",
        )
    from ._families import records_for_member, safe_label

    values: dict[str, float] = {}
    for record in records_for_member(trace, source.member):
        value = scalar_or_none(getattr(record, source.member, None))
        if value is not None:
            values[safe_label(record)] = value
    return source, values


def _no_member_anywhere(trace: Trace, family_token: str) -> bool:
    """Return True when neither family member has any evidence (headline gap)."""

    for view in ("unrolled", "rolled"):
        member = resolve_source_family(family_token, view).member
        if source_coverage(trace, member).encoded:
            return False
    return True


def _compose_node_spec_fns(fns: list[Any], user_fn: Any) -> Any:
    """Chain preset spec functions with the user's fn LAST (the last word)."""

    stages = [fn for fn in fns if fn is not None]
    if user_fn is not None:
        stages.append(user_fn)
    if not stages:
        return None
    if len(stages) == 1:
        return stages[0]

    def _chained(layer: Any, spec: Any) -> Any:
        """Fold the spec through every stage; a None return keeps the last spec."""

        current = spec
        for stage in stages:
            result = stage(layer, current)
            if result is not None:
                current = result
        return current

    return _chained


def _speed_callout_spec_fn(values: dict[str, float], unit_wording: str) -> Any:
    """Return the internal top-5 callout preset fn for the speed lens."""

    total = sum(values.values())
    top = sorted(values.items(), key=lambda pair: pair[1], reverse=True)[:5]
    ranks = {label: index + 1 for index, (label, _) in enumerate(top)}
    shares = {label: (value / total if total else 0.0) for label, value in top}
    del unit_wording  # units ride the legend; the callout stays compact

    def _apply(layer: Any, spec: Any) -> Any:
        """Append the top-5 rank callout line to a ranked layer's spec."""

        from ._families import safe_label

        for candidate in (safe_label(layer), getattr(layer, "layer_label", None)):
            if candidate in ranks:
                spec.lines.append(f"top-{ranks[candidate]} of 5 ({shares[candidate]:.0%} of total)")
                break
        return spec

    return _apply


def _debug_label_spec_fn() -> Any:
    """Return the debug preset fn adding dtype/device label rows."""

    def _apply(layer: Any, spec: Any) -> Any:
        """Append dtype/device label rows where the layer carries them."""

        dtype = getattr(layer, "dtype", None)
        device = getattr(layer, "output_device", None)
        parts = []
        if dtype is not None:
            parts.append(f"dtype: {str(dtype).replace('torch.', '')}")
        if device:
            parts.append(f"device: {device}")
        if parts:
            spec.lines.append(", ".join(parts))
        return spec

    return _apply


def _attention_role_spec_fn() -> Any:
    """Return the transformer preset fn attaching per-layer role rows."""

    def _apply(layer: Any, spec: Any) -> Any:
        """Append the observed attention-role row for recognized layers."""

        layer_type = str(getattr(layer, "layer_type", ""))
        if layer_type in _ATTENTION_OP_TYPES:
            spec.lines.append("attention: scaled dot-product (observed)")
            return spec
        for address in getattr(layer, "modules", ()) or ():
            tail = str(address).split(":", 1)[0].rsplit(".", 1)[-1].lower()
            row = _ATTENTION_ROLE_SUFFIXES.get(tail)
            if row is not None:
                spec.lines.append(row)
                break
        return spec

    return _apply


def _neutral_collapsed_spec_fn(neutral_fill: str, user_fn: Any) -> Any:
    """Return a collapsed-box spec fn applying the N17 neutral fill.

    With a colour channel active a collapsed box must never carry a fill
    that already means something else in the same picture (the measured
    trainable-params-grey channel collision); the declared neutral
    "aggregate, not encoded" appearance replaces it, and the user's own
    collapsed fn keeps the last word.
    """

    def _apply(module: Any, spec: Any) -> Any:
        """Paint the neutral fill and disclosure line; user fn keeps the last word."""

        spec.fillcolor = neutral_fill
        spec.lines.append("aggregate, not encoded")
        if user_fn is not None:
            result = user_fn(module, spec)
            if result is not None:
                return result
        return spec

    return _apply


def _channel_exclusivity_spec_fn(theme: Any) -> Any:
    """Clear semantic PARAM fills on unencoded nodes while a channel paints.

    The measured CHANNEL-COLLISION defect: trainable-params grey inside a
    colour-channel picture reads as a scale value. With a channel active an
    unencoded param node keeps the DEFAULT ground instead (encoded nodes
    carry ramp fills, which are not in the semantic set, so they pass
    through untouched). Input/output boundary hues stay: they never collide
    with the sequential ramp and carry wayfinding.
    """

    param_fills = {
        theme.semantic_palette.get("params_generic", "").lower(),
        theme.semantic_palette.get("params_trainable", "").lower(),
        theme.semantic_palette.get("params_frozen", "").lower(),
        theme.semantic_palette.get("params_gradient", "").lower(),
    }

    def _apply(layer: Any, spec: Any) -> Any:
        """Clear a colliding semantic PARAM fill on an unencoded node."""

        if str(spec.fillcolor).lower() in param_fills:
            spec.fillcolor = None
        return spec

    return _apply


def _stack_license_holds(trace: Trace) -> bool:
    """Cheap pre-check of the stack_by='auto' lockstep license."""

    last = 0
    for op in trace.ops:
        if getattr(op, "num_passes", 1) <= 1:
            continue
        pass_index = getattr(op, "pass_index", None)
        if pass_index is None:
            return False
        if pass_index < last:
            return False
        last = pass_index
    return True


def _warn_secondary(lens_name: str, member: str, why: str) -> str:
    """Emit the coded SECONDARY degrade warning; return the rendered notice."""

    notice = f"{member} degraded: {why}"
    warnings.warn(
        TorchLensWarning(
            f"lens {lens_name!r} SECONDARY member {member} degraded: {why}. "
            "Remedy: the lens renders without it; re-capture or adjust the "
            "view to restore the member.",
            code="lens_secondary_degraded",
        ),
        stacklevel=4,
    )
    return notice


@dataclass
class _LensBuild:
    """Mutable build state one resolution's stage helpers share."""

    effective: dict[str, Any]
    user_kwargs: dict[str, Any]
    view: str
    disclosure: list[str]
    notices: list[str] = dataclass_field(default_factory=list)
    preset_fns: list[Any] = dataclass_field(default_factory=list)


def _split_request_kwargs(all_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split explicit kwargs into (request fields, passthrough kwargs).

    Output-target and other non-request draw kwargs (vis_outpath,
    vis_fileformat, ...) pass through untouched: they are never lens
    members, so precedence does not apply to them.
    """

    from dataclasses import fields as dataclass_fields

    from ..request import ResolvedRenderRequest

    request_field_names = {
        request_field.name for request_field in dataclass_fields(ResolvedRenderRequest)
    }
    request = {name: value for name, value in all_kwargs.items() if name in request_field_names}
    passthrough = {
        name: value for name, value in all_kwargs.items() if name not in request_field_names
    }
    return request, passthrough


def _entry_gates(trace: Trace, lens: LensPreset, user_kwargs: dict[str, Any]) -> None:
    """Refuse typed at entry: legend-dependence gate + detail ceiling."""

    # -- Legend-dependence gate (mechanism; populated by the battery). -----
    if lens.name in LEGEND_DEPENDENT_LENSES and user_kwargs.get("show_legend", True) is False:
        raise InvalidArgumentError(
            f"the legibility gate ruled the {lens.name!r} lens "
            "legend-DEPENDENT (naive readers could not identify the channel "
            "without it); explicit show_legend=False is refused for this lens",
            code="lens_legend_suppression_refused",
            remedy="drop show_legend=False, or use a lens the gate ruled legend-optional",
            argument="show_legend",
        )

    # -- Detail ceiling for the collapse='none' rows (debug, dims). --------
    if lens.name in ("debug", "dims") and len(trace.ops) > LENS_DETAIL_CEILING:
        raise InvalidArgumentError(
            f"the {lens.name!r} lens renders uncollapsed and this "
            f"trace has {len(trace.ops)} ops (ceiling {LENS_DETAIL_CEILING})",
            code="lens_above_detail_budget",
            remedy=(
                "focus the render (module= or a Selection region), or pass an "
                "explicit collapse= override at your own risk"
            ),
            argument="lens",
        )


def _perf_stage(
    trace: Trace, lens: LensPreset, build: _LensBuild
) -> tuple[ResolvedSource | None, dict[str, float] | None]:
    """N16 source-family resolution + zero-coverage refusal for perf rows."""

    if lens.name not in PERF_FAMILIES:
        return None, None
    source, channel_values = _resolve_perf_source(trace, lens, build.view)
    if "color_by" not in build.user_kwargs:
        from .._encoding import EncodingChannelRequest

        build.effective["color_by"] = EncodingChannelRequest(
            source=source.member,
            transform="rank",
            display_name=f"{source.family.name} ({source.member})",
        )
    coverage = source_coverage(trace, source.member)
    build.disclosure.append(
        f"coverage: encoded {coverage.encoded} of {coverage.total} ops ({source.member})"
    )
    build.disclosure.append(source.family.unit_wording)
    if source.aggregation_line is not None:
        build.disclosure.append(source.aggregation_line)
    if lens.name == "speed":
        build.preset_fns.append(_speed_callout_spec_fn(channel_values, source.family.unit_wording))
    return source, channel_values


def _budget_stage(
    trace: Trace, build: _LensBuild, channel_values: dict[str, float] | None
) -> BudgetResolution | None:
    """N10 visible-detail budget resolution for rows that compact."""

    if build.effective.get("collapse") != "auto" or "collapse" in build.user_kwargs:
        return None
    from ..request import ResolvedRenderRequest

    # The dial is measured on the EFFECTIVE view's render context: a
    # dial priced on the unrolled universe does not map onto a rolled
    # render (the measured rolled-x-float mispricing cell).
    budget_context = ResolvedRenderRequest(
        vis_mode=build.view,  # type: ignore[arg-type]
        show_buffer_layers=build.effective.get("show_buffer_layers", "meaningful"),
        show_containers=build.effective.get("show_containers", False),
    )
    budget = resolve_budget(trace, values=channel_values, context=budget_context)
    build.effective.update(budget.draw_kwargs)
    build.disclosure.extend(budget.disclosure)
    return budget


def _debug_stage(trace: Trace, lens: LensPreset, build: _LensBuild) -> NonfiniteChannel | None:
    """Debug lens: six-state nonfinite channel (SECONDARY, two-mode degrade)."""

    if lens.name != "debug":
        return None
    build.preset_fns.append(_debug_label_spec_fn())
    nonfinite = derive_nonfinite_channel(trace)
    build.disclosure.extend(nonfinite.legend_lines)
    if nonfinite.zero_coverage:
        warnings.warn(
            TorchLensWarning(
                "nonfinite status: NOT CHECKED -- no saved payloads to "
                "examine; the debug lens renders without per-node status "
                "motifs. Remedy: re-capture with saved payloads or "
                "CaptureOptions(track_nonfinite=True).",
                code="lens_nonfinite_not_checked",
            ),
            stacklevel=3,
        )
    else:
        build.preset_fns.append(nonfinite_spec_fn(nonfinite))
    return nonfinite


def _secondary_stages(trace: Trace, lens: LensPreset, build: _LensBuild) -> None:
    """Per-lens SECONDARY degrades (sequence, memory) + transformer rows."""

    # -- Sequence: stack license pre-check (SECONDARY degrade). ------------
    if (
        lens.name == "sequence"
        and build.effective.get("stack_by") == "auto"
        and not _stack_license_holds(trace)
    ):
        build.effective.pop("stack_by", None)
        build.notices.append(
            _warn_secondary(
                lens.name,
                "stack_by",
                "the lockstep license does not hold on this trace "
                "(pass indexes are not globally non-decreasing)",
            )
        )

    # -- Memory: saved-for-backward SECONDARY degrade. ---------------------
    if lens.name == "memory" and build.effective.get("show_saved_for_backward"):
        has_autograd_evidence = any(
            scalar_or_none(getattr(op, "autograd_memory", None)) is not None for op in trace.ops
        )
        if not has_autograd_evidence:
            build.effective["show_saved_for_backward"] = False
            build.notices.append(
                _warn_secondary(
                    lens.name,
                    "show_saved_for_backward",
                    "no autograd-memory evidence on this capture",
                )
            )

    # -- Transformer: per-layer role rows. ---------------------------------
    if lens.name == "transformer":
        build.preset_fns.append(_attention_role_spec_fn())


def _filter_stage(
    trace: Trace, display_filter: DisplayFilter | None, build: _LensBuild
) -> CompiledDisplayFilter | None:
    """Compile the declarative display filter down to ``skip_fn``."""

    if display_filter is None:
        return None
    if "skip_fn" in build.user_kwargs and build.user_kwargs["skip_fn"] is not None:
        raise InvalidArgumentError(
            "display_filter and an explicit skip_fn cannot combine: the "
            "filter compiles to the same hiding machinery",
            code="display_filter_skip_fn_conflict",
            remedy="pass either display_filter=... or skip_fn=..., not both",
            argument="display_filter",
        )
    compiled_filter = compile_display_filter(trace, display_filter)
    build.effective["skip_fn"] = compiled_filter.skip_fn
    build.disclosure.append(compiled_filter.caption)
    build.disclosure.append(compiled_filter.legend_line)
    return compiled_filter


def _skin_stage(skin: str | None, build: _LensBuild) -> None:
    """Apply the skin + the N17 neutral collapsed fill under a channel."""

    if skin is not None:
        build.effective["theme"] = skin
    from ..themes import resolve_theme

    theme = resolve_theme(str(build.effective.get("theme", "torchlens")))
    channel_active = build.effective.get("color_by") is not None
    if channel_active:
        # Channel exclusivity (composition row 13): one channel, one meaning.
        build.preset_fns.append(_channel_exclusivity_spec_fn(theme))
    if channel_active and build.effective.get("collapse") not in (None, "none"):
        build.effective["collapsed_node_spec_fn"] = _neutral_collapsed_spec_fn(
            theme.neutral_aggregate_fill, build.user_kwargs.get("collapsed_node_spec_fn")
        )
        build.disclosure.append("collapsed boxes: neutral fill = aggregate, not encoded")


def _finalize_draw_kwargs(build: _LensBuild, passthrough: dict[str, Any]) -> dict[str, Any]:
    """Compose node specs, render the disclosure caption, map spellings."""

    # -- Compose the node-spec chain (preset slots first, user LAST). ------
    user_node_spec_fn = build.effective.pop("node_spec_fn", None)
    composed = _compose_node_spec_fns(build.preset_fns, user_node_spec_fn)
    if composed is not None:
        build.effective["node_spec_fn"] = composed

    # -- Rendered disclosure caption (never a tooltip). ---------------------
    caption = "\\n".join(line.replace("\n", " ") for line in build.disclosure)
    graph_overrides = dict(build.effective.get("graph_overrides") or {})
    existing_label = graph_overrides.get("label")
    graph_overrides["label"] = f"{existing_label}\\n{caption}" if existing_label else caption
    graph_overrides.setdefault("labelloc", "b")
    graph_overrides.setdefault("fontsize", "10")
    build.effective["graph_overrides"] = graph_overrides

    draw_kwargs = {
        _DRAW_KWARG_SPELLINGS.get(name, name): value for name, value in build.effective.items()
    }
    draw_kwargs.update(passthrough)
    return draw_kwargs


def resolve_lens(
    trace: Trace,
    lens: str | LensPreset,
    user_kwargs: dict[str, Any] | None = None,
    *,
    skin: str | None = None,
    display_filter: DisplayFilter | None = None,
) -> LensResolution:
    """Resolve a lens draw against one trace, enforcing the honesty terms.

    Parameters
    ----------
    trace:
        The trace to render.
    lens:
        Registered lens name (default-surface lookup) or row. A lens on a
        non-default surface passes its resolved :class:`LensPreset` row
        (``get_lens(name, surface)``); the row carries its surface.
    user_kwargs:
        EXPLICITLY-passed draw parameters only (request-field spellings);
        they win over every lens member.
    skin:
        Cosmetic skin name (``themes.THEME_PRESETS`` key); composes freely
        with any lens.
    display_filter:
        Optional declarative display filter (no v1 lens sets one).

    Returns
    -------
    LensResolution
        Draw kwargs in ``Trace.draw`` spellings plus the disclosure record.
    """

    request_kwargs, passthrough = _split_request_kwargs(dict(user_kwargs or {}))
    # The surgery row (lane F43) registers at its module's import; resolving
    # it by name must not depend on the caller having imported that module.
    from .. import surgery_visuals as _surgery_visuals  # noqa: F401

    resolved_lens = lens if isinstance(lens, LensPreset) else get_lens(lens)
    _check_subject(trace, resolved_lens)
    _entry_gates(trace, resolved_lens, request_kwargs)

    effective = resolve_lens_request(resolved_lens, request_kwargs, resolved_lens.surface)
    effective.pop("preset_spec_fn", None)
    build = _LensBuild(
        effective=effective,
        user_kwargs=request_kwargs,
        view=_effective_view(resolved_lens, request_kwargs),
        disclosure=[f"lens: {resolved_lens.name} -- {resolved_lens.question}"],
    )
    if resolved_lens.preset_spec_fn is not None:
        build.preset_fns.append(resolved_lens.preset_spec_fn)

    source, channel_values = _perf_stage(trace, resolved_lens, build)
    budget = _budget_stage(trace, build, channel_values)
    nonfinite = _debug_stage(trace, resolved_lens, build)
    if resolved_lens.name == "surgery":
        # Surgery lens (lane F43): headline check + mark family + census
        # disclosure resolve in the surgery module; this stage only mounts
        # them on the build (missing evidence refuses surgery_evidence_missing).
        mark_spec_fn, census_lines = _surgery_visuals.surgery_stage(trace)
        build.preset_fns.append(mark_spec_fn)
        build.disclosure.extend(census_lines)
    _secondary_stages(trace, resolved_lens, build)
    compiled_filter = _filter_stage(trace, display_filter, build)
    _skin_stage(skin, build)

    build.disclosure.extend(resolved_lens.disclosure)
    build.disclosure.extend(build.notices)
    draw_kwargs = _finalize_draw_kwargs(build, passthrough)
    return LensResolution(
        lens=resolved_lens,
        draw_kwargs=draw_kwargs,
        disclosure=tuple(build.disclosure),
        source=source,
        budget=budget,
        nonfinite=nonfinite,
        display_filter=compiled_filter,
        notices=tuple(build.notices),
    )


def draw_with_lens(
    trace: Trace,
    lens: str | LensPreset,
    *,
    skin: str | None = None,
    display_filter: DisplayFilter | None = None,
    **user_kwargs: Any,
) -> Any:
    """Resolve ``lens`` against ``trace`` and draw it (the v1 door).

    The public ``theme=``/``skin=`` draw spelling is a named [UI-SPRINT]
    fork; until it is ratified this documented-unstable function is how a
    lens reaches ``Trace.draw``. Explicit ``user_kwargs`` (request-field
    spellings) always win over lens members.
    """

    resolution = resolve_lens(trace, lens, user_kwargs, skin=skin, display_filter=display_filter)
    return trace.draw(**resolution.draw_kwargs)
