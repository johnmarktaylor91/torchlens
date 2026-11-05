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
- Refusal taxonomy (memo section 2 item 5): overview never
  refuses; speed/memory/compute/dims refuse on missing EVIDENCE;
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
from ..._utils._display_bounded import short_dataclass_repr
from ..._vocab.node_spec import LENS_BANNER_OMIT_KEY, LENS_CAPTION_KEY
from ...errors._base import TorchLensWarning
from .._encoding_request import EncodingChannelRequest
from ..theme_registry import LensPreset, get_lens, resolve_lens_request
from ._attention import attention_role_spec_fn, first_attention_block, has_attention_subject
from ._budget import BudgetResolution, resolve_budget
from ._families import ResolvedSource, resolve_source_family, scalar_or_none, source_coverage
from ._filter import CompiledDisplayFilter, DisplayFilter, compile_display_filter
from ._nonfinite import NonfiniteChannel, derive_nonfinite_channel, nonfinite_spec_fn
from ._roster import LENS_SUBJECTS, PERF_FAMILIES

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

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
    "engine": "layout",
    "show_buffer_layers": "show_buffers",
    "node_mode": "node_style",
    "scale": "size_scale",
    "vis_call_depth": "depth",
}

#: Trace.draw kwarg name -> request-field name, for the explicit kwargs a
#: caller passes in draw's dialect (the inverse of the table above, plus the
#: ``rolled`` bool that stands for the request's ``vis_mode``).
_REQUEST_SPELLINGS = {draw_name: field for field, draw_name in _DRAW_KWARG_SPELLINGS.items()}


def _request_dialect(user_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return explicit ``Trace.draw`` kwargs in request-field spellings."""

    normalized: dict[str, Any] = {}
    for name, value in user_kwargs.items():
        if name == "rolled":
            normalized["vis_mode"] = "rolled" if value else "unrolled"
        else:
            normalized[_REQUEST_SPELLINGS.get(name, name)] = value
    return normalized


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


def _has_multipass_subject(trace: Trace) -> bool:
    """Return whether a parameterised layer runs more than one pass.

    A loop reuses weights (an RNN cell, a tied block); a feed-forward model
    that calls one stateless ``nn.ReLU`` twice has nothing to sequence, so
    reused parameter-free ops alone do not count.
    """

    return any(
        getattr(op, "num_passes", 1) > 1 and bool(getattr(op, "has_params", False))
        for op in trace._layer_list
    )


def _check_subject(trace: Trace, lens: LensPreset) -> None:
    """Refuse typed on factual absence of the lens subject (memo item 5)."""

    subject = LENS_SUBJECTS.get(lens.name)
    if subject is None:
        return
    if subject == "attention_structure" and not has_attention_subject(trace):
        raise InvalidArgumentError(
            "no attention structure detected in this trace; the transformer "
            "view never draws an ordinary graph under a domain name",
            code="view_attention_subject_absent",
            remedy="leave view= unset, or pass view='overview', for a domain-neutral picture",
            argument="view",
        )
    if subject == "multi_pass" and not _has_multipass_subject(trace):
        raise InvalidArgumentError(
            "no layer with weights runs more than once in this trace; nothing to sequence",
            code="view_single_pass_no_sequence",
            remedy="leave view= unset, or pass view='overview', for a domain-neutral picture",
            argument="view",
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
            code = "view_headline_evidence_missing"
        else:
            code = "view_source_zero_coverage"
        raise InvalidArgumentError(
            f"the {lens.name!r} view needs {family.name} evidence "
            f"({source.member}) and this capture has none: 0 of "
            f"{coverage.total} ops carry it -- the picture would answer a "
            "different question under a populated legend",
            code=code,
            remedy=family.capture_remedy,
            argument="view",
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
    # Zero-time records (model inputs and outputs) never rank.
    timed = [(label, value) for label, value in values.items() if value > 0]
    top = sorted(timed, key=lambda pair: pair[1], reverse=True)[:5]
    ranks = {label: index + 1 for index, (label, _) in enumerate(top)}
    shares = {label: (value / total if total else 0.0) for label, value in top}
    del unit_wording  # units ride the legend; the callout stays compact

    def _apply(layer: Any, spec: Any) -> Any:
        """Append the top-5 rank callout line to a ranked layer's spec."""

        from ._families import safe_label

        for candidate in (safe_label(layer), getattr(layer, "layer_label", None)):
            if candidate in ranks:
                spec.lines.append(
                    f"top-{ranks[candidate]} of {len(top)} ({shares[candidate]:.0%} of total)"
                )
                break
        return spec

    return _apply


def _dtype_device(record: Any) -> tuple[str | None, str | None]:
    """Return one record's output dtype and device as display strings."""

    dtype = getattr(record, "dtype", None)
    device = getattr(record, "device_ref", None)
    return (
        None if dtype is None else str(dtype).replace("torch.", ""),
        None if device is None else str(device),
    )


def _usual_dtype_device(trace: Trace) -> tuple[str | None, str | None]:
    """Return the trace's most common (dtype, device) pair over its ops."""

    counts: dict[tuple[str | None, str | None], int] = {}
    for op in trace._layer_list:
        pair = _dtype_device(op)
        counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return None, None
    return max(counts.items(), key=lambda item: item[1])[0]


def _debug_label_spec_fn(usual: tuple[str | None, str | None]) -> Any:
    """Return the debug preset fn marking ops whose dtype or device is unusual.

    The usual pair goes in the caption once; a node gets a row only where it
    differs (a cast, a device move), which is what a debugging reader is
    looking for.
    """

    def _apply(layer: Any, spec: Any) -> Any:
        """Append a dtype/device row when the layer differs from the usual pair."""

        dtype, device = _dtype_device(layer)
        parts = []
        if dtype is not None and dtype != usual[0]:
            parts.append(f"dtype: {dtype}")
        if device is not None and device != usual[1]:
            parts.append(f"device: {device}")
        if parts:
            spec.lines.append(", ".join(parts))
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
    for op in trace._layer_list:
        if getattr(op, "num_passes", 1) <= 1:
            continue
        pass_index = getattr(op, "pass_number", None)
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
            f"view {lens_name!r} SECONDARY member {member} degraded: {why}. "
            "Remedy: the view renders without it; re-capture or adjust the "
            "draw keywords to restore the member.",
            code="view_secondary_degraded",
        ),
        stacklevel=4,
    )
    return notice


@dataclass
class _LensBuild:
    """Mutable build state one resolution's stage helpers share."""

    __repr__ = short_dataclass_repr

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
            f"the legibility gate ruled the {lens.name!r} view "
            "legend-DEPENDENT (naive readers could not identify the channel "
            "without it); explicit show_legend=False is refused for this view",
            code="view_legend_suppression_refused",
            remedy="drop show_legend=False, or use a view the gate ruled legend-optional",
            argument="show_legend",
        )

    # -- Detail ceiling for the collapse='none' rows (debug, dims). --------
    if lens.name in ("debug", "dims") and len(trace._layer_list) > LENS_DETAIL_CEILING:
        raise InvalidArgumentError(
            f"the {lens.name!r} view renders uncollapsed and this "
            f"trace has {len(trace._layer_list)} ops (ceiling {LENS_DETAIL_CEILING})",
            code="view_above_detail_budget",
            remedy=(
                "focus the render (module= or a Selection region), or pass an "
                "explicit collapse= override at your own risk"
            ),
            argument="view",
        )


def _perf_stage(
    trace: Trace, lens: LensPreset, build: _LensBuild
) -> tuple[ResolvedSource | None, dict[str, float] | None]:
    """N16 source-family resolution + zero-coverage refusal for perf rows."""

    if lens.name not in PERF_FAMILIES:
        return None, None
    source, channel_values = _resolve_perf_source(trace, lens, build.view)
    if "color_by" not in build.user_kwargs:
        build.effective["color_by"] = EncodingChannelRequest(
            source=source.member,
            transform="rank",
            display_name=f"{source.family.name} ({source.member})",
        )
    coverage = source_coverage(trace, source.member)
    build.disclosure.append(
        # Ops, not visible nodes: one visible node can aggregate several ops.
        f"coverage: encoded {coverage.encoded} of {coverage.total} ops; "
        f"colour: {source.family.unit_wording}"
    )
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
    usual = _usual_dtype_device(trace)
    build.preset_fns.append(_debug_label_spec_fn(usual))
    if usual[0] is not None and usual[1] is not None:
        build.disclosure.append(f"ops are {usual[0]} on {usual[1]} unless a node says otherwise")
    nonfinite = derive_nonfinite_channel(trace)
    build.disclosure.extend(nonfinite.legend_lines)
    if nonfinite.zero_coverage:
        warnings.warn(
            TorchLensWarning(
                "nonfinite status: NOT CHECKED -- no saved payloads to "
                "examine; the debug view renders without per-node status "
                "motifs. Remedy: re-capture with saved payloads or "
                "CaptureOptions(on_nonfinite='track').",
                code="view_nonfinite_not_checked",
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
                "the passes are not in order on this trace, so they are not stacked",
            )
        )

    # -- Memory: saved-for-backward SECONDARY degrade. ---------------------
    if lens.name == "memory" and build.effective.get("show_saved_for_backward"):
        has_autograd_evidence = any(
            scalar_or_none(getattr(op, "autograd_memory", None)) is not None
            for op in trace._layer_list
        )
        if not has_autograd_evidence:
            build.effective["show_saved_for_backward"] = False
            build.notices.append(
                _warn_secondary(
                    lens.name,
                    "show_saved_for_backward",
                    "this capture has no saved-for-backward sizes",
                )
            )

    # -- Transformer: per-layer role rows, one block in full. --------------
    if lens.name == "transformer":
        build.preset_fns.append(attention_role_spec_fn())
        block = None if build.user_kwargs.get("module") else first_attention_block(trace)
        if block is not None:
            address, class_name, count = block
            build.effective["module"] = address
            build.disclosure.append(
                f"showing {address}, the first of {count} {class_name} blocks; "
                "pass module= to draw another"
            )


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

    # -- Rendered disclosure caption (never a tooltip), under the header. ---
    graph_overrides = dict(build.effective.get("graph_overrides") or {})
    graph_overrides[LENS_CAPTION_KEY] = "\n".join(
        line.replace("\n", " ") for line in build.disclosure
    )
    build.effective["graph_overrides"] = graph_overrides

    draw_kwargs: dict[str, Any] = {}
    for name, value in build.effective.items():
        if name == "vis_mode":
            draw_kwargs["rolled"] = value == "rolled"
        else:
            draw_kwargs[_DRAW_KWARG_SPELLINGS.get(name, name)] = value
    draw_kwargs.update(passthrough)
    return draw_kwargs


def _start_build(trace: Trace, lens: LensPreset, request_kwargs: dict[str, Any]) -> _LensBuild:
    """Merge the lens members under the explicit kwargs into a fresh build."""

    effective = resolve_lens_request(lens, request_kwargs, lens.surface)
    effective.pop("preset_spec_fn", None)
    if (
        effective.get("vis_mode") == "rolled"
        and "vis_mode" not in request_kwargs
        and not _has_multipass_subject(trace)
    ):
        # Rolling a feed-forward model only folds reused stateless ops (one
        # nn.ReLU called twice) into loops that are not there.
        effective["vis_mode"] = "unrolled"
    build = _LensBuild(
        effective=effective,
        user_kwargs=request_kwargs,
        view=str(effective.get("vis_mode", _effective_view(lens, request_kwargs))),
        disclosure=[f"view: {lens.name} -- {lens.question}"],
    )
    if lens.preset_spec_fn is not None:
        build.preset_fns.append(lens.preset_spec_fn)
    return build


def _surgery_mount(trace: Trace, build: _LensBuild) -> None:
    """Mount the surgery stage: marks, plain caption, and the header lines it moves.

    The headline check, mark family and census resolve in the surgery module
    (missing evidence refuses ``surgery_evidence_missing``); the header's
    capture-status lines about the edit move to the census sidecar, since the
    header's edit line already says the trace is edited.
    """

    from .. import surgery_visuals as _surgery_visuals

    mark_spec_fn, census_lines = _surgery_visuals.surgery_stage(trace)
    build.preset_fns.append(mark_spec_fn)
    build.disclosure.extend(census_lines)
    overrides = dict(build.effective.get("graph_overrides") or {})
    overrides[LENS_BANNER_OMIT_KEY] = _surgery_visuals.EDIT_VIEW_OMITTED_BANNERS
    build.effective["graph_overrides"] = overrides


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

    request_kwargs, passthrough = _split_request_kwargs(_request_dialect(dict(user_kwargs or {})))
    # The surgery row (lane F43) registers at its module's import; resolving
    # it by name must not depend on the caller having imported that module.
    from .. import surgery_visuals as _surgery_visuals  # noqa: F401

    resolved_lens = lens if isinstance(lens, LensPreset) else get_lens(lens)
    _check_subject(trace, resolved_lens)
    _entry_gates(trace, resolved_lens, request_kwargs)
    build = _start_build(trace, resolved_lens, request_kwargs)

    source, channel_values = _perf_stage(trace, resolved_lens, build)
    budget = _budget_stage(trace, build, channel_values)
    nonfinite = _debug_stage(trace, resolved_lens, build)
    if resolved_lens.name == "interventions":
        _surgery_mount(trace, build)
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
