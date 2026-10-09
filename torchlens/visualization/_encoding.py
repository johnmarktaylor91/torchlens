"""Encoding-channel core (S5/L5): declarative value -> visual channels for draw().

v1 ships the COLOR channel (``color_by``): a value source (Layer/Op field
name, scalar node-overlay builtin, or callable ``node -> value``) compiled
down to the existing NodeSpec chain as a fillcolor transform in the C3
precedence slot (after the node-style preset and intervention styling, before
the user ``node_spec_fn``), plus an auto-legend that disclosures the
transform.

Wave 1 adds the SIZE channel (``size_by`` + ``scale=``): a scalar field,
callable, or the closed ``"dims"`` shape token mapped to node width/height
minimums (Graphviz ``fixedsize=false`` -- a label can never be truncated by
an encoding, and the max clamp bounds encoded area at
:data:`SIZE_BY_MAX_AREA_MULT` times the default node area). The shipped
multi-dim -> geometry mapping is the D4 DEFAULT (memo 3.2 C2, AREA-ONLY:
one scalar = numel of the non-batch shape, box area ~ value, default aspect
preserved) with the default ``scale="sqrt"`` -- D4 was UNRULED at this
merge, so the METAPLAN default applies and is marked as default-applied.
Size REFUSES where color degrades on rolled multi-pass nodes
(``size_by_rolled_varying``): an unencoded box is visually indistinguishable
from an encoded small box, so size has no honest "n/a" rendering.

Resolution is TWO-PHASE (design memo 2.3):

* PHASE A -- a presentation PREPASS over the already-chosen visible-node
  universe (:func:`populate_encoding_state`, called from
  ``build_render_ir``): collect each eligible node's raw value exactly once
  (user callables are invoked once per node HERE and never again), apply the
  closed value/type rules and the rolled-aggregate source allowlist, and
  min-max normalize over the finite values. This is a data prepass over
  records; it renders nothing.
* PHASE B -- during per-node spec resolution, apply the precomputed color as
  a spec transform (:meth:`EncodingState.fillcolor_for`); the user
  ``node_spec_fn`` still sees and may override the channel's output.

NAMING: ``color_by``, the ``encoding_*`` error codes, and the ``show_legend``
tri-state are DOCUMENTED-UNSTABLE spellings (no deprecation shim owed) until
the naming session ratifies them (METAPLAN naming protocol).

HONESTY TRIPWIRE (never weaken): an encoding must never imply uniformity it
cannot prove. On a rolled multi-pass Layer node, a FIELD source resolves
through the NAME-KEYED rolled-aggregate allowlist below; a source in no
declared row REFUSES rather than silently projecting pass-1 or an aggregate
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

# ---------------------------------------------------------------------------
# Rolled-aggregate source allowlist rows (design memo 2.3b, r4 shape).
#
# A FIELD or BUILTIN source resolving on a rolled MULTI-PASS Layer node
# (len(layer.ops) > 1) is classified by the NAME-KEYED table below. A name in
# no row refuses ``encoding_source_invalid`` at the prepass (allowlist
# default): a new record field cannot silently become encodable.
# ---------------------------------------------------------------------------

#: All passes agree unless annotations["varying_across_passes"] names the
#: field (the finalization reconciler's any-variation marker). Unmarked ->
#: exact -> encode; marked -> the stored aggregate is a per-pass MAXIMUM
#: (upper bound) -> color UNENCODES + legend note (size_by will refuse).
ROW_RECONCILED = "reconciled_aggregate"
#: Exact cross-pass aggregate (sum or distinct-union). Uniformly defined on
#: every node ("this node's total over the whole trace"), encode WITH a
#: mandatory aggregation legend line.
ROW_SUMMED = "summed_aggregate"
#: Mirrored per-call NUMERICS: stored names resolving through
#: _LAYER_MIRROR_SPEC to the representative FIRST pass with NO variation
#: marker and NO reconciliation -- a first-pass projection that cannot be
#: certified single-valued. Color unencodes + legend note; size refuses.
ROW_MIRRORED_PER_CALL = "mirrored_per_call_numeric"
#: Value determined by the grouping identity (function, parameters, output
#: slot, module address, non-tensor args) or by group construction
#: (num_passes). Encodes normally; op-backed members additionally get a
#: DEFENSIVE per-pass equality check at resolution (mismatch degrades to
#: unencoded + note rather than painting a dishonest uniform value).
ROW_STRUCTURAL = "structurally_uniform"
#: Properties computed from other classified fields; they inherit the MOST
#: CONSERVATIVE verdict among their inputs (see _DERIVED_COMPOSITE_INPUTS).
ROW_DERIVED = "derived_composite"
#: Per-pass attributes with no aggregate meaning: the multipass-safe read
#: degrades to None on a rolled aggregate -> unencoded + "n/a" legend note.
ROW_PER_PASS = "per_pass"
#: Shape-valued sources: never a scalar color source (size_by consumes them
#: via the wave-1 "dims" typed shape path only).
ROW_SHAPE = "shape"
#: Everything else readable off the record: strings, containers, callables,
#: bools (a bool is a truth value, not an encodable magnitude), weakrefs.
#: Resolving one raises the 2.2 wrong-type rule (encoding_value_invalid).
ROW_NON_NUMERIC = "non_numeric"

#: THE TRUE MIRRORED PER-CALL NUMERIC ENUMERATION (sol r4 MAJOR-1 residual
#: fix). The design memo claimed this set was exactly
#: {transformed_gradient_memory}; the live capture/postprocess code disproves
#: that five ways:
#:   raw_index               -- fresh incremented counter per emitted op
#:   step_index              -- labeling step 9 reassigns SEQUENTIALLY PER OP
#:                              (labeling.py: layer_entry.step_index = step_index)
#:   ordinal_index           -- zero-based position of EACH op in the final
#:                              ordered layer list (labeling.py step 11)
#:   grad_fn_object_id       -- per-call autograd object identity
#:   buffer_pass             -- sequential per-address buffer version number
#:                              (control_flow.py step 6)
#:   transformed_gradient_memory -- per-call Bytes, no marker (the memo's one)
#: Plus the first-pass-projection numeric PROPERTY conditional_depth
#: (len over the mirror-copied in_conditionals) classified in the same row.
MIRRORED_PER_CALL_NUMERIC_FIELDS = frozenset(
    {
        "raw_index",
        "step_index",
        "ordinal_index",
        "grad_fn_object_id",
        "buffer_pass",
        "transformed_gradient_memory",
        "conditional_depth",
    }
)

_RECONCILED_FIELDS = frozenset(
    {"activation_memory", "transformed_activation_memory", "flops_forward", "flops_backward"}
)

_SUMMED_FIELDS = frozenset(
    {
        # Stored fields _build_layer_logs overwrites with cross-pass sums.
        "autograd_memory",
        "total_autograd_memory",
        "num_autograd_tensors",
        # total_* sum properties over Layer.ops.
        "total_activation_memory",
        "total_gradient_memory",
        "total_flops_forward",
        "total_flops_backward",
        "total_flops_total",
        "total_macs_forward",
        "total_macs_backward",
        "total_macs_total",
        "total_func_duration",
        # Aggregate distinct-union cardinalities (exact on the rolled node).
        "num_children",
        "num_parents",
    }
)

#: Legend wording per summed-family source (default: "total across passes").
_AGGREGATE_LEGEND_WORDING = {
    "num_children": "distinct across all passes",
    "num_parents": "distinct across all passes",
}

_STRUCTURAL_FIELDS = frozenset(
    {
        "type_index",  # inherited from pass 1 for every pass (labeling step 8)
        "num_passes",  # group-level, assigned to every member
        "num_ops",  # == number of passes
        "num_args_total",
        "num_pos_args",
        "num_kwargs",
        "multi_output_index",  # output slot, part of the grouping identity
        "num_params",
        "num_params_trainable",
        "num_params_frozen",
        "total_param_memory",
        "num_param_tensors",
        "num_param_tensors_trainable",
        "num_param_tensors_frozen",
    }
)

#: Derived-composite properties -> their input fields. Verdict = the most
#: conservative verdict among the inputs' rows (aliasing rule: verdicts key
#: on the DECLARED row, never on string-prefix accident).
_DERIVED_COMPOSITE_INPUTS: dict[str, tuple[str, ...]] = {
    "flops_total": ("flops_forward", "flops_backward"),
    "macs_forward": ("flops_forward",),
    "macs_backward": ("flops_backward",),
    "macs_total": ("flops_forward", "flops_backward"),
    "buffer_overwrite_index": ("buffer_pass",),
    # len(modules): the reconciler writes a "modules" variation marker.
    "module_call_depth": ("modules",),
}

_PER_PASS_FIELDS = frozenset({"func_duration", "fx_call_index", "pass_index"})

_SHAPE_FIELDS = frozenset({"shape", "transformed_out_shape"})

#: Sources classified per-name for rolled multi-pass resolution. Built below
#: from the row sets plus the NON_NUMERIC remainder of the readable Layer
#: namespace; the classification completeness pin
#: (tests/test_encoding_channels.py) asserts this dict covers EXACTLY
#: _LAYER_STATE_ORDER + _LAYER_MIRROR_SPEC keys + the public Layer
#: properties, each name in exactly one row.
_NON_NUMERIC_FIELDS = frozenset(
    {
        # Identity / naming strings.
        "layer_label",
        "layer_label_short",
        # L1 grouping surface (site keys + across-pass shape summary): a
        # percent-escaped position key, a live peer view, and a plain-data
        # summary STRING ("2->4") — none is an encodable magnitude. The
        # completeness pin went red the moment L1's merge added these; this
        # row is the classification it demanded.
        "site_key",
        "site_peers",
        "shape_summary",
        "layer_type",
        "label",
        "label_short",
        "fx_label",
        "fx_qualpath",
        "lookup_keys",
        "op_labels",
        "call_labels",
        # Function / autograd objects and strings.
        "func",
        "func_name",
        "func_qualname",
        "func_config",
        "func_rng_states",
        "grad_fn",
        "grad_fn_handle",
        "grad_fn_class_name",
        "grad_fn_class_qualname",
        "code_context",
        "arg_names",
        "saved_args",
        "saved_kwargs",
        # Bools (a truth value is not an encodable magnitude; the runtime
        # wrong-type rule rejects bool explicitly).
        "is_inplace",
        "in_multi_output",
        "is_input",
        "input_was_parameter",
        "is_output",
        "is_final_output",
        "is_buffer",
        "is_buffer_source",
        "is_compute_layer",
        "is_internal_source",
        "is_internal_sink",
        "is_terminal_bool",
        "is_scalar_bool",
        "bool_value",
        "buffer_value_changed",
        "buffer_replay_validated",
        "has_input_ancestor",
        "is_atomic_module",
        "intervention_replaced",
        "detach_saved_activations",
        "save_grads",
        "edges_vary_across_ops",
        "has_children",
        "has_co_parents",
        "has_frozen_params",
        "has_grad",
        "has_parents",
        "has_saved_activation",
        "has_siblings",
        "has_trainable_params",
        "in_submodule",
        "is_in_conditional",
        "is_in_conditional_body",
        "is_in_conditional_evaluation",
        "is_module_input",
        "is_orphan",
        "uses_params",
        # Dtypes / devices / addresses / roles (marker-only family included).
        "dtype",
        "dtype_ref",
        "transformed_out_dtype",
        "transformed_grad_dtype",
        "device_ref",
        "output_device",
        "backend_address",
        "resolver_status",
        "address",
        "io_role",
        "equivalence_class",
        "multi_output_name",
        "buffer_source",
        "buffer_write_kind",
        "buffer_source_func_name",
        "visualizer_path",
        "activation_transform",
        # Containers / graph views / payloads.
        "transformed_grad_shape",
        "param_shapes",
        "params",
        "param_names",
        "param_dtypes",
        "modules",
        "module",
        "output_of_modules",
        "output_of_module_calls",
        "in_conditionals",
        "terminal_bool_for",
        "conditional_entry_children",
        "conditional_then_children",
        "conditional_elif_children",
        "conditional_else_children",
        "conditional_arm_children",
        "conditional_role_stacks",
        "conditional_branch_stack_ops",
        "annotations",
        "equivalent_ops",
        "ops",
        "children",
        "parents",
        "children_per_pass",
        "parents_per_pass",
        "child_ops_per_layer",
        "parent_ops_per_layer",
        "parent_arg_positions",
        "co_parents",
        "siblings",
        "leaf_module_ops",
        "out",
        "grad",
        "tensor",
        "transformed_out",
        "transformed_grad",
        "receptive_field",
        "projective_field",
        "source_trace",
        "trace",
        # Private stored state.
        "_source_trace_ref",
        "_is_in_conditional_body",
        "_param_barcodes",
        "_param_logs",
    }
)


def _build_source_rows() -> dict[str, str]:
    """Build the name -> row classification table."""

    rows: dict[str, str] = {}
    for names, row in (
        (_RECONCILED_FIELDS, ROW_RECONCILED),
        (_SUMMED_FIELDS, ROW_SUMMED),
        (MIRRORED_PER_CALL_NUMERIC_FIELDS, ROW_MIRRORED_PER_CALL),
        (_STRUCTURAL_FIELDS, ROW_STRUCTURAL),
        (frozenset(_DERIVED_COMPOSITE_INPUTS), ROW_DERIVED),
        (_PER_PASS_FIELDS, ROW_PER_PASS),
        (_SHAPE_FIELDS, ROW_SHAPE),
        (_NON_NUMERIC_FIELDS, ROW_NON_NUMERIC),
    ):
        for name in names:
            if name in rows:  # pragma: no cover - guarded by the completeness pin
                raise AssertionError(f"source {name!r} classified into two rows")
            rows[name] = row
    return rows


LAYER_SOURCE_ROWS: dict[str, str] = _build_source_rows()

# ---------------------------------------------------------------------------
# Builtin scalar sources: the node_overlay builtin names where scalar, so the
# two surfaces converge (memo 2.2). "nan" is boolean-valued and NOT a scalar
# color source. Field-backed builtins resolve through the FIELD path so the
# rolled-aggregate allowlist governs them; payload builtins (magnitude,
# grad_norm) resolve through the multipass-safe overlay helper.
# ---------------------------------------------------------------------------
_BUILTIN_FIELD_SOURCES = {
    "flops": "flops_forward",
    "bytes": "activation_memory",
    "time": "func_duration",
}
_BUILTIN_PAYLOAD_SOURCES = frozenset({"magnitude", "grad_norm"})
#: Session-time Kineto-join builtin (torchnative W2.3): resolves per-node
#: joined device nanoseconds through the F27 weak trace registry. Requesting
#: it on a trace with no joined session refuses typed
#: (``device_time_unavailable``) -- an explicitly requested inapplicable
#: column raises with cause and remedy, never silent zeros.
_BUILTIN_JOIN_SOURCES = frozenset({"device_time"})
SCALAR_BUILTIN_SOURCES = (
    frozenset(_BUILTIN_FIELD_SOURCES) | _BUILTIN_PAYLOAD_SOURCES | _BUILTIN_JOIN_SOURCES
)

#: Legend notes (fixed wording pinned by tests).
NOTE_NA_UNENCODED = "n/a = unencoded"
NOTE_VARIES = "varies across passes -- unencoded"
NOTE_FIRST_PASS_ONLY = "first-pass-only field -- unencoded on rolled nodes"
# Degenerate-domain honesty (themes memo build item 11): min==max renders
# UNENCODED with the note, never mid-ramp -- a uniform mid-ramp paint is a
# claim ("these differ from an unencoded node") the data cannot support.
NOTE_CONSTANT = "constant value -- unencoded (degenerate domain)"
NOTE_CALLABLE = "value from user callable"
NOTE_PER_PASS_ROLLED = "per-pass field -- unencoded on rolled nodes"
NOTE_LOG_NONPOSITIVE = "values <= 0 -- unencoded under the log transform"

#: Closed color-transform vocabulary (N5 slice shipped with the lens
#: roster): ``linear`` min-max, ``rank`` (ordinal; the v1 perf default --
#: the only measured candidate that cannot degenerate), and scale-invariant
#: ``log`` (disclosed floor: non-positive values unencode). ``log1p`` is
#: rejected from the vocabulary outright (measured no-op on seconds-valued
#: fields; unit-dependent behavior).
COLOR_TRANSFORM_VOCABULARY = ("linear", "rank", "log")

#: Sequential colormap anchors (Okabe-Ito adjacent, colorblind-safe).
LIGHT_RAMP = ("#FFFFFF", "#0072B2")
DARK_RAMP = ("#1F2937", "#56B4E9")

# ---------------------------------------------------------------------------
# SIZE channel constants (wave 1; D4 DEFAULT-APPLIED -- memo 3.1/3.2 C2).
# ---------------------------------------------------------------------------

#: Closed ``scale=`` vocabulary (slate 3.2). Log is REJECTED by design: it
#: flattens 512-vs-4096 and defeats the size motif.
SIZE_SCALE_VOCABULARY = ("sqrt", "linear")

#: Max clamp in value space: encoded node area never exceeds this multiple of
#: the default node area. PROVISIONAL value per the design memo (4.0), to be
#: tuned once at the D4 rendered-candidates session; the min clamp is free
#: (fixedsize=false -- the box only ever GROWS from the label's natural size).
SIZE_BY_MAX_AREA_MULT = 4.0

#: Graphviz's own default node geometry in inches (dot default width=0.75,
#: height=0.5). Emitted sizes are MINIMUMS scaled from this baseline with the
#: default aspect ratio preserved (D4 default mapping C2: area-only,
#: conservative -- no per-rank axis rules that could mislead about which axis
#: is which).
DEFAULT_NODE_WIDTH_IN = 0.75
DEFAULT_NODE_HEIGHT_IN = 0.5

#: The one shape-valued builtin source token, legal for ``size_by`` only
#: (memo 2.2): user callables must return scalars.
SIZE_DIMS_TOKEN = "dims"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an RGB tuple."""

    stripped = color.lstrip("#")
    return (int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16))


def interpolate_hex(start: str, end: str, fraction: float) -> str:
    """Linearly interpolate between two ``#RRGGBB`` colors.

    The ONE colormap interpolation home (generalized from
    ``bundle_diff._interpolate``, which now consumes this helper).

    Parameters
    ----------
    start:
        Start color.
    end:
        End color.
    fraction:
        Interpolation fraction in ``[0, 1]``.

    Returns
    -------
    str
        Interpolated hex color.
    """

    fraction = max(0.0, min(1.0, fraction))
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    values = [
        round(start_value + (end_value - start_value) * fraction)
        for start_value, end_value in zip(start_rgb, end_rgb, strict=True)
    ]
    return "#{:02X}{:02X}{:02X}".format(*values)


@dataclass(frozen=True)
class EncodingChannelRequest:
    """A color source paired with an explicit transform (lens layer door).

    The lens roster's performance rows pass this as ``color_by`` so the
    view-resolved member arrives WITH its rank transform; a bare string or
    callable ``color_by`` keeps the historical linear default.
    """

    source: Any
    transform: str = "linear"
    display_name: str | None = None


@dataclass(frozen=True)
class EncodingChannelSpec:
    """One resolved channel request (option-validation product).

    Parameters
    ----------
    channel:
        Channel kind; v1 supports ``"color"`` only.
    source_kind:
        ``"builtin"``, ``"field"``, or ``"callable"``.
    source:
        The validated user source (token, field name, or callable).
    display_name:
        Human-readable source name for legend disclosure.
    transform:
        Value-to-fraction mapping from :data:`COLOR_TRANSFORM_VOCABULARY`.
    """

    channel: str
    source_kind: str
    source: Any
    display_name: str
    transform: str = "linear"


@dataclass
class EncodingState:
    """Mutable per-draw channel state: prepass products + legend disclosures.

    Created at request resolution, populated exactly once by the Phase-A
    prepass in ``build_render_ir``, and read by Phase B and the legend.
    One state object carries every active channel for the draw: ``spec`` is
    the color channel (``None`` when only size is active) and ``size_spec``
    /``size_scale`` the size channel; the two compose freely (different
    Graphviz attrs, memo 2.4).
    """

    spec: EncodingChannelSpec | None = None
    dark_theme: bool = False
    populated: bool = False
    colors: dict[str, str] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    domain: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)
    aggregation_lines: list[str] = field(default_factory=list)
    eligible_count: int = 0
    # SIZE channel (wave 1). ``sizes`` maps node key -> (width_in, height_in)
    # emitted as Graphviz minimums under fixedsize=false.
    size_spec: EncodingChannelSpec | None = None
    size_scale: str = "sqrt"
    sizes: dict[str, tuple[float, float]] = field(default_factory=dict)
    size_values: dict[str, float] = field(default_factory=dict)
    size_domain: tuple[float, float] | None = None
    size_notes: list[str] = field(default_factory=list)
    size_aggregation_lines: list[str] = field(default_factory=list)
    # STACK (rank) channel (wave 1). ``stack_groups`` holds
    # (rank_key_repr, member node names) rows resolved at the prepass and
    # emitted as rank=same subgraphs under newrank=true.
    stack_spec: EncodingChannelSpec | None = None
    stack_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    stack_notes: list[str] = field(default_factory=list)
    # Skin-supplied 3-anchor ramp (N4): low/mid/high. None keeps the
    # historical 2-anchor module ramps.
    ramp_anchors: tuple[str, str, str] | None = None

    def note(self, text: str) -> None:
        """Record a color-channel legend note once."""

        if text not in self.notes:
            self.notes.append(text)

    def size_note(self, text: str) -> None:
        """Record a size-channel legend note once."""

        if text not in self.size_notes:
            self.size_notes.append(text)

    def stack_note(self, text: str) -> None:
        """Record a stack-channel legend note once."""

        if text not in self.stack_notes:
            self.stack_notes.append(text)

    def fillcolor_for(self, node: Any) -> str | None:
        """Phase B: return the precomputed fill for ``node`` (None = unencoded)."""

        return self.colors.get(_node_key(node))

    def size_for(self, node: Any) -> tuple[float, float] | None:
        """Phase B: return the precomputed (width, height) for ``node``."""

        return self.sizes.get(_node_key(node))

    def active_channels(self) -> tuple[str, ...]:
        """Return the active channel kwarg names for fence/refusal messages."""

        channels: list[str] = []
        if self.spec is not None:
            channels.append("color_by")
        if self.size_spec is not None:
            channels.append("size_by")
        if self.stack_spec is not None:
            channels.append("stack_by")
        return tuple(channels)

    @property
    def ramp(self) -> tuple[str, str]:
        """Return the theme-aware sequential ramp endpoints."""

        if self.ramp_anchors is not None:
            return (self.ramp_anchors[0], self.ramp_anchors[2])
        return DARK_RAMP if self.dark_theme else LIGHT_RAMP

    def ramp_color(self, fraction: float) -> str:
        """Map a [0, 1] fraction through the (possibly 3-anchor) ramp.

        With skin anchors the mapping is piecewise low->mid->high, so a
        future diverging map is a mapping change, not a schema change.
        """

        if self.ramp_anchors is not None:
            low, mid, high = self.ramp_anchors
            if fraction <= 0.5:
                return interpolate_hex(low, mid, fraction * 2.0)
            return interpolate_hex(mid, high, (fraction - 0.5) * 2.0)
        start, end = self.ramp
        return interpolate_hex(start, end, fraction)


def _require_channel_spec(state: EncodingState) -> EncodingChannelSpec:
    """Return the active color spec or fail on an internal phase-order bug."""

    if state.spec is None:
        raise RuntimeError("color encoding resolution requires an active color spec")
    return state.spec


def _require_size_spec(state: EncodingState) -> EncodingChannelSpec:
    """Return the active size spec or fail on an internal phase-order bug."""

    if state.size_spec is None:
        raise RuntimeError("size encoding resolution requires an active size spec")
    return state.size_spec


def _encoding_error(
    problem: str, *, code: str, remedy: str, **context: Any
) -> InvalidArgumentError:
    """Build a typed encoding refusal."""

    return InvalidArgumentError(problem, code=code, remedy=remedy, **context)


def resolve_color_by(color_by: Any) -> EncodingChannelSpec | None:
    """Validate ``color_by`` at option validation, before any render work.

    Parameters
    ----------
    color_by:
        ``None``, a field-name string, a scalar builtin token, or a callable
        ``node -> value``.

    Returns
    -------
    EncodingChannelSpec | None
        Resolved channel spec, or ``None`` when the channel is inactive.

    Raises
    ------
    InvalidArgumentError
        ``encoding_source_invalid`` for an unknown field name / builtin
        token or a non-string non-callable source.
    """

    if color_by is None:
        return None
    if isinstance(color_by, EncodingChannelRequest):
        if color_by.transform not in COLOR_TRANSFORM_VOCABULARY:
            raise _encoding_error(
                f"unknown color transform {color_by.transform!r}; the closed "
                f"vocabulary is {', '.join(COLOR_TRANSFORM_VOCABULARY)} "
                "(log1p is rejected by design: unit-dependent, measured no-op)",
                code="encoding_transform_invalid",
                remedy="pass one of the closed transform tokens",
                argument="color_by",
            )
        inner = resolve_color_by(color_by.source)
        if inner is None:
            raise _encoding_error(
                "EncodingChannelRequest.source must name an active source",
                code="encoding_source_invalid",
                remedy="pass a field name, builtin token, or callable as the source",
                argument="color_by",
            )
        return EncodingChannelSpec(
            channel=inner.channel,
            source_kind=inner.source_kind,
            source=inner.source,
            display_name=color_by.display_name or inner.display_name,
            transform=color_by.transform,
        )
    if callable(color_by) and not isinstance(color_by, str):
        name = getattr(color_by, "__name__", type(color_by).__name__)
        return EncodingChannelSpec(
            channel="color",
            source_kind="callable",
            source=color_by,
            display_name=f"callable {name}",
        )
    if isinstance(color_by, str):
        normalized = color_by.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in SCALAR_BUILTIN_SOURCES:
            return EncodingChannelSpec(
                channel="color",
                source_kind="builtin",
                source=normalized,
                display_name=normalized,
            )
        from ..constants import LAYER_PASS_LOG_FIELD_ORDER

        if color_by in LAYER_SOURCE_ROWS or color_by in LAYER_PASS_LOG_FIELD_ORDER:
            return EncodingChannelSpec(
                channel="color",
                source_kind="field",
                source=color_by,
                display_name=color_by,
            )
        raise _encoding_error(
            f"color_by source {color_by!r} is not a known record field, scalar "
            "builtin, or callable",
            code="encoding_source_invalid",
            remedy=(
                "pass a Layer/Op field name, one of the scalar builtins "
                f"({', '.join(sorted(SCALAR_BUILTIN_SOURCES))}), or a callable node -> value"
            ),
            argument="color_by",
        )
    raise _encoding_error(
        f"color_by must be a field-name string, builtin token, or callable; "
        f"received {type(color_by).__name__}",
        code="encoding_source_invalid",
        remedy="pass a string source name or a callable node -> value",
        argument="color_by",
    )


def resolve_size_by(size_by: Any) -> EncodingChannelSpec | None:
    """Validate ``size_by`` at option validation, before any render work.

    Parameters
    ----------
    size_by:
        ``None``, a field-name string, the closed ``"dims"`` shape token, or
        a callable ``node -> scalar``.

    Returns
    -------
    EncodingChannelSpec | None
        Resolved channel spec, or ``None`` when the channel is inactive.

    Raises
    ------
    InvalidArgumentError
        ``encoding_source_invalid`` for an unknown field name or a
        non-string non-callable source.
    """

    if size_by is None:
        return None
    if callable(size_by) and not isinstance(size_by, str):
        name = getattr(size_by, "__name__", type(size_by).__name__)
        return EncodingChannelSpec(
            channel="size",
            source_kind="callable",
            source=size_by,
            display_name=f"callable {name}",
        )
    if isinstance(size_by, str):
        normalized = size_by.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _BUILTIN_JOIN_SOURCES:
            return EncodingChannelSpec(
                channel="size",
                source_kind="builtin",
                source=normalized,
                display_name=normalized,
            )
        if normalized == SIZE_DIMS_TOKEN:
            return EncodingChannelSpec(
                channel="size",
                source_kind="dims",
                source=SIZE_DIMS_TOKEN,
                display_name=SIZE_DIMS_TOKEN,
            )
        from ..constants import LAYER_PASS_LOG_FIELD_ORDER

        if size_by in LAYER_SOURCE_ROWS or size_by in LAYER_PASS_LOG_FIELD_ORDER:
            return EncodingChannelSpec(
                channel="size",
                source_kind="field",
                source=size_by,
                display_name=size_by,
            )
        raise _encoding_error(
            f"size_by source {size_by!r} is not a known record field, the "
            '"dims" builtin, or a callable',
            code="encoding_source_invalid",
            remedy=(
                'pass a Layer/Op field name, "dims" for shape-driven sizing, '
                "or a callable node -> scalar"
            ),
            argument="size_by",
        )
    raise _encoding_error(
        f'size_by must be a field-name string, "dims", or callable; '
        f"received {type(size_by).__name__}",
        code="encoding_source_invalid",
        remedy='pass a string source name, "dims", or a callable node -> scalar',
        argument="size_by",
    )


def resolve_size_scale(scale: Any, *, size_by_active: bool) -> str:
    """Validate ``scale=`` at option validation (closed vocabulary).

    Parameters
    ----------
    scale:
        ``None`` (default -> ``"sqrt"``), ``"sqrt"``, or ``"linear"``.
    size_by_active:
        Whether a ``size_by`` source was supplied.

    Returns
    -------
    str
        The effective scale.

    Raises
    ------
    InvalidArgumentError
        ``scale_requires_size_by`` when ``scale`` is supplied without
        ``size_by``; ``encoding_scale_invalid`` for an unknown scale token.
    """

    if scale is None:
        return "sqrt"
    if not size_by_active:
        raise _encoding_error(
            "scale= was supplied without size_by; the scale transform applies "
            "to the size channel only",
            code="scale_requires_size_by",
            remedy="pass size_by= alongside scale=, or drop scale=",
            argument="scale",
        )
    if scale not in SIZE_SCALE_VOCABULARY:
        raise _encoding_error(
            f"scale must be one of {SIZE_SCALE_VOCABULARY}; received {scale!r} "
            "(log is rejected by design: it flattens 512-vs-4096 and defeats "
            "the size motif)",
            code="encoding_scale_invalid",
            remedy="pass scale='sqrt' (default) or scale='linear'",
            argument="scale",
        )
    return str(scale)


def _node_key(node: Any) -> str:
    """Stable per-draw key for a rendered record.

    Rolled Layer nodes key by ``layer_label``; per-pass Op nodes by their
    pass-qualified ``label``. The prepass and Phase B call this on the SAME
    record object, so the keying is consistent within one draw.
    """

    from ..data_classes.layer import Layer

    if isinstance(node, Layer):
        layer_label = getattr(node, "layer_label", None)
        if isinstance(layer_label, str):
            return layer_label
    from ..utils._multipass_access import get_multipass_attr

    label = get_multipass_attr(node, "label", None, multipass=None)
    if isinstance(label, str):
        return label
    layer_label = getattr(node, "layer_label", None)
    return layer_label if isinstance(layer_label, str) else str(id(node))


def _is_rolled_multipass(node: Any) -> bool:
    """Return whether ``node`` is a rolled multi-pass aggregate Layer."""

    from ..data_classes.layer import Layer

    return isinstance(node, Layer) and len(node.ops) > 1


def is_record_derived_image_node(trace: Trace, node: Any) -> bool:
    """THE closed image-origin predicate for pre-user image nodes (memo 2.4(i)).

    Covers all three record-derived image mechanisms: the visualizer_path
    branch, the annotation-image branch, and the raw-input montage input
    node. Image nodes are excluded from channel encoding AND from the
    normalization domain. Any future record-derived image producer must
    extend THIS predicate and its test pair, never add a local check
    elsewhere (classification drift between the trace-bearing prepass and
    the NodeSpec funnel is this design's named recurring risk).
    """

    visualizer_path = getattr(node, "visualizer_path", None)
    if isinstance(visualizer_path, str) and visualizer_path.lower().endswith(".png"):
        return True
    from ._render_nodes import _annotation_image_path_for_node

    if _annotation_image_path_for_node(trace, node) is not None:
        return True
    # Raw-input visual branch (montage / preview): input nodes whose label is
    # decorated from trace.raw_input by the post-user raw merge.
    return bool(getattr(node, "is_input", False)) and getattr(trace, "raw_input", None) is not None


def _coerce_scalar(
    state: EncodingState,
    node: Any,
    value: Any,
    *,
    argument: str = "color_by",
) -> float | None:
    """Apply the closed value/type rules (memo 2.2) to one resolved value.

    ``argument`` selects the owning channel so color and size share ONE
    closed failure table.
    """

    channel_spec = state.size_spec if argument == "size_by" else state.spec
    record_note = state.size_note if argument == "size_by" else state.note
    display_name = channel_spec.display_name if channel_spec is not None else argument
    if value is None:
        record_note(NOTE_NA_UNENCODED)
        return None
    if isinstance(value, bool):
        raise _encoding_error(
            f"{argument} source {display_name!r} produced a bool on node "
            f"{_node_key(node)!r}; a truth value is not an encodable magnitude",
            code="encoding_value_invalid",
            remedy="encode a numeric field, or map the bool to a number in a callable",
            argument=argument,
        )
    if isinstance(value, (int, float)):
        as_float = float(value)
        if not math.isfinite(as_float):
            record_note(NOTE_NA_UNENCODED)
            return None
        return as_float
    if argument == "size_by" and _is_shape_valued(value):
        raise _encoding_error(
            f"size_by source {display_name!r} produced a shape "
            f'({type(value).__name__}) on node {_node_key(node)!r}; "dims" is '
            "the only shape-valued size source",
            code="encoding_value_invalid",
            remedy='pass size_by="dims" for shape-driven sizing, or return a scalar',
            argument=argument,
        )
    numel = getattr(value, "numel", None)
    item = getattr(value, "item", None)
    if callable(numel) and callable(item):
        if numel() == 1:
            # Documented: float(x.item()) forces a device sync the user
            # opted into by passing a tensor-returning source.
            return _coerce_scalar(
                state,
                node,
                item(),
                argument=argument,
            )
        raise _encoding_error(
            f"{argument} source {display_name!r} produced a non-scalar "
            f"tensor on node {_node_key(node)!r}",
            code="encoding_value_invalid",
            remedy="reduce the tensor to one element (e.g. .mean()) in the callable",
            argument=argument,
        )
    raise _encoding_error(
        f"{argument} source {display_name!r} produced "
        f"{type(value).__name__!r} on node {_node_key(node)!r}; encoding channels "
        "accept python ints/floats and 1-element tensors",
        code="encoding_value_invalid",
        remedy="pick a numeric source or convert the value in a callable",
        argument=argument,
    )


def _is_shape_valued(value: Any) -> bool:
    """Return whether ``value`` is a shape (tuple/list/torch.Size of ints)."""

    if not isinstance(value, (tuple, list)):
        return False
    return all(isinstance(entry, int) and not isinstance(entry, bool) for entry in value)


def _varying_marker(node: Any) -> dict[str, Any]:
    """Return the reconciler's variation marker for a rolled Layer."""

    annotations = getattr(node, "annotations", None)
    if isinstance(annotations, dict):
        marker = annotations.get("varying_across_passes")
        if isinstance(marker, dict):
            return marker
    return {}


def _mirror_backed_op_field(field_name: str) -> str | None:
    """Return the Op field backing a mirrored Layer name, if any."""

    from ..data_classes._layer_spec import _LAYER_MIRROR_SPEC

    entry = _LAYER_MIRROR_SPEC.get(field_name)
    return entry[0] if entry is not None else None


def _rolled_reconciled(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Reconciled family: marker-varying -> unencode; unmarked -> exact encode."""

    if field_name in _varying_marker(node):
        state.note(NOTE_VARIES)
        return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_summed(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Summed family: exact aggregate, mandatory aggregation legend line."""

    value = _coerce_scalar(state, node, getattr(node, field_name, None))
    if value is not None:
        wording = _AGGREGATE_LEGEND_WORDING.get(field_name, "total across passes")
        line = f"{field_name}: {wording} on rolled nodes"
        if line not in state.aggregation_lines:
            state.aggregation_lines.append(line)
    return value


def _rolled_mirrored(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Mirrored per-call numeric: never certified single-valued -> unencode."""

    del node, field_name
    state.note(NOTE_FIRST_PASS_ONLY)
    return None


def _rolled_derived(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Derived composite: inherit the most conservative input verdict."""

    marker = _varying_marker(node)
    for input_name in _DERIVED_COMPOSITE_INPUTS[field_name]:
        if LAYER_SOURCE_ROWS.get(input_name) == ROW_MIRRORED_PER_CALL:
            state.note(NOTE_FIRST_PASS_ONLY)
            return None
        if input_name in marker:
            state.note(NOTE_VARIES)
            return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_structural(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Structurally uniform: encode, with a defensive op-backed equality check.

    The structural premise is the live code's own declaration; a
    counterexample degrades honestly instead of painting a dishonest uniform
    value.
    """

    op_field = _mirror_backed_op_field(field_name)
    if op_field is not None:
        per_pass = [getattr(op, op_field, None) for op in node.ops.values()]
        if len({repr(value) for value in per_pass}) > 1:
            state.note(NOTE_VARIES)
            return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_per_pass(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Per-pass attribute on a rolled multi-pass node: honest unencode.

    A per-pass field has one value PER PASS; a rolled node stands for every
    pass at once, so any single value (the old pass-1 read) is a claim the
    node cannot carry -- the measured min:1/max:1 uniform mid-ramp defect
    (themes memo build item 11). Unencode with the note, never resolve to
    pass 1.
    """

    del node, field_name
    state.note(NOTE_PER_PASS_ROLLED)
    return None


def _rolled_wrong_type_read(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Shape / non-numeric rows: read and let the wrong-type rule speak."""

    from ..utils._multipass_access import get_multipass_attr

    return _coerce_scalar(state, node, get_multipass_attr(node, field_name, None, multipass=None))


_ROLLED_ROW_HANDLERS = {
    ROW_RECONCILED: _rolled_reconciled,
    ROW_SUMMED: _rolled_summed,
    ROW_MIRRORED_PER_CALL: _rolled_mirrored,
    ROW_DERIVED: _rolled_derived,
    ROW_STRUCTURAL: _rolled_structural,
    ROW_PER_PASS: _rolled_per_pass,
    ROW_SHAPE: _rolled_wrong_type_read,
    ROW_NON_NUMERIC: _rolled_wrong_type_read,
}


def _node_pass_ops(node: Any) -> tuple[Any, ...]:
    """Return the op records behind one render node (Layer or Op)."""

    ops = getattr(node, "ops", None)
    if isinstance(ops, dict):
        return tuple(ops.values())
    if isinstance(ops, (list, tuple)):
        return tuple(ops)
    return (node,)


def _sum_node_device_ns(result: Any, node: Any) -> tuple[int, bool]:
    """Sum a node's pass-qualified joined device ns; found=False reads n/a."""

    total = 0
    found = False
    for op in _node_pass_ops(node):
        label = getattr(op, "label", None)
        if label is None:
            continue
        value = result.op_device_ns.get(str(label))
        if value is not None:
            total += int(value)
            found = True
    return total, found


def _resolve_device_time(
    state: EncodingState, trace: Any, node: Any, *, argument: str
) -> float | None:
    """Resolve one node's joined device time (torchnative W2.3).

    The value is the sum of exact-attribution device nanoseconds over the
    node's pass-qualified ops; on rolled multi-pass nodes that is an exact
    cross-pass total and lands the mandatory aggregation legend line. A node
    the join attributed no kernels to reads honest n/a, never zero.
    """

    from ..observability._join import require_availability
    from ..observability._native_profile import join_result_for

    result = join_result_for(trace)
    if result is None:
        from ..observability._errors import ProfilerSessionError

        raise ProfilerSessionError(
            f"{argument}='device_time' is unavailable: this trace carries no "
            "Kineto join (device time exists only for captures run under an "
            "owned profiler session).",
            code="device_time_unavailable",
            remedy=(
                "capture through torchlens.observability.native_profile on a "
                "CUDA host, then draw the returned result.trace"
            ),
        )
    require_availability(result, needs=f"{argument}='device_time'")
    total, found = _sum_node_device_ns(result, node)
    if not found:
        if argument == "size_by":
            state.size_note(NOTE_NA_UNENCODED)
        else:
            state.note(NOTE_NA_UNENCODED)
        return None
    if _is_rolled_multipass(node):
        line = "device_time: total across passes on rolled nodes"
        if argument == "size_by":
            if line not in state.size_aggregation_lines:
                state.size_aggregation_lines.append(line)
        elif line not in state.aggregation_lines:
            state.aggregation_lines.append(line)
    return float(total)


def _resolve_field_on_rolled(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Resolve a FIELD source on a rolled multi-pass Layer per the allowlist."""

    row = LAYER_SOURCE_ROWS.get(field_name)
    if row is None:
        raise _encoding_error(
            f"color_by source {field_name!r} has no declared rolled-aggregate "
            f"semantics row and cannot resolve on rolled multi-pass node "
            f"{_node_key(node)!r}",
            code="encoding_source_invalid",
            remedy=(
                "unroll the graph (vis_mode='unrolled'), use a callable that "
                "asserts its own aggregate semantics, or classify the field in "
                "the rolled-aggregate allowlist"
            ),
            argument="color_by",
        )
    return _ROLLED_ROW_HANDLERS[row](state, node, field_name)


def _resolve_payload_builtin(state: EncodingState, node: Any, token: str) -> float | None:
    """Resolve one payload-builtin (magnitude/grad_norm) color value."""

    from .overlays import builtin_overlay_value

    value = builtin_overlay_value(node, token)
    if value is None:
        state.note(NOTE_NA_UNENCODED)
        return None
    return _coerce_scalar(state, node, value)


def _resolve_source_value(state: EncodingState, trace: Trace, node: Any) -> float | None:
    """Resolve one node's raw color value (Phase A, exactly once per node)."""

    spec = _require_channel_spec(state)
    if spec.source_kind == "callable":
        try:
            value = spec.source(node)
        except Exception as error:
            raise _encoding_error(
                f"color_by callable {spec.display_name!r} raised on node "
                f"{_node_key(node)!r}: {error}",
                code="encoding_callable_error",
                remedy=(
                    "fix the callable; per-pass reads off rolled aggregates trip "
                    "the multipass tripwire -- read Layer.ops for per-pass truth"
                ),
                argument="color_by",
            ) from error
        state.note(NOTE_CALLABLE)
        return _coerce_scalar(state, node, value)

    if spec.source_kind == "builtin":
        token = spec.source
        if token in _BUILTIN_JOIN_SOURCES or token in _BUILTIN_PAYLOAD_SOURCES:
            return (
                _resolve_device_time(state, trace, node, argument="color_by")
                if token in _BUILTIN_JOIN_SOURCES
                else _resolve_payload_builtin(state, node, token)
            )
        field_name = _BUILTIN_FIELD_SOURCES[token]
    else:
        field_name = spec.source

    if _is_rolled_multipass(node):
        return _resolve_field_on_rolled(state, node, field_name)
    from ..utils._multipass_access import get_multipass_attr

    value = get_multipass_attr(node, field_name, None, multipass=None)
    if value is None:
        state.note(NOTE_NA_UNENCODED)
        return None
    return _coerce_scalar(state, node, value)


# ---------------------------------------------------------------------------
# SIZE channel resolution (wave 1). THE D4 DETECTOR (memo 3.1): the refusal
# predicate is LAYER-RECORD truth -- a rolled multi-pass Layer refuses
# size_by when the size source cannot be certified single-valued on it
# (marker-varying reconciled fields, mirrored/unreconciled per-call
# projections). Verdicts key on the source's DECLARED 2.3b row, never on
# string-prefix accident. Callables bypass the table (the user asserts their
# own aggregate semantics; disclosed in the legend).
# ---------------------------------------------------------------------------


def _size_rolled_refusal(node: Any, field_name: str, why: str) -> InvalidArgumentError:
    """Build the typed rolled-varying size refusal (widened semantics)."""

    detail = ""
    shape_summary = getattr(node, "shape_summary", None)
    if isinstance(shape_summary, str) and shape_summary:
        detail = f" (shapes across passes: {shape_summary})"
    return _encoding_error(
        f"size_by source {field_name!r} cannot be certified single-valued on "
        f"rolled multi-pass node {_node_key(node)!r}: {why}{detail}",
        code="size_by_rolled_varying",
        remedy=(
            "unroll the graph (vis_mode='unrolled') to size each pass by its "
            "own value, choose a cross-pass total (total_*), or pass a "
            "callable asserting your own aggregate semantics"
        ),
        argument="size_by",
    )


def _require_size_rolled_row(node: Any, field_name: str) -> str:
    """Return a declared rolled-size row or raise the typed source refusal."""

    row = LAYER_SOURCE_ROWS.get(field_name)
    if row is None:
        raise _encoding_error(
            f"size_by source {field_name!r} has no declared rolled-aggregate "
            f"semantics row and cannot resolve on rolled multi-pass node "
            f"{_node_key(node)!r}",
            code="encoding_source_invalid",
            remedy=(
                "unroll the graph (vis_mode='unrolled'), use a callable that "
                "asserts its own aggregate semantics, or classify the field in "
                "the rolled-aggregate allowlist"
            ),
            argument="size_by",
        )
    return row


def _resolve_size_field_on_rolled(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Resolve a FIELD size source on a rolled multi-pass Layer.

    Size REFUSES where color degrades: every verdict below that unencodes for
    color is a typed ``size_by_rolled_varying`` refusal here (memo 3.1).
    """

    _require_size_spec(state)
    row = _require_size_rolled_row(node, field_name)
    coerce = _size_coerce(state)
    marker = _varying_marker(node)
    if row == ROW_RECONCILED:
        if field_name in marker:
            raise _size_rolled_refusal(
                node, field_name, "the stored aggregate is a per-pass maximum (upper bound)"
            )
        return coerce(node, getattr(node, field_name, None))
    if row == ROW_SUMMED:
        value = coerce(node, getattr(node, field_name, None))
        if value is not None:
            wording = _AGGREGATE_LEGEND_WORDING.get(field_name, "total across passes")
            line = f"{field_name}: {wording} on rolled nodes"
            if line not in state.size_aggregation_lines:
                state.size_aggregation_lines.append(line)
        return value
    if row == ROW_MIRRORED_PER_CALL:
        raise _size_rolled_refusal(
            node, field_name, "the stored value is an unreconciled first-pass projection"
        )
    if row == ROW_DERIVED:
        _validate_derived_size_inputs(node, field_name, marker)
        return coerce(node, getattr(node, field_name, None))
    if row == ROW_STRUCTURAL:
        _validate_structural_size_value(node, field_name)
        return coerce(node, getattr(node, field_name, None))
    if row == ROW_PER_PASS:
        raise _size_rolled_refusal(
            node, field_name, "the field is per-pass with no aggregate meaning"
        )
    # ROW_SHAPE / ROW_NON_NUMERIC: read and let the wrong-type rule speak
    # (a shape field names its own remedy: size_by="dims").
    from ..utils._multipass_access import get_multipass_attr

    return coerce(node, get_multipass_attr(node, field_name, None, multipass=None))


def _validate_derived_size_inputs(node: Any, field_name: str, marker: dict[str, Any]) -> None:
    """Refuse a derived rolled size whose input is not provably single-valued."""

    for input_name in _DERIVED_COMPOSITE_INPUTS[field_name]:
        if LAYER_SOURCE_ROWS.get(input_name) == ROW_MIRRORED_PER_CALL:
            raise _size_rolled_refusal(
                node,
                field_name,
                f"input {input_name!r} is an unreconciled first-pass projection",
            )
        if input_name in marker:
            raise _size_rolled_refusal(node, field_name, f"input {input_name!r} varies")


def _validate_structural_size_value(node: Any, field_name: str) -> None:
    """Refuse a structural rolled size when its per-pass source values disagree."""

    op_field = _mirror_backed_op_field(field_name)
    if op_field is None:
        return
    per_pass = [getattr(op, op_field, None) for op in node.ops.values()]
    if len({repr(value) for value in per_pass}) > 1:
        raise _size_rolled_refusal(node, field_name, "per-pass values disagree (defensive check)")


def _size_coerce(state: EncodingState) -> Any:
    """Return a size-channel scalar coercion closure."""

    _require_size_spec(state)

    def coerce(node: Any, value: Any) -> float | None:
        """Coerce one node's ``size_by`` value to a float, or None if unusable."""
        return _coerce_scalar(
            state,
            node,
            value,
            argument="size_by",
        )

    return coerce


def _resolve_node_shape(state: EncodingState, node: Any) -> tuple[int, ...] | None:
    """Resolve the ``"dims"`` shape for one node (typed shape path, memo 2.2).

    ORDERING PIN: on a rolled multi-pass node the 3.1 refusal fires BEFORE
    resolution, so the string-bearing honest aggregates minted for varying
    rolled layers ("3..4" range tokens, ``("varies",)``) are unreachable by
    construction; a non-int entry reaching resolution anyway is a defensive
    ``encoding_value_invalid``.
    """

    if _is_rolled_multipass(node):
        if "shape" in _varying_marker(node):
            raise _size_rolled_refusal(node, "shape", "the output shape varies across passes")
        shape = getattr(node, "shape", None)
    else:
        from ..utils._multipass_access import get_multipass_attr

        shape = get_multipass_attr(node, "shape", None, multipass=None)
    if shape is None:
        state.size_note(NOTE_NA_UNENCODED)
        return None
    entries = tuple(shape)
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise _encoding_error(
                f'size_by="dims" resolved a non-integer shape entry {entry!r} on '
                f"node {_node_key(node)!r}",
                code="encoding_value_invalid",
                remedy="unroll the graph for per-pass shapes",
                argument="size_by",
            )
    return entries


def _non_batch_numel(shape: tuple[int, ...]) -> float:
    """D4 default mapping C2: numel of the non-batch shape (batch = dim 0).

    Rank <= 1 resolves to 1 (default box) -- conservative: a rank-1 dim may
    or may not be a batch axis, and C2 never misleads about which axis is
    which.
    """

    if len(shape) <= 1:
        return 1.0
    numel = 1.0
    for dim in shape[1:]:
        numel *= max(1, dim)
    return numel


def _resolve_size_source_value(state: EncodingState, trace: Trace, node: Any) -> float | None:
    """Resolve one node's raw size value (Phase A, exactly once per node)."""

    spec = _require_size_spec(state)
    if spec.source_kind == "callable":
        try:
            value = spec.source(node)
        except Exception as error:
            raise _encoding_error(
                f"size_by callable {spec.display_name!r} raised on node "
                f"{_node_key(node)!r}: {error}",
                code="encoding_callable_error",
                remedy=(
                    "fix the callable; per-pass reads off rolled aggregates trip "
                    "the multipass tripwire -- read Layer.ops for per-pass truth"
                ),
                argument="size_by",
            ) from error
        state.size_note(NOTE_CALLABLE)
        return _size_coerce(state)(node, value)

    if spec.source_kind == "builtin":
        value = _resolve_device_time(state, trace, node, argument="size_by")
        return None if value is None else _size_coerce(state)(node, value)

    if spec.source_kind == "dims":
        shape = _resolve_node_shape(state, node)
        return None if shape is None else _non_batch_numel(shape)

    field_name = spec.source
    if _is_rolled_multipass(node):
        return _resolve_size_field_on_rolled(state, node, field_name)
    from ..utils._multipass_access import get_multipass_attr

    value = get_multipass_attr(node, field_name, None, multipass=None)
    if value is None:
        state.size_note(NOTE_NA_UNENCODED)
        return None
    return _size_coerce(state)(node, value)


def _compute_size_geometry(state: EncodingState) -> None:
    """Map collected size values to (width, height) minimums (D4 default C2).

    ``scale="sqrt"`` (default) compresses the dynamic range; ``"linear"``
    keeps area ~ value for the literal motif. Normalization is min-max over
    the transformed finite values of visible nodes; the encoded area spans
    [1x .. SIZE_BY_MAX_AREA_MULT x] the default node area with the default
    aspect ratio preserved. Fonts NEVER scale.
    """

    if not state.size_values:
        state.size_note(NOTE_NA_UNENCODED)
        return
    if state.size_scale == "sqrt":
        for key, value in state.size_values.items():
            if value < 0:
                raise _encoding_error(
                    f"size_by produced a negative value ({value}) on node {key!r}; "
                    "scale='sqrt' requires non-negative magnitudes",
                    code="encoding_value_invalid",
                    remedy="use scale='linear' or map values to magnitudes in a callable",
                    argument="size_by",
                )
        transformed = {key: math.sqrt(value) for key, value in state.size_values.items()}
    else:
        transformed = dict(state.size_values)
    low = min(transformed.values())
    high = max(transformed.values())
    if low == high:
        # Degenerate domain: unencoded with the note, like the color channel.
        state.size_note(NOTE_CONSTANT)
        return
    state.size_domain = (min(state.size_values.values()), max(state.size_values.values()))
    default_area = DEFAULT_NODE_WIDTH_IN * DEFAULT_NODE_HEIGHT_IN
    aspect = DEFAULT_NODE_WIDTH_IN / DEFAULT_NODE_HEIGHT_IN
    span = high - low
    fractions = {key: (value - low) / span for key, value in transformed.items()}
    for key, fraction in fractions.items():
        area = default_area * (1.0 + fraction * (SIZE_BY_MAX_AREA_MULT - 1.0))
        width = math.sqrt(area * aspect)
        height = math.sqrt(area / aspect)
        state.sizes[key] = (round(width, 3), round(height, 3))


def _collect_encoding_values(state: EncodingState, trace: Trace, universe: Any) -> dict[str, float]:
    """Collect color and size values over the visible eligible node universe."""

    raw_values: dict[str, float] = {}
    for unit in universe.units:
        emission = unit.emission
        if emission.kind != "raw_op" or emission.node is None:
            continue
        node = emission.node
        if is_record_derived_image_node(trace, node):
            continue
        state.eligible_count += 1
        if state.spec is not None:
            value = _resolve_source_value(state, trace, node)
            if value is not None:
                raw_values[_node_key(node)] = value
        if state.size_spec is not None:
            size_value = _resolve_size_source_value(state, trace, node)
            if size_value is not None:
                state.size_values[_node_key(node)] = size_value
    return raw_values


def _transform_fractions(state: EncodingState, raw_values: dict[str, float]) -> dict[str, float]:
    """Map raw values to [0, 1] fractions under the active transform.

    ``rank`` is ordinal over the DISTINCT sorted values (ties share a rank
    fraction; invariant under any monotone unit change by construction).
    ``log`` unencodes non-positive values with the disclosed floor note.
    """

    transform = state.spec.transform if state.spec is not None else "linear"
    low = min(raw_values.values())
    high = max(raw_values.values())
    if transform == "rank":
        distinct = sorted(set(raw_values.values()))
        denominator = max(len(distinct) - 1, 1)
        rank_of = {value: index / denominator for index, value in enumerate(distinct)}
        return {key: rank_of[value] for key, value in raw_values.items()}
    if transform == "log":
        import math

        positive = {key: value for key, value in raw_values.items() if value > 0}
        if len(positive) < len(raw_values):
            state.note(NOTE_LOG_NONPOSITIVE)
        if not positive:
            return {}
        log_low = math.log(min(positive.values()))
        log_high = math.log(max(positive.values()))
        span = log_high - log_low
        if span == 0:
            return {}
        return {key: (math.log(value) - log_low) / span for key, value in positive.items()}
    span = high - low
    if span == 0:
        return {}
    return {key: (value - low) / span for key, value in raw_values.items()}


def _normalize_color_values(state: EncodingState, raw_values: dict[str, float]) -> None:
    """Normalize collected color values into the configured sequential ramp.

    Degenerate domains (min == max) render UNENCODED with the note, never
    mid-ramp (themes memo build item 11: a measured shipped defect -- a
    6-pass rolled trace reported min:1/max:1 and painted uniform mid-ramp).
    """

    if not raw_values:
        state.note(NOTE_NA_UNENCODED)
        return
    low = min(raw_values.values())
    high = max(raw_values.values())
    state.values = raw_values
    if low == high:
        state.note(NOTE_CONSTANT)
        return
    state.domain = (low, high)
    fractions = _transform_fractions(state, raw_values)
    if not fractions:
        return
    if state.spec is not None and state.spec.transform == "log":
        # The ramp spans the encoded (positive) values; values <= 0 are unencoded.
        encoded = [raw_values[key] for key in fractions]
        state.domain = (min(encoded), max(encoded))
    state.colors = {key: state.ramp_color(fraction) for key, fraction in fractions.items()}


def populate_encoding_state(state: EncodingState, trace: Trace, universe: Any) -> None:
    """PHASE A: collect values over the visible-node universe and normalize.

    Runs exactly once per draw (``build_render_ir`` calls it before any
    per-node spec resolution). User callables are invoked exactly once per
    visible eligible node here; Phase B only reads the precomputed map.
    """

    if state.populated:
        return
    state.populated = True
    raw_values = _collect_encoding_values(state, trace, universe)

    if state.size_spec is not None:
        _compute_size_geometry(state)

    if state.stack_spec is not None:
        from ._stacking import compute_stack_groups

        compute_stack_groups(state, trace, universe)

    if state.spec is None:
        return
    _normalize_color_values(state, raw_values)


def _format_domain_value(state: EncodingState, value: float) -> str:
    """Format a domain endpoint for the legend (builtin-aware)."""

    if state.spec is not None and state.spec.source_kind == "builtin":
        from .overlays import format_overlay_value

        formatted = format_overlay_value(state.spec.source, value)
        return formatted.split(": ", 1)[-1]
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def _color_legend_rows(state: EncodingState) -> list[Any]:
    """Return color-channel title and ramp rows for the disclosure legend."""

    from .node_spec import NodeSpec

    if state.spec is None:
        return []
    transform_wording = {
        "linear": "linear min-max",
        "rank": "rank mapping (ordinal, not ratio)",
        "log": "log scale (scale-invariant; values <= 0 unencoded)",
    }[state.spec.transform]
    title_lines = [
        f"color_by: {state.spec.display_name}",
        transform_wording,
        # The coverage line falls out of the same computation (N13 slice):
        # a legend may never advertise a scale over zero encoded nodes.
        f"encoded {len(state.colors)} of {state.eligible_count} eligible nodes",
        *state.aggregation_lines,
        *state.notes,
    ]
    rows = [NodeSpec(lines=title_lines, shape="box", style="filled,rounded")]
    if state.domain is None or not state.colors:
        return rows
    low, high = state.domain
    # The 0.5 swatch encodes the geometric mean under log and the midpoint
    # under linear; under rank it encodes the middle rank, not this midpoint.
    mid = math.sqrt(low * high) if state.spec.transform == "log" else (low + high) / 2.0
    for tag, fraction, value in (("min", 0.0, low), ("mid", 0.5, mid), ("max", 1.0, high)):
        rows.append(
            NodeSpec(
                lines=[f"{tag}: {_format_domain_value(state, value)}"],
                shape="box",
                fillcolor=state.ramp_color(fraction),
            )
        )
    return rows


def _non_color_legend_rows(state: EncodingState) -> list[Any]:
    """Return active size and stack disclosure rows."""

    from .node_spec import NodeSpec

    rows: list[NodeSpec] = []
    if state.size_spec is not None:
        size_lines = [
            f"size_by: {state.size_spec.display_name}",
            f"size ~ {state.size_scale}({state.size_spec.display_name}), min-max, clamped",
            *state.size_aggregation_lines,
            *state.size_notes,
        ]
        if state.size_domain is not None:
            low, high = state.size_domain
            size_lines.append(
                f"min {_format_size_domain_value(low)} .. max {_format_size_domain_value(high)}"
            )
        rows.append(NodeSpec(lines=size_lines, shape="box", style="filled,rounded"))
    if state.stack_spec is not None:
        stack_lines = [
            f"stack_by: {state.stack_spec.display_name}",
            "same rank = same annotation value",
            *state.stack_notes,
        ]
        rows.append(NodeSpec(lines=stack_lines, shape="box", style="filled,rounded"))
    return rows


def _format_size_domain_value(value: float) -> str:
    """Format a size-domain endpoint for the legend."""

    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


#: Notice emitted when an active channel forces the dot engine where AUTO
#: would have chosen the rank backend by cost (memo 2.1). Forcing dot on a
#: >threshold-cost graph is the case the rank backend exists for -- expect
#: layout time, bounded by the render timeout.
ENCODING_FORCES_DOT_NOTICE = (
    "An active encoding channel ({channel}) requires the Graphviz dot layout, "
    "overriding the automatic rank-layout choice for this graph "
    "(estimated cost={cost} > threshold={threshold}). Expect longer layout "
    "time, bounded by the render timeout."
)


def attach_encoding_state(
    request: Any,
    theme: Any,
    *,
    channel_specs: tuple[
        EncodingChannelSpec | None,
        EncodingChannelSpec | None,
        EncodingChannelSpec | None,
    ],
    size_scale: str = "sqrt",
) -> Any:
    """Return ``request`` with a fresh per-draw :class:`EncodingState` attached."""

    from dataclasses import replace

    color_spec, size_spec, stack_spec = channel_specs
    return replace(
        request,
        encoding=EncodingState(
            spec=color_spec,
            dark_theme=theme.name == "dark",
            size_spec=size_spec,
            size_scale=size_scale,
            stack_spec=stack_spec,
            # The skin's 3-anchor ramp (N4). The low anchor is off-ground by
            # construction: an encoded-lowest node is never invisible (the
            # measured white-on-white absence defect).
            ramp_anchors=tuple(theme.ramp) if getattr(theme, "ramp", None) else None,
        ),
    )


def resolve_encoding_engine(
    requested_engine: str,
    resolved_engine: str,
    layout_cost: int,
    channels: tuple[str, ...] = ("color_by",),
) -> str:
    """Apply the engine-resolution fence for an ACTIVE channel (memo 2.1).

    v1 encoding channels are dot-layout-only. EXPLICIT ``layout="rank"``
    refuses typed HERE (the earliest point the conflict is decidable); AUTO
    forces dot -- exactly mirroring the ``show_containers`` precedent -- with
    a notice when the force overrides what AUTO would have chosen by cost.
    """

    channel_names = ", ".join(channels) if channels else "color_by"
    if requested_engine == "rank":
        raise _encoding_error(
            f"encoding channels ({channel_names}) require the Graphviz dot "
            "layout; explicit layout='rank' cannot render them",
            code="encoding_requires_dot_layout",
            remedy=f"pass layout='dot' or layout='auto', or drop {channel_names}",
            argument="layout",
        )
    if resolved_engine == "rank":
        import warnings

        from ..utils.display import user_stacklevel
        from ._rank_layout_internal import layout as _rank_layout

        warnings.warn(
            ENCODING_FORCES_DOT_NOTICE.format(
                channel=channel_names,
                cost=layout_cost,
                threshold=_rank_layout.RANK_LAYOUT_COST_THRESHOLD,
            ),
            stacklevel=user_stacklevel(),
        )
    return "dot"


def channel_wrapped_node_spec_fn(
    encoding: EncodingState,
    node: Any,
    node_spec_fn: Any,
) -> Any:
    """Wrap ``node_spec_fn`` with the channels' per-node spec transforms.

    PHASE B application site: the precomputed fill and size minimums apply
    in the C3 slot (after the node-style preset, before the user callback,
    which still sees and may override them). Keyed on the rendered node
    itself: unrolled per-pass Op nodes encode their OWN pass value even
    though the user callback receives the aggregate Layer.
    """

    channel_fill = encoding.fillcolor_for(node)
    channel_size = encoding.size_for(node)
    if channel_fill is None and channel_size is None:
        return node_spec_fn

    def channel_then_user(layer_log: Any, spec: Any) -> Any:
        """Apply the channel fill/size, then let the user callback override them."""

        if channel_fill is not None:
            spec = spec.replace(fillcolor=channel_fill)
        if channel_size is not None:
            # Minimums only: fixedsize=false means the box can only GROW
            # from the label's natural size (a label can never be truncated
            # by an encoding; fonts never scale).
            spec = spec.replace(width=channel_size[0], height=channel_size[1], fixedsize="false")
        if node_spec_fn is None:
            return spec
        result = node_spec_fn(layer_log, spec)
        return spec if result is None else result

    return channel_then_user


def raise_encoding_dagua_refusal(channels: tuple[str, ...] = ("color_by",)) -> None:
    """Refuse an active channel on the dagua renderer (never a silent drop).

    An active channel silently dropped by the alternate label path would be
    a dishonest no-op; channels are Graphviz-dot-only in v1.
    """

    channel_names = ", ".join(channels) if channels else "color_by"
    raise _encoding_error(
        f"encoding channels ({channel_names}) are not supported by the dagua renderer",
        code="encoding_requires_dot_layout",
        remedy=f"use the graphviz renderer, or drop {channel_names}",
        argument="vis_renderer",
    )
