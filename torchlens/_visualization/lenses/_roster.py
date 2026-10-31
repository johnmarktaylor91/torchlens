"""The lens roster v1: nine registry rows, tiers, subjects, compositions.

The inclusion rule (themes memo section 3, the standing arbiter): a preset
earns a registry ROW iff it has (a) its own HEADLINE evidence requirement
-- a capture-time field whose absence makes the picture answer a different
question -- AND (b) its own mandatory disclosure text a composition would
not inherit. Applied uniformly this rule produced exactly nine rows; a
separate vision lens and a combined perf lens both fail it and survive as
tested COMPOSITIONS.

Tiers: CORE-6 (release blocks on each) = overview, blueprint, debug, speed,
dims, transformer. EXTENDED-3 (ship iff their strata pass; a failure drops
that row alone to wave 2 with the mandatory named promotion path in the
memo, never a silent disappearance) = memory, sequence, compute.

``overview`` and ``blueprint`` were registered by the C05 substrate
(``theme_registry``); this module registers the remaining seven and the
roster-level tables. Import is idempotent per process (the registry refuses
double registration; this module registers once at first import).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..theme_registry import LensPreset, register_lens

__all__ = [
    "COMPOSITIONS",
    "CORE_LENSES",
    "EXTENDED_LENSES",
    "LENS_SUBJECTS",
    "PERF_FAMILIES",
    "ROSTER",
]

#: CORE-6: a CORE row failing its full stratum at freeze blocks the release.
CORE_LENSES = ("overview", "blueprint", "debug", "speed", "dims", "transformer")

#: EXTENDED-3: ship iff their strata pass; individual drop to wave 2 with
#: the named promotion path (memo section 10), never silent.
EXTENDED_LENSES = ("memory", "sequence", "compute")

#: Perf rows -> their N16 source family token. The registry row deliberately
#: carries NO color_by member: resolution binds the per-view member and the
#: rank transform at resolve time (view-aware, memo section 2 item 7).
PERF_FAMILIES: Mapping[str, str] = MappingProxyType(
    {"speed": "time", "memory": "bytes", "compute": "flops"}
)

#: Subject requirements (refusal taxonomy, memo section 2 item 5):
#: transformer and sequence refuse on factual absence of SUBJECT, with the
#: teaching escape to theme='overview'.
LENS_SUBJECTS: Mapping[str, str] = MappingProxyType(
    {"transformer": "attention_structure", "sequence": "multi_pass"}
)

#: Named compositions (gallery recipes with full evaluation strata and
#: pre-authorized promotion; memo section 3).
COMPOSITIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "runtime_storage": MappingProxyType({"lens": "speed", "size_by": "bytes", "scale": "sqrt"}),
        "vision": MappingProxyType({"lens": "dims", "direction": "leftright", "color_by": "flops"}),
        "debug_edge_shapes": MappingProxyType(
            # The edge-shape channel spelling belongs to TRI-VIZMECH; the
            # recipe records the intent and the gallery spec pins the
            # stratum. Until that channel lands the recipe is dims-adjacent
            # label redundancy only.
            {"lens": "debug", "show_redundant_args": True}
        ),
    }
)


DEBUG_LENS = register_lens(
    LensPreset(
        name="debug",
        question="what is broken?",
        members={
            # view omitted: the caller's choice flows through. dtype/device
            # label rows ride the preset spec slot (closed node_label_fields
            # vocabulary has no dtype/device tokens yet).
            "collapse": "none",
            "show_buffer_layers": "always",
            "show_redundant_args": True,
            "show_legend": True,
            "show_orphans": True,
        },
        secondary_members=("show_orphans",),
        disclosure=(
            "six-state nonfinite status channel (SECONDARY; two-mode degrade)",
            "hue left FREE for the session's own color_by",
        ),
    )
)

SPEED_LENS = register_lens(
    LensPreset(
        name="speed",
        question="where does time go?",
        members={
            "node_mode": "profiling",
            "show_legend": True,
        },
        headline_evidence="time",
        disclosure=(
            "rank mapping: ordinal, not ratio",
            "instrumented capture: ~14x a plain forward; never quote as production latency",
            "aggregation line mandatory on rolled views",
        ),
    )
)

MEMORY_LENS = register_lens(
    LensPreset(
        name="memory",
        question="what is big in bytes?",
        members={
            "node_mode": "profiling",
            "show_legend": True,
            "show_saved_for_backward": True,
        },
        headline_evidence="bytes",
        secondary_members=("show_saved_for_backward",),
        disclosure=("output-storage bytes, not live/allocator/peak memory",),
    )
)

COMPUTE_LENS = register_lens(
    LensPreset(
        name="compute",
        question="where are the FLOPs?",
        members={
            "node_mode": "profiling",
            "show_legend": True,
        },
        headline_evidence="flops",
        disclosure=("FLOP convention: 2 FLOPs per multiply-accumulate (as measured)",),
    )
)

DIMS_LENS = register_lens(
    LensPreset(
        name="dims",
        question="what shape is the data?",
        members={
            # collapse="none" in v1 (memo dissent 1, the 2-1 cell): the size
            # channel has no honest N/A rendering on collapsed boxes yet;
            # budget resolution becomes available to dims the release an
            # honest N/A-size cue exists (the named wave-2 promotion path).
            "collapse": "none",
            "size_by": "dims",
            "scale": "sqrt",
            "show_legend": True,
        },
        headline_evidence="dims",
        disclosure=(
            "AREA encodes total non-batch element count; it does NOT encode "
            "individual axes -- a 1x512x7x7 and a 1x64x28x28 tensor draw the "
            "same size",
            "encoded area clamped to 4.0x the default node area",
        ),
    )
)

SEQUENCE_LENS = register_lens(
    LensPreset(
        name="sequence",
        question="how do the passes unfold?",
        members={
            # Passes ARE the subject: the view pin is the lens's meaning.
            "vis_mode": "unrolled",
            "collapse": "none",
            "color_by": "pass_index",
            "stack_by": "auto",
            "show_containers": "labels",
            "show_legend": True,
        },
        secondary_members=("stack_by",),
        disclosure=(
            "same rank = the licensed same execution window, not parallelism",
            "pass_index encodes linearly (the one field uniform by construction)",
        ),
    )
)

TRANSFORMER_LENS = register_lens(
    LensPreset(
        name="transformer",
        question="how is attention wired?",
        members={
            "collapse": "auto",
            "fold_repeats": True,
            "show_containers": "labels",
            "direction": "leftright",
            "show_legend": True,
        },
        disclosure=(
            "attention role rows attach where the facts are (per-layer, never "
            "trace classification)",
            "no display filter: masks/positions/casts can be causal to the "
            "failure under inspection",
        ),
    )
)


#: The full roster in tier order (data view for tests and the gallery).
ROSTER: tuple[str, ...] = CORE_LENSES + EXTENDED_LENSES
