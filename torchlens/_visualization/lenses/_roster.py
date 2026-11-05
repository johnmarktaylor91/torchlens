"""The lens roster: registry rows, tiers and subjects.

The public door is ``Trace.draw(lens=...)`` (``_visualization/_lens_door.py``),
whose allowed values are spelled in ``torchlens._literals.DrawLensLiteral``.

The inclusion rule (themes memo section 3): a preset earns a registry ROW
iff it has (a) its own HEADLINE evidence requirement -- a capture-time field
whose absence makes the picture answer a different question -- AND (b) its
own disclosure text. Combinations (a lens plus extra draw keywords, such as
``lens="debug", show_edge_shapes=True``) are spelled with keywords, not rows.

Tiers: CORE-6 = overview, blueprint, debug, speed, dims, transformer.
EXTENDED-3 = memory, sequence, compute. The tenth row, ``interventions``,
registers in ``surgery_visuals``.

``overview`` and ``blueprint`` were registered by the C05 substrate
(``theme_registry``); this module registers the remaining seven and the
roster-level tables. Import is idempotent per process (the registry refuses
double registration; this module registers once at first import).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ..theme_registry import LensPreset, register_lens

__all__ = [
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
        disclosure=("ops whose output holds NaN or Inf are marked",),
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
            "colour shows rank order, not proportion",
            "times come from an instrumented run: compare ops with each other, "
            "not with production latency",
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
        disclosure=(),
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
        disclosure=(),
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
            "box area shows the element count per batch item, not the shape: "
            "512x7x7 and 64x28x28 draw the same size (capped at 4x a plain box)",
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
            "color_by": "pass_number",
            "stack_by": "auto",
            "show_containers": "labels",
            "show_legend": True,
        },
        secondary_members=("stack_by",),
        disclosure=(
            "colour shows the pass number; side-by-side boxes ran in the same pass, "
            "not in parallel",
        ),
    )
)

TRANSFORMER_LENS = register_lens(
    LensPreset(
        name="transformer",
        question="how is attention wired?",
        members={
            # The resolver focuses the draw on the first repeated block that
            # holds attention (module=), so its wiring is drawn in full
            # instead of hidden in a folded module box.
            "collapse": "none",
            "fold_repeats": True,
            "show_legend": True,
        },
        disclosure=("attention rows come from module names and op types",),
    )
)


#: The full roster in tier order (data view for tests and the gallery).
ROSTER: tuple[str, ...] = CORE_LENSES + EXTENDED_LENSES
