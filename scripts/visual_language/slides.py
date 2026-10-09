"""The slide table of the TorchLens visual language deck, as data.

Every slide names its fixture models (by name, see ``models.FIXTURES``), the exact draw
call, the key entries with a selector over the slide's own DOT, and the inventory rows it
covers (``coverage.ROWS``). Nothing here imports torch, so the coverage check can read the
table cheaply. Captions are templates: ``{name}`` fields are filled from TorchLens module
constants by :func:`constants`, so a changed constant changes the caption.

Render conventions (applied by the renderer unless a panel sets ``conventions=False`` or
overrides a key): SVG output, save only, left to right, ``font_size=16``, no legend, the
graph caption hidden, ``collapse="none"``, and two label rows (label and shape) so a picture
fits a deck card at a legible size; slides that teach rows restore the defaults (``FULL``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any

CONVENTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "vis_fileformat": "svg",
        "vis_save_only": True,
        "direction": "leftright",
        "font_size": 16,
        "show_legend": False,
        "vis_graph_overrides": {"label": ""},
        "collapse": "none",
        "node_label_fields": ["label", "shape"],
    }
)

#: Selector over the panel DOT. ``node(...)``, ``edge(...)`` or ``cluster(...)`` with
#: comma-separated conditions: ``attr=value`` (exact, case-insensitive), ``attr~text``
#: (substring of the tag-stripped value; ``text`` searches every label attribute) and
#: ``attr^token`` (token of a comma-separated list such as ``style``).
Selector = str


@dataclass(frozen=True)
class Key:
    """One numbered key entry: words beside the picture and the mark they point at."""

    text: str
    select: Selector | None = None
    panel: str = "a"


@dataclass(frozen=True)
class Panel:
    """One TorchLens render on a slide."""

    name: str
    fixture: str
    call: str = "draw"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    capture: Mapping[str, Any] = field(default_factory=dict)
    intervene: str | None = None
    prep: str | None = None
    label: str = ""
    conventions: bool = True
    #: Selectors whose marks bound the part of the render shown (a detail crop, for
    #: marks whose fixed-size text is only legible enlarged); empty shows it all.
    crop: tuple[Selector, ...] = ()


@dataclass(frozen=True)
class Cell:
    """One crop of a panel in a grid slide: the region around the selected mark."""

    panel: str
    select: Selector | None
    label: str


@dataclass(frozen=True)
class Slide:
    """One slide: a title, a one-line rule, the picture's panels and the key."""

    id: str
    title: str
    rule: str
    panels: tuple[Panel, ...] = ()
    keys: tuple[Key, ...] = ()
    rows: tuple[str, ...] = ()
    layout: str = "auto"
    text: str = ""
    footnote: str = ""
    cells: tuple[Cell, ...] = ()
    caption_kept: bool = False


@cache
def constants() -> dict[str, Any]:
    """Caption constants read from the TorchLens modules that own them."""

    import inspect

    from torchlens._vocab import node_spec
    from torchlens.data_classes._trace_viz import TraceVisualizationMixin
    from torchlens.visualization import (
        _edge_multiplicity,
        _encoding,
        _render_common,
        _render_utils,
        code_panel,
        surgery_visuals,
    )
    from torchlens.visualization._rank_layout_internal import layout
    from torchlens.visualization._typography import DEFAULT_TYPOGRAPHY

    draw_defaults = {
        name: param.default
        for name, param in inspect.signature(TraceVisualizationMixin.draw).parameters.items()
    }
    typo = DEFAULT_TYPOGRAPHY
    return {
        "fan_in": _edge_multiplicity._ARG_LABEL_MIDPOINT_FANIN,
        "commute": ", ".join(_render_common.COMMUTE_FUNCS),
        "noise_buffers": ", ".join(sorted(_render_common._NOISE_BUFFER_NAMES)),
        "container_max_inline": draw_defaults["container_max_inline"],
        "rank_cost": f"{layout.RANK_LAYOUT_COST_THRESHOLD:,}",
        "max_code_lines": code_panel.MAX_CODE_PANEL_LINES,
        "size_max_area": f"{_encoding.SIZE_BY_MAX_AREA_MULT:g}",
        "pen_max": _render_utils.MAX_MODULE_PENWIDTH,
        "pen_min": _render_utils.MIN_MODULE_PENWIDTH,
        "site_color": node_spec.INTERVENTION_SITE_COLOR,
        "cone_color": node_spec.INTERVENTION_CONE_COLOR,
        "base_pt": f"{typo.base_size:g}",
        "annotation_pt": f"{typo.annotation_size:g}",
        "secondary_pt": f"{typo.secondary_size:g}",
        "emphasis_pt": f"{typo.emphasis_size:g}",
        "font": typo.family,
        "surgery_cap": surgery_visuals._MAX_MARK_ROWS_PER_NODE,
        "trainable": _render_common.TRAINABLE_PARAMS_BG_COLOR,
        "frozen": _render_common.FROZEN_PARAMS_BG_COLOR,
        "generic_params": _render_common.PARAMS_NODE_BG_COLOR,
        "bool_color": _render_common.BOOL_NODE_COLOR,
        "input_color": _render_common.INPUT_COLOR,
        "output_color": _render_common.OUTPUT_COLOR,
        "grad_color": _render_common.GRADIENT_ARROW_COLOR,
    }


def fill(template: str) -> str:
    """Fill a caption template from :func:`constants`."""

    return template.format(**constants()) if "{" in template else template


#: Draw arguments that restore TorchLens's own default label rows (title, shape and
#: memory, arguments, parameters, module path) on slides that teach them.
FULL = {"node_label_fields": None}
#: Label plus module path, for slides that point at the call a node belongs to.
CALLS = {"node_label_fields": ["label", "module"]}
CAP = {"vis_graph_overrides": {}}
QUIET = {"show_legend": False}
LEGEND = "node(text~TorchLens)"

SLIDES: tuple[Slide, ...] = (
    Slide(
        id="alphabet-nodes",
        title="At a glance: nodes",
        rule="Shape names the kind; fill, border and rows add facts. Each mark is copied from "
        "the slide that teaches it.",
        layout="sheet",
        rows=(
            "VN01",
            "VN02",
            "VN03",
            "VN04",
            "VN05",
            "VN06",
            "VN07",
            "VN08",
            "VN09",
            "VN10",
            "VN11",
            "VN12",
            "VN13",
            "VN14",
            "VN15",
            "VN16",
            "VN17",
            "VN18",
            "VN26",
        ),
    ),
    Slide(
        id="alphabet-lines",
        title="At a glance: lines and boxes",
        rule="Line style and arrow words describe the data flow; boxes group the calls of one "
        "module.",
        layout="sheet",
        rows=(
            "VE01",
            "VE02",
            "VE03",
            "VE04",
            "VE06",
            "VE07",
            "VE09",
            "VE10",
            "VR01",
            "VR02",
            "VR03",
            "VR11",
            "VR13",
        ),
    ),
    Slide(
        id="start-here",
        title="Start here: how a graph reads",
        rule="Arrows run from the op that made a tensor to the op that used it, so the default "
        "graph reads bottom to top.",
        panels=(Panel("a", "Tiny", kwargs={"direction": "bottomup", **FULL}),),
        keys=(
            Key("Green: the input, named @input.<argument>", "node(fillcolor={input_color})"),
            Key("Arrows point from producer to consumer", "edge(style=solid)"),
            Key("Red: the output, at the top", "node(fillcolor={output_color})"),
        ),
        footnote="A caption under the graph names the model and counts tensors, memory and "
        "parameters; bold warning lines join it when a capture was poisoned or unverified.",
        rows=("VN03", "VE01", "VC03", "VY01"),
    ),
    Slide(
        id="ovals-boxes",
        title="Ovals are calls, boxes are modules",
        rule="A leaf module that ran one op is drawn as that op in a box; the same matmul called "
        "as a function is an oval.",
        panels=(Panel("a", "TwoWays"),),
        keys=(
            Key("Box: nn.Linear called as a module", "node(shape=box)"),
            Key("Oval: F.linear called as a function", "node(shape=oval, text~linear)"),
            Key("Grey: the op uses parameters", "node(fillcolor={trainable})"),
            Key("White: no parameters", "node(fillcolor=white, shape=oval)"),
        ),
        rows=("VN01", "VN02"),
    ),
    Slide(
        id="node-rows",
        title="What a node says, row by row",
        rule="Bold title, shape and memory, the arguments the shapes do not prove, parameters, "
        "then the module path.",
        panels=(
            Panel("a", "Anatomy", kwargs=FULL, label="draw()", crop=("node(text~conv2d)",)),
            Panel(
                "b",
                "Anatomy",
                kwargs={"show_redundant_args": True, **FULL},
                label="show_redundant_args=True",
                crop=("node(text~conv2d)",),
            ),
        ),
        keys=(
            Key("Title: type, count of that type, execution index", "node(text~conv2d_1_1)"),
            Key("Output shape and memory", 'node(text~"(1, 2, 3, 3)")'),
            Key("stride and padding kept; in_channels dropped", "node(text~stride)"),
            Key("show_redundant_args=True restores it", "node(text~in_channels)", panel="b"),
        ),
        footnote="Rows are {font} {base_pt} pt by default (16 here); the first row is bold.",
        rows=("VL01", "VL02", "VL05", "VL06", "VL07", "VT03"),
    ),
    Slide(
        id="grey-params",
        title="Grey means parameters",
        rule="Light grey trains, dark grey is frozen, two-tone is both. Fill precedence: input, "
        "output, boolean, then parameters.",
        panels=(Panel("a", "Params", kwargs={**CAP, **FULL}),),
        keys=(
            Key("Light grey {trainable}: all trainable", "node(fillcolor={trainable})"),
            Key("Dark grey {frozen}: frozen, shapes in brackets", "node(fillcolor={frozen})"),
            Key("Two-tone: some of each", "node(fillcolor~:)"),
            Key("Caption counts trainable parameters", "graph(label~trainable)"),
        ),
        footnote="A pale {generic_params} fill marks an op that used parameters with no "
        "Param record; no small model produces it.",
        caption_kept=True,
        rows=("VN05", "VL06", "VC01"),
    ),
    Slide(
        id="dashed",
        title="Dashed: not from the input",
        rule="Anything computed only from parameters, buffers, constants or random draws is "
        "dashed until it meets the input.",
        panels=(Panel("a", "Gate"),),
        keys=(
            Key("Dashed sigmoid: made from a parameter only", "node(style^dashed, text~sigmoid)"),
            Key("Its arrows are dashed too", "edge(style=dashed)"),
            Key("The module box around it is dashed", "cluster(style^dashed)"),
            Key("The input arrives: solid again", "node(text~mul_2)"),
        ),
        rows=("VN06", "VE01", "VR01"),
    ),
    Slide(
        id="module-boxes",
        title="Module boxes: path, class, depth",
        rule="Each call is a box titled @path over (Class); the border thins with depth, "
        "{pen_max} pt outside to {pen_min} pt at the deepest.",
        panels=(Panel("a", "Nested", kwargs=CALLS),),
        keys=(
            Key("@block:1 over (Sequential): the first call", "cluster(label~@block:1)"),
            Key("Thick outer border, thinner inside", "cluster(penwidth={pen_max})"),
            Key("Second call of the same module: @block:2", "cluster(label~@block:2)"),
            Key("Rows name the call: @block.0.fc:2", "node(text~@block.0.fc:2)"),
        ),
        footnote="Empty modules are never drawn.",
        rows=("VR01", "VR02", "VR03", "VR04", "VL07"),
    ),
    Slide(
        id="arrow-words",
        title="Words on arrows: argument slots",
        rule="When argument order matters, each arrow names its slot.",
        panels=(
            Panel(
                "a",
                "SubXY",
                label="torch.sub(x, y)",
                crop=("node(text~input_1)", "node(text~input_2)", "node(text~sub)"),
            ),
        ),
        keys=(
            Key("arg 0: the first argument", 'edge(text~"arg 0")'),
            Key("arg 1: the second", 'edge(text~"arg 1")'),
            Key("Arrows into {commute} carry no slot", None),
        ),
        rows=("VE02",),
    ),
    Slide(
        id="fan-in",
        title="Many inputs: labels mid-arrow",
        rule="From {fan_in} inputs up, slot labels sit mid-arrow.",
        panels=(
            Panel(
                "a",
                "FanIn",
                kwargs={"direction": "topdown"},
                label="torch.stack of four",
                crop=(
                    "node(text~stack)",
                    'words:edge(text~"arg (0, 0)")',
                    'words:edge(text~"arg (0, 3)")',
                ),
            ),
        ),
        keys=(
            Key("arg (0, k): position k in the list", "edge(text~arg)"),
            Key("cat([a, a]) draws two arrows from one op", None),
        ),
        rows=("VE02",),
    ),
    Slide(
        id="arrows-as-one",
        title="Arrows drawn as one: xN",
        rule="xN on an arrow means N edges between the same two nodes, drawn as one.",
        panels=(
            Panel(
                "a",
                "NestedRes",
                kwargs={"depth": 1},
                label="depth=1",
                crop=("node(text~input_1)", "node(shape=box3d)"),
            ),
        ),
        keys=(
            Key("x2: the input feeds two ops inside the box", "edge(text~x2)"),
            Key("The box hides which ops they were", "node(shape=box3d)"),
        ),
        rows=("VE03",),
    ),
    Slide(
        id="loop-unrolled",
        title="Loops, unrolled: a node per pass",
        rule=':2 is the pass number, and the exact key trace["linear_1_1:2"] accepts.',
        panels=(Panel("a", "Loop2", kwargs={"view": "unrolled", **FULL}),),
        keys=(
            Key("linear_1_1:2: second pass of the same layer", "node(text~linear_1_1:2)"),
            Key("One call per pass: @cell:1, then @cell:2", "node(text~@cell:2)"),
            Key("Passes run left to right", "edge(head~relu_1_2pass2)"),
        ),
        rows=("VL01", "VR04", "VV01"),
    ),
    Slide(
        id="loop-rolled",
        title="Loops, rolled: pass counts",
        rule="(x3) means one node ran three times with the same weights; In and Out say which "
        "passes used each edge.",
        panels=(
            Panel(
                "a",
                "Loop",
                kwargs={"view": "rolled", **FULL},
                crop=("node(text~linear)", "node(text~relu)", "edge(text~In)"),
            ),
        ),
        keys=(
            Key("(x3): ran three times", "node(text~x3)"),
            Key("In 2-3 on the back edge", "edge(text~In)"),
            Key("A self-loop appears only when the loop carries state", None),
        ),
        rows=("VL01", "VE04", "VE05", "VR04"),
    ),
    Slide(
        id="reuse-shapes",
        title="Reuse versus changing shapes",
        rule="Separate call groups print as :1-2,3-4; a shape that changes across passes prints "
        "with an arrow.",
        panels=(
            Panel(
                "a",
                "LoopGroups",
                kwargs={"view": "rolled", **CALLS},
                label="two call groups",
                crop=("node(text~linear)",),
            ),
            Panel(
                "b",
                "LoopShapes",
                kwargs={"view": "rolled", "depth": 1, **FULL},
                label="two shapes, depth=1",
                crop=("node(shape=box3d)",),
            ),
        ),
        keys=(
            Key("Call groups 1-2 and 3-4", 'node(text~"1-2,3-4")'),
            Key("The shape row changes across calls", "node(text~shapes)", panel="b"),
            Key("Compare (xN): one weight-tied loop", None),
        ),
        rows=("VL01", "VL03"),
    ),
    Slide(
        id="branches",
        title="Branches: which arm ran",
        rule="Yellow TRUE is the comparison that chose the arm; it has no outgoing arrow because "
        "it steered Python, not tensors.",
        panels=(Panel("a", "Branch"),),
        keys=(
            Key("Yellow boolean, a dead end", "node(fillcolor={bool_color})"),
            Key("IF where the condition's computation starts", "edge(text~IF)"),
            Key("THEN into the first op of the arm that ran", "edge(text~THEN)"),
            Key("The arm that did not run is absent", None),
        ),
        footnote="Rolled arms print THEN(1,3); several conditionals print THEN@L12.",
        rows=("VN04", "VE06"),
    ),
    Slide(
        id="buffers",
        title="Buffers: cylinders",
        rule='"meaningful" hides {noise_buffers}; the op that reads them gets a double outline.',
        panels=(Panel("a", "Stateful", kwargs=FULL, label='show_buffer_layers="meaningful"'),),
        keys=(
            Key("@scale: a buffer drawn as a cylinder", "node(shape=cylinder, text~@scale)"),
            Key("Double outline: hidden buffers feed this op", "node(peripheries=2)"),
            Key('"always" draws the hidden statistics too; "never" hides every buffer', None),
        ),
        rows=("VN07", "VN08"),
    ),
    Slide(
        id="mutated-param",
        title="A parameter changed in place",
        rule="A dashed cylinder stands for the Parameter itself when an in-place op changed it "
        "during forward.",
        panels=(
            Panel("a", "Clamp", kwargs=FULL),
            Panel(
                "b",
                "Clamp",
                kwargs={"show_legend": True, **FULL},
                label="show_legend=True",
                crop=(LEGEND,),
            ),
        ),
        keys=(
            Key("parameter temp (1,)", "node(shape=cylinder, text~parameter)"),
            Key("Dashed black edges to its earlier readers", "edge(style=dashed)"),
            Key("The legend gains this row only now", 'node(text~"mutated parameter")', "b"),
        ),
        footnote="A frozen Parameter takes the dark grey fill.",
        rows=("VN09", "VE11", "VI01"),
    ),
    Slide(
        id="orphans",
        title="Computed but unused: orphans",
        rule="Grey dashed boxes with no edges, unreachable from inputs and outputs; hidden "
        "unless you ask.",
        panels=(
            Panel(
                "a",
                "Dead",
                capture={"keep_orphans": True},
                kwargs={"show_orphans": True},
                crop=("cluster(label~orphans)", "node(text~mul)"),
            ),
        ),
        keys=(
            Key("The orphans group", "cluster(label~orphans)"),
            Key("sin ran on a constant, and nothing used it", "node(text~sin)"),
            Key("Without keep_orphans at capture, the draw warns", None),
        ),
        rows=("VN14", "VR13"),
    ),
    Slide(
        id="containers",
        title="Dicts and tuples",
        rule='show_containers="nodes" adds a record box for a returned container; "labels" '
        "names its keys on the arrows.",
        panels=(
            Panel(
                "a",
                "DictOut",
                capture={"capture_container_structure": True},
                kwargs={"show_containers": "nodes", "direction": "topdown"},
                label='show_containers="nodes"',
                crop=("node(text~dict)", "node(text~output_1)", "node(text~output_2)"),
            ),
        ),
        keys=(
            Key("Dashed record box for the returned dict", "node(text~dict)"),
            Key("Arrow words name the keys", "edge(text~logits)"),
            Key("Arrowless ties mark membership", "edge(arrowhead=none)"),
        ),
        footnote='"cluster" draws a dotted group instead.',
        rows=("VN25", "VE12", "VR09"),
    ),
    Slide(
        id="container-lists",
        title="Long lists fold",
        rule='"collapsed" folds more than container_max_inline ({container_max_inline} by '
        "default) same-shape leaves into one node.",
        panels=(
            Panel(
                "a",
                "ListOut",
                capture={"capture_container_structure": True},
                kwargs={
                    "show_containers": "collapsed",
                    "container_max_inline": 2,
                    "direction": "topdown",
                },
                label='"collapsed", container_max_inline=2',
                crop=("node(text~mul_1_1)", "node(text~mul_3_3)", "node(text~x3)"),
            ),
        ),
        keys=(Key("Three same-shape leaves folded into one", "node(text~x3)"),),
        footnote='"auto" behaves as "collapsed" today.',
        rows=("VN24", "VR09"),
    ),
    Slide(
        id="depth",
        title="Whole modules as one node",
        rule="The 3D box names the module and counts the ops, buffers and parameters inside.",
        panels=(Panel("a", "Nested", kwargs={"depth": 1}),),
        keys=(
            Key("@block:1 as a 3D box", "node(shape=box3d, text~@block:1)"),
            Key("Class, output shape and memory", "node(shape=box3d, text~Sequential)"),
            Key("Ops and parameters inside", "node(shape=box3d, text~params)"),
        ),
        footnote="collapse_fn(module) chooses the modules to fold instead of a depth.",
        rows=("VN10", "VR05"),
    ),
    Slide(
        id="collapse-max",
        title='collapse="max": runs as chips',
        rule="A rounded chip stands for a run of ops and names its address range; a float in "
        "[0, 1] walks a monotone ladder.",
        panels=(
            Panel(
                "a",
                "Stack",
                kwargs={"collapse": "max", "direction": "topdown"},
                label='collapse="max"',
            ),
        ),
        keys=(
            Key("Chip: a run of ops and the modules it spans", "node(style^rounded)"),
            Key("collapse=0.5 picks a point between none and max", None),
        ),
        footnote="A floor fallback prints bold in the caption; an orange dashed edge is a "
        "projection artifact, not a cycle.",
        rows=("VN11", "VR06", "VR07", "VE13"),
    ),
    Slide(
        id="collapse-auto",
        title='collapse="auto" on a big graph',
        rule='"auto" folds just enough to reach the readable band; on a small graph it changes '
        "nothing.",
        panels=(
            Panel(
                "a",
                "BigStack",
                kwargs={"collapse": "auto"},
                label='collapse="auto", part of 45 ops',
                crop=("node(name=blocks.1pass1)", "node(text~linear_3_7)"),
            ),
        ),
        keys=(
            Key("Folded by auto: a whole block", "node(name=blocks.1pass1)"),
            Key("The rest stay expanded", "node(text~linear_3_7)"),
        ),
        rows=("VR06",),
    ),
    Slide(
        id="fold-repeats",
        title="Repeated blocks: one, +N more",
        rule="One representative and an honest count of separate blocks with their own weights; "
        "compare (xN), one block run N times.",
        panels=(Panel("a", "Stack", kwargs={"fold_repeats": True}),),
        keys=(
            Key("The representative, @blocks.0", "node(shape=box3d, text~@blocks.0)"),
            Key("... +3 more Block", "node(shape=plaintext, text~more)"),
            Key("Arrows route through the ellipsis", "edge(head~more)"),
        ),
        rows=("VN13", "VE16", "VR06"),
    ),
    Slide(
        id="fold-patterns",
        title="Named patterns: fold_patterns",
        rule="PATTERN chips replace runs that match a named idiom; patterns refuse collapse other "
        'than "none".',
        panels=(
            Panel(
                "a",
                "ConvBnRelu2",
                kwargs={"fold_patterns": "idiomatic", "direction": "topdown"},
                label='fold_patterns="idiomatic"',
            ),
        ),
        keys=(
            Key("A ConvBnRelu chip: three ops, one name", "node(text~PATTERN)"),
            Key("Two instances, different weights, same name", None),
            Key('With collapse other than "none", the draw refuses', None),
        ),
        rows=("VN12",),
    ),
    Slide(
        id="own-patterns",
        title="Your own patterns",
        rule="A mapping declares a pattern: a name and a path of op or module names joined by >.",
        panels=(
            Panel(
                "a",
                "Flow",
                kwargs={"fold_patterns": {"LinearTanh": "Linear > tanh"}},
                label='fold_patterns={"LinearTanh": "Linear > tanh"}',
            ),
        ),
        keys=(Key("Your pattern's chip", "node(text~LinearTanh)"),),
        rows=("VN12",),
    ),
    Slide(
        id="focus",
        title="Focus on one module: module=",
        rule="Only the module's own ops; green and red ovals stand for the outside. "
        'trace.modules["blocks.0"].draw() is the same.',
        panels=(Panel("a", "Stack", kwargs={"module": "blocks.0"}),),
        keys=(
            Key("Green: where data enters", "node(fillcolor={input_color})"),
            Key("Red: where data leaves", "node(fillcolor={output_color})"),
            Key("Only the module's own ops", "node(text~relu)"),
        ),
        rows=("VN15", "VR08"),
    ),
    Slide(
        id="skip",
        title="Hiding ops: skip_fn, lens filter",
        rule="A dashed bridge labelled via N hidden means reachable through omitted work, not "
        "adjacent.",
        panels=(
            Panel("a", "FlowReshape", kwargs={"skip_fn": "@skip_reshape"}, label="skip_fn"),
            Panel(
                "b",
                "FlowReshape",
                call="lens",
                kwargs={"lens": "overview", "display_filter": "@exclude_reshapes"},
                label="a lens with a display filter",
            ),
        ),
        keys=(
            Key("The reshape is gone: via 1 hidden", "edge(text~hidden)"),
            Key("The lens filter bridges the same way", "edge(text~hidden)", panel="b"),
        ),
        rows=("VE07", "VK10"),
    ),
    Slide(
        id="interventions",
        title="Interventions: site and cone",
        rule="Magenta 3 pt border is the edited site; pink 1.75 pt is its downstream cone; "
        '"as_node" draws the hook as a diamond.',
        panels=(
            Panel(
                "a",
                "Flow",
                capture={"intervention_ready": True},
                intervene="zero_tanh",
                kwargs={"vis_intervention_mode": "node_mark"},
                label='"node_mark"',
            ),
            Panel(
                "b",
                "Flow",
                capture={"intervention_ready": True},
                intervene="zero_tanh",
                kwargs={"vis_intervention_mode": "as_node"},
                label='"as_node"',
            ),
        ),
        keys=(
            Key("Site: magenta border {site_color}", "node(color={site_color})"),
            Key("Cone: pink border downstream", "node(color={cone_color})"),
            Key("The hook as a diamond", "node(shape=diamond)", panel="b"),
            Key("vis_show_cone=False hides the cone", None),
        ),
        rows=("VN16", "VN17", "VN18"),
    ),
    Slide(
        id="surgery",
        title="What an edit did: surgery lens",
        rule="Solid magenta cites a recorded fire; dashed marks a declared target that did not "
        "fire here.",
        panels=(
            Panel(
                "a",
                "Shared",
                call="surgery",
                capture={"intervention_ready": True, "save_arg_values": True},
                prep="fork_edit",
                crop=("node(name=linear_1_1pass2)",),
            ),
        ),
        keys=(
            Key("The edit: zero_ablate, replay engine", "node(text~edited)"),
            Key("The declared target: no fire recorded here", 'node(text~"no fire")'),
        ),
        footnote="A caption census names the replacement lane that ran, and a replay with "
        "direct writes adds Direct writes detected. Up to {surgery_cap} citation rows per node.",
        rows=("VN19", "VC02", "VC05"),
    ),
    Slide(
        id="surgery-diff",
        title="Two captures, honest joins",
        rule="Solid joins use site identity; dashed joins are heuristic; a missing record does "
        "not prove missing execution.",
        panels=(
            Panel(
                "a",
                "Shared",
                call="surgery_diff",
                capture={"intervention_ready": True, "save_arg_values": True},
                prep="fork_edit",
                crop=('node(text~"linear_1_1:1")', 'node(text~"linear_1_1:2")'),
            ),
        ),
        keys=(
            Key("Solid grey join: same site", "edge(style=solid, dir=none)"),
            Key("Dashed join: matched by label or position", None),
        ),
        rows=("VE15",),
    ),
    Slide(
        id="color-by",
        title="Colour a measurement: color_by",
        rule="Fill ramps light to blue by value; the legend appears by itself and states source, "
        "coverage, scale and range.",
        panels=(
            Panel("a", "Sizes", kwargs={"color_by": "flops", **QUIET}, label='color_by="flops"'),
            Panel(
                "b",
                "Sizes",
                kwargs={"color_by": "flops", "show_legend": None},
                label="its legend",
                crop=("node(text~color_by)",),
            ),
        ),
        keys=(
            Key("Most FLOPs: the darkest fill", "node(text~linear_2)"),
            Key("The encoding legend appears by itself", "node(text~color_by)", panel="b"),
            Key("Unencoded nodes stay white and are counted", "node(text~encoded)", panel="b"),
        ),
        rows=("VK01", "VK03", "VK06", "VI02"),
    ),
    Slide(
        id="color-transforms",
        title="Linear, rank and log differ",
        rule="Rank shows order, linear shows the position in the range, log leaves nonpositive "
        "values unencoded.",
        panels=tuple(
            Panel(
                name,
                "Sizes",
                kwargs={"color_by": f"@bytes_{transform}", **QUIET},
                label=transform,
                crop=("node(text~linear_1)", "node(text~linear_2)"),
            )
            for name, transform in zip("abc", ("linear", "rank", "log"), strict=True)
        ),
        footnote="Rank is ordinal, not a ratio. Rolled nodes whose value varies stay unencoded "
        "with the reason.",
        rows=("VK02",),
    ),
    Slide(
        id="unencoded",
        title="Unencoded is not zero",
        rule="A node left white carries no value for the channel; the legend says why, in these "
        "words.",
        layout="table",
        rows=("VK03",),
    ),
    Slide(
        id="size-by",
        title="Size a measurement: size_by",
        rule="Area, not side, follows the non-batch element count, clamped at {size_max_area}x; "
        "labels never shrink.",
        panels=(
            Panel("a", "Sizes", kwargs={"size_by": "dims", "scale": "sqrt", **QUIET}),
            Panel(
                "b",
                "Sizes",
                kwargs={"size_by": "dims", "scale": "sqrt", "show_legend": None},
                label="its legend",
                crop=("node(text~size_by)",),
            ),
        ),
        keys=(
            Key("Bigger tensor, bigger node", "node(text~linear_1)"),
            Key("The legend states the size rule", "node(text~size_by)", panel="b"),
            Key('scale="sqrt" by default; "linear" grows faster', None),
        ),
        rows=("VK04", "VK06"),
    ),
    Slide(
        id="stack-by",
        title="Line passes up: stack_by",
        rule="Nodes sharing a pass index share a rank: the same annotation, not parallel "
        "execution. Rolled views refuse.",
        panels=(
            Panel(
                "a",
                "Loop2",
                kwargs={"view": "unrolled", "stack_by": True, **QUIET},
                label="stack_by=True",
            ),
            Panel(
                "b",
                "Loop2",
                kwargs={"view": "unrolled", "stack_by": True, "show_legend": None},
                label="its legend",
                crop=("node(text~stack_by)",),
            ),
        ),
        keys=(
            Key("Each pass shares a rank", "node(text~linear_1_1:2)"),
            Key("The legend explains the rank rule", "node(text~stack_by)", panel="b"),
            Key("Sibling ordering is off while stacking", None),
        ),
        footnote="The caption adds the line stacked by pass_index.",
        rows=("VK05", "VR14", "VK06"),
    ),
    Slide(
        id="overlays",
        title="Overlays: a row and a border",
        rule="nan: yes adds an orange 3 pt border; n/a means not checkable; a nonzero numeric "
        "overlay thickens the border.",
        panels=(
            Panel("a", "NanMaker", kwargs={"node_overlay": "nan"}, label='node_overlay="nan"'),
            Panel("b", "Tiny", kwargs={"node_overlay": "@score_map"}, label="a mapping"),
        ),
        keys=(
            Key("nan: yes row", 'node(text~"nan: yes")'),
            Key("Orange 3 pt border flags it", "node(color=#D55E00)"),
            Key("A mapping adds a numeric row and a thicker border", "node(penwidth=2)", panel="b"),
        ),
        rows=("VN20", "VL08", "VK07"),
    ),
    Slide(
        id="more-rows",
        title="More rows on demand",
        rule="Profiling adds time, storage and call site; listed fields replace the default rows.",
        panels=(
            Panel(
                "a",
                "Grad",
                kwargs={
                    "node_style": "profiling",
                    "show_saved_for_backward": True,
                    **FULL,
                },
                label='node_style="profiling"',
                crop=("node(text~mul)",),
            ),
            Panel(
                "b",
                "Flow",
                kwargs={"node_label_fields": ["label", "time"]},
                label='node_label_fields=["label", "time"]',
            ),
        ),
        keys=(
            Key("t= time and out= storage", "node(text~t=)"),
            Key("call= and fn= name the source line", "node(text~call=)"),
            Key("show_saved_for_backward: retained tensors", "node(text~saved)"),
            Key("Fields you list replace the defaults", "node(text~tanh)", panel="b"),
        ),
        footnote="Vision and attention rows exist only as experimental node_spec_fn "
        "callbacks in torchlens.experimental.node_styles.",
        rows=("VL04", "VL09", "VL10", "VL11", "VK08"),
    ),
    Slide(
        id="lenses",
        title="Lenses: presets for one question",
        rule="A lens chooses sources, detail and disclosures; anything you pass wins; a lens may "
        "refuse when evidence is missing.",
        layout="grid",
        panels=(
            Panel("a", "Flow", call="lens", kwargs={"lens": "overview"}, label="overview"),
            Panel("b", "Flow", call="lens", kwargs={"lens": "speed"}, label="speed"),
            Panel("c", "Flow", call="lens", kwargs={"lens": "dims"}, label="dims"),
        ),
        cells=(
            Cell("a", "node(text~tanh)", "overview"),
            Cell("b", "node(text~tanh)", "speed"),
            Cell("c", "node(text~tanh)", "dims"),
        ),
        footnote="The same tanh node under three lenses.",
        rows=("VK09",),
    ),
    Slide(
        id="nonfinite",
        title="Not checked is not finite",
        rule="Nonfinite states use stripes, wedges or a border; grey dashed means not checked, "
        "never finite.",
        layout="grid",
        panels=(
            Panel(
                "a",
                "Nonfinite",
                call="lens",
                capture={"module_filter": "@unsaved_abs"},
                kwargs={"lens": "debug", "node_label_fields": ["label"]},
            ),
        ),
        cells=(
            Cell("a", "node(text~add)", "finite"),
            Cell("a", "node(text~log_1)", "nan"),
            Cell("a", "node(text~truediv_1)", "pos_inf"),
            Cell("a", "node(text~truediv_2)", "neg_inf"),
            Cell("a", "node(text~log_2)", "mixed"),
            Cell("a", "node(text~abs)", "not_checked (payload not saved)"),
        ),
        rows=("VN29",),
    ),
    Slide(
        id="backward",
        title="The backward graph",
        rule="Lavender ovals are grad_fns named by their forward op; dotted accum edges feed "
        "leaves.",
        panels=(
            Panel(
                "a",
                "Grad",
                call="draw_backward",
                capture={"save_grads": "all"},
                prep="log_backward",
                kwargs=QUIET,
                label="draw_backward()",
            ),
            Panel(
                "b",
                "HigherOrder",
                call="draw_backward",
                capture={"save_grads": "all"},
                prep="higher_order",
                kwargs={"vis_mode": "unrolled", "show_legend": True},
                label="the backward key, with every style painted",
                crop=('node(text~"backward key")',),
            ),
        ),
        keys=(
            Key("grad_fn node", "node(fillcolor=#F2F3FF)"),
            Key("Dotted accum edge to a leaf", "edge(style=dotted)"),
            Key("The backward key lists only the styles painted", "node(text~grad_fn)", "b"),
        ),
        rows=("VN26", "VE09", "VR11", "VI03", "VC04", "VV02"),
    ),
    Slide(
        id="grad-of-grad",
        title="Gradients of gradients",
        rule="A grad_fn of order two or more is cream; [i] marks a grad_fn with no forward op; "
        "[custom] a custom autograd Function.",
        panels=(
            Panel(
                "a",
                "HigherOrder",
                call="draw_backward",
                capture={"save_grads": "all"},
                prep="higher_order",
                kwargs={"vis_mode": "unrolled", **QUIET},
                crop=("node(text~sum_back_2_11)", "node(text~mul_back_1_7)"),
            ),
        ),
        keys=(
            Key("Order two or more: cream", "node(fillcolor=#FFF4D6)"),
            Key("[i]: no forward op", "node(text~[i])"),
        ),
        footnote="[custom] marks the custom Function's grad_fn elsewhere in this graph.",
        rows=("VN26",),
    ),
    Slide(
        id="combined",
        title="Forward and backward together",
        rule="Dashed ties join each forward op to its grad_fn; purple reverse arrows are "
        "captured gradients.",
        panels=(
            Panel(
                "a",
                "Grad",
                call="draw_combined",
                capture={"save_grads": "all"},
                prep="log_backward",
                label="draw_combined()",
                crop=("node(text~mul_1_1)", "node(text~mul_back)"),
            ),
            Panel(
                "b",
                "Grad",
                capture={"save_grads": "all"},
                prep="log_backward",
                kwargs={"vis_grad_edge_overrides": {"penwidth": "2"}},
                label="draw() with gradients",
            ),
        ),
        keys=(
            Key("Correspondence tie, dashed", "edge(constraint=false, style=dashed)"),
            Key("Gradient arrows on the forward graph", "edge(color={grad_color})", panel="b"),
        ),
        footnote="intervening_cluster places intervening grad_fns upstream (default), "
        "outside, downstream or in their own dashed group.",
        rows=("VE08", "VE10", "VR12", "VC04"),
    ),
    Slide(
        id="skins",
        title="Five skins",
        rule='Meaning stays, ink changes; for_paper=True forces "paper".',
        layout="grid",
        panels=tuple(
            Panel(name, "Tiny", kwargs={"vis_theme": skin}, label=skin)
            for name, skin in zip(
                "abcde", ("torchlens", "paper", "dark", "colorblind", "high_contrast"), strict=True
            )
        ),
        footnote='"colorblind" changes borders, not the palette. Orphans, container nodes, '
        "backward nodes and overlay borders ignore the skin.",
        rows=("VT01", "VT02"),
    ),
    Slide(
        id="direction-layout",
        title="Direction and sibling order",
        rule="Direction rotates the layout, not the computation.",
        panels=(
            Panel("a", "Tiny", kwargs={"direction": "bottomup"}, label="bottomup (default)"),
            Panel("b", "Tiny", kwargs={"direction": "topdown"}, label="topdown"),
            Panel("c", "Tiny", kwargs={"direction": "leftright"}, label="leftright"),
        ),
        footnote="order_siblings=True keeps fan-outs in call order using invisible ordering edges.",
        rows=("VY01", "VR10", "VE14"),
    ),
    Slide(
        id="layout-engine",
        title='Layout engines: layout="rank"',
        rule='layout="dot" is the default; "auto" switches to the rank engine above {rank_cost} '
        "cost units.",
        panels=(
            Panel("a", "Tiny", kwargs={"layout": "rank", **QUIET}, label='layout="rank"'),
            Panel(
                "b",
                "Tiny",
                kwargs={"layout": "rank", "show_legend": True},
                label="the rank engine's legend, pinned left",
                crop=(LEGEND,),
            ),
        ),
        keys=(Key("The rank engine pins its legend at the left", LEGEND, panel="b"),),
        footnote="The rank engine draws no orphans and does not order siblings.",
        rows=("VY02", "VI04"),
    ),
    Slide(
        id="text-size",
        title="Text size and role differ",
        rule="font_size grows node rows; arrow words stay {annotation_pt} pt.",
        panels=(
            Panel(
                "a",
                "SubXY",
                kwargs={"font_size": 12},
                label="font_size=12",
                crop=("node(text~sub)", 'edge(text~"arg 0")', 'edge(text~"arg 1")'),
            ),
            Panel(
                "b",
                "SubXY",
                kwargs={"font_size": 22},
                label="font_size=22",
                crop=("node(text~sub)", 'edge(text~"arg 0")', 'edge(text~"arg 1")'),
            ),
        ),
        keys=(
            Key("Node text grew", "node(text~sub)", panel="b"),
            Key("arg 0 did not", "edge(text~arg)", panel="b"),
        ),
        footnote="Legend text is {secondary_pt} pt and branch labels {emphasis_pt} pt; dpi "
        "changes raster pixels only.",
        rows=("VT03",),
    ),
    Slide(
        id="code-panel",
        title="Source beside the graph",
        rule="code_panel adds the captured source in Courier with an Open source link, up to "
        "{max_code_lines} lines.",
        panels=(
            Panel("a", "Tiny", kwargs={"code_panel": "forward"}, crop=("region:0,0,99999,150",)),
        ),
        keys=(
            Key("The forward source", None),
            Key('"class" and "init+forward" show more', None),
            Key("Missing source is stated in the panel", None),
        ),
        rows=("VO04",),
    ),
    Slide(
        id="builtin-legend",
        title="The legend TorchLens draws",
        rule="show_legend=True adds seven fill roles and nothing else; None shows only an active "
        "channel's section.",
        panels=(
            Panel(
                "a",
                "Stateful",
                kwargs={"show_legend": True},
                label="show_legend=True",
                crop=(LEGEND,),
            ),
        ),
        keys=(
            Key("The built-in legend table", 'node(text~"TorchLens legend")'),
            Key("No shapes, dashes, double borders, box weights or arrow words", None),
            Key("Channel and backward sections join when they apply", None),
            Key("show_legend=False hides even those", None),
        ),
        rows=("VN27", "VI01", "VI02"),
    ),
    Slide(
        id="custom",
        title="Custom styling is custom",
        rule="Callbacks and overrides change appearance; these marks are your choices, not "
        "built-in meaning.",
        panels=(
            Panel(
                "a",
                "Flow",
                kwargs={"node_spec_fn": "@badge", "vis_edge_overrides": {"color": "#0072B2"}},
                label="node_spec_fn, vis_edge_overrides",
            ),
            Panel(
                "b",
                "Nested",
                kwargs={
                    "depth": 1,
                    "collapsed_node_spec_fn": "@boxbadge",
                    "vis_graph_overrides": {"label": "", "bgcolor": "#F7F7F7"},
                    "vis_module_overrides": {"color": "#0072B2"},
                },
                label="collapsed_node_spec_fn, graph and module overrides",
            ),
        ),
        keys=(
            Key("NodeSpec callback restyles a node", "node(fillcolor=#FFE9A8)"),
            Key("Edge override recolours every arrow", "edge(color=#0072B2)"),
            Key("Collapsed-node callback", "node(fillcolor=#FFE9A8)", panel="b"),
            Key("Graph override: background", "graph(bgcolor=#F7F7F7)", panel="b"),
        ),
        rows=("VN28", "VE17", "VO03"),
    ),
    Slide(
        id="export",
        title="Export choices add no facts",
        rule="DPI changes raster pixels only; closed vocabularies refuse before any render.",
        layout="table",
        rows=("VO01", "VO02", "VO05", "VV01"),
    ),
    Slide(
        id="data-previews",
        title="Inputs as pictures",
        rule="A picture replaces the input outline; show_input_transform_summary names the "
        "transform.",
        panels=(
            Panel(
                "a",
                "ImageNet",
                capture={"transform": "@to_tensor"},
                kwargs={"show_input_transform_summary": True},
                crop=("node(text~preprocess)",),
            ),
        ),
        keys=(
            Key("The input images as a montage", "node(shape=none)"),
            Key("preprocess names the transform, UNVERIFIED here", "node(text~preprocess)"),
        ),
        footnote="Batch policy auto shows 4 examples, all 16, or first, first_n:N, shape_only.",
        rows=("VN21", "VN22", "VL12"),
    ),
    Slide(
        id="data-outputs",
        title="Outputs as decoded rows",
        rule="output_transform turns the output into label and score rows; output_style decodes "
        "a model that carries its labels.",
        panels=(
            Panel(
                "a",
                "ImageNet",
                capture={"transform": "@to_tensor", "output_transform": "@decode"},
                crop=("node(text~green)",),
            ),
        ),
        keys=(Key("Decoded rows replace the output's shape", "node(text~green)"),),
        rows=("VN23",),
    ),
    Slide(
        id="other-pictures",
        title="Other TorchLens pictures",
        rule="Bundle graph and diff, fastlog preview, summary, tviz and more have their own keys; "
        "red delta is not an intervention.",
        layout="table",
        rows=("VS01", "VS02", "VS03", "VS04", "VI05", "VY03"),
    ),
    Slide(
        id="cheat-sheet",
        title="Cheat sheet: to see X, pass Y",
        rule="Every draw() parameter once, grouped by what it shows (part 1 of 2).",
        layout="table",
        rows=("VO06",),
    ),
    Slide(
        id="cheat-sheet-2",
        title="Cheat sheet, continued",
        rule="The rest of draw(), grouped by what it shows.",
        layout="table",
    ),
    Slide(
        id="draw-parameters",
        title="Every draw() parameter",
        rule="Short spellings (view, depth, layout, renderer, node_style) win over their long "
        "twins when both are given.",
        layout="table",
    ),
    Slide(
        id="draw-parameters-2",
        title="Every draw() parameter, continued",
        rule="Pictures in this deck use font_size=16, left to right, label and shape rows unless "
        "a slide says otherwise.",
        layout="table",
    ),
)


#: Cheat-sheet groups: (to see, pass, the Trace.draw parameters listed under it). Every
#: ``Trace.draw`` parameter must appear in exactly one group (checked by the coverage test).
CHEAT_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Everything once, then less",
        "depth, collapse",
        ("depth", "vis_call_depth", "collapse", "collapse_fn"),
    ),
    ("Repeats", "fold_repeats, fold_patterns", ("fold_repeats", "fold_patterns")),
    ("One module", "module=", ("module",)),
    ("Fewer ops", "skip_fn", ("skip_fn",)),
    ("Loops", "view, stack_by", ("view", "vis_mode", "stack_by")),
    (
        "Buffers, containers, orphans",
        "show_buffer_layers, show_containers, show_orphans",
        ("show_buffer_layers", "show_containers", "container_max_inline", "show_orphans"),
    ),
    ("Edits", "vis_intervention_mode", ("vis_intervention_mode", "vis_show_cone")),
    (
        "Measurements",
        "color_by, size_by, node_overlay",
        ("color_by", "size_by", "scale", "node_overlay", "show_saved_for_backward"),
    ),
    (
        "Rows",
        "node_style, node_label_fields",
        ("node_style", "node_mode", "node_label_fields", "show_redundant_args"),
    ),
    ("Gradients", "vis_grad_edge_overrides", ("vis_grad_edge_overrides",)),
    (
        "Data previews and source",
        "show_input_transform_summary, code_panel",
        ("show_input_transform_summary", "code_panel"),
    ),
    (
        "The look",
        "vis_theme, direction, layout",
        (
            "vis_theme",
            "for_paper",
            "direction",
            "layout",
            "vis_node_placement",
            "order_siblings",
            "renderer",
            "vis_renderer",
            "font_size",
            "show_legend",
        ),
    ),
    (
        "Your own styling",
        "node_spec_fn, overrides",
        (
            "node_spec_fn",
            "collapsed_node_spec_fn",
            "vis_graph_overrides",
            "vis_edge_overrides",
            "vis_module_overrides",
        ),
    ),
    (
        "Files",
        "vis_fileformat, vis_outpath",
        ("vis_fileformat", "vis_outpath", "vis_save_only", "dpi", "return_graph"),
    ),
)

#: Unencoded reasons: the exact legend line and when TorchLens writes it.
UNENCODED: tuple[tuple[str, str], ...] = (
    ("n/a = unencoded", "a node has no value for the source; the legend counts them"),
    ("rank mapping (ordinal, not ratio)", 'transform="rank": the fill shows order only'),
    ("log scale (scale-invariant; values <= 0 unencoded)", 'transform="log"'),
    ("values <= 0 -- unencoded under the log transform", "a value at or below zero under log"),
    ("constant value -- unencoded (degenerate domain)", "every value is equal: no ramp is painted"),
    ("varies across passes -- unencoded", "a rolled node whose passes disagree"),
    ("per-pass field -- unencoded on rolled nodes", "a per-pass field on a rolled node"),
    (
        "first-pass-only field -- unencoded on rolled nodes",
        "a field recorded for the first pass only, on a rolled node",
    ),
    ("value from user callable", "the source is your function; TorchLens did not measure it"),
)

#: Side pictures for the other-pictures slide: name, what it shows, how to call it, doc.
OTHER_PICTURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "Bundle graph",
        "several traces of one model, grouped",
        "tl.show_bundle_graph(bundle)",
        "docs/reference/glossary.md",
    ),
    (
        "Bundle diff",
        "two traces side by side, white to red by difference",
        "bundle.show_diff()",
        "docs/reference/glossary.md",
    ),
    (
        "Fastlog preview",
        "which ops a predicate keeps (green), rejects (grey), cannot reach (yellow)",
        "trace.preview_fastlog(predicate)",
        "docs/reference/glossary.md",
    ),
    ("Summary", "a text table of layers", "trace.summary()", "docs/reference/summary.md"),
    ("tviz", "transformer pictures", "torchlens.tviz", "docs/reference/tviz.md"),
    (
        "Model Explorer",
        "export to Google Model Explorer",
        "torchlens.export",
        "docs/reference/model_explorer.md",
    ),
    ("Treescope cards", "notebook cards", "treescope", "docs/guides/treescope_cards.md"),
    ("Offline report", "a standalone HTML report", "report", "docs/guides/offline_report.md"),
    (
        "Dagua renderer",
        "experimental second renderer; channels refuse",
        'renderer="dagua"',
        "torchlens/experimental/dagua",
    ),
)


#: Alphabet sheets: (meaning, teaching slide id, panel, selector). Each cell copies the
#: exact Graphviz attributes of the first mark the selector finds on the teaching slide.
ALPHABET: Mapping[str, tuple[tuple[str, str, str, Selector], ...]] = MappingProxyType(
    {
        "alphabet-nodes": (
            ("input", "start-here", "a", "node(fillcolor={input_color})"),
            ("output", "start-here", "a", "node(fillcolor={output_color})"),
            ("function call", "ovals-boxes", "a", "node(shape=oval, fillcolor=white)"),
            ("leaf module call", "ovals-boxes", "a", "node(shape=box)"),
            ("trainable parameters", "grey-params", "a", "node(fillcolor={trainable})"),
            ("frozen parameters", "grey-params", "a", "node(fillcolor={frozen})"),
            ("trainable and frozen", "grey-params", "a", "node(fillcolor~:)"),
            ("boolean that chose a branch", "branches", "a", "node(fillcolor={bool_color})"),
            ("not from the input", "dashed", "a", "node(style^dashed, text~sigmoid)"),
            ("buffer", "buffers", "a", "node(shape=cylinder)"),
            ("hidden buffers feed it", "buffers", "a", "node(peripheries=2)"),
            ("parameter changed in place", "mutated-param", "a", "node(shape=cylinder)"),
            ("whole module", "depth", "a", "node(shape=box3d)"),
            ("run of ops (segment)", "collapse-max", "a", "node(style^rounded)"),
            ("named pattern", "fold-patterns", "a", "node(text~PATTERN)"),
            ("+N more repeats", "fold-repeats", "a", "node(shape=plaintext, text~more)"),
            ("orphan", "orphans", "a", "node(text~sin)"),
            ("outside a focused module", "focus", "a", "node(fillcolor={input_color})"),
            ("edited site", "interventions", "a", "node(color={site_color})"),
            ("downstream of an edit", "interventions", "a", "node(color={cone_color})"),
            ("hook drawn as a node", "interventions", "b", "node(shape=diamond)"),
            ("recorded edit (surgery)", "surgery", "a", "node(text~edited)"),
            ("grad_fn (backward)", "backward", "a", "node(fillcolor=#F2F3FF)"),
            ("grad_fn of order 2+", "grad-of-grad", "a", "node(fillcolor=#FFF4D6)"),
        ),
        "alphabet-lines": (
            ("data flows here", "start-here", "a", "edge(style=solid)"),
            ("not from the input", "dashed", "a", "edge(style=dashed)"),
            ("through hidden ops", "skip", "a", "edge(text~hidden)"),
            ("argument slot", "arrow-words", "a", 'edge(text~"arg 1")'),
            ("condition starts here", "branches", "a", "edge(text~IF)"),
            ("branch taken", "branches", "a", "edge(text~THEN)"),
            ("passes using this edge", "loop-rolled", "a", "edge(text~In)"),
            ("N edges drawn as one", "arrows-as-one", "a", "edge(text~x2)"),
            ("gradient flow", "combined", "b", "edge(color={grad_color})"),
            ("accumulates into a leaf", "backward", "a", "edge(style=dotted)"),
            ("forward op to its grad_fn", "combined", "a", "edge(constraint=false)"),
            ("reader of a changed parameter", "mutated-param", "a", "edge(style=dashed)"),
            ("member of a container", "containers", "a", "edge(arrowhead=none)"),
            ("module call (outermost)", "module-boxes", "a", "cluster(penwidth={pen_max})"),
            ("module call (deeper)", "module-boxes", "a", "cluster(label~@block.0:1)"),
            ("module not from the input", "dashed", "a", "cluster(style^dashed)"),
            ("orphans group", "orphans", "a", "cluster(label~orphans)"),
            ("one backward pass", "backward", "a", "cluster(label~backward)"),
        ),
    }
)


def slide_ids() -> tuple[str, ...]:
    """Every slide id, in deck order."""

    return tuple(slide.id for slide in SLIDES)


def iter_panel_kwargs() -> Iterator[tuple[str, Panel, str, Any]]:
    """Yield ``(slide_id, panel, kwarg, value)`` for every explicit draw argument."""

    for slide in SLIDES:
        for panel in slide.panels:
            for key, value in panel.kwargs.items():
                yield slide.id, panel, key, value


def deck_text() -> str:
    """All words the deck states: titles, rules, keys, footnotes, labels and tables."""

    parts: list[str] = []
    for slide in SLIDES:
        parts.extend((slide.title, slide.rule, slide.text, slide.footnote))
        parts.extend(key.text for key in slide.keys)
        parts.extend(panel.label for panel in slide.panels)
        parts.extend(repr(dict(panel.kwargs)) for panel in slide.panels)
        parts.extend(cell.label for cell in slide.cells)
    for group in CHEAT_GROUPS:
        parts.extend((group[0], group[1], " ".join(group[2])))
    for row in OTHER_PICTURES:
        parts.extend(row)
    for line in UNENCODED:
        parts.extend(line)
    return "\n".join(parts)
