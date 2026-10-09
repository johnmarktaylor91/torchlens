"""The slide table of the TorchLens visual language deck, as data.

Every slide names its fixture models (by name, see ``models.FIXTURES``), the exact draw
call, the key entries with a selector over the slide's own DOT, and the inventory rows it
covers (``coverage.ROWS``). Nothing here imports torch, so the coverage check can read the
table cheaply. Captions are templates: ``{name}`` fields are filled from TorchLens module
constants by :func:`constants`, so a changed constant changes the caption.

Render conventions (applied by the renderer unless a panel sets ``conventions=False`` or
overrides a key): SVG output, save only, left to right, ``font_size=16``, no legend, the
graph caption hidden, ``collapse="none"``.
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
    layout: str = "key"
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


LR = {"direction": "leftright"}
TRIM = {"node_label_fields": ["label", "shape"]}
CAP = {"vis_graph_overrides": {}}

SLIDES: tuple[Slide, ...] = (
    Slide(
        id="alphabet-nodes",
        title="The TorchLens visual language at a glance: nodes",
        rule="Shape names the kind; fill, border and rows add independent facts. "
        "Each mark is copied from the slide that teaches it.",
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
        title="The TorchLens visual language at a glance: lines and boxes",
        rule="Line style and words on arrows add facts about the data flow; boxes group calls.",
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
        title="Start here: caption, input, output, which way it reads",
        rule="Arrows run from the op that made a tensor to the op that used it, "
        "so the default graph reads bottom to top.",
        panels=(
            Panel(
                "a",
                "Flow",
                kwargs={"direction": "bottomup", "vis_graph_overrides": {}},
                label="draw()",
            ),
        ),
        keys=(
            Key("Caption: model class, tensor count and memory, parameters", "graph(label~Flow)"),
            Key("Green: the input, named @input.<argument>", "node(fillcolor={input_color})"),
            Key("Arrows point from producer to consumer", "edge(style=solid)"),
            Key("Red: the output, pinned to the top", "node(fillcolor={output_color})"),
        ),
        footnote="Bold warning lines join the caption when a capture was poisoned or unverified.",
        caption_kept=True,
        rows=("VN03", "VE01", "VC01", "VC03", "VY01"),
    ),
    Slide(
        id="ovals-boxes",
        title="Ovals are function calls; boxes are calls to a leaf module",
        rule="A module with no submodules that ran one op is drawn as that op in a box; "
        "the same matmul called as a function is an oval.",
        panels=(Panel("a", "TwoWays"),),
        keys=(
            Key("Box: nn.Linear called as a module, @fc", "node(shape=box, text~@fc)"),
            Key("Oval: F.linear called as a function", "node(shape=oval, text~linear_2)"),
            Key("Grey: the op uses parameters", "node(fillcolor={trainable})"),
            Key("White: no parameters", "node(fillcolor=white, shape=oval)"),
        ),
        rows=("VN01", "VN02"),
    ),
    Slide(
        id="node-rows",
        title="What a node says, row by row",
        rule="Bold title, then shape and memory, constructor arguments the shapes do not "
        "already prove, parameters, and the module path.",
        panels=(
            Panel("a", "Anatomy", label="draw()"),
            Panel(
                "b", "Anatomy", kwargs={"show_redundant_args": True}, label="show_redundant_args"
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
        title="Grey means parameters: light trains, dark is frozen, two-tone is both",
        rule="Fill precedence is input, output, boolean, then parameters.",
        panels=(Panel("a", "Params", kwargs=CAP),),
        keys=(
            Key("Light grey {trainable}: all trainable", "node(fillcolor={trainable})"),
            Key("Dark grey {frozen}: all frozen, shapes in brackets", "node(fillcolor={frozen})"),
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
        title="Dashed means not computed from the input",
        rule="A node, edge or module box computed only from parameters, buffers, constants "
        "or random draws is dashed until it meets the input.",
        panels=(Panel("a", "Gate"),),
        keys=(
            Key("Dashed sigmoid: made from a parameter only", "node(style^dashed, text~sigmoid)"),
            Key("Its arrow is dashed too", "edge(style=dashed)"),
            Key("The module box around it is dashed", "cluster(style^dashed)"),
            Key("mul meets the input: solid again", "node(style^solid, text~mul)"),
        ),
        rows=("VN06", "VE01", "VR01"),
    ),
    Slide(
        id="module-boxes",
        title="Module boxes: name, class, depth, repeated calls",
        rule="Each module call is a box titled @path over (Class); the border thins with "
        "depth, {pen_max} pt outermost to {pen_min} pt deepest.",
        panels=(Panel("a", "Nested"),),
        keys=(
            Key("Title @block:1 over (Sequential)", "cluster(label~@block:1)"),
            Key("Thick outer border, thin inner", "cluster(penwidth={pen_max})"),
            Key("Second call: @block:2", "cluster(label~@block:2)"),
            Key("Leaf rows show the call: @block.0:2", "node(text~@block.0:2)"),
        ),
        footnote="Empty modules are never drawn.",
        rows=("VR01", "VR02", "VR03", "VR04", "VL07"),
    ),
    Slide(
        id="arrow-words",
        title="Words on arrows: which argument, and how many",
        rule="Argument slots are named when order matters; {fan_in} or more inputs move the "
        "labels to the middle; xN is N edges drawn as one.",
        layout="text",
        panels=(
            Panel("a", "ArgOrder", kwargs=TRIM, label="ArgOrder"),
            Panel("b", "FanIn", kwargs=TRIM, label="FanIn"),
        ),
        keys=(
            Key("arg 0 and arg 1 into sub", 'edge(text~"arg 1")'),
            Key("No labels into {commute}", None),
            Key("cat([a, a]): two arrows from one op", 'edge(text~"arg 0", head~cat)'),
            Key("At fan-in {fan_in}, labels sit mid-arrow", "edge(label~arg)", panel="b"),
        ),
        rows=("VE02", "VE03"),
    ),
    Slide(
        id="loop-unrolled",
        title="Loops, unrolled: one node per pass",
        rule=':2 is the pass number, and the exact key trace["linear_1_1:2"] accepts.',
        panels=(Panel("a", "Loop", kwargs={"view": "unrolled"}),),
        keys=(
            Key("linear_1_1:2: second pass of the same layer", "node(text~linear_1_1:2)"),
            Key("One box per call: @cell:1 to @cell:3", "node(text~@cell:3)"),
            Key("Passes run left to right", "edge(head~relu_1_2pass2)"),
        ),
        rows=("VL01", "VR04", "VV01"),
    ),
    Slide(
        id="loop-rolled",
        title="Loops, rolled: one node per layer, with pass counts",
        rule="(x3) means one node ran three times with the same weights; In and Out say "
        "which passes used each edge.",
        layout="text",
        panels=(Panel("a", "Loop", kwargs={"view": "rolled", **TRIM}),),
        keys=(
            Key("(x3): ran three times", "node(text~x3)"),
            Key("In 2-3 on the back edge", "edge(text~In)"),
            Key("Out 1-2 at the tail", "edge(text~Out)"),
            Key("A self-loop appears only when the loop carries state", None),
        ),
        rows=("VL01", "VE04", "VE05", "VR04"),
    ),
    Slide(
        id="reuse-shapes",
        title="Reuse, recurrence and changing shapes differ",
        rule="Separate call groups print as :1-2,3-4; shapes that change across calls "
        "print as 2->4.",
        panels=(
            Panel("a", "LoopGroups", kwargs={"view": "rolled"}, label="two call groups"),
            Panel("b", "LoopShapes", kwargs={"view": "rolled"}, label="two shapes"),
        ),
        keys=(
            Key("Call-group suffix on the module", 'node(text~",")'),
            Key("Shape row changes across calls", "node(text~->)", panel="b"),
            Key("Compare (xN): one weight-tied loop", None),
        ),
        rows=("VL01", "VL03"),
    ),
    Slide(
        id="branches",
        title="Branches: the boolean that decided, and which arm ran",
        rule="Yellow TRUE is the comparison that chose the arm; it has no outgoing arrow "
        "because it steered Python, not tensors.",
        layout="text",
        panels=(Panel("a", "Branch", kwargs=TRIM),),
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
        title="Buffers: cylinders, and a double border when hidden",
        rule='"meaningful" hides {noise_buffers}; the consumer gets a double outline and a '
        'tooltip; "never" hides every buffer.',
        panels=(
            Panel("a", "Stateful", label='show_buffer_layers="meaningful"'),
            Panel("b", "Stateful", kwargs={"show_buffer_layers": "always"}, label='"always"'),
        ),
        keys=(
            Key("@scale: a buffer drawn as a cylinder", "node(shape=cylinder, text~@scale)"),
            Key("Double outline: hidden buffers feed this op", "node(peripheries=2)"),
            Key('"always" shows the hidden statistics', "node(text~running_mean)", panel="b"),
        ),
        footnote='show_buffer_layers="never" hides every buffer.',
        rows=("VN07", "VN08"),
    ),
    Slide(
        id="mutated-param",
        title="A parameter changed in place",
        rule="A dashed grey cylinder stands for the Parameter itself when an in-place op "
        "changed it during forward; dashed edges run to its earlier readers.",
        panels=(
            Panel("a", "Clamp", kwargs={"show_legend": True}, label="trainable"),
            Panel("b", "MutFrozen", label="frozen"),
        ),
        keys=(
            Key("parameter temp (1,)", "node(shape=cylinder, text~parameter)"),
            Key("Dashed black edges to pre-mutation readers", "edge(style=dashed)"),
            Key("The legend gains this row only now", 'node(text~"mutated parameter")'),
            Key("A frozen Parameter takes the dark grey", "node(shape=cylinder)", panel="b"),
        ),
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
            ),
        ),
        keys=(
            Key("The orphans group title", "cluster(label~orphans)"),
            Key("exp ran, but nothing used it", "node(text~exp, style^dashed)"),
            Key("Without keep_orphans at capture, the draw warns", None),
        ),
        rows=("VN14", "VR13"),
    ),
    Slide(
        id="containers",
        title="Inputs and outputs that are dicts, tuples, lists",
        rule='"labels" names keys on arrows; "cluster" draws a dotted group; "collapsed" '
        'folds more than {container_max_inline} same-shape leaves; "nodes" adds a record box.',
        panels=(
            Panel(
                "a",
                "DictOut",
                capture={"capture_container_structure": True},
                kwargs={"show_containers": "nodes"},
                label='"nodes"',
            ),
            Panel(
                "b",
                "ListOut",
                capture={"capture_container_structure": True},
                kwargs={"show_containers": "collapsed"},
                label='"collapsed"',
            ),
        ),
        keys=(
            Key("Dashed record box for the returned dict", "node(text~dict)"),
            Key("Arrow words name the keys", "edge(text~logits)"),
            Key("Arrowless dashed ties mark membership", "edge(arrowhead=none)"),
            Key("14 same-shape leaves folded into one", "node(text~x14)", panel="b"),
        ),
        footnote='"auto" behaves as "collapsed" today; "cluster" draws a dotted group.',
        rows=("VN24", "VN25", "VE12", "VR09"),
    ),
    Slide(
        id="depth",
        title="Whole modules as one node: depth and collapse_fn",
        rule="The 3D box names the module and counts the ops, buffers and parameters inside.",
        panels=(Panel("a", "Nested", kwargs={"depth": 1}),),
        keys=(
            Key("@block:1 as a 3D box", "node(shape=box3d, text~@block:1)"),
            Key("Class, output shape and memory", "node(shape=box3d, text~Sequential)"),
            Key("Ops and parameters inside", "node(shape=box3d, text~params)"),
        ),
        footnote="collapse_fn(module) chooses the modules to fold instead of a depth.",
        rows=("VN10", "VR05", "VE03"),
    ),
    Slide(
        id="collapse-max",
        title='collapse="max" and the float ladder',
        rule="A dashed rounded chip stands for a run of ops or blocks and names an address "
        "range; a float in [0, 1] walks a monotone ladder.",
        panels=(
            Panel("a", "Stack", kwargs={"collapse": "max", **CAP}, label='collapse="max"'),
            Panel("b", "Stack", kwargs={"collapse": 0.5}, label="collapse=0.5"),
        ),
        keys=(
            Key("Chip: a run of ops and the modules it spans", "node(style^rounded)"),
            Key("Caption discloses the collapse", "graph(label~collapse)"),
            Key("A float picks a point on the ladder", "node(style^rounded)", panel="b"),
        ),
        footnote="A floor fallback prints bold in the caption; an orange dashed edge is a "
        "projection artifact, not a cycle.",
        caption_kept=True,
        rows=("VN11", "VR06", "VR07", "VE13"),
    ),
    Slide(
        id="collapse-auto",
        title='collapse="auto" on a graph that needs it',
        rule='"auto" picks the first readable schedule point; on a small graph it changes nothing.',
        panels=(Panel("a", "BigStack", kwargs={"collapse": "auto"}, label='collapse="auto"'),),
        keys=(
            Key("36 ops become a few chips", "node(style^rounded)"),
            Key("Below the readable band, auto leaves the graph alone", None),
        ),
        rows=("VR06",),
    ),
    Slide(
        id="fold-repeats",
        title="Repeated blocks: one shown, +N more",
        rule="One representative and an honest count of separate blocks with their own "
        "weights; compare (xN), the same weights applied N times.",
        panels=(Panel("a", "Stack", kwargs={"fold_repeats": True}),),
        keys=(
            Key("The representative, scoped @blocks.0 only", "node(text~only)"),
            Key("... +3 more Block", "node(shape=plaintext, text~more)"),
            Key("Arrows route through the ellipsis", "edge(head~ellipsis)"),
        ),
        rows=("VN13", "VE16", "VR06"),
    ),
    Slide(
        id="fold-patterns",
        title="Named patterns: fold_patterns",
        rule="PATTERN chips replace runs that match a named idiom; a mapping declares your "
        'own; patterns refuse to combine with collapse other than "none".',
        panels=(
            Panel("a", "ConvBnRelu2", kwargs={"fold_patterns": "idiomatic"}, label="idiomatic"),
            Panel(
                "b",
                "AddRelu",
                kwargs={"fold_patterns": {"AddRelu": "add > relu"}},
                label='{"AddRelu": "add > relu"}',
            ),
        ),
        keys=(
            Key("First ConvBnRelu chip", "node(text~PATTERN)"),
            Key("Your own pattern from a mapping", "node(text~AddRelu)", panel="b"),
            Key('With collapse other than "none", the draw refuses', None),
        ),
        rows=("VN12",),
    ),
    Slide(
        id="focus",
        title="Focus on one module: module=",
        rule="Only the module's own ops; green and red ext: ovals stand for the outside; "
        'trace.modules["head"].draw() is the same.',
        panels=(Panel("a", "Nested", kwargs={"module": "head"}),),
        keys=(
            Key("Green ext: where data enters", "node(fillcolor={input_color}, text~ext)"),
            Key("Red: where data leaves", "node(fillcolor={output_color})"),
            Key("Only the module's own ops", "node(text~@head)"),
        ),
        rows=("VN15", "VR08"),
    ),
    Slide(
        id="skip",
        title="Hiding ops: skip_fn and the lens filter",
        rule="A dashed bridge labelled via N hidden means reachable through omitted work, "
        "not adjacent; a lens adds a bridge key and counts.",
        layout="text",
        panels=(
            Panel("a", "FlowReshape", kwargs={"skip_fn": "@skip_reshape", **TRIM}, label="skip_fn"),
            Panel(
                "b",
                "FlowReshape",
                call="lens",
                kwargs={"lens": "overview", "display_filter": "@exclude_reshapes", **TRIM},
                label="lens filter",
            ),
        ),
        keys=(
            Key("The reshape is gone", None),
            Key("via 1 hidden on a dashed bridge", "edge(text~hidden)"),
            Key("The lens filter adds its own disclosure", "edge(text~hidden)", panel="b"),
        ),
        rows=("VE07", "VK10"),
    ),
    Slide(
        id="interventions",
        title="Interventions: site, cone, or a hook node",
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
            Key("Hook drawn as a diamond", "node(shape=diamond)", panel="b"),
            Key("vis_show_cone=False hides the cone", None),
        ),
        rows=("VN16", "VN17", "VN18"),
    ),
    Slide(
        id="surgery",
        title="What an edit actually did: the surgery lens",
        rule="Solid magenta cites a recorded fire; dashed is a declared target with no fire "
        "here; the caption census says which replacement lane ran.",
        panels=(
            Panel(
                "a",
                "Shared",
                call="surgery",
                capture={"intervention_ready": True, "save_arg_values": True},
                prep="fork_edit",
                kwargs=CAP,
            ),
        ),
        keys=(
            Key("Fact: a recorded fire, solid", "node(text~spliced)"),
            Key("Heuristic: declared target, dashed", "node(text~declared)"),
            Key("Census lines in the caption", "graph(label~census)"),
        ),
        footnote="Up to {surgery_cap} citation rows per node. An intervention replay with "
        "direct writes adds the caption line Direct writes detected.",
        caption_kept=True,
        rows=("VN19", "VC02", "VC05"),
    ),
    Slide(
        id="surgery-diff",
        title="Two captures need honest joins",
        rule="Solid joins use site identity; dashed joins are heuristic; a missing record "
        "does not prove missing execution.",
        panels=(
            Panel(
                "a",
                "Shared",
                call="surgery_diff",
                capture={"intervention_ready": True, "save_arg_values": True},
                prep="fork_edit",
            ),
        ),
        keys=(
            Key("Solid grey join: same site", "edge(style=solid, dir=none)"),
            Key("Dashed join: matched by label or position", None),
            Key("not recorded: one side has no record", None),
        ),
        rows=("VE15",),
    ),
    Slide(
        id="color-by",
        title="Colour a measurement: color_by",
        rule="Fill ramps light to blue by value; the legend appears by itself and states "
        "source, coverage, scale and min, mid and max; a channel forces dot.",
        panels=(Panel("a", "SmallCnn", kwargs={"color_by": "flops", "show_legend": None}),),
        keys=(
            Key("Ramp: light is low, blue is high", "node(fillcolor~#0072B2)"),
            Key("The encoding legend appears by itself", "node(text~color_by)"),
            Key("Unencoded nodes stay white and are counted", "node(text~encoded)"),
        ),
        rows=("VK01", "VK06", "VI02"),
    ),
    Slide(
        id="color-transforms",
        title="Rank, linear and log are different comparisons; unencoded is not zero",
        rule="Rank shows order, linear shows position in the range, log leaves nonpositive "
        "values unencoded; varying and missing values stay unencoded with the reason.",
        layout="grid",
        panels=(
            Panel(
                "a",
                "Shapes",
                kwargs={"color_by": "@bytes_linear", "show_legend": None},
                label="linear",
            ),
            Panel(
                "b", "Shapes", kwargs={"color_by": "@bytes_rank", "show_legend": None}, label="rank"
            ),
            Panel(
                "c", "Shapes", kwargs={"color_by": "@bytes_log", "show_legend": None}, label="log"
            ),
            Panel(
                "d",
                "LoopShapes",
                kwargs={"view": "rolled", "color_by": "step_index", "show_legend": None},
                label="rolled",
            ),
        ),
        keys=(
            Key("rank mapping is ordinal, not a ratio", "node(text~rank)", panel="b"),
            Key("Rolled nodes whose value varies stay unencoded", "node(text~varies)", panel="d"),
        ),
        rows=("VK02", "VK03"),
    ),
    Slide(
        id="size-by",
        title="Size a measurement: size_by and scale",
        rule="Area, not side, follows the non-batch element count, clamped at {size_max_area}x; "
        "labels never shrink.",
        panels=(
            Panel(
                "a",
                "SmallCnn",
                kwargs={"size_by": "dims", "scale": "sqrt", "show_legend": None},
            ),
        ),
        keys=(
            Key("Bigger tensor, bigger node", "node(fixedsize=false)"),
            Key("sqrt by default; linear grows faster", None),
            Key("The legend states the size rule", "node(text~size)"),
        ),
        rows=("VK04", "VK06"),
    ),
    Slide(
        id="stack-by",
        title="Line passes up: stack_by",
        rule="Nodes sharing a pass index pin to one rank; same rank means the same "
        "annotation, not parallel execution; rolled views refuse.",
        panels=(
            Panel("a", "LockstepDecoder", kwargs={"stack_by": True, "show_legend": None, **CAP}),
        ),
        keys=(
            Key("Each column is one pass", None),
            Key("Caption: stacked by pass_index", "graph(label~stacked)"),
            Key("The legend explains the rank rule", "node(text~stack_by)"),
            Key("Sibling ordering is off while stacking", None),
        ),
        caption_kept=True,
        rows=("VK05", "VR14", "VK06"),
    ),
    Slide(
        id="overlays",
        title="Overlays: one more row, and a border that flags",
        rule="nan: yes adds an orange 3 pt border; n/a means not checkable; any nonzero "
        "numeric overlay thickens the border.",
        panels=(
            Panel("a", "NanMaker", kwargs={"node_overlay": "nan"}, label='node_overlay="nan"'),
            Panel("b", "Flow", kwargs={"node_overlay": "@score_map"}, label="a mapping"),
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
        rule="Profiling appends time, storage and call site; saved for backward measures "
        "retained tensors; explicit fields replace the default rows.",
        panels=(
            Panel(
                "a",
                "Grad",
                kwargs={"node_style": "profiling", "show_saved_for_backward": True},
                label='node_style="profiling"',
            ),
            Panel(
                "b",
                "Flow",
                kwargs={"node_label_fields": ["label", "shape", "time"]},
                label="node_label_fields",
            ),
        ),
        keys=(
            Key("t= time and out= storage", "node(text~t=)"),
            Key("call= and fn= name the source line", "node(text~call=)"),
            Key("saved for backward: N tensors", "node(text~saved)"),
            Key("Fields you list replace the defaults", "node(text~tanh)", panel="b"),
        ),
        footnote="Vision and attention rows exist only as experimental node_spec_fn "
        "callbacks in torchlens.experimental.node_styles.",
        rows=("VL04", "VL09", "VL10", "VL11", "VK08"),
    ),
    Slide(
        id="lenses",
        title="Lenses: presets that answer one question",
        rule="A lens chooses sources, detail and disclosures; a skin changes the ink; anything "
        "you pass wins; a lens may refuse when evidence is missing.",
        layout="grid",
        panels=(
            Panel("a", "SmallCnn", call="lens", kwargs={"lens": "overview"}, label="overview"),
            Panel("b", "SmallCnn", call="lens", kwargs={"lens": "speed"}, label="speed"),
            Panel("c", "SmallCnn", call="lens", kwargs={"lens": "dims"}, label="dims"),
        ),
        rows=("VK09",),
    ),
    Slide(
        id="nonfinite",
        title="Not checked is different from finite",
        rule="Nonfinite states use stripes, wedges or a border; grey dashed means not "
        "checked, never finite.",
        layout="grid",
        panels=(Panel("a", "Nonfinite", call="lens", kwargs={"lens": "debug"}),),
        cells=(
            Cell("a", "node(text~add)", "finite"),
            Cell("a", "node(text~log_1)", "nan"),
            Cell("a", "node(text~truediv_1)", "pos_inf"),
            Cell("a", "node(text~truediv_2)", "neg_inf"),
            Cell("a", "node(text~log_2)", "mixed"),
            Cell("a", "node(text~stack)", "not_checked"),
        ),
        rows=("VN29",),
    ),
    Slide(
        id="backward",
        title="The backward graph",
        rule="Lavender ovals are grad_fns named by their forward op; dotted accum edges feed "
        "leaves; cream is order two or more; the key lists only styles painted.",
        panels=(
            Panel(
                "a",
                "Grad",
                call="draw_backward",
                capture={"save_grads": "all"},
                prep="log_backward",
                kwargs={"show_legend": True},
                label="draw_backward",
            ),
            Panel(
                "b",
                "HigherOrder",
                call="draw_backward",
                capture={"save_grads": "all"},
                prep="higher_order",
                kwargs={"vis_mode": "unrolled", "show_legend": True},
                label="grad of grad",
            ),
        ),
        keys=(
            Key("grad_fn node", "node(fillcolor=#F2F3FF)"),
            Key("Dotted accum edge to a leaf", "edge(style=dotted)"),
            Key("Order two or more: cream", "node(fillcolor=#FFF4D6)", panel="b"),
            Key("The backward key", "node(text~grad_fn)"),
        ),
        rows=("VN26", "VE09", "VR11", "VI03", "VC04", "VV02"),
    ),
    Slide(
        id="combined",
        title="Forward and backward together",
        rule="Dashed lavender ties join each forward op to its grad_fn; on a forward graph, "
        "purple reverse arrows are captured gradients.",
        panels=(
            Panel(
                "a",
                "Grad",
                call="draw_combined",
                capture={"save_grads": "all"},
                prep="log_backward",
                label="draw_combined",
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
        rule='Meaning stays, ink changes; for_paper=True forces "paper"; "colorblind" '
        "changes borders, not the palette.",
        layout="grid",
        panels=tuple(
            Panel(name, "Flow", kwargs={"vis_theme": skin}, label=skin)
            for name, skin in zip(
                "abcde", ("torchlens", "paper", "dark", "colorblind", "high_contrast"), strict=True
            )
        ),
        footnote="Orphans, container nodes, backward nodes and overlay borders ignore the skin.",
        rows=("VT01", "VT02"),
    ),
    Slide(
        id="direction-layout",
        title="Direction, layout engine, sibling order",
        rule='Direction rotates the layout, not the computation; "auto" switches to the rank '
        "engine above {rank_cost} cost units, losing orphans and sibling ordering.",
        layout="grid",
        panels=(
            Panel("a", "Flow", kwargs={"direction": "bottomup"}, label="bottomup (default)"),
            Panel("b", "Flow", kwargs={"direction": "topdown"}, label="topdown"),
            Panel("c", "Flow", kwargs={"direction": "leftright"}, label="leftright"),
            Panel(
                "d",
                "FanIn",
                kwargs={"layout": "rank", "show_legend": True, "direction": "topdown"},
                label='layout="rank"',
            ),
        ),
        footnote="order_siblings=True keeps fan-outs in call order using invisible ordering "
        "edges; the rank engine pins its legend at the left.",
        rows=("VY01", "VY02", "VR10", "VE14", "VI04"),
    ),
    Slide(
        id="text-size",
        title="Text size and role are separate controls",
        rule="font_size grows node rows; edge labels ({annotation_pt} pt), legend text "
        "({secondary_pt} pt) and branch labels ({emphasis_pt} pt) are fixed roles.",
        panels=(
            Panel("a", "ArgOrder", kwargs={"font_size": 14, **TRIM}, label="font_size=14"),
            Panel("b", "ArgOrder", kwargs={"font_size": 22, **TRIM}, label="font_size=22"),
        ),
        keys=(
            Key("Node text grew", "node(text~sub)", panel="b"),
            Key("arg 0 did not", "edge(text~arg)", panel="b"),
            Key("dpi changes raster pixels only", None),
        ),
        rows=("VT03",),
    ),
    Slide(
        id="code-panel",
        title="Source beside the graph: code_panel",
        rule="Captured source in Courier on the right with an Open source link, up to "
        "{max_code_lines} lines; class, init+forward or a function choose the text.",
        panels=(Panel("a", "Flow", kwargs={"code_panel": "forward"}),),
        keys=(
            Key("The forward source", None),
            Key('"class" and "init+forward" show more', None),
            Key("Missing source is stated in the panel", None),
        ),
        rows=("VO04",),
    ),
    Slide(
        id="builtin-legend",
        title="The legend TorchLens draws, and what it leaves out",
        rule="show_legend=True adds seven fill roles and nothing else; None shows only an "
        "active channel's section; False hides even that.",
        panels=(Panel("a", "Stateful", kwargs={"show_legend": True}),),
        keys=(
            Key("The built-in legend table", 'node(text~"TorchLens legend")'),
            Key("No shapes, dashes, double borders, box weights or arrow words", None),
            Key("Channel and backward sections join when they apply", None),
        ),
        rows=("VN27", "VI01", "VI02"),
    ),
    Slide(
        id="custom",
        title="Custom styling is explicitly custom",
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
        title="Export choices create no new facts",
        rule="DPI changes raster pixels; vector geometry, save behaviour and the returned "
        "object are separate controls; closed vocabularies refuse before any render.",
        layout="table",
        rows=("VO01", "VO02", "VO05", "VV01"),
    ),
    Slide(
        id="data-previews",
        title="Inputs and outputs shown as data",
        rule="A picture node replaces the outline; the transform row names the preprocessing "
        "and whether it was verified; output rows can show decoded labels and scores.",
        panels=(
            Panel(
                "a",
                "ImageNet",
                capture={"transform": "@to_tensor", "output_style": "classification"},
                kwargs={"show_input_transform_summary": True},
            ),
        ),
        keys=(
            Key("The input images as a montage", "node(shape=none)"),
            Key("preprocess names the transform, UNVERIFIED here", "node(text~preprocess)"),
            Key("Decoded output rows", "node(text~output)"),
        ),
        footnote="Batch policy auto shows 4 examples, all 16, or first, first_n:N, shape_only.",
        rows=("VN21", "VN22", "VN23", "VL12"),
    ),
    Slide(
        id="other-pictures",
        title="Other TorchLens pictures",
        rule="Bundle graph and diff, fastlog preview, summary, tviz, Model Explorer, treescope "
        "cards and the offline report have their own keys; red delta is not an intervention.",
        layout="table",
        rows=("VS01", "VS02", "VS03", "VS04", "VI05", "VY03"),
    ),
    Slide(
        id="cheat-sheet",
        title="Cheat sheet: to see X, pass Y",
        rule="Every draw() parameter once, grouped by what it shows, with its default.",
        layout="table",
        rows=("VO06",),
    ),
    Slide(
        id="draw-parameters",
        title="Every draw() parameter and its default",
        rule="Short spellings (view, depth, layout, renderer, node_style) win over their "
        "long twins when both are given.",
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
            ("leaf module call", "ovals-boxes", "a", "node(shape=box, text~@fc)"),
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
            ("orphan", "orphans", "a", "node(text~exp, style^dashed)"),
            ("outside a focused module", "focus", "a", "node(text~ext)"),
            ("edited site", "interventions", "a", "node(color={site_color})"),
            ("downstream of an edit", "interventions", "a", "node(color={cone_color})"),
            ("hook drawn as a node", "interventions", "b", "node(shape=diamond)"),
            ("recorded edit (surgery)", "surgery", "a", "node(text~spliced)"),
            ("grad_fn (backward)", "backward", "a", "node(fillcolor=#F2F3FF)"),
            ("grad_fn of order 2+", "backward", "b", "node(fillcolor=#FFF4D6)"),
        ),
        "alphabet-lines": (
            ("data flows here", "start-here", "a", "edge(style=solid)"),
            ("not from the input", "dashed", "a", "edge(style=dashed)"),
            ("through hidden ops", "skip", "a", "edge(text~hidden)"),
            ("argument slot", "arrow-words", "a", 'edge(text~"arg 1")'),
            ("condition starts here", "branches", "a", "edge(text~IF)"),
            ("branch taken", "branches", "a", "edge(text~THEN)"),
            ("passes using this edge", "loop-rolled", "a", "edge(text~In)"),
            ("N edges drawn as one", "depth", "a", "edge(text~x2)"),
            ("gradient flow", "backward", "a", "edge(color={grad_color})"),
            ("accumulates into a leaf", "backward", "a", "edge(style=dotted)"),
            ("forward op to its grad_fn", "combined", "a", "edge(constraint=false)"),
            ("reader of a changed parameter", "mutated-param", "a", "edge(style=dashed)"),
            ("member of a container", "containers", "a", "edge(arrowhead=none)"),
            ("module call (outermost)", "module-boxes", "a", "cluster(penwidth={pen_max})"),
            ("module call (deepest)", "module-boxes", "a", "cluster(penwidth={pen_min})"),
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
    return "\n".join(parts)
