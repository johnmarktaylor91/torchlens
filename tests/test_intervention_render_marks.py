"""How a forward picture marks an intervention (JMT's 29 Oct 2026 pick, F2).

The edited node gets a purple ring and a bold purple edit row (``Δ zero_ablate``);
the header names the edit; the downstream highlighter (a translucent purple
underlay under changed arrows and node outlines) is off by default and drawn
as an SVG step after layout on both layout engines.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from example_models import TinyReluAdd
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.intervention.selectors import grad_fn_label
from torchlens.intervention.types import Edit, FireRecord
from torchlens.options import VisualizationOptions
from torchlens.visualization._edit_highlighter import HighlighterStyle, highlight_svg
from torchlens.visualization._edit_marks import (
    EditMark,
    EditRow,
    edit_ink,
    edit_marks_by_site,
    edit_rows,
    header_edit_line,
    short_edit_text,
)
from torchlens.visualization.node_spec import intervention_sites_for_log, render_lines_to_html
from torchlens.visualization.themes import THEME_PRESETS

PURPLE = "#7B1FA2"
DARK_PURPLE = "#C084FC"


def _traced(intervene: Any = None) -> tl.Trace:
    """Trace ``relu(x) + 1`` with a zero ablation at the relu."""

    return tl.trace(
        TinyReluAdd(),
        torch.randn(2, 3),
        intervene=intervene or tl.when(tl.func("relu"), tl.zero_ablate()),
    )


def test_edit_text_rule() -> None:
    """Name alone without arguments; short literals written in; stand-ins otherwise."""

    def named_fn(grad: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Return the gradient unchanged."""

        del ctx
        return grad

    assert short_edit_text(tl.zero_ablate()) == "zero_ablate"
    assert short_edit_text(tl.scale(0.5)) == "scale(0.5)"
    assert short_edit_text(tl.clamp(low=0, high=1)) == "clamp(low=0, high=1)"
    assert short_edit_text(tl.add_noise(0.1)) == "add_noise(0.1)"
    assert short_edit_text(tl.steer(torch.ones(3))) == "steer(v)"
    # A label argument is written in (built directly: swap_with refuses a bare label at run time).
    assert short_edit_text(Edit("swap_with", args=("relu_2_1",))) == "swap_with(relu_2_1)"
    assert short_edit_text(tl.hook_grad(named_fn)) == "hook_grad(named_fn)"
    assert short_edit_text(tl.hook_grad(lambda g, *, ctx: g)) == "hook_grad(fn)"
    assert short_edit_text(tl.zero_grad()) == "zero_grad"


def test_node_rows_cap_at_three_then_more() -> None:
    """Several edits take one row each in order, three at most, then ``+N more``."""

    marks = [EditMark(short=f"e{index}", full="", detail="") for index in range(5)]
    rows = edit_rows(marks, PURPLE)
    assert [str(row) for row in rows] == ["Δ e0", "Δ e1", "Δ e2", "+2 more"]
    assert all(isinstance(row, EditRow) and row.color == PURPLE for row in rows)


def test_edit_row_html_spec() -> None:
    """The edit row is the last row after a 4 pt gap: delta 11 pt, text 10 pt, bold purple."""

    # On 11 pt op labels (the mockups) the row is 11 pt over 10 pt; it scales with the label.
    label = render_lines_to_html(["relu_1_1", "(2, 3)", EditRow("Δ zero_ablate", PURPLE, 11.0)])
    assert label.index("(2, 3)") < label.index('POINT-SIZE="4"') < label.index("zero_ablate")
    assert (
        f'<FONT COLOR="{PURPLE}"><B><FONT POINT-SIZE="11">Δ</FONT>'
        '<FONT POINT-SIZE="10">&#160;zero_ablate</FONT></B></FONT>'
    ) in label
    assert "<HR" not in label.upper()
    default = render_lines_to_html(["relu_1_1", EditRow("Δ zero_ablate", PURPLE)])
    assert (
        '<FONT POINT-SIZE="14">Δ</FONT><FONT POINT-SIZE="12.7">&#160;zero_ablate</FONT>' in default
    )


def _fired_site(label: str, *edits: Any) -> SimpleNamespace:
    """Return a stand-in site whose fire records name ``edits``."""

    records = [
        FireRecord(target_label=label, helper=edit, timestamp=float(index))
        for index, edit in enumerate(edits)
    ]
    return SimpleNamespace(label=label, layer_label=label, interventions=records)


def test_header_edit_line_forms() -> None:
    """One edit, several comma-separated, one edit on many sites, and the cap."""

    trace = SimpleNamespace(_intervention_spec=None)
    one = header_edit_line(trace, [_fired_site("linear_1_1", tl.zero_ablate())], PURPLE)
    assert one == (
        f'1 edit:&#160;<FONT COLOR="{PURPLE}"><B>Δ zero_ablate</B></FONT>&#160;on linear_1_1'
    )
    many = [_fired_site(f"linear_{i}_1", tl.zero_ablate()) for i in range(12)]
    assert header_edit_line(trace, many, PURPLE).endswith("&#160;on 12 layers")
    two = header_edit_line(
        trace,
        [_fired_site("relu_1_1", tl.zero_ablate()), _fired_site("relu_2_1", tl.scale(0.5))],
        PURPLE,
    )
    assert two is not None and two.startswith("2 edits:&#160;")
    assert ",&#160;<FONT" in two and "scale(0.5)" in two
    four = [
        _fired_site("a_1_1", tl.zero_ablate()),
        _fired_site("b_1_1", tl.scale(0.5)),
        _fired_site("c_1_1", tl.add_noise(0.1)),
        _fired_site("d_1_1", tl.zero_grad()),
    ]
    assert header_edit_line(trace, four, PURPLE) == "4 edits (marked)"


def test_edit_ink_per_skin() -> None:
    """The four light skins draw ``#7B1FA2``; the dark skin its lighter tint."""

    for name in ("torchlens", "paper", "colorblind", "high_contrast"):
        assert edit_ink(THEME_PRESETS[name]) == PURPLE
    assert edit_ink(THEME_PRESETS["dark"]) == DARK_PURPLE


def test_cone_is_off_by_default() -> None:
    """JMT's 29 Apr ruling, confirmed 29 Oct: downstream marking off unless asked for."""

    assert VisualizationOptions().show_cone is False


def test_edited_node_dot(tmp_path: Path) -> None:
    """The edited node keeps its fill and gains the ring and edit row; the header names it."""

    trace = _traced()
    try:
        dot = trace.draw(vis_save_only=True, file_format="dot", path=str(tmp_path / "toy"))
    finally:
        trace.cleanup()
    relu_line = next(line for line in dot.splitlines() if "tl_edited" in line)
    assert f'color="{PURPLE}"' in relu_line or f"color={PURPLE}" in relu_line.replace('"', "")
    assert "penwidth=2.5" in relu_line.replace('"', "")
    assert "zero_ablate" in relu_line
    assert "1 edit:" in dot and "on relu_1_1" in dot
    assert "tl_downstream" not in dot
    assert "#FF00FF" not in dot and "#FFB3FF" not in dot


def test_dark_skin_uses_its_tint(tmp_path: Path) -> None:
    """The dark skin draws ring, row and header token in ``#C084FC``."""

    trace = _traced()
    try:
        dot = trace.draw(
            vis_save_only=True,
            file_format="dot",
            path=str(tmp_path / "dark"),
            theme="dark",
        )
    finally:
        trace.cleanup()
    assert dot.count(DARK_PURPLE) >= 3
    assert PURPLE not in dot


def _svg_group(svg: str, cls: str) -> str:
    """Return the first SVG group whose class list contains ``cls``."""

    match = re.search(rf'<g id="[^"]+" class="[^"]*\b{cls}\b[^"]*">', svg)
    assert match is not None, cls
    return svg[match.start() : svg.index("</g>", match.end())]


_SAMPLE_SVG = """<svg><g id="graph0" class="graph"><title>G</title>
<g id="node1" class="node tl_edited"><title>a</title>
<g id="a_node1"><a xlink:title="x"><polygon fill="grey" stroke="#7B1FA2" points="0,0 1,1"/></a></g>
</g>
<g id="node2" class="node tl_downstream"><title>b</title>
<ellipse fill="white" stroke="black" cx="5" cy="5" rx="2" ry="1"/>
</g>
<g id="node3" class="node"><title>c</title><ellipse fill="white" stroke="black" cx="9" cy="9" rx="2" ry="1"/></g>
<g id="edge1" class="edge"><title>a&#45;&gt;b</title>
<path fill="none" stroke="black" d="M0,0C1,1 2,2 3,3"/>
<polygon fill="black" stroke="black" points="3,3 4,4 3,4"/>
</g>
<g id="edge2" class="edge"><title>c&#45;&gt;a</title><path fill="none" stroke="black" d="M9,9L0,0"/></g>
<g id="edge3" class="edge"><title>b:out&#45;&gt;c</title><path fill="none" stroke="black" d="M5,5L9,9"/></g>
</g></svg>"""


def test_highlight_svg_lays_underlays_where_f2_says() -> None:
    """Out-edges of marked nodes and downstream outlines get an underlay; nothing else does."""

    style = HighlighterStyle(color=PURPLE, opacity=0.25)
    out = highlight_svg(_SAMPLE_SVG, style)
    underlay = 'stroke="#7B1FA2" stroke-opacity="0.25" stroke-width="8"'
    edge1 = out[out.index('<g id="edge1"') : out.index('<g id="edge2"')]
    assert edge1.count(underlay) == 2  # shaft and arrowhead
    assert edge1.index(underlay) < edge1.index('stroke="black"')  # painted underneath
    edge2 = out[out.index('<g id="edge2"') : out.index('<g id="edge3"')]
    assert underlay not in edge2  # into the edited node: plain
    edge3 = out[out.index('<g id="edge3"') :]
    assert underlay in edge3  # from a downstream node, through a port
    assert _svg_group(out, "tl_downstream").count(underlay) == 1
    assert underlay not in _svg_group(out, "tl_edited")  # the ring is the edited mark
    node3 = out[out.index('<g id="node3"') : out.index('<g id="edge1"')]
    assert underlay not in node3
    assert highlight_svg(
        _SAMPLE_SVG.replace("tl_edited", "").replace("tl_downstream", ""), style
    ) == (_SAMPLE_SVG.replace("tl_edited", "").replace("tl_downstream", ""))


@pytest.mark.parametrize("placement", ["dot", "rank"])
def test_highlighter_in_svg_on_both_engines(tmp_path: Path, placement: str) -> None:
    """``vis_show_cone=True`` lays the highlighter into the SVG on dot and the rank layout."""

    trace = _traced()
    try:
        trace.draw(
            vis_save_only=True,
            file_format="svg",
            path=str(tmp_path / f"cone_{placement}"),
            vis_show_cone=True,
            vis_node_placement=placement,
        )
        trace.draw(
            vis_save_only=True,
            file_format="svg",
            path=str(tmp_path / f"plain_{placement}"),
            vis_node_placement=placement,
        )
    finally:
        trace.cleanup()
    cone = (tmp_path / f"cone_{placement}.svg").read_text(encoding="utf-8")
    plain = (tmp_path / f"plain_{placement}.svg").read_text(encoding="utf-8")
    assert "tl_downstream" in cone
    assert cone.count('stroke-opacity="0.25"') >= 3  # node outline plus two arrow shapes
    assert "stroke-opacity" not in plain
    assert "tl_edited" in plain


def test_highlighter_dark_opacity(tmp_path: Path) -> None:
    """The dark skin's highlighter is ``#C084FC`` at 0.35."""

    trace = _traced()
    try:
        trace.draw(
            vis_save_only=True,
            file_format="svg",
            path=str(tmp_path / "dark"),
            vis_show_cone=True,
            theme="dark",
        )
    finally:
        trace.cleanup()
    svg = (tmp_path / "dark.svg").read_text(encoding="utf-8")
    assert f'stroke="{DARK_PURPLE}" stroke-opacity="0.35"' in svg


def test_highlighter_pdf_and_png_from_processed_svg(tmp_path: Path) -> None:
    """PDF and PNG carry the highlighter: they are written from the processed SVG."""

    pytest.importorskip("cairosvg")
    trace = _traced()
    try:
        for fileformat in ("pdf", "png"):
            trace.draw(
                vis_save_only=True,
                file_format=fileformat,
                path=str(tmp_path / "cone"),
                vis_show_cone=True,
            )
    finally:
        trace.cleanup()
    assert (tmp_path / "cone.pdf").read_bytes().startswith(b"%PDF")
    png = (tmp_path / "cone.png").read_bytes()
    assert png.startswith(b"\x89PNG")
    from PIL import Image

    with Image.open(tmp_path / "cone.png") as image:
        pixels = image.convert("RGB").getdata()
        # The translucent purple over white is a pale violet no other mark uses.
        assert any(r > g + 20 and b > g + 20 and r > 180 and b > 180 for r, g, b in pixels)
    assert not list(tmp_path.glob("*.highlight.svg"))


def test_highlighter_warns_when_the_format_cannot_carry_it(tmp_path: Path) -> None:
    """A raster format other than PDF or PNG draws without the highlighter and says so."""

    trace = _traced()
    try:
        with pytest.warns(TorchLensWarning) as record:
            trace.draw(
                vis_save_only=True,
                file_format="jpg",
                path=str(tmp_path / "cone"),
                vis_show_cone=True,
            )
    finally:
        trace.cleanup()
    codes = [getattr(entry.message, "fields", {}).get("code") for entry in record]
    assert codes.count("viz_highlighter_not_drawn") == 1
    assert (tmp_path / "cone.jpg").exists()


def test_as_node_hook_is_purple(tmp_path: Path) -> None:
    """The as_node hook diamond is restyled in the edit purple and tagged as the edit."""

    trace = _traced()
    try:
        dot = trace.draw(
            vis_save_only=True,
            file_format="dot",
            path=str(tmp_path / "as_node"),
            vis_intervention_mode="as_node",
        )
    finally:
        trace.cleanup()
    hook_line = next(
        line
        for line in dot.splitlines()
        if line.lstrip().startswith("intervention_hook_") and "->" not in line
    )
    assert "shape=diamond" in hook_line
    assert PURPLE in hook_line and "#F3E5F5" in hook_line
    assert "tl_edited" in hook_line


class _SharedLinearTwice(nn.Module):
    """One linear layer called twice, so its layer has two passes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared projection twice around a ReLU."""

        for _call in range(2):
            x = torch.relu(self.lin(x))
        return x


def test_backward_edit_marks_its_own_pass(tmp_path: Path) -> None:
    """A backward edit staged on pass 2 of a two-pass layer names pass 2, and only it.

    A GradFn site resolves to the one pass it differentiates (``GradFn.op`` is
    the multi-pass Layer); the staged edit row must key by that same pass, or
    the ringed node would read ``edit`` instead of the edit's name.
    """

    def named_backward(grad: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        return grad

    torch.manual_seed(0)
    trace = tl.trace(
        _SharedLinearTwice(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    try:
        linear = next(layer for layer in trace.layer_logs.values() if layer.layer_type == "linear")
        assert linear.num_passes == 2
        second = trace[f"{linear.layer_label}:2"]
        # GradFn records exist once a backward pass has run.
        trace[trace.output_layers[0]].out.sum().backward()
        grad_fn = next(
            gf for gf in trace.autograd.grad_fns if gf.grad_fn_object_id == second.grad_fn_object_id
        )
        trace.attach_hooks(
            grad_fn_label(grad_fn.label),
            tl.hook_grad(named_backward),
            confirm_mutation=True,
        )
        sites = intervention_sites_for_log(trace)
        assert [site.label for site in sites] == [second.label]
        marks = edit_marks_by_site(trace, sites)
        assert [mark.short for mark in marks[second.label]] == ["hook_grad(named_backward)"]
        dot = trace.draw(
            vis_save_only=True,
            file_format="dot",
            vis_mode="unrolled",
            path=str(tmp_path / "shared"),
        )
    finally:
        trace.cleanup()
    edited = [line for line in dot.splitlines() if "tl_edited" in line]
    assert len(edited) == 1
    assert "hook_grad(named_backward)" in edited[0]


def test_unrolled_forward_edit_marks_only_its_pass(tmp_path: Path) -> None:
    """An edit on pass 2 rings pass 2 alone, tags only what pass 2 feeds, and keeps its pass.

    The node slot hands an unrolled node its aggregate Layer; the ring, edit row
    and downstream class must follow the rendered pass, not the layer.
    """

    torch.manual_seed(0)
    trace = tl.trace(
        _SharedLinearTwice(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        fork = trace.fork("edited")
        layers = {layer.layer_type: layer.layer_label for layer in fork.layer_logs.values()}
        linear, relu = layers["linear"], layers["relu"]
        fork.do(tl.label(f"{linear}:2"), tl.zero_ablate())
        dot = fork.draw(
            vis_save_only=True,
            file_format="dot",
            vis_mode="unrolled",
            vis_show_cone=True,
            path=str(tmp_path / "shared_fwd"),
        )
        rolled = fork.draw(
            vis_save_only=True,
            file_format="dot",
            vis_mode="rolled",
            path=str(tmp_path / "shared_fwd_rolled"),
        )
    finally:
        trace.cleanup()
    # Rolled, the one node stands for both passes; the header keeps the pass.
    assert f"&#160;on {linear}:2" in rolled
    nodes = {
        line.split(" [", 1)[0].strip(): line
        for line in dot.splitlines()
        if " [label=" in line and "->" not in line
    }
    edited = sorted(name for name, line in nodes.items() if "tl_edited" in line)
    assert edited == [f"{linear}pass2"]
    assert "zero_ablate" not in nodes[f"{linear}pass1"]
    assert "tl_downstream" in nodes[f"{relu}pass2"]
    assert "tl_downstream" not in nodes[f"{relu}pass1"]
    assert "tl_downstream" not in nodes[f"{linear}pass1"]


_PATCHED_IMPORT_PROBE = """
import sys
from torchlens.visualization import node_spec
if "torchlens.visualization._render_common" in sys.modules:
    print("PRELOADED")
    raise SystemExit(0)
real = node_spec.intervention_site_and_cone_labels
node_spec.intervention_site_and_cone_labels = lambda trace, *, show_cone: (set(), {"x"})
try:
    node_spec.make_intervention_node_spec_fn(
        object(), show_cone=True, graph_overrides=None, user_node_spec_fn=None
    )
finally:
    node_spec.intervention_site_and_cone_labels = real
from torchlens.visualization import _render_common
print("BOUND_REAL" if _render_common.intervention_site_and_cone_labels is real else "BOUND_PATCH")
"""


def test_building_the_edit_callback_imports_no_render_module() -> None:
    """Building the intervention callback never first-imports the render modules.

    The render modules bind ``node_spec``'s helpers by value when imported, so a
    first import while a helper is patched (``test_node_spec_api`` monkeypatches
    ``intervention_site_and_cone_labels``) kept the patch for the rest of the
    process: later ``as_node`` renders lost their hook node. A fresh interpreter
    makes the first import deterministic.
    """

    result = subprocess.run(
        [sys.executable, "-c", _PATCHED_IMPORT_PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    outcome = result.stdout.strip().splitlines()[-1]
    if outcome == "PRELOADED":
        pytest.skip("importing node_spec already loads the render modules")
    assert outcome == "BOUND_REAL"
