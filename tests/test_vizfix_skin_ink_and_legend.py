"""Skins own all of a render's ink, and the legend lists only what is drawn.

Defects pinned here (UI sprint visual track, renderer findings):

- the dark skin left edges, edge labels and module-box borders black and
  module boxes white on its near-black page, and kept its near-white node
  text on light semantic fills (about 1.2:1 on the input fill);
- the role legend listed every category whatever the graph held;
- the legend had no plain-op row, and buffers were filled like plain ops,
  so the buffer swatch and a plain op looked the same in every skin;
- the header's capture advisory printed the model file's absolute path.

White-page skins must keep their historical edge and module-box DOT.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._header_block import KEY_TABLE_ID, key_swatches
from torchlens.visualization._legend import LEGEND_NODE_NAME, drawn_legend_roles
from torchlens.visualization._render_utils import strip_path_directories
from torchlens.visualization.lenses.audit import palette_distinguishability
from torchlens.visualization.themes import THEME_PRESETS, _contrast_ratio

_ATTR_RE = r"(?:^|[\s\[,])({name})=(\"[^\"]*\"|[^\s\],]+)"
_WHITE_PAGE_SKINS = ("torchlens", "paper", "colorblind", "high_contrast")


class _InnerRes(nn.Module):
    """A residual block: one Linear plus an add, so its module box holds two ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + x


class _NestedRes(nn.Module):
    """Two nested module boxes around the residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(_InnerRes())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Stateful(nn.Module):
    """A registered buffer read by a plain op."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * self.scale)


class _Branch(nn.Module):
    """A tensor-driven if; capture records a scalar-escape advisory at the ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        return x - 1


def _draw(model: nn.Module, x: torch.Tensor, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Trace ``model`` and return the DOT source of an SVG render."""

    torch.manual_seed(0)
    trace = tl.trace(model, x)
    return str(
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / name),
            **kwargs,
        )
    )


def _attr_values(line: str, name: str) -> list[str]:
    """Return every value of attribute ``name`` on one DOT line, unquoted."""

    return [
        value.strip('"') for _, value in re.findall(_ATTR_RE.format(name=re.escape(name)), line)
    ]


def _edge_lines(dot: str) -> list[str]:
    """Return the drawn edge statements (the legend's invisible tie excluded)."""

    return [line for line in dot.splitlines() if " -> " in line and "style=invis" not in line]


def _node_lines(dot: str) -> list[str]:
    """Return node statements that carry a fill."""

    return [
        line
        for line in dot.splitlines()
        if " -> " not in line and "fillcolor=" in line and LEGEND_NODE_NAME not in line
    ]


def _legend_rows(dot: str) -> list[str]:
    """Return the key's entries, in order (the key rides in the header label)."""

    label = next(line for line in dot.splitlines() if line.strip().startswith("graph ["))
    if f'ID="{KEY_TABLE_ID}"' not in label:
        return []
    key = label[label.index(f'ID="{KEY_TABLE_ID}"') :]
    return re.findall(r"<FONT [^>]*>([^<]+)</FONT></TD>", key)


def _fills(skin: str) -> tuple[str, str]:
    """Return the plain-op fill and the key's buffer fill under ``skin``."""

    from torchlens.visualization.node_spec import NodeSpec
    from torchlens.visualization.themes import apply_theme_to_spec

    theme = THEME_PRESETS[skin]
    plain = str(apply_theme_to_spec(NodeSpec(["x"], fillcolor="white"), theme).fillcolor)
    buffer = next(s.fill for s in key_swatches(theme, frozenset({"buffer"}), has_module=False))
    return plain, buffer


def test_dark_skin_edges_and_edge_labels_follow_the_skin(tmp_path: Path) -> None:
    """No dark edge or edge label keeps the renderer's black ink."""

    dark = THEME_PRESETS["dark"]
    dot = _draw(_NestedRes(), torch.randn(1, 3), tmp_path, "dark_edges", vis_theme="dark")
    edges = _edge_lines(dot)
    assert edges
    for line in edges:
        assert "black" not in _attr_values(line, "color"), line
        assert "black" not in _attr_values(line, "fontcolor"), line
    assert any(dark.edge["color"] in _attr_values(line, "color") for line in edges)


def test_dark_skin_module_boxes_follow_the_skin(tmp_path: Path) -> None:
    """Dark module boxes take the page fill and the skin's line and label ink."""

    dark = THEME_PRESETS["dark"]
    dot = _draw(_NestedRes(), torch.randn(1, 3), tmp_path, "dark_boxes", vis_theme="dark")
    assert "subgraph cluster_" in dot
    assert "fillcolor=white" not in dot
    page = dark.graph["bgcolor"]
    assert f'fillcolor="{page}"' in dot
    assert f'color="{dark.edge["color"]}"' in dot.split("subgraph cluster_", 1)[1]


@pytest.mark.parametrize("skin", _WHITE_PAGE_SKINS)
def test_white_page_skins_keep_their_edge_and_box_ink(skin: str, tmp_path: Path) -> None:
    """The fix moves only skins that set edge ink: white-page DOT is unchanged."""

    dot = _draw(_NestedRes(), torch.randn(1, 3), tmp_path, f"{skin}_ink", vis_theme=skin)
    edges = _edge_lines(dot)
    assert edges and all(_attr_values(line, "color") == ["black"] for line in edges)
    assert "fillcolor=white" in dot
    assert THEME_PRESETS["dark"].edge["color"] not in dot


def test_dark_skin_node_text_is_readable_on_every_fill(tmp_path: Path) -> None:
    """Every filled dark node meets 4.5:1 text contrast (was 1.2:1 on the input fill)."""

    dot = _draw(_Stateful(), torch.randn(1, 3), tmp_path, "dark_text", vis_theme="dark")
    checked = 0
    for line in _node_lines(dot):
        fills, fonts = _attr_values(line, "fillcolor"), _attr_values(line, "fontcolor")
        if not fills or not fonts:
            continue
        for segment in fills[0].split(":"):
            ratio = _contrast_ratio(fonts[0], segment)
            if ratio is not None:
                checked += 1
                assert ratio >= 4.5, line
    assert checked >= 3


def test_legend_lists_only_the_roles_drawn(tmp_path: Path) -> None:
    """A Linear+ReLU graph keys input, output, both weight grays and the module box."""

    dot = _draw(
        nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
        torch.randn(1, 3),
        tmp_path,
        "drawn_roles",
        show_legend=True,
    )
    assert _legend_rows(dot) == [
        "input",
        "output",
        "trainable weights",
        "frozen weights",
        "module",
    ]


def test_legend_buffer_row_appears_with_a_drawn_buffer(tmp_path: Path) -> None:
    """A drawn buffer brings its entry, drawn as a cylinder."""

    dot = _draw(
        _Stateful(),
        torch.randn(1, 3),
        tmp_path,
        "buffer_row",
        show_legend=True,
        show_buffer_layers="always",
    )
    rows = _legend_rows(dot)
    assert "buffer" in rows
    assert "trainable weights" not in rows and "true/false test" not in rows
    assert "key-cylinder-" in dot


def test_legend_intervention_row_only_on_an_intervened_render(tmp_path: Path) -> None:
    """The intervention row is gated on a drawn site or cone outline."""

    torch.manual_seed(0)
    trace = tl.trace(
        _NestedRes(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    plain = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "plain"),
        show_legend=True,
    )
    assert not any(row.startswith("intervention") for row in _legend_rows(str(plain)))
    fork = trace.fork("edited")
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    edited = fork.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "edited"),
        show_legend=True,
    )
    rows = _legend_rows(str(edited))
    assert "edited" in rows
    # The downstream highlighter is off by default, so its key row is absent.
    assert "changed by the edit" not in rows


@pytest.mark.parametrize("skin", sorted(THEME_PRESETS))
def test_plain_op_row_and_buffer_fill_are_distinguishable(skin: str) -> None:
    """Every skin's buffer fill differs from its plain-op fill under CVD."""

    plain, buffer = _fills(skin)
    findings = palette_distinguishability({"plain op": plain, "buffer": buffer})
    assert len(findings) == 4
    assert all(finding.passed for finding in findings), [f.detail for f in findings]


@pytest.mark.parametrize("skin", sorted(THEME_PRESETS))
def test_buffer_nodes_are_painted_with_the_legend_buffer_fill(skin: str, tmp_path: Path) -> None:
    """The drawn buffer's fill is the buffer swatch, never the plain-op fill."""

    plain, buffer = _fills(skin)
    dot = _draw(
        _Stateful(),
        torch.randn(1, 3),
        tmp_path,
        f"{skin}_buffer",
        vis_theme=skin,
        show_buffer_layers="always",
    )
    cylinders = [line for line in _node_lines(dot) if "shape=cylinder" in line]
    assert cylinders
    for line in cylinders:
        assert _attr_values(line, "fillcolor") == [buffer]
        assert _attr_values(line, "fillcolor") != [plain]


def test_drawn_roles_ignore_layout_helpers_and_collapsed_boxes() -> None:
    """Invisible helpers and default-fill collapsed boxes are not plain ops."""

    theme = THEME_PRESETS["torchlens"]
    roles = drawn_legend_roles(
        [
            {"shape": "point", "fillcolor": "white"},
            {"shape": "oval", "style": "invis", "fillcolor": "white"},
            {"shape": "box3d", "fillcolor": "white"},
        ],
        theme,
    )
    assert roles == frozenset()


def test_key_lists_only_the_roles_drawn() -> None:
    """A picture of plain ops only has nothing to key: plain ops get no entry."""

    theme = THEME_PRESETS["torchlens"]
    plain_only = drawn_legend_roles([{"shape": "oval", "fillcolor": "white"}], theme)
    assert key_swatches(theme, plain_only, has_module=False) == ()
    assert key_swatches(theme, frozenset(), has_module=False) == ()


def test_strip_path_directories_keeps_file_and_line() -> None:
    """Absolute paths lose their directories; other slashes are left alone."""

    assert (
        strip_path_directories("(first at /srv/project/src/models.py:224)")
        == "(first at models.py:224)"
    )
    assert strip_path_directories("first at C:\\Users\\me\\m.py:3") == "first at m.py:3"
    assert strip_path_directories("and/or ratio 1/2") == "and/or ratio 1/2"


def test_header_advisory_names_no_local_directory(tmp_path: Path) -> None:
    """The capture advisory in the header cites the file name and line only."""

    with warnings.catch_warnings(record=True):
        # The scalar escape at the ``if`` is warned once at capture; it is
        # the advisory under test, not a defect.
        warnings.simplefilter("always")
        trace = tl.trace(_Branch(), torch.ones(1, 3))
    dot = str(
        trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=str(tmp_path / "adv"))
    )
    assert "capture advisory" in dot
    assert str(Path(__file__).resolve().parent) not in dot
    assert f"{Path(__file__).name}:" in dot


def test_as_node_intervention_keeps_the_site_inside_its_module_boxes(tmp_path: Path) -> None:
    """The site-to-hook edge sits in the same cluster as the hook-to-child edge."""

    torch.manual_seed(0)
    trace = tl.trace(
        _NestedRes(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    fork = trace.fork("edited")
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    dot = str(
        fork.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "as_node"),
            vis_intervention_mode="as_node",
        )
    )
    into_hook = [line for line in _edge_lines(dot) if "-> intervention_hook_" in line]
    out_of_hook = [
        line for line in _edge_lines(dot) if line.lstrip().startswith("intervention_hook_")
    ]
    assert len(into_hook) == 1 and len(out_of_hook) == 1

    def depth(line: str) -> int:
        return len(line) - len(line.lstrip("\t"))

    assert depth(into_hook[0]) == depth(out_of_hook[0]) > 1


@pytest.mark.parametrize("skin", ["dark", "torchlens"])
def test_rank_path_page_follows_the_skin(skin: str, tmp_path: Path) -> None:
    """The rank engine paints the dark page; white pages stay implicit, as before."""

    dot = _draw(
        nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
        torch.randn(1, 3),
        tmp_path,
        f"rank_{skin}",
        vis_theme=skin,
        vis_node_placement="rank",
    )
    page = THEME_PRESETS[skin].graph["bgcolor"]
    if skin == "dark":
        assert f'bgcolor="{page}"' in dot
    else:
        assert "bgcolor" not in dot
