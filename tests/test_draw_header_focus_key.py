"""The draw header, the ``module=`` focus scope, the key, and the format doors.

- Both draw doors (``Trace.draw`` and ``Bundle.draw``) take the format from the
  path extension; a ``file_format`` that disagrees refuses
  (``draw_format_mismatch``), so the extension never loses silently.
- The header's count names what it counts, and the count and the byte total
  cover the same nodes: the drawn nodes that show a tensor size (a focus's
  input and output stubs show none and are not counted).
- Parameter counts carry thousands separators; every size uses the binary
  unit its 1024 divisor implies (KiB, MiB).
- A focused draw opens with a scope line (model class, then each enclosing
  module, then the focus with its class), labels which line covers the drawn
  view and which the whole model, shows the key and a reading key by default,
  and draws no frame for the focus's ancestors.
- Key entries match drawn marks: a frame for an expanded module, a box for a
  module drawn as one op, a 3-D box for a collapsed module, a double outline
  for hidden buffers. Frozen weights take the Okabe-Ito dark blue with white
  label ink.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._header_block import KEY_TABLE_ID, drawn_module_marks, key_swatches
from torchlens._visualization._label_format import binary_size, format_memory
from torchlens._visualization.themes import (
    OKABE_ITO_SEMANTIC_PALETTE,
    RELEASED_SEMANTIC_PALETTE,
    resolve_theme,
)
from torchlens.errors import ArgumentConflictError

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot not installed")

_FOCUS = "transformer.h.1.mlp"
_COUNT = re.compile(r"(\d[\d,]*) (ops|layers|nodes) \(([\d.]+ (?:B|KiB|MiB|GiB))\)")
_SIZE_LINE = re.compile(r"\), [\d.]+ (?:B|KiB|MiB|GiB)<")
_NODE = re.compile(r"^\t(\S+) \[label=<")
_CLUSTER = re.compile(r'subgraph "?cluster_([^"\s{]+)')


class _MLP(nn.Module):
    """A GPT-2 style feed-forward block: expand, tanh-GELU, project, dropout."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class _Block(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = _MLP(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln_2(x))


class _Body(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.h = nn.ModuleList([_Block(width), _Block(width)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.h:
            x = block(x)
        return x


class _Toy(nn.Module):
    """Model, ``transformer``, ``h.N``, ``mlp``: the nesting of a GPT-2 focus."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.transformer = _Body(width)
        self.lm_head = nn.Linear(width, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(x))


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    torch.manual_seed(0)
    captured = tl.trace(_Toy().eval(), torch.randn(1, 5, 8))
    try:
        yield captured
    finally:
        captured.cleanup()


def _dot(trace: tl.Trace, folder: Path, name: str, **kwargs: object) -> str:
    return trace.draw(str(folder / f"{name}.svg"), **kwargs).dot


def _header(dot: str) -> str:
    return dot[dot.index("label=<") : dot.index("labeljust")]


def _sized_nodes(dot: str) -> list[str]:
    """Names of the drawn nodes whose label shows a tensor size."""

    names = []
    for line in dot.splitlines():
        match = _NODE.match(line)
        if match is not None and _SIZE_LINE.search(line):
            names.append(match.group(1))
    return names


# -- format doors (AGT-2531) -------------------------------------------------


@pytest.mark.parametrize(("ext", "magic"), [("svg", b"<"), ("png", b"\x89PNG"), ("pdf", b"%PDF")])
def test_trace_draw_writes_the_extension_format(
    trace: tl.Trace, tmp_path: Path, ext: str, magic: bytes
) -> None:
    drawing = trace.draw(str(tmp_path / f"g.{ext}"), module=_FOCUS)
    assert drawing.file_format == ext and drawing.path == tmp_path / f"g.{ext}"
    head = (tmp_path / f"g.{ext}").read_bytes().lstrip()
    assert head.startswith(b"<?xml") or head.startswith(magic)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [f"g.{ext}"]


@pytest.mark.parametrize(("ext", "magic"), [("svg", b"<"), ("png", b"\x89PNG"), ("pdf", b"%PDF")])
def test_bundle_draw_writes_the_extension_format(tmp_path: Path, ext: str, magic: bytes) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 5, 8)
    bundle = tl.Bundle({"a": tl.trace(_MLP(), x), "b": tl.trace(_MLP(), x)}, baseline="a")
    drawing = bundle.draw(str(tmp_path / f"b.{ext}"), open_viewer=False)
    assert drawing.file_format == ext and drawing.path == tmp_path / f"b.{ext}"
    head = (tmp_path / f"b.{ext}").read_bytes().lstrip()
    assert head.startswith(b"<?xml") or head.startswith(magic)


@pytest.mark.parametrize("door", ["trace", "bundle"])
def test_a_disagreeing_file_format_refuses_on_both_doors(
    trace: tl.Trace, tmp_path: Path, door: str
) -> None:
    x = torch.randn(1, 5, 8)
    pair = {"a": tl.trace(_MLP(), x), "b": tl.trace(_MLP(), x)}
    drawer = trace if door == "trace" else tl.Bundle(pair, baseline="a")
    with pytest.raises(ArgumentConflictError) as excinfo:
        drawer.draw(str(tmp_path / "g.svg"), file_format="png")
    assert excinfo.value.fields["code"] == "draw_format_mismatch"
    assert list(tmp_path.iterdir()) == []


# -- header count and units (AGT-2536) ----------------------------------------


def test_binary_units_match_the_1024_divisor() -> None:
    assert binary_size(0) == "0 B"
    assert binary_size(1023) == "1023 B"
    assert binary_size(1024) == "1.0 KiB"
    assert binary_size(1.5 * 1024**2) == "1.5 MiB"
    assert binary_size(3 * 1024**3) == "3.0 GiB"
    assert format_memory(61440) == "60.0 KiB"
    assert format_memory(True) == "True"
    with pytest.raises(ValueError):
        binary_size(-1)


def test_whole_model_count_and_bytes_cover_the_same_nodes(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "whole")
    count, noun, size = _COUNT.search(_header(dot)).groups()  # type: ignore[union-attr]
    sized = _sized_nodes(dot)
    assert noun == "ops"
    assert int(count.replace(",", "")) == len(sized)
    labels = [re.sub(r"pass(\d+)$", r":\1", name) for name in sized]
    assert size == binary_size(sum(int(trace.ops[label].activation_memory) for label in labels))
    assert f"{trace.num_params:,} params" in dot
    assert not re.search(r"\d (?:KB|MB|GB)\b", dot)


@pytest.mark.parametrize("rolled", [False, True])
def test_focus_count_skips_the_boundary_stubs(
    trace: tl.Trace, tmp_path: Path, rolled: bool
) -> None:
    dot = _dot(trace, tmp_path, "focus", module=_FOCUS, rolled=rolled)
    header = _header(dot)
    count, noun, _size = _COUNT.search(header).groups()  # type: ignore[union-attr]
    sized = _sized_nodes(dot)
    assert noun == ("layers" if rolled else "ops")
    assert int(count) == len(sized)
    assert not any(name.startswith("__module_focus") for name in sized)
    assert "__module_focus_input" in dot and "__module_focus_output" in dot
    if not rolled:
        labels = [re.sub(r"pass(\d+)$", r":\1", name) for name in sized]
        expected = sum(int(trace.ops[label].activation_memory) for label in labels)
        assert f"drawn: {count} ops ({binary_size(expected)})" in header


# -- focus scope line and reading key (AGT-2538) ------------------------------


def test_focus_header_names_its_scope_and_its_lines(trace: tl.Trace, tmp_path: Path) -> None:
    header = _header(_dot(trace, tmp_path, "scope", module=_FOCUS))
    scope = "<B>_Toy</B> &gt; transformer &gt; h.1 &gt; <B>mlp</B> (_MLP)"
    assert scope in header
    assert "drawn: " in header
    assert f"whole model: {trace.num_params:,} params" in header


def test_focus_key_is_on_by_default_and_off_on_request(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "keyed", module=_FOCUS)
    assert KEY_TABLE_ID in dot
    assert "@name: the module at that address" in dot
    assert re.search(r"\w+_\d+_\d+: the \d+(st|nd|rd|th) \w+ call, op \d+ of the whole", dot)
    assert "(shape), size: the op's output tensor and its memory" in dot
    assert "params: the weights the op used (filled as the key shows)" in dot
    assert "module run as one op" in dot  # the dropout and GELU are drawn as boxes
    bare = _dot(trace, tmp_path, "bare", module=_FOCUS, show_legend=False)
    assert KEY_TABLE_ID not in bare and "@name: the module" not in bare
    assert "&gt; <B>mlp</B> (_MLP)" in bare  # the scope line is the header, not the key


def test_whole_model_draw_keeps_its_key_opt_in(trace: tl.Trace, tmp_path: Path) -> None:
    dot = _dot(trace, tmp_path, "plain")
    assert KEY_TABLE_ID not in dot and "@name: the module" not in dot
    assert "drawn: " not in _header(dot) and "&gt;" not in _header(dot)


# -- ancestor frames (AGT-2537) -----------------------------------------------


@pytest.mark.parametrize("rolled", [False, True])
def test_focus_draws_no_ancestor_frames(trace: tl.Trace, tmp_path: Path, rolled: bool) -> None:
    dot = _dot(trace, tmp_path, "frames", module=_FOCUS, rolled=rolled)
    frames = _CLUSTER.findall(dot)
    assert frames, dot
    assert all(frame.startswith(_FOCUS) for frame in frames), frames
    widths = re.findall(r"penwidth=([\d.]+)", dot[dot.index("subgraph") :])
    assert max(float(width) for width in widths) == 5.0  # the focus frame is the outermost


def test_whole_model_draw_keeps_every_frame(trace: tl.Trace, tmp_path: Path) -> None:
    frames = _CLUSTER.findall(_dot(trace, tmp_path, "all_frames"))
    assert any(frame.startswith("transformer_pass") for frame in frames), frames
    assert any(frame.startswith(_FOCUS) for frame in frames), frames


# -- key entries and palette (rulings-gap-4) ----------------------------------


@pytest.mark.parametrize(
    ("attrs", "regions", "marks"),
    [
        ({"shape": "box", "style": "filled,solid"}, False, {"box"}),
        ({"shape": "box3d", "style": "filled"}, False, {"box3d"}),
        ({"shape": "box", "style": "rounded,dashed,filled"}, False, set()),
        ({"shape": "oval", "style": "filled", "peripheries": "2"}, False, {"double"}),
        ({"shape": "oval", "style": "filled"}, True, {"frame"}),
        ({"shape": "box", "style": "invis"}, False, set()),
    ],
)
def test_module_marks_follow_the_drawn_style(
    attrs: dict[str, str], regions: bool, marks: set[str]
) -> None:
    assert drawn_module_marks([attrs], regions) == frozenset(marks)


def test_each_module_mark_gets_its_own_key_entry() -> None:
    theme = resolve_theme("torchlens")
    texts = {
        mark: [
            swatch.text
            for swatch in key_swatches(
                theme, frozenset(), has_module=False, module_marks=frozenset({mark})
            )
        ]
        for mark in ("frame", "box", "box3d", "double")
    }
    assert texts == {
        "frame": ["module"],
        "box": ["module run as one op"],
        "box3d": ["collapsed module"],
        "double": ["hidden buffers"],
    }
    shapes = [
        s.shape
        for s in key_swatches(
            theme, frozenset(), has_module=False, module_marks=frozenset({"box3d", "double"})
        )
    ]
    assert shapes == ["box3d", "double"]
    assert [s.text for s in key_swatches(theme, frozenset(), has_module=True)] == ["module"]


def test_frozen_weights_take_dark_grey_with_black_ink(tmp_path: Path) -> None:
    # No skin draws weights in blue: trainable is light grey, frozen darker grey, and black
    # ink reads on both (the default skin keeps the released #D9D9D9 / #B0B0B0).
    assert RELEASED_SEMANTIC_PALETTE["params_frozen"] == "#B0B0B0"
    assert RELEASED_SEMANTIC_PALETTE["params_trainable"] == "#D9D9D9"
    assert OKABE_ITO_SEMANTIC_PALETTE["params_frozen"] == "#808080"
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    model[0].requires_grad_(False)
    captured = tl.trace(model, torch.randn(1, 4))
    try:
        dot = captured.draw(str(tmp_path / "frozen.svg")).dot
    finally:
        captured.cleanup()
    frozen = [line for line in dot.splitlines() if "#B0B0B0" in line and "[label=<" in line]
    assert frozen, dot
    assert not any("#FFFFFF" in line.upper() or "fontcolor=white" in line for line in frozen)
