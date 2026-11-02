"""``Bundle.draw`` and the Bundle split: real methods, one draw door, private delta reads.

Also pinned here (JMT drawing rulings, Sun 1 Nov 2026):

- ``Bundle.draw(path, ...)`` returns a :class:`tl.viz.Drawing` like ``Trace.draw``;
  the path's extension picks the format, ``file_format=`` applies to a path with
  no extension, and a disagreement refuses (``draw_format_mismatch``);
- a bare ``b.draw()`` displays inline in a Jupyter kernel and writes nothing;
  elsewhere it writes a collision-safe ``bundle_modelgraph`` file;
- any keyword ``Bundle.draw`` does not take refuses typed and names the
  supported list (``bundle_draw_keyword_unsupported``);
- ``Bundle.show()`` with no kind is the merged graph, ``kind="strip"`` each
  member's own graph, ``"repr"`` / ``"html"`` stay, any other kind refuses;
- ``show_bundle_graph`` is gone: importing it fails and the root name teaches
  ``Bundle.draw``.
"""

from __future__ import annotations

import inspect
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.viz._drawing as drawing_module
from torchlens.errors import (
    ArgumentConflictError,
    ArgumentTypeError,
    FacadeTeachingError,
    InvalidArgumentError,
)

_REAL_METHODS = (
    "store_comparison",
    "stored_comparison",
    "stored_comparison_names",
    "upstream",
    "downstream",
    "token_pairs",
    "token_divergence",
    "add_alternative",
    "draw",
)


class _TinyRelu(torch.nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.relu = torch.nn.ReLU()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x)) + self.offset


def _pair() -> tl.Bundle:
    """Two unrelated members: comparisons refuse members joined by an ordering relation."""
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    clean = tl.trace(_TinyRelu(), x)
    changed = tl.trace(_TinyRelu(offset=0.5), x)
    return tl.Bundle({"clean": clean, "changed": changed}, baseline="clean")


def test_bundle_draw_returns_a_drawing_of_the_union_graph(tmp_path: Path) -> None:
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    bundle = _pair()
    drawing = bundle.draw(
        str(tmp_path / "union.svg"),
        node_overrides={"linear_1_1": {"fillcolor": "#ffeeaa"}},
        open_viewer=False,
    )
    assert isinstance(drawing, tl.viz.Drawing)
    assert drawing.path == tmp_path / "union.svg" and drawing.file_format == "svg"
    dot = drawing.dot
    assert "digraph" in dot
    assert "#ffeeaa" in dot
    assert (tmp_path / "union.svg").exists()


def test_bundle_draw_diff_folds_show_diff(tmp_path: Path) -> None:
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    bundle = _pair()
    dot = bundle.draw(
        graph="diff", members=("clean", "changed"), path=str(tmp_path / "d"), open_viewer=False
    ).dot
    assert dot.startswith("// TorchLens bundle diff")
    assert "clean vs changed" in dot
    assert (tmp_path / "d.svg").exists()
    with pytest.raises(FacadeTeachingError) as excinfo:
        _ = bundle.show_diff
    assert "draw(graph='diff'" in str(excinfo.value)


def test_former_helper_table_methods_are_real_methods() -> None:
    for name in _REAL_METHODS:
        assert inspect.isfunction(inspect.getattr_static(tl.Bundle, name)), name
    assert "__dir__" not in vars(tl.Bundle)
    bundle = _pair()
    name = bundle.store_comparison("relative_l2")
    assert bundle.stored_comparison_names() == (name,)
    assert bundle.stored_comparison(name) == bundle._delta_map("relative_l2")
    bundle.relate("forked_from", "changed", "clean", at_step=0)
    assert bundle.upstream("changed") == ("clean",)
    assert bundle.downstream("clean") == ("changed",)


@pytest.mark.parametrize("name", ["delta_map", "norm_delta", "output_delta", "aligned_pairs"])
def test_delta_family_is_private_behind_compare(name: str) -> None:
    bundle = _pair()
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(bundle, name)
    assert excinfo.value.fields["attribute"] == name
    assert callable(getattr(bundle, f"_{name}"))
    rows = bundle.compare("clean", "changed", metric="relative_l2")
    assert rows and set(dict(rows)) <= set(bundle._delta_map("relative_l2"))


def test_bundle_draw_refuses_an_unknown_graph_typed() -> None:
    """``graph=`` outside forward/backward/combined/diff refuses before any rendering."""
    bundle = _pair()
    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.draw(graph="sideways", open_viewer=False)
    assert excinfo.value.fields["code"] == "bundle_draw_graph_invalid"


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class _Large(_Small):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.relu(self.fc(x)))


@pytest.fixture(scope="module")
def bundle() -> Iterator[tl.Bundle]:
    """Two members that differ by one ReLU."""

    pytest.importorskip("graphviz")
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    torch.manual_seed(0)
    x = torch.randn(1, 3)
    small = tl.trace(_Small(), x)
    large = tl.trace(_Large(), x)
    try:
        yield tl.Bundle({"small": small, "large": large}, baseline="small")
    finally:
        small.cleanup()
        large.cleanup()


def test_draw_format_follows_the_path_extension(bundle: tl.Bundle, tmp_path: Path) -> None:
    drawing = bundle.draw(str(tmp_path / "diff.svg"))
    assert "TorchLens bundle graph" in drawing.dot
    assert "[large]" in drawing.dot  # the ReLU only one member runs keeps its member list
    assert (tmp_path / "diff.svg").read_text().lstrip().startswith(("<?xml", "<svg"))

    png = bundle.draw(str(tmp_path / "diff.png"))
    assert png.file_format == "png" and png.path == tmp_path / "diff.png"
    assert (tmp_path / "diff.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / "diff.pdf").exists()


def test_draw_file_format_applies_to_a_path_without_extension(
    bundle: tl.Bundle, tmp_path: Path
) -> None:
    bundle.draw(str(tmp_path / "stem"), file_format="svg")
    assert (tmp_path / "stem.svg").is_file()
    bundle.draw(str(tmp_path / "same.svg"), file_format="svg")
    assert (tmp_path / "same.svg").is_file()
    with pytest.raises(ArgumentConflictError) as excinfo:
        bundle.draw(str(tmp_path / "clash.svg"), file_format="png")
    assert excinfo.value.fields["code"] == "draw_format_mismatch"
    assert not (tmp_path / "clash.svg").exists() and not (tmp_path / "clash.png").exists()


def test_bare_draw_outside_a_kernel_never_overwrites(
    bundle: tl.Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(drawing_module, "_in_notebook", lambda: False)
    monkeypatch.chdir(tmp_path)
    first = bundle.draw(file_format="svg")
    bundle.draw(file_format="svg")
    assert first.path == Path("bundle_modelgraph.svg")
    assert (tmp_path / "bundle_modelgraph.svg").is_file()
    assert (tmp_path / "bundle_modelgraph-2.svg").is_file()
    assert "bundle_modelgraph-2.svg" in capsys.readouterr().out


def test_bare_draw_in_a_kernel_displays_inline_and_writes_nothing(
    bundle: tl.Bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(drawing_module, "_in_notebook", lambda: True)
    monkeypatch.chdir(tmp_path)
    drawing = bundle.draw()
    assert drawing.path is None
    assert drawing.file_format == "svg"
    assert drawing.data is not None and b"<svg" in drawing.data
    assert drawing._repr_svg_() is not None
    assert list(tmp_path.iterdir()) == []
    assert "written to" not in capsys.readouterr().out
    written = bundle.draw(str(tmp_path / "kept.svg"))  # a path always writes
    assert written.path == tmp_path / "kept.svg" and written.path.is_file()


def test_draw_members_nodes(bundle: tl.Bundle, tmp_path: Path) -> None:
    source = bundle.draw(str(tmp_path / "members.svg"), nodes="members").dot
    assert (tmp_path / "members.svg").is_file()
    assert "TorchLens bundle graph (members)" in source
    assert "member_0" in source and "member_1" in source
    assert "fwd_" not in source  # no op nodes in the members picture


def test_draw_graph_rolled_and_direction(bundle: tl.Bundle, tmp_path: Path) -> None:
    combined = bundle.draw(str(tmp_path / "combined.svg"), graph="combined").dot
    assert "cluster_backward_small" in combined and "cluster_backward_large" in combined
    assert "fwd_" in combined or "[large]" in combined  # the forward ops are drawn too
    rolled = bundle.draw(str(tmp_path / "rolled.svg"), rolled=True, direction="leftright").dot
    assert "rolled" in rolled
    assert "rankdir=LR" in rolled
    with pytest.raises(InvalidArgumentError, match="graph"):
        bundle.draw(str(tmp_path / "bad.svg"), graph="both")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "keyword", ["depth", "collapse", "module", "color_by", "vis_mode", "vis_save_only", "signal"]
)
def test_draw_refuses_keywords_it_does_not_take(
    bundle: tl.Bundle, tmp_path: Path, keyword: str
) -> None:
    with pytest.raises(ArgumentTypeError) as excinfo:
        bundle.draw(str(tmp_path / "old.svg"), **{keyword: None})
    error = excinfo.value
    assert error.fields["code"] == "bundle_draw_keyword_unsupported"
    assert error.fields["argument"] == keyword
    assert keyword in str(error)
    assert "node_overrides" in str(error) and "open_viewer" in str(error)  # the supported list
    assert not (tmp_path / "old.svg").exists()


def test_show_defaults_to_the_merged_graph(bundle: tl.Bundle, tmp_path: Path) -> None:
    target = str(tmp_path / "merged.svg")
    shown = bundle.show(path=target)
    assert isinstance(shown, tl.viz.Drawing)
    assert "TorchLens bundle graph" in shown.dot
    assert shown.dot == bundle.draw(target).dot
    graph_kind = bundle.show(kind="graph", path=target)
    assert isinstance(graph_kind, tl.viz.Drawing) and graph_kind.dot == shown.dot


def test_show_strip_is_each_members_own_graph(bundle: tl.Bundle, tmp_path: Path) -> None:
    strip = bundle.show(kind="strip", path=str(tmp_path / "strip"), file_format="svg")
    assert isinstance(strip, dict)
    assert list(strip) == ["small", "large"]
    for name in ("small", "large"):
        assert (tmp_path / f"strip_{name}.svg").is_file()
        member = bundle.members[name].draw(path=str(tmp_path / f"strip_{name}"), file_format="svg")
        picture = strip[name]
        assert isinstance(picture, tl.viz.Drawing) and picture.dot == member.dot
        assert "TorchLens bundle graph" not in member.dot


def test_show_text_kinds_stay_and_unknown_kinds_refuse(bundle: tl.Bundle) -> None:
    assert bundle.show(kind="repr") == {
        name: repr(member) for name, member in bundle.members.items()
    }
    html = bundle.show(kind="html")
    assert isinstance(html, dict) and set(html) == {"small", "large"}
    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.show(kind="side_by_side")
    assert excinfo.value.fields["code"] == "bundle_show_kind_invalid"


def test_show_bundle_graph_is_removed() -> None:
    with pytest.raises(ImportError):
        from torchlens._visualization import show_bundle_graph  # noqa: F401
    with pytest.raises(AttributeError, match="Bundle.draw"):
        tl.show_bundle_graph  # noqa: B018
    assert not hasattr(tl.Bundle, "show_bundle_graph")
