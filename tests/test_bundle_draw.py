"""``Bundle.draw`` draws the merged bundle graph; ``Bundle.show`` defaults to it.

Pins here:

- ``Bundle.draw(path, ...)`` takes the ``Trace.draw`` keyword spellings plus
  ``nodes=``; the path's extension picks the format, a bare call writes a
  collision-safe ``bundle_modelgraph`` file;
- ``Bundle.show()`` with no kind is the merged graph, ``kind="strip"`` is each
  member's own graph (the former default), ``"repr"`` / ``"html"`` stay;
- ``show_bundle_graph`` is gone: importing it fails and the root name teaches
  ``Bundle.draw``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("graphviz")


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
    source = bundle.draw(str(tmp_path / "diff.svg"))
    assert "TorchLens bundle graph" in source
    assert "[large]" in source  # the ReLU only one member runs keeps its member list
    assert (tmp_path / "diff.svg").read_text().lstrip().startswith(("<?xml", "<svg"))

    bundle.draw(str(tmp_path / "diff.png"))
    assert (tmp_path / "diff.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / "diff.png.pdf").exists()


def test_draw_file_format_applies_to_a_bare_stem(bundle: tl.Bundle, tmp_path: Path) -> None:
    bundle.draw(str(tmp_path / "stem"), file_format="svg")
    assert (tmp_path / "stem.svg").is_file()
    bundle.draw(str(tmp_path / "same.svg"), file_format="svg")
    assert (tmp_path / "same.svg").is_file()
    with pytest.raises(ValueError, match="file_format"):
        bundle.draw(str(tmp_path / "clash.svg"), file_format="png")


def test_bare_draw_never_overwrites(
    bundle: tl.Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle.draw(file_format="svg")
    bundle.draw(file_format="svg")
    assert (tmp_path / "bundle_modelgraph.svg").is_file()
    assert (tmp_path / "bundle_modelgraph-2.svg").is_file()
    assert "bundle_modelgraph-2.svg" in capsys.readouterr().out


def test_draw_members_nodes(bundle: tl.Bundle, tmp_path: Path) -> None:
    source = bundle.draw(str(tmp_path / "members.svg"), nodes="members")
    assert (tmp_path / "members.svg").is_file()
    assert "TorchLens bundle graph (members)" in source
    assert "member_0" in source and "member_1" in source
    assert "fwd_" not in source  # no op nodes in the members picture


def test_draw_graph_rolled_and_direction(bundle: tl.Bundle, tmp_path: Path) -> None:
    combined = bundle.draw(str(tmp_path / "combined.svg"), graph="combined")
    assert "cluster_backward_small" in combined and "cluster_backward_large" in combined
    assert "fwd_" in combined or "[large]" in combined  # the forward ops are drawn too
    rolled = bundle.draw(str(tmp_path / "rolled.svg"), rolled=True, direction="leftright")
    assert "rolled" in rolled
    assert "rankdir=LR" in rolled
    with pytest.raises(ValueError, match="graph"):
        bundle.draw(str(tmp_path / "bad.svg"), graph="both")  # type: ignore[arg-type]


def test_draw_takes_trace_draw_spellings_only(bundle: tl.Bundle, tmp_path: Path) -> None:
    for old in ("vis_mode", "vis_save_only", "vis_direction", "vis_graph_overrides"):
        with pytest.raises(TypeError):
            bundle.draw(str(tmp_path / "old.svg"), **{old: None})


def test_show_defaults_to_the_merged_graph(bundle: tl.Bundle, tmp_path: Path) -> None:
    target = str(tmp_path / "merged.svg")
    shown = bundle.show(path=target)
    assert isinstance(shown, str)
    assert "TorchLens bundle graph" in shown
    assert shown == bundle.draw(target)
    assert bundle.show(kind="graph", path=target) == shown


def test_show_strip_is_each_members_own_graph(bundle: tl.Bundle, tmp_path: Path) -> None:
    strip = bundle.show(kind="strip", path=str(tmp_path / "strip"), file_format="svg")
    assert isinstance(strip, dict)
    assert list(strip) == ["small", "large"]
    for name in ("small", "large"):
        assert (tmp_path / f"strip_{name}.svg").is_file()
        member_dot = bundle.members[name].draw(
            path=str(tmp_path / f"strip_{name}"), file_format="svg"
        )
        assert strip[name] == member_dot
        assert "TorchLens bundle graph" not in member_dot


def test_show_text_kinds_stay_and_unknown_kinds_refuse(bundle: tl.Bundle) -> None:
    assert bundle.show(kind="repr") == {
        name: repr(member) for name, member in bundle.members.items()
    }
    html = bundle.show(kind="html")
    assert isinstance(html, dict) and set(html) == {"small", "large"}
    with pytest.raises(ValueError, match="kind"):
        bundle.show(kind="side_by_side")


def test_show_bundle_graph_is_removed() -> None:
    with pytest.raises(ImportError):
        from torchlens._visualization import show_bundle_graph  # noqa: F401
    with pytest.raises(AttributeError, match="Bundle.draw"):
        tl.show_bundle_graph  # noqa: B018
    assert not hasattr(tl.Bundle, "show_bundle_graph")
