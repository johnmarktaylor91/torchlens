"""A rolled loop drawn left to right renders without a Graphviz router warning.

Graphviz 2.43 (the build CI and current Ubuntu ship) fails to route the README loop's
return arc between its inner corner ports under the bundled font's metrics and reports
``libpath/shortest.c:324: triangulation failed``; newer builds route it.  When the
installed Graphviz reports such a router failure for the text an image is rendered
from, the image is laid out again with the edges' compass ports dropped, and that
layout is kept only when it routes cleanly.  The DOT a drawing returns or saves keeps
its ports under every build.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import _render_utils
from torchlens._visualization._loop_edges import reports_route_failure, without_compass_ports

ROUTER_FAILURE = b"Warning: libpath/shortest.c:324: triangulation failed\n"


class ReadmeLoop(nn.Module):
    """The README loop: a branch, then two passes of fc2(relu(fc1(x)))."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() < 0:
            x = -x
        for _ in range(2):
            x = self.fc2(torch.relu(self.fc1(x)))
        return x


def _layout_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(item.message)
        for item in caught
        if getattr(item.message, "fields", {}).get("code") == "layout_engine_stderr"
    ]


def _draw(tmp_path: Path, fill: float, file_format: str = "svg") -> tuple[Any, list[str]]:
    trace = tl.trace(ReadmeLoop(), torch.full((1, 4), fill))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawing = trace.draw(
                rolled=True,
                direction="leftright",
                path=str(tmp_path / f"loop_{fill}"),
                file_format=file_format,
            )
        return drawing, _layout_warnings(caught)
    finally:
        trace.cleanup()


def _source_of(cmd: list[str], kwargs: dict[str, Any]) -> str:
    data = kwargs.get("input")
    if data is None and cmd and Path(str(cmd[-1])).is_file():
        return Path(str(cmd[-1])).read_text(encoding="utf-8")
    return data.decode("utf-8") if isinstance(data, bytes) else str(data or "")


def _graphviz_failing_on(monkeypatch: pytest.MonkeyPatch, fails: Any) -> list[str]:
    """Make every Graphviz child append the 2.43 router failure when ``fails(source)``."""

    real = _render_utils.run_bounded_subprocess
    sources: list[str] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        completed = real(cmd, **kwargs)
        source = _source_of(cmd, kwargs)
        sources.append(source)
        if not fails(source):
            return completed
        stderr = completed.stderr or b""
        extra: Any = ROUTER_FAILURE.decode() if isinstance(stderr, str) else ROUTER_FAILURE
        return subprocess.CompletedProcess(
            completed.args, completed.returncode, completed.stdout, stderr + extra
        )

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", run)
    return sources


@pytest.mark.parametrize("fill", [-1.0, 1.0])
def test_readme_loop_rolled_leftright_renders_without_layout_warning(
    tmp_path: Path, fill: float
) -> None:
    """On the installed Graphviz (2.43 in CI) the README fence warns about nothing."""

    drawing, layout_warnings = _draw(tmp_path, fill)
    assert layout_warnings == []
    assert "tailport=sw" in drawing.dot and "headport=se" in drawing.dot
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawing.save(tmp_path / f"saved_{fill}.png")
    assert _layout_warnings(caught) == []


def test_router_failure_relays_out_the_image_without_compass_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that cannot route the loop's ports gets an image laid out without them."""

    sources = _graphviz_failing_on(monkeypatch, lambda source: "tailport=" in source)
    drawing, layout_warnings = _draw(tmp_path, -1.0)
    assert layout_warnings == []
    rendered = [source for source in sources if "linear_2_6 -> linear_1_4" in source]
    assert rendered and "tailport=" not in rendered[-1], rendered[-1:]
    assert "tailport=sw" in drawing.dot and "headport=se" in drawing.dot
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawing.save(tmp_path / "saved.png")
    assert _layout_warnings(caught) == []


ROUTER_ABORT = (
    b"Error: in routesplines, Pshortestpath failed\nError: lost linear_2_6 linear_1_4 edge\n"
)


def _graphviz_aborting_on(monkeypatch: pytest.MonkeyPatch, fails: Any) -> list[str]:
    """Make every Graphviz child exit 1 on the 2.43 router failure when ``fails(source)``.

    2.43 does not always route on after ``Pshortestpath failed``: on some layouts (two
    return arcs into one node) it exits 1 with ``lost ... edge`` and writes nothing.
    """

    real = _render_utils.run_bounded_subprocess
    sources: list[str] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        source = _source_of(cmd, kwargs)
        sources.append(source)
        if fails(source):
            raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=ROUTER_ABORT)
        return real(cmd, **kwargs)

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", run)
    return sources


def test_router_abort_relays_out_the_image_without_compass_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that aborts on the loop's ports still draws, from the portless layout."""

    sources = _graphviz_aborting_on(monkeypatch, lambda source: "tailport=" in source)
    drawing, _layout_warnings = _draw(tmp_path, -1.0)
    rendered = [source for source in sources if "linear_2_6 -> linear_1_4" in source]
    assert rendered and "tailport=" not in rendered[-1], rendered[-1:]
    assert "tailport=sw" in drawing.dot and "headport=se" in drawing.dot
    assert (tmp_path / "loop_-1.0.svg").exists()


def test_router_failure_without_ports_keeps_the_original_render_and_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the portless layout fails too, the ported image and its warning stand."""

    sources = _graphviz_failing_on(monkeypatch, lambda source: "->" in source)
    _drawing, layout_warnings = _draw(tmp_path, -1.0)
    assert len(layout_warnings) == 1 and "triangulation failed" in layout_warnings[0]
    rendered = [source for source in sources if "linear_2_6 -> linear_1_4" in source]
    assert "tailport=sw" in rendered[-1], rendered[-1]


def test_without_compass_ports_drops_only_edge_compass_ports() -> None:
    source = "\n".join(
        [
            "digraph g {",
            "\ta [label=<<TABLE><TR><TD>tailport=sw</TD></TR></TABLE>> shape=box]",
            "\ta -> b [constraint=false headport=se style=solid tailport=sw]",
            '\tb -> a [headport="ne" tailport=s]',
            "\tc:out -> d:in [color=black]",
            "}",
        ]
    )
    assert without_compass_ports(source) == "\n".join(
        [
            "digraph g {",
            "\ta [label=<<TABLE><TR><TD>tailport=sw</TD></TR></TABLE>> shape=box]",
            "\ta -> b [constraint=false style=solid]",
            "\tb -> a []",
            "\tc:out -> d:in [color=black]",
            "}",
        ]
    )
    assert without_compass_ports("digraph g {\n\ta -> b\n}") == "digraph g {\n\ta -> b\n}"


def test_reports_route_failure_names_router_failures_only() -> None:
    assert reports_route_failure(ROUTER_FAILURE)
    assert reports_route_failure("Error: in routesplines, Pshortestpath failed\n")
    assert reports_route_failure("libpath/shortest.c:200: source point not in any triangle")
    assert not reports_route_failure(None)
    assert not reports_route_failure("")
    assert not reports_route_failure(
        "Warning: graph is too large for cairo-renderer bitmaps. Scaling by 0.5 to fit"
    )
