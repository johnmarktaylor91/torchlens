"""``Trace.draw`` output rules shared with ``Bundle.draw`` (JMT drawing rulings, Sun 1 Nov 2026).

- The path's extension picks the format; ``file_format=`` applies only to a path
  with no extension; a ``file_format`` that disagrees with the extension refuses
  with ``draw_format_mismatch``.
- Outside a Jupyter kernel a bare ``trace.draw()`` writes ``modelgraph.<format>``
  (collision-safe) and prints the path; inside a kernel it writes no file and
  returns the picture for inline display. A path argument always writes.

Kernel detection is ``torchlens.viz._drawing._in_notebook``, the same check
``tl.draw`` uses; the tests monkeypatch it. The bare-call tests pass ``path=None``
explicitly: the suite's autouse leak guard (``tests/_leak_guards.py``) points the
``path`` default of ``Trace.draw`` into ``tmp_path``, so an argument-free call in a
test is not a bare call.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.viz._drawing as drawing_module
from torchlens.errors import ArgumentConflictError

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot not installed")


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    torch.manual_seed(0)
    captured = tl.trace(_Tiny(), torch.randn(1, 3))
    try:
        yield captured
    finally:
        captured.cleanup()


def _files(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


def test_extension_picks_the_format(trace: tl.Trace, tmp_path: Path) -> None:
    drawing = trace.draw(str(tmp_path / "g.png"))
    assert drawing.file_format == "png"
    assert drawing.path == tmp_path / "g.png"
    assert (tmp_path / "g.png").read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "g.png.pdf").exists()
    assert not (tmp_path / "g.pdf").exists()


def test_file_format_applies_to_a_path_without_extension(trace: tl.Trace, tmp_path: Path) -> None:
    drawing = trace.draw(str(tmp_path / "g"), file_format="svg")
    assert drawing.file_format == "svg"
    assert drawing.path == tmp_path / "g.svg"
    assert (tmp_path / "g.svg").exists()


def test_matching_file_format_and_extension_is_accepted(trace: tl.Trace, tmp_path: Path) -> None:
    drawing = trace.draw(str(tmp_path / "g.SVG"), file_format="svg")
    assert drawing.file_format == "svg"


def test_disagreeing_file_format_refuses(trace: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(ArgumentConflictError) as excinfo:
        trace.draw(str(tmp_path / "g.png"), file_format="pdf")
    assert excinfo.value.fields["code"] == "draw_format_mismatch"
    assert _files(tmp_path) == []


def test_bare_draw_outside_a_kernel_writes_and_prints(
    trace: tl.Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(drawing_module, "_in_notebook", lambda: False)
    monkeypatch.chdir(tmp_path)
    first = trace.draw(path=None)
    second = trace.draw(path=None)
    assert first.file_format == "pdf"
    assert first.path is not None and first.path.name == "modelgraph.pdf"
    assert second.path is not None and second.path.name == "modelgraph-2.pdf"
    assert (tmp_path / "modelgraph.pdf").exists()
    assert "modelgraph.pdf" in capsys.readouterr().out


def test_bare_draw_in_a_kernel_writes_nothing_and_displays_inline(
    trace: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(drawing_module, "_in_notebook", lambda: True)
    monkeypatch.chdir(tmp_path)
    drawing = trace.draw(path=None)
    assert drawing.path is None
    assert drawing.file_format == "svg"
    svg = drawing._repr_svg_()  # the inline display path; the payload bytes are private
    assert svg is not None and "<svg" in svg
    assert _files(tmp_path) == []


def test_path_in_a_kernel_still_writes(
    trace: tl.Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(drawing_module, "_in_notebook", lambda: True)
    drawing = trace.draw(str(tmp_path / "g.svg"))
    assert drawing.path == tmp_path / "g.svg"
    assert (tmp_path / "g.svg").exists()


def test_tl_draw_takes_the_format_from_the_extension(tmp_path: Path) -> None:
    torch.manual_seed(0)
    drawing = tl.draw(_Tiny(), torch.randn(1, 3), path=str(tmp_path / "g.png"))
    assert drawing.file_format == "png"
    assert (tmp_path / "g.png").exists()
    assert not (tmp_path / "g.png.pdf").exists()


def test_visualization_options_path_extension_picks_the_format(
    trace: tl.Trace, tmp_path: Path
) -> None:
    options = tl.options.VisualizationOptions(container_path=str(tmp_path / "g.png"))
    drawing = trace.draw(**tl.options.visualization_to_render_kwargs(options))
    assert drawing.file_format == "png"
    assert (tmp_path / "g.png").exists()
    assert not (tmp_path / "g.pdf").exists()


def _detached_svg_with_key(
    trace: tl.Trace, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> tuple[str, str, list[str]]:
    """Draw with the key to a path, then re-render it as a notebook displays it.

    Returns the displayed SVG, the Graphviz text written to stderr and the
    warning messages raised by the draw and the display.
    """

    import warnings

    capfd.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drawing = trace.draw(str(tmp_path / "keyed.svg"), show_legend=True)
        svg = drawing._repr_svg_()
    assert svg is not None
    return svg, capfd.readouterr().err, [str(item.message) for item in caught]


def test_displayed_drawing_with_key_links_no_temp_file(
    trace: tl.Trace, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The key's icons are embedded in the displayed SVG, never linked from the temp folder."""

    import re
    import tempfile

    svg, _, _ = _detached_svg_with_key(trace, tmp_path, capfd)
    hrefs = re.findall(r'href="([^"]*)"', svg)
    assert hrefs, "the key draws its swatches as images"
    assert all(href.startswith("data:image/") for href in hrefs), hrefs[:3]
    assert tempfile.gettempdir() not in svg


def test_displayed_drawing_with_key_renders_without_graphviz_warning(
    trace: tl.Trace, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Drawing and displaying the key draws no Graphviz warning (the key table fits its words)."""

    _, stderr, messages = _detached_svg_with_key(trace, tmp_path, capfd)
    assert "Warning" not in stderr, stderr
    assert not [message for message in messages if "Graphviz" in message], messages
