"""F16 B5: the single-file offline report (treescope memo decision 4).

The row gate's JS-off/no-network regression lives here: the report must
carry zero external references, read fully without JavaScript, scrub home
paths, regenerate byte-exactly under ``deterministic=True``, and disclose
exactly what it embeds and omits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


class _Poisoned(nn.Module):
    """Mid-graph NaN injection for the frontier legs."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with a deliberate 0/0 between the layers."""
        h = torch.relu(self.a(x))
        h = h / 0.0 * 0.0
        return self.b(h)


@pytest.fixture(scope="module")
def dirty_log() -> object:
    """One NaN-injected capture shared across the module."""

    log = tl.trace(_Poisoned(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def clean_log() -> object:
    """One healthy capture shared across the module."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _write(log: object, path: Path, *, share_safe: bool = False, **kwargs: object) -> str:
    """Write one report and return its text."""

    options = tl.export.ReportOptions(deterministic=True, share_safe=share_safe)
    tl.export.html(log, path, options=options, **kwargs)
    return path.read_text(encoding="utf-8")


@pytest.mark.smoke
def test_report_anatomy(dirty_log, tmp_path: Path) -> None:
    """Banner, honesty, summary, graph, ops, arrays, manifest -- in one file."""

    text = _write(dirty_log, tmp_path / "r.html")
    for marker in (
        "TorchLens report:",
        "tl-report-banner",
        "<h2>summary</h2>",
        "<h2>graph</h2>",
        "<h2>ops</h2>",
        "<h2>arrays</h2>",
        "<h2>manifest</h2>",
        "Content-Security-Policy",
    ):
        assert marker in text, marker
    # Stable per-op anchors.
    assert 'id="op-relu-1-2-1"' in text


@pytest.mark.smoke
def test_no_network_regression(dirty_log, tmp_path: Path) -> None:
    """The one-file contract: zero external references of any kind."""

    text = _write(dirty_log, tmp_path / "r.html")
    assert "<script src=" not in text
    assert "<link " not in text
    assert not re.search(r'(?:src|href)="https?://', text)
    assert "url(http" not in text and "@import" not in text


@pytest.mark.smoke
def test_js_disabled_readability(dirty_log, tmp_path: Path) -> None:
    """Strip every script: the sections and data all remain readable."""

    text = _write(dirty_log, tmp_path / "r.html")
    without_js = re.sub(r"<script>.*?</script>", "", text, flags=re.S)
    for marker in (
        "TorchLens report:",
        "<h2>summary</h2>",
        "<h2>ops</h2>",
        "flagged,",
        "report manifest",
    ):
        assert marker in without_js, marker
    # The graph area relies on native scrolling, not scripts.
    assert "tl-report-graph" in without_js


def test_frontier_default_embeds_only_the_frontier(dirty_log, tmp_path: Path) -> None:
    """Dirty capture: thumbnails for the frontier, disclosure exact."""

    text = _write(dirty_log, tmp_path / "r.html")
    manifest = _manifest_of(text)
    assert manifest["arrays"]["mode"] == "frontier"
    assert manifest["arrays"]["flagged_total"] >= 3
    shown = manifest["arrays"]["sites_shown"]
    assert 1 <= shown <= 8
    assert f"{manifest['arrays']['flagged_total']} flagged, {shown} shown" in text
    assert "tl-grid-table" in text  # thumbnails present
    assert "truediv" in text  # the injection site is in the frontier


@pytest.mark.smoke
def test_clean_capture_embeds_zero_array_bytes(clean_log, tmp_path: Path) -> None:
    """Healthy captures embed no values by default; graph depth disclosed."""

    text = _write(clean_log, tmp_path / "c.html")
    manifest = _manifest_of(text)
    assert manifest["arrays"]["sites_shown"] == 0
    assert manifest["arrays"]["bytes_embedded"] == 0
    assert "0 flagged, 0 shown" in text
    assert "tl-grid-table" not in text
    assert manifest["graph"].get("collapse") in ("auto", "max", "none")


def test_share_safe_strips_everything(dirty_log, tmp_path: Path) -> None:
    """share_safe: no thumbnails, banner and manifest say so."""

    text = _write(dirty_log, tmp_path / "s.html", share_safe=True)
    assert "tl-grid-table" not in text
    assert "share_safe: every embedded value stripped" in text
    manifest = _manifest_of(text)
    assert manifest["arrays"]["share_safe"] is True
    assert manifest["arrays"]["sites_shown"] == 0


def test_arrays_flagged_is_capped_with_disclosure(dirty_log, tmp_path: Path) -> None:
    """The debugging preset stays under the same hard budgets."""

    text = _write(dirty_log, tmp_path / "f.html", arrays="flagged")
    manifest = _manifest_of(text)
    assert manifest["arrays"]["mode"] == "flagged"
    assert manifest["arrays"]["sites_shown"] <= manifest["arrays"]["budgets"]["max_sites"]


def test_arrays_selection_form(dirty_log, tmp_path: Path) -> None:
    """Row X15: arrays=<Selection> composes with the selection algebra."""

    selection = tl.units("relu_1_2", [(0, 0)])
    text = _write(dirty_log, tmp_path / "sel.html", arrays=selection)
    manifest = _manifest_of(text)
    assert manifest["arrays"]["mode"] == "selection"
    assert manifest["arrays"]["sites_shown"] >= 1
    assert "selected," in text


@pytest.mark.smoke
def test_invalid_arrays_refuses_typed(dirty_log, tmp_path: Path) -> None:
    """Unknown arrays= values refuse with report_arrays_invalid."""

    with pytest.raises(Exception) as excinfo:
        tl.export.html(dirty_log, tmp_path / "x.html", arrays=42)
    assert excinfo.value.fields["code"] == "report_arrays_invalid"


@pytest.mark.smoke
def test_deterministic_regeneration_and_scrub(dirty_log, tmp_path: Path) -> None:
    """deterministic=True regenerates byte-exactly; home paths never leak."""

    first = _write(dirty_log, tmp_path / "a.html")
    second = _write(dirty_log, tmp_path / "b.html")
    assert first == second
    home = str(Path.home())
    if home not in ("/", ""):
        assert home not in first


def test_partial_trace_report_is_first_class(tmp_path: Path) -> None:
    """A failed capture writes a shippable one-file failure report."""

    class Dies(nn.Module):
        """Fixture model failing mid-forward."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Raise after one committed op."""
            torch.relu(self.a(x))
            raise RuntimeError("cluster boom")

    with pytest.raises(RuntimeError):
        tl.trace(Dies(), torch.randn(2, 4))
    try:
        tl.trace(Dies(), torch.randn(2, 4))
    except RuntimeError as error:
        partial = tl.partial.from_failed_capture(error)
    assert partial is not None
    text = _write(partial, tmp_path / "p.html")
    assert "FAILED CAPTURE" in text
    assert "committed prefix" in text
    assert "<h2>manifest</h2>" in text
    assert "<script src=" not in text


def test_graph_fallback_warns_and_labels(
    clean_log, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-layout failure: report_graph_fallback fires, canvas is labeled.

    The fallback picture is a schematic grid, NOT the real geometry -- the
    report must say so inline, in the manifest, and through the coded
    warning (never a silent stand-in).
    """

    from torchlens.export import _report as report_module

    monkeypatch.setattr(report_module, "_draw_svg", lambda *args, **kwargs: None)
    path = tmp_path / "fb.html"
    with pytest.warns(UserWarning) as caught:
        tl.export.html(clean_log, path, options=tl.export.ReportOptions(deterministic=True))
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "report_graph_fallback" in codes
    text = path.read_text(encoding="utf-8")
    assert "fallback layout (schematic grid, NOT the real graph geometry)" in text
    manifest = _manifest_of(text)
    assert manifest["graph"]["fallback"] is True


def test_interactive_viewer_markers_survive(clean_log, tmp_path: Path) -> None:
    """The report keeps a self-contained pan/zoom viewer (wheel + drag)."""

    text = _write(clean_log, tmp_path / "v.html")
    assert "addEventListener('wheel'" in text
    assert "onmousemove" in text


def _manifest_of(text: str) -> dict:
    """Parse the embedded manifest JSON back out of the report."""

    match = re.search(r"<pre>(\{.*?\})</pre>", text, flags=re.S)
    assert match, "manifest block missing"
    import html as html_module

    return json.loads(html_module.unescape(match.group(1)))
