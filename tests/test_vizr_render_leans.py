"""The graph-rendering leans of forks.11 (FK1, FK3, FK4).

FK1: an explicit vector render past the page ceiling is written exactly as
asked, with a coded warning naming the size and the remedies.

FK3: every Graphviz render resolves the pinned ``Helvetica`` family to the
bundled open font (Liberation Sans, SIL OFL 1.1) instead of whatever face the
machine's fontconfig picks, while the emitted DOT keeps its family name.

FK4: an over-ceiling residual renders a bounded overview and writes an
expansion manifest beside the picture instead of an uncollapsed wall or a
refusal.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
import zlib
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.visualization import _fonts, render_execution


class _Tiny(nn.Module):
    """Two-op model for render checks."""

    def __init__(self) -> None:
        """Initialize the layer."""
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.fc(x))


def _pdf_text_blobs(data: bytes) -> bytes:
    """Return the PDF bytes plus every inflatable stream, for name searches."""

    blobs = [data]
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        try:
            blobs.append(zlib.decompress(match.group(1)))
        except zlib.error:
            continue
    return b"\n".join(blobs)


def test_fk3_bundled_font_files_ship_with_license() -> None:
    """The bundled faces and their OFL license are present beside the module."""
    status = _fonts.bundled_font_status()
    assert status["available"] is True
    assert status["family"] == "Liberation Sans"
    font_dir = Path(str(status["font_dir"]))
    for name in _fonts.BUNDLED_FONT_FILES:
        assert (font_dir / name).stat().st_size > 100_000
    license_text = (font_dir / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
    assert "Reserved Font Name Liberation" in license_text


def test_fk3_pinned_family_resolves_to_bundled_font() -> None:
    """Under the render environment fontconfig resolves the pinned family to the bundled file."""
    env = _fonts.graphviz_font_env()
    assert env is not None
    fc_match = shutil.which("fc-match")
    assert fc_match is not None, "fontconfig's fc-match is required on the render host"
    resolved = {
        pattern: subprocess.run(
            [fc_match, "-f", "%{file}", pattern],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        for pattern in ("Helvetica", "Helvetica:bold", "Arial", "Arial:bold")
    }
    assert os.path.basename(resolved["Helvetica"]) == "LiberationSans-Regular.ttf"
    assert os.path.basename(resolved["Helvetica:bold"]) == "LiberationSans-Bold.ttf"
    # Graphviz's PostScript alias table asks fontconfig for Arial, not Helvetica.
    assert os.path.basename(resolved["Arial"]) == "LiberationSans-Regular.ttf"
    assert os.path.basename(resolved["Arial:bold"]) == "LiberationSans-Bold.ttf"
    font_dir = str(_fonts.bundled_font_status()["font_dir"])
    assert all(path.startswith(font_dir) for path in resolved.values())


def test_fk3_pdf_render_embeds_bundled_font(tmp_path: Path) -> None:
    """A real PDF render embeds Liberation Sans; the DOT still names Helvetica."""
    trace = tl.trace(_Tiny(), torch.randn(2, 3))
    try:
        dot = trace.draw(
            vis_outpath=str(tmp_path / "fk3"), vis_save_only=True, vis_fileformat="pdf"
        )
    finally:
        trace.cleanup()
    assert "fontname=Helvetica" in dot or 'fontname="Helvetica"' in dot
    searchable = _pdf_text_blobs((tmp_path / "fk3.pdf").read_bytes())
    assert b"LiberationSans" in searchable


def _codes(records: list[warnings.WarningMessage]) -> list[str]:
    """Return the TorchLens warning codes among captured warnings."""

    return [
        str(getattr(record.message, "fields", {}).get("code"))
        for record in records
        if isinstance(record.message, TorchLensWarning)
    ]


def test_fk1_oversize_vector_warns_and_obeys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the page ceiling the explicit render is still written, with a coded warning."""
    monkeypatch.setattr(render_execution, "VECTOR_PAGE_CEILING_POINTS", 50.0)
    trace = tl.trace(_Tiny(), torch.randn(2, 3))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trace.draw(vis_outpath=str(tmp_path / "big"), vis_save_only=True, vis_fileformat="svg")
    finally:
        trace.cleanup()
    assert (tmp_path / "big.svg").stat().st_size > 0, "the oversize render was not written"
    assert "render_vector_oversize" in _codes(records)
    message = next(
        str(record.message) for record in records if "render_vector_oversize" in _codes([record])
    )
    assert "50 pt" in message and "collapse='auto'" in message and "module=" in message


def test_fk1_ordinary_vector_render_is_quiet(tmp_path: Path) -> None:
    """A small render under the real ceiling raises no oversize warning."""
    trace = tl.trace(_Tiny(), torch.randn(2, 3))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trace.draw(
                vis_outpath=str(tmp_path / "small"), vis_save_only=True, vis_fileformat="pdf"
            )
    finally:
        trace.cleanup()
    assert "render_vector_oversize" not in _codes(records)


class _Block(nn.Module):
    """Three-layer child module (a box the overview can collapse)."""

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.fc2(torch.relu(self.fc1(x)))


class _RootFloodWithBlocks(nn.Module):
    """Root owns more units than K_CAP, so auto falls to the floor plan."""

    def __init__(self) -> None:
        """Initialize the child blocks."""
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        for _ in range(70):
            x = torch.tanh(x)
        for block in self.blocks:
            x = block(x)
        return x


def _manifest(path: Path) -> dict:
    """Load a written expansion manifest."""

    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_fk4_pathological_trace_renders_bounded_overview_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the pre-gate, auto renders top-level boxes and writes the manifest."""
    from torchlens.visualization import collapse_optimizer

    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
    trace = tl.trace(_RootFloodWithBlocks(), torch.randn(2, 4))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            dot = trace.draw(
                collapse="auto",
                vis_outpath=str(tmp_path / "ov"),
                vis_save_only=True,
                vis_fileformat="svg",
            )
    finally:
        trace.cleanup()
    codes = _codes(records)
    assert "collapse_pathological_skip" in codes
    assert "collapse_expansion_manifest" in codes
    assert (tmp_path / "ov.svg").stat().st_size > 0
    assert "collapse overview: top-level modules only" in dot
    manifest = _manifest(tmp_path / "ov.expansion.json")
    assert manifest["schema"] == "torchlens.expansion_manifest.v1"
    assert manifest["planner"] == "overview_fallback"
    expandable = {entry["module"] for entry in manifest["entries"]}
    assert "blocks" in expandable or {"blocks.0", "blocks.1", "blocks.2"} <= expandable
    for entry in manifest["entries"]:
        assert entry["expand"], "every collapsed unit names the draw() call that opens it"


def test_fk4_floor_residual_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A floor-fallback plan still above the band ships the manifest beside it.

    ``_RootFloodWithBlocks``' root owns more units than K_CAP, so the frontier
    empties and the floor plan serves. The readable band is narrowed to one
    unit so that plan is certainly over the band: the residual FK4 discloses.
    """
    from torchlens.visualization import auto_collapse

    monkeypatch.setattr(auto_collapse, "_readable_band_high", lambda trace: 1)
    trace = tl.trace(_RootFloodWithBlocks(), torch.randn(2, 4))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            trace.draw(
                collapse="auto",
                vis_outpath=str(tmp_path / "floor"),
                vis_save_only=True,
                vis_fileformat="svg",
            )
    finally:
        trace.cleanup()
    codes = _codes(records)
    assert "collapse_floor_fallback" in codes
    assert "collapse_expansion_manifest" in codes
    manifest = _manifest(tmp_path / "floor.expansion.json")
    assert manifest["planner"] == "floor_fallback"
    assert manifest["readable_band_high"] == 1
    assert manifest["visible_count"] > 1
    assert all(entry["expansion_id"] and entry["expand"] for entry in manifest["entries"])


def test_fk4_ordinary_auto_render_writes_no_manifest(tmp_path: Path) -> None:
    """An in-band auto render writes only the picture."""
    trace = tl.trace(_Block(), torch.randn(2, 4))
    try:
        trace.draw(
            collapse="auto",
            vis_outpath=str(tmp_path / "plain"),
            vis_save_only=True,
            vis_fileformat="svg",
        )
    finally:
        trace.cleanup()
    assert not (tmp_path / "plain.expansion.json").exists()
