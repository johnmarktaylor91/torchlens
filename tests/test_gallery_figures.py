"""The gallery figures draw from saved result files, without torch and within budget."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "examples" / "gallery"
PANE_WIDTH = 1520  # the docs figure target in pixels, exported at 2x


def _load_figures() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_gallery_figures", GALLERY / "_figures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _check_outputs(paths: list[Path], stems: list[str]) -> None:
    assert [p.name for p in paths] == [f"{s}.{x}" for s in stems for x in ("png", "svg")]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
        if path.suffix == ".svg":
            text = path.read_text("utf-8")
            assert "<svg" in text and "<text" in text, "SVG text must stay text, not paths"
        else:
            width, _height = _png_size(path)
            assert PANE_WIDTH - 60 <= width <= PANE_WIDTH + 60, (path, width)


def test_causal_tracing_figure_from_result_file(tmp_path: Path) -> None:
    figures = _load_figures()
    rng = np.random.default_rng(0)
    grid = {c: [float(v) for v in rng.normal(0, 0.001, 12)] for c in figures.TOKEN_CATEGORIES}
    grid["last subject token"][3] = 0.02
    summary = {
        "measured": {"ate": 0.1, "aie_grid_single_layer": grid},
        "oracles": {
            "paper_rule": {
                "measured": {
                    "peak_layer": 3,
                    "peak_category": "last subject token",
                    "peak_aie": 0.02,
                }
            }
        },
        "controls": {
            "wrong_layer": {"measured": {"aie": -0.001, "wrong_layer": 9}},
            "wrong_fact": {"measured": {"aie": 0.0002}},
        },
        "pinned": {"facts": "first 2 kept records", "noise": "3 samples per fact"},
    }
    (tmp_path / "causal_tracing.json").write_text(json.dumps(summary))
    _check_outputs(figures.draw_causal_tracing(tmp_path), ["causal_tracing"])


def test_cnn_fields_figure_from_result_files(tmp_path: Path) -> None:
    figures = _load_figures()
    side = 16
    spatial = np.zeros((side, side), dtype=bool)
    spatial[2:12, 3:13] = True
    patch = np.zeros((side, side), dtype=bool)
    patch[7:9, 7:9] = True
    projected = np.zeros((4, 4), dtype=bool)
    projected[1:3, 1:3] = True
    np.savez(
        tmp_path / "cnn_fields_figure.npz",
        image=np.random.default_rng(0).random((side, side, 3)),
        spatial=spatial,
        empirical=spatial,
        projected=projected,
        changed=projected,
        patch=patch,
    )
    summary = {
        "oracles": {"gradient_support": {"measured": 1.0}, "projective_support": {"measured": 1.0}},
        "pinned": {"layer": "layer3"},
    }
    (tmp_path / "cnn_fields.json").write_text(json.dumps(summary))
    _check_outputs(figures.draw_cnn_fields(tmp_path), ["cnn_fields"])


def test_neuro_rdms_figure_from_result_files(tmp_path: Path) -> None:
    figures = _load_figures()
    rng = np.random.default_rng(0)
    rdms = np.stack([1 - np.corrcoef(rng.normal(size=(5, 8))) for _ in range(2)])
    np.savez(
        tmp_path / "neuro_rdms_figure.npz",
        rdms=rdms,
        stages=np.array(["layer1", "avgpool"]),
        correlation_min=0.999,
        shuffled_max=0.5,
    )
    keys = ["avgpool", "layer1", "residual_0", "residual_1"]
    summary = {
        "oracles": {"per_layer_rdm": {"measured": dict.fromkeys(keys, 1.0)}},
        "controls": {"shuffled_order": {"measured": dict.fromkeys(keys, -0.04)}},
    }
    (tmp_path / "neuro_rdms.json").write_text(json.dumps(summary))
    _check_outputs(figures.draw_neuro_rdms(tmp_path), ["neuro_rdms"])


def test_ioi_circuit_figures_from_figure_file(tmp_path: Path) -> None:
    figures = _load_figures()
    data = {
        "rows": 10,
        "steps": 2,
        "auc": 0.75,
        "roc": [[0.0, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.0]],
        "k": 2,
        "recall_at_k": 0.5,
        "fpr_at_k": 0.25,
        "positives": 2,
        "heads": 6,
        "layers": 3,
        "heads_per_layer": 2,
        "groups": {"name mover": [[2, 1]], "negative": [[2, 0]]},
        "edges": [
            {"source": [0, 0], "target": [2, 1], "channel": "q", "score": 0.9},
            {"source": [1, 1], "target": [2, 0], "channel": "v", "score": -0.4},
        ],
    }
    (tmp_path / "ioi_circuit_figure.json").write_text(json.dumps(data))
    _check_outputs(figures.draw_ioi_circuit(tmp_path), ["ioi_roc", "ioi_circuit"])


def test_ioi_figure_data_reduces_scores() -> None:
    sys.path.insert(0, str(GALLERY))
    try:
        ioi = importlib.import_module("ioi_circuit")
    finally:
        sys.path.remove(str(GALLERY))
    scores = {
        "a9.h9->a10.h7<v>": 0.9,
        "a5.h5->a8.h6<q>": -0.6,
        "a0.h1->a1.h1<k>": 0.2,
        "resid_pre->a9.h9<q>": 0.3,
    }
    data = ioi.figure_data(scores, k=2, edges=2, run={"rows": 3, "steps": 1})
    assert data["rows"] == 3 and data["steps"] == 1
    assert data["roc"][0] == [0.0, 0.0] and data["roc"][-1] == [1.0, 1.0]
    assert 0.0 <= data["auc"] <= 1.0
    assert data["heads"] == 144 and data["positives"] == 26
    assert data["layers"] == 12 and data["heads_per_layer"] == 12
    assert set(data["groups"]) == set(ioi.HEAD_GROUPS)
    assert [edge["source"] for edge in data["edges"]] == [[9, 9], [5, 5]]
    assert data["edges"][1]["channel"] == "q" and data["edges"][1]["score"] == -0.6


@pytest.mark.parametrize(
    "script", sorted(p.name for p in GALLERY.glob("*.py") if not p.name.startswith("_"))
)
def test_gallery_scripts_define_everything_before_the_main_guard(script: str) -> None:
    """A def below ``if __name__ == "__main__"`` is undefined while the script runs."""
    import ast

    tree = ast.parse((GALLERY / script).read_text("utf-8"))
    guard_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    ]
    assert len(guard_lines) == 1, f"{script}: expected one main guard"
    late = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.lineno > guard_lines[0]
    ]
    assert late == [], f"{script}: defined after the main guard: {late}"
