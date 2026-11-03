"""Behavioral tests for removed capture-field migration."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from tools.codemods.r2_options import rewrite_source


def _rewrite(source: str) -> tuple[str, list[str]]:
    result: list[str] = []
    reports: list[str] = []
    rewrite_source(source, Path("example.py"), False, Counter(), reports, result)
    rewritten = result[0] if result else source
    ast.parse(rewritten)
    return rewritten, reports


@pytest.mark.parametrize("prefix", ["tl", "torchlens", "_tl()"])
def test_inline_selection_moves_and_drops_empty_capture(prefix: str) -> None:
    source = f'{prefix}.trace(model, x, capture=CaptureOptions(layers_to_save=["relu"], save_preview=False))'
    rewritten, reports = _rewrite(source)
    assert not reports
    assert "capture=" not in rewritten
    assert 'save=["relu"]' in rewritten
    assert _rewrite(rewritten) == (rewritten, [])


def test_selection_preserves_other_capture_fields() -> None:
    rewritten, reports = _rewrite(
        "tl.record(model, x, capture=CaptureOptions(seed=2, layers_to_save=None))"
    )
    assert not reports
    assert "seed=2" in rewritten
    assert "save=None" in rewritten
    assert "layers_to_save" not in rewritten


def test_visible_variable_migrates_to_each_door() -> None:
    source = 'def f():\n    cap = CaptureOptions(layers_to_save=["relu"], seed=1)\n    a = tl.trace(model, x, capture=cap)\n    return tl.record(model, x, capture=cap)\n'
    rewritten, reports = _rewrite(source)
    assert not reports
    assert rewritten.count('save=["relu"]') == 2
    assert "layers_to_save" not in rewritten


@pytest.mark.parametrize(
    "source",
    [
        'tl.trace(model, x, save="all", capture=CaptureOptions(layers_to_save=["relu"]))',
        'cap = CaptureOptions(layers_to_save=["relu"])\nunknown(cap)',
        "cap = CaptureOptions(save_preview=True)",
        'cap = CaptureOptions(layers_to_save="all")\nunknown(cap)',
        'def f():\n    cap = CaptureOptions(layers_to_save=["relu"])\n    cap = other\n    return tl.trace(model, x, capture=cap)',
    ],
)
def test_ambiguous_migrations_require_review(source: str) -> None:
    rewritten, reports = _rewrite(source)
    assert reports and all("REVIEW" in report for report in reports)
    assert rewritten == source


def test_imported_constructor_alias_migrates() -> None:
    source = 'from torchlens.options import CaptureOptions as Cap\ntl.trace(model, x, capture=Cap(layers_to_save=["relu"]))'
    rewritten, reports = _rewrite(source)
    assert not reports
    assert 'save=["relu"]' in rewritten
    assert "capture=" not in rewritten
