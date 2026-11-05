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


def _record_keywords(source: str) -> dict[str, ast.expr]:
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record"
    )
    return {str(kw.arg): kw.value for kw in call.keywords}


def _ctor_fields(node: ast.expr) -> dict[str, str]:
    assert isinstance(node, ast.Call)
    return {str(kw.arg): ast.unparse(kw.value) for kw in node.keywords}


def test_record_recording_options_is_kept_not_spilled() -> None:
    # Fails if the codemod spills recording=RecordingOptions(...) back onto record.
    source = (
        "import torchlens as tl\n"
        "tl.record(model, x, recording=tl.options.RecordingOptions("
        'default_op=True, history_size=4, postprocess="none"))\n'
    )
    rewritten, reports = _rewrite(source)
    assert not reports
    assert rewritten == source


def test_record_flat_recording_keywords_fold_into_recording_options() -> None:
    source = (
        "import torchlens as tl\n"
        "tl.record(model, x, default_op=True, seed=1, history_size=4, "
        'on_forward_error="raise")\n'
    )
    rewritten, reports = _rewrite(source)
    assert not reports
    keywords = _record_keywords(rewritten)
    assert set(keywords) == {"capture", "recording"}
    assert ast.unparse(keywords["recording"]).startswith("tl.options.RecordingOptions(")
    assert _ctor_fields(keywords["recording"]) == {
        "default_op": "True",
        "history_size": "4",
        "on_forward_error": "'raise'",
    }
    assert _ctor_fields(keywords["capture"]) == {"seed": "1"}
    assert _rewrite(rewritten) == (rewritten, [])


def test_record_flat_recording_keywords_merge_into_existing_constructor() -> None:
    source = "tl.record(model, x, recording=RecordingOptions(history_size=3), default_op=True)"
    rewritten, reports = _rewrite(source)
    assert not reports
    keywords = _record_keywords(rewritten)
    assert set(keywords) == {"recording"}
    assert _ctor_fields(keywords["recording"]) == {"history_size": "3", "default_op": "True"}


@pytest.mark.parametrize(
    "source",
    [
        "tl.record(model, x, recording=RecordingOptions(history_size=3), history_size=4)",
        "tl.record(model, x, recording=opts, default_op=True)",
    ],
)
def test_record_recording_fold_conflicts_require_review(source: str) -> None:
    rewritten, reports = _rewrite(source)
    assert reports and all("move by hand" in report for report in reports)
    assert rewritten == source


def test_recorder_keywords_stay_flat() -> None:
    source = "rec = tl.recording.Recorder(model, default_op=True, history_size=3)"
    rewritten, reports = _rewrite(source)
    assert not reports
    assert rewritten == source


@pytest.mark.parametrize(
    "source",
    [
        "tl.record(model, x, seed=1, default_op=True, lookback=2, history_size=4)",
        "tl.record(\n    model,\n    x,\n    default_op=True,\n    history_size=4,\n)\n",
    ],
)
def test_record_trailing_keywords_fold_without_stray_commas(source: str) -> None:
    rewritten, reports = _rewrite(source)
    assert not reports
    keywords = _record_keywords(rewritten)
    assert _ctor_fields(keywords["recording"]) == {"default_op": "True", "history_size": "4"}
    assert ", ," not in rewritten and ", )" not in rewritten
    assert _rewrite(rewritten) == (rewritten, [])
