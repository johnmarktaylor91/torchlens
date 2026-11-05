"""``HAS_CAIROSVG``: one lazy probe decides every PDF/PNG composition path and its skip reason."""

from __future__ import annotations

import builtins

import pytest

from torchlens._visualization import _cairosvg, _edit_highlighter, _render_flow
from torchlens.errors import TorchLensWarning


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HAS_CAIROSVG", "_CAIROSVG_PROBED", "_CAIROSVG_UNAVAILABLE_REASON"):
        monkeypatch.setattr(_cairosvg, name, getattr(_cairosvg, name))
    monkeypatch.setattr(_cairosvg, "_CAIROSVG_PROBED", False)


def _fail_cairosvg_import(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cairosvg":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_cairo_library_is_reported_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present package without libcairo raises OSError; the reason names the library."""

    _reset(monkeypatch)
    _fail_cairosvg_import(
        monkeypatch, OSError('no library called "cairo-2" was found\nno library called "cairo"')
    )
    assert _cairosvg.get_cairosvg_support(force_probe=True) is False
    reason = _cairosvg.cairosvg_unavailable_reason()
    assert "cairo library is missing" in reason and 'no library called "cairo-2"' in reason
    assert "\n" not in reason


def test_missing_cairosvg_package_is_reported_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    _fail_cairosvg_import(monkeypatch, ImportError("No module named 'cairosvg'"))
    assert _cairosvg.get_cairosvg_support(force_probe=True) is False
    assert "package not installed" in _cairosvg.cairosvg_unavailable_reason()


def test_probe_latches_after_one_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    calls = 0

    def counting_probe() -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, ""

    monkeypatch.setattr(_cairosvg, "_probe_cairosvg", counting_probe)
    assert _cairosvg.get_cairosvg_support() is True
    assert _cairosvg.get_cairosvg_support() is True
    assert _cairosvg.cairosvg_unavailable_reason() == ""
    assert calls == 1
    assert _cairosvg.get_cairosvg_support(force_probe=True) is True
    assert calls == 2


def test_live_flag_agrees_with_a_real_import() -> None:
    """The latched flag answers what ``import cairosvg`` does on this install."""

    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError):
        importable = False
    else:
        importable = True
    assert _cairosvg.get_cairosvg_support(force_probe=True) is importable
    assert (_cairosvg.cairosvg_unavailable_reason() == "") is importable


def test_pdf_highlighter_warns_with_the_reason_when_cairosvg_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_edit_highlighter, "get_cairosvg_support", lambda: False)
    monkeypatch.setattr(
        _edit_highlighter, "cairosvg_unavailable_reason", lambda: "no library called cairo-2"
    )
    with pytest.warns(TorchLensWarning, match="no library called cairo-2") as record:
        assert _edit_highlighter.highlighter_supported("pdf") is False
    assert [getattr(w.message, "fields", {}).get("code") for w in record] == [
        "viz_highlighter_not_drawn"
    ]
    monkeypatch.setattr(_edit_highlighter, "get_cairosvg_support", lambda: True)
    assert _edit_highlighter.highlighter_supported("png") is True


def test_code_panel_composition_follows_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_render_flow, "get_cairosvg_support", lambda: False)
    assert _render_flow._code_panel_composition_available("pdf", "dot") is False
    assert _render_flow._code_panel_composition_available("svg", "dot") is True
    monkeypatch.setattr(_render_flow, "get_cairosvg_support", lambda: True)
    assert _render_flow._code_panel_composition_available("pdf", "dot") is True
