"""Tests for TensorFlow private compatibility probes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from torchlens.backends.registry import BackendUnsupportedError
from torchlens.backends.tf import _tf_compat as tfc


def test_op_callbacks_absence_marks_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing TensorFlow op callbacks raise the public unsupported-backend error."""

    monkeypatch.setattr(tfc, "HAS_TF_OP_CALLBACKS", True)
    tfc._warned_missing_capabilities.discard("HAS_TF_OP_CALLBACKS")
    monkeypatch.setattr(tfc, "_import_op_callbacks_module", lambda: None)

    with pytest.warns(UserWarning, match="HAS_TF_OP_CALLBACKS"):
        with pytest.raises(BackendUnsupportedError):
            tfc.get_op_callbacks_module()

    assert tfc.HAS_TF_OP_CALLBACKS is False


def _fake_tf_spec(location: Path) -> SimpleNamespace:
    """A stand-in TensorFlow package spec rooted at ``location``."""

    return SimpleNamespace(submodule_search_locations=[str(location)])


def test_op_callbacks_probe_locates_the_module_without_importing_tensorflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With TF not loaded, the probe reads the package layout and never imports."""

    framework = tmp_path / "python" / "framework"
    framework.mkdir(parents=True)
    (framework / "op_callbacks.py").write_text("", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(tfc.importlib.util, "find_spec", lambda name: _fake_tf_spec(tmp_path))

    def _no_import() -> None:
        raise AssertionError("the probe imported TensorFlow")

    monkeypatch.setattr(tfc, "_import_op_callbacks_module", _no_import)

    assert tfc._probe_op_callbacks() is True
    (framework / "op_callbacks.py").unlink()
    assert tfc._probe_op_callbacks() is False


def test_op_callbacks_probe_is_false_without_tensorflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """No TensorFlow package spec means no op callbacks."""

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(tfc.importlib.util, "find_spec", lambda name: None)

    assert tfc._probe_op_callbacks() is False
