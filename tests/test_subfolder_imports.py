"""Tests for optional appliance subfolder imports."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


def _drop_module(module_name: str) -> None:
    """Remove a module and its loaded children from ``sys.modules``.

    Parameters
    ----------
    module_name : str
        Fully qualified module name to remove.
    """
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]


def test_viewer_imports() -> None:
    """Verify the viewer namespace is reserved and importable."""
    _drop_module("torchlens.viewer")

    module = importlib.import_module("torchlens.viewer")

    assert module.__all__ == []


def test_paper_imports() -> None:
    """Verify the paper namespace is reserved and importable."""
    _drop_module("torchlens.paper")

    module = importlib.import_module("torchlens.paper")

    assert module.__all__ == []


def test_notebook_imports() -> None:
    """Verify the notebook namespace imports when its extra deps are present."""
    _drop_module("torchlens.notebook")

    pytest.importorskip("IPython")
    pytest.importorskip("jupyter_client")
    module = importlib.import_module("torchlens.notebook")

    assert module.__all__ == []


def test_notebook_import_reports_missing_dependency() -> None:
    """Verify notebook import errors name the missing optional dependency."""
    _drop_module("torchlens.notebook")

    with patch.dict("sys.modules", {"IPython": None}):
        with pytest.raises(ImportError, match=r"torchlens\.notebook requires extra"):
            importlib.import_module("torchlens.notebook")


def test_llm_imports() -> None:
    """Verify the LLM namespace is reserved and importable."""
    _drop_module("torchlens.llm")

    module = importlib.import_module("torchlens.llm")

    assert module.__all__ == []


def test_neuro_imports() -> None:
    """Verify the neuro namespace imports when its extra deps are present."""
    _drop_module("torchlens.neuro")

    pytest.importorskip("rsatoolbox")
    pytest.importorskip("brainscore_core")
    module = importlib.import_module("torchlens.neuro")

    assert module.__all__ == []


def test_neuro_import_reports_missing_dependency() -> None:
    """Verify neuro import errors name the missing optional dependency."""
    _drop_module("torchlens.neuro")

    with patch.dict("sys.modules", {"rsatoolbox": None}):
        with pytest.raises(ImportError, match=r"torchlens\.neuro requires extra"):
            importlib.import_module("torchlens.neuro")
