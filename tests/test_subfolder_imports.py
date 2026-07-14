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


def test_notebook_import_stays_inert() -> None:
    """``import torchlens.notebook`` never imports its foreign extra deps.

    Package import must not touch ``IPython`` / ``jupyter_client`` as a side
    effect, regardless of whether the extra is installed -- the dependency
    check is deferred to first attribute access (see the security fix that
    closes a foreign-import-at-untrusted-bundle-load class).
    """
    _drop_module("torchlens.notebook")
    _drop_module("IPython")
    _drop_module("jupyter_client")

    module = importlib.import_module("torchlens.notebook")

    assert module.__all__ == []
    assert "IPython" not in sys.modules
    assert "jupyter_client" not in sys.modules


def test_notebook_attribute_access_reports_missing_dependency() -> None:
    """First attribute access still names the missing optional dependency."""
    _drop_module("torchlens.notebook")

    module = importlib.import_module("torchlens.notebook")

    with patch.dict("sys.modules", {"IPython": None}):
        with pytest.raises(ImportError, match=r"torchlens\.notebook requires extra"):
            module.anything


def test_notebook_attribute_access_when_deps_present() -> None:
    """When deps are installed, attribute access raises AttributeError, not ImportError."""
    pytest.importorskip("IPython")
    pytest.importorskip("jupyter_client")
    _drop_module("torchlens.notebook")

    module = importlib.import_module("torchlens.notebook")

    with pytest.raises(AttributeError):
        module.anything


def test_neuro_import_stays_inert() -> None:
    """``import torchlens.neuro`` never imports its foreign extra deps.

    Package import must not touch ``rsatoolbox`` / ``brainscore_core`` as a
    side effect, regardless of whether the extra is installed -- the
    dependency check is deferred to first attribute access.
    """
    _drop_module("torchlens.neuro")
    _drop_module("rsatoolbox")
    _drop_module("brainscore_core")

    module = importlib.import_module("torchlens.neuro")

    assert module.__all__ == []
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules


def test_neuro_attribute_access_reports_missing_dependency() -> None:
    """First attribute access still names the missing optional dependency."""
    _drop_module("torchlens.neuro")

    module = importlib.import_module("torchlens.neuro")

    with patch.dict("sys.modules", {"rsatoolbox": None}):
        with pytest.raises(ImportError, match=r"torchlens\.neuro requires extra"):
            module.anything


def test_neuro_attribute_access_when_deps_present() -> None:
    """When deps are installed, attribute access raises AttributeError, not ImportError."""
    pytest.importorskip("rsatoolbox")
    pytest.importorskip("brainscore_core")
    _drop_module("torchlens.neuro")

    module = importlib.import_module("torchlens.neuro")

    with pytest.raises(AttributeError):
        module.anything
