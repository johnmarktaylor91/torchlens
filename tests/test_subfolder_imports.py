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

    # F22 facade contract: __all__ advertises only dependency-present real
    # names, resolved via find_spec WITHOUT importing the foreign packages.
    assert set(module.__all__) <= {"cards", "cardtree", "frontier"}
    assert "IPython" not in sys.modules
    assert "jupyter_client" not in sys.modules


def test_notebook_attribute_access_reports_missing_dependency() -> None:
    """Unknown attribute access teaches AttributeError-lineage regardless of deps."""
    _drop_module("torchlens.notebook")

    module = importlib.import_module("torchlens.notebook")

    with patch.dict("sys.modules", {"IPython": None}):
        # F22 facade contract: unknown names never mask as ImportError --
        # dependency reporting belongs to KNOWN names only.
        with pytest.raises(AttributeError, match=r"torchlens\.notebook"):
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

    # F22 facade contract: __all__ advertises only dependency-present real
    # names, resolved via find_spec WITHOUT importing the foreign packages.
    assert set(module.__all__) <= {"datasets", "rdms"}
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules


def test_neuro_attribute_access_reports_missing_dependency() -> None:
    """Known-name access with the dependency absent names the missing package."""
    pytest.importorskip("rsatoolbox")
    _drop_module("torchlens.neuro")

    module = importlib.import_module("torchlens.neuro")

    from torchlens._errors import MissingDependencyError

    with patch.dict("sys.modules", {"rsatoolbox": None}):
        # F22 teaching refusal: a KNOWN name whose optional dependency cannot
        # import refuses typed, naming the package (AttributeError never masks
        # a real missing dependency; ImportError-shaped masking is the old
        # pre-F22 contract).
        with pytest.raises(MissingDependencyError, match="rsatoolbox"):
            module.datasets


def test_neuro_attribute_access_when_deps_present() -> None:
    """When deps are installed, attribute access raises AttributeError, not ImportError."""
    pytest.importorskip("rsatoolbox")
    pytest.importorskip("brainscore_core")
    _drop_module("torchlens.neuro")

    module = importlib.import_module("torchlens.neuro")

    with pytest.raises(AttributeError):
        module.anything
