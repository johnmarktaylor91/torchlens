"""Tests for optional appliance subfolder imports."""

from __future__ import annotations

import importlib
import importlib.util
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
    """``import torchlens._notebook`` never imports its foreign extra deps.

    Package import must not touch ``IPython`` / ``jupyter_client`` as a side
    effect, regardless of whether the extra is installed -- the dependency
    check is deferred to first attribute access (see the security fix that
    closes a foreign-import-at-untrusted-bundle-load class).
    """
    _drop_module("torchlens._notebook")
    _drop_module("IPython")
    _drop_module("jupyter_client")

    module = importlib.import_module("torchlens._notebook")

    # F22 facade contract: __all__ advertises only dependency-present real
    # names, resolved via find_spec WITHOUT importing the foreign packages.
    assert set(module.__all__) <= {"cards", "cardtree", "frontier"}
    assert "IPython" not in sys.modules
    assert "jupyter_client" not in sys.modules


def test_notebook_attribute_access_reports_missing_dependency() -> None:
    """Unknown attribute access teaches AttributeError-lineage regardless of deps."""
    _drop_module("torchlens._notebook")

    module = importlib.import_module("torchlens._notebook")

    with patch.dict("sys.modules", {"IPython": None}):
        # F22 facade contract: unknown names never mask as ImportError --
        # dependency reporting belongs to KNOWN names only.
        with pytest.raises(AttributeError, match=r"torchlens\._notebook"):
            module.anything


def test_notebook_attribute_access_when_deps_present() -> None:
    """When deps are installed, attribute access raises AttributeError, not ImportError."""
    pytest.importorskip("IPython")
    pytest.importorskip("jupyter_client")
    _drop_module("torchlens._notebook")

    module = importlib.import_module("torchlens._notebook")

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

    # F22 facade contract: __all__ advertises exactly the dependency-present
    # real names, resolved via find_spec WITHOUT importing the foreign
    # packages. ``compare`` is numpy-only and the extraction-plan names need
    # no extra, so they are always present.
    expected = {"ExtractionPlan", "PlanSite", "compare", "export_npz", "plan_extraction"}
    if importlib.util.find_spec("rsatoolbox") is not None:
        expected |= {"datasets", "rdms"}
    if importlib.util.find_spec("brainscore_vision") is not None:
        expected |= {"activations_extractor", "get_activations_fn"}
    assert set(module.__all__) == expected
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules
    # Every real name stays lazy: the package import loads no implementation
    # module, the numpy-only ones included.
    for implementation in ("_compare", "_datasets", "_rdms"):
        assert f"torchlens.neuro.{implementation}" not in sys.modules


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
