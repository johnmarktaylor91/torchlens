"""Entry-point phase audits (F35; memo item 11 + composition row 9).

Two obligations the merged test plan names that no shipped suite proves:

- **NO-CODE-ON-ATTRIBUTE-EXISTENCE**: with an installed distribution
  declaring ``torchlens.recipes``/``torchlens.appliances`` entry points
  whose provider module writes a sentinel at import, a fresh interpreter
  may ``import torchlens``, sweep ``hasattr``/``dir()`` across every lazy
  name, and run a real capture -- and the sentinel stays ABSENT.
  ``hasattr``, ``dir()``, IDE sweeps, and agent surface walks cannot
  express consent to execute installed code.
- **ARTIFACT LOAD NEVER IMPORTS A PROVIDER**: saving and loading a real
  ``.tlspec`` artifact with the provider installed leaves the sentinel
  ABSENT; typed refusal or declared analysis-only degrade is the only
  legal missing-provider behavior.

The positive arm keeps both negatives honest: explicit activation through
:func:`torchlens.ecosystem.plugins.activate` imports the provider and the
sentinel appears. Everything runs in ONE fresh subprocess (warm sweeps lie;
the in-process suite for discovery/activation mechanics is
tests/test_ecosystem_rt_plugins.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

REPO = Path(__file__).resolve().parent.parent

_PROVIDER_MODULE = '''
"""F35 sentinel provider: importing this module IS the detected event."""

import os
import pathlib

pathlib.Path(os.environ["F35_SENTINEL"]).write_text("imported")


def factory():
    """Zero-arg factory (shape irrelevant: the import already happened)."""

    return {"name": "f35_sentinel", "fn": lambda trace, path=None: None, "tier": "present"}
'''

_DRIVER = """
import os
import pathlib
import sys

sentinel = pathlib.Path(os.environ["F35_SENTINEL"])

import torch
import torchlens

# Phase 1: attribute existence executes nothing.
names = sorted(set(torchlens.__all__) | set(torchlens._LAZY_ATTRS))
for name in names:
    assert hasattr(torchlens, name), f"facade name vanished: {name}"
dir(torchlens)
assert not sentinel.exists(), "provider imported during hasattr/dir sweep"

# Phase 2: a real capture + artifact save/load imports no provider.
model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
trace = torchlens.trace(model, torch.randn(2, 4))
artifact = pathlib.Path(os.environ["F35_ARTIFACT"])
torchlens.save(trace, str(artifact))
loaded = torchlens.load(str(artifact))
assert loaded is not None
assert not sentinel.exists(), "provider imported during capture or artifact load"

# Phase 3 (the positive arm): explicit activation imports the provider.
from torchlens.ecosystem import plugins

plugins.activate("f35-sentinel", strict=False)
assert sentinel.exists(), "explicit activation failed to import the provider"
print("F35_ENTRY_AUDIT_PASS")
"""


def _plant_provider(site: Path) -> None:
    (site / "f35_sentinel_provider.py").write_text(
        textwrap.dedent(_PROVIDER_MODULE), encoding="utf-8"
    )
    info = site / "f35_sentinel-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: f35-sentinel\nVersion: 0.1.0\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        "[torchlens.recipes]\nf35recipe = f35_sentinel_provider:factory\n"
        "[torchlens.appliances]\nf35appl = f35_sentinel_provider:factory\n",
        encoding="utf-8",
    )


def test_attribute_existence_and_artifact_load_execute_no_provider(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    _plant_provider(site)
    sentinel = tmp_path / "sentinel.txt"
    env = dict(os.environ)
    env["F35_SENTINEL"] = str(sentinel)
    env["F35_ARTIFACT"] = str(tmp_path / "audit.tlspec")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop(  # a stale kill switch would make the positive arm vacuous
        "TORCHLENS_PLUGINS", None
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_DRIVER)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=REPO,
        env=env,
    )
    assert result.returncode == 0, (
        f"entry-point audit subprocess failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "F35_ENTRY_AUDIT_PASS" in result.stdout
