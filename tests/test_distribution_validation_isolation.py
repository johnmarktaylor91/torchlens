"""``torch.compile`` must not switch distribution argument checks off for later tests.

Creating any ``torch.compile`` callable sets the process-wide default of
``torch.distributions.Distribution`` argument validation to off. A test module
that compiled at import therefore changed every later test in the process: a
policy model whose forward builds a distribution lost its batch-wide
``_is_all_true`` check, so its chunked capture ran instead of refusing, and the
outcome depended on which files were collected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from _state_isolation import distribution_validation_restored
from torch.distributions import Distribution

from torchlens._utils._torch_compat import HAS_SET_STANCE

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_compile_stance_module_keeps_distribution_validation() -> None:
    """Collecting the compile-stance tests leaves distribution validation as found.

    On a torch without ``torch.compiler.set_stance`` the module refuses at import
    with pytest's module-level skip, before its import-time compile; the probe
    records that outcome and still checks validation, so both torches assert the
    isolation property.
    """

    probe = (
        "import importlib\n"
        "from _pytest.outcomes import Skipped\n"
        "from torch.distributions import Distribution\n"
        "before = Distribution._validate_args\n"
        "try:\n"
        "    importlib.import_module('tests.test_compile_set_stance')\n"
        "except Skipped:\n"
        "    print('skipped-at-import')\n"
        "else:\n"
        "    print('imported')\n"
        "print(before, Distribution._validate_args)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    outcome, before, after = result.stdout.split()[-3:]
    assert outcome == ("imported" if HAS_SET_STANCE else "skipped-at-import"), outcome
    assert after == before, f"importing the module changed validation from {before} to {after}"


def test_compile_inside_a_restored_block_leaves_validation_as_found() -> None:
    """The isolation helper puts the validation default back after a compile."""

    incoming = Distribution._validate_args
    with distribution_validation_restored():
        torch.compile(lambda t: torch.tanh(t), backend="eager")
    assert Distribution._validate_args is incoming
