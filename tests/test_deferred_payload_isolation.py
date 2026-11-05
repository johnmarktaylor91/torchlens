"""A failed capture's dead trace must not break a later test's ``torch.compile``.

A trace that dies inside a reference cycle keeps its deferred-payload aliases
registered until the cyclic collector runs. Dynamo guards on that registry
when it traces TorchLens's torch wrappers, so a collection during a later
compile removed a guarded key and the compile failed with
``InternalTorchDynamoError: KeyError``. Running the failed-capture test right
before the compile-count test reproduced it on every run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from torchlens._utils import _torch_compat

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STANCE_FILE = "tests/test_compile_set_stance.py"


@pytest.mark.heavy
@pytest.mark.skipif(not _torch_compat.HAS_SET_STANCE, reason="needs torch >= 2.6")
def test_compile_after_a_failed_capture_in_the_same_process() -> None:
    """The failed-capture test followed by the compile-count test passes."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{_STANCE_FILE}::test_failed_capture_restores_force_eager_stance",
            f"{_STANCE_FILE}::test_count_compiles_verifies_the_coexistence_contract",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
