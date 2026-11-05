"""A failed capture must free its partial trace by refcount.

A failed capture's escaping error, its traceback frames and the capture's run
state formed reference cycles, so the partial trace and its deferred-payload
aliases stayed registered until the cyclic collector ran. Dynamo guards on
that registry when it traces TorchLens's torch wrappers, so a collection
during a later compile removed a guarded key and the compile failed with
``InternalTorchDynamoError: KeyError``. Running the failed-capture test right
before the compile-count test reproduced it on every run.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import warnings
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat
from torchlens.errors import TraceCaptureError

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


class _SinThenRaise(nn.Module):
    """Record one op, then fail the forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``sin`` and raise.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This path never returns.

        Raises
        ------
        RuntimeError
            Always, after the first op.
        """

        _ = torch.sin(x)
        raise RuntimeError("injected forward failure")


def _failed_capture_trace_ref() -> weakref.ref[object]:
    """Fail a capture, drop its error, and return a weak reference to its trace.

    Returns
    -------
    weakref.ref
        Reference to the partial trace's underlying ``Trace``.
    """

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            tl.trace(_SinThenRaise(), torch.randn(2, 4))
        except TraceCaptureError as err:
            return weakref.ref(err.partial_trace.trace)
    raise AssertionError("the forward was expected to fail")


def test_failed_capture_partial_trace_is_freed_by_refcount() -> None:
    """Dropping a failed capture's error frees its partial trace without the collector."""

    gc.collect()
    gc.disable()
    try:
        trace_ref = _failed_capture_trace_ref()
        assert trace_ref() is None, "the partial trace outlived its error (a reference cycle)"
    finally:
        gc.enable()
