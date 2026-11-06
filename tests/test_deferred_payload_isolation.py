"""A failed capture must free its partial trace by refcount.

A failed capture's escaping error, its traceback frames and the capture's run
state formed reference cycles, so the partial trace and its deferred-payload
aliases stayed registered until the cyclic collector ran. Dynamo guards on
that registry when it traces TorchLens's torch wrappers, so a collection
during a later compile removed a guarded key and the compile failed with
``InternalTorchDynamoError: KeyError``. Running the failed-capture test right
before the compile-count test reproduced it on every run.

Any trace collected during a compile removed keys the same way, so the wrappers now read
a single armed flag that Dynamo can guard, and the copy-out before a write runs behind
``torch.compiler.disable``. The tests below rebuild the wrap order in which Dynamo inlines
the wrappers (its rule tables built before the wrap) and drop a trace mid-compile.
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
from _forward_stash import stashed
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat, tensor_utils
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


class _Stash(nn.Module):
    """Keep a handle on an op output, so the caller can write to a saved payload's storage."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stash ``relu(fc(x)) * 2`` on the module and return it plus one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The stashed tensor plus one.
        """

        z = torch.relu(self.fc(x)) * 2
        self.stash = z
        return z + 1


def _rewrap_after_dynamo() -> None:
    """Re-wrap torch after Dynamo's rule tables are built, so Dynamo inlines the wrappers."""

    pytest.importorskip("torch._dynamo.trace_rules")
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    unwrap_torch()
    wrap_torch()


def _no_grad_capture() -> tl.Trace:
    """Capture ``_Stash`` under ``no_grad``, so its saved payloads are pending aliases."""

    with torch.no_grad():
        trace = tl.trace(_Stash(), torch.randn(2, 4))
    assert tensor_utils._DEFER_PENDING or not tensor_utils._DEFER_ENABLED
    return trace


@pytest.mark.usefixtures("dynamo_free_process")  # these captures must defer
def test_compile_survives_a_trace_collected_mid_compile() -> None:
    """A trace whose pending aliases die between tracing and guard building breaks no compile."""

    _rewrap_after_dynamo()
    holder: dict[str, object] = {}

    def dropping_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> object:
        holder.clear()
        gc.collect()
        return gm.forward

    for _ in range(5):
        torch.compiler.reset()
        holder["trace"] = _no_grad_capture()
        x = torch.randn(3)
        out = torch.compile(lambda t: torch.relu(t) + 1, backend=dropping_backend)(x)
        assert torch.equal(out, torch.relu(x) + 1)
        assert not holder
    torch.compiler.reset()


@pytest.mark.usefixtures("dynamo_free_process")  # these captures must defer
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_inplace_write_copies_out_pending_payloads(backend: str) -> None:
    """A compiled in-place torch function on a saved payload's storage keeps the trace intact."""

    _rewrap_after_dynamo()
    torch.compiler.reset()
    trace = _no_grad_capture()
    live = stashed(trace, "stash")
    labels = [label for label in trace.layer_labels if isinstance(trace[label].out, torch.Tensor)]
    saved = {label: trace[label].out.clone() for label in labels}
    expected = torch.exp(live)

    torch.compile(lambda t: torch.exp_(t), backend=backend)(live)

    assert torch.equal(live, expected)
    changed = [label for label in labels if not torch.equal(saved[label], trace[label].out)]
    assert not changed, f"a compiled in-place write changed saved payloads: {changed}"
    torch.compiler.reset()
