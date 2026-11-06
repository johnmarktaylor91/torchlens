"""Compiled in-place writes never reach a deferred saved payload.

A no-grad capture saves payloads as deferred aliases of the live activations
(clone-on-write): the copy is made only when a TorchLens torch wrapper sees a
write to a pending storage. Code compiled by ``torch.compile`` on the default
Inductor backend keeps an in-place write to an input inside its generated
kernel, so at execution time no wrapper runs. Before the compiled-write guard,
such a write silently replaced the saved payload with the written bytes.

Covered here, for every in-place spelling (method, ``torch.*_``, functional
``inplace=True`` and ``__setitem__``):

* compile after the capture: the compiler traces the write through the wrapper
  on tracing tensors, which materializes every pending alias first;
* compile before the capture (the guard cache hits, no compile event at all):
  a capture made once Dynamo is loaded keeps eager clones;
* a fresh process where Dynamo is first imported after the capture;
* ``unwrap_torch`` materializes pending aliases, since unwrapped writes go unseen.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from _forward_stash import stashed
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._utils import _compiled_write_guard, _torch_compat, tensor_utils
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

pytestmark = [
    pytest.mark.skipif(
        not _torch_compat.get_torch_cpp_extension_support(),
        reason="HAS_TORCH_CPP_EXTENSION is False: inductor's CPU backend cannot import "
        "torch.utils.cpp_extension on this install",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SPELLINGS: dict[str, Callable[[torch.Tensor], object]] = {
    "tensor_add_": lambda t: t.add_(1000.0),
    "torch_relu_": lambda t: torch.relu_(t),
    "functional_relu_inplace": lambda t: F.relu(t, inplace=True),
    "setitem": lambda t: t.__setitem__(0, 77.0),
}


class _Stash(nn.Module):
    """Keep a live reference to an activation whose payload the capture saves."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stash ``fc(x) * 2`` (mixed signs, so ``relu_`` changes it) and return it plus one.

        Parameters
        ----------
        x:
            Input batch of shape ``(2, 4)``.

        Returns
        -------
        torch.Tensor
            The stashed activation plus one.
        """

        stash = self.fc(x) * 2
        self.stash = stash
        return stash + 1


@contextmanager
def _dynamo_not_loaded() -> Iterator[None]:
    """Make the guard read Dynamo as not imported, as in a fresh process.

    Yields
    ------
    None
        Control while the guard reports no Dynamo.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_compiled_write_guard, "compiled_writes_seen", lambda: False)
        patch.setattr(_compiled_write_guard, "compiled_writes_possible", lambda: False)
        yield


def _is_pending(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor``'s storage still backs a deferred payload alias.

    Parameters
    ----------
    tensor:
        A live activation.

    Returns
    -------
    bool
        ``True`` when a saved payload still aliases this storage.
    """

    with tensor_utils._paused_internal_reads():
        key = tensor_utils._deferred_storage_key(tensor)
    return key is not None and bool(tensor_utils._DEFER_PENDING.get(key))


def _capture() -> tuple[tl.Trace, torch.Tensor, dict[str, torch.Tensor]]:
    """Capture ``_Stash`` without grad; return the trace, live stash and saved copies.

    Returns
    -------
    tuple
        The trace, the live stashed activation and a clone of every saved
        tensor payload keyed by layer label.
    """

    torch.manual_seed(0)
    model = _Stash()
    with torch.no_grad():
        trace = tl.trace(model, torch.randn(2, 4))
    live = stashed(trace, "stash")
    saved = {
        label: trace[label].out.clone()
        for label in trace.layer_labels
        if isinstance(trace[label].out, torch.Tensor)
    }
    assert any(torch.equal(value, live) for value in saved.values())
    assert bool((live < 0).any()) and bool((live > 0).any())
    return trace, live, saved


def _assert_payloads_unchanged(trace: tl.Trace, saved: dict[str, torch.Tensor]) -> None:
    """Assert every saved payload still equals its value before the compiled write.

    Parameters
    ----------
    trace:
        The capture under test.
    saved:
        Clones of its saved payloads taken right after capture.
    """

    changed = [label for label, value in saved.items() if not torch.equal(trace[label].out, value)]
    assert changed == []


@pytest.mark.heavy
@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_write_compiled_after_capture_leaves_saved_payload_intact(spelling: str) -> None:
    """Compiling the write traces it through the wrapper, which copies pending aliases out."""

    torch.compiler.reset()
    with _dynamo_not_loaded():
        trace, live, saved = _capture()
        assert _is_pending(live), (
            "the capture must defer the payload for this test to mean anything"
        )
    compiled = torch.compile(_SPELLINGS[spelling])
    version = live._version
    compiled(live)
    assert live._version > version
    _assert_payloads_unchanged(trace, saved)


@pytest.mark.heavy
@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_write_compiled_before_capture_leaves_saved_payload_intact(spelling: str) -> None:
    """A guard-cache hit runs the kernel with no compile event; the capture must not defer."""

    torch.compiler.reset()
    compiled = torch.compile(_SPELLINGS[spelling])
    torch.manual_seed(1)
    compiled(torch.randn(2, 8) * 2)
    trace, live, saved = _capture()
    assert not _is_pending(live)
    compiled(live)
    _assert_payloads_unchanged(trace, saved)


def test_capture_with_dynamo_loaded_materializes_earlier_pending_payloads() -> None:
    """The first capture that sees Dynamo loaded copies out what earlier captures left pending."""

    import torch._dynamo  # noqa: F401  (already loaded by the compiles above; explicit here)

    with _dynamo_not_loaded():
        _, earlier_live, _ = _capture()
        assert _is_pending(earlier_live)
    _, live, _ = _capture()
    assert not _is_pending(live)
    assert not _is_pending(earlier_live)


def test_unwrap_torch_materializes_pending_payloads() -> None:
    """Writes after ``unwrap_torch`` are not intercepted, so no alias may stay pending."""

    escape_mode = _state._escape_detector_mode
    witness_mode = _state._completeness_witness_mode
    with _dynamo_not_loaded():
        trace, live, saved = _capture()
        assert _is_pending(live)
        unwrap_torch()
        try:
            assert not _is_pending(live)
            live.add_(1000.0)
            _assert_payloads_unchanged(trace, saved)
        finally:
            wrap_torch(escape_detector=escape_mode, completeness_witness=witness_mode)


_FRESH_PROCESS_SCRIPT = textwrap.dedent(
    """
    import sys

    import torch
    import torch.nn.functional as F
    from torch import nn

    import torchlens as tl
    from torchlens._utils import tensor_utils

    class Stash(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            stash = self.fc(x) * 2
            self.stash = stash
            return stash + 1

    spellings = {
        "tensor_add_": lambda t: t.add_(1000.0),
        "torch_relu_": lambda t: torch.relu_(t),
        "functional_relu_inplace": lambda t: F.relu(t, inplace=True),
        "setitem": lambda t: t.__setitem__(0, 77.0),
    }
    captures = []
    for name in spellings:
        torch.manual_seed(0)
        model = Stash()
        with torch.no_grad():
            trace = tl.trace(model, torch.randn(2, 4))
        saved = {k: trace[k].out.clone() for k in trace.layer_labels
                 if isinstance(trace[k].out, torch.Tensor)}
        live = trace.modules["self"].attribute_writes["stash"].value
        captures.append((name, trace, live, saved))
    assert "torch._dynamo" not in sys.modules, "the capture must run before Dynamo loads"
    assert len(tensor_utils._DEFER_PENDING) >= len(spellings), "the captures must defer"
    for name, trace, live, saved in captures:
        torch.compile(spellings[name])(live)
        changed = [k for k, v in saved.items() if not torch.equal(trace[k].out, v)]
        print(name, "changed", changed)
        assert changed == [], name
    print("ok")
    """
)


@pytest.mark.slow
def test_fresh_process_compiled_writes_leave_saved_payloads_intact() -> None:
    """End to end in a process where Dynamo is first imported after the captures."""

    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")
