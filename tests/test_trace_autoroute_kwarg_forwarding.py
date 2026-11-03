"""Autoroute kwarg-forwarding pins: no public trace option silently vanishes.

The autoroute detector branch of :func:`torchlens.trace` forwards the public
keyword bundle into detectors (which re-enter ``tl.trace`` through bridge
helpers such as ``bridge.transformers.trace_text``). Any ``trace()`` parameter missing
from that bundle is SILENTLY DROPPED on autoroute dispatch — the worst API
failure mode: the user configured something and nothing complained. Found
live 2026-08-19: ``tl.trace(hf_model, "text", structure_only=True)`` returned
a full value-bearing capture with ``structure_only=False``; ``episode=`` was
likewise discarded without its typed torch-only refusal.

Two pins:

1. LOCKSTEP — the ``autoroute_kwargs`` literal in ``user_funcs.trace`` must
   cover every ``trace()`` parameter except the closed, documented set that
   is provably guarded before the autoroute branch runs. A new public trace
   parameter that skips the bundle goes red here.
2. BEHAVIORAL — a recorder detector actually receives the once-dropped keys.
"""

from __future__ import annotations

import ast

import pytest
import torch
from _source_corpus import PACKAGE_ROOT, package_ast
from torch import nn

import torchlens as tl
from torchlens import _autoroute as autoroute

#: trace() parameters legitimately absent from the autoroute bundle. Every
#: entry names the guard that makes the omission safe; widening this set
#: requires the same style of proof.
AUTOROUTE_EXEMPT_PARAMS = {
    # the dispatch subjects themselves
    "model",
    "input_args",
    "input_kwargs",
    # autoroute only runs when backend is None
    "backend",
    # chunk_paths without CaptureOptions(chunk_size=) raises before the branch
    "chunk_paths",
    # quickstart input ladder (F17): any non-None input_size returns via
    # _trace_via_ladder before the autoroute branch, so autoroute only ever
    # runs with input_shape=None (the ladder re-enters trace() with a real
    # input, which forwards normally)
    "input_shape",
}


def _trace_function_def() -> ast.FunctionDef:
    """The public ``trace`` FunctionDef from the parsed package corpus."""

    tree = package_ast(PACKAGE_ROOT / "_user_funcs.py")
    candidates = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "trace"
    ]
    assert candidates, "public trace() not found in _user_funcs.py"
    return candidates[-1]


def _autoroute_forwarded_names() -> set[str]:
    """The ``_AUTOROUTE_FORWARDED_KWARGS`` tuple of the PREPARE dispatch module."""

    tree = package_ast(PACKAGE_ROOT / "_capture" / "prepare" / "dispatch.py")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "_AUTOROUTE_FORWARDED_KWARGS" for target in node.targets
        ):
            assert isinstance(node.value, ast.Tuple)
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    raise AssertionError("_AUTOROUTE_FORWARDED_KWARGS tuple not found in prepare/dispatch.py")


def test_autoroute_bundle_covers_every_unguarded_trace_param() -> None:
    """Every trace() parameter is forwarded to detectors or provably guarded."""

    node = _trace_function_def()
    params = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
    bundle_keys = _autoroute_forwarded_names()

    dropped = params - bundle_keys - AUTOROUTE_EXEMPT_PARAMS
    assert not dropped, (
        f"trace() parameters silently dropped on the autoroute path: "
        f"{sorted(dropped)}. Forward them in _AUTOROUTE_FORWARDED_KWARGS (detectors "
        f"re-enter tl.trace, which honors or refuses them typed) or add a "
        f"guard that provably fires before the autoroute branch and document "
        f"it in AUTOROUTE_EXEMPT_PARAMS."
    )
    stale = AUTOROUTE_EXEMPT_PARAMS - params
    assert not stale, f"AUTOROUTE_EXEMPT_PARAMS names vanished trace() params: {sorted(stale)}"


class _Identity(nn.Module):
    """Minimal model for the recorder-detector run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a trivial tensor expression."""

        return x + 1


@pytest.mark.smoke
def test_detectors_receive_the_once_dropped_kwargs() -> None:
    """A registered detector sees capture/episode/stop_after in kwargs."""

    received: dict[str, object] = {}

    def recorder(model: object, payload: object, **kwargs: object) -> None:
        """Record the forwarded bundle and decline the dispatch."""

        received.update(kwargs)
        return None

    with autoroute.input.snapshot():
        autoroute.input.register(name="test_recorder", priority=-1000)(recorder)
        log = tl.trace(_Identity(), torch.ones(2), save="none")
    assert log is not None
    for key in ("capture", "episode", "stop_after"):
        assert key in received, (
            f"autoroute detectors no longer receive {key!r}: it is being "
            f"silently dropped on the autoroute dispatch path again"
        )
    # structure_only travels inside the grouped capture object now.
    capture = received["capture"]
    assert capture is not None
    assert hasattr(capture, "structure_only")
