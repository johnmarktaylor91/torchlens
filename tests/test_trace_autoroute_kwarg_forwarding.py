"""Autoroute kwarg-forwarding pins: no public trace option silently vanishes.

The autoroute detector branch of :func:`torchlens.trace` forwards the public
keyword bundle into detectors (which re-enter ``tl.trace`` through bridge
helpers such as ``bridge.hf.trace_text``). Any ``trace()`` parameter missing
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
from torchlens import autoroute

pytestmark = pytest.mark.smoke


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
    # chunked forwards skip autoroute (chunk_size gate; chunk_paths without
    # chunk_size raises before the branch)
    "chunk_size",
    "chunk_paths",
    # explicit backend-only options: raise before autoroute when backend is
    # None
    "jax_static_argnums",
    "grad_options",
}


def _trace_function_def() -> ast.FunctionDef:
    """The public ``trace`` FunctionDef from the parsed package corpus."""

    tree = package_ast(PACKAGE_ROOT / "user_funcs.py")
    candidates = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "trace"
    ]
    assert candidates, "public trace() not found in user_funcs.py"
    return candidates[-1]


def test_autoroute_bundle_covers_every_unguarded_trace_param() -> None:
    """Every trace() parameter is forwarded to detectors or provably guarded."""

    node = _trace_function_def()
    params = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
    bundle_keys: set[str] | None = None
    for inner in ast.walk(node):
        if isinstance(inner, ast.Assign) and any(
            getattr(target, "id", "") == "autoroute_kwargs" for target in inner.targets
        ):
            assert isinstance(inner.value, ast.Dict)
            bundle_keys = {
                key.value
                for key in inner.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            break
    assert bundle_keys is not None, "autoroute_kwargs literal not found in trace()"

    dropped = params - bundle_keys - AUTOROUTE_EXEMPT_PARAMS
    assert not dropped, (
        f"trace() parameters silently dropped on the autoroute path: "
        f"{sorted(dropped)}. Forward them in autoroute_kwargs (detectors "
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


def test_detectors_receive_the_once_dropped_kwargs() -> None:
    """A registered detector sees structure_only/episode/grouping in kwargs."""

    received: dict[str, object] = {}

    def recorder(model: object, payload: object, **kwargs: object) -> None:
        """Record the forwarded bundle and decline the dispatch."""

        received.update(kwargs)
        return None

    with autoroute.input.snapshot():
        autoroute.input.register(name="test_recorder", priority=-1000)(recorder)
        log = tl.trace(
            _Identity(), torch.ones(2), capture=tl.options.CaptureOptions(layers_to_save="none")
        )
    assert log is not None
    for key in ("capture", "episode", "grouping"):
        assert key in received, (
            f"autoroute detectors no longer receive {key!r}: it is being "
            f"silently dropped on the autoroute dispatch path again"
        )
    # structure_only travels inside the grouped capture object now.
    capture = received["capture"]
    assert capture is not None
    assert hasattr(capture, "structure_only")
