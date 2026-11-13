"""Capture hooks other lanes implement for Paddle: random-draw logging and native backward.

Each hook module is looked up by name and called only when it imports, so the
Paddle capture runs unchanged until its owner lands (backend parity wave B
contract): ``backends/_rng_capture/paddle.py`` (``rng_scope``) and
``backends/_backward_native/paddle.py`` (``backward_ready_scope``,
``retain_backward_state``).

The hooks share one capture session, a plain namespace holding the trace under
construction, so the owners can attach their state without touching declared
``Trace`` fields.
"""

from __future__ import annotations

import importlib
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from typing import Any

RNG_HOOK_MODULE = "torchlens.backends._rng_capture.paddle"
BACKWARD_HOOK_MODULE = "torchlens.backends._backward_native.paddle"


def optional_hook_module(name: str) -> ModuleType | None:
    """Import an owner lane's hook module, or return ``None`` when it does not exist.

    Parameters
    ----------
    name
        Dotted module name.

    Returns
    -------
    ModuleType | None
        The module, or ``None`` when the module (or its package) is absent. A
        missing dependency inside an existing module still raises.
    """

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            return None
        raise


class PaddleCaptureHooks:
    """The optional hook calls of one Paddle capture.

    Parameters
    ----------
    tensor_store
        The backend's label store; its ``backward_scope`` notes every labelled
        tensor (and so its grad node) while the backward scope is open.
    trace
        The trace under construction.
    backward_ready
        Whether the capture arms backward recording.
    """

    def __init__(self, tensor_store: Any, trace: Any, *, backward_ready: bool) -> None:
        self._store = tensor_store
        self._backward_ready = backward_ready
        self._stack = ExitStack()
        self._backward: ModuleType | None = None
        self.session = SimpleNamespace(trace=trace)

    def open(self) -> None:
        """Enter the hook scopes that exist; call after the wrappers are installed."""

        rng = optional_hook_module(RNG_HOOK_MODULE)
        if rng is not None:
            self._stack.enter_context(rng.rng_scope(self.session))
        if not self._backward_ready:
            return
        self._backward = optional_hook_module(BACKWARD_HOOK_MODULE)
        if self._backward is not None:
            scope = self._stack.enter_context(self._backward.backward_ready_scope(self.session))
            self._store.backward_scope = scope

    def see_output(self, output: object) -> None:
        """Hand the model's live return value to the open backward scope."""

        scope = self._store.backward_scope
        if scope is not None:
            scope.output = output

    def retain(self, trace: Any) -> None:
        """Keep backward state on the materialized trace and stop noting tensors."""

        self._store.backward_scope = None
        if self._backward is not None:
            self._backward.retain_backward_state(trace, self.session)

    def close(self) -> None:
        """Leave every open hook scope; safe to call more than once."""

        self._store.backward_scope = None
        self._stack.close()
