"""Optional capture hooks the TensorFlow eager path calls around its forward.

Each hook belongs to another private module (random-state capture in
``backends/_rng_capture``, native backward capture in
``backends/_backward_native``, variable-read verification in
``tf/_variable_reads``). The capture path calls a hook only when its module is
importable and, for the first two, when the matching capture option is on.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import Any

_RNG_PACKAGE = "torchlens.backends._rng_capture"
_BACKWARD_MODULE = "torchlens.backends._backward_native.tf"
_VARIABLE_READS_MODULE = "torchlens.backends.tf._variable_reads"


def _optional_module(name: str) -> ModuleType | None:
    """Import ``name`` or return ``None`` when that module (or its package) is absent.

    An import error raised from inside the module itself is not hidden.
    """

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing and name.startswith(missing):
            return None
        raise


def _rng_hooks() -> ModuleType | None:
    """The TensorFlow random-state hook module, when installed."""

    package = _optional_module(_RNG_PACKAGE)
    if package is None:
        return None
    module: ModuleType = package.rng_module("tf")
    return module


@contextmanager
def forward_hooks(
    session: Any, *, save_rng: bool, backward_ready: bool
) -> Iterator[dict[str, Any]]:
    """Run the eager forward inside the random-state and backward scopes it asked for.

    Parameters
    ----------
    session
        The ``TFEagerCaptureSession`` about to run.
    save_rng
        Whether random draws are logged for replay (``rng_scope``).
    backward_ready
        Whether a persistent tape is held for later backward passes
        (``backward_ready_scope``).

    Yields
    ------
    dict[str, Any]
        What the scopes leave for :func:`after_forward`.
    """

    state: dict[str, Any] = {}
    with ExitStack() as stack:
        rng = _rng_hooks() if save_rng else None
        if rng is not None:
            stack.enter_context(rng.rng_scope(session))
        backward = _optional_module(_BACKWARD_MODULE) if backward_ready else None
        if backward is not None:
            stack.enter_context(backward.backward_ready_scope(session))
        reads = _optional_module(_VARIABLE_READS_MODULE)
        if reads is not None:
            scope = reads.variable_read_scope(session.tf, session.module_tree)
            state["variable_reads"] = (reads, stack.enter_context(scope))
        yield state


def after_forward(trace: Any, session: Any, state: dict[str, Any], *, backward_ready: bool) -> None:
    """Keep what validation and later backward passes need, while the stream is still live."""

    if "variable_reads" in state:
        reads, ledger = state["variable_reads"]
        reads.attach_variable_reads(trace, ledger)
    if not backward_ready:
        return
    backward = _optional_module(_BACKWARD_MODULE)
    if backward is not None:
        backward.retain_backward_state(trace, session)


__all__ = ("after_forward", "forward_hooks")
