"""Autocast (and grad/inference mode) state capture and re-entry for replay.

Rides in the rng package because replay restores RNG streams and autocast state in one
bracket (:func:`torchlens._utils.rng.restore.execute_with_restored_rng_autocast`).
"""

from __future__ import annotations

from types import (
    TracebackType,
)
from typing import Any, cast

import torch

from .._torch_compat import (
    autocast_get_dtype,
    autocast_is_enabled,
)

_AUTOCAST_DEVICES = ("cpu", "cuda")


def log_current_autocast_state() -> dict[str, dict[str, Any]]:
    """Capture the current ``torch.amp.autocast`` enabled/dtype state.

    Checked for each device in :data:`_AUTOCAST_DEVICES`.  If a device
    doesn't support autocast queries, it is silently skipped.

    Returns:
        Dict mapping device name to ``{"enabled": bool, "dtype": torch.dtype}``.
    """
    state: dict[str, dict[str, Any]] = {}
    # r35 corr2_8: record the grad/inference execution mode alongside autocast in the
    # same per-op KEEP field, under a reserved non-device key. ``enabled`` stays False
    # so every autocast consumer (AutocastRestore) skips it structurally.
    state["__execution__"] = {
        "enabled": False,
        "dtype": None,
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode": bool(torch.is_inference_mode_enabled()),
    }
    for device in _AUTOCAST_DEVICES:
        try:
            # Routed through the version-neutral shim so TorchLens runs on
            # torch 2.1+ (the per-device ``device_type`` argument to these
            # query helpers is torch 2.4+ only). On torch>=2.4 the shim calls
            # ``torch.is_autocast_enabled``/``torch.get_autocast_dtype``
            # directly (identical behavior); on torch 2.1-2.3 it routes to the
            # legacy per-device helpers. See _utils/_torch_compat.py.
            state[device] = {
                "enabled": autocast_is_enabled(device),
                "dtype": autocast_get_dtype(device),
            }
        except (RuntimeError, TypeError):
            # Device doesn't support autocast queries (e.g. no CUDA).
            pass
    return state


class AutocastRestore:
    """Context manager that re-enters saved autocast contexts during replay.

    Only devices that were *enabled* at capture time get an autocast
    context opened.  Contexts are exited in reverse order on ``__exit__``.

    Usage::

        with AutocastRestore(saved_state):
            result = func(*args, **kwargs)
    """

    __slots__ = ("_autocast_state", "_contexts")

    def __init__(self, autocast_state: dict[str, dict[str, Any]]) -> None:
        """Store serialized autocast state for later context restoration.

        Parameters
        ----------
        autocast_state:
            Mapping from device type to captured autocast enabled/dtype state.
        """

        self._autocast_state = autocast_state
        self._contexts: list[Any] = []

    def __enter__(self) -> AutocastRestore:
        """Enter captured autocast contexts, unwinding fully if any entry fails.

        Python never calls ``__exit__`` when ``__enter__`` raises, so a multi-device
        replay whose second device fails (an artifact-controlled ``dtype``, an
        unsupported device) used to leave the FIRST device's autocast entered
        thread-globally for the rest of the process -- every later user op silently
        running under an autocast nobody asked for. The unwind arm is
        ``BaseException``-wide because a Ctrl-C between two entries leaves exactly the
        same unbalanced nesting.

        Returns
        -------
        AutocastRestore
            This context manager instance.
        """

        try:
            self._enter_contexts()
        except BaseException:
            self._exit_contexts(None, None, None)
            raise
        return self

    def _enter_contexts(self) -> None:
        """Open one autocast context per captured device entry."""

        for device, state in self._autocast_state.items():
            if device.startswith("__"):
                # Reserved non-device entries (e.g. ``__execution__`` grad/inference
                # mode) are not autocast device records and open no context here.
                continue
            # Open an autocast context for EVERY captured device -- including
            # devices that were DISABLED at capture time. Previously a saved
            # ``enabled=False`` device opened NO context, so if the replay caller
            # had live autocast enabled for that device the replayed op silently
            # ran under the caller's autocast (wrong dtype / arithmetic, returned
            # normally). Opening an explicit ``enabled=False`` context shields the
            # replay against the caller's live state, reproducing the captured
            # autocast posture exactly.
            autocast = cast(Any, getattr(torch.amp, "autocast"))
            ctx = autocast(device, dtype=state["dtype"], enabled=bool(state["enabled"]))
            ctx.__enter__()
            self._contexts.append(ctx)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit restored autocast contexts in reverse nesting order.

        Parameters
        ----------
        exc_type:
            Exception type propagated by the managed block, if any.
        exc_value:
            Exception instance propagated by the managed block, if any.
        traceback:
            Traceback propagated by the managed block, if any.
        """

        self._exit_contexts(exc_type, exc_value, traceback)

    def _exit_contexts(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit every opened context in reverse nesting order, guarding each one.

        A raise from the innermost ``__exit__`` used to skip every OUTER context,
        leaving ``autocast_increment_nesting`` permanently unbalanced for the thread.
        Each exit is fenced independently (the same per-item pattern the monitor's
        restore queue uses) and the list is cleared so a repeat exit is a no-op; the
        FIRST failure is re-raised once every context has been given its chance.
        """

        contexts = self._contexts
        self._contexts = []
        first_error: BaseException | None = None
        for ctx in reversed(contexts):
            try:
                ctx.__exit__(exc_type, exc_value, traceback)
            except BaseException as error:  # noqa: PERF203 - per-item fence is the point
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
