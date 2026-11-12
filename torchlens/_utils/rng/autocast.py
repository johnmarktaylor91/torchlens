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
    AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED,
    autocast_get_dtype,
    autocast_is_enabled,
)
from ..container_tokens import TOKEN_INERT

_AUTOCAST_DEVICES = ("cpu", "cuda")

# The per-device queries, resolved once. On torch>=2.4 the shim functions only forward to
# these two torch callables, so calling them directly records the same values for two Python
# frames less per device on every logged call; torch 2.1 to 2.3 keep the legacy shims.
_IS_ENABLED: Any = torch.is_autocast_enabled if AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED else None
_GET_DTYPE: Any = torch.get_autocast_dtype if AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED else None


def log_current_autocast_state(trace: Any = None) -> dict[str, dict[str, Any]]:
    """Capture the current ``torch.amp.autocast`` enabled/dtype state.

    Checked for each device in :data:`_AUTOCAST_DEVICES`.  If a device
    doesn't support autocast queries, it is silently skipped.

    Args:
        trace: The capturing trace, when the caller has it. The snapshot is then
            registered with a content token in the trace's runtime workspace
            (``container_tokens``): the token holds every value the snapshot is
            built from, in build order, so two snapshots are equal exactly when
            their tokens are, and the freeze's container pooling keys the
            snapshot by it instead of walking it (``_trace_core.pools.ContainerTokens``).
            Nothing registers without a trace, on the torch 2.1 to 2.3 path, or
            once the trace's freeze has taken the table. A snapshot equal to one
            already registered under the same trace is that registered object
            (the freeze pools equal snapshots into one object anyway); the state
            itself is read live on every call.

    Returns:
        Dict mapping device name to ``{"enabled": bool, "dtype": torch.dtype}``.
    """
    grad_enabled = bool(torch.is_grad_enabled())
    inference_mode = bool(torch.is_inference_mode_enabled())
    is_enabled = _IS_ENABLED
    if is_enabled is None:
        state = _execution_only_state(grad_enabled, inference_mode)
        _legacy_device_states(state)
        return state
    get_dtype = _GET_DTYPE
    token: tuple[Any, ...] = (grad_enabled, inference_mode)
    for device in _AUTOCAST_DEVICES:
        try:
            # torch>=2.4: the same two calls the version-neutral shim makes.
            enabled = bool(is_enabled(device))
            dtype = get_dtype(device)
        except (RuntimeError, TypeError):
            # Device doesn't support autocast queries (e.g. no CUDA).
            continue
        token += (device, enabled, dtype)
    table = _token_table(trace)
    if table is not None:
        # The trace already holds a snapshot with this content: hand it out again. The
        # table entry proves the remembered snapshot was registered under THIS trace
        # (the table holds it, so its id is not reused while the entry exists).
        kept = _KEPT_SNAPSHOTS.get(token)
        if kept is not None:
            entry = table.get(id(kept))
            if entry is not None and entry[0] is kept:
                return kept
    state = _execution_only_state(grad_enabled, inference_mode)
    for index in range(2, len(token), 3):
        state[token[index]] = {"enabled": token[index + 1], "dtype": token[index + 2]}
    if table is not None:
        table[id(state)] = (state, token, TOKEN_INERT)
        _KEPT_SNAPSHOTS[token] = state
    return state


#: The latest registered snapshot per content token. A snapshot is only handed out again
#: under the trace whose token table holds it; the tokens are few (execution mode and the
#: per-device autocast settings), so the map stays small.
_KEPT_SNAPSHOTS: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}


def _execution_only_state(grad_enabled: bool, inference_mode: bool) -> dict[str, dict[str, Any]]:
    """Return a snapshot holding only the reserved ``__execution__`` entry.

    The grad/inference execution mode rides in the same per-op KEEP field as autocast,
    under a reserved non-device key. ``enabled`` stays False so every autocast consumer
    (AutocastRestore) skips it structurally.
    """

    return {
        "__execution__": {
            "enabled": False,
            "dtype": None,
            "grad_enabled": grad_enabled,
            "inference_mode": inference_mode,
        }
    }


def _token_table(trace: Any) -> dict[int, Any] | None:
    """Return ``trace``'s container-token table, creating it; ``None`` when nothing registers."""

    if trace is None:
        return None
    try:
        workspace = trace._wrapper_runtime_ws
    except AttributeError:
        return None
    table = workspace.container_tokens
    if table is None:
        table = workspace.container_tokens = {}
    elif table is False:
        return None
    return table


def _legacy_device_states(state: dict[str, dict[str, Any]]) -> None:
    """Fill ``state`` per device through the torch 2.1 to 2.3 shims.

    Parameters
    ----------
    state:
        The snapshot being built; each device that answers gets its entry.
    """

    for device in _AUTOCAST_DEVICES:
        try:
            # The per-device ``device_type`` argument is torch 2.4+ only; the
            # shim routes to the legacy per-device helpers. See _utils/_torch_compat.py.
            state[device] = {
                "enabled": autocast_is_enabled(device),
                "dtype": autocast_get_dtype(device),
            }
        except (RuntimeError, TypeError):
            # Device doesn't support autocast queries (e.g. no CUDA).
            pass


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
