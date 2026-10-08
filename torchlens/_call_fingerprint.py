"""Native-forward twin of the capture-time ordered call fingerprint.

Every torch capture folds one token per unpaused owner-thread torch call (the
decorated wrappers) and one per non-root module forward entry (the module
forward decoration) into ``_state.CallFingerprint``; ``active_logging`` stores
the result as ``Trace._raw_call_fingerprint``. This module reproduces the same
stream around a plain, uncaptured forward so a guarded fast re-run can prove
it executed the same op structure, e.g. on a different-length input.

Contract shared with capture:

* torch calls count through the persistent wrappers, so torch must be wrapped
  (any earlier capture of the model did that);
* a module entry counts at forward pre-hook time; capture counts it inside the
  decorated ``forward``, after the module's other pre-hooks, so the token hook
  is appended last;
* the ROOT module gets no token in either engine: capture leaves the root's
  ``forward`` undecorated and calls it exactly once;
* calls under ``pause_logging()`` (TorchLens internals, intervention hook
  callables) never count, in either engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn
from torch.utils.hooks import RemovableHandle

from . import _state
from .backends.torch._tl import get_module_meta


def module_fingerprint_address(module: nn.Module, fallback: str) -> str:
    """Return the address capture uses for ``module``'s fingerprint token.

    Parameters
    ----------
    module:
        Module whose entry is fingerprinted.
    fallback:
        ``named_modules`` path, used when the module was never prepared.

    Returns
    -------
    str
        The prepared ``_tl.address`` when present, else ``fallback``.
    """

    meta = get_module_meta(module)
    if meta is None or meta.address is None:
        return fallback
    return meta.address


def _token_pre_hook(token: int) -> Any:
    """Build a forward pre-hook that notes one module-entry token.

    Parameters
    ----------
    token:
        Precomputed ``_state.module_token`` of the module's address.

    Returns
    -------
    Any
        Pre-hook callable returning ``None`` (inputs unchanged).
    """

    def hook(_module: nn.Module, _args: Any) -> None:
        """Fold this module's entry token into the active fingerprint."""
        _state.note_fingerprint_token(token)

    return hook


def install_module_token_hooks(model: nn.Module) -> list[RemovableHandle]:
    """Register a module-entry token pre-hook on every non-root module.

    Parameters
    ----------
    model:
        Root module; it gets no hook (capture gives the root no token).

    Returns
    -------
    list[RemovableHandle]
        Handles the caller must ``remove()`` when the run ends.
    """

    handles: list[RemovableHandle] = []
    try:
        for name, module in model.named_modules():
            if module is model:
                continue
            token = _state.module_token(module_fingerprint_address(module, name))
            handles.append(module.register_forward_pre_hook(_token_pre_hook(token)))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


@contextmanager
def fingerprinting() -> Iterator[_state.CallFingerprint]:
    """Collect a fresh ordered call fingerprint for the current thread.

    Yields
    ------
    _state.CallFingerprint
        The fingerprint; read ``.value`` after the block.
    """

    with _state.call_fingerprinting() as fp:
        yield fp


def fingerprint_native_forward(
    model: nn.Module, input_args: Any, input_kwargs: dict[str, Any] | None
) -> tuple[tuple[int, int], Any]:
    """Run ``model`` natively and return its call fingerprint and output.

    Parameters
    ----------
    model:
        Module to run; torch must already be wrapped (a prior capture).
    input_args:
        Positional inputs (a tuple/list, or a single non-sequence input).
    input_kwargs:
        Keyword inputs, or ``None``.

    Returns
    -------
    tuple[tuple[int, int], Any]
        ``((count, digest), output)``, comparable with
        ``Trace._raw_call_fingerprint``.
    """

    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    kwargs = dict(input_kwargs or {})
    handles = install_module_token_hooks(model)
    try:
        with fingerprinting() as fp:
            output = model(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return fp.value, output
