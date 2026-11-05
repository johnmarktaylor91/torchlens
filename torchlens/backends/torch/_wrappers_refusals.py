"""Typed refusals raised from the torch function wrapper's logged call path."""

from __future__ import annotations

from typing import Any

import torch

from ..._utils._torch_compat import HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
from ._modes import TensorSubclassUnsupportedError


def _raise_if_subclass_ctor_under_dispatch_mode(
    exc: RuntimeError, func_name: str, pauses_owned_modes: bool
) -> None:
    """Translate torch's subclass-under-dispatch-mode crash into the typed refusal."""
    if (
        pauses_owned_modes
        and not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
        and "already associated to a python object" in str(exc)
    ):
        raise TensorSubclassUnsupportedError(
            f"torchlens could not construct the Tensor subclass via "
            f"{func_name!r} even after pausing its own dispatch mode on this "
            f"torch build (original torch error: {exc}). "
            "Remedy: upgrade torch (HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE reports "
            "False here), or avoid this construction while other TorchLens "
            "instrumentation observes the same forward on this build.",
            code="subclass_ctor_under_dispatch_mode_unsupported",
            func_name=func_name,
        ) from exc


def _constructs_strict_subclass(args: tuple[Any, ...], idx: int | None) -> bool:
    """Return whether ``args[idx]`` is a strict (non-``torch.Tensor``) subclass cls.

    Parameters
    ----------
    args:
        Positional call arguments.
    idx:
        Index of the candidate subclass ``cls`` argument, or ``None`` when the
        caller has none (always ``False``; lets callers pass their own
        ``... | None`` index constant without narrowing it themselves first).

    Returns
    -------
    bool
        ``True`` when ``args[idx]`` is a type strictly below ``torch.Tensor``.
    """

    if idx is None or len(args) <= idx:
        return False
    cls = args[idx]
    return isinstance(cls, type) and cls is not torch.Tensor and issubclass(cls, torch.Tensor)


def _converts_strict_subclass_receiver(func_name: str, args: tuple[Any, ...]) -> bool:
    """Return whether ``as_subclass`` converts OUT of a strict subclass instance.

    The receiver-side twin of :func:`_constructs_strict_subclass`: torch 2.1
    to 2.5 also refuse ``instance.as_subclass(torch.Tensor)`` when the
    instance's Python object is a strict Tensor subclass and a dispatch mode
    is active ("Creating a new Tensor subclass Tensor but the raw Tensor
    object is already associated to a python object of type X"). A
    ``Parameter`` receiver keeps its own frozen-receiver handling.

    Parameters
    ----------
    func_name:
        Name of the wrapped callable.
    args:
        Positional call arguments; ``args[0]`` is the ``as_subclass`` receiver.

    Returns
    -------
    bool
        ``True`` for ``as_subclass`` on a strict, non-``Parameter`` subclass.
    """

    if func_name != "as_subclass" or not args or not isinstance(args[0], torch.Tensor):
        return False
    return type(args[0]) not in (torch.Tensor, torch.nn.Parameter)


def _pauses_for_subclass_conversion(func_name: str, args: tuple[Any, ...], idx: int | None) -> bool:
    """Return whether a logged subclass construction must pause owned modes.

    Parameters
    ----------
    func_name:
        Name of the wrapped callable.
    args:
        Positional call arguments.
    idx:
        Index of the target ``cls`` argument, or ``None`` when there is none.

    Returns
    -------
    bool
        ``True`` when the call converts into, or out of, a strict subclass.
    """

    return _constructs_strict_subclass(args, idx) or _converts_strict_subclass_receiver(
        func_name, args
    )
