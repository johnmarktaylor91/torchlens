"""TorchScript call boundary: a ``torch.jit.ScriptFunction`` called from an eager forward.

A function compiled with ``@torch.jit.script`` (or ``torch.jit.trace``) runs its body
in the TorchScript interpreter, below the Python functions TorchLens wraps, so none of
its operations reach the graph and the completeness census reports every dispatch it
makes as a gap. This module records each such call as ONE opaque boundary op (type
``jitscript``, ``transform_kind="jit.script"``): the call's tensor arguments are the
parents, the returned tensors its outputs, and its callable calls the same scripted
function again, so forward replay and perturbation validate through it. It reuses the
foreign-kernel boundary's core (``_function_boundary.open_boundary_call`` /
``close_boundary_call``): every dispatch inside the call is re-owned to the audited
census token ``jit_script:boundary`` and any interior op logged by a Python fallback
(``@torch.jit.ignore``) is dropped in postprocess step 3.

The hook is ``ScriptFunction.__call__`` on the pybind type, gated by the
``HAS_SCRIPT_FUNCTION_CALL_HOOK`` capability flag. A scripted function that calls
another stays inside the interpreter, so a call tree is one boundary op. A call that
writes into one of its tensor arguments in place (or whose argument keeps no version
counter, so a write cannot be ruled out) is NOT collapsed: a functionless boundary
cannot attest that side effect, so the census and the
``scripted_function_not_captured`` advisory keep reporting it. Under ``PYTORCH_JIT=0``
``torch.jit.script`` returns the plain function and nothing here fires. Scripted
modules are refused before capture (``_capture_state_helpers._reject_opaque_wrappers``).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

from ... import _state
from ..._utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from ._function_boundary import BoundarySpec, close_boundary_call, open_boundary_call

SCRIPT_FUNCTION_FUNC_NAME = "jitscript"
"""Op type / func name of the boundary op."""

SCRIPT_FUNCTION_TRANSFORM_KIND = "jit.script"
"""``Op.transform_kind`` of the boundary op."""

SCRIPT_FUNCTION_WRAPPER_NAME = "jit_script:boundary"
"""Completeness-witness owner token of every dispatch inside a recorded call."""

SCRIPT_FUNCTION_RECORDER_ATTR = "__tl_script_function_recorder__"
"""Marker set on the boundary op's replay callable."""

SCRIPT_FUNCTION_WARNING_CODE = "torchscript_boundary"
"""``TorchLensWarning`` code of the once-per-function disclosure."""

_INSTALLED_CALL: Callable[..., Any] | None = None
_ORIGINAL_CALL: Callable[..., Any] | None = None


def is_script_function_recorder(func: Any) -> bool:
    """Return whether ``func`` is a TorchScript boundary replay callable.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for the callable built by :func:`record_script_call`.
    """

    return getattr(func, SCRIPT_FUNCTION_RECORDER_ATTR, False) is True


def install_script_function_call_wrapper() -> None:
    """Wrap ``torch.jit.ScriptFunction.__call__`` idempotently.

    Returns
    -------
    None
        Nothing is patched when ``HAS_SCRIPT_FUNCTION_CALL_HOOK`` is False or the type
        refuses the assignment; the census then reports scripted calls as before.
    """

    global _INSTALLED_CALL, _ORIGINAL_CALL
    if _INSTALLED_CALL is not None or not HAS_SCRIPT_FUNCTION_CALL_HOOK:
        return
    script_type = torch.jit.ScriptFunction
    original = script_type.__dict__["__call__"]

    def wrapped_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the scripted function, recording the call as one boundary op."""

        active_trace, logging_enabled = _state.active_capture()
        if not logging_enabled or active_trace is None:
            return original(self, *args, **kwargs)
        return record_script_call(active_trace, self, original, args, kwargs)

    try:
        script_type.__call__ = wrapped_call  # type: ignore[method-assign]
    except (AttributeError, TypeError):
        return
    _INSTALLED_CALL = wrapped_call
    _ORIGINAL_CALL = original


def uninstall_script_function_call_wrapper() -> None:
    """Restore ``ScriptFunction.__call__`` when the slot still holds our wrapper.

    Returns
    -------
    None
        A foreign patch layered over ours is left in place.
    """

    global _INSTALLED_CALL, _ORIGINAL_CALL
    if _INSTALLED_CALL is None or _ORIGINAL_CALL is None:
        return
    script_type = torch.jit.ScriptFunction
    if script_type.__dict__.get("__call__") is _INSTALLED_CALL:
        script_type.__call__ = _ORIGINAL_CALL  # type: ignore[method-assign]
    _INSTALLED_CALL = None
    _ORIGINAL_CALL = None


def _python_source(function: Any) -> Any:
    """Return the Python function torch kept for a scripted function, if any."""

    return getattr(function, "_torchdynamo_inline", None)


def _version_of(tensor: torch.Tensor) -> int | None:
    """Return a tensor's version counter, or ``None`` when it keeps none (inference mode)."""

    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def _make_replay_callable(function: Any, original_call: Callable[..., Any]) -> Callable[..., Any]:
    """Build the boundary op's callable: the same scripted function, called again."""

    from .wrappers import _set_transform_metadata

    def jitscript(*call_args: Any, **call_kwargs: Any) -> Any:
        """Call the scripted function through the unwrapped ``__call__``."""

        return original_call(function, *call_args, **call_kwargs)

    source = _python_source(function)
    code = getattr(source, "__code__", None)
    setattr(jitscript, SCRIPT_FUNCTION_RECORDER_ATTR, True)
    _set_transform_metadata(
        jitscript,
        transform_kind=SCRIPT_FUNCTION_TRANSFORM_KIND,
        tags=(SCRIPT_FUNCTION_TRANSFORM_KIND,),
        transform_config={
            "function": function.qualified_name,
            "function_module": getattr(source, "__module__", None),
            "source": None if code is None else f"{code.co_filename}:{code.co_firstlineno}",
        },
        inner_fn=source if source is not None else jitscript,
    )
    return jitscript


def _warn_script_boundary(trace: Any, function: Any) -> None:
    """Disclose one recorded scripted function per capture."""

    from ..._errors import TorchLensWarning

    name = function.qualified_name
    warned = trace.__dict__.setdefault("_foreign_kernel_warned", set())
    if name in warned:
        return
    warned.add(name)
    warnings.warn(
        TorchLensWarning(
            f"TorchLens captured the TorchScript function {name} as one boundary op: its "
            "body runs in the TorchScript interpreter, so the operations inside it are not "
            "logged; the boundary op replays by calling the scripted function again. Set "
            "PYTORCH_JIT=0 before the model's code is imported to capture its interior.",
            code="torchscript_boundary",  # literal: the warning-contract census reads it
            function=name,
        ),
        stacklevel=4,
    )


def record_script_call(
    trace: Any,
    function: Any,
    original_call: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run one scripted call under capture and record it as one boundary op.

    Parameters
    ----------
    trace:
        Active capture trace.
    function:
        The ``torch.jit.ScriptFunction`` being called.
    original_call:
        The unwrapped ``ScriptFunction.__call__``.
    args:
        Positional call arguments.
    kwargs:
        Keyword call arguments.

    Returns
    -------
    Any
        Whatever the scripted function returned.
    """

    marks = open_boundary_call(trace, args, kwargs)
    versions = [(tensor, _version_of(tensor)) for tensor, _ in marks.arg_labels]
    out = original_call(function, *args, **kwargs)
    if any(version is None or _version_of(tensor) != version for tensor, version in versions):
        return out
    spec = BoundarySpec(
        replay=_make_replay_callable(function, original_call),
        func_name=SCRIPT_FUNCTION_FUNC_NAME,
        wrapper_name=SCRIPT_FUNCTION_WRAPPER_NAME,
        original=function,
    )
    if close_boundary_call(trace, marks, spec, (args, kwargs, out)):
        _warn_script_boundary(trace, function)
    return out
