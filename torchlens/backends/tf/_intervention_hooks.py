"""Leaf helpers for the TensorFlow intervention layer (no imports from the capture core)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..registry import BackendUnsupportedError


def _selector_kinds(selector: Any) -> frozenset[str]:
    """Return every selector kind used in a selector tree.

    Parameters
    ----------
    selector
        Static selector or composite.

    Returns
    -------
    frozenset[str]
        Selector kinds present in the tree.
    """

    kinds: set[str] = set()
    stack = [selector]
    while stack:
        node = stack.pop()
        kind = getattr(node, "selector_kind", None)
        if kind is None:
            continue
        kinds.add(str(kind))
        value = getattr(node, "selector_value", None)
        if isinstance(value, (list, tuple)):
            stack.extend(value)
        elif hasattr(value, "selector_kind"):
            stack.append(value)
        inner = getattr(node, "inner", None)
        if hasattr(inner, "selector_kind"):
            stack.append(inner)
        members = getattr(node, "selectors", None)
        if isinstance(members, (list, tuple)):
            stack.extend(members)
    return frozenset(kinds)


def _tf_helper_hook(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """Return the TensorFlow implementation for a curated helper spec.

    Parameters
    ----------
    spec
        Portable helper spec built by ``tl.zero_ablate``/``tl.scale``/``tl.add``.
    tf
        Imported TensorFlow module.

    Returns
    -------
    Callable[[Any], Any]
        TensorFlow replacement callable.
    """

    if spec.name == "zero_ablate":
        return lambda out: tf.zeros_like(out)
    if spec.name == "scale":
        factor = spec.args[0]
        return lambda out: out * tf.cast(factor, out.dtype)
    if spec.name == "add":
        delta = spec.args[0]
        return lambda out: out + tf.cast(delta, out.dtype)
    raise BackendUnsupportedError(
        f"tf backend interventions implement the curated helpers zero_ablate, scale, "
        f"and add; helper {spec.name!r} has no TensorFlow implementation. Pass "
        "a callable action for custom replacements."
    )


def _tf_callable_hook(hook: Callable[..., Any]) -> Callable[[Any], Any]:
    """Normalize a user callable into the single-argument hook convention.

    Parameters
    ----------
    hook
        User replacement callable, optionally accepting a ``hook`` keyword.

    Returns
    -------
    Callable[[Any], Any]
        Callable invoked with the original output tensor only.
    """

    try:
        parameter_names = set(inspect.signature(hook).parameters)
    except (TypeError, ValueError):
        parameter_names = set()
    if "ctx" in parameter_names:
        return lambda out: hook(out, ctx=None)
    return hook


@dataclass(frozen=True)
class _TFModuleExitContext:
    """Selector subject for one TensorFlow module-boundary exit.

    Parameters
    ----------
    module
        Pass-qualified module-call label such as ``"dense:1"``.
    address
        Module address.
    module_type
        Module class name.
    module_stack
        Active module frames including the exiting module.
    source_trace
        Owning trace, so selector evaluation sees a non-torch backend.
    """

    module: str
    address: str
    module_type: str
    module_stack: tuple[dict[str, Any], ...]
    source_trace: Any
    kind: str = "module_exit"
    label: None = None
    raw_label: None = None


def _producer_label(session: Any, tensor: Any) -> str | None:
    """Return the raw label of the op event that produced ``tensor``.

    Parameters
    ----------
    session
        Active capture session.
    tensor
        Candidate TensorFlow tensor.

    Returns
    -------
    str | None
        Producer raw label when the callback recorded the tensor.
    """

    ref = getattr(tensor, "ref", None)
    if not callable(ref):
        return None
    try:
        return session.producer_by_ref.get(ref())
    except TypeError:
        return None
