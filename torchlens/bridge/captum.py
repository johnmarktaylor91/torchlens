"""Captum bridge helpers."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError

# ``source_model`` stays importable from this module (audit notebook 16 calls
# ``bridge.captum.source_model``).
from ._utils import module_for_site, source_model


def attribute(
    log: Any,
    method: Any,
    target: Any | None = None,
    *,
    inputs: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Captum attribution method using inputs retained by a TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` from the model being attributed.
    method:
        Captum attribution object exposing ``attribute``.
    target:
        Captum target, forwarded as ``target=`` only when given: the
        activation and neuron classes (``LayerActivation``, ``NeuronGradient``,
        ...) take no ``target`` and reject one.
    inputs:
        Optional explicit Captum input (a tensor or a tuple of tensors). When
        omitted, every logged forward input is forwarded: the single tensor,
        or the tuple of positional tensors in ``forward`` order.
    **kwargs:
        Additional keyword arguments forwarded to Captum (``baselines``,
        ``neuron_selector``, ``additional_forward_args``, ...).

    Returns
    -------
    Any
        Captum attribution result.

    Raises
    ------
    ImportError
        If Captum is unavailable.
    TypeError
        If ``method`` does not expose ``attribute``.
    InvalidArgumentError
        ``bridge_captum_inputs_unresolved`` when ``inputs`` is omitted and the
        logged inputs cannot be replayed positionally (a keyword-only or
        nested input, a missing positional tensor, or no saved input).
    """

    try:
        import captum  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Captum bridge requires the `captum` extra: install torchlens[captum]."
        ) from exc

    if not hasattr(method, "attribute"):
        raise TypeError("Captum bridge expected a method object with an attribute(...) method.")

    captum_inputs = _logged_inputs(log) if inputs is None else inputs
    if target is not None:
        kwargs["target"] = target
    return method.attribute(captum_inputs, **kwargs)


def _logged_inputs(log: Any) -> Any:
    """Return every logged forward input in the shape Captum's ``inputs`` takes.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with saved input tensors.

    Returns
    -------
    Any
        The input tensor for a one-input forward, else the tuple of input
        tensors in ``forward``'s positional order.

    Raises
    ------
    InvalidArgumentError
        ``bridge_captum_inputs_unresolved`` when an input was passed by
        keyword only, sits inside a container, was not saved, or a positional
        parameter between two logged inputs has no tensor.
    """

    tensors, problem = _positional_inputs(log)
    if problem is not None:
        raise InvalidArgumentError(
            f"Captum bridge cannot forward the logged inputs: {problem}.",
            code="bridge_captum_inputs_unresolved",
            remedy=(
                "pass inputs= explicitly (a tensor or a tuple of the positional "
                "tensors) and the rest through additional_forward_args="
            ),
        )
    return tensors[0] if len(tensors) == 1 else tensors


def _positional_inputs(log: Any) -> tuple[tuple[torch.Tensor, ...], str | None]:
    """Collect the logged inputs in ``forward`` order, or say why they cannot be."""

    by_name: dict[str, torch.Tensor] = {}
    for layer in getattr(log, "layer_list", []):
        if not getattr(layer, "is_input", False):
            continue
        role = str(getattr(layer, "io_role", "") or "")
        parts = role.split(".")
        out = getattr(layer, "out", None)
        if len(parts) != 2 or parts[0] != "input" or not isinstance(out, torch.Tensor):
            return (), f"input {role or layer.layer_label!r} is nested or unsaved"
        by_name[parts[1]] = out
    if not by_name:
        return (), "the trace holds no saved input tensor"
    positional = _positional_names(source_model(log))
    order = [name for name in positional if name in by_name]
    if set(order) != set(by_name) or order != positional[: len(order)]:
        return (), (
            f"logged inputs {sorted(by_name)} are not the leading positional "
            f"parameters {positional}"
        )
    return tuple(by_name[name] for name in order), None


def _positional_names(model: nn.Module) -> list[str]:
    """Return ``model.forward``'s positional parameter names in order."""

    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    try:
        parameters = inspect.signature(model.forward).parameters.values()
    except (TypeError, ValueError):
        return []
    return [param.name for param in parameters if param.kind in kinds]


def layer(log: Any, site: Any) -> nn.Module:
    """Resolve a TorchLens module/site to the live PyTorch module Captum expects.

    Shares Grad-CAM's resolver: a module address returns that module; an op
    label returns the outermost module whose output the op's tensor is,
    refusing when that module runs more than once.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Module address, module pass label, layer selector, or layer object.

    Returns
    -------
    nn.Module
        PyTorch module corresponding to the requested TorchLens site.

    Raises
    ------
    ImportError
        If Captum is unavailable.
    InvalidArgumentError
        If the site maps to no module, to sibling modules, or to a module that
        runs more than once (see :func:`torchlens.bridge._utils.module_for_site`).
    """

    try:
        import captum  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Captum bridge requires the `captum` extra: install torchlens[captum]."
        ) from exc

    return module_for_site(log, site, bridge="Captum")


__all__ = ["attribute", "layer"]
