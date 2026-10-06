"""Captum bridge helpers."""

from __future__ import annotations

from typing import Any

from torch import nn

# ``source_model`` stays importable from this module (audit notebook 16 calls
# ``bridge.captum.source_model``).
from ._utils import first_input_tensor, module_for_site, source_model  # noqa: F401


def attribute(
    log: Any,
    method: Any,
    target: Any,
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
        Captum target forwarded to ``method.attribute``.
    inputs:
        Optional explicit Captum input. When omitted, the first logged input
        tensor is used.
    **kwargs:
        Additional keyword arguments forwarded to Captum.

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
    """

    try:
        import captum  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Captum bridge requires the `captum` extra: install torchlens[captum]."
        ) from exc

    if not hasattr(method, "attribute"):
        raise TypeError("Captum bridge expected a method object with an attribute(...) method.")

    captum_inputs = first_input_tensor(log) if inputs is None else inputs
    return method.attribute(captum_inputs, target=target, **kwargs)


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
