"""Captum bridge helpers."""

from __future__ import annotations

from typing import Any

from torch import nn

from ._utils import first_input_tensor, resolve_one_site, source_model


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
        TorchLens ``ModelLog`` from the model being attributed.
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

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` with a live source model reference.
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
    ValueError
        If the site cannot be mapped to a live module.
    """

    try:
        import captum  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Captum bridge requires the `captum` extra: install torchlens[captum]."
        ) from exc

    model = source_model(log)
    modules = dict(model.named_modules())
    if site == "self":
        return model
    if isinstance(site, str):
        address = site.rsplit(":", maxsplit=1)[0]
        if address in modules:
            return modules[address]

    resolved = resolve_one_site(log, site)
    candidates = list(getattr(resolved, "module_passes_exited", ()) or [])
    containing_module = getattr(resolved, "containing_module", None)
    if containing_module is not None:
        candidates.append(str(containing_module))
    for candidate in reversed(candidates):
        address = str(candidate).rsplit(":", maxsplit=1)[0]
        if address in modules:
            return modules[address]
    raise ValueError(f"Could not resolve Captum layer for site {site!r}.")


__all__ = ["attribute", "layer"]
