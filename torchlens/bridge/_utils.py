"""Shared helpers for optional bridge adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn


def source_model(log: Any) -> nn.Module:
    """Return the live source model retained by a model log.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with a live ``_source_model_ref``.

    Returns
    -------
    nn.Module
        Source model used for the captured forward pass.

    Raises
    ------
    ValueError
        If the source model is no longer available.
    """

    source_ref = getattr(log, "_source_model_ref", None)
    model = cast(nn.Module | None, source_ref() if source_ref is not None else None)
    if model is None:
        raise ValueError(
            "This bridge requires a live source model. Re-run trace and keep "
            "the model object alive while using the bridge."
        )
    return model


def resolve_one_site(log: Any, site: Any) -> Any:
    """Resolve a TorchLens site-like value to one layer-pass record.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    site:
        Layer label, selector, or already-resolved layer object.

    Returns
    -------
    Any
        Layer-pass-like object with an ``out`` attribute.
    """

    if hasattr(site, "out") and hasattr(site, "layer_label"):
        return site
    resolved = log.resolve_sites(site, max_fanout=1)
    return resolved.first()


def out_at(log: Any, site: Any) -> torch.Tensor:
    """Return a tensor out for one resolved site.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    site:
        Layer label, selector, or layer object.

    Returns
    -------
    torch.Tensor
        Saved tensor out.

    Raises
    ------
    ValueError
        If the site does not carry a tensor out.
    """

    layer = resolve_one_site(log, site)
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor):
        label = getattr(layer, "layer_label", site)
        raise ValueError(f"Bridge site {label!r} does not have a saved tensor out.")
    return out


def first_input_tensor(log: Any) -> torch.Tensor:
    """Return the first saved input tensor in a model log.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    torch.Tensor
        First saved input out.

    Raises
    ------
    ValueError
        If no tensor input out is present.
    """

    for layer in getattr(log, "layer_list", []):
        out = getattr(layer, "out", None)
        if getattr(layer, "is_input", False) and isinstance(out, torch.Tensor):
            return out
    raise ValueError("Could not find a saved tensor input in this Trace.")


def tensor_layers(log: Any, sites: Iterable[Any] | None = None) -> list[Any]:
    """Return layer-pass records with tensor outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    sites:
        Optional iterable of site-like values. When omitted, all saved tensor
        layers except input placeholders are returned.

    Returns
    -------
    list[Any]
        Layer-pass records in requested order.
    """

    if sites is not None:
        return [resolve_one_site(log, site) for site in sites]
    return [
        layer
        for layer in getattr(log, "layer_list", [])
        if isinstance(getattr(layer, "out", None), torch.Tensor)
        and not getattr(layer, "is_input", False)
    ]


__all__ = [
    "out_at",
    "first_input_tensor",
    "resolve_one_site",
    "source_model",
    "tensor_layers",
]
