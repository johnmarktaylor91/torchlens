"""Shared helpers for optional bridge adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def source_model(log: Any) -> nn.Module:
    """Return the live source model retained by a model log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` with a live ``_source_model_ref``.

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
    model = source_ref() if source_ref is not None else None
    if model is None:
        raise ValueError(
            "This bridge requires a live source model. Re-run log_forward_pass and keep "
            "the model object alive while using the bridge."
        )
    return model


def resolve_one_site(log: Any, site: Any) -> Any:
    """Resolve a TorchLens site-like value to one layer-pass record.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    site:
        Layer label, selector, or already-resolved layer object.

    Returns
    -------
    Any
        Layer-pass-like object with an ``activation`` attribute.
    """

    if hasattr(site, "activation") and hasattr(site, "layer_label"):
        return site
    resolved = log.resolve_sites(site, max_fanout=1)
    return resolved.first()


def activation_at(log: Any, site: Any) -> torch.Tensor:
    """Return a tensor activation for one resolved site.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    site:
        Layer label, selector, or layer object.

    Returns
    -------
    torch.Tensor
        Saved tensor activation.

    Raises
    ------
    ValueError
        If the site does not carry a tensor activation.
    """

    layer = resolve_one_site(log, site)
    activation = getattr(layer, "activation", None)
    if not isinstance(activation, torch.Tensor):
        label = getattr(layer, "layer_label", site)
        raise ValueError(f"Bridge site {label!r} does not have a saved tensor activation.")
    return activation


def first_input_tensor(log: Any) -> torch.Tensor:
    """Return the first saved input tensor in a model log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.

    Returns
    -------
    torch.Tensor
        First saved input activation.

    Raises
    ------
    ValueError
        If no tensor input activation is present.
    """

    for layer in getattr(log, "layer_list", []):
        activation = getattr(layer, "activation", None)
        if getattr(layer, "is_input_layer", False) and isinstance(activation, torch.Tensor):
            return activation
    raise ValueError("Could not find a saved tensor input in this ModelLog.")


def tensor_layers(log: Any, sites: Iterable[Any] | None = None) -> list[Any]:
    """Return layer-pass records with tensor activations.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
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
        if isinstance(getattr(layer, "activation", None), torch.Tensor)
        and not getattr(layer, "is_input_layer", False)
    ]


__all__ = [
    "activation_at",
    "first_input_tensor",
    "resolve_one_site",
    "source_model",
    "tensor_layers",
]
