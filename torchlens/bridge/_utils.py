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

    The accepted vocabulary is deliberately WIDE (neuro memo item 9): any
    ``resolve_sites`` selector or raw op label, or -- when that resolution
    misses -- any ``log[...]`` lookup, which covers module dotted paths
    (``"encoder.layer.3"``), the spelling Brain-Score users already write.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    site:
        Layer label, selector, module dotted path, or already-resolved
        layer object.

    Returns
    -------
    Any
        Layer-pass-like object with an ``out`` attribute.
    """

    from ..intervention.errors import SiteResolutionError

    if hasattr(site, "out") and hasattr(site, "layer_label"):
        return site
    try:
        resolved = log.resolve_sites(site, max_fanout=1)
        return resolved.first()
    except SiteResolutionError:
        if isinstance(site, str):
            # Module dotted paths and other getitem-only spellings: the
            # trace lookup teaches its own error when this also misses.
            return log[site]
        raise


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


def tensor_layers(
    log: Any, sites: Iterable[Any] | None = None, *, verb: str = "bridge"
) -> list[Any]:
    """Return layer-pass records with tensor outs.

    Default sweeps run through the core stimulus-indexed eligibility gate
    (neuro memo item 9): a partially saved trace no longer crashes with
    ``PayloadUnavailableError`` (unsaved layers are simply not saved sites),
    and buffer overwrites are skipped with one summarized disclosure instead
    of being scored as if they were stimulus responses.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    sites:
        Optional iterable of site-like values (labels, selectors, module
        dotted paths). When omitted, all stimulus-indexed saved tensor
        layers except input placeholders are returned.
    verb:
        Public caller name used in skip disclosures and refusals.

    Returns
    -------
    list[Any]
        Layer-pass records in requested order, each carrying a tensor ``out``.

    Raises
    ------
    ValueError
        If an explicitly requested site does not carry a saved tensor
        ``out`` (the documented contract -- raised BEFORE any payload read,
        so it can no longer be preempted by ``PayloadUnavailableError``),
        or is not stimulus-indexed.
    """

    from ..repgeom._annotation_gate import (
        _expected_stimulus_counts,
        _raise_ineligible_site,
        _site_ineligibility,
    )

    expected_counts = _expected_stimulus_counts(log)
    if sites is not None:
        resolved: list[Any] = []
        for site in sites:
            layer = resolve_one_site(log, site)
            label = getattr(layer, "layer_label", site)
            if not bool(getattr(layer, "has_saved_activation", False)):
                raise ValueError(f"Bridge site {label!r} does not have a saved tensor out.")
            reason = _site_ineligibility(layer, expected_counts)
            if reason is not None:
                _raise_ineligible_site(verb, str(label), reason)
            out = getattr(layer, "out", None)
            if not isinstance(out, torch.Tensor):
                raise ValueError(f"Bridge site {label!r} does not have a saved tensor out.")
            resolved.append(layer)
        return resolved

    return _default_tensor_layer_sweep(log, expected_counts, verb=verb)


def _default_tensor_layer_sweep(
    log: Any, expected_counts: frozenset[int], *, verb: str
) -> list[Any]:
    """Sweep every stimulus-indexed saved tensor layer (default ``sites=None``).

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    expected_counts:
        Leading-axis sizes of the capture's input sites.
    verb:
        Public caller name used in the one summarized skip disclosure.

    Returns
    -------
    list[Any]
        Eligible layer-pass records in trace order.
    """

    from collections import OrderedDict

    from ..repgeom._annotation_gate import _site_ineligibility, _warn_skipped_sites

    selected: list[Any] = []
    skipped: OrderedDict[str, str] = OrderedDict()
    for layer in getattr(log, "layer_list", []):
        if getattr(layer, "is_input", False):
            continue
        if not bool(getattr(layer, "has_saved_activation", False)):
            continue
        reason = _site_ineligibility(layer, expected_counts)
        if reason is not None:
            skipped[str(getattr(layer, "layer_label", "?"))] = reason
            continue
        out = getattr(layer, "out", None)
        if isinstance(out, torch.Tensor):
            selected.append(layer)
    _warn_skipped_sites(verb, skipped)
    return selected


__all__ = [
    "out_at",
    "first_input_tensor",
    "resolve_one_site",
    "source_model",
    "tensor_layers",
]
