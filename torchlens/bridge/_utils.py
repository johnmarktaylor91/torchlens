"""Shared helpers for optional bridge adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn

from .._errors import InvalidArgumentError


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


def module_for_site(log: Any, site: Any, *, bridge: str) -> nn.Module:
    """Resolve a TorchLens site to the live module whose forward hook sees it.

    Attribution bridges (Grad-CAM, Captum layer methods) hook a module, so a
    site must name exactly one module call. A plain module address
    (``"layer4"``) or ``"self"`` returns that module. Any other site resolves
    to one op; the op's tensor may be the output of several nested modules
    (``layer4.1.relu``, ``layer4.1``, ``layer4``), and the OUTERMOST of them
    is the module whose output that tensor is. A pass-qualified address
    (``"layer4.1.relu:2"``) names one module call; its output tensor resolves
    the same way (``layer4.1.relu:2`` returns ``layer4``'s output, so it
    resolves to ``layer4``). A pass-qualified address or an op label names
    one call, so the resolved module must run exactly once in the trace: a
    module hook would see every call, not the one named.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with a live source model reference.
    site:
        Module address, module pass label, op label, selector, or layer object.
    bridge:
        Bridge name used in refusal messages.

    Returns
    -------
    nn.Module
        Live PyTorch module.

    Raises
    ------
    InvalidArgumentError
        ``bridge_module_site_unresolved`` when the site is not the output of
        any module, ``bridge_module_site_ambiguous`` when two sibling modules
        share the outermost depth, and ``bridge_module_site_multi_call`` when
        the resolved module runs more than once in the trace.
    """

    model = source_model(log)
    modules = dict(model.named_modules())
    if site == "self":
        return model
    if isinstance(site, str) and site in modules:
        return cast(nn.Module, modules[site])
    if isinstance(site, str):
        address, _, suffix = site.rpartition(":")
        if address in modules and suffix.isdigit():
            owner = _pass_output_owner(log, modules, site) or address
            return _single_call_module(log, modules, owner, site=site, bridge=bridge)

    resolved = resolve_one_site(log, site)
    addresses = _owning_addresses(resolved, modules)
    if not addresses:
        raise InvalidArgumentError(
            f"Could not resolve {bridge} layer for site {site!r}: the site is not the "
            "output of any module of the source model",
            code="bridge_module_site_unresolved",
            remedy="pass a module address such as 'layer4', or the label of an op whose "
            "tensor a module returns",
        )
    depth = min(_address_depth(address) for address in addresses)
    outermost = [address for address in addresses if _address_depth(address) == depth]
    if len(outermost) > 1:
        raise InvalidArgumentError(
            f"{bridge} site {site!r} is the output of sibling modules {outermost!r}; "
            "no one module owns it",
            code="bridge_module_site_ambiguous",
            remedy="pass one of those module addresses directly",
        )
    return _single_call_module(log, modules, outermost[0], site=site, bridge=bridge)


def _owning_addresses(layer: Any, modules: dict[str, nn.Module]) -> list[str]:
    """Return the source-model module addresses whose output ``layer``'s tensor is.

    Parameters
    ----------
    layer:
        Layer-pass-like record with ``output_of_module_calls``.
    modules:
        ``named_modules()`` mapping of the source model.

    Returns
    -------
    list[str]
        Distinct module addresses, in the record's order.
    """

    calls = getattr(layer, "output_of_module_calls", ()) or ()
    return [
        address
        for address in dict.fromkeys(_module_call_address(call) for call in calls)
        if address in modules
    ]


def _pass_output_owner(log: Any, modules: dict[str, nn.Module], site: str) -> str | None:
    """Return the outermost module owning the output of module call ``site``.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    modules:
        ``named_modules()`` mapping of the source model.
    site:
        Pass-qualified module address such as ``"layer4.1.relu:2"``.

    Returns
    -------
    str | None
        The single outermost owning address shared by every op that call
        returns; None when no op names the call or the owners disagree (the
        caller then falls back to the call's own module).
    """

    owners: set[str] = set()
    for layer in getattr(log, "layer_list", []):
        calls = getattr(layer, "output_of_module_calls", ()) or ()
        if not any(_module_call_label(call) == site for call in calls):
            continue
        addresses = _owning_addresses(layer, modules)
        depth = min(_address_depth(address) for address in addresses)
        outermost = [address for address in addresses if _address_depth(address) == depth]
        if len(outermost) != 1:
            return None
        owners.add(outermost[0])
    return owners.pop() if len(owners) == 1 else None


def _module_call_label(call: Any) -> str:
    """Return one ``output_of_module_calls`` entry as ``"address:call_index"``."""

    if isinstance(call, tuple) and len(call) >= 2:
        return f"{call[0]}:{call[1]}"
    return str(call)


def _module_call_address(call: Any) -> str:
    """Return the module address of one ``output_of_module_calls`` entry.

    Parameters
    ----------
    call:
        ``(address, call_index)`` tuple or ``"address:call_index"`` string.

    Returns
    -------
    str
        Module dotted address.
    """

    if isinstance(call, tuple) and call:
        return str(call[0])
    text = str(call)
    address, _, suffix = text.rpartition(":")
    return address if address and suffix.isdigit() else text


def _address_depth(address: str) -> int:
    """Return the nesting depth of a module address (root is 0)."""

    return 0 if not address else address.count(".") + 1


def _single_call_module(
    log: Any, modules: dict[str, nn.Module], address: str, *, site: Any, bridge: str
) -> nn.Module:
    """Return ``modules[address]`` after refusing a module that runs more than once.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    modules:
        ``named_modules()`` mapping of the source model.
    address:
        Resolved module address.
    site:
        Original site, for the refusal message.
    bridge:
        Bridge name used in refusal messages.

    Returns
    -------
    nn.Module
        The resolved live module.
    """

    calls = _module_num_calls(log, address)
    if calls > 1:
        enclosing = _enclosing_single_call(log, modules, address)
        hint = (
            f"pass the enclosing module {enclosing!r}, which runs once (it hooks that "
            "module's own output, not this tensor), or "
            if enclosing
            else ""
        )
        raise InvalidArgumentError(
            f"{bridge} site {site!r} resolves to module {address!r}, which runs {calls} "
            "times in this trace; a module hook sees every call, not the one the site names",
            code="bridge_module_site_multi_call",
            remedy=f"{hint}pick a site whose outermost owning module runs once",
        )
    return modules[address]


def _enclosing_single_call(log: Any, modules: dict[str, nn.Module], address: str) -> str | None:
    """Return the nearest enclosing module address that runs once, if any.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    modules:
        ``named_modules()`` mapping of the source model.
    address:
        Address of a module that runs more than once.

    Returns
    -------
    str | None
        Nearest proper ancestor (never the root) with one call; None if none.
    """

    parts = address.split(".")
    for end in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:end])
        if parent in modules and _module_num_calls(log, parent) == 1:
            return parent
    return None


def _module_num_calls(log: Any, address: str) -> int:
    """Return how many times module ``address`` ran in the trace.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    address:
        Module dotted address.

    Returns
    -------
    int
        Call count; the trace's module record when available, else the number
        of distinct calls named in the ops' ``output_of_module_calls``.
    """

    modules = log.modules if hasattr(log, "modules") else None
    record = modules[address] if modules is not None and address in modules else None
    num_calls = getattr(record, "num_calls", None)
    if num_calls is not None:
        return int(num_calls)
    # Duck-typed logs without a module record: count the distinct calls instead.
    seen: set[Any] = set()
    for layer in getattr(log, "layer_list", []):
        for call in getattr(layer, "output_of_module_calls", ()) or ():
            if _module_call_address(call) == address:
                seen.add(call if isinstance(call, tuple) else str(call))
    return max(len(seen), 1)


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
    "module_for_site",
    "out_at",
    "first_input_tensor",
    "resolve_one_site",
    "source_model",
    "tensor_layers",
]
