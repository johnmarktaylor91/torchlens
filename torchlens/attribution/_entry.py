"""Integer inputs for the layer methods: the input path starts where the integers become floats.

Token ids carry no gradient, so a layer method given only integer inputs
attributes from the first floating-point values computed from them: the
outputs of the innermost modules (embeddings, in practice) that take only
integer tensors from the caller's inputs and return a floating-point tensor.
Those outputs play the role a floating-point input plays otherwise: the
straight path of the path methods runs between their baseline and input
values, and they are substituted into each forward pass with forward hooks.
The result therefore equals the same method run on a model that takes the
embedding output as its input.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._result import AttributionError

_DOCS_POINTER = "See docs/reference/attribution.md, section Integer inputs."


def _is_float_tensor(value: Any) -> bool:
    """Return whether ``value`` is a floating-point or complex tensor."""

    return isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex())


def _tensor_leaves(tree: Any) -> list[Tensor]:
    """Return every tensor leaf of a tuple/list/dict tree in traversal order."""

    if isinstance(tree, Tensor):
        return [tree]
    if isinstance(tree, tuple | list):
        return [leaf for item in tree for leaf in _tensor_leaves(item)]
    if isinstance(tree, dict):
        return [leaf for value in tree.values() for leaf in _tensor_leaves(value)]
    return []


def _replace_tensors(tree: Any, replacements: dict[int, Tensor]) -> Any:
    """Return ``tree`` with tensors whose ``id`` is a key replaced by the mapped tensor."""

    if isinstance(tree, Tensor):
        return replacements.get(id(tree), tree)
    if isinstance(tree, tuple):
        return tuple(_replace_tensors(item, replacements) for item in tree)
    if isinstance(tree, list):
        return [_replace_tensors(item, replacements) for item in tree]
    if isinstance(tree, dict):
        return {key: _replace_tensors(value, replacements) for key, value in tree.items()}
    return tree


@dataclass(frozen=True)
class _EntrySite:
    """One call of a module that turns caller integers into floats.

    Attributes
    ----------
    layer
        Module name in ``model.named_modules()``.
    module
        The module.
    call
        Index of this call among the module's calls in one forward pass.
    sources
        Positions, in the integer-leaf list, of the caller integers it consumed.
    """

    layer: str
    module: Module
    call: int
    sources: tuple[int, ...]


@dataclass(frozen=True)
class _EntryRoute:
    """The integer-input route of a layer method.

    Attributes
    ----------
    sites
        Entry calls, in the order their outputs appear among the attributed leaves.
    values
        Detached floating-point outputs of ``sites`` on the caller's inputs.
    integer_leaves
        The caller's integer tensor leaves in traversal order.
    """

    sites: tuple[_EntrySite, ...]
    values: tuple[Tensor, ...]
    integer_leaves: tuple[Tensor, ...]

    @contextmanager
    def substituted(self, replacements: tuple[Tensor, ...]) -> Iterator[None]:
        """Replace each entry call's output with its replacement during one forward pass.

        Raises
        ------
        AttributionError
            ``layer_path_embedding_inconsistent`` when an entry module's output
            differs in shape from its replacement (different control flow on
            this call).
        """

        by_module: dict[int, dict[int, Tensor]] = {}
        for site, replacement in zip(self.sites, replacements, strict=True):
            by_module.setdefault(id(site.module), {})[site.call] = replacement
        counts: dict[int, int] = {}
        handles: list[RemovableHandle] = []

        def _hook(module: Module, _args: tuple[Any, ...], output: Any) -> Any:
            """Return the replacement for this call of an entry module."""

            call = counts.get(id(module), 0)
            counts[id(module)] = call + 1
            replacement = by_module[id(module)].get(call)
            if replacement is None:
                return None
            if not isinstance(output, Tensor) or output.shape != replacement.shape:
                raise AttributionError(
                    f"internal error: an embedding module returned shape "
                    f"{tuple(getattr(output, 'shape', ()))} on call {call} where "
                    f"{tuple(replacement.shape)} was recorded; the model took different "
                    "control flow between forward passes. This is a contract breach of the "
                    "integer-input route, not a user error. Remedy: report this as a bug.",
                    code="layer_path_embedding_inconsistent",
                )
            return replacement

        # Prepended so capture hooks on the same module (a layer method naming
        # the embedding itself) record the substituted, differentiable value.
        modules = {id(site.module): site.module for site in self.sites}
        for module in modules.values():
            handles.append(module.register_forward_hook(_hook, prepend=True))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def sources_of(self, entry_values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Return the caller integer leaves consumed by the given entry values."""

        positions: list[int] = []
        for value in entry_values:
            for index, own in enumerate(self.values):
                if own is value:
                    positions.extend(self.sites[index].sources)
        unique = sorted(set(positions))
        return tuple(self.integer_leaves[position] for position in unique)


def _record_entries(
    model: Module,
    positional: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[tuple[_EntrySite, Tensor]], list[str]]:
    """Run one forward pass and record the calls that turn caller integers into floats.

    Returns
    -------
    tuple[list[tuple[_EntrySite, Tensor]], list[str]]
        Innermost entry calls with their outputs, and the names of modules
        whose output was a floating-point tensor, in call order (for remedy
        spellings).
    """

    from torchlens.attribution._core import _temporarily_eval

    leaves = [leaf for leaf in _tensor_leaves((positional, kwargs)) if not _is_float_tensor(leaf)]
    position_by_id = {id(leaf): index for index, leaf in enumerate(leaves)}
    records: list[tuple[_EntrySite, Tensor]] = []
    float_outputs: list[str] = []
    counts: dict[str, int] = {}
    handles: list[RemovableHandle] = []

    def _hook_for(name: str) -> Any:
        """Return a with-kwargs forward hook recording entry calls of ``name``."""

        def _hook(
            module: Module, args: tuple[Any, ...], call_kwargs: dict[str, Any], output: Any
        ) -> None:
            """Record this call when it takes only caller integers and returns floats."""

            call = counts.get(name, 0)
            counts[name] = call + 1
            if not _is_float_tensor(output):
                return
            float_outputs.append(name)
            tensors = _tensor_leaves((args, call_kwargs))
            if not tensors or any(_is_float_tensor(tensor) for tensor in tensors):
                return
            sources = tuple(position_by_id[id(t)] for t in tensors if id(t) in position_by_id)
            if sources:
                site = _EntrySite(layer=name, module=module, call=call, sources=sources)
                records.append((site, output.detach()))

        return _hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(_hook_for(name), with_kwargs=True))
    try:
        with _temporarily_eval(model), torch.no_grad():
            model(*positional, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    names = {site.layer for site, _ in records}
    innermost = [
        (site, value)
        for site, value in records
        if not any(other.startswith(site.layer + ".") for other in names)
    ]
    return innermost, float_outputs


def _detect_entry_route(
    model: Module,
    positional: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[_EntryRoute, list[str]]:
    """Build the integer-input route for a call whose inputs hold no floating-point tensor.

    Returns
    -------
    tuple[_EntryRoute, list[str]]
        The route (possibly with no entry sites) and the names of modules
        whose output was a floating-point tensor, in call order.
    """

    records, float_outputs = _record_entries(model, positional, kwargs)
    integer_leaves = tuple(
        leaf for leaf in _tensor_leaves((positional, kwargs)) if not _is_float_tensor(leaf)
    )
    route = _EntryRoute(
        sites=tuple(site for site, _ in records),
        values=tuple(value for _, value in records),
        integer_leaves=integer_leaves,
    )
    return route, float_outputs


def _baseline_ids_invalid(why: str, *, code: str) -> AttributionError:
    """Build the coded refusal for baseline ids that do not mirror the integer inputs."""

    return AttributionError(
        f"baseline for integer inputs must hold integer ids mirroring them: {why}. Remedy: "
        "pass reference ids of the same shape and dtype, e.g. "
        "baseline=torch.zeros_like(inputs), or omit baseline= for a zero embedding. "
        f"{_DOCS_POINTER}",
        code=code,
    )


def _baseline_leaves(route: _EntryRoute, baseline: Any) -> list[Tensor]:
    """Return the baseline's tensor leaves, validated against the caller's integer leaves."""

    leaves = _tensor_leaves(baseline)
    if len(leaves) != len(route.integer_leaves):
        raise _baseline_ids_invalid(
            f"it has {len(leaves)} tensors where the inputs have {len(route.integer_leaves)} "
            "integer tensors",
            code="layer_path_baseline_ids_invalid",
        )
    for given, expected in zip(leaves, route.integer_leaves, strict=True):
        if given.shape != expected.shape or given.dtype != expected.dtype:
            raise _baseline_ids_invalid(
                f"got shape {tuple(given.shape)} and dtype {given.dtype} where the input has "
                f"shape {tuple(expected.shape)} and dtype {expected.dtype}",
                code="layer_path_baseline_ids_invalid",
            )
    return leaves


def _entry_baselines(
    model: Module,
    route: _EntryRoute,
    positional: tuple[Any, ...],
    kwargs: dict[str, Any],
    baseline: Any | None,
) -> tuple[Tensor, ...]:
    """Return baseline values at the entry calls.

    ``None`` gives zeros (a zero embedding, matching the zeros default of a
    floating-point input). Integer baseline ids mirroring the integer inputs
    give the entry outputs computed on those ids.

    Raises
    ------
    AttributionError
        ``layer_path_baseline_ids_invalid`` when the baseline does not mirror
        the integer inputs or does not reach the same entry calls.
    """

    if baseline is None:
        return tuple(torch.zeros_like(value) for value in route.values)
    leaves = _baseline_leaves(route, baseline)
    mapping = {id(own): given for own, given in zip(route.integer_leaves, leaves, strict=True)}
    records, _ = _record_entries(
        model, _replace_tensors(positional, mapping), _replace_tensors(kwargs, mapping)
    )
    found = {(site.layer, site.call): value for site, value in records}
    values: list[Tensor] = []
    for site, own in zip(route.sites, route.values, strict=True):
        value = found.get((site.layer, site.call))
        if value is None or value.shape != own.shape:
            raise _baseline_ids_invalid(
                f"on the baseline ids, embedding {site.layer!r} (call {site.call}) did not "
                "return an output of the same shape",
                code="layer_path_baseline_ids_invalid",
            )
        values.append(value)
    return tuple(values)
