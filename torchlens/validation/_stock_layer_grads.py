"""Stock per-module-output gradient capture for backward validation."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

_PASS_INDEX_PARSE_WARNED = False
ModuleOutputGradKey = tuple[str, int, int]


class _StockModuleGradCollector:
    """Capture stock per-module-output gradients with forward hooks.

    The collector installs one ``register_forward_hook`` on each module returned
    by ``named_modules()``. Each hook records every tensor leaf in the module
    output, keyed by ``(module_address, call_index, output_index)``. After backward,
    ``collect_grads_after_backward`` snapshots retained output gradients.
    """

    def __init__(self) -> None:
        """Initialize empty hook, call-count, and gradient state."""

        self.stock_module_output_grads: dict[ModuleOutputGradKey, torch.Tensor] = {}
        self._call_counts: dict[str, int] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._tensor_hook_handles: list[RemovableHandle] = []
        self.identity_output_addresses: set[ModuleOutputGradKey] = set()

    def install(self, model: nn.Module) -> None:
        """Install one forward hook per named module.

        Parameters
        ----------
        model:
            Model whose module outputs should be retained.
        """

        for address, module in model.named_modules():
            handle = module.register_forward_hook(_make_post_hook(self, address))
            self._hook_handles.append(handle)

    def collect_grads_after_backward(self) -> None:
        """Finalize gradients captured directly by tensor hooks."""

    def cleanup(self) -> None:
        """Remove installed hooks and drop retained output references."""

        for handle in self._hook_handles:
            handle.remove()
        for handle in self._tensor_hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._tensor_hook_handles.clear()


def _make_post_hook(
    collector: _StockModuleGradCollector,
    address: str,
) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    """Build a closure-bound module forward hook.

    Parameters
    ----------
    collector:
        Collector that owns retained-output state.
    address:
        Named-module address associated with this hook.

    Returns
    -------
    Callable[[nn.Module, tuple[Any, ...], Any], None]
        Forward hook.
    """

    def _post_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """Register gradient hooks for tensor leaves emitted by one module call."""

        collector._call_counts[address] = collector._call_counts.get(address, 0) + 1
        call_index = collector._call_counts[address]
        key = (address, call_index)

        output_leaves = _tensor_leaves(output)
        if not output_leaves:
            return
        input_ids = {id(leaf) for leaf in _tensor_leaves(args)}
        for output_index, leaf_out in enumerate(output_leaves):
            output_key = (*key, output_index)
            if isinstance(module, nn.Identity) or id(leaf_out) in input_ids:
                collector.identity_output_addresses.add(output_key)
                continue
            if leaf_out.requires_grad or leaf_out.grad_fn is not None:
                handle = leaf_out.register_hook(_make_tensor_grad_hook(collector, output_key))
                collector._tensor_hook_handles.append(handle)

    return _post_hook


def _make_tensor_grad_hook(
    collector: _StockModuleGradCollector,
    key: ModuleOutputGradKey,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a tensor hook that snapshots one module-output gradient.

    Parameters
    ----------
    collector:
        Collector receiving detached gradient clones.
    key:
        Module-address and call-index key for the hooked output.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Tensor hook that records a clone and returns the incoming gradient unchanged.
    """

    def _tensor_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        """Snapshot a stock module-output gradient without retaining output grads."""

        collector.stock_module_output_grads[key] = grad.detach().clone()
        return grad

    return _tensor_grad_hook


def _first_leaf_tensor(obj: Any) -> torch.Tensor | None:
    """Return the first non-parameter tensor leaf in DFS order.

    Parameters
    ----------
    obj:
        Arbitrary tensor container.

    Returns
    -------
    torch.Tensor | None
        First tensor leaf, or None when no tensor leaf is present.
    """

    if isinstance(obj, torch.Tensor):
        if isinstance(obj, nn.Parameter):
            return None
        return obj
    if (
        hasattr(obj, "to_tuple")
        and callable(getattr(obj, "to_tuple", None))
        and isinstance(obj, dict)
    ):
        to_tuple = getattr(obj, "to_tuple")
        try:
            leaf = _first_leaf_tensor(to_tuple())
        except Exception:
            leaf = None
        if leaf is not None:
            return leaf
    if isinstance(obj, (tuple, list)):
        for item in obj:
            leaf = _first_leaf_tensor(item)
            if leaf is not None:
                return leaf
        return None
    if isinstance(obj, dict):
        for value in obj.values():
            leaf = _first_leaf_tensor(value)
            if leaf is not None:
                return leaf
        return None
    if hasattr(obj, "to_tuple") and callable(getattr(obj, "to_tuple", None)):
        try:
            return _first_leaf_tensor(obj.to_tuple())
        except Exception:
            return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            leaf = _first_leaf_tensor(getattr(obj, field.name))
            if leaf is not None:
                return leaf
    return None


def _tensor_leaves(obj: Any) -> list[torch.Tensor]:
    """Return non-parameter tensor leaves in DFS order.

    Parameters
    ----------
    obj:
        Arbitrary tensor container.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves, excluding ``nn.Parameter`` instances.
    """

    if isinstance(obj, torch.Tensor):
        if isinstance(obj, nn.Parameter):
            return []
        return [obj]
    if (
        hasattr(obj, "to_tuple")
        and callable(getattr(obj, "to_tuple", None))
        and isinstance(obj, dict)
    ):
        to_tuple = getattr(obj, "to_tuple")
        try:
            return _tensor_leaves(to_tuple())
        except Exception:
            return []
    if isinstance(obj, (tuple, list)):
        leaves: list[torch.Tensor] = []
        for item in obj:
            leaves.extend(_tensor_leaves(item))
        return leaves
    if isinstance(obj, dict):
        leaves = []
        for value in obj.values():
            leaves.extend(_tensor_leaves(value))
        return leaves
    if hasattr(obj, "to_tuple") and callable(getattr(obj, "to_tuple", None)):
        try:
            return _tensor_leaves(obj.to_tuple())
        except Exception:
            return []
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        leaves = []
        for field in dataclasses.fields(obj):
            leaves.extend(_tensor_leaves(getattr(obj, field.name)))
        return leaves
    return []


def _innermost_module_address(layer: Any) -> str | None:
    """Return the innermost module address from a layer's module stack.

    Parameters
    ----------
    layer:
        Layer-like object with a ``modules`` field.

    Returns
    -------
    str | None
        Innermost module address, or None for module-less layers.
    """

    modules = getattr(layer, "modules", None) or []
    if not modules:
        return None
    last = modules[-1]
    if isinstance(last, tuple):
        return str(last[0])
    if isinstance(last, str):
        return last.rsplit(":", 1)[0]
    return None


def _pass_index_from_layer_modules(layer: Any) -> int | None:
    """Return the innermost module pass index from a layer's module stack.

    Parameters
    ----------
    layer:
        Layer-like object with a ``modules`` field.

    Returns
    -------
    int | None
        Parsed pass index, or None when absent or malformed.
    """

    global _PASS_INDEX_PARSE_WARNED
    modules = getattr(layer, "modules", None) or []
    if not modules:
        return None
    last = modules[-1]
    if isinstance(last, tuple) and len(last) >= 2:
        try:
            return int(last[1])
        except (TypeError, ValueError):
            pass
    if isinstance(last, str) and ":" in last:
        try:
            return int(last.rsplit(":", 1)[1])
        except ValueError:
            pass
    if not _PASS_INDEX_PARSE_WARNED:
        warnings.warn(
            f"Could not parse module pass index from {last!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        _PASS_INDEX_PARSE_WARNED = True
    return None


def _candidate_module_call_for(trace: Any, address: str, call_index: int) -> Any | None:
    """Return a candidate ``ModuleCall`` by address and call index.

    Parameters
    ----------
    trace:
        Candidate trace.
    address:
        Module address.
    call_index:
        One-based module call index.

    Returns
    -------
    Any | None
        Matching module call log, if present.
    """

    modules = getattr(trace, "modules", None)
    if modules is None:
        return None
    key = f"{address}:{call_index}"
    pass_dict = getattr(modules, "_pass_dict", {})
    if key in pass_dict:
        return pass_dict[key]
    if hasattr(modules, "__contains__") and key in modules:
        return modules[key]
    return None


def _candidate_root_module(trace: Any) -> Any | None:
    """Return the candidate root module log if present.

    Parameters
    ----------
    trace:
        Candidate trace.

    Returns
    -------
    Any | None
        Root module log.
    """

    modules = getattr(trace, "modules", None)
    if modules is None:
        return None
    pass_dict = getattr(modules, "_pass_dict", {})
    if "self:1" in pass_dict:
        return pass_dict["self:1"]
    if hasattr(modules, "__contains__") and "self" in modules:
        return modules["self"]
    return None


def _stock_layer_grads(
    model: nn.Module,
    input_args: Any,
    input_kwargs: Mapping[str, Any],
    *,
    loss_fn: Callable[[Any], torch.Tensor],
    random_seed: int,
    state_dict_snapshot: Mapping[str, torch.Tensor],
) -> tuple[dict[ModuleOutputGradKey, torch.Tensor], set[ModuleOutputGradKey]]:
    """Run a stock forward/backward pass and collect module-output grads.

    Parameters
    ----------
    model:
        Model to execute.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.
    loss_fn:
        Loss function mapping model output to a scalar tensor.
    random_seed:
        Seed used for deterministic stock execution.
    state_dict_snapshot:
        State dict restored after the stock pass.

    Returns
    -------
    tuple[dict[ModuleOutputGradKey, torch.Tensor], set[ModuleOutputGradKey]]
        Captured stock module-output grads and identity-output addresses.
    """

    from ..utils.rng import set_random_seed

    collector = _StockModuleGradCollector()
    collector.install(model)
    try:
        set_random_seed(random_seed)
        output = model(*input_args, **dict(input_kwargs))
        loss = loss_fn(output)
        loss.backward()  # type: ignore[no-untyped-call]
        collector.collect_grads_after_backward()
        return dict(collector.stock_module_output_grads), set(collector.identity_output_addresses)
    finally:
        collector.cleanup()
        model.load_state_dict(state_dict_snapshot)
