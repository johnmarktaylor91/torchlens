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


class _StockModuleGradCollector:
    """Capture stock per-module-output gradients with forward hooks.

    The collector installs one ``register_forward_hook`` on each module returned
    by ``named_modules()``. Each hook retains the first tensor leaf in the module
    output, keyed by ``(module_address, call_index)``. After backward,
    ``collect_grads_after_backward`` snapshots retained output gradients.
    """

    def __init__(self) -> None:
        """Initialize empty hook, call-count, and gradient state."""

        self._retained_outputs: dict[tuple[str, int], torch.Tensor] = {}
        self.stock_module_output_grads: dict[tuple[str, int], torch.Tensor] = {}
        self._call_counts: dict[str, int] = {}
        self._hook_handles: list[RemovableHandle] = []
        self.identity_output_addresses: set[tuple[str, int]] = set()

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
        """Collect detached gradients from retained module outputs."""

        for key, retained in self._retained_outputs.items():
            grad = getattr(retained, "grad", None)
            if grad is not None:
                self.stock_module_output_grads[key] = grad.detach().clone()

    def cleanup(self) -> None:
        """Remove installed hooks and drop retained output references."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._retained_outputs.clear()


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
        """Retain the first tensor leaf emitted by one module call."""

        collector._call_counts[address] = collector._call_counts.get(address, 0) + 1
        call_index = collector._call_counts[address]
        key = (address, call_index)

        leaf_out = _first_leaf_tensor(output)
        if leaf_out is None:
            return
        leaf_in = _first_leaf_tensor(args) if args else None
        identity_output = leaf_out is leaf_in
        if (
            not identity_output
            and leaf_in is not None
            and leaf_out.shape == leaf_in.shape
            and torch.allclose(leaf_out.detach(), leaf_in.detach(), atol=1e-5, rtol=1e-5)
        ):
            identity_output = True
        if isinstance(module, nn.Identity) or identity_output:
            collector.identity_output_addresses.add(key)
            return
        if leaf_out.requires_grad or leaf_out.grad_fn is not None:
            leaf_out.retain_grad()
            collector._retained_outputs[key] = leaf_out

    return _post_hook


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
) -> tuple[dict[tuple[str, int], torch.Tensor], set[tuple[str, int]]]:
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
    tuple[dict[tuple[str, int], torch.Tensor], set[tuple[str, int]]]
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
