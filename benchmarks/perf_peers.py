"""Peer-tool capture implementations for TorchLens performance benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from typing import Any, Iterator

import torch
from torch import nn


class PeerSkip(RuntimeError):
    """Structured peer skip raised when a peer package or mode is unavailable."""

    def __init__(self, peer: str, reason: str) -> None:
        """Initialize a peer skip.

        Parameters
        ----------
        peer:
            Peer tool name.
        reason:
            Skip reason.
        """

        super().__init__(reason)
        self.peer = peer
        self.reason = reason


def _require_package(module_name: str, peer: str) -> None:
    """Require an importable peer package.

    Parameters
    ----------
    module_name:
        Python module name.
    peer:
        Human-readable peer name.
    """

    if importlib.util.find_spec(module_name) is None:
        raise PeerSkip(peer, f"{module_name} is not importable")


def _detach_tree(value: Any) -> Any:
    """Detach tensors in a nested output structure.

    Parameters
    ----------
    value:
        Module output.

    Returns
    -------
    Any
        Detached structure suitable for retention in a hook cache.
    """

    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _module_names(model: nn.Module) -> list[str]:
    """Return all non-root module names.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    list[str]
        Named-module paths excluding the root module.
    """

    return [name for name, _module in model.named_modules() if name]


def run_vanilla_hooks_manual_dict(model: nn.Module, x: Any) -> dict[str, Any]:
    """Capture all module outputs with direct ``register_forward_hook`` calls.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Forward input.

    Returns
    -------
    dict[str, Any]
        Detached outputs keyed by module name.
    """

    outputs: dict[str, Any] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not name:
            continue

        def hook(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, key: str = name
        ) -> None:
            outputs[key] = _detach_tree(output)

        handles.append(module.register_forward_hook(hook))
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


@contextmanager
def _capture_modules(model: nn.Module) -> Iterator[dict[str, Any]]:
    """Context manager wrapping vanilla module hooks.

    Parameters
    ----------
    model:
        Model to instrument.

    Yields
    ------
    dict[str, Any]
        Mutable output cache populated during the forward pass.
    """

    outputs: dict[str, Any] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not name:
            continue

        def hook(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, key: str = name
        ) -> None:
            outputs[key] = _detach_tree(output)

        handles.append(module.register_forward_hook(hook))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def run_vanilla_hooks_context_manager(model: nn.Module, x: Any) -> dict[str, Any]:
    """Capture all module outputs with a context-manager hook helper.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Forward input.

    Returns
    -------
    dict[str, Any]
        Detached outputs keyed by module name.
    """

    with _capture_modules(model) as outputs:
        model(x)
    return outputs


def run_baukit(model: nn.Module, x: Any) -> dict[str, Any]:
    """Capture module outputs with baukit ``TraceDict``.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Forward input.

    Returns
    -------
    dict[str, Any]
        Detached outputs keyed by module name.
    """

    _require_package("baukit", "baukit")
    from baukit import TraceDict

    names = _module_names(model)
    if not names:
        raise PeerSkip("baukit", "model has no named child modules")
    with TraceDict(model, names) as traces:
        model(x)
    return {name: _detach_tree(traces[name].output) for name in names if name in traces}


def run_transformer_lens(model: nn.Module, x: Any) -> dict[str, Any]:
    """Capture activations with TransformerLens ``run_with_cache``.

    Parameters
    ----------
    model:
        HookedTransformer model.
    x:
        Token tensor.

    Returns
    -------
    dict[str, Any]
        Detached cache entries.
    """

    _require_package("transformer_lens", "transformer_lens")
    if not hasattr(model, "run_with_cache"):
        raise PeerSkip("transformer_lens", "model does not expose run_with_cache")
    _logits, cache = model.run_with_cache(x)
    return {str(key): _detach_tree(value) for key, value in cache.items()}


def run_nnsight(model: nn.Module, x: Any) -> dict[str, Any]:
    """Attempt a generic nnsight trace capture.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Forward input.

    Returns
    -------
    dict[str, Any]
        Captured values when the generic nnsight API is applicable.
    """

    _require_package("nnsight", "nnsight")
    if not hasattr(model, "trace"):
        raise PeerSkip("nnsight", "generic nn.Module does not expose nnsight trace")
    raise PeerSkip("nnsight", "generic all-module nnsight capture was not applicable")
