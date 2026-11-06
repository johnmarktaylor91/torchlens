"""Fast path for :func:`torchlens.extract` when every request is a module's output.

A full capture records every operation of the forward pass, which costs many
times a plain forward on a small transformer. When each requested name is
an exact module address, the model is in eval mode and the input is one tensor
already on the model's device (or raw text, which the input door tokenizes
exactly as ``tl.trace`` does), the same values come from one plain forward with
forward hooks on those modules. The fast path returns only when the result is
the one a full capture gives: every requested module ran exactly once and
returned a single tensor. Anything else (a module called twice or never, a
tuple or dict output, an exception) restores the random-number state and
returns ``None`` so :func:`torchlens.extract` runs the full capture, which also
owns every refusal and error message.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn

from ._input_coerce import _coerce_inputs
from ._state import active_capture

#: A finalized op label (``linear_1_1``, ``relu_3_7:2``); such a name may resolve to an op.
_OP_LABEL = re.compile(r"^[A-Za-z_]+_\d+_\d+(:\d+)?$")


def _plan(model: nn.Module, layers: Iterable[str] | Mapping[str, str]) -> dict[str, str] | None:
    """Return ``output_key -> module address`` when every entry is an exact module address."""

    if isinstance(layers, Mapping):
        items = list(layers.items())
    elif isinstance(layers, (list, tuple)):
        items = [(layer, layer) for layer in layers]
    else:
        return None
    if not items:
        return None
    addresses = dict(model.named_modules())
    plan: dict[str, str] = {}
    for key, name in items:
        if not isinstance(key, str) or not isinstance(name, str):
            return None
        if not name or name not in addresses or _OP_LABEL.match(name):
            return None
        plan[key] = name
    return plan


def _is_text(x: Any) -> bool:
    """Return whether ``x`` is a string or a non-empty list of strings."""

    if isinstance(x, str):
        return True
    return isinstance(x, list) and bool(x) and all(isinstance(item, str) for item in x)


def _forward_call(model: nn.Module, x: Any) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Return the ``(args, kwargs)`` of the plain forward, or ``None`` for a full capture.

    One plain tensor is the forward's only argument. Raw text goes through the
    input door every entry shares (:func:`torchlens._input_coerce._coerce_inputs`,
    the resolution ``tl.trace`` applies), so the forward sees the same token
    tensors a capture sees; a door that does not yield plain tensors (or
    refuses) leaves the input to the full capture, which owns every error.
    """

    if type(x) is torch.Tensor:
        return (x,), {}
    if not _is_text(x):
        return None
    try:
        args, kwargs = _coerce_inputs(model, x, None)
    except Exception:  # noqa: BLE001 -- the full capture re-resolves the input and owns the error
        return None
    if kwargs:
        if args or not all(
            isinstance(k, str) and type(v) is torch.Tensor for k, v in kwargs.items()
        ):
            return None
        return (), dict(kwargs)
    return ((args,), {}) if type(args) is torch.Tensor else None


def _inputs_on_model_device(model: nn.Module, call: tuple[tuple[Any, ...], dict[str, Any]]) -> bool:
    """Return whether every forward tensor sits on the device of the model's parameters."""

    tensors = [*call[0], *call[1].values()]
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    return all(devices <= {t.device} for t in tensors)


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Snapshot the CPU and (initialized) CUDA random-number states."""

    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return torch.random.get_rng_state(), cuda


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    """Restore a snapshot taken by :func:`_rng_state`."""

    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def try_module_output_extract(
    model: nn.Module,
    x: Any,
    layers: Iterable[str] | Mapping[str, str],
    *,
    detach: bool,
) -> dict[str, torch.Tensor] | None:
    """Return module outputs from one hooked forward, or ``None`` to run a full capture.

    Parameters
    ----------
    model:
        Model passed to :func:`torchlens.extract`.
    x:
        Input passed to :func:`torchlens.extract`.
    layers:
        Requested names (list) or ``user_label -> name`` mapping.
    detach:
        Only the default ``True`` takes the fast path.

    Returns
    -------
    dict[str, torch.Tensor] | None
        Detached copies keyed as :func:`torchlens.extract` keys them, or ``None``.
    """

    if not detach or model.training or active_capture()[1]:
        return None
    plan = _plan(model, layers)
    if plan is None:
        return None
    call = _forward_call(model, x)
    if call is None or not _inputs_on_model_device(model, call):
        return None
    modules = dict(model.named_modules())
    calls: dict[str, list[Any]] = {name: [] for name in set(plan.values())}

    def recorder(name: str) -> Any:
        """Return a forward hook that appends module ``name``'s detached output."""

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            """Record one call's output, or ``None`` when it is not a plain tensor."""
            calls[name].append(output.detach().clone() if type(output) is torch.Tensor else None)

        return hook

    handles = [modules[name].register_forward_hook(recorder(name)) for name in calls]
    rng = _rng_state()
    try:
        model(*call[0], **call[1])
    except Exception:  # the full capture re-runs the forward and owns the error
        _restore_rng(rng)
        return None
    finally:
        for handle in handles:
            handle.remove()
    if any(len(seen) != 1 or seen[0] is None for seen in calls.values()):
        _restore_rng(rng)
        return None
    return {key: calls[name][0] for key, name in plan.items()}


def extract_outputs(
    model: nn.Module,
    x: Any,
    layers: Iterable[str] | Mapping[str, str],
    full_capture: Callable[..., tuple[Any, dict[str, torch.Tensor], Any]],
    *,
    detach: bool,
) -> dict[str, torch.Tensor]:
    """Return ``tl.extract``'s outputs: the hooked fast path when it applies, else ``full_capture``.

    ``full_capture`` is the package's trace-backed extract; it receives the same
    arguments (``detach`` as ``detach_saved_activations``) and returns
    ``(trace, outputs, views)``.
    """

    fast = try_module_output_extract(model, x, layers, detach=detach)
    if fast is not None:
        return fast
    _trace, outputs, _views = full_capture(model, x, layers, detach_saved_activations=detach)
    return outputs
