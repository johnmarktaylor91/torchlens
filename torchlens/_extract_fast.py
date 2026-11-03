"""Fast path for :func:`torchlens.extract` when every request is a module's output.

A full capture records every operation of the forward pass (about 15-20x the
cost of the plain forward on a small transformer). When each requested name is
an exact module address, the model is in eval mode and the input is one tensor
already on the model's device, the same values come from one plain forward with
forward hooks on those modules. The fast path returns only when the result is
the one a full capture gives: every requested module ran exactly once and
returned a single tensor. Anything else (a module called twice or never, a
tuple or dict output, an exception) restores the random-number state and
returns ``None`` so :func:`torchlens.extract` runs the full capture, which also
owns every refusal and error message.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

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


def _input_on_model_device(model: nn.Module, x: Any) -> bool:
    """Return whether ``x`` is one plain tensor on the device of the model's parameters."""

    if type(x) is not torch.Tensor:
        return False
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    return devices <= {x.device}


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
    detach_saved_activations: bool,
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
    detach_saved_activations:
        Only the default ``True`` takes the fast path.

    Returns
    -------
    dict[str, torch.Tensor] | None
        Detached copies keyed as :func:`torchlens.extract` keys them, or ``None``.
    """

    if not detach_saved_activations or model.training or active_capture()[1]:
        return None
    plan = _plan(model, layers)
    if plan is None or not _input_on_model_device(model, x):
        return None
    modules = dict(model.named_modules())
    calls: dict[str, list[Any]] = {name: [] for name in set(plan.values())}

    def recorder(name: str) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            calls[name].append(output.detach().clone() if type(output) is torch.Tensor else None)

        return hook

    handles = [modules[name].register_forward_hook(recorder(name)) for name in calls]
    rng = _rng_state()
    try:
        model(x)
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
