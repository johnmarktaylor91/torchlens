"""Hook axis: capture-time TorchLens hooks as options rows, torch's own hooks as model rows.

A TorchLens hook is ``fn(out, *, hook)`` (``hook`` is the per-firing context)
and runs at capture through ``tl.trace(..., intervene=tl.when(site, fn))``.
The options rows aim at ``tl.func("linear")``, the op almost every model row
carries, so each hook behaviour is crossed with every architecture; a model
without that op records a zero-fire warning and runs as a plain capture.

Default validation re-traces the model without the options row, so
``replay_equals_eager`` and ``invariants_pass`` judge the plain capture; the
hook rows are judged by the capture itself, the save/load round trip of the
intervened trace, the typed-refusal, torch-clean and memory oracles.

Hooks attached after capture (``Trace.attach_hooks``, ``Trace.set``) and a
TorchLens hook removed through its handle mid-pass need a post-capture step
the runner does not have; they are listed as remaining work, not rows.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidArgumentError

from ..generator import Inputs, ModelSpec, OptionsSpec
from .concurrency import StressUserError, side_capture

WIDTH = 8
HOOK_SITE = "linear"
BUFFER_SITE = "buffer_1:1"


def hook_reads(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Read the output and return it unchanged."""

    float(out.detach().float().abs().sum())
    return out


def hook_scales(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return a new, scaled output."""

    return out * 0.5


def hook_scales_in_place(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Scale the output in place and return the same tensor."""

    with torch.no_grad():
        out.mul_(0.5)
    return out


def hook_raises_typed(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Refuse with a typed TorchLens error carrying a stable code."""

    raise InvalidArgumentError(
        "stress hook refuses on purpose",
        code="stress_hook_refusal",
        remedy="none: the stress row raises by design",
        argument="hook",
    )


def hook_raises_plain(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Raise a plain user exception."""

    raise StressUserError("stress hook raises a plain exception on purpose")


def hook_nested_capture(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Start a second capture from inside a capture-time hook."""

    side_capture(torch.zeros(1, WIDTH))
    return out


def _on_linear(fn: Any, **kwargs: Any) -> dict[str, Any]:
    return {"intervene": tl.when(tl.func(HOOK_SITE), fn, **kwargs)}


OPTIONS: tuple[OptionsSpec, ...] = (
    OptionsSpec(
        "hook_reads",
        "a capture-time hook on every linear output that only reads",
        trace_kwargs=_on_linear(hook_reads),
    ),
    OptionsSpec(
        "hook_modifies_out",
        "a capture-time hook that returns a new, scaled linear output",
        trace_kwargs=_on_linear(hook_scales),
    ),
    OptionsSpec(
        "hook_inplace_out",
        "a capture-time hook that scales its output in place and returns it",
        trace_kwargs=_on_linear(hook_scales_in_place),
    ),
    OptionsSpec(
        "hook_modifies_input",
        "a capture-time hook on the model's first input that scales it (a pre-hook)",
        trace_kwargs={"intervene": tl.when(tl.model_input(0), hook_scales)},
    ),
    OptionsSpec(
        "hook_backward_dir",
        "a scaling hook aimed at linear grads (direction='backward') during capture",
        trace_kwargs=_on_linear(hook_scales, direction="backward"),
    ),
    OptionsSpec(
        "hook_buffer_read",
        "a scaling hook on the first read of a never-written buffer (behaviour net F45)",
        trace_kwargs={"intervene": tl.when(tl.label(BUFFER_SITE), hook_scales)},
    ),
    OptionsSpec(
        "hook_raises_typed",
        "a capture-time hook that raises a typed TorchLens error: it must surface typed",
        trace_kwargs=_on_linear(hook_raises_typed),
    ),
    OptionsSpec(
        "hook_raises_plain",
        "a capture-time hook that raises a plain exception: capture must unwind clean",
        trace_kwargs=_on_linear(hook_raises_plain),
    ),
    OptionsSpec(
        "hook_nested_capture",
        "a capture-time hook that starts a second capture: typed refusal expected",
        trace_kwargs=_on_linear(hook_nested_capture),
    ),
)


class BufferReader(nn.Module):
    """Read a registered buffer it never writes (the F45 replay-threading shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(WIDTH, WIDTH)
        self.register_buffer("mask", (torch.arange(WIDTH) % 2).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x)) * self.mask


def _scale_input_pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (args[0] * 0.5, *args[1:])


def _shift_output_hook(_module: nn.Module, _args: tuple[Any, ...], out: Any) -> Any:
    return out + 1.0


def _read_output_hook(module: nn.Module, _args: tuple[Any, ...], out: Any) -> None:
    module.last_norm = float(out.detach().norm())


class TorchHooked(nn.Module):
    """Two linears carrying torch's own pre-hook and forward hooks, left installed."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc1.register_forward_pre_hook(_scale_input_pre_hook)
        self.fc1.register_forward_hook(_shift_output_hook)
        self.fc2.register_forward_hook(_read_output_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TorchHookRemovedMidPass(nn.Module):
    """A torch forward hook on ``fc2`` is removed by ``fc1``'s hook during the same pass."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc3 = nn.Linear(WIDTH, WIDTH)
        self.fc1.register_forward_hook(self._remove_fc2_hook)
        self._fc2_handle: Any = None

    def _remove_fc2_hook(self, _module: nn.Module, _args: tuple[Any, ...], out: Any) -> Any:
        if self._fc2_handle is not None:
            self._fc2_handle.remove()
            self._fc2_handle = None
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._fc2_handle = self.fc2.register_forward_hook(_shift_output_hook)
        h = self.fc1(x)
        h = self.fc2(torch.relu(h))
        return self.fc3(torch.relu(h))


def _input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH),), {})


def _model(row_id: str, factory: type[nn.Module], reason: str) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=factory,
        example_input=_input,
        reason=reason,
        tags=frozenset({"hooks"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _model(
        "hk_buffer_reader",
        BufferReader,
        "reads a never-written buffer, the site the hook_buffer_read row aims at",
    ),
    _model(
        "hk_torch_hooks",
        TorchHooked,
        "torch pre-hook and forward hooks that edit inputs and outputs, installed before capture",
    ),
    _model(
        "hk_torch_hook_removed",
        TorchHookRemovedMidPass,
        "one torch forward hook removes another during the same forward pass",
    ),
)
