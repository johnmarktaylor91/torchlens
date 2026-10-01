"""AUD-CODE 4.3 regression: the admitted-meta forward's scoped torch swaps.

(a) The W1-AC null-autocast shim replaced ``torch.autocast`` (a class) with
a FUNCTION for the duration of an admitted forward, so ``isinstance(ctx,
torch.autocast)`` / ``inspect.isclass`` / decorator use broke inside the
scope. The shim is now a subclass of the shipped class.
(b) The D19 DeviceContext absorption restored popped device contexts ON TOP
of the kept modes, losing the caller's interleaving. The exact original
stack order is now restored.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.utils._torch_compat import autocast_is_enabled

pytestmark = pytest.mark.smoke


class _IntrospectingAutocastModel(nn.Module):
    """A forward that inspects torch.autocast the way user code does."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.seen: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = torch.autocast(device_type=x.device.type, enabled=False)
        self.seen["isclass"] = inspect.isclass(torch.autocast)
        self.seen["isinstance"] = isinstance(ctx, torch.autocast)
        self.seen["issubclass_amp"] = issubclass(torch.autocast, torch.amp.autocast_mode.autocast)
        self.seen["name"] = torch.autocast.__name__
        with ctx:
            y = self.fc(x)

        @torch.autocast(device_type=x.device.type, enabled=False)
        def _decorated(t: torch.Tensor) -> torch.Tensor:
            return torch.relu(t)

        return _decorated(y)


def test_autocast_stays_a_class_inside_the_admitted_forward() -> None:
    with torch.device("meta"):
        model = _IntrospectingAutocastModel()
    model.eval()
    original = torch.autocast
    trace = tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    try:
        assert trace.outcome.status.value == "complete"
        assert model.seen == {
            "isclass": True,
            "isinstance": True,
            "issubclass_amp": True,
            "name": "autocast",
        }
        assert torch.autocast is original, "the shipped class is restored after the forward"
    finally:
        trace.cleanup()


def test_shim_class_delegates_non_null_calls_to_the_shipped_class() -> None:
    from torchlens.capture._weightsfree_admission import _null_on_meta_autocast_class

    shim = _null_on_meta_autocast_class(torch.autocast)
    assert issubclass(shim, torch.autocast)
    null = shim("meta", enabled=False)
    assert isinstance(null, torch.autocast)
    with null:
        pass
    real = shim("cpu", dtype=torch.bfloat16, enabled=True)
    assert real.fast_dtype == torch.bfloat16
    with real:
        assert autocast_is_enabled("cpu")
    assert not autocast_is_enabled("cpu")


class _Passthrough(TorchFunctionMode):
    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        return func(*args, **(kwargs or {}))


class _Other(_Passthrough):
    pass


def test_device_context_absorption_restores_the_exact_interleaving() -> None:
    from torch.overrides import _get_current_function_mode_stack as stack

    from torchlens.capture._weightsfree_admission import _absorbed_ambient_device_context
    from torchlens.utils._torch_compat import get_torch_function_stack_surgery

    if get_torch_function_stack_surgery() is None:
        pytest.skip("mode-stack surgery unavailable; absorption refuses typed instead")
    with _Passthrough(), torch.device("meta"), _Other():
        before = list(stack())
        assert [type(m).__name__ for m in before] == ["_Passthrough", "DeviceContext", "_Other"]
        with _absorbed_ambient_device_context():
            inside = list(stack())
            assert [type(m).__name__ for m in inside] == ["_Passthrough", "_Other"]
            assert inside[0] is before[0] and inside[1] is before[2]
        after = list(stack())
        assert len(after) == len(before)
        assert all(a is b for a, b in zip(after, before, strict=True)), (
            "the caller's mode-stack interleaving must survive the admitted forward"
        )
