"""Perturbation retry at a TorchScript region that thresholds a tensor (spiking neuron).

spikingjelly's LIF node scripts its single step: charge the membrane ``v`` from the input,
then fire ``(v >= v_threshold).to(x)``. With no neuron near the threshold every spike is
zero, and the plain unit steps on ``v`` or the input stay below the threshold, so the real
edge looked dead and validation failed it as ``perturbation_insensitive`` (Menagerie m10178).
A scripted body that thresholds a tensor now gets the geometric magnitude ladder explicit
quantizers get, so a larger step crosses the threshold and proves the edge. Armed negative:
a membrane that the body multiplies by zero before the threshold still fails under every rung.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from torchlens.backends.torch._script_boundary import SCRIPT_FUNCTION_FUNC_NAME
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation._core import _check_whether_func_on_saved_parents_yields_saved_tensor
from torchlens.validation._script_metadata_proof import script_body_thresholds_tensor

pytestmark = pytest.mark.skipif(
    not HAS_SCRIPT_FUNCTION_CALL_HOOK, reason="torch build without a ScriptFunction call hook"
)


@torch.jit.script
def lif_fire(x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float) -> torch.Tensor:
    """Charge with decay input, then fire where the membrane reaches the threshold."""

    v = v + (x - v) / tau
    return (v >= v_threshold).to(x)


@torch.jit.script
def lif_fire_dead_membrane(
    x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
) -> torch.Tensor:
    """Planted bug: the membrane is multiplied by zero, so no finite value reaches the spike."""

    charged = x / tau + v * 0.0
    return (charged >= v_threshold).to(x)


@torch.jit.script
def scaled(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain arithmetic with no threshold."""

    return x * 2.0 + v


class _SpikingLayer(nn.Module):
    """A linear layer feeding a scripted LIF step over a stored membrane buffer."""

    def __init__(self, step: Any) -> None:
        super().__init__()
        self.step = step
        self.fc = nn.Linear(4, 6)
        with torch.no_grad():
            self.fc.weight.mul_(0.1)
            self.fc.bias.zero_()
        self.register_buffer("v", torch.linspace(-0.7, 0.3, 6).repeat(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(self.fc(x), self.v, 1.0, 2.0)


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    torch.manual_seed(0)
    return tl.trace(model, x, capture=CaptureOptions(save_args=True, seed=0), save="all")


def _script_op(trace: Any) -> Any:
    return next(op for op in trace._layer_list if op.func_name == SCRIPT_FUNCTION_FUNC_NAME)


def _membrane_edge(trace: Any) -> Any:
    op = _script_op(trace)
    parent = op.call.parent_arg_positions["args"][1]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )


def _input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 4)


def test_silent_spiking_step_validates() -> None:
    """No neuron fires, yet a ladder step on the membrane crosses the threshold."""

    model = _SpikingLayer(lif_fire)
    trace = _capture(model, _input())
    op = _script_op(trace)
    assert not bool(op.out.any()), "the stand-in must not fire on the saved values"
    result = _membrane_edge(trace)
    assert (result.decision, result.reason) == ("validated", "perturbation_changed"), (
        result.decision,
        result.reason,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, _input(), seed=0)


def test_dead_membrane_still_fails() -> None:
    """A membrane the body multiplies by zero stays unchanged under every ladder rung."""

    trace = _capture(_SpikingLayer(lif_fire_dead_membrane), _input())
    result = _membrane_edge(trace)
    assert (result.decision, result.reason) == ("failed", "perturbation_insensitive"), (
        result.decision,
        result.reason,
    )


def test_threshold_detection_reads_the_graph() -> None:
    """A thresholding body qualifies; a non-script op does not."""

    trace = _capture(_SpikingLayer(lif_fire), _input())
    assert script_body_thresholds_tensor(_script_op(trace))
    linear = next(op for op in trace._layer_list if op.func_name == "linear")
    assert not script_body_thresholds_tensor(linear)


class _ScaledLayer(_SpikingLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scaled(self.fc(x), self.v)


def test_arithmetic_script_body_has_no_threshold() -> None:
    """A scripted body with no threshold node keeps the plain unit steps."""

    trace = _capture(_ScaledLayer(scaled), _input())
    assert not script_body_thresholds_tensor(_script_op(trace))
