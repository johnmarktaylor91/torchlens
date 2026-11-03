"""Validation replay hands each op arguments with the captured ``requires_grad`` and grad mode.

ATen can pick a different kernel when an input requires grad, or when grad mode
differs: on macOS arm64 a Cin=1 3x3 conv runs ``Slow2d`` for a grad-requiring
weight and ``Winograd3x3Depthwise`` otherwise, and the two differ by 1-2 ULP,
so ``test_masked_conv`` failed forward replay there. Linux oneDNN serves both
calls with one kernel and hides the split, so these tests observe the replay
arguments and modes themselves instead of the numeric outcome.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

import example_models
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.validation import _autograd_grad_replay, validate_forward_pass
from torchlens.validation.core import validate_saved_outs


def _tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor inside a nested list/tuple/dict structure."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)


class _ReplayRecorder:
    """Wrap the plain-op replay executor and record what each replay received."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install the recording wrapper.

        Parameters
        ----------
        monkeypatch:
            Pytest monkeypatch fixture that undoes the wrapper after the test.
        """

        self.calls: list[dict[str, Any]] = []
        self.refs: list[weakref.ref[torch.Tensor]] = []
        original = _autograd_grad_replay.execute_with_restored_rng_autocast

        def recording(
            func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], **state: Any
        ) -> Any:
            self.calls.append(
                {
                    "func": getattr(func, "__name__", str(func)),
                    "grad_enabled": torch.is_grad_enabled(),
                    "inference_mode": torch.is_inference_mode_enabled(),
                    "args": args,
                }
            )
            output = original(func, args, kwargs, **state)
            self.refs.extend(
                weakref.ref(leaf)
                for leaf in _tensor_leaves((args, kwargs, output))
                if not isinstance(leaf, nn.Parameter)
            )
            return output

        monkeypatch.setattr(_autograd_grad_replay, "execute_with_restored_rng_autocast", recording)

    def calls_to(self, func_name: str) -> list[dict[str, Any]]:
        """Return the recorded replays of one function, failing when there are none."""

        calls = [call for call in self.calls if call["func"] == func_name]
        assert calls, f"validation replayed no {func_name}"
        return calls

    def drop_arg_values(self) -> None:
        """Forget the strong references the recorder itself took."""

        for call in self.calls:
            call.pop("args")


def _assert_conv_weights_require_grad(recorder: _ReplayRecorder) -> None:
    """Every conv replay ran with grad enabled on a grad-requiring non-leaf weight."""

    derived_weights = 0
    for call in recorder.calls_to("conv2d"):
        weight = call["args"][1]
        assert isinstance(weight, torch.Tensor)
        assert weight.requires_grad, "replay weight lost the captured requires_grad"
        assert call["grad_enabled"] is True
        assert call["inference_mode"] is False
        if not isinstance(weight, nn.Parameter):
            # A masked weight (not the live parameter the last conv uses) was a
            # non-leaf at capture and must be rebuilt as one.
            assert not weight.is_leaf, "the derived weight was a non-leaf at capture"
            derived_weights += 1
    assert derived_weights, "no conv replay received a derived weight"


def _capture_for_replay(model: nn.Module, x: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Capture ``model`` with the saved arguments replay needs, plus its ground truth."""

    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(save_args=True, save_rng=True),
    )
    return trace, model(x)


@pytest.mark.parametrize("ambient", ["no_grad", "inference_mode"])
def test_validation_under_another_ambient_mode_replays_the_captured_mode(
    monkeypatch: pytest.MonkeyPatch, ambient: str
) -> None:
    """A grad-mode capture validated inside no_grad or inference_mode replays with grad."""

    model = example_models.MaskedConvModel()
    trace, ground_truth = _capture_for_replay(model, torch.rand(2, 1, 16, 16))
    recorder = _ReplayRecorder(monkeypatch)
    ambient_ctx = torch.no_grad() if ambient == "no_grad" else torch.inference_mode()

    with ambient_ctx:
        result = validate_saved_outs(trace, [ground_truth])

    assert bool(result)
    _assert_conv_weights_require_grad(recorder)


class _NoGradBranchModel(nn.Module):
    """One op runs under ``torch.no_grad()`` inside an otherwise grad-enabled forward."""

    def __init__(self) -> None:
        """Build the two linear layers."""

        super().__init__()
        self.inner = nn.Linear(4, 4)
        self.outer = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``inner`` without grad and ``outer`` with grad."""

        with torch.no_grad():
            frozen = torch.tanh(self.inner(x))
        return self.outer(frozen) + x


class _LeafAndOpOutputModel(nn.Module):
    """One multiply takes an op output (a non-leaf) and an in-forward grad leaf.

    TorchLens feeds the model clones of its inputs, so a grad-requiring model
    input is a non-leaf by the time any op sees it; a factory tensor built
    with ``requires_grad=True`` inside ``forward`` is a genuine leaf.
    """

    def __init__(self) -> None:
        """Build the linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale a linear output by a grad-requiring factory tensor."""

        scale = torch.full((4,), 3.0, requires_grad=True)
        return torch.mul(self.lin(x), scale)


def test_replay_rebuilds_leaves_as_leaves_and_op_outputs_as_non_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leafness follows the producing op's recorded autograd node."""

    model = _LeafAndOpOutputModel()
    trace, ground_truth = _capture_for_replay(model, torch.rand(3, 4))
    recorder = _ReplayRecorder(monkeypatch)

    assert bool(validate_saved_outs(trace, [ground_truth]))

    for call in recorder.calls_to("mul"):
        op_output, factory_leaf = call["args"][0], call["args"][1]
        assert op_output.requires_grad and not op_output.is_leaf
        assert factory_leaf.requires_grad and factory_leaf.is_leaf


class _RequiresGradToggleModel(nn.Module):
    """Re-grad a detached output, then turn grad off again (legal only on a leaf)."""

    def __init__(self) -> None:
        """Build the linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Toggle ``requires_grad`` on a detached sigmoid output."""

        frozen = torch.sigmoid(self.lin(x)).detach()
        frozen.requires_grad_(True)
        frozen.requires_grad_(False)
        return frozen * 2.0 + self.lin(x)


def test_requires_grad_toggle_replays_on_a_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    """``requires_grad_(False)`` replays on a leaf, as its successful capture proves it had."""

    recorder = _ReplayRecorder(monkeypatch)

    assert validate_forward_pass(_RequiresGradToggleModel(), torch.rand(3, 4))

    for call in recorder.calls_to("requires_grad_"):
        assert call["args"][0].is_leaf
