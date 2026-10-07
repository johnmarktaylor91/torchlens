"""Validation replay hands each op arguments with the captured ``requires_grad`` and grad mode.

ATen can pick a different kernel when an input requires grad, or when grad mode
differs: on macOS arm64 a Cin=1 3x3 conv runs ``Slow2d`` for a grad-requiring
weight and ``Winograd3x3Depthwise`` otherwise, and the two differ by 1-2 ULP,
so ``test_masked_conv`` failed forward replay there. Linux oneDNN serves both
calls with one kernel and hides the split, so these tests observe the replay
arguments themselves instead of the numeric outcome.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from typing import Any

import example_models
import pytest
import torch
import torch.nn as nn

from torchlens.validation import _autograd_grad_replay, validate_forward_pass


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
        self.arg_refs: list[weakref.ref[torch.Tensor]] = []
        original = _autograd_grad_replay.execute_with_restored_rng_autocast

        def recording(
            func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], **state: Any
        ) -> Any:
            leaves = list(_tensor_leaves((args, kwargs)))
            self.calls.append(
                {
                    "func": getattr(func, "__name__", str(func)),
                    "grad_enabled": torch.is_grad_enabled(),
                    "args": args,
                    "requires_grad": [leaf.requires_grad for leaf in leaves],
                }
            )
            self.arg_refs.extend(
                weakref.ref(leaf) for leaf in leaves if not isinstance(leaf, nn.Parameter)
            )
            return original(func, args, kwargs, **state)

        monkeypatch.setattr(_autograd_grad_replay, "execute_with_restored_rng_autocast", recording)

    def drop_arg_values(self) -> None:
        """Forget the strong references the recorder itself took."""

        for call in self.calls:
            call.pop("args")


def test_masked_conv_replay_weight_requires_grad_like_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derived masked weight of every conv reaches replay requiring grad.

    ``w1 = conv1.weight * mask`` is a non-leaf that requires grad. Before the
    fix replay received a detached copy (``requires_grad=False``) for the
    first conv, which is what sent macOS arm64 to a different kernel.
    """

    recorder = _ReplayRecorder(monkeypatch)
    model = example_models.MaskedConvModel()
    x = torch.rand(2, 1, 16, 16)

    assert validate_forward_pass(model, x)

    conv_calls = [call for call in recorder.calls if call["func"] == "conv2d"]
    assert conv_calls, "validation replayed no conv2d"
    for call in conv_calls:
        weight = call["args"][1]
        assert isinstance(weight, torch.Tensor)
        assert weight.requires_grad, "replay weight lost the captured requires_grad"
        # The input image x does not require grad, and replay must not invent it.
        assert call["grad_enabled"] is True
    first_conv_input = conv_calls[0]["args"][0]
    assert not first_conv_input.requires_grad
    # The derived weight is rebuilt as a non-leaf, like the captured value.
    first_weight = conv_calls[0]["args"][1]
    assert not first_weight.is_leaf


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


def test_replay_runs_under_each_ops_captured_grad_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay restores the grad mode each op was captured under, whatever the caller's mode."""

    recorder = _ReplayRecorder(monkeypatch)
    model = _NoGradBranchModel()
    x = torch.rand(3, 4, requires_grad=True)

    assert validate_forward_pass(model, x)

    by_func: dict[str, set[bool]] = {}
    for call in recorder.calls:
        by_func.setdefault(call["func"], set()).add(call["grad_enabled"])
    assert by_func.get("tanh") == {False}, by_func
    assert by_func.get("__add__", by_func.get("add")) == {True}, by_func
    assert {False, True} <= by_func.get("linear", set()), by_func


def test_replay_arguments_do_not_outlive_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grad-requiring replay copies and their autograd graphs are freed after validation."""

    recorder = _ReplayRecorder(monkeypatch)
    model = example_models.MaskedConvModel()
    x = torch.rand(2, 1, 16, 16)

    assert validate_forward_pass(model, x)
    assert any(any(call["requires_grad"]) for call in recorder.calls)
    recorder.drop_arg_values()
    gc.collect()

    alive = [ref for ref in recorder.arg_refs if ref() is not None]
    assert recorder.arg_refs
    assert not alive, f"{len(alive)} replay argument tensors outlived validation"
