"""A06 capture-options truth: ``save_grads`` predicates are honored.

WALKTHROUGH list-A row 10 (first clause): the trace path silently DISCARDED
selector/callable ``save_grads`` predicates -- ``callable -> "all"`` -- so a
narrowing predicate saved EVERY gradient (memory blowups on real models).
Predicates now ride the deferred-gradient path: selectors resolve through the
unchecked resolver, bare callables evaluate per finalized op with a strict
bool contract, and only matching ops get gradient hooks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions


class ThreeStep(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _grad_labels(trace: object) -> list[str]:
    return [op.layer_label for op in trace if getattr(op, "grad", None) is not None]  # type: ignore[attr-defined]


def _backward(trace: object) -> None:
    list(trace)[-1].out.sum().backward()  # type: ignore[attr-defined,call-overload]


@pytest.mark.smoke
def test_save_grads_callable_restricts_retention() -> None:
    """A bare callable keeps gradients ONLY on matching ops."""

    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(
            save_grads=lambda op: bool(op.layer_label.startswith("relu")),
            backward_ready=True,
        ),
        save_mode="reference",
    )
    _backward(log)
    assert _grad_labels(log) == ["relu_1_2"]


@pytest.mark.smoke
def test_save_grads_selector_restricts_retention() -> None:
    """A tl.* selector keeps gradients ONLY on matching ops."""

    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(save_grads=tl.func("relu"), backward_ready=True),
        save_mode="reference",
    )
    _backward(log)
    assert _grad_labels(log) == ["relu_1_2"]


def test_save_grads_true_keeps_all_gradients() -> None:
    """The boolean policy is unchanged: True retains every op's gradient."""

    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(save_grads=True, backward_ready=True),
        save_mode="reference",
    )
    _backward(log)
    assert len(_grad_labels(log)) >= 4


def test_save_grads_label_string_still_works() -> None:
    """The legacy label-string selector spelling is unchanged."""

    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(save_grads=["relu_1_2"], backward_ready=True),
        save_mode="reference",
    )
    _backward(log)
    assert "relu_1_2" in _grad_labels(log)
    assert "linear_1_1" not in _grad_labels(log)


def test_save_grads_callable_non_bool_refuses_typed() -> None:
    """The predicate contract is strict bool: truthy junk refuses typed."""

    from torchlens.fastlog.exceptions import PredicateError

    with pytest.raises(PredicateError) as excinfo:
        tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(save_grads=lambda op: 1, backward_ready=True),
            save_mode="reference",
        )
    assert excinfo.value.fields["code"] == "predicate_return_invalid"


@pytest.mark.real_model
def test_save_grads_selector_gpt2_narrow_retention() -> None:
    """R0 realism row: a narrowing selector on the REAL GPT-2 class."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_gpt2

    model = build_gpt2("eager")
    generator = torch.Generator().manual_seed(20260826)
    input_ids = torch.randint(0, 512, (1, 8), generator=generator)
    log = tl.trace(
        model,
        input_ids,
        capture=CaptureOptions(save_grads=tl.func("layer_norm"), backward_ready=True),
        save_mode="reference",
    )
    logits = list(log)[-1].out
    logits.sum().backward()
    grad_labels = _grad_labels(log)
    assert grad_labels, "the layer_norm selector matched nothing on GPT-2"
    assert all(label.startswith("layernorm") for label in grad_labels), grad_labels
