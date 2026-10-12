"""U40.1 default state contract: a capture is a pure observer; ``carry_state`` opts out.

JMT's ruling (2026-10-11): "Pure observer: the trace snapshots and restores
buffers, random state, modes and attributes; an explicit opt-in gives
forward-equivalent behavior for people who train through a trace."

- After a default ``tl.trace``, buffers (persistent and non-persistent), the
  torch / CUDA (when present) / Python / NumPy random states and every module's
  ``training`` flag are bit-identical to before, on success and when the
  forward raises. Plain-attribute restore is built but opt-in for now (it
  collides with the stale-reference holder contract); attributes the forward
  writes stay as it leaves them.
- ``CaptureOptions(carry_state=True)`` is forward-equivalent: buffers, modes
  and attributes advance exactly as one direct ``model(x)`` would (train-mode
  running statistics bit-exact); no capture ever advances the caller's random
  streams (R57).
- Bug 7 (the D18 contract change): train-mode BatchNorm runs through ``run``
  and ``run(until=)``; unproven buffer-write evidence and non-norm buffer
  writers still refuse typed.
"""

from __future__ import annotations

import copy
import random
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import BufferSinkRoutingError
from torchlens.options import CaptureOptions


class _Stateful(nn.Module):
    """Train-mode BN, dropout, a non-persistent buffer and forward-mutated attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(4, 2)
        self.register_buffer("seen", torch.zeros(2), persistent=False)
        self.calls = 0
        self.history: list[int] = [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.history.append(self.calls)
        self.last_batch = int(x.shape[0])
        with torch.no_grad():
            self.seen.copy_(torch.stack([torch.tensor(float(self.calls)), x.mean()]))
        _ = (random.random(), np.random.rand())
        h = self.drop(torch.relu(self.bn(self.conv(x))))
        return self.head(h.mean(dim=(2, 3)))


class _Boom(_Stateful):
    """Mutates every kind of state, then raises mid-forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        self.bn.eval()
        raise RuntimeError("boom after the state writes")


def _rng_state() -> tuple:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    return torch.get_rng_state(), cuda, random.getstate(), np.random.get_state()


def _assert_rng_equal(before: tuple, after: tuple) -> None:
    assert torch.equal(before[0], after[0])
    assert len(before[1]) == len(after[1])
    assert all(torch.equal(a, b) for a, b in zip(before[1], after[1], strict=True))
    assert before[2] == after[2]
    assert before[3][0] == after[3][0] and np.array_equal(before[3][1], after[3][1])
    assert before[3][2:] == after[3][2:]


def _restored_state(model: nn.Module) -> dict:
    """Buffers and modes: what a default capture restores."""

    return {
        "buffers": {name: buf.detach().clone() for name, buf in model.named_buffers()},
        "modes": {name: module.training for name, module in model.named_modules()},
    }


def _assert_restored_equal(before: dict, after: dict) -> None:
    assert before["buffers"].keys() == after["buffers"].keys()
    for name, value in before["buffers"].items():
        assert torch.equal(value, after["buffers"][name]), name
    assert before["modes"] == after["modes"]


def _observed_state(model: nn.Module) -> dict:
    return {
        "buffers": {name: buf.detach().clone() for name, buf in model.named_buffers()},
        "modes": {name: module.training for name, module in model.named_modules()},
        "attrs": {
            name: {
                key: copy.deepcopy(value)
                for key, value in vars(module).items()
                if not key.startswith("_") and key != "training"
            }
            for name, module in model.named_modules()
        },
    }


def _assert_state_equal(before: dict, after: dict) -> None:
    assert before["buffers"].keys() == after["buffers"].keys()
    for name, value in before["buffers"].items():
        assert torch.equal(value, after["buffers"][name]), name
    assert before["modes"] == after["modes"]
    assert before["attrs"] == after["attrs"]


def test_default_trace_restores_buffers_rng_and_modes():
    """A default capture of a train-mode stateful model restores buffers, RNG and modes."""

    torch.manual_seed(0)
    model = _Stateful().train()
    x = torch.randn(2, 3, 8, 8)
    before, rng_before = _restored_state(model), _rng_state()
    tl.trace(model, x)
    _assert_restored_equal(before, _restored_state(model))
    _assert_rng_equal(rng_before, _rng_state())
    # Plain attributes the forward wrote stay (attribute restore is opt-in).
    assert model.calls == 1 and model.last_batch == 2


def test_default_trace_restores_state_on_the_exception_path():
    """A capture whose forward raises after writing state still restores it."""

    torch.manual_seed(0)
    model = _Boom().train()
    x = torch.randn(2, 3, 8, 8)
    before, rng_before = _restored_state(model), _rng_state()
    with pytest.raises(RuntimeError, match="boom after the state writes"):
        tl.trace(model, x)
    _assert_restored_equal(before, _restored_state(model))
    _assert_rng_equal(rng_before, _rng_state())


def test_plain_attribute_restore_is_built_but_opt_in():
    """The bracket restores plain attributes only when asked (a one-line default flip)."""

    from torchlens._runnable_state_live import (
        RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT,
        capture_observer_bracket,
    )

    assert RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT is False
    torch.manual_seed(0)
    model = _Stateful().train()
    x = torch.randn(2, 3, 8, 8)
    before = _observed_state(model)
    with capture_observer_bracket(model, carry_state=False, restore_attributes=True):
        model(x)
    _assert_state_equal(before, _observed_state(model))
    assert not hasattr(model, "last_batch")


def test_carry_state_is_forward_equivalent_bit_exact():
    """carry_state=True: buffers, modes and attributes advance exactly as one model(x) does.

    Random state is the exception under both settings: every capture draws from
    its own seed and restores the global engines (R57), so the caller's random
    streams never advance.
    """

    torch.manual_seed(0)
    model = _Stateful().train()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 8, 8)
    reference(x)
    expected = _observed_state(reference)
    rng_before = _rng_state()
    tl.trace(model, x, capture=CaptureOptions(carry_state=True))
    _assert_state_equal(expected, _observed_state(model))
    _assert_rng_equal(rng_before, _rng_state())


def _train_bn_stack() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(
            conv0=nn.Conv2d(3, 4, 3, padding=1),
            bn0=nn.BatchNorm2d(4),
            act0=nn.ReLU(),
            conv1=nn.Conv2d(4, 4, 3, padding=1),
            bn1=nn.BatchNorm2d(4),
            act1=nn.ReLU(),
            flat=nn.Flatten(),
            head=nn.Linear(4 * 8 * 8, 2),
        )
    ).train()


def test_train_mode_batchnorm_runs_and_default_run_restores_stats():
    """Bug 7: run() and run(until=) on train-mode BN; the default run restores stats."""

    model = _train_bn_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    before = {name: buf.clone() for name, buf in model.named_buffers()}
    full = log.run(inputs=torch.randn(2, 3, 8, 8))
    full.output.pow(2).mean().backward()
    truncated = log.run(inputs=torch.randn(2, 3, 8, 8), until="act0")
    assert truncated.report.truncated is True
    truncated.trace["act0"].out.pow(2).mean().backward()
    for name, value in before.items():
        assert torch.equal(value, dict(model.named_buffers())[name]), name


def test_run_carry_state_advances_train_stats_like_a_plain_forward():
    """run(carry_state=True) running statistics equal a plain forward's, bit-exact."""

    model = _train_bn_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    reference = copy.deepcopy(model)
    x2 = torch.randn(2, 3, 8, 8)
    reference(x2)
    log.run(inputs=x2, carry_state=True)
    for (name, got), (_, expected) in zip(
        model.named_buffers(), reference.named_buffers(), strict=True
    ):
        assert torch.equal(got, expected), name


def test_train_mode_run_backward_matches_eager_gradients():
    """Default train-mode run(): backward works and equals the eager model's gradients."""

    model = _train_bn_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    reference = copy.deepcopy(model)
    x2 = torch.randn(2, 3, 8, 8)
    log.run(inputs=x2).output.pow(2).mean().backward()
    reference(x2).pow(2).mean().backward()
    got = {name: p.grad for name, p in model.named_parameters()}
    expected = {name: p.grad for name, p in reference.named_parameters()}
    assert len(expected) == 10 and all(grad is not None for grad in expected.values())
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0, msg=name)


def test_d18_still_refuses_unproven_train_mode_evidence():
    """Unproven (None) buffer-write evidence still refuses typed in train mode."""

    model = _train_bn_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    for label in log.internal_sink_ops:
        layer = log.layer_dict_all_keys[label]
        if layer.layer_type == "buffer":
            layer._internal_set("buffer_value_changed", None)
    with pytest.raises(BufferSinkRoutingError, match="unproven"):
        log.run(inputs=torch.randn(2, 3, 8, 8))


def test_d18_still_refuses_a_non_norm_buffer_writer():
    """A counter buffer written in place is not a proven train-mode norm writer."""

    class _Counter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 3)
            self.register_buffer("steps", torch.zeros(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.steps.add_(1)
            return self.fc(x)

    model = _Counter()
    log = tl.trace(model, torch.randn(2, 3))
    assert float(model.steps) == 0.0
    with pytest.raises(BufferSinkRoutingError, match="not proven train-mode"):
        log.run(inputs=torch.randn(2, 3))


def test_resnet18_train_mode_runs_through_run_and_until():
    """Acceptance: stock resnet18 in train mode runs through run and run(until=)."""

    torchvision_models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = torchvision_models.resnet18(weights=None, num_classes=10).train()
    x = torch.randn(2, 3, 32, 32)
    before = {name: buf.clone() for name, buf in model.named_buffers()}
    log = tl.trace(model, x, save=tl.in_module("layer2"))
    log.run(inputs=torch.randn(2, 3, 32, 32)).output.pow(2).mean().backward()
    result = log.run(inputs=torch.randn(2, 3, 32, 32), until="layer2")
    result.trace["layer2"].out.pow(2).mean().backward()
    for name, value in before.items():
        assert torch.equal(value, dict(model.named_buffers())[name]), name


def test_carry_state_does_not_change_the_trace():
    """carry_state changes only the model's post-capture state, never the trace."""

    torch.manual_seed(0)
    model = _Stateful().train()
    twin = copy.deepcopy(model)
    x = torch.randn(2, 3, 8, 8)
    pure = tl.trace(model, x, capture=CaptureOptions(random_seed=3))
    carried = tl.trace(twin, x, capture=CaptureOptions(random_seed=3, carry_state=True))
    assert [op.layer_label for op in pure.layer_list] == [
        op.layer_label for op in carried.layer_list
    ]
    for left, right in zip(pure.layer_list, carried.layer_list, strict=True):
        if isinstance(left.out, torch.Tensor):
            assert torch.equal(left.out, right.out), left.layer_label
