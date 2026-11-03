"""Default state contract: a capture is a pure observer; ``carry_state`` opts out.

JMT's ruling (2026-10-11): "Pure observer: the trace snapshots and restores
buffers, random state, modes and attributes; an explicit opt-in gives
forward-equivalent behavior for people who train through a trace." AGT-2360
(2026-10-12, option c): "Full pure observer: restore attributes too; the trace
records what the forward wrote so stashed values are read from it;
carry_state=True keeps them".

- After a default ``tl.trace``, buffers (persistent and non-persistent), the
  torch / CUDA (when present) / Python / NumPy random states, every module's
  ``training`` flag and every plain attribute are as before, on success and
  when the forward raises. Attributes keep their identity: a rebound object is
  put back, a container changed in place is restored in place.
- What the forward wrote is read from the trace:
  ``trace.modules[address].attribute_writes[name]`` (``AttributeWrite``);
  a loaded trace has no record (``None``).
- ``CaptureOptions(carry_state=True)`` is forward-equivalent: buffers, modes
  and attributes advance exactly as one direct ``model(x)`` would (train-mode
  running statistics bit-exact), and the same writes are recorded; no capture
  ever advances the caller's random streams (R57).
- Bug 7 (the D18 contract change): train-mode BatchNorm runs through ``run``
  and ``run(until=)``; unproven buffer-write evidence and non-norm buffer
  writers still refuse typed.
"""

from __future__ import annotations

import copy
import random
import warnings
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._record_type import record_type
from torchlens.errors import BufferSinkRoutingError, TraceCaptureError
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


class _Stash(nn.Module):
    """Stashes a tensor, deletes an attribute, rebinds one and edits a tensor in place."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.scratch = "gone after the forward"
        self.cfg = {"mode": "a"}
        self.count = torch.zeros(3)
        self.big = torch.zeros(5000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        self.last_out = y.detach()
        del self.scratch
        self.cfg = {"mode": "b"}
        with torch.no_grad():
            self.count += 1
            self.big[0] = 7.0
        return y


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_attr_observer.py.
def test_carry_state_keeps_the_writes_and_records_them():
    """carry_state=True leaves the forward's attribute writes and records the same rows."""

    model = _Stash()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(carry_state=True))
    assert not hasattr(model, "scratch") and model.cfg == {"mode": "b"}
    assert torch.equal(model.count, torch.ones(3)) and model.last_out is not None
    writes = trace.modules["self"].attribute_writes
    assert set(writes) == {"big", "cfg", "count", "last_out", "scratch"}
    assert writes["last_out"].value is model.last_out


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_attr_observer.py, campaign mutant kills.
def test_loaded_trace_has_no_attribute_write_record(tmp_path):
    """The record is session-time only: a loaded trace reads None, never raises."""

    trace = tl.trace(_Stash(), torch.randn(2, 4))
    assert trace.modules["self"].attribute_writes is not None
    path = tmp_path / "stash.tlspec"
    trace.save(path)
    loaded = tl.load(str(path))
    assert loaded.modules["self"].attribute_writes is None
    assert loaded.modules["fc"].attribute_writes is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _runnable/state_live.py, campaign mutant kills.
def test_default_trace_restores_state_on_the_exception_path():
    """A capture whose forward raises after writing state still restores all of it."""

    torch.manual_seed(0)
    model = _Boom().train()
    x = torch.randn(2, 3, 8, 8)
    before, rng_before = _observed_state(model), _rng_state()
    with pytest.raises(TraceCaptureError, match="boom after the state writes") as _tce_info:
        tl.trace(model, x)
    assert isinstance(_tce_info.value.__cause__, RuntimeError)
    _assert_state_equal(before, _observed_state(model))
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/projectors.py, _data_classes/_fork_lineage.py, intervention/_trace_run.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _runnable/transaction.py, intervention/_trace_run.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/projectors.py.
def test_d18_still_refuses_unproven_train_mode_evidence():
    """Unproven (None) buffer-write evidence still refuses typed in train mode."""

    model = _train_bn_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    for label in log.internal_sink_ops:
        layer = log._layer_dict_all_keys[label]
        if record_type(layer) == "buffer":
            layer._internal_set("buffer_value_changed", None)
    with pytest.raises(BufferSinkRoutingError, match="unproven"):
        log.run(inputs=torch.randn(2, 3, 8, 8))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/projectors.py.
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


def test_carry_state_does_not_change_the_trace():
    """carry_state changes only the model's post-capture state, never the trace."""

    torch.manual_seed(0)
    model = _Stateful().train()
    twin = copy.deepcopy(model)
    x = torch.randn(2, 3, 8, 8)
    pure = tl.trace(model, x, capture=CaptureOptions(seed=3))
    carried = tl.trace(twin, x, capture=CaptureOptions(seed=3, carry_state=True))
    assert [op.layer_label for op in pure._layer_list] == [
        op.layer_label for op in carried._layer_list
    ]
    for left, right in zip(pure._layer_list, carried._layer_list, strict=True):
        if isinstance(left.out, torch.Tensor):
            assert torch.equal(left.out, right.out), left.layer_label


# Kept by shrink-r4 (per-id subset rule): sole coverage of _runnable/state_live.py, _utils/rng/streams.py.
def test_unrestorable_state_warns_capture_state_restore_incomplete(monkeypatch):
    """A restore step that fails is named in one coded warning, never raised."""

    import torchlens._utils.rng as rng
    from torchlens.errors import TorchLensWarning

    def _broken(_states):
        raise RuntimeError("generator unavailable")

    model = _Stateful().train()
    x = torch.randn(2, 3, 8, 8)
    monkeypatch.setattr(rng, "set_rng_from_saved_states", _broken)
    with pytest.warns(TorchLensWarning) as caught:
        tl.trace(model, x)
    codes = [
        w.message.fields.get("code") for w in caught if isinstance(w.message, TorchLensWarning)
    ]
    assert "capture_state_restore_incomplete" in codes


class _BigHistory(nn.Module):
    """Writes in place into builtin containers past the old 128-item / 64-level bounds."""

    def __init__(self, n_items: int = 129, depth: int = 0) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.history = [0] * n_items
        self.table = {str(i): i for i in range(n_items)}
        self.deep: list = [0]
        for _ in range(depth):
            self.deep = [self.deep]
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.history[0] += 1
        self.table["0"] = -1
        leaf = self.deep
        for _ in range(self.depth):
            leaf = leaf[0]
        leaf[0] += 1
        return self.fc(x)


def _restore_incomplete_messages(caught: list) -> list[str]:
    from torchlens.errors import TorchLensWarning

    return [
        str(w.message)
        for w in caught
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") == "capture_state_restore_incomplete"
    ]


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py, _capture/prepare/door.py, _capture_attr_observer.py ....
def test_nesting_past_the_observer_bound_warns_naming_the_attribute():
    """Past the observer's depth bound the capture names the attribute; never silent."""

    from torchlens._capture_attr_observer import _MAX_DEPTH

    depth = _MAX_DEPTH + 10
    model = _BigHistory(n_items=2, depth=depth)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, torch.ones(1, 1))
    messages = _restore_incomplete_messages(caught)
    assert len(messages) == 1 and "<root>.deep" in messages[0]
    assert model.history[0] == 0  # the other attributes are still restored
