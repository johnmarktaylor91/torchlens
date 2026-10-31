"""The host RNG monitor's profile hook: inlined first exits, unchanged verdicts.

The profile hook answers the overwhelmingly common profile events (a C call
on an inert built-in receiver, a return into a cached implementation frame)
without entering the classifier helpers. These tests pin two things: the
helpers are entered only for events their own first checks would not
dismiss, and the verdicts on host-RNG toys (module draws, a held alias, NumPy
generators held and returned, a clock read) are the ones the monitor reported
before the fast exits, on the capture and on the rerun's fresh capture.
"""

from __future__ import annotations

import random
import time
from random import random as held_random
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.intervention.rerun as rerun_module
import torchlens.utils.rng as rng
from torchlens.options import CaptureOptions, InterventionOptions

_FIELDS = (
    "host_rng_consumed",
    "host_rng_unreplayable",
    "host_rng_channels",
    "host_rng_replayable_reads",
    "rng_monitor_uncertain",
    "rng_monitor_uncertain_detail",
)


def _make_generator() -> np.random.Generator:
    return np.random.default_rng(0)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.body(torch.relu(self.fc1(x))))


class _Clean(_Base):
    pass


class _ModuleDraw(_Base):
    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h * random.random()


class _HeldAlias(_Base):
    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h * held_random()


class _NumpyHeld(_Base):
    def __init__(self) -> None:
        super().__init__()
        self.generator = np.random.default_rng(0)

    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h * float(self.generator.normal())


class _NumpyReturned(_Base):
    def body(self, h: torch.Tensor) -> torch.Tensor:
        generator = _make_generator()
        return h * float(generator.normal())


class _Clock(_Base):
    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + 0.0 * time.time())


class _Dropout(_Base):
    def __init__(self) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.5)

    def body(self, h: torch.Tensor) -> torch.Tensor:
        return self.drop(h)


class _TorchRand(_Base):
    def body(self, h: torch.Tensor) -> torch.Tensor:
        return h + torch.rand(h.shape)


def _clean_verdict(consumed: bool = False) -> dict[str, Any]:
    return {
        "host_rng_consumed": consumed,
        "host_rng_unreplayable": False,
        "host_rng_channels": (),
        "host_rng_replayable_reads": (),
        "rng_monitor_uncertain": False,
        "rng_monitor_uncertain_detail": (),
    }


def _unreplayable(*channels: str) -> dict[str, Any]:
    return {**_clean_verdict(True), "host_rng_unreplayable": True, "host_rng_channels": channels}


#: The verdicts the monitor reported before the inlined exits (measured on the
#: base with the same toys), for the capture and for the rerun's fresh capture.
EXPECTED = {
    "clean": (_Clean, _clean_verdict()),
    "module_draw": (_ModuleDraw, _clean_verdict(True)),
    "held_alias": (_HeldAlias, _clean_verdict(True)),
    "numpy_held": (
        _NumpyHeld,
        _unreplayable("frame_reachable_generator", "model_attribute_generator"),
    ),
    "numpy_returned": (
        _NumpyReturned,
        _unreplayable("c_rng_instance_draw", "frame_reachable_generator", "np.random.default_rng"),
    ),
    "clock": (_Clock, _unreplayable("time.time")),
    "dropout_train": (_Dropout, _clean_verdict()),
    "torch_rand": (_TorchRand, _clean_verdict()),
}


def _verdict(trace: tl.Trace) -> dict[str, Any]:
    state = trace.__dict__["_runnable"]
    return {name: getattr(state, name) for name in _FIELDS}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_verdicts_match_the_base_on_capture_and_rerun(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls, expected = EXPECTED[name]
    torch.manual_seed(0)
    random.seed(0)
    model = cls()
    model.train(name == "dropout_train")
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(replayable=True))
    assert _verdict(trace) == expected
    fresh: list[dict[str, Any]] = []
    original = rerun_module._capture_with_active_spec

    def grab(*args: Any, **kwargs: Any) -> tl.Trace:
        new_log = original(*args, **kwargs)
        fresh.append(_verdict(new_log))
        return new_log

    monkeypatch.setattr(rerun_module, "_capture_with_active_spec", grab)
    trace.fork().do(
        tl.module("fc1"),
        tl.scale(0.5),
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )
    assert fresh == [expected]


def test_unarmed_capture_keeps_the_witness_gap() -> None:
    model = _ModuleDraw()
    trace = tl.trace(model, torch.randn(2, 4))
    verdict = _verdict(trace)
    assert verdict["rng_monitor_uncertain"] is True
    assert verdict["rng_monitor_uncertain_detail"] == ("monitor_not_armed",)


def test_classifiers_entered_only_past_their_own_first_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every classifier entry is one its first checks would not have dismissed."""

    monitor_cls = rng.host_nondeterminism_monitor
    c_call_entries: list[tuple[bool, type | None]] = []
    return_entries: list[bool] = []
    original_c_call = monitor_cls._classify_c_call
    original_return = monitor_cls._snapshot_returned_numpy_rngs

    def recording_c_call(self: Any, frame: Any, arg: Any) -> None:
        receiver = getattr(arg, "__self__", None)
        c_call_entries.append(
            (id(arg) in self._held_ref_marks, None if receiver is None else type(receiver))
        )
        original_c_call(self, frame, arg)

    def recording_return(self: Any, frame: Any, returned: Any) -> None:
        caller = frame.f_back
        scope = self._numpy_frame_digest_scope_cache.get(id(caller.f_code))
        return_entries.append(scope is not None and scope[0] is caller.f_code and scope[1] is False)
        original_return(self, frame, returned)

    monkeypatch.setattr(monitor_cls, "_classify_c_call", recording_c_call)
    monkeypatch.setattr(monitor_cls, "_snapshot_returned_numpy_rngs", recording_return)
    model = _NumpyReturned()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(replayable=True))
    assert _verdict(trace) == EXPECTED["numpy_returned"][1]
    assert c_call_entries
    inert = rng._INERT_PROFILE_C_CALL_RECEIVER_TYPES
    for held, receiver_type in c_call_entries:
        assert held or (receiver_type is not None and receiver_type not in inert)
    # A return reaches the helper only when its caller is uncached or a user frame.
    if rng._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST:
        assert return_entries
    assert not any(return_entries)
