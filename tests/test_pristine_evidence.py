"""``Trace.is_pristine`` is derived from the evidence a trace carries, never a stored flag.

A plain capture reads pristine before and after ``Trace.save``/``tl.load``; any
intervention, staged hook, value replacement or hooked run reads not pristine, and
no reset (clearing hooks, reassigning ``Trace.state``, forking, saving) makes it
pristine again. An artifact that cannot carry the evidence reads ``None`` (no
evidence), never pristine. The reading never changes a verdict.
"""

from __future__ import annotations

import pickle
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _refuses_as_2x import assert_refuses_as_2x
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_pristine import derive_pristine, pristine_touch_evidence
from torchlens._runnable import PathFaithfulness
from torchlens._trace_state import TraceState
from torchlens.backends._intervention_evidence import records_intervention_evidence
from torchlens.intervention.errors import BaselineUndeterminedError
from torchlens.options import CaptureOptions

_DIM = 8
_HERE = Path(__file__).parent
_V9_ARTIFACT = _HERE / "fixtures" / "tlspec_v9" / "clean.tlspec"
_V7_ARTIFACT = _HERE / "fixtures" / "tlspec_v7" / "tiny_v7.tlspec"


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


def _model() -> nn.Module:
    torch.manual_seed(0)
    return _Net().eval()


def _x(seed: int = 0) -> torch.Tensor:
    return torch.randn(1, 3, _DIM, generator=torch.Generator().manual_seed(seed))


def _saved(trace: tl.Trace, tmp_path: Path, name: str = "t") -> tl.Trace:
    path = tmp_path / f"{name}.tlspec"
    trace.save(str(path))
    return tl.load(str(path))


def _ablated(model: nn.Module) -> tl.Trace:
    return tl.trace(model, _x(), intervene=tl.when(tl.func("relu"), tl.zero_ablate()))


def _steered(model: nn.Module) -> tl.Trace:
    direction = torch.randn(_DIM, generator=torch.Generator().manual_seed(3))
    spec = tl.when(tl.module("block"), tl.steer(direction, strength=3.0, dim=-1))
    return tl.trace(model, _x(), save=tl.module("block") | tl.module("head"), intervene=spec)


def _ready(model: nn.Module) -> tl.Trace:
    return tl.trace(model, _x(), capture=CaptureOptions(intervention_ready=True))


def _double(out: torch.Tensor) -> torch.Tensor:
    return out * 2.0


def test_plain_capture_reads_pristine_before_and_after_save(tmp_path: Path) -> None:
    trace = tl.trace(_model(), _x())
    assert trace.is_pristine is True
    assert pristine_touch_evidence(trace) == ()
    assert _saved(trace, tmp_path).is_pristine is True
    assert pickle.loads(pickle.dumps(trace)).is_pristine is True


def test_intervened_capture_reads_not_pristine_although_its_state_says_pristine(
    tmp_path: Path,
) -> None:
    trace = _ablated(_model())
    # The operational state machine never left PRISTINE for a func-targeted edit;
    # the derived reading comes from the evidence the capture recorded.
    assert trace.state is TraceState.PRISTINE
    assert trace.is_pristine is False
    assert {"intervention_audit", "state_history", "op_marks"} <= set(
        pristine_touch_evidence(trace)
    )
    loaded = _saved(trace, tmp_path)
    assert loaded.is_pristine is False
    assert {"intervention_audit", "state_history", "op_marks"} <= set(
        pristine_touch_evidence(loaded)
    )


def test_steered_capture_reads_not_pristine_after_save(tmp_path: Path) -> None:
    trace = _steered(_model())
    assert trace.is_pristine is False
    assert _saved(trace, tmp_path).is_pristine is False


def test_zero_fire_capture_reads_not_pristine(tmp_path: Path) -> None:
    never = tl.when(tl.func("sigmoid"), tl.zero_ablate())
    trace = tl.trace(_model(), _x(), intervene=never, on_zero_fire="disclose")
    assert trace.is_pristine is False
    assert _saved(trace, tmp_path).is_pristine is False


def test_staged_hook_stays_not_pristine_after_clear_and_save(tmp_path: Path) -> None:
    trace = _ready(_model())
    trace.attach_hooks(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    assert trace.is_pristine is False
    assert "staged_spec" in pristine_touch_evidence(trace)
    trace.clear_hooks(confirm_mutation=True)
    assert trace.is_pristine is False
    # The staged spec is session-time; the staging row persists.
    loaded = _saved(trace, tmp_path)
    assert loaded.is_pristine is False
    assert "state_history" in pristine_touch_evidence(loaded)


def test_value_replacement_survives_a_state_reassignment(tmp_path: Path) -> None:
    trace = _ready(_model())
    trace.set(tl.module("block"), _double, confirm_mutation=True)
    trace.state = TraceState.PRISTINE
    assert trace.is_pristine is False
    loaded = _saved(trace, tmp_path)
    loaded.state = TraceState.PRISTINE
    assert loaded.is_pristine is False


def test_clear_hooks_on_an_untouched_trace_stays_pristine(tmp_path: Path) -> None:
    trace = _ready(_model())
    trace.clear_hooks(confirm_mutation=True)
    assert trace.state is TraceState.SPEC_STALE
    assert trace.is_pristine is True
    assert _saved(trace, tmp_path).is_pristine is True


def test_plain_reruns_and_runs_stay_pristine(tmp_path: Path) -> None:
    model = _model()
    rerun = tl.trace(model, _x())
    rerun.run(model, _x(1))
    assert rerun.state is TraceState.RERUN_PROPAGATED
    assert rerun.is_pristine is True
    assert _saved(rerun, tmp_path, "rerun").is_pristine is True

    source = tl.trace(model, _x())
    result = source.run(inputs=_x(1))
    assert source.is_pristine is True and result.trace.is_pristine is True


def test_fork_inherits_the_reading_and_never_touches_its_parent() -> None:
    model = _model()
    plain = tl.trace(model, _x(), capture=CaptureOptions(intervention_ready=True))
    fork = plain.fork("f")
    assert fork.is_pristine is True and plain.is_pristine is True
    fork.set(tl.module("block"), _double, confirm_mutation=True)
    assert fork.is_pristine is False
    assert plain.is_pristine is True
    assert fork.fork("g").is_pristine is False


def test_hooked_fast_run_agrees_with_the_flag() -> None:
    model = _model()
    plain = tl.trace(model, _x(), save=tl.module("block") | tl.module("head"))
    plain_report = plain.run(inputs=_x(1), fast=True).report
    assert plain_report.path_faithfulness is PathFaithfulness.VERIFIED
    assert plain.is_pristine is True

    hooked = _steered(model)
    result = hooked.run(inputs=_x(1), fast=True)
    mismatch = result.report.first_mismatch
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert mismatch is not None and ("pristine", "false") in tuple(mismatch.details)
    assert result.trace.is_pristine is False
    assert pickle.loads(pickle.dumps(result.trace)).is_pristine is False


def test_v9_artifact_refuses_as_2x(tmp_path: Path) -> None:
    """A v9 artifact written by torchlens 2.x refuses typed at load."""

    work = tmp_path / "clean_v9.tlspec"
    shutil.copytree(_V9_ARTIFACT, work)
    assert_refuses_as_2x(work)


def test_pre_v9_artifact_refuses_as_2x(tmp_path: Path) -> None:
    """A pre-v9 artifact written by torchlens 2.x refuses typed at load."""

    work = tmp_path / "tiny_v7.tlspec"
    shutil.copytree(_V7_ARTIFACT, work)
    assert_refuses_as_2x(work)


def test_saved_artifact_reads_what_it_carries(tmp_path: Path) -> None:
    """An untouched trace saved by this release loads with no touch evidence, pristine."""

    path = tmp_path / "clean.tlspec"
    tl.trace(_model(), _x()).save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.load(path)
    assert pristine_touch_evidence(trace) == ()
    assert trace.is_pristine is True


@pytest.mark.parametrize(
    ("backend", "version", "history", "expected"),
    [
        ("torch", 10, [], True),
        ("torch", 8, [], None),
        ("torch", 8, [{"op": "do"}], False),
        ("tf", 10, [], None),
        ("tf", 10, [{"op": "set"}], False),
        ("torch", 10, [{"op": "rerun", "hook_count": 0, "hook_fire_count": 0}], True),
        ("torch", 10, [{"op": "rerun", "hook_count": 1, "hook_fire_count": 0}], False),
        ("torch", 10, [{"op": "append", "hook_count": 0, "hook_fire_count": 2}], False),
        ("torch", 10, [{"op": "clear_hooks"}, {"op": "fork"}, {"op": "detach_hooks"}], True),
    ],
)
def test_derivation_rule(backend: str, version: int, history: list[Any], expected: Any) -> None:
    stub = SimpleNamespace(
        backend=backend,
        tlspec_version=version,
        state_history=history,
        intervention_audit=[],
        state=TraceState.RERUN_PROPAGATED,
        _has_direct_writes=False,
        layer_list=[],
    )
    assert derive_pristine(stub) is expected


def test_non_torch_backends_read_not_recorded_never_pristine() -> None:
    names = {spec.name for spec in tl.backends.specs()}
    assert "torch" in names and len(names) > 1
    for name in sorted(names):
        stub = SimpleNamespace(
            backend=name,
            tlspec_version=10,
            state_history=[],
            intervention_audit=[],
            state=TraceState.PRISTINE,
            _has_direct_writes=False,
            layer_list=[],
        )
        expected = True if name == "torch" else None
        assert records_intervention_evidence(name) is (name == "torch")
        assert derive_pristine(stub) is expected, name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intervention_audit", [{"kind": "ACT"}]),
        ("_has_direct_writes", True),
        ("state", TraceState.DIRECT_WRITE_DIRTY),
        ("state", TraceState.REPLAY_PROPAGATED),
        ("state", TraceState.LIVE_CAPTURED),
        ("layer_list", [SimpleNamespace(intervention_replaced=True)]),
    ],
)
def test_each_evidence_kind_reads_not_pristine(field: str, value: Any) -> None:
    stub = SimpleNamespace(
        backend="torch",
        tlspec_version=10,
        state_history=[],
        intervention_audit=[],
        state=TraceState.PRISTINE,
        _has_direct_writes=False,
        layer_list=[],
    )
    setattr(stub, field, value)
    assert derive_pristine(stub) is False


def test_bundle_auto_baseline_reads_the_derived_flag() -> None:
    model = _model()
    clean = tl.trace(model, _x())
    ablated = _ablated(model)
    bundle = tl.Bundle({"clean": clean, "ablated": ablated})
    assert bundle.baseline_name == "clean"
    assert "pristine" in bundle.help().splitlines()[1]
    assert "pristine" not in bundle.help().splitlines()[2]


def test_bundle_without_a_pristine_member_refuses_typed() -> None:
    model = _model()
    bundle = tl.Bundle({"a": _ablated(model), "b": _steered(model)})
    assert bundle.baseline_name is None
    with pytest.raises(BaselineUndeterminedError):
        bundle.rank_changes()


def test_reading_the_flag_never_changes_the_plain_run_verdict() -> None:
    model = _model()
    first = tl.trace(model, _x())
    unread = first.run(inputs=_x(1)).report
    second = tl.trace(model, _x())
    assert second.is_pristine is True
    read = second.run(inputs=_x(1)).report
    card = ("path_faithfulness", "first_mismatch", "is_poisoned", "numeric_attestation")
    assert [getattr(read, name) for name in card] == [getattr(unread, name) for name in card]
    assert read.path_faithfulness is PathFaithfulness.VERIFIED
    assert tl.validate(model, _x(), scope="forward") is True
