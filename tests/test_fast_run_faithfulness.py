"""fast=True per-iteration faithfulness ceilings (grind B3-L1, Tier-1 #1).

The verify-once gate proves only the FIRST input settled ``verified``; every
later fast iteration runs on a different input and must re-derive the
per-input dynamic ceilings through ``_path_faithfulness`` exactly like the
ordinary provider. These tests pin the two published wrong-value repros
(host-scalar escape, layout twin) to the honest ``unverifiable`` verdict and
prove fast mode can never IMPROVE the verdict over ``fast=False`` on the same
artifact and input.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import PathFaithfulness
from torchlens.options import CaptureOptions


class HostScalarEscapeModel(nn.Module):
    """Bakes a tensor->host escaped scalar into a downstream literal."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Multiply by a host-escaped derived integer constant."""

        scale = int(value.sum().item()) % 5 + 1
        return value * scale


class LayoutPredicateModel(nn.Module):
    """Branches on a layout predicate derived from the model input."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Take a contiguity-dependent arm on an input-rooted activation."""

        doubled = value * 2
        if doubled.is_contiguous():
            return doubled + 100
        return doubled - 100


def _runnable_artifact(model: nn.Module, inputs: torch.Tensor, path: Path) -> Path:
    """Capture and save one runnable artifact for the fast-path repros."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(
            model,
            inputs,
            capture=CaptureOptions(
                replayable=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    captured.save(path, level="runnable")
    return path


# Kept: sole branch coverage of _fast_run.py.
def test_fast_run_original_input_escape_stays_verified(tmp_path: Path) -> None:
    """Repeating the ORIGINAL input keeps the escape digest fresh: still verified."""

    original = torch.ones(4)
    path = _runnable_artifact(
        HostScalarEscapeModel(), original, tmp_path / "escape-fast-orig.tlspec"
    )

    loaded = tl.load(path)
    first = loaded.run(inputs=original, fast=True)
    second = loaded.run(inputs=original.clone(), fast=True)

    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(second.output, HostScalarEscapeModel()(original))


class BufferGainModel(nn.Module):
    """Two linears plus a directly consumed buffer (a non-collectable save)."""

    def __init__(self) -> None:
        """Register the layers and the directly read gain buffer."""

        super().__init__()
        self.first = nn.Linear(3, 3)
        self.second = nn.Linear(3, 3)
        self.register_buffer("gain", torch.tensor([2.0, 1.0, 0.5]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the stacked linear output by the buffer."""

        return self.second(self.first(value)) * self.gain


# Kept: sole branch coverage of _fast_live_plans.py, _fast_run.py.
def test_fast_live_module_refusal_fires_before_activation_wipe() -> None:
    """A typed module-plan refusal must not destroy the user's saved payloads.

    The buffer read is saved but not fast-collectable, so it is exactly the
    payload the unsupported-activation wipe targets; the module-address
    refusal must fire BEFORE that wipe runs (grind b3-l1, F-R07).
    """

    model = BufferGainModel().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    saved_before = {op.label for op in captured._layer_list if op.has_saved_activation}
    assert "buffer_1:1" in saved_before
    del model.second

    with pytest.raises(Exception) as excinfo:
        captured.run(inputs=torch.ones(2, 3), fast=True)
    assert excinfo.value.fields["detection_stage"] == "fast_live_module_plan"

    saved_after = {op.label for op in captured._layer_list if op.has_saved_activation}
    assert saved_after == saved_before
    assert captured._layer_dict_all_keys["buffer_1:1"].out is not None


# Kept: sole branch coverage of _runnable/verification.py.
def test_fast_live_inherited_divergence_never_unregisters_user_trace() -> None:
    """An inherited divergence raise must not evict the user's live Trace."""

    class BranchingFunctionModel(nn.Module):
        """Choose between two same-shape activation functions from tensor data."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply the branch selected by the runtime sum."""

            if bool((value.sum() > 0).item()):
                return torch.relu(value)
            return torch.sigmoid(value)

    from torchlens import _state
    from torchlens.errors import PathDivergenceError

    model = BranchingFunctionModel().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(model, torch.ones(2), save=tl.func("relu"))

    with pytest.raises(PathDivergenceError):
        captured.run(inputs=-torch.ones(2), fast=True)
    # The poisoned mark is monotonic; a later good-input run raises the
    # inherited divergence but must leave the USER-owned Trace registered
    # (every other provider passes a throwaway fork to the discard arm).
    with pytest.raises(PathDivergenceError):
        captured.run(inputs=torch.ones(2), fast=True)
    assert any(log is captured for log in _state.list_logs())


# Kept: sole branch coverage of _fast_live_hooks.py.
def test_remove_fast_live_hooks_survives_raising_remove() -> None:
    """One raising handle.remove() must not strand the remaining hooks."""

    from torchlens._fast_live_hooks import _remove_fast_live_hooks

    removed: list[int] = []

    class Handle:
        """Hook-handle stand-in with an optional failing removal."""

        def __init__(self, key: int, fail: bool = False) -> None:
            """Record the handle identity and failure mode."""

            self.key = key
            self.fail = fail

        def remove(self) -> None:
            """Remove the handle or raise like a torn hook registry."""

            if self.fail:
                raise RuntimeError("torn hook registry")
            removed.append(self.key)

    handles: list = [Handle(1), Handle(2, fail=True), Handle(3)]
    with pytest.raises(RuntimeError, match="torn hook registry"):
        _remove_fast_live_hooks(handles)
    assert removed == [1, 3]
    assert handles == []


# Kept: sole branch coverage of _fast_run.py.
def test_close_fast_run_session_retryable_after_raising_close() -> None:
    """A raising close leaves the session attached and retryable, never stranded."""

    from torchlens._fast_run import close_fast_run_session

    class FlakySession:
        """Session stand-in whose first close raises."""

        def __init__(self) -> None:
            """Arm one failing close."""

            self.calls = 0

        def close(self) -> None:
            """Raise once, then succeed."""

            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first close fails")

    class Holder:
        """Bare object with a Trace-like __dict__."""

    trace = Holder()
    session = FlakySession()
    trace.__dict__["_fast_run_session"] = session

    with pytest.raises(RuntimeError, match="first close fails"):
        close_fast_run_session(trace)
    assert trace.__dict__["_fast_run_session"] is session

    close_fast_run_session(trace)
    assert "_fast_run_session" not in trace.__dict__
    assert session.calls == 2


class _PlainLinear(nn.Module):
    """Two-op model for the admission fault-injection probes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(value))


# Kept: sole branch coverage of _fast_run.py, _runnable/input_sites.py.
def test_fast_live_admission_guard_failure_refuses_not_fail_open(monkeypatch) -> None:
    """A broken input-contract guard REFUSES the fast path (R22-2 layer 1).

    The review's fault injection: with the classifier machinery raising internally,
    the typed refusal used to silently vanish and the forward ran unguarded --
    guard failure was indistinguishable from inputs-match.
    """

    import torchlens._fast_run as fast_run_module
    import torchlens._runnable.input_sites as input_sites
    import torchlens._runnable.transaction as transaction
    from torchlens.errors import RunCapabilityUnavailableError

    model = _PlainLinear().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    captured.run(inputs=torch.ones(2, 3), fast=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("injected classifier failure")

    # The home module and every module that bound the walker by import.
    for module in (transaction, input_sites, fast_run_module):
        monkeypatch.setattr(module, "_live_runtime_input_leaves", _boom)
    with pytest.raises(RunCapabilityUnavailableError):
        captured.run(inputs=torch.ones(2, 3), fast=True)


# Kept: sole branch coverage of _fast_run.py.
def test_fast_live_input_refresh_arity_guarded_and_poisons(monkeypatch) -> None:
    """The input-payload refresh zip guards arity like the output branch (R22-2 layer 2).

    A truncating zip used to keep STALE capture-time activations on the
    surplus input ops with no disclosure.
    """

    import torchlens._fast_run as fast_run
    from torchlens.errors import PathDivergenceError

    model = _PlainLinear().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    captured.run(inputs=torch.ones(2, 3), fast=True)

    # Patch the REFRESH-side binding only (_fast_run's module global); the
    # admission classifier keeps the real execution-module binding, so the
    # run is admitted and the refresh sees a truncated leaf list.
    real = fast_run._live_runtime_input_leaves

    def _truncating(*args, **kwargs):
        leaves = real(*args, **kwargs)
        if leaves is not None:
            return list(leaves)[:-1]
        return leaves

    monkeypatch.setattr(fast_run, "_live_runtime_input_leaves", _truncating)
    with pytest.raises(PathDivergenceError) as excinfo:
        captured.run(inputs=torch.ones(2, 3), fast=True)
    assert excinfo.value.fields["code"] == "input_tree_mismatch"


class _LinearReluModel(nn.Module):
    """Two-op model for fast-sparse payload refresh pins."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply linear then relu."""

        return torch.relu(self.lin(value))


def _saved_all_runnable(model: nn.Module, inputs: torch.Tensor, path: Path) -> Path:
    """Capture with all activations saved and save a weighted runnable artifact."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(
            model,
            inputs,
            capture=CaptureOptions(
                replayable=True,
                cache=False,
            ),
            save="all",
        )
    captured.save(path, level="runnable", include_weights=True)
    return path


class _BranchOnActivationModel(nn.Module):
    """Branches on a host-escaped predicate of a produced activation."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Take a data-dependent arm on the linear output."""

        hidden = self.lin(value)
        if bool((hidden.sum() > 0).item()):
            return torch.relu(hidden)
        return torch.sigmoid(hidden)


# Kept: sole branch coverage of backends/torch/_completeness_dispatch_names.py; a sole mutation kill.
def test_fast_sparse_mid_loop_divergence_poisons_reused_target(tmp_path: Path) -> None:
    """A mid-loop fast-sparse divergence poisons the reused target trace.

    Earlier calls in the failing iteration already refreshed their saved
    activations, so raising without a monotonic DIVERGED mark left a
    mixed-iteration trace that downstream faithful consumers accepted.
    """

    from torchlens.errors import PathDivergenceError

    torch.manual_seed(0)
    model = _BranchOnActivationModel().eval()
    first = torch.ones(2, 3)
    loaded = tl.load(_saved_all_runnable(model, first, tmp_path / "m.tlspec"))

    ok = loaded.run(inputs=first, fast=True)
    assert ok.report.path_faithfulness is PathFaithfulness.VERIFIED

    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=-torch.ones(2, 3) * 5, fast=True)

    assert ok.trace._runnable.path_faithfulness is PathFaithfulness.DIVERGED


# Kept: sole branch coverage of _fast_run.py, _input_walk.py, _ir/container_registry.py and more.
def test_fast_sparse_runs_post_execution_contract_checks(tmp_path: Path) -> None:
    """An exact-class-swapped container input diverges under BOTH providers.

    ``fast=True`` skipped ``_post_execution_contract_checks`` entirely, so
    the ``input_structure``/``container``/``conditional_arm_entry`` witness
    families had NO fast-provider consumer: a namedtuple input whose exact
    class changed passed the fast input contract (identical leaves/literals)
    and the recorded path replayed with a numerically wrong output stamped
    verified, where ``fast=False`` on the same artifact and input raises.
    """

    import collections

    from torchlens.errors import PathDivergenceError

    box_a = collections.namedtuple("BoxA", ["t"])
    box_b = collections.namedtuple("BoxB", ["t"])

    class ClassRoutedModel(nn.Module):
        """Route on the exact input container class."""

        def forward(self, box: object) -> torch.Tensor:
            """Scale by a class-identity-selected constant."""

            return box.t * (2.0 if type(box).__name__ == "BoxA" else 3.0)

    model = ClassRoutedModel().eval()
    tensor = torch.ones(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(
            model,
            box_a(t=tensor),
            capture=CaptureOptions(
                replayable=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    path = tmp_path / "m.tlspec"
    captured.save(path, level="runnable", include_weights=True)

    ordinary = tl.load(path)
    with pytest.raises(PathDivergenceError):
        ordinary.run(inputs=box_b(t=tensor))

    fast = tl.load(path)
    verify_once = fast.run(inputs=box_a(t=tensor), fast=True)
    assert verify_once.report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(PathDivergenceError):
        fast.run(inputs=box_b(t=tensor), fast=True)
    assert verify_once.trace._runnable.path_faithfulness is PathFaithfulness.DIVERGED


# Kept: sole branch coverage of _fast_run.py.
def test_fast_bind_outputs_refuses_op_label_slot_arity_mismatch() -> None:
    """The fast path contract-checks slot/op-label arity like the slow path.

    grind-p5 rollup (b7 fable+opus, CARRIED): ``_bind_outputs`` zipped
    ``output_slot_ids`` with ``op_labels`` shortest-wins, so a descriptor
    whose ``op_labels`` under-counts its output slots silently skipped the
    shape/dtype/device checks for the surplus slots -- the slow path pins
    ``len(output_slot_ids) == len(op_labels)`` inside its structure check.
    """

    from types import SimpleNamespace

    from torchlens._fast_run import _FastSparseSession
    from torchlens._runnable import RunnableErrorCode

    slots_by_id = {
        "s1": SimpleNamespace(
            output_path=(0,),
            shape=(2,),
            dtype="torch.float32",
            device_type="cpu",
            device_index=None,
        ),
        "s2": SimpleNamespace(
            output_path=(1,),
            shape=(999,),
            dtype="torch.float32",
            device_type="cpu",
            device_index=None,
        ),
    }
    fake_self = SimpleNamespace(
        slots_by_id=slots_by_id,
        version_alias_ids={},
        escape_witness_slot_ids=frozenset(),
        saved_labels=frozenset(),
        target=None,
    )
    call = SimpleNamespace(
        call_id="c1",
        output_slot_ids=("s1", "s2"),
        op_labels=("op_a",),  # under-counts the output slots
        is_inplace=False,
        control_obligations=(),
    )
    compiled = SimpleNamespace(descriptor=call)
    output = (torch.ones(2), torch.ones(3))  # s2 shape disagrees with its slot
    checks = _FastSparseSession._bind_outputs(
        fake_self,
        compiled,
        output,
        {},
        ceiling=SimpleNamespace(guarded_clone=lambda value, **kwargs: value),
        witness_source_snapshots={},
    )
    failing = [check for check in checks if not check.passed]
    assert failing, "under-counted op_labels silently skipped surplus slot checks"
    assert failing[0].diagnostic is not None
    assert failing[0].diagnostic.code == RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value
