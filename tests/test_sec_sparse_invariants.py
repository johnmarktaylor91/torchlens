"""Bundle replay validates sparse tensors: out-of-range indices are rejected, not built.

PyTorch skips its sparse index checks by default, so a recorded sparse constructor
whose indices exceed the declared shape would build an invalid tensor that later ops
read or write out of range. Every replay or execution of recorded calls therefore runs
under ``torch.sparse.check_sparse_tensor_invariants`` (restored afterwards), and a
recorded ``check_invariants`` keyword that turns the checks off is refused at bundle
parse and again at execution. Crash-prone cases run in a child process, so a
regression shows as a failing test rather than a dead runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from _format3_envelope import mutate_runnable_descriptor

import torchlens as tl
from torchlens._io.runnable_load import DescriptorStructuralBoundError, _validate_call_arity
from torchlens._runnable import (
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    ReadinessStatus,
    RunnableErrorCode,
    TensorArgumentRef,
)
from torchlens._utils import _torch_compat
from torchlens._utils._sparse_invariant_guard import call_with_sparse_invariants
from torchlens.errors import RunPreconditionError
from torchlens.intervention._types import FunctionRegistryKey
from torchlens.validation._autograd_grad_replay import execute_replay_func
from torchlens.validation._influence_probes import _ProbeCall, _substitutes_move_output
from torchlens.validation._loaded_callables import resolve_loaded_callables

_SWITCH = "check_invariants"


def _invariants_on() -> bool:
    """Return torch's current global sparse invariant-check setting."""

    return bool(torch._C._check_sparse_tensor_invariants())


def _bad_coo_args() -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Indices up to 2 against a declared 1x1 shape."""

    return torch.tensor([[0, 1, 2], [1, 0, 2]]), torch.ones(3), (1, 1)


# --------------------------------------------------------------------------- #
# (1) the guard: checks on for the call, previous setting restored            #
# --------------------------------------------------------------------------- #


def test_guard_rejects_out_of_range_indices() -> None:
    """A sparse constructor run through the guard refuses indices beyond the shape."""

    assert not _invariants_on()
    with pytest.raises(RuntimeError, match="(?i)index|indices"):
        call_with_sparse_invariants(torch.sparse_coo_tensor, _bad_coo_args(), {})
    assert not _invariants_on()


def test_guard_restores_the_previous_setting_on_return_and_on_raise() -> None:
    """The checks are on only inside the call; both settings come back afterwards."""

    seen: list[bool] = []

    def _record(*_args: Any, **_kwargs: Any) -> int:
        seen.append(_invariants_on())
        return 1

    def _boom(*_args: Any, **_kwargs: Any) -> int:
        seen.append(_invariants_on())
        raise ValueError("boom")

    assert not _invariants_on()
    assert call_with_sparse_invariants(_record, (), {}) == 1
    with pytest.raises(ValueError, match="boom"):
        call_with_sparse_invariants(_boom, (), {})
    assert seen == [True, True]
    assert not _invariants_on()
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        call_with_sparse_invariants(_record, (), {})
        assert _invariants_on()
    assert not _invariants_on()


def test_guard_refuses_the_switch_keyword_before_the_call() -> None:
    """``check_invariants=False`` would override the global checks; it never runs."""

    calls: list[Any] = []
    with pytest.raises(RunPreconditionError) as refusal:
        call_with_sparse_invariants(
            lambda *a, **k: calls.append(k), _bad_coo_args(), {_SWITCH: False}
        )
    assert refusal.value.fields["code"] == RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
    assert _SWITCH in str(refusal.value)
    assert calls == []
    for keeps in (True, None):
        call_with_sparse_invariants(lambda *a, **k: calls.append(k), (), {_SWITCH: keeps})
    assert len(calls) == 2


def test_without_the_check_sparse_construction_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where torch has no invariant check, replay refuses sparse construction outright."""

    monkeypatch.setattr(_torch_compat, "HAS_SPARSE_INVARIANT_CHECK", False)
    with pytest.raises(RunPreconditionError, match="sparse"):
        call_with_sparse_invariants(torch.sparse_coo_tensor, _bad_coo_args(), {})
    assert call_with_sparse_invariants(torch.add, (torch.ones(1), torch.ones(1)), {}).item() == 2


def test_capability_flag_is_published() -> None:
    """The flag is named in the capability snapshot that the doctor report shows."""

    snapshot = _torch_compat.get_torch_capability_snapshot()
    assert snapshot["HAS_SPARSE_INVARIANT_CHECK"] is True


def test_eager_capture_leaves_the_global_setting_alone() -> None:
    """Ordinary capture of a user's own model does not switch the checks on."""

    seen: list[bool] = []

    class _Peek(nn.Module):
        """Records the global setting while its forward runs."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``x + 1`` and note the current setting."""

            seen.append(_invariants_on())
            return x + 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.trace(_Peek(), torch.randn(2))
    assert seen and not any(seen)


# --------------------------------------------------------------------------- #
# (2) the switch keyword is refused at parse and at every replay boundary     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "value", "refused"),
    [
        (LiteralAtomKind.BOOL, False, True),
        (LiteralAtomKind.INT, 0, True),
        (LiteralAtomKind.BOOL, True, False),
        (LiteralAtomKind.NONE, None, False),
    ],
)
def test_parse_refuses_a_switch_that_turns_the_checks_off(
    kind: LiteralAtomKind, value: Any, refused: bool
) -> None:
    """``_validate_call_arity`` refuses ``check_invariants`` unless it is None or True."""

    literal = LiteralArgumentRef(
        argument_path=("kwargs", _SWITCH), value=LiteralAtom(kind=kind, value=value)
    )
    if not refused:
        _validate_call_arity("call-0", 0, 1, (), (literal,))
        return
    with pytest.raises(DescriptorStructuralBoundError) as refusal:
        _validate_call_arity("call-0", 0, 1, (), (literal,))
    assert refusal.value.code == RunnableErrorCode.CALL_ARITY_MISMATCH.value
    assert _SWITCH in str(refusal.value)


def test_parse_refuses_a_tensor_in_the_switch_keyword() -> None:
    """A tensor leaf filling ``check_invariants`` is refused like a False literal."""

    tensor = TensorArgumentRef(argument_path=("kwargs", _SWITCH), slot_id="s0")
    with pytest.raises(DescriptorStructuralBoundError, match=_SWITCH):
        _validate_call_arity("call-0", 0, 1, (tensor,), ())


class _Tiny(nn.Module):
    """One linear plus a softmax: the smallest replayable capture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a softmax over the linear output."""

        return torch.softmax(self.fc(x), dim=1)


def test_crafted_bundle_with_the_switch_off_refuses_at_load(tmp_path: Path) -> None:
    """End to end: a bundle whose call carries ``check_invariants=False`` never runs."""

    trace = tl.trace(
        _Tiny().eval(), torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True)
    )
    bundle = tmp_path / "switch.tlspec"
    trace.save(str(bundle), level="runnable", include_weights=True)

    def _craft(descriptor: dict[str, Any]) -> None:
        """Add a ``check_invariants=False`` keyword leaf to the first call."""

        call = descriptor["calls"][0]
        call["literal_arguments"].append(
            {"argument_path": ["kwargs", _SWITCH], "value": {"kind": "bool", "value": False}}
        )
        call["num_keyword_args"] = len(
            {
                tuple(argument["argument_path"])
                for argument in call["literal_arguments"]
                if argument["argument_path"][0] == "kwargs"
            }
        )

    mutate_runnable_descriptor(bundle, _craft)
    readiness = tl.load(str(bundle))._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert "descriptor_parse" in {d.detection_stage for d in readiness.diagnostics}
    assert any(_SWITCH in (d.message or "") for d in readiness.diagnostics)


class _Relu(nn.Module):
    """A linear and a relu: the relu op is rewritten for the replay-level checks."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relu of the linear output plus a residual."""

        return torch.relu(self.fc(x)) + x * 2.0


def _replay_trace() -> Any:
    """Capture with re-execution payloads."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(
            _Relu().eval(),
            torch.randn(3, 4),
            capture=tl.options.CaptureOptions(save_args=True, save_rng=True),
        )


@pytest.mark.parametrize("switch_off", [True, False])
def test_loaded_resolution_leaves_a_switch_off_op_unresolved(
    tmp_path: Path, switch_off: bool
) -> None:
    """A loaded op whose saved keywords turn the checks off is never armed for replay.

    The control (no switch) resolves, so the refusal keys on the keyword alone.
    """

    saved = {_SWITCH: False} if switch_off else {}
    trace = _replay_trace()
    target = next(op for op in trace._layer_list if op.func_name == "relu")
    target._internal_set("func_name", "sparse_coo_tensor")
    target._internal_set("func_id", FunctionRegistryKey("torch", "sparse_coo_tensor", "function"))
    target._internal_set("saved_kwargs", dict(saved))
    target._internal_set("non_tensor_kwargs", dict(saved))
    path = tmp_path / "switch.tlz"
    trace.save(path, level="executable_with_callables")
    loaded = tl.load(path)
    crafted = next(op for op in loaded._layer_list if op.func_name == "sparse_coo_tensor")
    resolved = resolve_loaded_callables(loaded)
    try:
        assert any(op is crafted for op in resolved) is not switch_off
    finally:
        for op in resolved:
            op.func = None


def test_replay_and_probe_belts_refuse_the_switch() -> None:
    """Validation replay and influence probes refuse before calling the op."""

    trace = _replay_trace()
    op = next(op for op in trace._layer_list if op.func_name == "relu")
    calls: list[Any] = []
    op.func = lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(3, 4)
    with pytest.raises(RunPreconditionError) as refusal:
        execute_replay_func(op, {"args": list(op.call.saved_args), "kwargs": {_SWITCH: False}}, [])
    assert refusal.value.fields["code"] == RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
    layer = SimpleNamespace(
        func=lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(2),
        out=torch.zeros(2),
        layer_label="crafted",
        func_name="sparse_coo_tensor",
    )
    call = _ProbeCall((torch.zeros(2),), {_SWITCH: False}, 0)
    with pytest.raises(RunPreconditionError):
        _substitutes_move_output(layer, call, lambda: (torch.ones(2),))  # type: ignore[arg-type]
    assert calls == []


# --------------------------------------------------------------------------- #
# (3) end to end: out-of-range sparse indices in a saved bundle, in a child   #
# --------------------------------------------------------------------------- #


def _child(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter (a memory fault kills only the child)."""

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


_RUN_CHILD = """
import torch, torchlens as tl
loaded = tl.load(__PATH__)
try:
    result = loaded.run_transaction(inputs=torch.arange(3, dtype=torch.float32))
    print("RAN", result.report.path_faithfulness, result.output.tolist())
except Exception as exc:
    print("RAISED", type(exc).__name__, getattr(exc, "fields", {}).get("code"), exc)
"""


class _SparseFromBuffer(nn.Module):
    """Builds a COO tensor from a recorded index buffer, then densifies it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", torch.tensor([[0, 1, 2], [1, 0, 2]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter ``x`` into a 3x3 sparse matrix and return it dense, doubled."""

        return torch.sparse_coo_tensor(self.idx, x, (3, 3)).to_dense() * 2.0


def _sparse_bundle(tmp_path: Path, shrink: bool) -> Path:
    """Save a runnable sparse bundle; ``shrink`` declares a 1x1 shape for 3x3 indices."""

    trace = tl.trace(
        _SparseFromBuffer().eval(),
        torch.arange(3, dtype=torch.float32),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    bundle = tmp_path / ("shrunk.tlspec" if shrink else "sparse.tlspec")
    trace.save(str(bundle), level="runnable", include_weights=True)

    def _shrink(node: Any) -> None:
        """Rewrite every integer 3 in a literal to 1."""

        if isinstance(node, dict):
            if node.get("kind") == "int" and node.get("value") == 3:
                node["value"] = 1
            for child in node.values():
                _shrink(child)
        elif isinstance(node, list):
            for child in node:
                _shrink(child)

    def _craft(descriptor: dict[str, Any]) -> None:
        """Shrink the size literal of the sparse constructor call."""

        for call in descriptor["calls"]:
            for argument in call.get("literal_arguments", []):
                if argument["argument_path"][:2] == ["args", 2]:
                    _shrink(argument)

    if shrink:
        mutate_runnable_descriptor(bundle, _craft)
    return bundle


def test_run_path_rejects_indices_beyond_the_declared_shape(tmp_path: Path) -> None:
    """``run_transaction`` refuses at construction with the signature-drift code."""

    child = _child(_RUN_CHILD.replace("__PATH__", repr(str(_sparse_bundle(tmp_path, shrink=True)))))
    assert child.returncode == 0, f"child died rc={child.returncode}: {child.stderr[-2000:]}"
    assert "RAISED RuntimeSignatureDriftError" in child.stdout, child.stdout + child.stderr
    assert RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value in child.stdout
    assert "inconsistent with indices" in child.stdout, child.stdout


def test_valid_sparse_bundle_still_runs(tmp_path: Path) -> None:
    """An honest sparse bundle replays and matches the eager result."""

    child = _child(
        _RUN_CHILD.replace("__PATH__", repr(str(_sparse_bundle(tmp_path, shrink=False))))
    )
    assert child.returncode == 0, child.stderr[-2000:]
    expected = (
        torch.sparse_coo_tensor(_SparseFromBuffer().idx, torch.arange(3.0), (3, 3)).to_dense() * 2
    )
    assert "RAN" in child.stdout, child.stdout + child.stderr[-1500:]
    assert str(expected.tolist()) in child.stdout, child.stdout


def test_valid_sparse_capture_still_validates() -> None:
    """Validation replay of an honest sparse capture still passes under the checks."""

    torch.manual_seed(0)
    model = _SparseFromBuffer().eval()
    x = torch.randn(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))
        assert trace.validate_forward_pass(model(x)) is True
    assert not _invariants_on()


class _Scale(nn.Module):
    """Scales a recorded index buffer by the input; the op is rewritten into a sparse ctor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", torch.tensor([[0, 1, 2], [1, 0, 2]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the index buffer times ``x`` (buffer at args 0, input at args 1)."""

        return torch.mul(self.idx, x)


def test_validate_path_rejects_indices_beyond_the_declared_shape(tmp_path: Path) -> None:
    """A crafted bundle replaying ``sparse_coo_tensor`` out of range fails, never verifies.

    The ``mul(idx, x)`` op is rewritten into ``torch.sparse_coo_tensor(idx, x, (1, 1))``,
    keeping both parents at their argument positions, with indices up to 2; replay must
    stop at construction with torch's invariant error.
    """

    torch.manual_seed(0)
    model = _Scale().eval()
    x = torch.randn(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))
    target = next(op for op in trace._layer_list if op.func_name == "mul")
    indices, values = target.call.saved_args[0], target.call.saved_args[1]
    assert indices.dtype == torch.int64 and values.dtype == torch.float32
    target._internal_set("func_name", "sparse_coo_tensor")
    target._internal_set("func_id", FunctionRegistryKey("torch", "sparse_coo_tensor", "function"))
    target._internal_set("saved_args", (indices, values, (1, 1)))
    target._internal_set("saved_kwargs", {})
    path = tmp_path / "crafted.tlz"
    trace.save(path, level="executable_with_callables")
    torch.save(model(x).detach(), tmp_path / "ref.pt")
    child = _child(
        f"""
        import warnings, torch, torchlens as tl
        ref = torch.load({str(tmp_path / "ref.pt")!r})
        import torchlens.validation.core as core
        filed = core.note_replay_exception
        def note(exc):
            print("REPLAY-EXC", type(exc).__name__, exc)
            filed(exc)
        core.note_replay_exception = note
        loaded = tl.load({str(path)!r})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verdict = loaded.validate_forward_pass(ref, verbose=True)
        print("VERDICT", verdict, getattr(loaded.validation_replay_status, "state", None))
        print("WARNED", any("treating as failed" in str(w.message) for w in caught))
        """
    )
    output = child.stdout + child.stderr[-1500:]
    assert child.returncode == 0, f"child died rc={child.returncode}: {output}"
    assert "VERDICT True" not in child.stdout and "passed" not in child.stdout, output
    assert "WARNED True" in child.stdout, output
    assert "inconsistent with indices" in child.stdout, output
