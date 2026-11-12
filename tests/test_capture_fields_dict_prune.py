"""Deterministic counters for the per-op capture fields dict.

Capture builds one fields dict per op and commits it as the op's row
(``_ops_row_journal.commit_row_record``, which reads what the record freeze
``_op_record_from_log`` reads). The dict used to carry placeholder and computed keys
the freeze never reads: graph placeholders postprocess fills (children, output
and distance flags, buffer and conditional fields), gradient placeholders,
module-I/O placeholders, final labels, and computed values materialize
recomputes from the record (parameter totals and memory, the parent-param map,
``equivalent_ops``, ``grad_fn_object_id``, ``internal_source_parents``). These
tests pin, on a tiny GPT-2 ``generate`` and on toy models that reach the
parameter, in-place, multi-output and internal-source paths, that no such key is
built, the number of keys built per op, the Python calls capture makes per op,
and the Python calls dropping the returned trace makes per op.
Source tensors and module-boundary outputs have their own builders (no function
call: ``func_call_id`` is None) and are not counted.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_pkg
from torchlens.backends.torch import _ops_row_journal, _thin_emit

# Keys the per-op build wrote that nothing reads before or at the freeze.
_UNREAD_KEYS = frozenset(
    {
        "_param_barcodes",
        "_param_logs",
        "address",
        "atomic_module_call",
        "buffer_pass",
        "buffer_replay_validated",
        "buffer_source",
        "buffer_source_func_name",
        "buffer_value_changed",
        "buffer_write_kind",
        "children",
        "conditional_arm_children",
        "conditional_branch_depth",
        "conditional_branch_stack",
        "conditional_context_kind",
        "conditional_elif_children",
        "conditional_else_children",
        "conditional_entry_children",
        "conditional_then_children",
        "conditional_wrapper_kind",
        "edits",
        "equivalent_ops",
        "grad",
        "grad_dtype",
        "grad_fn",
        "grad_fn_object_id",
        "grad_shape",
        "gradient_memory",
        "has_children",
        "has_grad",
        "has_input_ancestor",
        "has_out_variations",
        "has_output_descendant",
        "in_conditionals",
        "input_to_module_calls",
        "internal_source_parents",
        "io_role",
        "is_atomic_module",
        "is_final_output",
        "is_in_conditional_body",
        "is_internal_sink",
        "is_module_output",
        "is_orphan",
        "is_output",
        "is_terminal_bool",
        "is_terminal_conditional_bool",
        "label",
        "layer_label",
        "lookup_keys",
        "max_distance_from_input",
        "max_distance_to_output",
        "min_distance_from_input",
        "min_distance_to_output",
        "module_call_stack",
        "module_entry_arg_keys",
        "multi_output_name",
        "num_params",
        "num_params_frozen",
        "num_params_trainable",
        "num_passes",
        "output_descendants",
        "output_of_module_calls",
        "output_of_modules",
        "param_memory",
        "parent_param_ops",
        "recurrent_ops",
        "short_label",
        "short_layer_label",
        "source_trace",
        "terminal_bool_for",
        "terminal_conditional_id",
        "transformed_grad",
        "transformed_grad_dtype",
        "transformed_grad_shape",
        "transformed_gradient_memory",
        "var_names",
    }
)

# Most keys any one function-call op's frozen fields dict holds on the tiny GPT-2
# generate: 91 (it was 167 while the unread keys were built).
_MAX_KEYS_PER_OP = 91
# Python calls into TorchLens code per function-call op while capture runs
# (postprocess and the returned trace's teardown excluded). Lowered from 1082 when the
# op-time argument snapshot's common path stopped making Python calls: measured 1077.05
# serial (448052 calls over 416 ops) and up to 1077.10 under xdist, where neighbouring
# tests' process state moves it; the pin keeps about one call per op for that drift.
_MAX_TORCHLENS_CALLS_PER_OP = 1078.01
# Python calls into TorchLens code per function-call op while the returned trace
# is dropped (it dies by refcount; the payload eviction walk dominates). Pinned at
# the value measured on CI's pinned environment (torch 2.13.0+cpu): 7973 calls over
# 416 ops, 19.17 (36163 calls, 86.93, before the walk skipped opaque record types
# and all-scalar tuples).
_MAX_TEARDOWN_CALLS_PER_OP = 7973 / 416
_TORCHLENS_ROOT = os.path.dirname(tl.__file__)


@pytest.fixture(autouse=True)
def _thin_emit_off() -> Iterator[None]:
    """Pin today's emit: these tests spy on its commit (``commit_row_record``).

    The thin emit appends its rows without that commit; its rows are checked
    against today's in ``tests/test_thin_emit_soundness.py``.
    """

    previous = _thin_emit.set_thin_emit_enabled(False)
    yield
    _thin_emit.set_thin_emit_enabled(previous)


class _InPlaceMlp(nn.Module):
    """Parameters, an in-place op, a buffer update, a split and a fresh tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.norm = nn.BatchNorm1d(6)
        self.register_buffer("count", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu_(self.fc(x))
        h = self.norm(h)
        self.count.add_(1)
        left, right = torch.split(h, 3, dim=1)
        return left * right + torch.ones(1)


def _tiny_gpt2_generate() -> tuple[Callable[..., Any], torch.Tensor, dict[str, Any]]:
    """A random-weight two-layer GPT-2 and its bound ``generate`` (three new tokens)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2, pad_token_id=0
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    kwargs = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    return model.generate, torch.tensor([[3, 7, 1, 9]]), kwargs


def _is_function_call_draft(fields_dict: dict[str, Any]) -> bool:
    """Whether a draft came from the per-op build (a function call), not a source."""

    return fields_dict.get("func_call_id") is not None


def _frozen_key_sets(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any], x: Any, kw: dict[str, Any]
) -> list[frozenset[str]]:
    """Trace ``fn`` and return each function-call op's fields-dict key set at the freeze."""

    seen: list[frozenset[str]] = []
    real = _ops_row_journal.commit_row_record

    def spy(draft: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the draft's keys, then commit it for real."""

        if _is_function_call_draft(draft.fields_dict):
            seen.append(frozenset(draft.fields_dict))
        return real(draft, *args, **kwargs)

    monkeypatch.setattr(_ops_row_journal, "commit_row_record", spy)
    try:
        tl.trace(fn, x, input_kwargs=kw or None)
    finally:
        monkeypatch.setattr(_ops_row_journal, "commit_row_record", real)
    return seen


def _capture_and_teardown_calls_per_op(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any], x: Any, kw: dict[str, Any]
) -> tuple[float, float]:
    """Python calls into TorchLens code per function-call op frozen, capture then teardown.

    Capture is ``tl.trace`` minus postprocess. Teardown is dropping the returned
    trace: it dies by refcount, and its payload eviction runs then.
    """

    owner = threading.get_ident()
    state: dict[str, Any] = {"phase": "capture", "capture": 0, "teardown": 0, "ops": 0}
    real_freeze = _ops_row_journal.commit_row_record
    real_postprocess = postprocess_pkg.postprocess

    def profile(frame: FrameType, event: str, _arg: Any) -> None:
        """Count Python-level calls whose code lives in the torchlens package."""

        if (
            event == "call"
            and state["phase"] != "paused"
            and frame.f_code.co_filename.startswith(_TORCHLENS_ROOT)
            and threading.get_ident() == owner
        ):
            state[state["phase"]] += 1

    def freeze(draft: Any, *args: Any, **kwargs: Any) -> Any:
        """Count one function-call op, then commit it."""

        if _is_function_call_draft(draft.fields_dict):
            state["ops"] += 1
        return real_freeze(draft, *args, **kwargs)

    def postprocess(*args: Any, **kwargs: Any) -> Any:
        """Run postprocess with counting paused."""

        state["phase"] = "paused"
        try:
            return real_postprocess(*args, **kwargs)
        finally:
            state["phase"] = "capture"

    previous = sys.getprofile()
    monkeypatch.setattr(_ops_row_journal, "commit_row_record", freeze)
    monkeypatch.setattr(postprocess_pkg, "postprocess", postprocess)
    with quiet_gc():
        sys.setprofile(profile)
        try:
            traced = tl.trace(fn, x, input_kwargs=kw or None)
            state["phase"] = "teardown"
            del traced
        finally:
            sys.setprofile(previous)
            monkeypatch.setattr(_ops_row_journal, "commit_row_record", real_freeze)
            monkeypatch.setattr(postprocess_pkg, "postprocess", real_postprocess)
    assert state["ops"] > 0
    return state["capture"] / state["ops"], state["teardown"] / state["ops"]


def test_toy_model_fields_dicts_build_no_unread_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parameter, in-place, buffer, multi-output and fresh-tensor ops build no unread key."""

    torch.manual_seed(0)
    model = _InPlaceMlp().train()
    key_sets = _frozen_key_sets(monkeypatch, model, torch.randn(4, 6), {})
    assert len(key_sets) >= 8
    built = frozenset().union(*key_sets)
    assert sorted(built & _UNREAD_KEYS) == []


def test_gpt2_generate_fields_dict_keys_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPT-2 generate builds no unread key and at most the pinned keys per op."""

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    key_sets = _frozen_key_sets(monkeypatch, fn, ids, kw)
    assert len(key_sets) > 100
    built = frozenset().union(*key_sets)
    sizes = [len(keys) for keys in key_sets]
    print(f"[fields-dict] ops={len(sizes)} max_keys={max(sizes)} min_keys={min(sizes)}")
    assert sorted(built & _UNREAD_KEYS) == []
    assert max(sizes) <= _MAX_KEYS_PER_OP


def test_gpt2_generate_capture_python_calls_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture's Python calls into TorchLens per op on a GPT-2 generate stay at the pin."""

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    per_op, _teardown = _capture_and_teardown_calls_per_op(monkeypatch, fn, ids, kw)
    print(f"[fields-dict] torchlens_calls_per_op={per_op:.2f}")
    assert per_op <= _MAX_TORCHLENS_CALLS_PER_OP


def test_gpt2_generate_trace_teardown_python_calls_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping a GPT-2 generate trace makes at most the pinned Python calls per op."""

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    _capture, per_op = _capture_and_teardown_calls_per_op(monkeypatch, fn, ids, kw)
    print(f"[fields-dict] torchlens_teardown_calls_per_op={per_op:.2f}")
    assert per_op <= _MAX_TEARDOWN_CALLS_PER_OP
