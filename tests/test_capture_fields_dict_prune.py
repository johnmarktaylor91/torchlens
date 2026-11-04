"""Deterministic counters for the per-op capture fields dict.

Capture builds one fields dict per op and freezes it into the op record
(``_op_record_from_log``). The dict used to carry placeholder and computed keys
the freeze never reads: graph placeholders postprocess fills (children, output
and distance flags, buffer and conditional fields), gradient placeholders,
module-I/O placeholders, final labels, and computed values materialize
recomputes from the record (parameter totals and memory, the parent-param map,
``equivalent_ops``, ``grad_fn_object_id``, ``internal_source_parents``). These
tests pin, on a tiny GPT-2 ``generate`` and on toy models that reach the
parameter, in-place, multi-output and internal-source paths, that no such key is
built, the number of keys built per op, and the Python calls capture makes per op.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_pkg
import torchlens.backends.torch.ops as torch_ops

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

# Most keys any one op's frozen fields dict holds on the tiny GPT-2 generate.
_MAX_KEYS_PER_OP = 0
# Python calls into TorchLens code per op while capture runs (postprocess excluded).
_MAX_TORCHLENS_CALLS_PER_OP = 0.0
_TORCHLENS_ROOT = os.path.dirname(tl.__file__)


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


def _frozen_key_sets(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any], x: Any, kw: dict[str, Any]
) -> list[frozenset[str]]:
    """Trace ``fn`` and return each op's fields-dict key set at the freeze."""

    seen: list[frozenset[str]] = []
    real = torch_ops._op_record_from_log

    def spy(trace: Any, fields_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Record the draft's keys, then freeze it for real."""

        seen.append(frozenset(fields_dict))
        return real(trace, fields_dict, *args, **kwargs)

    monkeypatch.setattr(torch_ops, "_op_record_from_log", spy)
    try:
        tl.trace(fn, x, input_kwargs=kw or None)
    finally:
        monkeypatch.setattr(torch_ops, "_op_record_from_log", real)
    return seen


def _capture_calls_per_op(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any], x: Any, kw: dict[str, Any]
) -> float:
    """Python calls into TorchLens code during capture, divided by the ops frozen."""

    owner = threading.get_ident()
    state = {"capture": True, "calls": 0, "ops": 0}
    real_freeze = torch_ops._op_record_from_log
    real_postprocess = postprocess_pkg.postprocess

    def profile(frame: FrameType, event: str, _arg: Any) -> None:
        """Count Python-level calls whose code lives in the torchlens package."""

        if (
            event == "call"
            and state["capture"]
            and frame.f_code.co_filename.startswith(_TORCHLENS_ROOT)
            and threading.get_ident() == owner
        ):
            state["calls"] += 1

    def freeze(*args: Any, **kwargs: Any) -> Any:
        """Count one op, then freeze it."""

        state["ops"] += 1
        return real_freeze(*args, **kwargs)

    def postprocess(*args: Any, **kwargs: Any) -> Any:
        """Run postprocess with counting paused."""

        state["capture"] = False
        try:
            return real_postprocess(*args, **kwargs)
        finally:
            state["capture"] = True

    previous = sys.getprofile()
    monkeypatch.setattr(torch_ops, "_op_record_from_log", freeze)
    monkeypatch.setattr(postprocess_pkg, "postprocess", postprocess)
    with quiet_gc():
        sys.setprofile(profile)
        try:
            tl.trace(fn, x, input_kwargs=kw or None)
        finally:
            sys.setprofile(previous)
            monkeypatch.setattr(torch_ops, "_op_record_from_log", real_freeze)
            monkeypatch.setattr(postprocess_pkg, "postprocess", real_postprocess)
    assert state["ops"] > 0
    return state["calls"] / state["ops"]


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
    print(f"[fields-dict] ops={len(key_sets)} max_keys={max(map(len, key_sets))}")
    assert sorted(built & _UNREAD_KEYS) == []
    assert max(map(len, key_sets)) <= _MAX_KEYS_PER_OP


def test_gpt2_generate_capture_python_calls_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture's Python calls into TorchLens per op on a GPT-2 generate stay at the pin."""

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    per_op = _capture_calls_per_op(monkeypatch, fn, ids, kw)
    print(f"[fields-dict] torchlens_calls_per_op={per_op:.2f}")
    assert per_op <= _MAX_TORCHLENS_CALLS_PER_OP
