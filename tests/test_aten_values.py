"""ATen output values: ``CaptureOptions(aten_ops="values")`` and ``AtenOp.out``.

``aten_ops=True`` keeps the rows value-free; ``"values"`` also stores each ATen
call's returned tensor(s) as ``AtenOp.out`` under the ``Op.out`` save rules
(the owning torch call's ``save=`` selection, ``detach_saved_activations``,
``output_device``, ``save_mode``, the save budget). Values are snapshots taken
at the dispatcher return seam, below autograd, so they never carry a
``grad_fn``; in-place and view ops store what the call returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from tests.classics_corpus._loader import build_entry, corpus_entries
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.backends.torch import _aten_capture
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants

_VALUES = tl.options.CaptureOptions(aten_ops="values")
_VALUES_REFERENCE = tl.options.CaptureOptions(aten_ops="values", save_mode="reference")
_ROWS_ONLY = tl.options.CaptureOptions(aten_ops=True)


class _BlockModel(nn.Module):
    """linear (addmm), conv2d, layer_norm, tanh-GELU, an in-place add, a view."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block the value tests recompute by hand."""

        h = self.linear(x)
        h = h.view(2, 1, 4, 4)
        h = self.conv(h)
        h = h.view(2, 2, 16)
        h = self.norm(h)
        h = F.gelu(h, approximate="tanh")
        h = h.add_(1.0)
        return h


class _MaxModel(nn.Module):
    """One multi-output torch call (``torch.max`` along a dim)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return values and indices."""

        values, indices = torch.max(x * 2, dim=1)
        return values, indices


def _rows(trace: tl.Trace, operator: str) -> list[Any]:
    """Return the recorded rows for one ATen operator in dispatch order."""

    return [row for row in trace.aten_ops if row.operator == operator]


def _leaves(value: Any) -> tuple[torch.Tensor, ...]:
    """Return the tensor leaves of one ``AtenOp.out``."""

    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def _recompute(model: _BlockModel, x: torch.Tensor) -> dict[str, Any]:
    """Recompute every block of ``_BlockModel`` with plain torch, in order."""

    with torch.no_grad():
        lin = torch.addmm(model.linear.bias, x, model.linear.weight.t())
        as_image = lin.view(2, 1, 4, 4)
        conv = F.conv2d(as_image, model.conv.weight, model.conv.bias, padding=1)
        flat = conv.view(2, 2, 16)
        mean = flat.mean(-1, keepdim=True)
        var = flat.var(-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + model.norm.eps)
        normed = F.layer_norm(flat, (16,), model.norm.weight, model.norm.bias, model.norm.eps)
        gelu = F.gelu(normed, approximate="tanh")
        final = gelu + 1.0
    return {
        "addmm": lin,
        "view_1": as_image,
        "convolution": conv,
        "view_2": flat,
        "native_layer_norm": (normed, mean, rstd),
        "gelu": gelu,
        "add_": final,
    }


def _block_trace(**capture: Any) -> tuple[_BlockModel, torch.Tensor, tl.Trace]:
    """Capture ``_BlockModel`` with values on plus any extra capture options."""

    torch.manual_seed(0)
    model = _BlockModel().eval()
    x = torch.randn(2, 16)
    options = tl.options.CaptureOptions(aten_ops="values", **capture)
    return model, x, tl.trace(model, x.clone(), capture=options)


def test_values_equal_a_plain_torch_recomputation() -> None:
    """addmm, conv, layer_norm, tanh-GELU, an in-place op and a view match plain torch."""

    model, x, trace = _block_trace()
    expected = _recompute(model, x)

    assert torch.equal(_rows(trace, "addmm")[0].out, expected["addmm"])
    assert torch.equal(_rows(trace, "convolution")[0].out, expected["convolution"])
    assert torch.equal(_rows(trace, "gelu")[0].out, expected["gelu"])

    norm_out = _rows(trace, "native_layer_norm")[0].out
    assert isinstance(norm_out, tuple) and len(norm_out) == 3
    assert torch.equal(norm_out[0], expected["native_layer_norm"][0])
    assert torch.allclose(norm_out[1].reshape(-1), expected["native_layer_norm"][1].reshape(-1))
    assert torch.allclose(norm_out[2].reshape(-1), expected["native_layer_norm"][2].reshape(-1))

    views = _rows(trace, "view")
    assert [tuple(row.out.shape) for row in views] == [(2, 1, 4, 4), (2, 2, 16)]
    assert torch.equal(views[0].out, expected["view_1"])
    assert torch.equal(views[1].out, expected["view_2"])
    assert views[0].view_copy_kind == "view"

    inplace = _rows(trace, "add_")[0]
    assert inplace.mutation_kind == "in_place"
    assert inplace.view_copy_kind == "alias"
    assert inplace.owner_status == "forward_op"
    assert torch.equal(inplace.out, expected["add_"])
    # The in-place row stores what the call returned: the mutated tensor.
    assert torch.equal(inplace.out, trace.output_ops[0].out)
    assert check_metadata_invariants(trace)


def test_values_are_snapshots_and_every_op_reproduces_its_out() -> None:
    """A kept value is a copy at return time; each saved Op's out equals one of its values."""

    _, _, trace = _block_trace()
    for op in trace.ops:
        leaves = [leaf for row in op.aten_ops for leaf in _leaves(row.out) if row.out is not None]
        assert leaves, f"{op.layer_label} owns no valued ATen rows"
        assert any(leaf.shape == op.out.shape and torch.equal(leaf, op.out) for leaf in leaves), (
            op.layer_label
        )
        for leaf in leaves:
            assert leaf.data_ptr() != op.out.data_ptr()
    for row in trace.aten_ops:
        for leaf, fact in zip(_leaves(row.out), row.output_tensor_facts, strict=True):
            assert tuple(leaf.shape) == tuple(fact.shape)
            assert str(leaf.dtype) == fact.dtype
    assert trace.aten_gaps == ()


def test_multi_output_aten_op_keeps_one_value_per_output_fact() -> None:
    """A tuple-returning ATen op stores a tuple, one tensor per output fact, in order."""

    torch.manual_seed(0)
    x = torch.randn(3, 5)
    trace = tl.trace(_MaxModel(), x, capture=_VALUES)
    row = _rows(trace, "max")[0]
    assert row.overload == "dim"
    assert isinstance(row.out, tuple) and len(row.out) == len(row.output_tensor_facts) == 2
    values, indices = torch.max(x * 2, dim=1)
    assert torch.equal(row.out[0], values)
    assert torch.equal(row.out[1], indices)
    assert row.out[1].dtype == torch.int64
    max_ops = [op for op in trace.ops if op.func_name == "max"]
    assert len(max_ops) == 2 and all(row in op.aten_ops for op in max_ops)
    assert check_metadata_invariants(trace)


def test_save_selection_restricts_which_calls_keep_values() -> None:
    """Rows beneath unsaved torch calls drop their values; the saved call keeps them."""

    torch.manual_seed(0)
    model = _BlockModel().eval()
    trace = tl.trace(model, torch.randn(2, 16), save=tl.func("conv2d"), capture=_VALUES)

    saved = [op for op in trace.ops if op.has_saved_activation]
    assert [op.func_name for op in saved] == ["conv2d"]
    conv_rows = saved[0].aten_ops
    assert conv_rows and all(row.out is not None for row in conv_rows)
    assert torch.equal(_rows(trace, "convolution")[0].out, saved[0].out)
    for op in trace.ops:
        if op is saved[0]:
            continue
        assert op.aten_ops, op.layer_label
        assert all(row.out is None for row in op.aten_ops), op.layer_label
        # Facts stay: "not kept" is distinguishable from "no tensor result".
        assert all(row.output_tensor_facts for row in op.aten_ops)
    assert check_metadata_invariants(trace)


def test_save_selection_frees_dropped_values_during_the_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped owner's values are released by the next recorded call, not at the end."""

    freed: list[int] = []
    real_release = _aten_capture._release_dropped_aten_values

    def counting_release(state: Any, *, keep_owner: int | None) -> None:
        """Count how many pending owners the eager prune frees."""

        before = len(state.aten_value_cells)
        real_release(state, keep_owner=keep_owner)
        freed.append(before - len(state.aten_value_cells))

    monkeypatch.setattr(_aten_capture, "_release_dropped_aten_values", counting_release)
    torch.manual_seed(0)
    trace = tl.trace(
        _BlockModel().eval(), torch.randn(2, 16), save=tl.func("gelu"), capture=_VALUES
    )
    assert sum(freed) >= 3, freed
    gelu_op = next(op for op in trace.ops if op.func_name == "gelu")
    assert all(row.out is not None for row in gelu_op.aten_ops)
    assert check_metadata_invariants(trace)


def test_detach_policy_and_graph_connection() -> None:
    """Values are detached snapshots: no grad_fn, no requires_grad, either way."""

    # backward_ready requires attached activations, so only the attach case asks for it.
    for detach, extra in ((True, {}), (False, {"backward_ready": True})):
        _, _, trace = _block_trace(detach_saved_activations=detach, **extra)
        for row in trace.aten_ops:
            for leaf in _leaves(row.out):
                assert leaf.grad_fn is None
                assert leaf.requires_grad is False
        assert check_metadata_invariants(trace)
    # The Op path keeps its own contract: an attached Op.out stays graph-connected.
    assert any(op.out.requires_grad for op in trace.ops)


def test_output_device_and_save_mode_are_honored() -> None:
    """``output_device`` places the copies; ``save_mode="reference"`` stores aliases."""

    _, _, trace = _block_trace(output_device="cpu")
    assert all(leaf.device.type == "cpu" for row in trace.aten_ops for leaf in _leaves(row.out))

    torch.manual_seed(0)
    model = _BlockModel().eval()
    referenced = tl.trace(model, torch.randn(2, 16), capture=_VALUES_REFERENCE)
    addmm = _rows(referenced, "addmm")[0]
    assert addmm.out is not None
    # In-place calls fall back to a copy under reference mode, as Op.out does.
    inplace = _rows(referenced, "add_")[0]
    assert torch.equal(inplace.out, referenced.output_ops[0].out)
    # Reading a reference out that a later in-place call mutated raises by design
    # (MutatedReferenceError), so the invariant sweep runs on a mutation-free model.
    torch.manual_seed(0)
    plain_ref = tl.trace(_MaxModel(), torch.randn(3, 5), capture=_VALUES_REFERENCE)
    assert all(row.out is not None for row in plain_ref.aten_ops)
    assert check_metadata_invariants(plain_ref)


def test_values_count_against_the_save_budget() -> None:
    """ATen values are admitted through the same accountant as Op.out."""

    def committed(trace: tl.Trace) -> int:
        """Bytes the capture's accountant holds for CPU payloads."""

        ledger = trace._save_budget_accountant._ledger_for(torch.device("cpu"))
        return int(ledger.committed_bytes)

    def capture(aten_ops: Any, budget: int) -> tl.Trace:
        """Capture the block model under one switch spelling and budget."""

        torch.manual_seed(0)
        return tl.trace(
            _BlockModel().eval(),
            torch.randn(2, 16),
            capture=tl.options.CaptureOptions(aten_ops=aten_ops, save_budget=budget),
        )

    unlimited = 1 << 30
    rows_only, valued = capture(True, unlimited), capture("values", unlimited)
    rows_bytes, valued_bytes = committed(rows_only), committed(valued)
    value_bytes = sum(
        leaf.numel() * leaf.element_size() for row in valued.aten_ops for leaf in _leaves(row.out)
    )
    # The accountant charges physical storage (alias-aware, credited on release),
    # so the charge for the values is bounded by their logical bytes, never above.
    charged = valued_bytes - rows_bytes
    assert 0 < charged <= value_bytes
    assert charged >= value_bytes * 0.8, (charged, value_bytes)
    budget = (rows_bytes + valued_bytes) // 2
    assert capture(True, budget).has_aten_ops
    with pytest.raises(SaveBudgetExceededError, match="aten:"):
        capture("values", budget)


def test_rows_only_switch_never_copies_and_keeps_rows_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``aten_ops=True`` never enters the value path; ``aten_ops=False`` never records."""

    def fail_if_called(*args: object, **kwargs: object) -> None:
        """Fail if the value path runs without ``"values"``."""

        del args, kwargs
        raise AssertionError("ATen value retention ran without aten_ops='values'")

    monkeypatch.setattr(_aten_capture, "_retain_aten_values", fail_if_called)
    torch.manual_seed(0)
    trace = tl.trace(_BlockModel().eval(), torch.randn(2, 16), capture=_ROWS_ONLY)
    assert trace.aten_ops and all(row.out is None for row in trace.aten_ops)

    monkeypatch.setattr(_aten_capture, "_prepare_aten_call", fail_if_called)
    off = tl.trace(_BlockModel().eval(), torch.randn(2, 16))
    assert off.has_aten_ops is False


def test_switch_rejects_unknown_spellings() -> None:
    """Only ``False``, ``True`` and ``"values"`` are accepted."""

    with pytest.raises(tl.errors.InvalidArgumentError) as excinfo:
        tl.options.CaptureOptions(aten_ops="all")
    assert excinfo.value.fields["code"] == "aten_ops_invalid"
    with pytest.raises(tl.errors.InvalidArgumentError):
        tl.options.CaptureOptions(aten_ops=1)
    assert tl.options.CaptureOptions(aten_ops="values").aten_ops == "values"


def test_values_round_trip_through_save_and_load(tmp_path: Path) -> None:
    """Values persist where Op.out persists and drop with ``include_outs=False``."""

    _, _, trace = _block_trace()
    path = tmp_path / "values.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    assert loaded.has_aten_ops
    assert len(loaded.aten_ops) == len(trace.aten_ops)
    for before, after in zip(trace.aten_ops, loaded.aten_ops, strict=True):
        assert type(after.out) is type(before.out)
        for a, b in zip(_leaves(before.out), _leaves(after.out), strict=True):
            assert torch.equal(a, b)
    assert check_metadata_invariants(loaded)

    stripped = tmp_path / "stripped.tlspec"
    trace.save(stripped, include_outs=False)
    bare = tl.load(stripped)
    assert bare.has_aten_ops and len(bare.aten_ops) == len(trace.aten_ops)
    assert all(row.out is None for row in bare.aten_ops)
    assert all(row.output_tensor_facts for row in bare.aten_ops)
    assert check_metadata_invariants(bare)


def test_backward_rows_keep_values_when_the_profile_records_them() -> None:
    """Backward ATen rows carry values with the same per-fact shape contract."""

    _, _, trace = _block_trace(backward_ready=True)
    trace.log_backward(trace.output_ops[0].out.sum())
    backward = [row for row in trace.aten_ops if row.capture_phase == "backward"]
    assert backward
    valued = [row for row in backward if row.out is not None]
    assert valued
    for row in valued:
        for leaf, fact in zip(_leaves(row.out), row.output_tensor_facts, strict=True):
            assert tuple(leaf.shape) == tuple(fact.shape)
    assert check_metadata_invariants(trace)


def test_values_leave_the_op_graph_unchanged() -> None:
    """Recording values never changes what the torch-level capture logs (in-place ops too)."""

    captures = []
    for options in (tl.options.CaptureOptions(), _ROWS_ONLY, _VALUES):
        torch.manual_seed(0)
        model = _BlockModel().eval()
        captures.append(tl.trace(model, torch.randn(2, 16), capture=options))
    plain, rows_only, valued = captures
    labels = [[op.layer_label for op in trace.ops] for trace in captures]
    assert labels[0] == labels[1] == labels[2]
    assert any(label.startswith("add_") for label in labels[0])
    assert plain._raw_event_shape_hash == rows_only._raw_event_shape_hash
    assert plain._raw_event_shape_hash == valued._raw_event_shape_hash
    # The recorder's own tensor reads never count as calls: the ordered call
    # fingerprint (the fast tier's structure guard) matches a plain capture.
    assert plain._raw_call_fingerprint is not None
    assert plain._raw_call_fingerprint == rows_only._raw_call_fingerprint
    assert plain._raw_call_fingerprint == valued._raw_call_fingerprint
    inplace = _rows(valued, "add_")[0]
    assert inplace.owner_status == "forward_op"
    assert inplace.parent_op_refs[0].op_label.startswith("add_")


def test_refresh_re_records_values() -> None:
    """``run()`` refreshes re-record the profile, so values follow the new input."""

    torch.manual_seed(0)
    model = _BlockModel().eval()
    trace = tl.trace(model, torch.randn(2, 16), capture=_VALUES)
    stale = _rows(trace, "gelu")[0].out.clone()
    x2 = torch.randn(2, 16)
    refreshed = trace.run(model, x2)
    assert refreshed.has_aten_ops
    assert all(row.out is not None for row in refreshed.aten_ops if row.output_tensor_facts)
    fresh = _rows(refreshed, "gelu")[0].out
    assert torch.equal(fresh, _recompute(model, x2)["gelu"])
    assert not torch.equal(fresh, stale)
    assert check_metadata_invariants(refreshed)


def test_fast_run_refuses_a_values_capture() -> None:
    """The fast tier never re-records ATen values, so it refuses a values capture typed."""

    from torchlens.errors import RunCapabilityUnavailableError

    torch.manual_seed(0)
    model = _BlockModel().eval()
    x = torch.randn(2, 16)
    trace = tl.trace(model, x, capture=_VALUES)
    with pytest.raises(RunCapabilityUnavailableError) as refused:
        trace.run(inputs=torch.randn(2, 16), fast=True)
    assert refused.value.fields["detection_stage"] == "fast_run_aten_values_unsupported"
    assert torch.equal(_rows(trace, "gelu")[0].out, _recompute(model, x)["gelu"])


def test_rows_capture_admits_the_fast_tier() -> None:
    """A rows-only capture keeps the fast tier: its fingerprint is a plain capture's."""

    torch.manual_seed(0)
    model = _BlockModel().eval()
    trace = tl.trace(model, torch.randn(2, 16), capture=_ROWS_ONLY)
    fast = trace.run(inputs=torch.randn(2, 16), fast=True)
    assert fast.trace is not trace
    assert fast.trace.has_aten_ops is True
    assert len(fast.trace.ops) == len(trace.ops)


def test_invariant_rejects_a_value_that_disagrees_with_its_facts_or_owner() -> None:
    """``aten_values_consistent`` is a tripwire: a wrong value fails validation."""

    _, _, trace = _block_trace()
    row = _rows(trace, "gelu")[0]
    good = row.out
    row.out = torch.zeros(1)
    with pytest.raises(MetadataInvariantError) as excinfo:
        check_metadata_invariants(trace)
    assert "aten_values_consistent" in str(excinfo.value)

    row.out = good + 1
    with pytest.raises(MetadataInvariantError, match="equals its saved output"):
        check_metadata_invariants(trace)

    row.out = good
    assert check_metadata_invariants(trace)
    for other in trace.aten_ops:
        if other is not row and other.owner_func_call_id == row.owner_func_call_id:
            pytest.fail("gelu should own exactly one ATen row")
    gelu_op = next(op for op in trace.ops if op.func_name == "gelu")
    gelu_op.has_saved_activation = False
    with pytest.raises(MetadataInvariantError, match="no owning Op saved"):
        check_metadata_invariants(trace)


_SMOKE_ENTRIES = {entry.id: entry for entry in corpus_entries("smoke")}


@pytest.mark.parametrize(
    "entry_id",
    ["Avalanche multi-task simple MLP compact", "Sionna-NeuralReceiver"],
)
def test_classics_smoke_models_keep_values_and_validate(entry_id: str) -> None:
    """Classics smoke models with values on replay and pass every invariant."""

    model, inputs = build_entry(_SMOKE_ENTRIES[entry_id])
    trace = tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(aten_ops="values", save_args=True),
        save="all",
    )
    assert trace.has_aten_ops and trace.aten_ops
    assert any(row.out is not None for row in trace.aten_ops)
    with torch.no_grad():
        expected = model(*inputs)
    assert trace.validate_forward_pass(expected, validate_metadata=True) is True
