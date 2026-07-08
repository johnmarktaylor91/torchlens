"""Regression gate for the uninitialized-destination perturbation class (B2).

The dominant ``failed:replay`` class in the menagerie sweep (the diffusers VIDEO
transformers -- Wan*/SanaVideo/SkyReels/ChronoEdit/Helios -- plus the
normalizing-flow families normflows/zuko and torch_geometric's NeuralFingerprint)
was the perturbation check firing on a BENIGN idiom:

    out = torch.empty_like(x)      # or x.new_empty(...) / out.index_copy_(...)
    out[..., 0::2] = a             # partition-write into uninitialized memory
    out[..., 1::2] = b

Perturbing the uninitialized destination of a partial in-place write is
meaningless -- the unwritten positions are garbage overwritten by a sibling
write and never meaningfully consumed. The fix exempts ONLY that case (the
perturbed parent occupies the destination slot AND is ITSELF a direct
uninitialized-memory allocation: empty/empty_like/new_empty).

LOCKED tripwire invariant under test: the exemption is NARROW. A REAL data
tensor feeding an in-place-write destination is NOT exempted -- a genuine dropped
dependency must still fail. This INCLUDES a prior in-place write whose buffer was
allocated by empty_like but which now holds live written data: the exemption does
NOT chain through intermediate writes (B1 adversarial-review hole). The
``test_real_data_destination_*`` and ``test_chained_index_copy_dropped_dest_*``
cases guard that the exemption never widens into a bug mask.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchlens.user_funcs import (
    _run_model_and_save_specified_outs,
    _validate_forward_pass_torch,
)
from torchlens.validation.core import validate_saved_outs
from torchlens.validation.diagnostics import (
    CHECK_PERTURBATION,
    CHECK_REPLAY,
    get_validation_failure,
)
from torchlens.validation.exemptions import (
    INPLACE_DESTINATION_WRITE_FUNCS,
    _uninitialized_value_origin,
)


def _validate_with_failure(model: nn.Module, x: torch.Tensor):
    """Run forward validation and return (result, structured_failure)."""

    captured: dict[str, object] = {}

    def observe(trace):
        captured["failure"] = get_validation_failure(trace)

    result = _validate_forward_pass_torch(model, x, validate_metadata=True, _trace_observer=observe)
    return result, captured.get("failure")


# ---------------------------------------------------------------------------
# Helper-level strictness: _uninitialized_value_origin
# ---------------------------------------------------------------------------


def _fake_op(func_name: str, dest_label: str | None = None):
    """Build a minimal op stand-in for the value-origin walk."""

    args = {} if dest_label is None else {0: dest_label}
    return SimpleNamespace(func_name=func_name, parent_arg_positions={"args": args})


def test_uninitialized_origin_true_for_direct_empty_source() -> None:
    """An empty/empty_like source is itself uninitialized."""

    trace = {"e": _fake_op("empty_like")}
    assert _uninitialized_value_origin(trace["e"], trace) is True


def test_uninitialized_origin_does_not_follow_inplace_write_chain() -> None:
    """A chained in-place write is NOT uninitialized -- the B1 bug-mask guard.

    The first ``index_copy_`` into ``new_empty`` writes REAL data into the buffer,
    so the second write's destination parent (``ic1``) holds live values that flow
    to the output. Classifying it as "uninitialized" by chaining back to the
    allocation masked a real perturbation failure (B1 adversarial review,
    ``TwoIndexCopyDim2``). Only a DIRECT allocation op is uninitialized.
    """

    trace = {
        "alloc": _fake_op("new_empty"),
        "ic1": _fake_op("index_copy_", dest_label="alloc"),
        "ic2": _fake_op("index_copy_", dest_label="ic1"),
    }
    # The allocation itself is uninitialized; an in-place write over it is NOT.
    assert _uninitialized_value_origin(trace["alloc"], trace) is True
    assert _uninitialized_value_origin(trace["ic1"], trace) is False
    assert _uninitialized_value_origin(trace["ic2"], trace) is False


def test_uninitialized_origin_false_for_real_data_destination() -> None:
    """A real-data destination is NOT uninitialized -- the bug-mask guard."""

    trace = {
        "data": _fake_op("__mul__"),  # genuine data, not an allocation
        "setitem": _fake_op("__setitem__", dest_label="data"),
    }
    assert _uninitialized_value_origin(trace["setitem"], trace) is False


def test_uninitialized_origin_terminates_on_cycle() -> None:
    """A pathological cyclic destination chain cannot spin forever."""

    op = _fake_op("__setitem__", dest_label="self")
    trace = {"self": op}
    # Must return (False) within the bounded depth, not hang.
    assert _uninitialized_value_origin(op, trace) is False


def test_inplace_write_func_set_covers_setitem_and_index_copy() -> None:
    """The exempted-op set names the idiom's write ops."""

    assert "__setitem__" in INPLACE_DESTINATION_WRITE_FUNCS
    assert "index_copy_" in INPLACE_DESTINATION_WRITE_FUNCS
    assert "indexcopy_" in INPLACE_DESTINATION_WRITE_FUNCS


# ---------------------------------------------------------------------------
# End-to-end: benign uninitialized-dest models validate; real-data stays strict
# ---------------------------------------------------------------------------


def test_empty_like_partial_setitem_validates() -> None:
    """The RoPE/flow idiom (empty_like + strided __setitem__) validates True.

    This is the minimal reproduction of the Wan*/normflows/zuko class. Before the
    fix it failed:replay with error="False"; it must now validate cleanly.
    """

    class EmptyLikeInterleave(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            out[..., 0::2] = torch.cos(x[..., 0::2])
            out[..., 1::2] = torch.sin(x[..., 1::2])
            return out

    model = EmptyLikeInterleave()
    model.eval()
    x = torch.randn(2, 8)
    result, failure = _validate_with_failure(model, x)
    assert result is True
    assert failure is None


def test_index_copy_chain_into_new_empty_validates() -> None:
    """Chained index_copy_ into x.new_empty (NeuralFingerprint class) validates."""

    class IndexCopyChain(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty(x.shape)
            idx_a = torch.arange(0, x.shape[0], 2)
            idx_b = torch.arange(1, x.shape[0], 2)
            out.index_copy_(0, idx_a, x[idx_a] + 1.0)
            out.index_copy_(0, idx_b, x[idx_b] * 2.0)
            return out

    model = IndexCopyChain()
    model.eval()
    x = torch.randn(4, 8)
    result, failure = _validate_with_failure(model, x)
    assert result is True
    assert failure is None


def test_real_data_destination_partial_setitem_stays_validatable() -> None:
    """Partial __setitem__ into a REAL data tensor is genuinely sensitive.

    The exemption must NOT fire here -- the unwritten half carries real data, so
    perturbing the destination DOES change the output and the perturbation check
    passes legitimately (never reaching the exemption). This guards that the
    uninitialized-dest exemption did not widen into masking real dependencies.
    """

    class RealDataDest(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = (x * 2.0).clone()  # real data destination, NOT empty_like
            out[:, 0::2] = x[:, 0::2] + 1.0
            return out

    model = RealDataDest()
    model.eval()
    x = torch.randn(4, 8)
    result, failure = _validate_with_failure(model, x)
    assert result is True
    assert failure is None


def test_real_replay_mismatch_on_empty_like_dest_still_fails() -> None:
    """A GENUINE forward-replay value mismatch is caught even with empty_like dest.

    The exemption only waives sensitivity to UNINITIALIZED memory; it must NOT
    waive the forward-replay value check. Here a __setitem__ op's replay is
    corrupted so the recomputed value diverges -- validation must still FAIL with
    a forward_replay diagnostic. This is the locked-principle guard: the carve-out
    covers ONLY the intended uninitialized-memory case and never masks a real bug.
    """

    class EmptyLikeInterleave(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            out[..., 0::2] = torch.cos(x[..., 0::2])
            out[..., 1::2] = torch.sin(x[..., 1::2])
            return out

    model = EmptyLikeInterleave()
    model.eval()
    x = torch.randn(2, 8)
    gt = model(x).detach().clone()
    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=(x,),
        input_kwargs={},
        layers_to_save="all",
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        save_arg_values=True,
        random_seed=0,
        save_rng_states=True,
    )
    try:
        # Corrupt a __setitem__ op's replay so the recomputed value diverges.
        patched = False
        for layer in trace.layer_list:
            op = layer.ops[1] if hasattr(layer.ops, "_list") else layer
            if getattr(op, "func_name", None) == "__setitem__":
                original = op.func

                def corrupt(*args, _orig=original, **kwargs):
                    result = _orig(*args, **kwargs)
                    args[0].add_(7.0)  # corrupt the written tensor
                    return result

                op.func = corrupt
                patched = True
                break
        assert patched, "expected a __setitem__ op to corrupt"

        # Under cert10 replay the in-place corruption raises inside the replayed
        # func; torchlens treats a raising replay as FAILED and warns. Both
        # routes (value mismatch or raising replay) must end in a failed replay
        # check -- the tripwire concern is unchanged.
        with pytest.warns(UserWarning, match="treating as failed validation"):
            result = validate_saved_outs(trace, [gt], validate_metadata=False)
        # ValidationReplayStatus API (da60a76e): tripwire still fires on a real
        # value mismatch.
        assert result.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == CHECK_REPLAY
    finally:
        trace.cleanup()


def test_chained_index_copy_dropped_dest_dependency_still_fails() -> None:
    """B1 BUG-MASK GUARD: a dropped destination dependency on a CHAINED in-place
    write must still fail the perturbation check.

    This is the exact masking case from the B1 adversarial review. ``out`` is
    allocated by ``empty_like`` and written by two ``index_copy_`` calls into
    DISJOINT-but-live columns. The second write's destination parent is the FIRST
    write, which already holds real data in columns 1,3 that flow to the output.

    A wrong replay that REBUILDS from the saved full output (ignoring the
    destination parent) reproduces the saved value and keeps the source-value
    sensitive, so only the PERTURBATION check on the destination parent can catch
    the dropped dependency. The pre-fix exemption labeled the chained destination
    "uninitialized" (following the write chain back to ``empty_like``) and waived
    that failure -- masking a real capture bug. The fix must make this FAIL.
    """

    class TwoIndexCopyDim2(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            idx1 = torch.tensor([1, 3], device=x.device)
            idx0 = torch.tensor([0, 2], device=x.device)
            out.index_copy_(2, idx1, x[:, :, idx1] * 2)
            out.index_copy_(2, idx0, x[:, :, idx0] + 1)
            return out

    model = TwoIndexCopyDim2()
    model.eval()
    x = torch.randn(2, 3, 4)
    gt = model(x).detach().clone()
    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=(x,),
        input_kwargs={},
        layers_to_save="all",
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        save_arg_values=True,
        random_seed=0,
        save_rng_states=True,
    )
    try:
        # Locate the SECOND index_copy_ (destination parent is the first write).
        second = None
        for layer in trace.layer_list:
            op = layer.ops[1] if hasattr(layer.ops, "_list") else layer
            if op.func_name in ("index_copy_", "indexcopy_") and str(
                op.parent_arg_positions["args"].get(0, "")
            ).startswith("indexcopy"):
                second = op
        assert second is not None, "expected a chained index_copy_ op"

        # Corrupt its replay so the destination-parent dependency is DROPPED:
        # rebuild from the saved full output, then re-apply only the source slice.
        # Normal replay still matches and the source value stays sensitive, but
        # the destination parent no longer influences the result.
        saved = second.out.detach().clone()

        def fake_index_copy(dest, dim, index, source, _saved=saved):
            dest.copy_(_saved)  # BUG: ignore the destination parent's value
            dest.index_copy_(dim, index, source)  # source still honored
            return dest

        fake_index_copy.__name__ = getattr(second.func, "__name__", "index_copy_")
        second.func = fake_index_copy

        result = validate_saved_outs(trace, [gt], validate_metadata=False)
        assert result.state == "failed", "dropped chained-dest dependency must FAIL validation"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == CHECK_PERTURBATION
        assert "indexcopy" in (failure.op_label or "")
    finally:
        trace.cleanup()
