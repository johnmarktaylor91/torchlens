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
perturbed parent's value traces to an uninitialized-memory source through the
destination slot).

LOCKED tripwire invariant under test: the exemption is NARROW. A REAL data
tensor feeding an in-place-write destination is NOT exempted -- a genuine dropped
dependency must still fail. ``test_real_data_destination_*`` guards that the
exemption never widens into a bug mask.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchlens.user_funcs import (
    _run_model_and_save_specified_outs,
    _validate_forward_pass_torch,
)
from torchlens.validation.core import validate_saved_outs
from torchlens.validation.diagnostics import CHECK_REPLAY, get_validation_failure
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


def test_uninitialized_origin_follows_inplace_write_chain() -> None:
    """index_copy_ into new_empty traces to an uninitialized origin."""

    trace = {
        "alloc": _fake_op("new_empty"),
        "ic1": _fake_op("index_copy_", dest_label="alloc"),
        "ic2": _fake_op("index_copy_", dest_label="ic1"),
    }
    assert _uninitialized_value_origin(trace["ic2"], trace) is True


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

        result = validate_saved_outs(trace, [gt], validate_metadata=False)
        assert result is False  # tripwire still fires on a real value mismatch
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == CHECK_REPLAY
    finally:
        trace.cleanup()
