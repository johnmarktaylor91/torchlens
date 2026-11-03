"""Round-28/31 validation identity-witness pins (FN-1..6 closure).

The round-31 false-negative hunt proved that a dropped parent edge whose saved
value is "trivial" (bool mask, all-zero, all-abs-one) was INVISIBLE to every
validation layer, and that a wrong-parent swap between value-identical
producers passed by construction. The closure is the capture-time identity
witness: ``dropped_edge_tensor_args`` records, per arg slot, a live traced
producer that is absent from the recorded parent edges, and
``_check_unattributed_arg_slots`` fails on it regardless of the slot's value.

These tests inject the exact capture-bug shapes from the round-31 attack
scripts (edge drop / wrong-parent swap inside
``_build_graph_relationship_fields``) and pin that the untouched public
``tl.validate`` now FAILS, while unmodified captures still validate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch._ops_exhaustive
import torchlens.backends.torch._ops_predicate_events
import torchlens.backends.torch._ops_shared_fields


class _MaskedFillBoolMask(nn.Module):
    """Consume a bool mask (trivial-valued) plus a second mask consumer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask the input and keep the mask reachable.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Masked values plus the mask count.
        """

        mask = x > 0
        y = x.masked_fill(mask, -1.0)
        z = mask.float().sum()
        return y + z


class _MulByAllOnes(nn.Module):
    """Multiply by an all-ones (trivial-valued) derived tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply through the all-ones gate.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Gated values plus the gate sum.
        """

        w = x * 0 + 1
        y = x.relu() * w
        z = w.sum()
        return y + z


class _AddAllZeros(nn.Module):
    """Add an all-zeros (trivial-valued) derived tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the zero gate and keep it reachable.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum with the zero gate plus its scaled sum.
        """

        w = x * 0
        y = x.relu() + w
        z = w.sum() * 2.0
        return y + z * 0.5


class _TwinRelu(nn.Module):
    """Two value-identical producers; only one is the true mul parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the FIRST relu; keep both relus alive.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Combined output keeping both producers reachable.
        """

        a = torch.relu(x)
        b = torch.relu(x + 0)
        y = a * 2.0
        return y + a.sum() + b.sum()


@pytest.fixture()
def drop_edge_injector(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, Callable[..., bool]], dict]:
    """Return an armer that drops targeted parent edges at CAPTURE time.

    Wrapping ``_build_graph_relationship_fields`` to filter a parent before
    the original runs makes the whole capture+postprocess pipeline behave
    exactly as if parent detection had missed that tensor -- an honest
    simulation of a capture attribution bug, judged by the untouched public
    ``tl.validate``.
    """

    original = torchlens.backends.torch._ops_predicate_events._build_graph_relationship_fields
    state: dict[str, Any] = {"active": False, "func_name": None, "pred": None, "dropped": 0}

    def patched(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    ):  # type: ignore[no-untyped-def]
        """Filter targeted parents, then delegate to the real builder."""

        if state["active"] and fields_dict.get("func_name") == state["func_name"]:
            kept_labels, kept_entries = [], []
            for label, entry in zip(parent_layer_labels, parent_layer_entries):
                if state["pred"](label, entry):
                    state["dropped"] += 1
                    continue
                kept_labels.append(label)
                kept_entries.append(entry)
            parent_layer_labels, parent_layer_entries = kept_labels, kept_entries
        return original(
            self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
        )

    # The home module and every module that bound the function by import.
    for module in (
        torchlens.backends.torch._ops_predicate_events,
        torchlens.backends.torch._ops_exhaustive,
        torchlens.backends.torch._ops_shared_fields,
    ):
        monkeypatch.setattr(module, "_build_graph_relationship_fields", patched)

    def arm(func_name: str, pred: Callable[..., bool]) -> dict:
        """Activate the injector for one wrapped function name."""

        state.update(active=True, func_name=func_name, pred=pred, dropped=0)
        return state

    return arm


def _parent_out_is_bool(_label: str, entry: Any) -> bool:
    """Return whether the candidate parent's out is a bool tensor."""

    out = getattr(entry, "out", None)
    return isinstance(out, torch.Tensor) and out.dtype == torch.bool


def _parent_label_prefix(prefix: str) -> Callable[..., bool]:
    """Return a predicate matching parent labels by prefix."""

    def pred(label: str, _entry: Any) -> bool:
        """Match the injected parent label."""

        return label.startswith(prefix)

    return pred


@pytest.mark.parametrize(
    "model,func_name,pred_factory",
    [
        (_MaskedFillBoolMask, "masked_fill", lambda: _parent_out_is_bool),
        (_MulByAllOnes, "__mul__", lambda: _parent_label_prefix("add_")),
        (_AddAllZeros, "__add__", lambda: _parent_label_prefix("mul_")),
    ],
    ids=["bool-mask", "all-ones", "all-zeros"],
)
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments:UserWarning")
def test_fn_trivial_value_dropped_edge_now_fails_validation(
    drop_edge_injector: Callable[[str, Callable[..., bool]], dict],
    model: type[nn.Module],
    func_name: str,
    pred_factory: Callable[[], Callable[..., bool]],
) -> None:
    """A dropped trivial-valued parent edge must fail forward validation."""

    torch.manual_seed(0)
    x = torch.randn(4)
    state = drop_edge_injector(func_name, pred_factory())

    result = tl.validate(model().eval(), x, scope="forward")

    assert state["dropped"] > 0, "injection did not drop any edge -- inconclusive"
    assert not bool(result)


@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments:UserWarning")
def test_fn_capture_witness_records_dropped_edge_positions(
    drop_edge_injector: Callable[[str, Callable[..., bool]], dict],
) -> None:
    """The capture-side identity witness marks the dropped slot itself."""

    torch.manual_seed(0)
    x = torch.randn(4)
    state = drop_edge_injector("masked_fill", _parent_out_is_bool)
    trace = tl.trace(_MaskedFillBoolMask().eval(), x)
    masked_op = next(op for op in trace.ops if op.func_name == "masked_fill")

    assert state["dropped"] > 0
    assert "arg1" in tuple(masked_op.call.dropped_edge_tensor_args)
    assert "arg1" in tuple(masked_op.call.unattributed_tensor_args)


@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments:UserWarning")
def test_fn6_wrong_parent_between_value_identical_producers_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consistent wrong-parent swap to a value-identical producer fails."""

    original = torchlens.backends.torch._ops_predicate_events._build_graph_relationship_fields
    state = {"hits": 0}

    def patched(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    ):  # type: ignore[no-untyped-def]
        """Swap the recorded mul parent from relu_1 to relu_2 post-detection."""

        result = original(
            self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
        )
        if fields_dict.get("func_name") != "__mul__":
            return result
        live = self.capture_events.live_index.by_raw_label
        replacements = [label for label in live if label.startswith("relu_2")]
        if not replacements:
            return result
        parents = fields_dict.get("parents") or []
        for index, label in enumerate(list(parents)):
            if label.startswith("relu_1"):
                parents[index] = replacements[0]
                state["hits"] += 1
        positions = fields_dict.get("parent_arg_positions") or {}
        for domain in ("args", "kwargs"):
            for key, label in list((positions.get(domain) or {}).items()):
                if label.startswith("relu_1"):
                    positions[domain][key] = replacements[0]
        return result

    # The home module and every module that bound the function by import.
    for module in (
        torchlens.backends.torch._ops_predicate_events,
        torchlens.backends.torch._ops_exhaustive,
        torchlens.backends.torch._ops_shared_fields,
    ):
        monkeypatch.setattr(module, "_build_graph_relationship_fields", patched)
    torch.manual_seed(0)
    x = torch.randn(4)

    result = tl.validate(_TwinRelu().eval(), x, scope="forward")

    assert state["hits"] > 0, "injection never swapped a parent -- inconclusive"
    assert not bool(result)


@pytest.mark.parametrize(
    "model",
    [_MaskedFillBoolMask, _MulByAllOnes, _AddAllZeros, _TwinRelu],
    ids=["bool-mask", "all-ones", "all-zeros", "twin-relu"],
)
def test_fn_unmodified_captures_still_validate(model: type[nn.Module]) -> None:
    """No-false-positive control: honest captures of the FN models pass."""

    torch.manual_seed(0)
    x = torch.randn(4)

    assert tl.validate(model().eval(), x, scope="forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
