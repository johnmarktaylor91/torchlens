"""Phase 3a capture-unification regression tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext


class PredicateToy(nn.Module):
    """Small model with mixed operation types for selective-save checks."""

    def __init__(self) -> None:
        """Initialize the test module."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with a relu and non-relu op."""

        y = self.fc(x)
        y = torch.relu(y)
        return y + 1


class RecurrentToy(nn.Module):
    """Model that calls the same module multiple times."""

    def __init__(self, passes: int = 3) -> None:
        """Initialize the recurrent test module."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.passes = passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run repeated calls through the same module."""

        for _ in range(self.passes):
            x = torch.relu(self.attn(x))
        return x


class FinalLabelDriftToy(nn.Module):
    """Model whose raw and final labels differ after the input source."""

    def __init__(self) -> None:
        """Initialize a linear layer used before repeated relu ops."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ops whose final labels are offset from raw labels."""

        x = self.fc(x)
        x = torch.relu(x)
        return torch.relu(x)


class IntegerSelectorToy(nn.Module):
    """Model where raw ordinals and per-type indices do not match layer-list indices."""

    def __init__(self) -> None:
        """Initialize three linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a chain with mixed repeated operation types."""

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _pseudo_random_subset(ctx: RecordContext) -> bool:
    """Select a deterministic subset of operation contexts."""

    label = ctx.raw_label or ctx.label
    return ctx.kind == "op" and sum(ord(char) for char in label) % 3 == 0


def _assert_saved_ops_match_full_trace(
    model: nn.Module,
    x: torch.Tensor,
    predicate: Callable[[RecordContext], bool],
) -> None:
    """Assert selective saved payloads are byte-identical to full trace payloads."""

    full = tl.trace(model, x.clone(), layers_to_save="all", random_seed=123)
    selective = tl.trace(model, x.clone(), save=predicate, random_seed=123)
    saved_ops = [
        op
        for op in selective.layer_list
        if op.has_saved_activation and op.layer_type not in {"input", "output"}
    ]
    assert saved_ops
    for saved_op in saved_ops:
        expected = full.layer_dict_all_keys[saved_op._label_raw].out
        assert torch.equal(saved_op.out, expected), saved_op.label
    for unsaved_op in selective.layer_list:
        if unsaved_op.layer_type in {"input", "output"} or unsaved_op.has_saved_activation:
            continue
        with pytest.raises(ValueError, match="not saved"):
            _ = unsaved_op.out


def test_trace_save_func_selector_keeps_only_matching_payloads() -> None:
    """trace(save=tl.func(...)) saves only matching op payloads in one pass."""

    model = PredicateToy()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, save=tl.func("relu"), random_seed=11)
    saved_ops = [
        op
        for op in log.layer_list
        if op.has_saved_activation and op.layer_type not in {"input", "output"}
    ]
    assert saved_ops
    assert {op.func_name for op in saved_ops} == {"relu"}
    relu_op = saved_ops[0]
    assert torch.equal(log[relu_op.label].out, relu_op.out)
    unsaved = next(op for op in log.layer_list if op.layer_type == "add")
    with pytest.raises(ValueError, match="not saved"):
        _ = unsaved.out


def test_selective_save_keeps_unsaved_non_orphan_op_metadata() -> None:
    """Selective save keeps unsaved non-orphan ops addressable with metadata."""

    model = PredicateToy()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, save=tl.func("relu"), random_seed=13)

    unsaved = next(
        op for op in log.layer_list if op.layer_type == "add" and not op.has_saved_activation
    )
    assert unsaved in log.layer_list
    assert log.layer_dict_main_keys[unsaved.label] is unsaved
    assert log.layer_dict_all_keys[unsaved.label] is unsaved
    assert log[unsaved.label] is unsaved
    assert unsaved.label in log.op_labels
    assert unsaved.layer_label in log.layer_labels

    assert unsaved.label
    assert unsaved.parents
    assert unsaved.children
    assert unsaved.shape == (2, 4)
    assert unsaved.dtype == torch.float32
    with pytest.raises(ValueError, match="not saved"):
        _ = unsaved.out


def test_selective_save_oracle_matches_full_trace_for_recurrent_passes() -> None:
    """Selective-save payloads match full trace payloads, including recurrent ops."""

    x = torch.randn(2, 4)
    _assert_saved_ops_match_full_trace(PredicateToy(), x, _pseudo_random_subset)
    _assert_saved_ops_match_full_trace(RecurrentToy(passes=4), x, _pseudo_random_subset)


def test_layers_to_save_retains_output_parent_when_selector_misses_parent() -> None:
    """Absorbed selective layers_to_save keeps output payloads available."""

    model = PredicateToy()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save=["relu"], random_seed=17)

    output = log["output_1"]
    assert output.has_saved_activation is True
    assert isinstance(output.out, torch.Tensor)
    assert any(
        op.layer_type == "add" and op.has_saved_activation
        for op in log.layer_list
        if op.layer_type not in {"input", "output"}
    )


@pytest.mark.parametrize(
    ("selector", "expected_relu_labels"),
    [
        ("relu", {"relu_1_2", "relu_2_3"}),
        ("relu_1", {"relu_1_2", "relu_2_3"}),
        ("relu_1_2", {"relu_1_2", "relu_2_3"}),
        ("relu_1_2:1", {"relu_1_2", "relu_2_3"}),
        ("relu_1_3", {"relu_1_2", "relu_2_3"}),
        ("relu_1_3_raw", {"relu_1_2", "relu_2_3"}),
        ("relu_2", {"relu_2_3"}),
        ("relu_2_3", {"relu_2_3"}),
        ("relu_2_3:1", {"relu_2_3"}),
        ("relu_2_4", {"relu_2_3"}),
        ("relu_2_4_raw", {"relu_2_3"}),
    ],
)
def test_layers_to_save_matches_legacy_label_spellings(
    selector: str,
    expected_relu_labels: set[str],
) -> None:
    """Absorbed label matching accepts legacy final, short, raw, and pass forms."""

    model = FinalLabelDriftToy()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save=[selector], random_seed=19)

    saved_relu_labels = {
        op.layer_label
        for op in log.layer_list
        if op.layer_type == "relu" and op.has_saved_activation
    }
    assert saved_relu_labels == expected_relu_labels


def test_integer_layers_to_save_uses_single_legacy_layer_index() -> None:
    """Integer selectors resolve through the legacy layer-list index only."""

    model = IntegerSelectorToy()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save=[2], random_seed=23)

    saved_compute_labels = [
        op.layer_label
        for op in log.layer_list
        if op.has_saved_activation and op.layer_type not in {"input", "output"}
    ]
    assert saved_compute_labels == ["relu_1_2", "linear_3_5"]


def test_layers_to_save_unqualified_module_label_saves_all_passes() -> None:
    """Unqualified repeated module labels save all passes under §6a."""

    model = RecurrentToy(passes=3)
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save=["attn"], random_seed=7)
    linear_layer = next(layer for layer in log.layers if layer.layer_type == "linear")
    assert linear_layer.num_passes == 3
    assert [op.has_saved_activation for op in linear_layer.ops._list] == [True, True, True]
    assert linear_layer.ops[0].out is not None
    with pytest.raises(ValueError, match="has 3 ops"):
        _ = linear_layer.out


def test_layers_to_save_pass_qualified_module_label_saves_one_pass() -> None:
    """Pass-qualified repeated module labels save exactly the requested 1-based pass."""

    model = RecurrentToy(passes=3)
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save=["attn:2"], random_seed=7)
    linear_layer = next(layer for layer in log.layers if layer.layer_type == "linear")
    assert [op.has_saved_activation for op in linear_layer.ops._list] == [False, True, False]
    assert linear_layer.ops[1].out is not None
    with pytest.raises(ValueError, match="has 3 ops"):
        _ = linear_layer.out
