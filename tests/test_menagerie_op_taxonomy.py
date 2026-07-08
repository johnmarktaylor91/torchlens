"""Tests for the menagerie operation taxonomy artifact."""

from __future__ import annotations

from menagerie.op_taxonomy import (
    OP_CATEGORIES,
    classify_activation_type,
    classify_norm_type,
    classify_op,
)


def test_classify_op_is_total_and_deterministic() -> None:
    """Unknown and repeated inputs always map to a closed category."""

    cases = [
        ("conv2d", None),
        ("torch.nn.functional.layer_norm", None),
        ("not_a_real_op", "CustomBlock"),
        (None, None),
    ]
    for func_name, module_class_name in cases:
        first = classify_op(func_name, module_class_name)
        repeated = [classify_op(func_name, module_class_name) for _ in range(10)]
        assert first in OP_CATEGORIES
        assert repeated == [first] * 10


def test_classify_op_load_bearing_category_mappings() -> None:
    """Common Torch operations land in the schema-required categories."""

    expected = {
        ("conv2d", None): "conv",
        ("conv_transpose2d", None): "conv",
        ("linear", None): "linear",
        ("matmul", None): "linear",
        ("scaled_dot_product_attention", None): "attention",
        ("linear", "MultiheadAttention"): "attention",
        ("batch_norm", None): "norm",
        ("add", None): "elementwise",
        ("sum", None): "reduction",
        ("reshape", None): "reshape",
        ("embedding", None): "embedding",
        ("max_pool2d", None): "pooling",
        ("relu", None): "activation",
        ("does_not_exist", None): "other",
    }
    for inputs, category in expected.items():
        assert classify_op(*inputs) == category


def test_norm_and_activation_subtypes() -> None:
    """Subtype helpers return concrete subtype names or none."""

    assert classify_norm_type("batch_norm") == "batch_norm"
    assert classify_norm_type("linear", "LayerNorm") == "layer_norm"
    assert classify_norm_type("matmul") == "none"
    assert classify_activation_type("relu") == "relu"
    assert classify_activation_type("leaky_relu") == "leaky_relu"
    assert classify_activation_type("linear", "SiLU") == "silu"
    assert classify_activation_type("conv2d") == "none"
