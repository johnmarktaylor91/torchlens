"""Tests for the func_config feature.

Verifies that func_config is correctly extracted for various operation types,
is empty for source/output nodes, survives the postprocessing pipeline, and
is accessible on both LayerPassLog and LayerLog.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.capture.salient_args import extract_salient_args, _build_arg_name_map


# ---------------------------------------------------------------------------
# Unit tests for extract_salient_args
# ---------------------------------------------------------------------------


class TestExtractSalientArgs:
    """Unit tests for the extractor registry."""

    def test_conv2d_basic(self):
        weight = torch.randn(16, 3, 3, 3)
        result = extract_salient_args(
            "conv2d", "conv2d", (torch.randn(1, 3, 8, 8), weight), {}, [(16, 3, 3, 3)]
        )
        assert result["out_channels"] == 16
        assert result["in_channels"] == 3
        assert result["kernel_size"] == (3, 3)
        assert "stride" not in result  # default stride=1 suppressed
        assert "dilation" not in result  # default dilation=1 suppressed

    def test_conv2d_with_stride(self):
        result = extract_salient_args(
            "conv2d",
            "conv2d",
            (torch.randn(1, 3, 8, 8), torch.randn(16, 3, 3, 3)),
            {"stride": (2, 2)},
            [(16, 3, 3, 3)],
        )
        assert result["stride"] == (2, 2)

    def test_conv2d_with_groups(self):
        result = extract_salient_args(
            "conv2d",
            "conv2d",
            (torch.randn(1, 4, 8, 8), torch.randn(4, 1, 3, 3)),
            {"groups": 4},
            [(4, 1, 3, 3)],
        )
        assert result["groups"] == 4

    def test_linear(self):
        result = extract_salient_args(
            "linear", "linear", (torch.randn(1, 128), torch.randn(64, 128)), {}, [(64, 128)]
        )
        assert result["out_features"] == 64
        assert result["in_features"] == 128

    def test_dropout(self):
        result = extract_salient_args("dropout", "dropout", (torch.randn(1, 10),), {"p": 0.3}, [])
        assert result["p"] == 0.3

    def test_batch_norm(self):
        result = extract_salient_args(
            "batchnorm",
            "batch_norm",
            (torch.randn(1, 16, 4, 4),),
            {"eps": 1e-05, "momentum": 0.1},
            [],
        )
        assert result["eps"] == 1e-05
        assert result["momentum"] == 0.1

    def test_layer_norm(self):
        result = extract_salient_args(
            "layernorm", "layer_norm", (torch.randn(1, 10),), {"normalized_shape": (10,)}, []
        )
        assert result["normalized_shape"] == (10,)

    def test_max_pool(self):
        result = extract_salient_args(
            "maxpool2d",
            "max_pool2d",
            (torch.randn(1, 3, 8, 8),),
            {"kernel_size": 2, "stride": 2},
            [],
        )
        assert result["kernel_size"] == 2
        assert result["stride"] == 2

    def test_adaptive_avg_pool(self):
        result = extract_salient_args(
            "adaptiveavgpool2d",
            "adaptive_avg_pool2d",
            (torch.randn(1, 3, 8, 8),),
            {"output_size": (1, 1)},
            [],
        )
        assert result["output_size"] == (1, 1)

    def test_softmax(self):
        result = extract_salient_args("softmax", "softmax", (torch.randn(1, 10),), {"dim": 1}, [])
        assert result["dim"] == 1

    def test_cat(self):
        t = torch.randn(1, 3)
        result = extract_salient_args("cat", "cat", ([t, t],), {"dim": 1}, [])
        assert result["dim"] == 1

    def test_reduction_mean(self):
        result = extract_salient_args(
            "mean", "mean", (torch.randn(2, 3, 4),), {"dim": (1, 2), "keepdim": True}, []
        )
        assert result["dim"] == (1, 2)
        assert result["keepdim"] is True

    def test_clamp(self):
        result = extract_salient_args(
            "clamp", "clamp", (torch.randn(3),), {"min": 0.0, "max": 1.0}, []
        )
        assert result["min"] == 0.0
        assert result["max"] == 1.0

    def test_transpose(self):
        result = extract_salient_args("transpose", "transpose", (torch.randn(2, 3), 0, 1), {}, [])
        assert result["dim0"] == 0
        assert result["dim1"] == 1

    def test_leaky_relu(self):
        result = extract_salient_args(
            "leakyrelu", "leaky_relu", (torch.randn(3),), {"negative_slope": 0.2}, []
        )
        assert result["negative_slope"] == 0.2

    def test_embedding(self):
        result = extract_salient_args(
            "embedding",
            "embedding",
            (torch.tensor([0, 1, 2]), torch.randn(100, 64)),
            {},
            [(100, 64)],
        )
        assert result["num_embeddings"] == 100
        assert result["embedding_dim"] == 64

    def test_interpolate(self):
        result = extract_salient_args(
            "interpolate",
            "interpolate",
            (torch.randn(1, 1, 4, 4),),
            {"scale_factor": 2.0, "mode": "bilinear"},
            [],
        )
        assert result["scale_factor"] == 2.0
        assert result["mode"] == "bilinear"

    def test_unregistered_op_returns_empty(self):
        result = extract_salient_args("relu", "relu", (torch.randn(3),), {}, [])
        assert result == {}

    def test_never_contains_tensors(self):
        """Values must be simple Python types, never tensors."""
        result = extract_salient_args("softmax", "softmax", (torch.randn(1, 10),), {"dim": 1}, [])
        for v in result.values():
            assert not isinstance(v, torch.Tensor)

    def test_sdpa(self):
        result = extract_salient_args(
            "scaleddotproductattention",
            "scaled_dot_product_attention",
            (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),
            {"dropout_p": 0.1, "is_causal": True},
            [],
        )
        assert result["dropout_p"] == 0.1
        assert result["is_causal"] is True


class TestBuildArgNameMap:
    """Unit tests for arg name mapping helper."""

    def test_basic_mapping(self):
        from torchlens import _state

        _state._func_argnames["softmax"] = ("input", "dim", "dtype")
        result = _build_arg_name_map("softmax", (torch.randn(3), 1), {})
        assert result["dim"] == 1

    def test_kwargs_take_precedence(self):
        from torchlens import _state

        _state._func_argnames["softmax"] = ("input", "dim", "dtype")
        result = _build_arg_name_map("softmax", (torch.randn(3), 0), {"dim": 1})
        assert result["dim"] == 1


# ---------------------------------------------------------------------------
# Integration tests: end-to-end with log_forward_pass
# ---------------------------------------------------------------------------


class TestFuncConfigIntegration:
    """Integration tests verifying func_config through the full pipeline."""

    def test_conv_bn_linear_model(self):
        """Conv2d, BatchNorm, Linear all populate func_config correctly."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = torch.relu(x)
                x = x.mean(dim=[2, 3])
                x = self.fc(x)
                return x

        model = Model()
        log = tl.log_forward_pass(model, torch.randn(1, 3, 8, 8))

        # Find layers by type
        conv_layer = next(ly for ly in log.layers if ly.layer_type == "conv2d")
        assert conv_layer.func_config["out_channels"] == 16
        assert conv_layer.func_config["in_channels"] == 3
        assert conv_layer.func_config["stride"] == (2, 2)
        assert conv_layer.func_config["padding"] == (1, 1)

        bn_layer = next(ly for ly in log.layers if ly.layer_type == "batchnorm")
        assert "eps" in bn_layer.func_config

        linear_layer = next(ly for ly in log.layers if ly.layer_type == "linear")
        assert linear_layer.func_config["out_features"] == 10
        assert linear_layer.func_config["in_features"] == 16

    def test_source_tensors_have_empty_func_config(self):
        """Input and buffer layers should have func_config == {}."""
        model = nn.BatchNorm2d(3)
        log = tl.log_forward_pass(model, torch.randn(1, 3, 4, 4))

        for layer in log.layers:
            if layer.is_input_layer or layer.is_buffer_layer:
                assert layer.func_config == {}, (
                    f"Source layer {layer.layer_label} has non-empty func_config: "
                    f"{layer.func_config}"
                )

    def test_output_nodes_have_empty_func_config(self):
        """Synthetic output nodes should have func_config == {}."""
        model = nn.Linear(10, 5)
        log = tl.log_forward_pass(model, torch.randn(1, 10))

        for layer in log.layers:
            if layer.is_output_layer:
                assert layer.func_config == {}, (
                    f"Output layer {layer.layer_label} has non-empty func_config"
                )

    def test_func_config_on_layer_pass_log(self):
        """func_config should be accessible on LayerPassLog (per-pass) objects."""
        model = nn.Linear(10, 5)
        log = tl.log_forward_pass(model, torch.randn(1, 10))

        linear_layer = next(ly for ly in log.layers if ly.layer_type == "linear")
        # Access via pass
        pass_log = linear_layer.passes[1]
        assert pass_log.func_config["out_features"] == 5

    def test_func_config_in_str_output(self):
        """func_config should appear in the string representation when non-empty."""
        model = nn.Linear(10, 5)
        log = tl.log_forward_pass(model, torch.randn(1, 10))

        linear_layer = next(ly for ly in log.layers if ly.layer_type == "linear")
        s = str(linear_layer)
        assert "Config:" in s
        assert "out_features=5" in s

    def test_func_config_not_in_str_when_empty(self):
        """Layers with no func_config should not show the config line."""
        model = nn.ReLU()
        log = tl.log_forward_pass(model, torch.randn(1, 10))

        relu_layer = next(ly for ly in log.layers if ly.layer_type == "relu")
        s = str(relu_layer)
        assert "Config:" not in s

    def test_dropout_func_config(self):
        """Dropout should capture the p parameter."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(0.3)

            def forward(self, x):
                return self.drop(x)

        model = Model()
        log = tl.log_forward_pass(model, torch.randn(1, 10))

        dropout_layer = next(ly for ly in log.layers if ly.layer_type == "dropout")
        assert dropout_layer.func_config["p"] == 0.3

    def test_reduction_func_config(self):
        """Reduction ops should capture dim and keepdim."""

        class Model(nn.Module):
            def forward(self, x):
                return x.sum(dim=1, keepdim=True)

        log = tl.log_forward_pass(Model(), torch.randn(2, 3, 4))
        sum_layer = next(ly for ly in log.layers if ly.layer_type == "sum")
        assert sum_layer.func_config["dim"] == 1
        assert sum_layer.func_config["keepdim"] is True

    def test_pooling_func_config(self):
        """Pooling ops should capture kernel_size and stride."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                return self.pool(x)

        log = tl.log_forward_pass(Model(), torch.randn(1, 3, 8, 8))
        pool_layer = next(ly for ly in log.layers if "maxpool" in ly.layer_type)
        assert pool_layer.func_config["kernel_size"] == 2
        assert pool_layer.func_config["stride"] == 2

    def test_save_new_activations_preserves_func_config(self):
        """func_config should survive save_new_activations (fast path)."""
        model = nn.Linear(10, 5)
        x1 = torch.randn(1, 10)
        log = tl.log_forward_pass(model, x1, layers_to_save="all")

        # Run with new input via ModelLog method
        x2 = torch.randn(1, 10)
        log.save_new_activations(model, x2)

        linear_layer = next(ly for ly in log.layers if ly.layer_type == "linear")
        assert linear_layer.func_config["out_features"] == 5
        assert linear_layer.func_config["in_features"] == 10

    def test_conv_default_stride_not_shown(self):
        """Conv with default stride/padding/dilation should not include them."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

            def forward(self, x):
                return self.conv(x)

        log = tl.log_forward_pass(Model(), torch.randn(1, 3, 8, 8))
        conv_layer = next(ly for ly in log.layers if ly.layer_type == "conv2d")
        assert "stride" not in conv_layer.func_config
        assert "padding" not in conv_layer.func_config
        assert "dilation" not in conv_layer.func_config
        assert conv_layer.func_config["kernel_size"] == (3, 3)

    def test_all_layers_have_func_config_attribute(self):
        """Every layer in the log should have a func_config attribute (dict)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)
                self.fc = nn.Linear(16 * 6 * 6, 10)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.bn(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        log = tl.log_forward_pass(Model(), torch.randn(1, 3, 8, 8))
        for layer in log.layers:
            assert hasattr(layer, "func_config"), f"Missing func_config on {layer.layer_label}"
            assert isinstance(layer.func_config, dict), (
                f"func_config should be dict on {layer.layer_label}, got {type(layer.func_config)}"
            )
