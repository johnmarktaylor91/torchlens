"""Regression coverage for tensor replacement during active TorchLens captures.

These tests cover the fleet bug where user replacement tensors lacked
TorchLens raw-label metadata and broke the next module boundary. The audited access sites
fall into two groups: output/module-exit paths now repair missing labels, while
postprocess, visualization, and backward-hook consumers only run after forward
instrumentation has assigned raw labels. In-place ops keep tensor attributes on
the same object; existing in-place coverage remains the relevant guard there.
"""

from __future__ import annotations

import torch

import torchlens as tl


class _HookedMlp(torch.nn.Module):
    """Small module graph with named replacement sites."""

    def __init__(self) -> None:
        """Create a deterministic MLP fixture."""

        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return self.fc3(x)


class _NestedFreshTensorModel(torch.nn.Module):
    """Nested module fixture for raw-hook fresh tensor replacements."""

    def __init__(self) -> None:
        """Initialize nested modules."""

        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run nested modules and return the model output."""

        return self.head(self.block(x))


class _DiscardedInplaceReturnModel(torch.nn.Module):
    """Use a mutated tensor after discarding an in-place operation's return."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discard ``relu_``'s return and consume its mutated input storage."""

        y = x * 2
        y.relu_()
        return y + 1


def _zero_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return a zero replacement for a TorchLens live hook."""

    del hook
    return torch.zeros_like(out)


def _quantize_dequantize_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return a tiny fake-quantized replacement tensor."""

    del hook
    scale = out.detach().abs().max().clamp_min(torch.tensor(1e-6, device=out.device)) / 127
    return torch.round(out / scale).clamp(-128, 127) * scale


def _replacement_layers(log: tl.Trace) -> list[tl.Op]:
    """Return layer-pass logs marked as replacement sites.

    Parameters
    ----------
    log:
        Captured TorchLens trace.

    Returns
    -------
    list[tl.Op]
        Replacement-marked layer passes.
    """

    return [layer for layer in log.layer_list if getattr(layer, "intervention_replaced", False)]


def test_intervention_api_replaces_op_output_preserves_graph() -> None:
    """Official live hooks preserve graph labels and mark the replaced op."""

    model = _HookedMlp()
    log = tl.trace(
        model,
        torch.randn(3, 4),
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )

    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")
    fc1_layer = next(layer for layer in log.layer_list if layer.layer_label.startswith("linear_1"))

    assert relu_layer.intervention_replaced is True
    assert relu_layer.parents == [fc1_layer.layer_label]
    assert len(relu_layer.interventions) == 1
    assert torch.count_nonzero(relu_layer.out) == 0


def test_output_hook_replaces_discarded_inplace_return_storage() -> None:
    """An output hook on ``relu_`` updates storage even when its return is discarded."""

    log = tl.trace(
        _DiscardedInplaceReturnModel(),
        torch.tensor([-2.0, 3.0]),
        intervention_ready=True,
        hooks={tl.func("relu_"): tl.zero_ablate()},
    )
    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu_")

    assert torch.equal(log[log.output_layers[0]].out, torch.ones(2))
    assert torch.equal(relu_layer.out, torch.zeros(2))
    assert relu_layer.intervention_replaced is True
    assert relu_layer.interventions[-1].replaced is True


def test_raw_forward_hook_replaces_module_output_does_not_crash() -> None:
    """Raw ``register_forward_hook`` replacement tensors are re-instrumented."""

    model = _HookedMlp()

    def raw_hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fresh tensor from a raw PyTorch forward hook."""

        del module, args
        return output * 0.5

    handle = model.relu.register_forward_hook(raw_hook)
    try:
        log = tl.trace(model, torch.randn(3, 4))
    finally:
        handle.remove()

    mul_layer = next(layer for layer in log.layer_list if layer.func_name == "__mul__")
    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert mul_layer.intervention_replaced is True
    assert mul_layer.parents == [relu_layer.layer_label]
    assert log.layer_dict_main_keys[mul_layer.label] is mul_layer


def test_chain_of_interventions_preserves_graph() -> None:
    """Multiple replacements in one live capture keep consistent parent chains."""

    model = _HookedMlp()
    log = tl.trace(
        model,
        torch.randn(3, 4),
        intervention_ready=True,
        hooks={
            tl.func("relu"): _zero_hook,
            tl.func("sigmoid"): _zero_hook,
        },
    )

    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")
    sigmoid_layer = next(layer for layer in log.layer_list if layer.func_name == "sigmoid")
    fc2_layer = next(layer for layer in log.layer_list if layer.layer_label.startswith("linear_2"))

    assert relu_layer.intervention_replaced is True
    assert sigmoid_layer.intervention_replaced is True
    assert sigmoid_layer.parents == [fc2_layer.layer_label]
    assert len(_replacement_layers(log)) >= 2


def test_quantization_sensitivity_pattern() -> None:
    """A minimal fake-quant replacement pattern captures without graph breakage."""

    log = tl.trace(
        _HookedMlp(),
        torch.randn(3, 4),
        intervention_ready=True,
        hooks={tl.func("sigmoid"): _quantize_dequantize_hook},
    )

    sigmoid_layer = next(layer for layer in log.layer_list if layer.func_name == "sigmoid")

    assert sigmoid_layer.intervention_replaced is True
    assert sigmoid_layer.parents
    assert sigmoid_layer.out is not None


def test_fresh_tensor_replacement_at_nested_boundary_and_model_output() -> None:
    """Fresh tensors from nested and root raw hooks keep TorchLens metadata intact."""

    model = _NestedFreshTensorModel()

    def nested_hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fresh tensor at a nested module boundary."""

        del module, args
        return output.clone()

    def root_hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fresh tensor as the model output."""

        del module, args
        return output + torch.zeros_like(output)

    nested_handle = model.block.register_forward_hook(nested_hook)
    root_handle = model.register_forward_hook(root_hook)
    try:
        log = tl.trace(model, torch.randn(3, 4), intervention_ready=True)
    finally:
        nested_handle.remove()
        root_handle.remove()

    assert log.output_layers
    assert log[log.output_layers[0]].out is not None
    assert any(getattr(layer, "intervention_replaced", False) for layer in log.layer_list)
    assert all(getattr(layer, "layer_label", None) for layer in log.layer_list)
