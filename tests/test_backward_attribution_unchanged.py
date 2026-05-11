"""Phase 5(e): backward op module attribution unchanged by refactor."""

from __future__ import annotations

import torch

import torchlens as tl


def test_backward_pass_module_attribution() -> None:
    """Backward capture keeps forward-op module attribution available."""

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    x = torch.randn(2, 8, requires_grad=True)
    trace = tl.trace(model, x, vis_opt="none", grads_to_save="all")
    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    forward_ops_with_modules = [op for op in trace.layer_list if getattr(op, "modules", [])]
    backward_ops_with_forward_modules = [
        grad_fn
        for grad_fn in trace.grad_fns
        if getattr(getattr(grad_fn, "op", None), "modules", [])
    ]
    assert forward_ops_with_modules, "expected forward modules to be populated"
    assert backward_ops_with_forward_modules, "expected backward logs to retain forward links"
