"""Phase 5a tests for sequential hook composition and handles."""

from __future__ import annotations

import torch

import torchlens as tl


class ReluModel(torch.nn.Module):
    """Small model with a ReLU site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ReLU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(x)


def test_sequential_hooks_run_in_order_during_rerun() -> None:
    """Composed hooks should run left-to-right."""

    order: list[str] = []

    def add_one(activation: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Add one to the activation.

        Parameters
        ----------
        activation:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Shifted activation.
        """

        del hook
        order.append("add")
        return activation + 1

    def times_two(activation: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Double the activation.

        Parameters
        ----------
        activation:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Scaled activation.
        """

        del hook
        order.append("mul")
        return activation * 2

    model = ReluModel()
    x = torch.tensor([[-1.0, 2.0]])
    log = tl.log_forward_pass(model, x, intervention_ready=True)
    log.attach_hooks(tl.func("relu"), add_one, times_two, confirm_mutation=True)

    rerun_log = log.rerun(model, x)

    assert order == ["add", "mul"]
    assert torch.equal(rerun_log[rerun_log.output_layers[0]].activation, (torch.relu(x) + 1) * 2)


def test_hook_handle_remove_and_context_manager_cleanup() -> None:
    """Hook handles should remove specs explicitly and on context exit."""

    log = tl.log_forward_pass(ReluModel(), torch.randn(1, 2), intervention_ready=True)

    def identity(activation: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Return activation unchanged.

        Parameters
        ----------
        activation:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Original activation.
        """

        del hook
        return activation

    handle = log.attach_hooks(tl.func("relu"), identity, confirm_mutation=True)
    assert len(log._intervention_spec.hook_specs) == 1
    handle.remove()
    assert log._intervention_spec.hook_specs == []

    with log.attach_hooks(tl.func("relu"), identity, confirm_mutation=True):
        assert len(log._intervention_spec.hook_specs) == 1
    assert log._intervention_spec.hook_specs == []
