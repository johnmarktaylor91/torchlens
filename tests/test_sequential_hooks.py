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

    def add_one(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Add one to the out.

        Parameters
        ----------
        out:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Shifted out.
        """

        del hook
        order.append("add")
        return out + 1

    def times_two(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Double the out.

        Parameters
        ----------
        out:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Scaled out.
        """

        del hook
        order.append("mul")
        return out * 2

    model = ReluModel()
    x = torch.tensor([[-1.0, 2.0]])
    log = tl.trace(model, x, intervention_ready=True)
    log.attach_hooks(tl.func("relu"), add_one, times_two, confirm_mutation=True)

    rerun_log = log.rerun(model, x)

    assert order == ["add", "mul"]
    assert torch.equal(rerun_log[rerun_log.output_layers[0]].out, (torch.relu(x) + 1) * 2)


def test_hook_handle_remove_and_context_manager_cleanup() -> None:
    """Hook handles should remove specs explicitly and on context exit."""

    log = tl.trace(ReluModel(), torch.randn(1, 2), intervention_ready=True)

    def identity(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Return out unchanged.

        Parameters
        ----------
        out:
            Activation at the site.
        hook:
            Hook context.

        Returns
        -------
        torch.Tensor
            Original out.
        """

        del hook
        return out

    handle = log.attach_hooks(tl.func("relu"), identity, confirm_mutation=True)
    assert len(log._intervention_spec.hook_specs) == 1
    handle.remove()
    assert log._intervention_spec.hook_specs == []

    with log.attach_hooks(tl.func("relu"), identity, confirm_mutation=True):
        assert len(log._intervention_spec.hook_specs) == 1
    assert log._intervention_spec.hook_specs == []
