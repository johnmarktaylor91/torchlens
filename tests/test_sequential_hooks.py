"""Phase 5a tests for sequential hook composition and clearing staged hooks."""

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

    def add_one(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
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

        del ctx
        order.append("add")
        return out + 1

    def times_two(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
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

        del ctx
        order.append("mul")
        return out * 2

    model = ReluModel()
    x = torch.tensor([[-1.0, 2.0]])
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    stage = tl.options.InterventionOptions(engine="set_only", confirm_mutation=True)
    log.do([(tl.func("relu"), add_one), (tl.func("relu"), times_two)], intervention=stage)

    rerun_log = log.run(model, x)

    assert order == ["add", "mul"]
    assert torch.equal(rerun_log[rerun_log.output_layers[0]].out, (torch.relu(x) + 1) * 2)


def test_clear_edits_removes_the_staged_hook_specs() -> None:
    """``clear_edits(site)`` removes the sticky hook specs a staged ``do`` added."""

    log = tl.trace(
        ReluModel(),
        torch.randn(1, 2),
        capture=tl.options.CaptureOptions(replayable=True),
    )

    def identity(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
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

        del ctx
        return out

    stage = tl.options.InterventionOptions(engine="set_only", confirm_mutation=True)
    log.do({tl.func("relu"): identity}, intervention=stage)
    assert len(log._intervention_spec.hook_specs) == 1
    log.clear_edits(tl.func("relu"), confirm_mutation=True)
    assert log._intervention_spec.hook_specs == []
