"""Phase 5a tests for site collections."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.intervention.sites import SiteCollection


def test_sites_builds_structured_sweep() -> None:
    """``tl.sites`` should compose layer, op, and mode sweep dimensions."""

    collection = tl.sites("encoder.layer", ops=["relu", "add"], modes=["clean", "corrupt"])

    assert isinstance(collection, SiteCollection)
    assert len(collection) == 4
    assert len(collection.selectors()) == 4
    assert [entry.mode for entry in collection] == ["clean", "clean", "corrupt", "corrupt"]


def test_sites_can_expand_to_hook_pairs() -> None:
    """Site collections should be usable as hook attachment inputs."""

    log = tl.log_forward_pass(
        torch.nn.ReLU(),
        torch.randn(2, 3),
        intervention_ready=True,
    )

    def hook_fn(activation: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Return the activation unchanged.

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

    handle = log.attach_hooks(
        tl.sites("relu", ops=["relu"], modes=["observe"]),
        hook_fn,
        confirm_mutation=True,
    )
    assert len(log._intervention_spec.hook_specs) == 1
    handle.remove()
    assert log._intervention_spec.hook_specs == []
