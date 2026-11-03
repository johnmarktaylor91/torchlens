"""Phase 5a tests for site collections."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.intervention.sites import SiteCollection, sites as _sites


def test_sites_builds_structured_sweep() -> None:
    """``sites()`` should compose layer, op, and mode sweep dimensions."""

    collection = _sites("encoder.layer", ops=["relu", "add"], modes=["clean", "corrupt"])

    assert isinstance(collection, SiteCollection)
    assert len(collection) == 4
    assert len(collection.selectors()) == 4
    assert [entry.mode for entry in collection] == ["clean", "clean", "corrupt", "corrupt"]


def test_sites_can_expand_to_hook_pairs() -> None:
    """Site collections should be usable as hook attachment inputs."""

    log = tl.trace(
        torch.nn.ReLU(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )

    def hook_fn(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
        """Return the out unchanged.

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

    log.do(
        [(_sites("relu", ops=["relu"], modes=["observe"]), hook_fn)],
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    assert len(log._intervention_spec.hook_specs) == 1
    log.clear_hooks(confirm_mutation=True)
    assert log._intervention_spec.hook_specs == []
