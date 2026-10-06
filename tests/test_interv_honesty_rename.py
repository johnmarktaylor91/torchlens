"""Intervention honesty: the ``resample_ablate`` -> ``scramble_elements``
honest rename (edits memo row 1, decision D19; alias posture: hard rename,
no alias, artifact-load compatibility KEPT).

The helper is an elementwise iid scramble -- a false friend of the field's
"resampling ablation" (coherent donor patching). Same bytes, honest name:

- ``scramble_elements`` is the only constructor; the ``resample_ablate``
  spelling is gone from ``torchlens.intervention`` and the ``tl.*`` facade.
- Persisted specs naming ``resample_ablate`` keep loading (artifact format).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention import scramble_elements
from torchlens.intervention.helpers import rebuild_builtin_helper


def test_old_spelling_is_gone() -> None:
    """Clean break: no alias survives in the helpers module, the package, or tl.*."""

    import torchlens.intervention as intervention
    from torchlens.intervention import helpers

    assert not hasattr(helpers, "resample_ablate")
    assert not hasattr(intervention, "resample_ablate")
    assert "resample_ablate" not in intervention.__all__
    assert "resample_ablate" not in tl.__all__
    with pytest.raises(AttributeError):
        tl.resample_ablate  # noqa: B018 - the attribute access IS the assertion


def test_constructor_mints_the_honest_name() -> None:
    """Specs carry the honest helper name."""

    assert scramble_elements(torch.ones(3), seed=0).helper_name == "scramble_elements"


@pytest.mark.smoke
def test_seeded_draws_are_reproducible() -> None:
    """Same seed, same draws (the rename changed no behavior)."""

    source = torch.arange(12.0)
    out = torch.zeros(2, 3)

    first = scramble_elements(source, seed=7).factory()(out, hook=None)
    second = scramble_elements(source, seed=7).factory()(out, hook=None)
    assert torch.equal(first, second)
    assert set(first.flatten().tolist()) <= set(source.tolist())


@pytest.mark.parametrize("persisted_name", ["scramble_elements", "resample_ablate"])
def test_serialized_specs_load_under_both_names(persisted_name: str) -> None:
    """Artifact-load compatibility: legacy persisted helper names reconstruct."""

    spec = rebuild_builtin_helper(persisted_name, (torch.ones(3),), {"seed": 3})
    assert spec.helper_name == "scramble_elements"
    hook = spec.factory()
    sampled = hook(torch.zeros(4), hook=None)
    assert torch.equal(sampled, torch.ones(4))


def test_empty_source_refusal_names_the_honest_helper() -> None:
    """Fire-time refusals teach the current name, not the retired one."""

    from torchlens.intervention.errors import HookValueError

    hook = scramble_elements(torch.ones(0), seed=1).factory()
    with pytest.raises(HookValueError, match="scramble_elements"):
        hook(torch.zeros(3), hook=None)
