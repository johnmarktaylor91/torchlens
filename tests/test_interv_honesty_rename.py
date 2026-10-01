"""Intervention honesty: the ``resample_ablate`` -> ``scramble_elements``
honest rename (edits memo row 1, decision D19; alias posture: hard rename,
no warn-shim, artifact-load compatibility KEPT).

The helper is an elementwise iid scramble -- a false friend of the field's
"resampling ablation" (coherent donor patching). Same bytes, honest name:

- ``scramble_elements`` is the canonical in-package constructor.
- The old in-package binding still resolves (the top-level facade flip is the
  facade owner's amendment) and mints specs carrying the HONEST name.
- Persisted specs naming ``resample_ablate`` keep loading.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention import scramble_elements
from torchlens.intervention.helpers import rebuild_builtin_helper

pytestmark = pytest.mark.smoke


def test_old_binding_is_the_same_function() -> None:
    """The transitional binding aliases the canonical constructor exactly."""

    from torchlens.intervention import helpers

    assert helpers.resample_ablate is helpers.scramble_elements
    assert tl.resample_ablate is scramble_elements


def test_both_spellings_mint_the_honest_name() -> None:
    """Specs carry ``scramble_elements`` regardless of construction spelling."""

    via_new = scramble_elements(torch.ones(3), seed=0)
    via_old = tl.resample_ablate(torch.ones(3), seed=0)
    assert via_new.helper_name == "scramble_elements"
    assert via_old.helper_name == "scramble_elements"


def test_seeded_draws_identical_across_spellings() -> None:
    """Same bytes: the rename changes no behavior."""

    source = torch.arange(12.0)
    out = torch.zeros(2, 3)

    hook_new = scramble_elements(source, seed=7).factory()
    hook_old = tl.resample_ablate(source, seed=7).factory()
    sampled_new = hook_new(out, hook=None)
    sampled_old = hook_old(out, hook=None)
    assert torch.equal(sampled_new, sampled_old)


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
