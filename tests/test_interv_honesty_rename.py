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
    assert "resample_ablate" not in dir(tl)


@pytest.mark.parametrize("owner", ["torchlens", "intervention", "helpers"])
def test_old_spelling_raises_typed_error_naming_the_replacement(owner: str) -> None:
    """Every removed spelling raises facade_redirect naming scramble_elements."""

    import torchlens.intervention as intervention
    from torchlens._errors import FacadeTeachingError
    from torchlens.intervention import helpers

    module = {"torchlens": tl, "intervention": intervention, "helpers": helpers}[owner]
    with pytest.raises(FacadeTeachingError) as excinfo:
        module.resample_ablate  # noqa: B018 - the attribute access IS the assertion
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert "torchlens.intervention.scramble_elements" in str(excinfo.value)


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


def test_import_ref_to_the_retired_name_fails_closed_typed() -> None:
    """An import ref ``torchlens.intervention.helpers:resample_ablate`` fails closed, teaching.

    Built-in helpers persist by NAME (above), so only a hand-wrapped
    import-ref helper saved before 2.35.0 could carry this path. Resolution
    walks the module attribute, hits the helpers module's facade redirect,
    and the resolver wraps that in ``ReplayPreconditionError``. The cause
    must be the typed redirect naming ``scramble_elements``: without the
    helpers-module redirect it would be a bare AttributeError.
    """

    from torchlens._errors import FacadeTeachingError
    from torchlens.intervention import helpers
    from torchlens.intervention.errors import ReplayPreconditionError
    from torchlens.intervention.resolver import resolve_import_ref

    current = resolve_import_ref("torchlens.intervention.helpers:scramble_elements")
    assert current is helpers.scramble_elements
    with pytest.raises(ReplayPreconditionError) as excinfo:
        resolve_import_ref("torchlens.intervention.helpers:resample_ablate")
    cause = excinfo.value.__cause__
    assert isinstance(cause, FacadeTeachingError)
    assert cause.fields["code"] == "facade_redirect"
    assert "torchlens.intervention.scramble_elements" in str(cause)


def test_empty_source_refusal_names_the_honest_helper() -> None:
    """Fire-time refusals teach the current name, not the retired one."""

    from torchlens.intervention.errors import HookValueError

    hook = scramble_elements(torch.ones(0), seed=1).factory()
    with pytest.raises(HookValueError, match="scramble_elements"):
        hook(torch.zeros(3), hook=None)
