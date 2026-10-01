"""GATE-FIX row 1: the three paper-era teaching redirect rows are back.

The root redirect table (``torchlens._REDIRECTS``) exists to teach every
removed top-level spelling its canonical home. W051-GATE dropped the three
paper-era rows because the repo-wide removed-spelling lint forbade spelling
them in package code; the lint's own sanctioned mechanism for audited
teaching rows is its ``_ALLOWED`` ledger, which now admits
``torchlens/__init__.py`` (group ``paper_era``) and this pin.
"""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens._errors import FacadeTeachingError

pytestmark = pytest.mark.smoke

_PAPER_ERA_ROWS = {
    "log_forward_pass": "torchlens.trace",
    "ModelHistory": "torchlens.Trace",
    "validate_saved_activations": "torchlens.validate",
}


@pytest.mark.parametrize(("name", "home"), sorted(_PAPER_ERA_ROWS.items()))
def test_paper_era_spelling_is_a_typed_teaching_redirect(name: str, home: str) -> None:
    """Each paper-era name raises the typed facade_redirect error naming its home."""

    assert name in tl._REDIRECTS
    with pytest.raises(AttributeError) as excinfo:
        getattr(tl, name)
    assert type(excinfo.value) is FacadeTeachingError
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert home in str(excinfo.value)
    assert not hasattr(tl, name)
    assert name not in dir(tl)
