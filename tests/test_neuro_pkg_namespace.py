"""Lane F22 neuro namespace mechanics (neuro MEMO D14-D16, test row T13).

Pins:

- ``hasattr`` ANSWERS (never raises) for every table row and for absent
  names, with or without the extra installed.
- Dunder/underscore probes raise plain ``AttributeError``, never a
  dependency error (IPython canaries must fail fast).
- Redirect and refusal rows raise typed teaching errors that ARE
  ``AttributeError`` subclasses; nothing from the tables enters ``dir()``.
- The redirect-table walk: every torchlens spelling a redirect names
  actually resolves, so redirects cannot rot.
- The Brain-Score live gate PASSED (D14 alias branch): the verified
  adapters ``activations_extractor``/``get_activations_fn`` are real names
  gated on ``brainscore_vision`` alone, and the ``brain_score`` spelling
  stays a redirect that names them (``hasattr`` stays ``False``).
- Per-name dependency gates carry ImportError SEMANTICS (package + install
  command) with AttributeError lineage.
"""

from __future__ import annotations

import pytest

import torchlens as tl
import torchlens.neuro as neuro
from torchlens._errors import FacadeTeachingError, MissingDependencyError


@pytest.mark.smoke
def test_real_names_and_dir_contract() -> None:
    """dir()/__all__ advertise exactly the dependency-present real names."""

    import importlib.util

    expected: list[str] = []
    if importlib.util.find_spec("rsatoolbox") is not None:
        expected += ["datasets", "rdms"]
    if importlib.util.find_spec("brainscore_vision") is not None:
        expected += ["activations_extractor", "get_activations_fn"]
    expected.sort()
    public = [name for name in dir(neuro) if not name.startswith("_")]
    assert public == expected
    assert sorted(neuro.__all__) == expected
    pytest.importorskip("rsatoolbox")
    assert callable(neuro.datasets)
    assert callable(neuro.rdms)


def test_brain_score_adapters_gate_on_brainscore_vision() -> None:
    """The D14 alias pair resolves iff brainscore_vision is installed."""

    import importlib.util

    if importlib.util.find_spec("brainscore_vision") is None:
        assert hasattr(neuro, "activations_extractor") is False
        with pytest.raises(AttributeError) as excinfo:
            _ = neuro.activations_extractor
        assert isinstance(excinfo.value, MissingDependencyError)
        assert excinfo.value.fields["dependency"] == "brainscore_vision"
        assert "torchlens[brainscore]" in excinfo.value.fields["install"]
    else:  # pragma: no cover - exercised only on py3.11+ gate environments
        assert callable(neuro.activations_extractor)
        assert callable(neuro.get_activations_fn)


def test_hasattr_answers_and_underscore_short_circuit() -> None:
    """hasattr never raises; underscore probes get plain AttributeError."""

    assert hasattr(neuro, "brain_score") is False
    assert hasattr(neuro, "noise_ceiling") is False
    assert hasattr(neuro, "definitely_absent_name") is False
    with pytest.raises(AttributeError) as excinfo:
        _ = neuro._repr_html_
    assert type(excinfo.value) is AttributeError
    with pytest.raises(AttributeError):
        _ = neuro._ipython_canary_method_should_not_exist_
    assert getattr(neuro, "no_such_name", None) is None


def test_redirect_rows_are_typed_teaching_attribute_errors() -> None:
    """Redirect rows raise FacadeTeachingError (AttributeError lineage)."""

    for name in neuro._REDIRECTS:
        with pytest.raises(AttributeError) as excinfo:
            getattr(neuro, name)
        error = excinfo.value
        assert isinstance(error, FacadeTeachingError)
        assert error.fields["code"] == "facade_redirect"
        assert error.fields["attribute"] == name


def test_refusal_rows_are_typed_teaching_attribute_errors() -> None:
    """Refusal rows raise FacadeTeachingError with the refusal code."""

    for name in neuro._REFUSALS:
        with pytest.raises(AttributeError) as excinfo:
            getattr(neuro, name)
        error = excinfo.value
        assert isinstance(error, FacadeTeachingError)
        assert error.fields["code"] == "facade_refusal"
        assert error.fields["attribute"] == name


def test_brain_score_redirect_names_verified_adapters() -> None:
    """D14 alias branch: the redirect names the two verified spellings."""

    with pytest.raises(AttributeError) as excinfo:
        _ = neuro.brain_score
    message = str(excinfo.value)
    assert "tl.bridge.brain_score" in message
    assert "activations_extractor" in message
    assert hasattr(neuro, "brain_score") is False


#: The torchlens spelling each redirect row teaches, resolved by the walk
#: test below so redirects cannot rot (MEMO D16). brain_score's target is
#: the bridge module itself (its own dependency teaching is separate).
_REDIRECT_TARGETS: dict[str, str] = {
    "rdm": "repgeom.rdm",
    "cka": "stats.cka",
    "mds": "repgeom.classical_mds",
    "classical_mds": "repgeom.classical_mds",
    "scree": "repgeom.scree",
    "pca": "repgeom.effective_dimensionality",
    "effective_dimensionality": "repgeom.effective_dimensionality",
    "procrustes_align": "repgeom.procrustes_align",
    "extract_dataset": "extract_dataset",
    "load_extraction": "load_extraction",
    "brain_score": "bridge.brain_score",
}


@pytest.mark.smoke
def test_redirect_table_walk_every_target_resolves() -> None:
    """Every torchlens spelling named by a redirect resolves (D16)."""

    assert set(_REDIRECT_TARGETS) == set(neuro._REDIRECTS)
    for name, dotted in _REDIRECT_TARGETS.items():
        target: object = tl
        for part in dotted.split("."):
            target = getattr(target, part)
        assert target is not None, f"redirect {name!r} names a dead spelling {dotted!r}"


def test_refusal_rows_name_an_owner() -> None:
    """Each refusal names an owning neighbour or the feeding spelling."""

    owners = (
        "rsatoolbox",
        "Brain-Score",
        "himalaya",
        "netrep",
        "scikit-learn",
        "thingsvision",
        "tl.",
    )
    for name, message in neuro._REFUSALS.items():
        assert any(owner in message for owner in owners), name


def test_dependency_gate_teaches_install_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent dependency raises AttributeError with install semantics."""

    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "rsatoolbox":
            return None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.delitem(neuro.__dict__, "datasets", raising=False)
    with pytest.raises(AttributeError) as excinfo:
        _ = neuro.datasets
    error = excinfo.value
    assert isinstance(error, MissingDependencyError)
    assert error.fields["dependency"] == "rsatoolbox"
    assert "torchlens[neuro]" in error.fields["install"]
    assert hasattr(neuro, "datasets") is False
    monkeypatch.undo()


def test_module_docstring_is_the_five_section_map() -> None:
    """The docstring carries the extract->geometry->handoff->score->refusal map."""

    doc = neuro.__doc__ or ""
    for section in ("EXTRACT", "GEOMETRY", "HANDOFF", "SCORE", "REFUSALS"):
        assert section in doc
