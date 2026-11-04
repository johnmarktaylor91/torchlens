"""``tl.errors.quiet``: the one switch that silences TorchLens warnings."""

from __future__ import annotations

import warnings

import pytest

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.errors._quiet import quiet_all_active


class _CodedWarning(TorchLensWarning):
    """A coded TorchLens warning for the switch tests."""


def _emitted(emit: object) -> list[str]:
    """Messages that reach the warning record while ``emit`` runs."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit()  # type: ignore[operator]
    return [str(item.message) for item in caught]


def test_quiet_with_codes_silences_only_those_codes() -> None:
    """``quiet("a")`` hides code ``a`` and keeps every other warning."""

    def emit() -> None:
        with tl.errors.quiet("code_a"):
            warnings.warn(_CodedWarning("hidden", code="code_a"), stacklevel=1)
            warnings.warn(_CodedWarning("shown", code="code_b"), stacklevel=1)
            warnings.warn("plain", _CodedWarning, stacklevel=1)
        warnings.warn(_CodedWarning("after", code="code_a"), stacklevel=1)

    assert _emitted(emit) == ["shown", "plain", "after"]


def test_quiet_without_codes_silences_every_torchlens_warning() -> None:
    """``quiet()`` hides coded, uncoded and string-form TorchLens warnings, not others."""

    def emit() -> None:
        with tl.errors.quiet():
            assert quiet_all_active()
            warnings.warn(_CodedWarning("coded", code="code_a"), stacklevel=1)
            warnings.warn("string form", _CodedWarning, stacklevel=1)
            warnings.warn("not torchlens", UserWarning, stacklevel=1)
        assert not quiet_all_active()

    assert _emitted(emit) == ["not torchlens"]


def test_quiet_nests_and_restores_each_level() -> None:
    """Nested scopes widen the code set and each exit restores the prior level."""

    def emit() -> None:
        with tl.errors.quiet("code_a"):
            with tl.errors.quiet("code_b"):
                warnings.warn(_CodedWarning("a inner", code="code_a"), stacklevel=1)
                warnings.warn(_CodedWarning("b inner", code="code_b"), stacklevel=1)
            warnings.warn(_CodedWarning("a outer", code="code_a"), stacklevel=1)
            warnings.warn(_CodedWarning("b outer", code="code_b"), stacklevel=1)

    assert _emitted(emit) == ["b outer"]


def test_quiet_session_form_is_in_force_before_any_with() -> None:
    """A bare call silences at once; leaving its scope restores the warning."""

    def emit() -> None:
        scope = tl.errors.quiet("code_a")
        try:
            warnings.warn(_CodedWarning("hidden", code="code_a"), stacklevel=1)
        finally:
            scope.__exit__(None, None, None)
        warnings.warn(_CodedWarning("back", code="code_a"), stacklevel=1)

    assert _emitted(emit) == ["back"]


def test_quiet_beats_an_error_filter_and_the_error_returns_after() -> None:
    """A quieted code is not promoted to an error; outside the scope it raises again."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with tl.errors.quiet("code_a"):
            warnings.warn(_CodedWarning("quiet", code="code_a"), stacklevel=1)
        with pytest.raises(_CodedWarning):
            warnings.warn(_CodedWarning("loud", code="code_a"), stacklevel=1)


def test_quieted_warning_keeps_its_class_and_fields() -> None:
    """Under a later ``simplefilter`` a quieted warning still is its own class."""

    with warnings.catch_warnings(record=True) as caught:
        with tl.errors.quiet("code_a"):
            warnings.simplefilter("always")
            warnings.warn(_CodedWarning("seen", code="code_a"), stacklevel=1)
    assert len(caught) == 1
    assert isinstance(caught[0].message, _CodedWarning)
    assert caught[0].category.__name__ == "_CodedWarning"
    assert caught[0].message.fields["code"] == "code_a"


def test_the_removed_mutate_switch_names_quiet() -> None:
    """The removed ``suppress_mutate_warnings`` spelling names its replacement."""

    assert not hasattr(tl.io, "suppress_mutate_warnings")
    assert not hasattr(tl.options, "suppress_mutate_warnings")
    with pytest.raises(AttributeError, match=r"torchlens\.errors\.quiet"):
        _ = tl.suppress_mutate_warnings
