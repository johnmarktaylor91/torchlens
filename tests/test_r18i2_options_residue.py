"""Regression tests for r18i2 options residue (finding L2).

L2 (de-bloat): ``torchlens.options._merge_grouped_options`` carried a
keyword-only ``fields`` parameter that was documented but never read in the
function body. Every one of its five call sites threaded a ``_*_FIELDS``
constant into it that the merge never consumed -- misleading dead plumbing that
implied a duplicate schema. Removing the parameter (and the five dead
``fields=`` arguments) must not change any merge behavior; these tests guard the
removal and pin the behavior that must stay identical.
"""

import inspect

import pytest

import torchlens.options as options


def test_l2_merge_grouped_options_has_no_dead_fields_param() -> None:
    """The unread ``fields`` parameter must stay removed from the signature.

    Regression guard: this assertion fails on the pre-fix tree where ``fields``
    was a keyword-only parameter of ``_merge_grouped_options`` that the body
    never referenced.
    """

    params = inspect.signature(options._merge_grouped_options).parameters
    assert "fields" not in params, (
        "`_merge_grouped_options` re-grew the unread `fields` parameter (dead schema plumbing)."
    )


def test_l2_merge_capture_flat_kwarg_still_applies() -> None:
    """A flat capture kwarg still overrides the default after fields removal."""

    merged = options.merge_capture_options(capture=None, verbose=True)
    assert isinstance(merged, options.CaptureOptions)
    assert merged.options.verbose is True


def test_l2_merge_save_flat_kwarg_still_applies() -> None:
    """A flat save kwarg still overrides its default after fields removal."""

    merged = options.merge_save_options(save=None, save_raw_activations=False)
    assert isinstance(merged, options.SaveOptions)
    assert merged.options.save_raw_activations is False


def test_l2_merge_default_when_none_unchanged() -> None:
    """Merging with ``None`` still yields the plain defaults."""

    merged = options.merge_capture_options(capture=None)
    assert merged.options.verbose is False
    assert merged.save_args is False


def test_l2_merge_grouped_flat_conflict_still_raises() -> None:
    """Passing both a grouped object and a conflicting flat kwarg still raises."""

    with pytest.raises(ValueError):
        options.merge_capture_options(capture=options.CaptureOptions(verbose=True), verbose=True)


def test_l2_all_merge_entrypoints_round_trip_defaults() -> None:
    """Every merge entrypoint still constructs its grouped type from ``None``."""

    assert isinstance(options.merge_capture_options(capture=None), options.CaptureOptions)
    assert isinstance(options.merge_save_options(save=None), options.SaveOptions)
    assert isinstance(options.merge_replay_options(replay=None), options.ReplayOptions)
    assert isinstance(
        options.merge_intervention_options(intervention=None),
        options.InterventionOptions,
    )
    assert isinstance(options.merge_streaming_options(streaming=None), options.StorageOptions)
