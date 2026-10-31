"""Fixwave-2 FW2-POLISH regression pins for the R47 dead-config debloat.

Two families of silent no-op configuration killed in grind b7 R47:

- R47-1: nine write-only "future" option fields (declared, validated, read by
  NOTHING). The fields and their keywords are deleted outright (shim-removal
  lane, 2026-08-19: the loud-no-op deprecation window was dropped).
- R47-5: the ``_module_containment_engine`` capture knob was validated and
  plumbed but never compared anywhere; the whole plumbing is gone.
"""

from __future__ import annotations

import pytest

import torchlens as tl

INERT_OPTION_KWARGS = [
    (tl.options.InterventionOptions, "helper_validation", "default"),
    (tl.options.InterventionOptions, "auto_promote", False),
    (tl.options.InterventionOptions, "cohort_migration", True),
    (tl.options.InterventionOptions, "error_severity_threshold", "recoverable"),
    (tl.options.SaveOptions, "output_dir", "/tmp/nowhere"),
    (tl.options.SaveOptions, "save_level", "full"),
    (tl.options.SaveOptions, "bundle_format", "directory"),
    (tl.options.ReplayOptions, "is_appended", True),
    (tl.options.ReplayOptions, "device_override", "cpu"),
]


@pytest.mark.parametrize(
    "cls,kwarg,value",
    INERT_OPTION_KWARGS,
    ids=[f"{cls.__name__}.{kwarg}" for cls, kwarg, _ in INERT_OPTION_KWARGS],
)
def test_inert_option_kwarg_is_gone(cls, kwarg, value):
    """A deleted write-only field refuses as an unknown keyword (R47-1)."""

    with pytest.raises(TypeError):
        cls(**{kwarg: value})
    assert not hasattr(cls(), kwarg)
    assert kwarg not in cls().as_dict()


def test_module_containment_engine_kwarg_is_deleted():
    """The never-compared containment-engine knob is gone entirely (R47-5)."""

    with pytest.raises(TypeError):
        tl.options.CaptureOptions(_module_containment_engine="hook_stack")
    assert not hasattr(tl.options.CaptureOptions(), "_module_containment_engine")
