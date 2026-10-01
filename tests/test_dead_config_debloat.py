"""Fixwave-2 FW2-POLISH regression pins for the R47 dead-config debloat.

Three families of silent no-op configuration killed in grind b7 R47:

- R47-1: nine write-only "future" option fields (declared, validated, read by
  NOTHING). The fields and their keywords are deleted outright (shim-removal
  lane, 2026-08-19: the loud-no-op deprecation window was dropped).
- R47-3: the postprocess audit env knobs parsed permissively — a typo silently
  DISARMED the audit. They now refuse unrecognized values.
- R47-5: the ``_module_containment_engine`` capture knob was validated and
  plumbed but never compared anywhere; the whole plumbing is gone.
"""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.postprocess import (
    _POSTPROCESS_ASSERT_ENV,
    _READ_AUDIT_ENV,
    _WRITE_AUDIT_RECORD_ENV,
    _postprocess_assertions_enabled,
    _read_audit_mode,
    _write_audit_record_mode,
)

pytestmark = pytest.mark.smoke


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


@pytest.mark.parametrize(
    "env_name,parser,junk",
    [
        (_POSTPROCESS_ASSERT_ENV, _postprocess_assertions_enabled, "2"),
        (_POSTPROCESS_ASSERT_ENV, _postprocess_assertions_enabled, "ON "),
        (_WRITE_AUDIT_RECORD_ENV, _write_audit_record_mode, "recrod"),
        (_READ_AUDIT_ENV, _read_audit_mode, "enfroce"),
        (_READ_AUDIT_ENV, _read_audit_mode, "1"),
        (_READ_AUDIT_ENV, _read_audit_mode, "record "),
    ],
)
def test_postprocess_knob_refuses_unrecognized_values(monkeypatch, env_name, parser, junk):
    """A typo can no longer silently disarm or reroute an audit knob (R47-3).

    The refusal is the typed ``InvalidArgumentError`` door
    (``postprocess_audit_env_invalid``), not a bare ``RuntimeError``.
    """

    monkeypatch.setenv(env_name, junk)
    with pytest.raises(InvalidArgumentError, match=env_name) as exc_info:
        parser()
    assert exc_info.value.fields["code"] == "postprocess_audit_env_invalid"


def test_postprocess_knob_legal_values_still_parse(monkeypatch):
    """The closed vocabularies keep every documented spelling working."""

    monkeypatch.delenv(_POSTPROCESS_ASSERT_ENV, raising=False)
    monkeypatch.delenv(_WRITE_AUDIT_RECORD_ENV, raising=False)
    monkeypatch.delenv(_READ_AUDIT_ENV, raising=False)
    assert _postprocess_assertions_enabled() is False
    assert _write_audit_record_mode() is False
    assert _read_audit_mode() == ""

    monkeypatch.setenv(_POSTPROCESS_ASSERT_ENV, "1")
    assert _postprocess_assertions_enabled() is True
    monkeypatch.setenv(_POSTPROCESS_ASSERT_ENV, "off")
    assert _postprocess_assertions_enabled() is False
    monkeypatch.setenv(_WRITE_AUDIT_RECORD_ENV, "record")
    assert _write_audit_record_mode() is True
    for mode in ("record", "enforce"):
        monkeypatch.setenv(_READ_AUDIT_ENV, mode)
        assert _read_audit_mode() == mode
