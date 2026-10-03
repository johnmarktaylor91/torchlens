"""Unit tests for the shared closed-vocabulary env-knob parser (round-7 R47)."""

from __future__ import annotations

import pytest

from torchlens._errors import InvalidArgumentError
from torchlens.utils.env_flags import closed_bool_env

_KNOB = "TORCHLENS_TEST_ENV_FLAG"


@pytest.mark.parametrize("spelling", ["1", "true", "True", "YES", "on", " on "])
def test_affirmative_spellings_parse_true(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    monkeypatch.setenv(_KNOB, spelling)
    assert closed_bool_env(_KNOB) is True


@pytest.mark.parametrize("spelling", ["0", "false", "False", "NO", "off", " off "])
def test_negative_spellings_parse_false(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    """The explicit-off vocabulary parses False.

    RED before the fix at the ``TORCHLENS_DEBUG_FORK_COPY`` site: raw
    truthiness treated ``"0"`` as ENABLED.
    """

    monkeypatch.setenv(_KNOB, spelling)
    assert closed_bool_env(_KNOB) is False


def test_unset_and_empty_take_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_KNOB, raising=False)
    assert closed_bool_env(_KNOB) is False
    assert closed_bool_env(_KNOB, default=True) is True
    monkeypatch.setenv(_KNOB, "")
    assert closed_bool_env(_KNOB, default=True) is True


def test_unrecognized_value_refuses_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must refuse, never silently select a state (disarmed tripwire)."""

    monkeypatch.setenv(_KNOB, "typo")
    with pytest.raises(InvalidArgumentError) as exc_info:
        closed_bool_env(_KNOB)
    assert exc_info.value.fields["code"] == "env_flag_invalid"
    assert _KNOB in str(exc_info.value)
