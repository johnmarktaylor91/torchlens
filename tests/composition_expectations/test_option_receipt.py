"""Option receipt oracles (compo row 0.5): requested / effective / explicit / reason.

The receipt is M3's generic oracle substrate: effective-only reporting would
bless silently-discarded options (SG#18/19), so every public field carries
its requested value, effective value, explicitness, and a machine-readable
reason -- and FORCED-SILENTLY is banned as a steady state (forcings must ride
the ``adjustments`` channel with a closed-vocabulary reason).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.compo]


def test_receipt_covers_every_public_field_in_order() -> None:
    """One entry per public CaptureOptions field, dataclass order, no privates."""

    from tests.composition_expectations._censuses import capture_option_fields
    from torchlens.options import CaptureOptions, option_receipt

    receipt = option_receipt(CaptureOptions())
    assert tuple(entry.name for entry in receipt) == capture_option_fields()
    # 48 -> 49 (T82d re-reconcile, F44): each parent pinned 47+1 for its own
    # new option (landed track_device_memory x F44 log_injections); the
    # merged tree carries both. As of T85 (F01-AMENDED landed) both parents
    # carry log_injections, so the pins agree at 49.
    assert len(receipt) == 49


def test_receipt_distinguishes_explicit_from_default() -> None:
    """Explicitness and requested-value truth ride _specified_fields."""

    from torchlens._deprecations import MISSING
    from torchlens.options import CaptureOptions, option_receipt

    options = CaptureOptions(save_grads=True)
    by_name = {entry.name: entry for entry in option_receipt(options)}

    explicit = by_name["save_grads"]
    assert explicit.explicit and explicit.reason == "explicit"
    assert explicit.requested is True and explicit.effective is True

    default = by_name["raise_on_nan"]
    assert not default.explicit and default.reason == "default"
    assert default.requested is MISSING
    assert default.effective == CaptureOptions().raise_on_nan

    # An EXPLICIT value equal to the default is still explicit (the
    # explicit-default cell of the option axis, memo 3.3).
    explicit_default = CaptureOptions(raise_on_nan=CaptureOptions().raise_on_nan)
    entry = {e.name: e for e in option_receipt(explicit_default)}["raise_on_nan"]
    assert entry.explicit and entry.reason == "explicit"


def test_receipt_adjustments_disclose_forcing() -> None:
    """A consumer-forced option is disclosed with its reason, never silent."""

    from torchlens.options import CaptureOptions, option_receipt

    options = CaptureOptions(save_grads=True)
    receipt = option_receipt(options, adjustments={"save_grads": (False, "forced")})
    entry = {e.name: e for e in receipt}["save_grads"]
    assert entry.requested is True, "the requested value must survive the forcing"
    assert entry.effective is False
    assert entry.explicit
    assert entry.reason == "forced"


def test_receipt_refusals_are_typed_teaching() -> None:
    """The three receipt refusals carry stable codes and remedies."""

    from torchlens._errors import InvalidArgumentError
    from torchlens.options import CaptureOptions, option_receipt

    with pytest.raises(InvalidArgumentError) as not_options:
        option_receipt(object())
    assert not_options.value.fields["code"] == "option_receipt_not_options"
    assert not_options.value.fields["remedy"]

    with pytest.raises(InvalidArgumentError) as unknown:
        option_receipt(CaptureOptions(), adjustments={"no_such_option": (1, "forced")})
    assert unknown.value.fields["code"] == "option_receipt_unknown_field"

    with pytest.raises(InvalidArgumentError) as bad_reason:
        option_receipt(CaptureOptions(), adjustments={"save_grads": (False, "explicit")})
    assert bad_reason.value.fields["code"] == "option_receipt_reason_invalid"


def test_receipt_works_across_the_options_families() -> None:
    """Any _specified_fields-carrying options object yields a receipt."""

    from torchlens.options import ReplayOptions, SaveOptions, option_receipt

    save_receipt = option_receipt(SaveOptions())
    replay_receipt = option_receipt(ReplayOptions(append=True))
    assert all(entry.reason == "default" for entry in save_receipt)
    appended = {e.name: e for e in replay_receipt}["append"]
    assert appended.explicit and appended.effective is True
