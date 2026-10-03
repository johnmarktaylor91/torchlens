"""Single-sourcing gate for the lookback payload policy vocabulary.

The canonical authority is the ``LookbackPayloadPolicy`` literal in
``torchlens/fastlog/options.py``; the runtime tuple is derived from it with
``typing.get_args``, so validation and the type can never drift apart. These
tests pin that derivation and the validation behavior at both boundaries.
"""

from __future__ import annotations

from typing import get_args

import pytest

from torchlens._errors import InvalidArgumentError
from torchlens.fastlog.options import (
    LOOKBACK_PAYLOAD_POLICIES,
    LookbackPayloadPolicy,
    RecordingOptions,
)


def test_runtime_tuple_derives_from_the_canonical_literal() -> None:
    """The runtime vocabulary is the literal's args, in the literal's order."""

    assert get_args(LookbackPayloadPolicy) == LOOKBACK_PAYLOAD_POLICIES
    assert LOOKBACK_PAYLOAD_POLICIES == (
        "metadata_only",
        "detached_raw",
        "transformed",
        "grad_connected",
        "disk_spilled",
    )


@pytest.mark.parametrize("policy", LOOKBACK_PAYLOAD_POLICIES)
def test_every_documented_policy_validates(policy: str) -> None:
    """Each vocabulary member is accepted by option validation."""

    options = RecordingOptions(lookback_payload_policy=policy)
    assert options.lookback_payload_policy == policy


def test_unknown_policy_refuses_typed_and_names_the_vocabulary() -> None:
    """An unknown policy fails closed with the typed code and the full list."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        RecordingOptions(lookback_payload_policy="bogus")
    assert excinfo.value.fields["code"] == "lookback_payload_policy_invalid"
    message = str(excinfo.value)
    for policy in LOOKBACK_PAYLOAD_POLICIES:
        assert repr(policy) in message
