"""The two structural validation reasons: ``op_unclassified`` and ``capture_inconsistent``.

Both are public members of ``tl.backends.VALIDATION_REASONS``. The TF and JAX
checkers emit each one from a pure decision helper, so every case is driven
here without the frameworks installed: an unclassifiable op and an initializer
contamination are ``op_unclassified``; a callback self-consistency failure, an
equation or op count mismatch, a kind multiset mismatch and a replay kind that
differs from the captured kind are ``capture_inconsistent``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchlens.backends._backend_protocol import VALIDATION_REASONS
from torchlens.validation._status import known_validation_reasons


def test_structural_reasons_are_public_members() -> None:
    assert {"op_unclassified", "capture_inconsistent"} <= VALIDATION_REASONS
    assert {"op_unclassified", "capture_inconsistent"} <= known_validation_reasons()


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("unclassified_op:relu_1_raw:Erfc", "op_unclassified"),
        ("initializer_contamination:fill_2_raw:Fill", "op_unclassified"),
        ("captured_output_missing_materialized_op:add_3_raw", "capture_inconsistent"),
        ("materialized_op_missing_capture:mul_4_raw", "capture_inconsistent"),
    ],
)
def test_tf_structural_failure_names_its_reason(failure: str, reason: str) -> None:
    from torchlens.backends.tf.validation import _structural_failure_decision

    raw_label = failure.split(":")[1]
    op = SimpleNamespace(label=raw_label.removesuffix("_raw"), func_name="f")
    decision = _structural_failure_decision(failure, {raw_label: op})

    assert decision["reason"] == reason
    assert decision["decision"] == "failed"
    assert decision["phase"] == "metadata"
    assert decision["op_label"] == op.label
    assert decision["justification"] == failure
    assert reason in VALIDATION_REASONS


def _jax_op(kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        label=f"{kind}_op", func_name=kind, annotations={"jax_capture_kind": kind}
    )


def _jax_capture(kind: str) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, primitive=kind)


def test_jax_equation_count_mismatch_is_capture_inconsistent() -> None:
    from torchlens.backends.jax._validation_log import _JaxValidationLog
    from torchlens.backends.jax.backend import _jax_captures_consistent

    log = _JaxValidationLog()
    consistent = _jax_captures_consistent(
        (_jax_capture("primitive"),), [_jax_op("primitive"), _jax_op("scan_read")], log
    )

    assert consistent is False
    assert log.first_failure == ("<trace>", "", "capture_inconsistent")
    assert log.decisions[-1]["reason"] == "capture_inconsistent"
    assert log.decisions[-1]["phase"] == "metadata"
    assert "2 equation ops" in log.decisions[-1]["justification"]


def test_jax_equation_kind_mismatch_is_capture_inconsistent() -> None:
    from torchlens.backends.jax._validation_log import _JaxValidationLog
    from torchlens.backends.jax.backend import _jax_captures_consistent

    log = _JaxValidationLog()
    consistent = _jax_captures_consistent(
        (_jax_capture("primitive"), _jax_capture("primitive")),
        [_jax_op("primitive"), _jax_op("scan_read")],
        log,
    )

    assert consistent is False
    assert log.first_failure == ("<trace>", "", "capture_inconsistent")
    assert (
        log.decisions[-1]["justification"] == "equation capture kinds differ from op capture kinds"
    )


def test_jax_consistent_captures_record_nothing() -> None:
    from torchlens.backends.jax._validation_log import _JaxValidationLog
    from torchlens.backends.jax.backend import _jax_captures_consistent

    log = _JaxValidationLog()
    assert (
        _jax_captures_consistent((_jax_capture("primitive"),), [_jax_op("primitive")], log) is True
    )
    assert log.decisions == []
    assert log.first_failure is None
