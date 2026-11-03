"""Every selector refusal in ``torchlens._ir.selector_eval`` carries a stable code.

The stress matrix's ``hook_live_final_label`` row found ``LiveModeLabelError``
typed but code-less (``fields["code"]`` was ``None``), so a caller could only
branch on message text. Each raise site in the selector evaluator now names its
code; these tests provoke every one through its production raise site. The
``LiveModeLabelError`` sites are pinned in ``tests/test_stress_refusal_codes.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import torchlens as tl
from torchlens._ir.selector_eval import (
    _compile_predicate,
    evaluate,
    normalize_selector_like,
    selector_from_spec,
)
from torchlens.intervention.errors import SelectorCompositionError, SiteResolutionError

_LIVE_SUBJECT = SimpleNamespace(_layer_label_raw="relu_1")


def test_where_payload_must_be_callable() -> None:
    selector = SimpleNamespace(selector_value="not callable")
    with pytest.raises(SiteResolutionError) as excinfo:
        _compile_predicate(selector, "site")(_LIVE_SUBJECT)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "where_predicate_not_callable"


def test_live_predicate_failure_chains_user_error_with_code() -> None:
    def explode(_ctx: object) -> bool:
        raise ZeroDivisionError("boom")

    with pytest.raises(SiteResolutionError) as excinfo:
        evaluate(tl.where(explode), _LIVE_SUBJECT, lifecycle="live")
    assert excinfo.value.fields["code"] == "live_predicate_failed"
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_standalone_followed_by_at_capture_time_carries_code() -> None:
    with pytest.raises(SelectorCompositionError) as excinfo:
        evaluate(tl.followed_by(tl.func("relu")), _LIVE_SUBJECT, lifecycle="capture")
    assert excinfo.value.fields["code"] == "followed_by_composition_unsupported"


@pytest.mark.parametrize(
    ("kind", "value", "lifecycle", "code"),
    [
        ("facet", {"module_address": "encoder"}, "site", "facet_selector_invalid"),
        ("backward_pass", "second", "site", "backward_pass_selector_invalid"),
        ("and", "not-a-sequence", "site", "composite_selector_invalid"),
        ("followed_by", "<lambda>", "capture", "lookback_selector_inner_opaque"),
        ("preceded_by", "<lambda>", "capture", "lookback_selector_inner_opaque"),
        ("no_such_kind", None, "site", "selector_kind_unsupported"),
    ],
)
def test_target_spec_payload_refusals_carry_codes(
    kind: str, value: object, lifecycle: str, code: str
) -> None:
    with pytest.raises(SiteResolutionError) as excinfo:
        selector_from_spec(kind, value, None, lifecycle=lifecycle)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == code


def test_unsupported_selector_like_inputs_carry_codes() -> None:
    with pytest.raises(SiteResolutionError) as live:
        normalize_selector_like(object(), lifecycle="live")
    assert live.value.fields["code"] == "live_hook_selector_unsupported"
    with pytest.raises(SiteResolutionError) as site:
        normalize_selector_like(object(), lifecycle="site")
    assert site.value.fields["code"] == "site_query_unsupported"
