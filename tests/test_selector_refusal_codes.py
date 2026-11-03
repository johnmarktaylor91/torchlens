"""Every selector refusal in ``torchlens._ir.selector_eval`` carries a stable code.

The stress matrix's ``hook_live_final_label`` row found ``LiveModeLabelError``
typed but code-less (``fields["code"]`` was ``None``), so a caller could only
branch on message text. Each raise site in the selector evaluator now names its
code; these tests provoke every one through its production raise site. The
capability refusals (a selector kind a lifecycle cannot evaluate) and the
post-capture site-resolution refusals (no match, excess fanout, a bad
``max_fanout``, a strict bare string, an empty site table) carry codes too, so
``Trace.set`` and ``attach_hooks`` refuse a model-input site with one code. The
``LiveModeLabelError`` sites are pinned in ``tests/test_stress_refusal_codes.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._ir.selector_eval import (
    _capability_error,
    _compile_predicate,
    evaluate,
    normalize_selector_like,
    selector_from_spec,
)
from torchlens._selection.resolver import _SiteTable, resolve_site_plan
from torchlens.intervention.errors import (
    AmbiguousSiteError,
    SelectorCapabilityError,
    SelectorCompositionError,
    SiteResolutionError,
)
from torchlens.options import CaptureOptions

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


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _ready_trace() -> tl.Trace:
    return tl.trace(_TwoLinear(), [torch.randn(2, 4)], capture=CaptureOptions(replayable=True))


def _scale_hook(out: torch.Tensor, *, ctx: object) -> torch.Tensor:
    return out * 0.5


def test_model_input_site_refusals_carry_one_code_from_both_doors() -> None:
    """``Trace.set`` and ``attach_hooks`` refuse a model-input site by its condition."""

    trace = _ready_trace()
    with pytest.raises(SelectorCapabilityError) as set_info:
        trace.set(tl.model_input(0), lambda out: out * 0.5, confirm_mutation=True)
    with pytest.raises(SelectorCapabilityError) as attach_info:
        trace._attach_hooks(tl.model_input(0), _scale_hook, confirm_mutation=True)
    assert set_info.value.fields["code"] == "model_input_selector_not_live"
    assert attach_info.value.fields["code"] == "model_input_selector_not_live"
    assert "not live hook application sites" in str(set_info.value)
    assert "matched 0 sites" not in str(set_info.value)


@pytest.mark.parametrize(
    ("kind", "lifecycle", "code"),
    [
        ("followed_by", "site", "followed_by_selector_unsupported"),
        ("preceded_by", "live", "preceded_by_selector_unsupported"),
        ("input_at", "live", "model_input_selector_not_live"),
        ("facet", "site", "facet_selector_unsupported"),
        ("site", "live", "site_selector_not_live"),
        ("mystery", "live", "selector_kind_unsupported"),
    ],
)
def test_capability_refusals_carry_codes(kind: str, lifecycle: str, code: str) -> None:
    error = _capability_error(kind, lifecycle)
    assert isinstance(error, SelectorCapabilityError)
    assert error.fields["code"] == code
    assert error.fields["selector_kind"] == kind
    assert error.fields["lifecycle"] == lifecycle


def test_site_resolution_refusals_carry_codes() -> None:
    trace = _ready_trace()
    with pytest.raises(SiteResolutionError) as none_info:
        trace._resolve_sites(tl.label("nope_9_9"))
    assert none_info.value.fields["code"] == "site_selector_no_match"
    with pytest.raises(AmbiguousSiteError) as fanout_info:
        trace._resolve_sites(tl.func("linear"), max_fanout=1)
    assert fanout_info.value.fields["code"] == "site_selector_excess_fanout"
    with pytest.raises(AmbiguousSiteError) as find_fanout_info:
        trace.find_sites(tl.func("linear"), max_fanout=1)
    assert find_fanout_info.value.fields["code"] == "site_selector_excess_fanout"
    with pytest.raises(SiteResolutionError) as none_fanout_info:
        trace._resolve_sites(tl.func("linear"), max_fanout=None)  # type: ignore[arg-type]
    assert none_fanout_info.value.fields["code"] == "max_fanout_invalid"
    with pytest.raises(AmbiguousSiteError) as zero_fanout_info:
        trace.find_sites(tl.func("linear"), max_fanout=0)
    assert zero_fanout_info.value.fields["code"] == "max_fanout_invalid"
    with pytest.raises(SiteResolutionError) as strict_info:
        trace._resolve_sites("linear", strict=True)
    assert strict_info.value.fields["code"] == "bare_string_selector_strict"
    with pytest.raises(SiteResolutionError) as strict_find_info:
        trace.find_sites("linear", strict=True)
    assert strict_find_info.value.fields["code"] == "bare_string_selector_strict"
    with pytest.raises(SiteResolutionError) as strict_plan_info:
        resolve_site_plan(trace, "linear", strict=True)
    assert strict_plan_info.value.fields["code"] == "bare_string_selector_strict"
    with pytest.raises(SiteResolutionError) as empty_info:
        _SiteTable(()).first()
    assert empty_info.value.fields["code"] == "site_table_empty"
