"""Regression tests for the dual predicate-consolidation review (opus + sol).

Every test here encodes one reproducer from the 2026-08-11 predicate reviews
(``results/review-predicate-opus.md`` / ``results/review-predicate-sol.md``).
Tests marked FIX failed at 4f568d5f (pre-fix arming proof); tests marked PIN
characterize semantics the reviews demanded be pinned explicitly.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, get_args

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import (
    SelectorCapabilityError,
    SiteResolutionError,
)
from torchlens.intervention.selectors import (
    BaseSelector,
    CompositeSelector,
    SelectorKind,
)
from torchlens.intervention.types import TargetSpec
from torchlens.ir.selector_eval import ensure_supported, selector_from_spec

pytestmark = pytest.mark.smoke


class TinyConvNet(nn.Module):
    """Nested-module CNN matching the selector-matrix oracle model."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Conv2d(2, 2, 3))
        self.head = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = x + 1
        x = x.flatten(1)
        return self.head(x)


def _conv_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 1, 8, 8)


@pytest.fixture(scope="module")
def conv_trace() -> Iterator[Any]:
    """Yield one warning-suppressed convolution Trace and clean it up."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(TinyConvNet(), _conv_input())
    try:
        yield trace
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# opus F-1 / sol#2 — n-ary and/or composites (FIX)
# ---------------------------------------------------------------------------


def test_nary_composite_repr_does_not_crash() -> None:
    """FIX: repr of a flat three-child composite raised ValueError."""

    selector = CompositeSelector("and", (tl.func("a"), tl.func("b"), tl.func("c")))
    text = repr(selector)
    assert text.count("&") == 2, text


def test_flat_nary_and_spec_resolves(conv_trace: Any) -> None:
    """FIX: a deserialized flat three-child ``and`` spec must resolve."""

    spec = TargetSpec(
        selector_kind="and",
        selector_value=(
            TargetSpec("func", "relu"),
            TargetSpec("in_module", "features"),
            TargetSpec("contains", "relu"),
        ),
    )
    rebuilt = selector_from_spec("and", spec.selector_value, None, lifecycle="site")
    assert repr(rebuilt).count("&") == 2
    expected = conv_trace.find_sites(
        tl.func("relu") & tl.in_module("features"), max_fanout=10**6
    ).labels()
    assert conv_trace.find_sites(spec, max_fanout=10**6).labels() == expected


def test_flat_nary_or_spec_resolves(conv_trace: Any) -> None:
    """FIX: a deserialized flat three-child ``or`` spec must resolve."""

    spec = TargetSpec(
        selector_kind="or",
        selector_value=(
            TargetSpec("func", "relu"),
            TargetSpec("func", "conv2d"),
            TargetSpec("func", "flatten"),
        ),
    )
    expected = conv_trace.find_sites(
        tl.func("relu") | tl.func("conv2d") | tl.func("flatten"), max_fanout=10**6
    ).labels()
    assert conv_trace.find_sites(spec, max_fanout=10**6).labels() == expected


def test_nary_live_backward_matcher_does_not_crash() -> None:
    """FIX: three-child composites crashed ``_live_backward_context_matches``."""

    from torchlens.intervention.hooks import live_backward_selector_matches

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        armed = tl.trace(
            TinyConvNet(),
            _conv_input().requires_grad_(True),
            capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
        )
        armed.log_backward(armed[armed.output_layers[0]].out.sum(), retain_graph=True)
    grad_fns = {g.label: g for g in armed.grad_fns}
    selector = CompositeSelector("and", (tl.grad_input(), tl.func("relu"), tl.contains("relu")))
    grads = (torch.ones(1),)
    matches = {
        label: live_backward_selector_matches(
            selector, site, 1, grad_input=grads, grad_output=grads
        )
        for label, site in grad_fns.items()
    }
    assert any(matches[label] for label in matches if label.startswith("relu_back"))
    assert not any(matches[label] for label in matches if label.startswith("sum_back"))


def test_flat_nary_followed_by_conjunction_capture() -> None:
    """FIX: a flat ``(candidate, candidate, followed_by)`` conjunction crashed.

    The supported sugar generalizes: the candidate is the conjunction of every
    non-``followed_by`` child. The flat spelling must save the same set as the
    nested binary spelling.
    """

    flat = CompositeSelector(
        "and",
        (tl.func("conv2d"), tl.contains("conv"), tl.followed_by(tl.func("relu"))),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat_log = tl.trace(
            TinyConvNet(),
            _conv_input(),
            save=flat,
            lookback=4,
            lookback_payload_policy="detached_raw",
        )
        binary_log = tl.trace(
            TinyConvNet(),
            _conv_input(),
            save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
            lookback=4,
            lookback_payload_policy="detached_raw",
        )

    def _saved(log: Any) -> list[str]:
        return sorted(
            str(op.layer_label)
            for op in log.layer_list
            if getattr(op, "has_saved_activation", False)
        )

    assert _saved(flat_log) == _saved(binary_log)
    assert _saved(flat_log), "expected the conv-before-relu candidates to be saved"


# ---------------------------------------------------------------------------
# opus F-2 — raw-label substring widening (FIX)
# ---------------------------------------------------------------------------


def test_contains_raw_only_substring_matches_nothing_posthoc(conv_trace: Any) -> None:
    """FIX: ``contains("_raw")`` matched every op via ``_layer_label_raw``."""

    assert conv_trace.find_sites(tl.contains("_raw"), max_fanout=10**6).labels() == ()


def test_regex_raw_suffix_matches_nothing_posthoc(conv_trace: Any) -> None:
    """FIX: ``regex(r"_raw$")`` matched every op via ``_layer_label_raw``."""

    assert conv_trace.find_sites(tl.regex(r"_raw$"), max_fanout=10**6).labels() == ()


def test_exact_raw_label_still_matches_posthoc(conv_trace: Any) -> None:
    """Exact ``tl.label`` keeps the wide universe (raw spelling included)."""

    assert conv_trace.find_sites(tl.label("relu_1_3_raw"), max_fanout=10**6).labels() == (
        "relu_1_2",
    )


# ---------------------------------------------------------------------------
# sol#3 — post-hoc composite short-circuit semantics (PIN, enumerated change)
# ---------------------------------------------------------------------------


def test_posthoc_and_short_circuits_where_predicate(conv_trace: Any) -> None:
    """PIN: ``a & tl.where(p)`` evaluates ``p`` only on sites matching ``a``.

    Enumerated behavior change: the old post-hoc resolver evaluated every
    branch over the full site set; the unified interpreter short-circuits
    per site in every lifecycle. ``tl.where`` predicates must not rely on
    being invoked for non-matching sites.
    """

    calls: list[str] = []

    def _pred(p: Any) -> bool:
        calls.append(str(getattr(p, "layer_label", "?")))
        return True

    labels = conv_trace.find_sites(tl.func("relu") & tl.where(_pred), max_fanout=10**6).labels()
    assert labels == ("relu_1_2", "relu_2_4")
    assert sorted(calls) == ["relu_1_2", "relu_2_4"]


def test_posthoc_or_short_circuits_where_predicate(conv_trace: Any) -> None:
    """PIN: ``a | tl.where(p)`` skips ``p`` on sites already matching ``a``."""

    calls: list[str] = []

    def _pred(p: Any) -> bool:
        calls.append(str(getattr(p, "layer_label", "?")))
        return False

    labels = conv_trace.find_sites(tl.func("relu") | tl.where(_pred), max_fanout=10**6).labels()
    assert labels == ("relu_1_2", "relu_2_4")
    assert "relu_1_2" not in calls and "relu_2_4" not in calls
    assert calls, "predicate should still run on non-relu sites"


# ---------------------------------------------------------------------------
# sol#4 — live capability validation must not silently complete (FIX)
# ---------------------------------------------------------------------------


def _noop_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    return out


def test_live_followed_by_refused_under_nonmatching_and() -> None:
    """FIX: ``func("never") & followed_by(...)`` completed silently at live."""

    with pytest.raises(SelectorCapabilityError, match="retroactive"):
        tl.trace(
            TinyConvNet(),
            _conv_input(),
            capture=tl.options.CaptureOptions(
                hooks=[(tl.func("zzz_never") & tl.followed_by(tl.func("relu")), _noop_hook)]
            ),
        )


def test_live_preceded_by_refused_under_nonmatching_and() -> None:
    """FIX: ``func("never") & preceded_by(...)`` completed silently at live."""

    with pytest.raises(SelectorCapabilityError, match="lookback"):
        tl.trace(
            TinyConvNet(),
            _conv_input(),
            capture=tl.options.CaptureOptions(
                hooks=[(tl.func("zzz_never") & tl.preceded_by(tl.func("relu")), _noop_hook)]
            ),
        )


def test_live_capability_refusal_is_site_resolution_subclass() -> None:
    """The live refusal stays catchable as ``SiteResolutionError``."""

    with pytest.raises(SiteResolutionError):
        tl.trace(
            TinyConvNet(),
            _conv_input(),
            capture=tl.options.CaptureOptions(
                hooks=[(tl.followed_by(tl.func("relu")), _noop_hook)]
            ),
        )


# ---------------------------------------------------------------------------
# sol#5 — capture-lifecycle spec rebuild + facet module_address (FIX)
# ---------------------------------------------------------------------------


def test_followed_by_spec_rebuilds_for_capture_lifecycle() -> None:
    """FIX: capture-lifecycle deserialization refused its own window sugar."""

    spec = tl.followed_by(tl.func("relu")).to_target_spec()
    rebuilt = selector_from_spec(
        spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="capture"
    )
    assert repr(rebuilt) == repr(tl.followed_by(tl.func("relu")))


def test_preceded_by_spec_rebuilds_for_capture_lifecycle() -> None:
    """FIX: capture-lifecycle deserialization refused the lookback selector."""

    spec = tl.preceded_by(tl.func("conv2d")).to_target_spec()
    rebuilt = selector_from_spec(
        spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="capture"
    )
    assert repr(rebuilt) == repr(tl.preceded_by(tl.func("conv2d")))


@pytest.mark.parametrize("lifecycle", ["site", "live"])
def test_followed_by_spec_still_refused_outside_capture(lifecycle: str) -> None:
    """followed_by specs keep the typed refusal in non-capture lifecycles."""

    spec = tl.followed_by(tl.func("relu")).to_target_spec()
    with pytest.raises(SelectorCapabilityError):
        selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle=lifecycle
        )


@pytest.mark.parametrize(
    "make_selector",
    [
        lambda: tl.facet("q").head(3).in_module("encoder.block.0"),
        lambda: tl.facet("q").in_module("encoder.block.0"),
        lambda: tl.facet("q").head(3),
        lambda: tl.facet("q"),
        lambda: tl.head(2),
    ],
)
def test_facet_spec_roundtrip_is_lossless(make_selector: Any) -> None:
    """FIX: facet spec rebuild dropped ``module_address``."""

    selector = make_selector()
    spec = selector.to_target_spec()
    rebuilt = selector_from_spec(
        spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="site"
    )
    assert repr(rebuilt) == repr(selector)
    assert getattr(rebuilt, "module_address", None) == getattr(selector, "module_address", None)


# ---------------------------------------------------------------------------
# opus F-6 item 11 — the capture-only refusal message is user-visible (PIN)
# ---------------------------------------------------------------------------


def test_followed_by_posthoc_refusal_message(conv_trace: Any) -> None:
    """The helpful capture-only message is the deliverable of the refusal."""

    with pytest.raises(SelectorCapabilityError) as excinfo:
        conv_trace.find_sites(tl.func("conv2d") & tl.followed_by(tl.func("relu")), max_fanout=10**6)
    message = str(excinfo.value)
    assert "capture-time-only retroactive save sugar" in message
    assert "no retroactive window" in message


# ---------------------------------------------------------------------------
# opus F-7 — ensure_supported refusal set matches the evaluator (contract)
# ---------------------------------------------------------------------------

_KIND_PAYLOADS: dict[str, Any] = {
    "label": "x",
    "func": "x",
    "func_transform": None,
    "module": "x",
    "output": 0,
    "output_at": (0,),
    "input_at": (0,),
    "contains": "x",
    "predicate": lambda p: False,
    "in_module": "x",
    "facet": {"name": "q"},
    "grad_fn": {"type": None},
    "grad_fn_label": "x",
    "grad_kind": "grad_input",
    "backward_pass": 1,
    "intervening": None,
    "without_op": None,
    "regex": "x",
    "followed_by": None,
    "preceded_by": None,
}


def test_site_upfront_refusal_set_matches_evaluator_contract() -> None:
    """Every kind the site evaluator refuses must refuse upfront, and only those.

    Guards the hand-maintained set in ``ensure_supported`` against new
    ``SelectorKind`` members: a kind that raises ``SelectorCapabilityError``
    inside per-site evaluation but not upfront would hide its refusal behind a
    short-circuiting sibling.
    """

    from torchlens.ir.selector_eval import _evaluate_subject

    site = SimpleNamespace(
        layer_label="relu_1_1", lookup_keys=(), func_name="relu", layer_type="relu"
    )
    for kind in get_args(SelectorKind):
        if kind in {"and", "or", "not"}:
            continue
        assert kind in _KIND_PAYLOADS, f"new SelectorKind {kind!r}: extend this contract test"
        selector = BaseSelector(kind, _KIND_PAYLOADS[kind])
        evaluator_refuses = False
        try:
            _evaluate_subject(selector, site, "site")
        except SelectorCapabilityError:
            evaluator_refuses = True
        except Exception:  # noqa: BLE001 - payload-shape errors are not refusals
            pass
        upfront_refuses = False
        try:
            ensure_supported(selector, lifecycle="site")
        except SelectorCapabilityError:
            upfront_refuses = True
        assert evaluator_refuses == upfront_refuses, (
            f"{kind!r}: evaluator refusal ({evaluator_refuses}) and upfront "
            f"ensure_supported refusal ({upfront_refuses}) disagree"
        )


# ---------------------------------------------------------------------------
# sol#7 — record(save=) signature (PIN, enumerated change)
# ---------------------------------------------------------------------------


def test_record_save_default_is_none() -> None:
    """PIN: ``record(save=...)`` defaults to ``None`` after the alias removal.

    Enumerated signature change: the ``MISSING`` sentinel existed only to
    arbitrate between ``save=`` and the removed ``keep_op=`` alias; with one
    spelling left, omitted and ``None`` are equivalent.
    """

    assert inspect.signature(tl.record).parameters["save"].default is None


# ---------------------------------------------------------------------------
# opus F-6 items 6/10 — bare-string normalization + non-torch fallback (PIN)
# ---------------------------------------------------------------------------


def test_bare_string_posthoc_is_substring_search(conv_trace: Any) -> None:
    """Post-hoc bare strings keep the historical substring contract."""

    assert conv_trace.find_sites("relu", max_fanout=10**6).labels() == (
        "relu_1_2",
        "relu_2_4",
    )


def test_bare_string_live_is_exact_raw_label() -> None:
    """Live bare strings keep the historical exact-label contract."""

    fired: list[str] = []

    def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fired.append(str(hook.layer_log.get("layer_label")))
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.trace(
            TinyConvNet(),
            _conv_input(),
            capture=tl.options.CaptureOptions(hooks=[("relu_1_3_raw", _hook)]),
        )
    assert fired == ["relu_1_3_raw"]


def test_strict_rejects_nested_where(conv_trace: Any) -> None:
    """strict=True refuses ``tl.where`` anywhere in the tree (walk sweep)."""

    with pytest.raises(SiteResolutionError, match="non-portable"):
        conv_trace.find_sites(
            tl.func("relu") & tl.where(lambda p: True), strict=True, max_fanout=10**6
        )


def test_capture_module_fallback_for_nontorch_backend() -> None:
    """Non-torch capture subjects fall back to the single ``module`` candidate."""

    from torchlens.ir.selector_eval import evaluate

    subject = SimpleNamespace(
        output_of_module_calls=(),
        source_trace=SimpleNamespace(backend="mlx"),
        module="encoder.block",
        func_name=None,
        layer_type=None,
    )
    assert evaluate(tl.module("encoder.block"), subject, lifecycle="capture")
    assert not evaluate(tl.module("decoder"), subject, lifecycle="capture")


# ---------------------------------------------------------------------------
# sol closure round — final three temporal-edge residuals (FIX)
# ---------------------------------------------------------------------------


def test_followed_by_conjunction_is_association_insensitive() -> None:
    """FIX: ``a & followed_by(x) & b`` raised while ``a & b & followed_by(x)`` worked.

    ``&`` builds nested binary composites, so the one-followed_by check and the
    retroactive split must normalize incrementally built conjunctions: every
    association of the same conjuncts saves the same set as the flat spec.
    """

    def _mid() -> BaseSelector:
        return tl.func("conv2d") & tl.followed_by(tl.func("relu")) & tl.contains("conv")

    def _tail() -> BaseSelector:
        return tl.func("conv2d") & tl.contains("conv") & tl.followed_by(tl.func("relu"))

    def _flat() -> BaseSelector:
        return CompositeSelector(
            "and",
            (tl.func("conv2d"), tl.contains("conv"), tl.followed_by(tl.func("relu"))),
        )

    def _saved(predicate: BaseSelector) -> list[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log = tl.trace(
                TinyConvNet(),
                _conv_input(),
                save=predicate,
                lookback=4,
                lookback_payload_policy="detached_raw",
            )
        return sorted(
            str(op.layer_label)
            for op in log.layer_list
            if getattr(op, "has_saved_activation", False)
        )

    mid_saved = _saved(_mid())
    assert mid_saved == _saved(_tail()) == _saved(_flat())
    assert mid_saved, "expected the conv-before-relu candidates to be saved"


def test_composite_degenerate_arity_eval_and_roundtrip_agree(conv_trace: Any) -> None:
    """FIX: empty/unary composites evaluated but refused their own spec round-trip.

    The one truth is the standard identity semantics: empty ``and`` matches
    everything, empty ``or`` matches nothing, a unary composite matches exactly
    like its child — and the target-spec round-trip preserves that behavior.
    """

    empty_and = CompositeSelector("and", ())
    empty_or = CompositeSelector("or", ())
    unary_or = CompositeSelector("or", (tl.func("relu"),))

    all_labels = conv_trace.find_sites(empty_and, max_fanout=10**6).labels()
    assert all_labels, "empty and is the identity: it matches every site"
    assert conv_trace.find_sites(empty_or, max_fanout=10**6).labels() == ()
    unary_labels = conv_trace.find_sites(unary_or, max_fanout=10**6).labels()
    assert unary_labels == conv_trace.find_sites(tl.func("relu"), max_fanout=10**6).labels()

    for selector, expected in (
        (empty_and, all_labels),
        (empty_or, ()),
        (unary_or, unary_labels),
    ):
        spec = selector.to_target_spec()
        rebuilt = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="site"
        )
        assert conv_trace.find_sites(rebuilt, max_fanout=10**6).labels() == expected


def test_followed_by_target_spec_serializes_structurally() -> None:
    """FIX: portable JSON refused ``followed_by(func(x))`` as an opaque callable
    and audit JSON silently mangled it into ``followed_by(contains(repr))``.

    A selector inner must round-trip structurally at every save level.
    """

    import json

    from torchlens.intervention.save import (
        SaveLevel,
        _target_spec_from_json,
        _target_spec_to_json,
    )

    for selector in (
        tl.followed_by(tl.func("relu")),
        tl.preceded_by(tl.func("conv2d")),
    ):
        spec = selector.to_target_spec()
        for level in (SaveLevel.PORTABLE, SaveLevel.AUDIT):
            payload = json.loads(json.dumps(_target_spec_to_json(spec, level)))
            rebuilt_spec = _target_spec_from_json(payload)
            rebuilt = selector_from_spec(
                rebuilt_spec.selector_kind,
                rebuilt_spec.selector_value,
                rebuilt_spec.metadata,
                lifecycle="capture",
            )
            assert repr(rebuilt) == repr(selector)


def test_followed_by_opaque_inner_refuses_typed_never_mangles() -> None:
    """FIX: an opaque-callable temporal inner must refuse typed, never rebuild
    as a ``contains`` selector over its repr string."""

    from torchlens.intervention.errors import OpaqueCallableInExecutableSaveError
    from torchlens.intervention.save import (
        SaveLevel,
        _target_spec_from_json,
        _target_spec_to_json,
    )

    spec = tl.followed_by(lambda ctx: True).to_target_spec()
    with pytest.raises(OpaqueCallableInExecutableSaveError):
        _target_spec_to_json(spec, SaveLevel.PORTABLE)
    audit_spec = _target_spec_from_json(_target_spec_to_json(spec, SaveLevel.AUDIT))
    with pytest.raises(SiteResolutionError, match="opaque audit payload"):
        selector_from_spec(
            audit_spec.selector_kind,
            audit_spec.selector_value,
            audit_spec.metadata,
            lifecycle="capture",
        )
