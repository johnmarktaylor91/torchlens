"""Characterization matrix for the public selector/predicate surface.

This is the behavioral oracle for the ONE-predicate-interpreter consolidation:
every public selector spelling is evaluated through each user-reachable
lifecycle (capture-time ``save=``, post-hoc ``find_sites``, live hook matching,
live backward matching, and spec round-trip) against fixed models, and the
resulting match sets are snapshotted in
``tests/golden/selector_semantics_matrix.json``.

Beyond the per-spelling grid, dedicated sections pin the seams the 2026-08-11
dual review demanded the oracle be able to SEE: typed ``output_at`` /
``input_at`` container paths against dict/namedtuple models, ``torch.func``
transform boundary ops, live backward matching (including n-ary composites),
default-``max_fanout`` resolution for the broadened ``func`` kind, per-site
short-circuit call semantics for stateful ``tl.where`` predicates, flat n-ary
target specs, and rebuilt-spec evaluation (spec cells resolve the rebuilt
selector, not just its repr).

The golden encodes TODAY'S behavior, including known divergences between the
lifecycles (case sensitivity, label universes, error shapes). A diff against
this file is therefore a *behavior change*: each one must be either an
explicitly intended, enumerated change or a bug. Regenerate deliberately with::

    TORCHLENS_UPDATE_SELECTOR_MATRIX=1 pytest tests/test_selector_semantics_matrix.py

(a regeneration run reports SKIP, never green — re-run without the flag to
verify against the freshly written golden)

Cell values are either a sorted list of matched labels, ``"ERROR:<Class>"``,
or a small dict of named sub-results.

Governance adjudication (b10 R78 round-3): this golden is ENVIRONMENT-
INDEPENDENT by design and therefore deliberately NOT routed through the
``tests/_oracle_env.py`` env-fingerprint resolver. It encodes selector
BEHAVIOR — pure-structure JSON of matched label sets, error class names, and
spec round-trip shapes over torchlens-owned label vocabulary. No float
formatting, qualname, repr, or emitter bytes enter the record; if a torch
upgrade ever changes a matched label set, that is a REAL behavior change this
oracle exists to surface loudly, not environmental drift to key away. The
family is registered in the environment-independent ledger enforced by
``tests/test_golden_governance_lint.py``.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from _oracle_env import (  # noqa: E402 - tests/ is on sys.path under pytest
    GOLDEN_FLAG_PREFIXES,
    flag_armed,
    guard_wrap_state_for_golden_update,
    require_update_reason,
    write_provenance,
)
from torch import nn

import torchlens as tl
from torchlens.intervention.selectors import (
    BaseSelector,
    CompositeSelector,
    grad_fn_label,
)
from torchlens.intervention.types import TargetSpec

_GOLDEN_PATH = Path(__file__).parent / "golden" / "selector_semantics_matrix.json"
_UPDATE_ENV = "TORCHLENS_UPDATE_SELECTOR_MATRIX"
# Armed on the exact value "1" only: bool(environ.get(...)) armed regen on
# TORCHLENS_UPDATE_SELECTOR_MATRIX=0 (b10 R78 round-3).
_REGEN = flag_armed(os.environ, _UPDATE_ENV)
if os.environ.get("TL_SELECTOR_MATRIX_REGEN"):
    # One regen-flag convention repo-wide (b10 R78-8b): fail loudly instead of
    # silently ignoring the retired spelling.
    raise RuntimeError(
        "TL_SELECTOR_MATRIX_REGEN was renamed; use TORCHLENS_UPDATE_SELECTOR_MATRIX=1"
    )

# Markers are additive: a file-level smoke pytestmark would keep the heavy test
# in the `-m smoke` tier, so tier marks are applied per test instead.


class TinyConvNet(nn.Module):
    """Nested-module CNN with relu/conv/add/flatten/linear ops."""

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


class SplitNet(nn.Module):
    """Multi-output chunk op feeding mul/add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return a * 2 + b


class LoopNet(nn.Module):
    """One linear block called twice (recurrent module passes)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.block = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.block(x))
        x = torch.relu(self.block(x))
        return x


class DictOutNet(nn.Module):
    """Dict-plus-tuple output: exercises DictKey/TupleIndex output paths."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        h = self.lin(x)
        return {"logits": torch.relu(h), "aux": (h * 2, h + 1)}


class _Pair(NamedTuple):
    main: torch.Tensor
    extra: torch.Tensor


class NamedTupleNet(nn.Module):
    """NamedTuple output: exercises NamedField output paths."""

    def forward(self, x: torch.Tensor) -> _Pair:
        return _Pair(main=torch.relu(x), extra=x * 2)


class DictInNet(nn.Module):
    """Dict input: exercises MODEL_INPUT container paths for ``input_at``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)

    def forward(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.lin(d["a"]) + d["b"]


class VmapNet(nn.Module):
    """``torch.func.vmap`` boundary op: exercises ``func_transform``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.func.vmap(torch.sin)(x)
        return torch.relu(y)


def _model_and_input(model_key: str) -> tuple[nn.Module, Any]:
    torch.manual_seed(0)
    if model_key == "conv":
        return TinyConvNet(), torch.randn(1, 1, 8, 8)
    if model_key == "split":
        return SplitNet(), torch.randn(1, 4, 4, 4)
    if model_key == "loop":
        return LoopNet(), torch.randn(2, 4)
    if model_key == "dictout":
        return DictOutNet(), torch.randn(2, 4)
    if model_key == "ntout":
        return NamedTupleNet(), torch.randn(2, 4)
    if model_key == "dictin":
        return DictInNet(), [{"a": torch.randn(2, 4), "b": torch.randn(2, 4)}]
    if model_key == "vmap":
        return VmapNet(), torch.randn(3, 4)
    raise KeyError(model_key)


def _extra_trace_kwargs(model_key: str) -> dict[str, Any]:
    """Per-model capture options (input containers need explicit opt-in)."""

    if model_key == "dictin":
        return {"capture": tl.options.CaptureOptions(capture_container_structure=True)}
    return {}


@cache
def _full_trace(model_key: str) -> Any:
    model, x = _model_and_input(model_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x, **_extra_trace_kwargs(model_key))


@cache
def _backward_trace(model_key: str) -> Any:
    model, x = _model_and_input(model_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(
            model,
            x.requires_grad_(True),
            capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
        )
        log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    return log


def _error_cell(exc: BaseException) -> str:
    return f"ERROR:{type(exc).__name__}"


def _probe_capture(model_key: str, make_selector: Callable[[], Any]) -> Any:
    """Labels saved by ``tl.trace(model, x, save=selector)``."""

    model, x = _model_and_input(model_key)
    try:
        selector = make_selector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log = tl.trace(
                model,
                x,
                save=selector,
                lookback=4,
                lookback_payload_policy="detached_raw",
                **_extra_trace_kwargs(model_key),
            )
    except Exception as exc:  # noqa: BLE001 - characterization records error shape
        return _error_cell(exc)
    return sorted(
        str(op.layer_label) for op in log.layer_list if getattr(op, "has_saved_activation", False)
    )


def _probe_sites(model_key: str, make_selector: Callable[[], Any], *, backward: bool) -> Any:
    """Labels returned by ``find_sites`` against a cached full trace."""

    try:
        selector = make_selector()
        log = _backward_trace(model_key) if backward else _full_trace(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = log.find_sites(selector, max_fanout=10**6)
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(str(label) for label in table.labels())


def _probe_live(model_key: str, make_selector: Callable[[], Any]) -> Any:
    """Raw labels of sites where a live hook attached to ``selector`` fires."""

    model, x = _model_and_input(model_key)
    fired: list[str] = []

    def _probe_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fired.append(str(hook.layer_log.get("layer_label")))
        return out

    try:
        selector = make_selector()
        # Fold the per-model capture opt-ins into ONE CaptureOptions so the
        # hook wiring never collides with _extra_trace_kwargs' capture= key.
        extra = dict(_extra_trace_kwargs(model_key))
        base_capture = extra.pop("capture", None)
        capture_kwargs: dict[str, Any] = {"hooks": [(selector, _probe_hook)]}
        if base_capture is not None:
            capture_kwargs.update(
                {
                    name: value
                    for name, value in base_capture.as_dict().items()
                    if base_capture.is_field_explicit(name)
                }
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.trace(
                model,
                x,
                **extra,
                capture=tl.options.CaptureOptions(**capture_kwargs),
            )
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(fired)


def _probe_sites_default_fanout(model_key: str, make_selector: Callable[[], Any]) -> Any:
    """``find_sites`` at DEFAULT ``max_fanout`` (pins fanout/ambiguity behavior)."""

    try:
        selector = make_selector()
        log = _full_trace(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = log.find_sites(selector)
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(str(label) for label in table.labels())


def _probe_live_backward(make_selector: Callable[[], Any]) -> Any:
    """GradFn labels matched by the live backward matcher on the armed trace.

    Exercises ``live_backward_selector_matches`` (and therefore the live-only
    backward context matcher) per grad_fn with synthetic grad tuples.
    """

    from torchlens.intervention.hooks import live_backward_selector_matches

    try:
        selector = make_selector()
        log = _backward_trace("conv")
        grads = (torch.ones(1),)
        matched = []
        for site in log.grad_fns:
            if live_backward_selector_matches(
                selector, site, 1, grad_input=grads, grad_output=grads
            ):
                matched.append(str(site.label))
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(matched)


def _probe_where_calls(
    model_key: str, make_selector: Callable[[Callable[[Any], bool]], Any], *, result: bool
) -> Any:
    """Post-hoc labels plus the exact site set a ``tl.where`` predicate saw.

    Pins the enumerated per-site short-circuit semantics: composite branches
    are not evaluated over the full site set, so a stateful ``tl.where``
    predicate observes only the sites its siblings did not already decide.
    """

    calls: list[str] = []

    def _tracking_predicate(p: Any) -> bool:
        calls.append(str(getattr(p, "layer_label", "?")))
        return result

    try:
        selector = make_selector(_tracking_predicate)
        log = _full_trace(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = sorted(
                str(label) for label in log.find_sites(selector, max_fanout=10**6).labels()
            )
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return {"labels": labels, "where_saw": sorted(calls)}


def _probe_spec(
    make_selector: Callable[[], Any],
    *,
    model_key: str | None = None,
    backward: bool = False,
) -> Any:
    """Round-trip a selector through the spec deserializer, all lifecycles.

    Deserializer equivalence is pinned by MATCH SET, not repr alone: when the
    site-lifecycle rebuild succeeds and a model is given, the rebuilt selector
    is resolved via ``find_sites`` and the labels are stored in the cell.
    """

    from torchlens.intervention.selectors import _classify_selector_direction
    from torchlens.ir.selector_eval import selector_from_spec

    try:
        selector = make_selector()
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    if not isinstance(selector, BaseSelector):
        return f"NOT_A_SELECTOR:{type(selector).__name__}"
    cell: dict[str, Any] = {"kind": str(selector.selector_kind)}
    try:
        cell["direction"] = _classify_selector_direction(selector)
    except Exception as exc:  # noqa: BLE001
        cell["direction"] = _error_cell(exc)
    try:
        spec = selector.to_target_spec()
    except Exception as exc:  # noqa: BLE001
        cell["to_spec"] = _error_cell(exc)
        return cell
    cell["spec_kind"] = str(spec.selector_kind)
    try:
        rebuilt = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="site"
        )
        cell["resolver_repr"] = repr(rebuilt)
        cell["resolver_direction"] = _classify_selector_direction(rebuilt)
        if model_key is not None:
            cell["resolver_labels"] = _rebuilt_selector_labels(
                rebuilt, model_key, backward=backward
            )
    except Exception as exc:  # noqa: BLE001
        cell["resolver_repr"] = _error_cell(exc)
    try:
        rebuilt_live = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="live"
        )
        cell["hooks_repr"] = repr(rebuilt_live)
        cell["hooks_direction"] = _classify_selector_direction(rebuilt_live)
    except Exception as exc:  # noqa: BLE001
        cell["hooks_repr"] = _error_cell(exc)
    try:
        rebuilt_capture = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="capture"
        )
        cell["capture_repr"] = repr(rebuilt_capture)
    except Exception as exc:  # noqa: BLE001
        cell["capture_repr"] = _error_cell(exc)
    return cell


def _rebuilt_selector_labels(rebuilt: Any, model_key: str, *, backward: bool) -> Any:
    """Resolve a spec-rebuilt selector so equivalence is pinned by match set."""

    try:
        log = _backward_trace(model_key) if backward else _full_trace(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = log.find_sites(rebuilt, max_fanout=10**6)
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(str(label) for label in table.labels())


def _where_relu() -> Any:
    return tl.where(lambda p: getattr(p, "layer_type", None) == "relu", name_hint="type_relu")


#: (cell_name, model_key, selector factory). Every entry gets capture/sites/live
#: cells on its model plus one spec round-trip cell.
FORWARD_CASES: tuple[tuple[str, str, Callable[[], Any]], ...] = (
    ("label_final", "conv", lambda: tl.label("relu_1_2")),
    ("label_raw", "conv", lambda: tl.label("relu_1_3_raw")),
    ("contains_lower", "conv", lambda: tl.contains("relu")),
    ("contains_upper", "conv", lambda: tl.contains("RELU")),
    ("contains_mixed", "conv", lambda: tl.contains("Conv2d")),
    ("regex_lower", "conv", lambda: tl.regex(r"relu_\d")),
    ("regex_upper", "conv", lambda: tl.regex(r"RELU")),
    ("regex_anchored", "conv", lambda: tl.regex(r"^conv2d_2")),
    ("func_relu", "conv", lambda: tl.func("relu")),
    ("func_add_type", "conv", lambda: tl.func("add")),
    ("func_add_dunder", "conv", lambda: tl.func("__add__")),
    ("func_transform_any", "conv", lambda: tl.func_transform()),
    ("module_container", "conv", lambda: tl.module("features")),
    ("module_address", "conv", lambda: tl.module("features.0")),
    ("in_module_container", "conv", lambda: tl.in_module("features")),
    ("in_module_pass", "conv", lambda: tl.in_module("features:1")),
    ("output_index", "conv", lambda: tl.output(0)),
    ("output_at_0", "conv", lambda: tl.output_at((0,))),
    ("input_at_0", "conv", lambda: tl.input_at(0)),
    ("where_type_relu", "conv", _where_relu),
    ("and_func_inmodule", "conv", lambda: tl.func("relu") & tl.in_module("features")),
    ("or_funcs", "conv", lambda: tl.func("relu") | tl.func("conv2d")),
    ("not_func", "conv", lambda: ~tl.func("relu")),
    (
        "and_nested_three",
        "conv",
        lambda: (tl.func("relu") & tl.in_module("features")) & tl.contains("relu"),
    ),
    (
        "followed_by_combo",
        "conv",
        lambda: tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    ),
    ("followed_by_bare", "conv", lambda: tl.followed_by(tl.func("relu"))),
    ("preceded_by_conv", "conv", lambda: tl.preceded_by(tl.func("conv2d"))),
    ("facet_named", "conv", lambda: tl.facet("q")),
    ("head_indexed", "conv", lambda: tl.head(0)),
    ("grad_fn_on_forward", "conv", lambda: tl.grad_fn("ReluBackward0")),
    ("split_output_0", "split", lambda: tl.output(0)),
    ("split_output_1", "split", lambda: tl.output(1)),
    ("split_func_chunk_out1", "split", lambda: tl.func("chunk", output=1)),
    ("split_contains_mul", "split", lambda: tl.contains("mul")),
    ("loop_in_module_block", "loop", lambda: tl.in_module("block")),
    ("loop_in_module_pass2", "loop", lambda: tl.in_module("block:2")),
    ("loop_module_block", "loop", lambda: tl.module("block")),
    ("loop_module_pass2", "loop", lambda: tl.module("block:2")),
    ("loop_label_recurrent", "loop", lambda: tl.label("linear_1_1")),
    # Raw/short spellings live only in the exact-label universe post-hoc:
    # substring/regex over them would make contains("raw") match every op.
    ("contains_raw_sub", "conv", lambda: tl.contains("_raw")),
    ("regex_raw_anchor", "conv", lambda: tl.regex(r"_raw$")),
    # Typed output-path components against real container outputs.
    ("dict_output_at_logits", "dictout", lambda: tl.output_at(("logits",))),
    ("dict_output_at_aux0", "dictout", lambda: tl.output_at(("aux", 0))),
    ("dict_output_at_aux1", "dictout", lambda: tl.output_at(("aux", 1))),
    ("dict_output_name", "dictout", lambda: tl.output("logits")),
    ("nt_output_at_main", "ntout", lambda: tl.output_at(("main",))),
    ("nt_output_at_extra", "ntout", lambda: tl.output_at(("extra",))),
    # MODEL_INPUT container paths (positive input_at cells).
    ("dictin_input_at_a", "dictin", lambda: tl.input_at("a")),
    ("dictin_input_at_b", "dictin", lambda: tl.input_at("b")),
    ("dictin_input_at_missing", "dictin", lambda: tl.input_at("zzz")),
    # torch.func transform boundary ops (positive func_transform cells).
    ("vmap_transform_any", "vmap", lambda: tl.func_transform()),
    ("vmap_transform_vmap", "vmap", lambda: tl.func_transform("vmap")),
    ("vmap_transform_grad", "vmap", lambda: tl.func_transform("grad")),
)

#: Post-hoc resolution at DEFAULT max_fanout (F-3: the broadened ``func`` kind
#: must be characterized where SiteAmbiguityError can actually fire).
DEFAULT_FANOUT_CASES: tuple[tuple[str, str, Callable[[], Any]], ...] = (
    ("func_relu", "conv", lambda: tl.func("relu")),
    ("func_conv2d", "conv", lambda: tl.func("conv2d")),
    ("func_add_type", "conv", lambda: tl.func("add")),
    ("contains_relu", "conv", lambda: tl.contains("relu")),
    ("not_func_missing", "conv", lambda: ~tl.func("zzz_missing")),
)

#: Live BACKWARD matching against the armed conv trace (the second-interpreter
#: seam: composite handling here crashed on n-ary children pre-fix).
LIVE_BACKWARD_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("grad_input", lambda: tl.grad_input()),
    ("grad_output", lambda: tl.grad_output()),
    ("grad_fn_class", lambda: tl.grad_fn("ReluBackward0")),
    ("without_op", lambda: tl.without_op()),
    ("func_relu_bridge", lambda: tl.func("relu")),
    ("bwd_and_binary", lambda: tl.grad_fn("ReluBackward0") & tl.grad_input()),
    (
        "bwd_and_nary_flat",
        lambda: CompositeSelector("and", (tl.grad_input(), tl.func("relu"), tl.contains("relu"))),
    ),
    ("bwd_not", lambda: ~tl.grad_input()),
    ("bwd_label_finalized", lambda: tl.label("relu_back_1_9")),
)

#: Stateful tl.where under composition: pins the enumerated per-site
#: short-circuit semantics (branches never evaluate over the full site set).
SHORT_CIRCUIT_CASES: tuple[tuple[str, str, Callable[[Callable[[Any], bool]], Any], bool], ...] = (
    ("and_where", "conv", lambda pred: tl.func("relu") & tl.where(pred), True),
    ("or_where", "conv", lambda pred: tl.func("relu") | tl.where(pred), False),
    ("not_where", "conv", lambda pred: ~tl.where(pred), False),
    (
        "and_where_left",
        "conv",
        lambda pred: tl.where(pred) & tl.func("relu"),
        True,
    ),
)

#: Flat n-ary target specs (deserialized shape unreachable via ``&``/``|``).
FLAT_SPEC_CASES: tuple[tuple[str, str, Callable[[], Any]], ...] = (
    (
        "flat_and_three",
        "conv",
        lambda: TargetSpec(
            selector_kind="and",
            selector_value=(
                TargetSpec("func", "relu"),
                TargetSpec("in_module", "features"),
                TargetSpec("contains", "relu"),
            ),
        ),
    ),
    (
        "flat_or_three",
        "conv",
        lambda: TargetSpec(
            selector_kind="or",
            selector_value=(
                TargetSpec("func", "relu"),
                TargetSpec("func", "conv2d"),
                TargetSpec("func", "flatten"),
            ),
        ),
    ),
    (
        "flat_and_single_child",
        "conv",
        lambda: TargetSpec(selector_kind="and", selector_value=(TargetSpec("func", "relu"),)),
    ),
)

#: Scoped facet chains: spec round-trips must not drop ``module_address``.
FACET_SPEC_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("facet_head_scoped", lambda: tl.facet("q").head(3).in_module("encoder.block.0")),
    ("facet_scoped", lambda: tl.facet("q").in_module("encoder.block.0")),
    ("facet_head", lambda: tl.facet("q").head(3)),
)

#: Backward selectors probed with find_sites against the armed conv trace.
BACKWARD_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("grad_fn_class", lambda: tl.grad_fn("ReluBackward0")),
    ("grad_fn_type", lambda: tl.grad_fn("relu")),
    ("grad_fn_label_pattern", lambda: tl.grad_fn(label="relu_back")),
    ("grad_fn_not_custom", lambda: tl.grad_fn(is_custom=False)),
    ("grad_fn_label_exact", lambda: grad_fn_label("relu_back_1_9")),
    ("without_op", lambda: tl.without_op()),
    ("grad_input", lambda: tl.grad_input()),
    ("grad_output", lambda: tl.grad_output()),
    ("backward_pass_1", lambda: tl.in_backward_pass(1)),
    ("backward_pass_2", lambda: tl.in_backward_pass(2)),
    (
        "bwd_and_direction_agnostic",
        lambda: tl.grad_fn("ReluBackward0") & tl.contains("relu"),
    ),
    ("bwd_plain_label", lambda: tl.label("relu_back_1_9")),
    ("bwd_not_accumulate", lambda: ~tl.grad_fn("AccumulateGrad")),
    # Direction-agnostic container kinds must intersect with backward kinds
    # through the grad_fn boundary-alias bridge (deleted-code invariant).
    ("bwd_output_at_and_grad_input", lambda: tl.output_at((0,)) & tl.grad_input()),
    ("bwd_output0_and_grad_input", lambda: tl.output(0) & tl.grad_input()),
    ("bwd_input_at_and_grad_output", lambda: tl.input_at(0) & tl.grad_output()),
)

#: Backward composition against container/multi-output models: the
#: direction-agnostic bridge (paired op + grad_fn boundary aliases) must make
#: ``output(0) & grad_input()`` and ``output_at(path) & grad_input()``
#: NON-vacuously intersect (deleted-code invariant; conv-only cells are []).
BACKWARD_CONTAINER_CASES: tuple[tuple[str, str, Callable[[], Any]], ...] = (
    ("split_output0_and_grad_input", "split", lambda: tl.output(0) & tl.grad_input()),
    ("split_output1_and_grad_input", "split", lambda: tl.output(1) & tl.grad_input()),
    (
        "dictout_output_at_logits_and_grad_input",
        "dictout",
        lambda: tl.output_at(("logits",)) & tl.grad_input(),
    ),
    ("dictout_output_at_logits", "dictout", lambda: tl.output_at(("logits",))),
)

#: Selector compositions expected to be decided at construction time.
CONSTRUCT_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("cross_direction_and", lambda: tl.func("relu") & tl.grad_fn("ReluBackward0")),
    ("or_with_followed_by", lambda: tl.func("relu") | tl.followed_by(tl.func("relu"))),
    ("not_followed_by", lambda: ~tl.followed_by(tl.func("relu"))),
    ("double_followed_by", lambda: tl.followed_by(tl.func("a")) & tl.followed_by(tl.func("b"))),
    ("func_non_string", lambda: tl.func(torch.relu)),  # type: ignore[arg-type]
)


def _construct_cell(make_selector: Callable[[], Any]) -> Any:
    try:
        return repr(make_selector())
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)


def _compute_matrix() -> dict[str, Any]:
    if _REGEN:
        # Regeneration is in-process: refuse to generate golden bytes on a
        # torch already wrapped by earlier tests (SF-53), and require the
        # WHY before any capture runs.
        guard_wrap_state_for_golden_update(_UPDATE_ENV)
        require_update_reason(_UPDATE_ENV)
    matrix: dict[str, Any] = {}
    for name, model_key, factory in FORWARD_CASES:
        matrix[f"capture/{model_key}/{name}"] = _probe_capture(model_key, factory)
        matrix[f"sites/{model_key}/{name}"] = _probe_sites(model_key, factory, backward=False)
        matrix[f"live/{model_key}/{name}"] = _probe_live(model_key, factory)
        matrix[f"spec/{name}"] = _probe_spec(factory, model_key=model_key)
    for name, factory in BACKWARD_CASES:
        matrix[f"sites_bwd/conv/{name}"] = _probe_sites("conv", factory, backward=True)
        matrix[f"spec_bwd/{name}"] = _probe_spec(factory, model_key="conv", backward=True)
    for name, model_key, factory in BACKWARD_CONTAINER_CASES:
        matrix[f"sites_bwd/{model_key}/{name}"] = _probe_sites(model_key, factory, backward=True)
    for name, factory in CONSTRUCT_CASES:
        matrix[f"construct/{name}"] = _construct_cell(factory)
    for name, model_key, factory in DEFAULT_FANOUT_CASES:
        matrix[f"sites_default_fanout/{model_key}/{name}"] = _probe_sites_default_fanout(
            model_key, factory
        )
    for name, factory in LIVE_BACKWARD_CASES:
        matrix[f"live_bwd/conv/{name}"] = _probe_live_backward(factory)
    for name, model_key, factory, result in SHORT_CIRCUIT_CASES:
        matrix[f"short_circuit/{model_key}/{name}"] = _probe_where_calls(
            model_key, factory, result=result
        )
    for name, model_key, factory in FLAT_SPEC_CASES:
        matrix[f"flat_spec/{model_key}/{name}"] = _probe_sites(model_key, factory, backward=False)
    for name, factory in FACET_SPEC_CASES:
        matrix[f"facet_spec/{name}"] = _probe_spec(factory)
    return matrix


@lru_cache(maxsize=1)
def _matrix() -> dict[str, Any]:
    return _compute_matrix()


def warm_scan_caches() -> None:
    """Pre-pay the one-time matrix build at collection time (uncharged).

    The conftest warm seam (``pytest_collection_modifyitems``) calls this for
    every collected module that exposes it. The 278-cell characterization
    matrix is ONE lru-cached build (~5-8s of genuine capture CPU); computed
    lazily it lands in whichever cell pytest-randomly happens to run first
    and sits ON the smoke duration-budget boundary (T45: two sessions tripped
    two DIFFERENT first-payer cells). Skipped whenever any golden
    update/regen flag is armed: golden generation guards that it starts on
    UNWRAPPED torch (SF-53), and this warm captures during collection --
    the same carve-out the conftest session-setup warmup capture takes.
    """

    if any(
        key.startswith(GOLDEN_FLAG_PREFIXES) and value == "1" for key, value in os.environ.items()
    ):
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _matrix()


_REGEN_WRITTEN = False


def _golden() -> dict[str, Any]:
    if _REGEN:
        # Write once, then SKIP every comparison: under the historical
        # behavior a regen run compared the matrix to itself and reported
        # green, blurring "verified" with "just rebaselined" (b10 R78-8d).
        global _REGEN_WRITTEN
        if not _REGEN_WRITTEN:
            _GOLDEN_PATH.write_text(json.dumps(_matrix(), indent=1, sort_keys=True) + "\n")
            write_provenance(
                _GOLDEN_PATH.parent,
                "tests/test_selector_semantics_matrix.py",
                _UPDATE_ENV,
                require_update_reason(_UPDATE_ENV),
            )
            _REGEN_WRITTEN = True
        pytest.skip(
            f"regenerated selector-semantics golden; re-run without {_UPDATE_ENV} to verify"
        )
    if not _GOLDEN_PATH.exists():
        pytest.fail(f"Missing golden {_GOLDEN_PATH}; regenerate with {_UPDATE_ENV}=1.")
    return _load_golden()


@lru_cache(maxsize=1)
def _load_golden() -> dict[str, Any]:
    return json.loads(_GOLDEN_PATH.read_text())


_CELL_KEYS: tuple[str, ...] = tuple(
    [
        f"{lifecycle}/{model_key}/{name}"
        for name, model_key, _ in FORWARD_CASES
        for lifecycle in ("capture", "sites", "live")
    ]
    + [f"spec/{name}" for name, _, _ in FORWARD_CASES]
    + [f"sites_bwd/conv/{name}" for name, _ in BACKWARD_CASES]
    + [f"sites_bwd/{model_key}/{name}" for name, model_key, _ in BACKWARD_CONTAINER_CASES]
    + [f"spec_bwd/{name}" for name, _ in BACKWARD_CASES]
    + [f"construct/{name}" for name, _ in CONSTRUCT_CASES]
    + [f"sites_default_fanout/{model_key}/{name}" for name, model_key, _ in DEFAULT_FANOUT_CASES]
    + [f"live_bwd/conv/{name}" for name, _ in LIVE_BACKWARD_CASES]
    + [f"short_circuit/{model_key}/{name}" for name, model_key, _, _ in SHORT_CIRCUIT_CASES]
    + [f"flat_spec/{model_key}/{name}" for name, model_key, _ in FLAT_SPEC_CASES]
    + [f"facet_spec/{name}" for name, _ in FACET_SPEC_CASES]
)


@pytest.mark.heavy
def test_matrix_covers_golden_exactly() -> None:
    """The computed cell-key set and the golden's key set must be identical."""

    assert set(_matrix()) == set(_golden())


# Cells that outgrew the 7s smoke budget on the merged tree (measured at
# T46's landing gate: live/split/split_contains_mul 7.8s wall / 18.6s cpu)
# carry a heavy param-mark; the rest stay smoke.
_HEAVY_CELLS = {"live/split/split_contains_mul"}


@pytest.mark.parametrize(
    "cell_key",
    [
        pytest.param(k, marks=pytest.mark.heavy) if k in _HEAVY_CELLS else pytest.param(k)
        for k in _CELL_KEYS
    ],
)
def test_selector_semantics_cell(cell_key: str) -> None:
    """One selector x lifecycle cell matches its committed characterization."""

    golden = _golden()
    matrix = _matrix()
    assert cell_key in golden, f"cell {cell_key} missing from golden; regenerate deliberately"
    assert matrix[cell_key] == golden[cell_key], (
        f"Behavior change in {cell_key}: golden={golden[cell_key]!r} "
        f"current={matrix[cell_key]!r}. If intended, enumerate it in the "
        f"consolidation report and regenerate with {_UPDATE_ENV}=1."
    )
