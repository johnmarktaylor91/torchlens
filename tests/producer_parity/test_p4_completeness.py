"""P4 completeness fixture: the live mutator sites against the closed registry.

Originally captured against the LIVE legacy ``replace_op_event`` sites BEFORE
any wiring or migration (DoR 4.2, review C7 / review 2 — that pre-migration form is
preserved at commit d4b1738d); re-targeted in P4c onto the amendment lane's
single chokepoint, ``CaptureEvents.append_amendment``. Four layers:

1. **Pair-wise identity** — every observed live amendment's patch paths are
   checked as ``(facet path, flat field)`` PAIRs against its family's registry
   row joined through ``PATH_TO_FLAT``, in registry order, and its call site
   must be the family's ONE intended site. A same-set identity permutation
   (the review's ``intervention_fired <-> intervention_replaced`` swap) is red.
2. **Emit-site value round-trip** — at every live call the folded journal
   record must carry each patch value at its own flat field AND (decomposed
   leg) at its own facet path, by object identity for non-interned values.
   The synthetic distinct-sentinel test below completes the value-level proof
   for interned (bool) paths with asymmetric patterns plus their complements.
3. **Reached-intended-family + observed >= 1** — each call site resolves to
   exactly its intended family (call-site identity recorded by the spy), and
   every torch registry family is observed at least once across the battery —
   zero observations for any family is red (the vacuity guard). The preview
   promotion families get their scenarios on the preview acceptance leg
   (skipped where the backend is not importable).
4. **Static AST guard** — no typed-constructor call site uses ``**`` expansion
   or a conditionally-present keyword, and the static site count matches the
   closed inventory — the property that makes the dynamic characterization a
   sound proof of the static set.

The spy NEVER raises inside the wrapper: a raised exception at a capture-time
site would be swallowed into failed-forward recovery (observed in P3), so
failures accumulate as strings and the test asserts at the end.
"""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.capture_events import CaptureEvents
from torchlens.ir.events import OpEvent
from torchlens.ir.op_record import (
    _FACET_ATTRIBUTES,
    AMENDMENT_FAMILIES,
    PATH_TO_FLAT,
    OpRecord,
    apply_patch_items,
)

pytestmark = pytest.mark.smoke

# Journal SHAPES for the fold-routing tests (P7 deleted the legacy torch
# producer): "decomposed" = captured OpRecord rows; "legacy" = genuine compat
# OpEvent rows (preview stand-in until S15), synthesized via the retained
# inverse adapter.
_LEGS = ("legacy", "decomposed")
_SEED = 20260812

# The seven torch families; the two preview families are characterized on the
# preview acceptance leg below.
#
# Reason-bearing retirement ledger: families that remain in the registry (the
# amendments lane still exercises them; artifact-safe — amendments are
# capture-session journal only) but deliberately have ZERO live emit sites.
# The vacuity guard skips exactly these; an unlisted dead family stays red.
_RETIRED_FAMILIES: dict[str, str] = {
    # 8858793e: a raw forward hook that recontainers an ALREADY-TRACED tensor
    # rewires the module boundary but does not replace the producing op's
    # value, so it must not mint replacement evidence that would exempt the
    # native op from replay validation (tripwire strengthening). The fresh-
    # tensor branch stamps intervention_replaced at record construction and
    # never used this amendment.
    "raw_hook_intervention": "emit site removed by 8858793e (recontainer hooks must not mint replacement evidence)",
}
_TORCH_FAMILIES = (
    frozenset(AMENDMENT_FAMILIES)
    - {
        "preview_output_parent_mark",
        "preview_output_parent_rebind",
    }
    - frozenset(_RETIRED_FAMILIES)
)

# (caller file basename, enclosing function) -> intended family. The spy
# resolves every observed call through this map; an unmapped caller is red.
# The torch sites plus the six preview promotion sites (which only fire on
# the preview acceptance leg).
_SITE_FAMILIES: dict[tuple[str, str], str] = {
    ("_ops_retention.py", "_replace_event_with_retained_payload"): "lookback_retention",
    ("user_funcs.py", "_register_live_tensor_connection"): "graph_edge_insertion",
    ("model_prep.py", "_record_module_exit_metadata"): "module_exit_intervention",
    (
        "model_prep.py",
        "_record_predicate_module_boundary_outputs",
    ): "module_boundary_retention",
    ("backend.py", "extract_and_mark_outputs"): "output_parent_promotion",
    ("graph_traversal.py", "_resolve_output_parent_labels"): "late_buffer_output_parent",
}

# Lane transport, not emit sites: concat re-appends already-emitted amendments
# when a per-pass journal merges into an accumulating one (the fastlog
# recorder path). Any family may legitimately pass through here.
_TRANSPORT_SITES: frozenset[tuple[str, str]] = frozenset({("capture_events.py", "concat")})


@dataclass
class _Observation:
    """One live amendment append recorded by the spy."""

    site: tuple[str, str]
    family: str
    kwarg_names: tuple[str, ...]
    label_raw: str
    record_type: str


@contextmanager
def _spy_append_amendment(observations: list[_Observation], failures: list[str]) -> Iterator[None]:
    """Wrap the ONE amendment chokepoint, ``CaptureEvents.append_amendment``."""

    real_append = CaptureEvents.append_amendment

    def wrapper(self: Any, amendment: Any) -> Any:
        frame = sys._getframe(1)
        site = (Path(frame.f_code.co_filename).name, frame.f_code.co_name)
        family = amendment.family
        label_raw = amendment.target_label_raw
        folded = real_append(self, amendment)
        observations.append(
            _Observation(
                site=site,
                family=family,
                kwarg_names=tuple(PATH_TO_FLAT[path] for path, _ in amendment.patch),
                label_raw=label_raw,
                record_type=type(folded).__name__,
            )
        )
        if site not in _TRANSPORT_SITES:
            expected_site_family = _SITE_FAMILIES.get(site)
            if expected_site_family != family:
                failures.append(
                    f"{site}: appended family {family!r} but the site map "
                    f"expects {expected_site_family!r} (reached-intended-family)"
                )
        view = self.amended_op_record(label_raw)
        for name in ("label_raw", "seq"):
            if getattr(view, name) != getattr(folded, name):
                failures.append(
                    f"{site}: folded journal view for {label_raw!r} disagrees "
                    f"with the appended fold on {name!r} (single-truth violation)"
                )
        for path, value in amendment.patch:
            flat = PATH_TO_FLAT[path]
            landed = getattr(folded, flat)
            identical = landed is value if not isinstance(value, bool) else landed == value
            if not identical:
                failures.append(
                    f"{site}: flat field {flat!r} does not carry the emitted "
                    f"value (got {landed!r}, sent {value!r})"
                )
            if isinstance(folded, OpRecord):
                owner, _, field_name = path.partition(".")
                holder = (
                    folded.core if owner == "core" else getattr(folded, _FACET_ATTRIBUTES[owner])
                )
                if holder is None:
                    failures.append(f"{site}: facet {owner!r} absent after folding {path!r}")
                    continue
                facet_value = getattr(holder, field_name)
                facet_ok = (
                    facet_value is value if not isinstance(value, bool) else facet_value == value
                )
                if not facet_ok:
                    failures.append(
                        f"{site}: facet path {path!r} does not carry the "
                        f"emitted value (got {facet_value!r}, sent {value!r})"
                    )
        return folded

    CaptureEvents.append_amendment = wrapper  # type: ignore[method-assign]
    try:
        yield
    finally:
        CaptureEvents.append_amendment = real_append  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Family scenario battery (each scenario is a fresh model + deterministic
# input + one capture call; the battery jointly reaches all seven families).
# ---------------------------------------------------------------------------


class _FCOnly(nn.Module):
    """Linear op IS the module output (module_exit_intervention carrier)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 31)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _BufferOut(nn.Module):
    """Returns an untouched registered buffer (late_buffer_output_parent)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 32)
        self.fc = nn.Linear(4, 4)
        self.register_buffer("gauge", torch.arange(4.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc(x), self.gauge


class _ManualEdge(nn.Module):
    """Calls the public register_tensor_connection (graph_edge_insertion)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 33)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc1(x)
        b = self.fc2(x)
        tl.register_tensor_connection(a, b)
        return a + b


class _Recurrent(nn.Module):
    """Linear cell applied twice with tanh (lookback_retention carrier)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 34)
        self.cell = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(2):
            h = torch.tanh(self.cell(h))
        return h


class _SmallCNN(nn.Module):
    """Conv + fc with module tree (module_boundary_retention carrier)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 35)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        return self.fc(h.flatten(1))


def _vec4() -> torch.Tensor:
    torch.manual_seed(_SEED + 40)
    return torch.randn(2, 4)


def _vec6() -> torch.Tensor:
    torch.manual_seed(_SEED + 41)
    return torch.randn(2, 6)


def _img() -> torch.Tensor:
    torch.manual_seed(_SEED + 42)
    return torch.randn(2, 1, 4, 4)


# name -> (build, capture) — the completeness battery. output_parent_promotion
# fires on every exhaustive/predicate capture; the named row keeps its intent
# explicit.
_BATTERY: tuple[tuple[str, Callable[[], tuple[nn.Module, Any]], Callable[..., Any]], ...] = (
    (
        "lookback_retention",
        lambda: (_Recurrent(), _vec6()),
        lambda m, x: tl.trace(
            m,
            x,
            save=tl.func("linear") & tl.followed_by(tl.func("tanh")),
            lookback=4,
            lookback_payload_policy="detached_raw",
        ),
    ),
    (
        "graph_edge_insertion",
        lambda: (_ManualEdge(), _vec4()),
        lambda m, x: tl.trace(m, x),
    ),
    (
        # A non-replacing module-boundary fire (tap) leaves live fire results
        # on a still-labeled output tensor — the only path into the module
        # exit handler's residual-fire branch (a replacing helper clears the
        # label and routes to the boundary-op channel instead).
        "module_exit_intervention",
        lambda: (_FCOnly(), _vec4()),
        lambda m, x: tl.trace(
            m,
            x,
            save=tl.func("linear"),
            intervene=tl.when(tl.module("fc"), tl.tap(lambda t: None)),
        ),
    ),
    (
        # Module-selector retention at module exit runs on the fastlog
        # recorder path (state.options.keep_op with kind "module"); a plain
        # predicate trace resolves the same selection elsewhere.
        "module_boundary_retention",
        lambda: (_SmallCNN(), _img()),
        lambda m, x: tl.record(m, x, save=tl.module("conv")),
    ),
    (
        "output_parent_promotion",
        lambda: (_SmallCNN(), _img()),
        lambda m, x: tl.trace(m, x, save=tl.func("relu")),
    ),
    (
        "late_buffer_output_parent",
        lambda: (_BufferOut(), _vec4()),
        lambda m, x: tl.trace(m, x),
    ),
)


def test_live_legacy_sites_completeness() -> None:
    """Layers 1-3: pair-wise identity, value round-trip, reached-family >= 1."""

    observations: list[_Observation] = []
    failures: list[str] = []
    with _spy_append_amendment(observations, failures):
        for _family, build, capture in _BATTERY:
            model, inputs = build()
            capture(model, inputs)

    for observation in observations:
        expected = tuple(PATH_TO_FLAT[path] for path, _ in AMENDMENT_FAMILIES[observation.family])
        assert observation.kwarg_names == expected, (
            f"{observation.site} ({observation.family}): observed kwargs "
            f"{observation.kwarg_names} != registry (path, flat) join {expected} "
            "(pair-wise, ordered)"
        )

    assert not failures, "emit-site round-trip failures:\n" + "\n".join(failures)

    observed_families = {o.family for o in observations}
    missing = _TORCH_FAMILIES - observed_families
    assert not missing, (
        f"vacuity guard: no scenario reached families {sorted(missing)} — a "
        "silently non-triggering scenario proves nothing"
    )

    wrong_shape = {(o.site, o.record_type) for o in observations if o.record_type != "OpRecord"}
    assert not wrong_shape, f"foreign record shapes in the journal: {wrong_shape}"


# ---------------------------------------------------------------------------
# Layer 2 completion: distinct-sentinel fold routing (both fold legs), with
# asymmetric bool patterns + complements for the interned paths.
# ---------------------------------------------------------------------------


def _journal_templates(monkeypatch: pytest.MonkeyPatch, producer: str) -> list[Any]:
    """Return raw journal entries from a tiny capture at the step-0 seam.

    ``producer`` selects the journal SHAPE; the ``"legacy"`` (OpEvent) shape
    is synthesized through the retained inverse adapter (S15 stand-in).
    """

    postprocess_module = importlib.import_module("torchlens.postprocess")
    materialize_module = importlib.import_module("torchlens.postprocess._materialize")
    original = materialize_module.materialize_from_events
    grabbed: list[Any] = []

    def observing(trace: Any, events: Any) -> None:
        if not grabbed:
            grabbed.extend(events.op_events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", observing)
    monkeypatch.setattr(materialize_module, "materialize_from_events", observing)
    model, inputs = _SmallCNN(), _img()
    tl.trace(model, inputs)
    assert grabbed, "journal interception grabbed no events"
    if producer == "legacy":
        from ._oracle_adapter import op_event_from_record

        return [op_event_from_record(entry) for entry in grabbed]
    return grabbed


def _sentinel_for(path: str, value_types: tuple[type, ...], template: Any, flip: bool) -> Any:
    """Build a per-path distinct, structurally valid sentinel value."""

    if value_types == (bool,):
        return flip
    current = getattr(template, PATH_TO_FLAT[path])
    if isinstance(current, dict):
        return {"__sentinel__": path}
    if isinstance(current, tuple) or value_types == (tuple,):
        return (f"__sentinel__{path}",)
    if current is not None:
        # A fresh copy of the current typed value is a distinct object with
        # the exact runtime type (OutputRef / CapturePolicy).
        return replace(current)
    return object()


_BOOL_PATTERNS = (False, True)


@pytest.mark.parametrize("producer", _LEGS)
def test_distinct_sentinel_fold_routing(monkeypatch: pytest.MonkeyPatch, producer: str) -> None:
    """Every registry path routes to its OWN destination on both fold legs.

    For each family, per-path distinct sentinels are applied through the ONE
    dual-leg fold primitive (``apply_patch_items``: PATH_TO_FLAT flat replace
    for compat ``OpEvent``s, facet replace for decomposed ``OpRecord``s). Each
    destination must carry exactly its sentinel and every other flat field
    must be untouched. Bool paths run BOTH asymmetric patterns, so a
    same-set permutation between two bool destinations is red.
    """

    templates = _journal_templates(monkeypatch, producer)
    template = templates[-1]
    flat_names = tuple(field.name for field in dataclass_fields(OpEvent))

    def _flat_read(record: Any, name: str) -> Any:
        try:
            return getattr(record, name)
        except AttributeError:
            return "__unreadable__"

    for family, schema in AMENDMENT_FAMILIES.items():
        bool_paths = [path for path, types in schema if types == (bool,)]
        patterns = _BOOL_PATTERNS if len(bool_paths) >= 1 else (False,)
        for pattern_start in patterns:
            flip = pattern_start
            sentinels: dict[str, Any] = {}
            for path, value_types in schema:
                sentinels[path] = _sentinel_for(path, value_types, template, flip)
                if value_types == (bool,):
                    flip = not flip  # asymmetric across the family's bool paths
            updates = {PATH_TO_FLAT[path]: value for path, value in sentinels.items()}
            before = {name: _flat_read(template, name) for name in flat_names}
            folded = apply_patch_items(template, tuple(sentinels.items()))
            for path, value in sentinels.items():
                flat = PATH_TO_FLAT[path]
                landed = getattr(folded, flat)
                ok = landed is value if not isinstance(value, bool) else landed == value
                assert ok, (
                    f"{family}: sentinel for {path!r} did not land at {flat!r} (got {landed!r})"
                )
                if isinstance(folded, OpRecord):
                    owner, _, field_name = path.partition(".")
                    holder = (
                        folded.core
                        if owner == "core"
                        else getattr(folded, _FACET_ATTRIBUTES[owner])
                    )
                    assert holder is not None, f"{family}: facet {owner!r} absent"
                    facet_value = getattr(holder, field_name)
                    facet_ok = (
                        facet_value is value
                        if not isinstance(value, bool)
                        else facet_value == value
                    )
                    assert facet_ok, f"{family}: sentinel for {path!r} not at its facet slot"
            touched = set(updates)
            for name in flat_names:
                if name in touched:
                    continue
                after = _flat_read(folded, name)
                assert after is before[name] or after == before[name], (
                    f"{family}: fold leaked into untouched flat field {name!r}"
                )


# ---------------------------------------------------------------------------
# Layer 4: static AST guard over the closed caller inventory.
# ---------------------------------------------------------------------------

# Names whose call sites must stay statically characterizable: the nine typed
# amendment constructors (post-migration; replace_op_event has zero production
# callers and dies in the P4 deletion step). Production sites only — the
# constructor definitions and tests are excluded by the file list.
_GUARDED_CALLEES = frozenset(
    {
        "amend_lookback_retention",
        "amend_graph_edge_insertion",
        # Retired by 8858793e, deliberately revived at ONE site by 56f8a29e:
        # the recontainer-hygiene skip swallowed genuine user raw-hook
        # replacements, so genuinely-new traced replacements mint the raw-hook
        # amendment again (evidence only; no replay exemption granted).
        "amend_raw_hook_intervention",
        "amend_module_exit_intervention",
        "amend_module_boundary_retention",
        "amend_output_parent_promotion",
        "amend_late_buffer_output_parent",
        "amend_preview_output_parent_mark",
        "amend_preview_output_parent_rebind",
        "replace_op_event",  # must stay at ZERO sites in these files
    }
)
# 7 torch sites (raw_hook_intervention's wrapped_hook site retired by
# 8858793e, then revived narrowly by 56f8a29e -- genuine user raw-hook
# replacements mint the amendment again) + 2 preview rebind sites (jax,
# tinygrad) + the ONE shared preview_output_parent_mark site in
# backends/_finalize.py (the tf x2 / mlx / paddle per-backend mark copies
# were hoisted there by R46-1, 5290caa7) + the tf module_exit_intervention
# site fire in backends/tf/interventions.py (typed amendment lane, ef08d2a9).
_EXPECTED_SITE_COUNT = 11


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_static_ast_guard_no_expansion_no_conditional_kwargs() -> None:
    """No ``**`` expansion / conditional keywords at any mutator call site."""

    package_root = Path(tl.__file__).parent
    relative_files = (
        "backends/torch/_ops_retention.py",
        "user_funcs.py",
        "backends/torch/model_prep.py",
        "backends/torch/backend.py",
        "postprocess/graph_traversal.py",
        "backends/tf/backend.py",
        "backends/tf/interventions.py",
        "backends/mlx/backend.py",
        "backends/paddle/backend.py",
        "backends/jax/backend.py",
        "backends/tinygrad/backend.py",
        "backends/_finalize.py",
    )
    sites: list[tuple[str, int]] = []
    for relative in relative_files:
        source_path = package_root / relative
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in _GUARDED_CALLEES:
                continue
            assert _call_name(node) != "replace_op_event", (
                f"{relative}:{node.lineno}: replace_op_event caller survived the P4 migration"
            )
            sites.append((relative, node.lineno))
            starred_kwargs = [kw for kw in node.keywords if kw.arg is None]
            assert not starred_kwargs, (
                f"{relative}:{node.lineno}: ** expansion at a mutator call "
                "site defeats the static completeness proof"
            )
            assert not any(isinstance(arg, ast.Starred) for arg in node.args), (
                f"{relative}:{node.lineno}: * expansion at a mutator call site"
            )
    assert len(sites) == _EXPECTED_SITE_COUNT, (
        f"closed mutator inventory drifted: expected {_EXPECTED_SITE_COUNT} "
        f"static sites, found {len(sites)}: {sites}"
    )


# Preview promotion families: their PATH_TO_FLAT fold routing is proven by
# test_distinct_sentinel_fold_routing above (it iterates the FULL registry,
# preview families included). Live-site scenarios (one static-output and one
# live-output promotion per affected backend) run on the preview acceptance
# leg in the P5 campaign — no preview backend is importable on this box, and
# a skipping placeholder here would prove nothing.
