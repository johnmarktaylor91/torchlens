"""Reachable-surface walk + 4-way classification gate (build item 2).

The cheapest row that would have caught the commissioning surface: the door
whose bugs commissioned the oracles panel was OUTSIDE ``__all__``, so every
``__all__``-rooted harness would never have tested it. The enumeration root
here is the walked reality; the declared ledger is a gated claim (D2), the
class layer gets closure by construction because ``Trace``/``Layer`` declare
nothing to lockstep against (D3), and the plants prove the gates can go red
(D7). The MODULE-layer baseline regenerates via ``python
tests/oracles/_regen.py`` -- the diff IS the public-surface change review.
The CLASS layer deliberately has NO frozen inventory: it asserts the closed
license rules of ``_surface.CLASS_MEMBER_LICENSES`` instead, because licenses
are local facts about committed source and therefore compose across sibling
branches (train T03 bounced when a frozen member inventory met a sibling
lane's legitimate schema-owner ClassVars).
"""

from __future__ import annotations

import csv
import types
from pathlib import Path

import pytest

import torchlens

from ._deprecation import load_deprecated_doors
from ._known_gaps import licensed_keys, load_known_gaps, partition_monotone
from ._surface import (
    CLASS_BODY,
    CLASS_MEMBER_LICENSES,
    CLASSIFICATIONS,
    DECLARED,
    INHERITED_EXTERNAL,
    INSTALLED_BY_PACKAGE,
    MODULE_ASSIGNED,
    UNDECLARED,
    UNLICENSED,
    surface_counts,
    walk_class_surface,
    walk_class_surface_classified,
    walk_module_surface,
)

pytestmark = pytest.mark.smoke

DATA_DIR = Path(__file__).resolve().parent / "data"

_REGEN_HINT = (
    "review the delta, then regenerate the baseline consciously: "
    "python tests/oracles/_regen.py (the diff is the surface-change review)"
)


def _deprecated_doors() -> frozenset[str]:
    """Return registered deprecated door names."""

    return frozenset(row.door.rsplit(".", 1)[-1] for row in load_deprecated_doors())


def _baseline_pairs(filename: str) -> frozenset[tuple[str, ...]]:
    """Load one baseline TSV as a frozen set of row tuples.

    Parameters
    ----------
    filename:
        Baseline under ``data/``.

    Returns
    -------
    frozenset[tuple[str, ...]]
        Row tuples (header excluded).
    """

    with (DATA_DIR / filename).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        return frozenset(tuple(row) for row in reader)


def test_module_surface_classification_matches_baseline() -> None:
    """Every module-layer reachable name classifies as committed; no drift.

    New deltas BLOCK from this gate's first merge (D25): a new reachable
    name fails here until classified consciously, and a vanished name fails
    until its baseline row is deleted.
    """

    live = frozenset(
        (entry.name, entry.classification)
        for entry in walk_module_surface(torchlens, _deprecated_doors())
    )
    baseline = _baseline_pairs("surface_module_classification.tsv")
    new = sorted(live - baseline)
    gone = sorted(baseline - live)
    assert not new, f"unclassified NEW reachable surface {new}; {_REGEN_HINT}"
    assert not gone, f"stale baseline rows (surface shrank) {gone}; {_REGEN_HINT}"


def test_undeclared_names_are_known_gap_licensed() -> None:
    """The 4-way gate: an undeclared reachable name needs a dated gap row."""

    entries = walk_module_surface(torchlens, _deprecated_doors())
    undeclared = frozenset(e.name for e in entries if e.classification == UNDECLARED)
    licensed = licensed_keys(load_known_gaps(), "surface_classification")
    new, stale = partition_monotone(undeclared, licensed)
    assert not new, (
        f"UNDECLARED reachable public names with no KNOWN-GAP row: {new} -- "
        "declare in __all__ consciously, underscore it, or add a dated owned "
        "gap row (D25; no grace for new surfaces)"
    )
    assert not stale, (
        f"KNOWN-GAP rows whose names are no longer undeclared: {stale} -- "
        "delete the rows; the manifest is monotone-down"
    )


def test_typing_leaks_stay_deleted() -> None:
    """Build item 1 stays done: the three typing leaks never come back."""

    for name in ("Any", "TYPE_CHECKING", "annotations"):
        assert not hasattr(torchlens, name), (
            f"typing-import leak {name!r} is reachable again (oracles fact 1/2 class)"
        )


def test_class_layer_every_member_is_licensed() -> None:
    """Class-layer closure by construction (D3): the RULES are the baseline.

    Undeclared classes carried MORE public surface than declared ones at the
    panel's SHA (fact 3). The gate asserts the closed CLASSIFICATION RULES,
    never a frozen member inventory: every reachable public member must carry
    a source-witnessed license (class body, static module assignment,
    package-defined installer, or external base). Licenses are LOCAL facts
    about committed source, so sibling branches that each consciously add
    class surface compose under merge -- the frozen inventory this replaces
    bounced train T03 on exactly that (mega/P05's schema-owner ClassVars).
    Runtime injections match no license and stay RED (the plants below).
    """

    rows = walk_class_surface_classified(torchlens)
    unlicensed = sorted((c, m) for c, m, lic in rows if lic == UNLICENSED)
    assert not unlicensed, (
        f"UNLICENSED class-layer public members {unlicensed[:20]} -- declare "
        "each in its class body, assign it statically at top level of the "
        "class's defining module, install it via a torchlens-defined "
        "descriptor/callable, or underscore it; runtime injection is never a "
        "licensed spelling"
    )
    assert {lic for _, _, lic in rows} <= set(CLASS_MEMBER_LICENSES)


def test_class_member_license_channels_are_alive() -> None:
    """Positive controls (D7): every license rule fires on a live exemplar.

    A classifier whose branch never fires is a dead measurement channel; pin
    one known member per licensed kind so a refactor that silently stops a
    rule from matching goes red here, not in production.
    """

    rows = {
        (cls_name, member): lic
        for cls_name, member, lic in walk_class_surface_classified(torchlens)
    }
    assert rows[("Recording", "to_trace")] == CLASS_BODY
    assert rows[("Trace", "FIELD_FORK_POLICY")] == MODULE_ASSIGNED
    assert rows[("Op", "gpu_kernels")] == INSTALLED_BY_PACKAGE
    assert rows[("Layer", "flops_forward")] == INSTALLED_BY_PACKAGE
    assert rows[("Flops", "bit_length")] == INHERITED_EXTERNAL


def test_class_walk_is_import_state_independent() -> None:
    """The class walk normalizes import state (train-bounce regression pin).

    ``torchlens.kernel_telemetry`` installs ``gpu_kernels`` on ``Op``/
    ``AtenOp`` at import time; without normalization the walk counted
    differently depending on which tests imported it first (green alone,
    red in a smoke session). The walk must always see the opt-in members,
    so the class-layer denominator is order-independent.
    """

    rows = frozenset(walk_class_surface(torchlens))
    assert ("Op", "gpu_kernels") in rows and ("AtenOp", "gpu_kernels") in rows, (
        "the class walk no longer normalizes import state; its denominator "
        "is test-order-dependent again (the T02 train-bounce failure mode)"
    )


def test_declared_gate_is_cited_not_forked() -> None:
    """Re-point law: the reachable gate CITES the declared ledger (item 2).

    The declared gates (tests/test_api_surface.py ``TARGET_ALL``,
    tests/test_docs_lockstep_names.py ``PUBLIC_SURFACE_SIZE``) keep the
    declared claim; this gate owns the reachable counts; one cites the
    other so the two can never fork silently.
    """

    from ._lints import declared_ledger_constant, declared_ledger_length

    entries = walk_module_surface(torchlens, _deprecated_doors())
    counts = surface_counts(entries)
    assert counts[DECLARED] == len(torchlens.__all__)
    assert declared_ledger_constant() == len(torchlens.__all__)
    assert declared_ledger_length() == len(torchlens.__all__)
    assert set(CLASSIFICATIONS) >= {e.classification for e in entries}


def test_reachable_counts_publish_with_counted_sets() -> None:
    """D8: no count publishes without its counted-set identity beside it."""

    from ._surface import CLASS_SURFACE_COUNTED_SET, MODULE_SURFACE_COUNTED_SET

    entries = walk_module_surface(torchlens, _deprecated_doors())
    counts = surface_counts(entries)
    class_rows = walk_class_surface(torchlens)
    assert counts["total"] == len(entries)
    assert counts["total"] >= len(torchlens.__all__)
    assert MODULE_SURFACE_COUNTED_SET and CLASS_SURFACE_COUNTED_SET
    print(
        f"reachable module-layer surface: {counts['total']} "
        f"[counted set: {MODULE_SURFACE_COUNTED_SET}]; "
        f"declared: {counts[DECLARED]}; "
        f"class-layer member rows: {len(class_rows)} "
        f"[counted set: {CLASS_SURFACE_COUNTED_SET}]"
    )


def test_plant_unlisted_reachable_callable_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANT (H0): an injected unlisted module-level callable is caught."""

    def planted_oracle_door() -> None:
        """Simulate a helper leaking onto the top level."""

    monkeypatch.setattr(torchlens, "planted_oracle_door", planted_oracle_door, raising=False)
    entries = walk_module_surface(torchlens, _deprecated_doors())
    classification = {e.name: e.classification for e in entries}
    assert classification.get("planted_oracle_door") == UNDECLARED
    live = frozenset((e.name, e.classification) for e in entries)
    baseline = _baseline_pairs("surface_module_classification.tsv")
    assert ("planted_oracle_door", UNDECLARED) in live - baseline, (
        "the planted unlisted callable did NOT register as a baseline delta; "
        "the surface gate is a dead tripwire"
    )


def test_plant_new_trace_member_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANT (H0): an injected public ``Trace`` method classifies UNLICENSED.

    The injected callable's ``__module__`` is this test module, not
    ``torchlens``, so no license rule matches -- proving a runtime injection
    cannot launder through the class it lands on.
    """

    monkeypatch.setattr(torchlens.Trace, "planted_oracle_member", lambda self: None, raising=False)
    rows = frozenset(walk_class_surface_classified(torchlens))
    assert ("Trace", "planted_oracle_member", UNLICENSED) in rows, (
        "the planted Trace member did NOT classify UNLICENSED; "
        "the class-layer license gate is a dead tripwire"
    )


def test_plant_new_trace_data_constant_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANT (H0): an injected public ``Trace`` DATA constant is caught too.

    Plain data attributes carry no ``__module__``, so they attribute to
    their type's module (``builtins``) -- a runtime-injected constant can
    never launder as ``installed_by_package``. Committed constants stay
    licensed because they are class-body or module-level assignments in
    committed source (``class_body`` / ``module_assigned``), which this
    injection is not.
    """

    monkeypatch.setattr(
        torchlens.Trace, "PLANTED_ORACLE_CONSTANT", frozenset({"planted"}), raising=False
    )
    rows = frozenset(walk_class_surface_classified(torchlens))
    assert ("Trace", "PLANTED_ORACLE_CONSTANT", UNLICENSED) in rows, (
        "the planted Trace data constant did NOT classify UNLICENSED; "
        "runtime data injections can launder past the class-layer gate"
    )


def test_walk_classifies_a_doctored_namespace_without_touching_torchlens() -> None:
    """The walkers take the namespace as an argument (plant isolation)."""

    doctored = types.ModuleType("doctored")
    doctored.__all__ = ["declared_thing"]
    doctored.declared_thing = object()
    doctored.stray_thing = lambda: None
    entries = walk_module_surface(doctored, frozenset())
    classification = {e.name: e.classification for e in entries}
    assert classification == {"declared_thing": DECLARED, "stray_thing": UNDECLARED}
