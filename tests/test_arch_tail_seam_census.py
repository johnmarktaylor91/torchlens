"""Seam-disposition census: memo 6.2 repairs, 6.3 doors, facet settlement (F35).

Items 15/19/21 in test form, analysis-class: one frozen disposition table
naming every existing seam the memo ruled on (6.2), every funded door
(6.3 / item 21), and the facet-settlement mechanics (item 15), each with
its live status. MET rows are probed structurally (the door exists and is
importable through its public spelling); OPEN rows are pinned the way the
relocation-trigger tests pin deferred moves -- the pin FAILS when the gap
closes, so the fix flips the row in the same change instead of rotting the
census. A census that cannot notice its own staleness is not a census.

Status vocabulary: MET (disposition executed), PARTIAL (door live, named
obligations open), OPEN (not built; the pinned trigger arms the flip).
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "torchlens"

#: The frozen census (memo 6.2 dispositions + 6.3 doors + item 15). The
#: third column names the evidence: a module/callable for MET, the open
#: obligation for PARTIAL/OPEN. Flip a row ONLY in the change that alters
#: the seam.
SEAM_DISPOSITIONS: dict[str, tuple[str, str]] = {
    # -- 6.2 existing seams -------------------------------------------------
    "backend_specs": ("MET", "torchlens.backends registry + BackendSpec provider contract"),
    "selection_term_resolvers": (
        "OPEN",
        "register_term_resolver still appends to a bare module list; the typed "
        "inspectable capability-declared registry promotion is unbuilt",
    ),
    "predicate_registry": (
        "PARTIAL",
        "register_predicate/coerce_predicate live under a lock; the "
        "inspection/snapshot/persistence obligations of 6.2 are open",
    ),
    "facet_recipes": (
        "MET",
        "metadata-only inventory + explicit activation (semantic/recipes; the "
        "autoloader defect class is dead, gate G2)",
    ),
    "autoroute": ("MET", "torchlens.autoroute._registry direction-specific doors"),
    "receptive_field_rules": ("MET", "register_rf_rule + epoch-backed invalidation"),
    "sidecar_families": ("MET", "L1 namespaced typed sidecar door (test_arch_spine_sidecar)"),
    "user_callable_identity_contract": (
        "OPEN",
        "per-seam gates exist (extract register_pure_module, netron key fn, "
        "themes node_spec_fn) but the ONE L0 callable-identity contract is unbuilt",
    ),
    # -- 6.3 / item 21 doors ------------------------------------------------
    "renderer_registry": ("MET", "capability-gated renderer door (test_arch_spine_registries)"),
    "export_target_registry": ("MET", "one L7 door, 17 builtins registered through it"),
    "transform_providers": ("MET", "torchlens.transforms._registry (transforms memo owns)"),
    "lens_skin_door": ("MET", "visualization.theme_registry + lenses roster (themes N12)"),
    "preprocessing_authorities": ("MET", "torchlens.preprocessing.register_authority_adapter"),
    "entry_point_activation": ("MET", "torchlens.ecosystem.plugins five-group consent law"),
    "edit_spec_providers": (
        "OPEN",
        "sequenced AFTER InterventionSpec per the memo; no provider door yet -- "
        "arbitrary callables remain the non-portable escape hatch",
    ),
    "read_policy_registry": (
        "OPEN",
        "item 15's L5 read-policy registry with structural defaults is unbuilt; "
        "reads still reach facet policy through the frozen resolver only",
    ),
    # -- item 15 facet settlement --------------------------------------------
    "facet_evidence_vocabulary": (
        "OPEN",
        "Facet/FacetSpec/FacetView still live at semantic/ (L6); the V3-gated "
        "L0/L1 relocation (settlement part 1) is unexecuted",
    ),
    "facet_deletability_contract": (
        "OPEN",
        "Module.facets defers `from ..semantic import FacetView`: engine-absent "
        "access raises ImportError today instead of returning typed-empty",
    ),
}


@functools.lru_cache(maxsize=1)
def _package_source() -> str:
    return "\n".join(
        path.read_text()
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def test_census_prints_the_disposition_table() -> None:
    lines = [
        f"{seam}: {status} -- {evidence}"
        for seam, (status, evidence) in sorted(SEAM_DISPOSITIONS.items())
    ]
    print("\nSEAM DISPOSITION CENSUS (memo 6.2/6.3/15):\n" + "\n".join(lines))
    counts = dict.fromkeys(("MET", "PARTIAL", "OPEN"), 0)
    for status, _ in SEAM_DISPOSITIONS.values():
        counts[status] += 1
    assert set(counts) == {"MET", "PARTIAL", "OPEN"}
    assert counts["MET"] >= 10, "MET rows vanished -- a door was deleted without a census flip"


def test_met_doors_are_importable_through_their_public_spellings() -> None:
    """Every MET row's door exists; a deleted door must flip its row."""

    probes: dict[str, tuple[str, str]] = {
        "backend_specs": ("torchlens.backends", "BackendSpec"),
        "facet_recipes": ("torchlens.semantic.recipes", "installed_recipe_providers"),
        "autoroute": ("torchlens.autoroute._registry", ""),
        "receptive_field_rules": ("torchlens.receptive_field._rules", "register_rf_rule"),
        "transform_providers": ("torchlens.transforms._registry", ""),
        "lens_skin_door": ("torchlens.visualization.theme_registry", ""),
        "preprocessing_authorities": ("torchlens.preprocessing", "register_authority_adapter"),
        "entry_point_activation": ("torchlens.ecosystem.plugins", "activate"),
    }
    failures = []
    for seam, (module_path, attr) in sorted(probes.items()):
        assert SEAM_DISPOSITIONS[seam][0] == "MET"
        try:
            module = importlib.import_module(module_path)
            if attr and not hasattr(module, attr):
                failures.append((seam, f"{module_path}.{attr} missing"))
        except Exception as exc:  # noqa: BLE001 -- census reports, never masks
            failures.append((seam, repr(exc)))
    assert failures == [], f"MET census rows whose doors are gone: {failures}"


def test_open_row_term_resolvers_trigger() -> None:
    """Armed flip trigger: the resolver list is still a bare append target."""

    from torchlens import selection

    resolvers = selection._TERM_RESOLVERS
    assert type(resolvers) is list, (
        "selection term resolvers are no longer a bare list -- the 6.2 promotion "
        "landed; flip SEAM_DISPOSITIONS['selection_term_resolvers'] to MET and "
        "replace this trigger with a capability-row probe in the same change"
    )


def test_open_row_facet_settlement_triggers() -> None:
    """Armed flip triggers for the two facet-settlement OPEN rows."""

    module_source = (PACKAGE_ROOT / "data_classes/module.py").read_text()
    assert "from ..semantic import FacetView" in module_source, (
        "Module.facets no longer defers into semantic/ -- settlement part 1 "
        "(vocabulary relocation) or the deletability contract landed; flip the "
        "facet census rows and add the typed-empty engine-absent probe here"
    )
    facets_home = PACKAGE_ROOT / "semantic/facets.py"
    assert facets_home.exists() and "class FacetSpec" in facets_home.read_text(), (
        "FacetSpec left semantic/facets.py -- execute the census flip: the "
        "evidence vocabulary found its L0/L1 home"
    )


def test_open_row_read_policy_and_edit_spec_doors_absent() -> None:
    """Armed flip triggers: landing either door must flip its census row."""

    source = _package_source()
    assert "def register_read_policy" not in source, (
        "a read-policy registry landed -- flip SEAM_DISPOSITIONS['read_policy_registry']"
    )
    assert "def register_edit_spec_provider" not in source, (
        "an edit-spec provider door landed -- flip SEAM_DISPOSITIONS['edit_spec_providers']"
    )
