"""Round-21 VS11 totality registry and host-attestation proofs."""

from __future__ import annotations

import ast
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pytest

from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests import test_round17_structural_inventories as structural
from menagerie.crawler.tests.test_round19_environment_authority_composition import (
    _run_host_denial_composition,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CRAWLER_ROOT = _REPOSITORY_ROOT / "menagerie/crawler"
_REGISTRY_PATH = _CRAWLER_ROOT / "conformance-round21.json"
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github/workflows/tests.yml"
_LINUX_NODES_PATH = _CRAWLER_ROOT / "tests/round21_linux_real_nodes.json"
_MACOS_NODES_PATH = _CRAWLER_ROOT / "tests/round21_macos_real_nodes.json"
_REVERSIONS_TOOL_PATH = _CRAWLER_ROOT / "tools/round21_reversions.py"
_PROOF_PREFIX = "menagerie/crawler/tests/"

P01 = (
    "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
    "test_round21_preclusion_real_v3_path_has_no_substitutable_fixture_edge"
)
P02 = (
    "menagerie/crawler/tests/test_round21_fingerprint_composition.py::"
    "test_round21_cheap_fingerprint_catches_stat_preserved_mutation_without_false_staling_clone"
)
P03 = (
    "menagerie/crawler/tests/test_round21_scale_composition.py::"
    "test_round21_pass_and_spawn_validation_walks_are_constant_bounded"
)
P07 = (
    "menagerie/crawler/tests/test_round21_transport_composition.py::"
    "test_round21_closed_transport_capability_awards_and_rejects_unlisted_library"
)
P08 = (
    "menagerie/crawler/tests/test_round21_cache_rebind_composition.py::"
    "test_round21_mismatched_rebind_preserves_active_authority_and_awards"
)
P09 = (
    "menagerie/crawler/tests/test_round21_ci_composition.py::"
    "test_round21_linux_committed_lock_provenance_awards_in_ci"
)
P10 = (
    "menagerie/crawler/tests/test_round21_ci_composition.py::"
    "test_round21_macos_committed_lock_seatbelt_award_and_denial"
)
P11 = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_conformance_registry_is_total_and_executed"
)
P11_CI = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_ci_attestations_cover_registry_without_skip"
)
P12_NONE = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[none]"
)
P12_STAT = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[statistical]"
)
P13 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_outside_selected_interpreter_is_rejected_at_binding"
)
P14 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
P15 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
P16 = (
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes"
)
P17 = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_unhashable_output_awards_runs_with_unverifiable_modes"
)
P18A = (
    "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
    "test_documented_dry_run_and_resume_use_real_environment"
)
P18B = (
    "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
    "test_dry_run_all_source_failure_is_acceptance_error"
)
P19 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_manifest_v3_rejects_changed_interpreter_association"
)
P20 = (P12_NONE, P12_STAT)
T01 = (
    "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
    "test_round21_tripwire_catches_python_evasion"
)
T02 = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_legacy_manifest_v1_is_quarantined_from_every_live_import_graph"
)
T03 = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_round21_verification_tree_walk_inventory_is_closed"
)
T04_LINUX = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_round21_linux_release_artifacts_and_provisioning_are_real"
)
T04_MACOS = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_round21_macos_release_artifacts_and_provisioning_are_real"
)
T05_LINUX = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_round21_linux_release_registry_is_exact"
)
T05_MACOS = (
    "menagerie/crawler/tests/test_round17_structural_inventories.py::"
    "test_round21_macos_release_registry_is_exact"
)
T06 = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_conformance_registry_is_total_and_executed"
)

ENVIRONMENT_IDS = frozenset(f"E{index:02d}" for index in range(1, 14))
SHUTDOWN_IDS = frozenset(f"S{index:02d}" for index in range(1, 14))
HANDOFF_IDS = frozenset(f"H{index:02d}" for index in range(1, 5))
EVASION_IDS = frozenset(
    {
        "EV-direct-patch",
        "EV-alias-patch",
        "EV-helper-indirection",
        "EV-decorator",
        "EV-assignment",
        "EV-dynamic-lookup",
        "EV-fake-environment-result",
        "EV-legacy-root",
        "EV-alternate-compiler",
        "EV-base-interpreter-argv",
        "EV-deleted-ci-node",
    }
)
PRESERVATION_IDS = frozenset(f"PRES{index:02d}" for index in range(1, 9))
DECISION_IDS = frozenset(f"DEC{index:02d}" for index in range(1, 13))
FINDING_IDS = frozenset(
    {
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "F07",
        "F08",
        "F09",
        "F10",
        "F11a",
        "F11b",
    }
)
INVARIANT_IDS = frozenset(f"I15-{index:02d}" for index in range(1, 16))
LIVING_INVARIANT_IDS = frozenset(f"LP-2.{index:02d}" for index in range(1, 18))
ACCEPTANCE_IDS = frozenset([*(f"LP-21.{index:02d}" for index in range(1, 29)), "LP-21.20a"])
REVERSIONS = frozenset(f"D{index:02d}" for index in range(1, 30))


def _node_for_environment(cell_id: str) -> str:
    """Return the expanded P04 node for one environment matrix cell."""

    return (
        "menagerie/crawler/tests/test_round21_environment_matrix_composition.py::"
        f"test_round21_environment_unit_matrix[{cell_id}]"
    )


def _node_for_shutdown(cell_id: str) -> str:
    """Return the expanded P05 node for one shutdown matrix cell."""

    return (
        "menagerie/crawler/tests/test_round21_shutdown_matrix_composition.py::"
        f"test_round21_shutdown_matrix[{cell_id}]"
    )


def _node_for_handoff(cell_id: str) -> str:
    """Return the expanded P06 node for one handoff matrix cell."""

    return (
        "menagerie/crawler/tests/test_round21_handoff_authority_composition.py::"
        f"test_round21_handoff_authority_identity_matrix[{cell_id}]"
    )


def _base_record(
    clause_id: str,
    *,
    source_locator: str,
    real_node_ids: Sequence[str],
    structural_node_ids: Sequence[str] = (),
    host: str = "linux",
    finding_ids: Sequence[str] = (),
    invariant_ids: Sequence[str] = (),
    deliberate_reversion_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one canonical conformance registry record."""

    return {
        "clause_id": clause_id,
        "source_locator": source_locator,
        "invariant_ids": sorted(set(invariant_ids)),
        "finding_ids": sorted(set(finding_ids)),
        "real_node_ids": sorted(set(real_node_ids)),
        "structural_node_ids": sorted(set(structural_node_ids)),
        "host": host,
        "expected_outcome": "passed",
        "real_prefix": True,
        "shipped_compiler": True,
        "deliberate_reversion_ids": sorted(set(deliberate_reversion_ids)),
    }


def _expand_nodes(aliases: Iterable[str]) -> tuple[str, ...]:
    """Expand proof aliases used by the Round-21 plan into full pytest node IDs."""

    nodes: list[str] = []
    for alias in aliases:
        if alias == "P01":
            nodes.append(P01)
        elif alias == "P02":
            nodes.append(P02)
        elif alias == "P03":
            nodes.append(P03)
        elif alias.startswith("P04-"):
            nodes.append(_node_for_environment(alias.removeprefix("P04-")))
        elif alias.startswith("P05-"):
            nodes.append(_node_for_shutdown(alias.removeprefix("P05-")))
        elif alias.startswith("P06-"):
            nodes.append(_node_for_handoff(alias.removeprefix("P06-")))
        elif alias == "P04":
            nodes.extend(_node_for_environment(cell_id) for cell_id in sorted(ENVIRONMENT_IDS))
        elif alias == "P05":
            nodes.extend(_node_for_shutdown(cell_id) for cell_id in sorted(SHUTDOWN_IDS))
        elif alias == "P06":
            nodes.extend(_node_for_handoff(cell_id) for cell_id in sorted(HANDOFF_IDS))
        elif alias == "P07":
            nodes.append(P07)
        elif alias == "P08":
            nodes.append(P08)
        elif alias == "P09":
            nodes.append(P09)
        elif alias == "P10":
            nodes.append(P10)
        elif alias == "P11":
            nodes.append(P11)
        elif alias == "P11-CI":
            nodes.append(P11_CI)
        elif alias == "P12":
            nodes.extend(P20)
        elif alias == "P13":
            nodes.append(P13)
        elif alias == "P14":
            nodes.append(P14)
        elif alias == "P15":
            nodes.append(P15)
        elif alias == "P16":
            nodes.append(P16)
        elif alias == "P17":
            nodes.append(P17)
        elif alias == "P18a":
            nodes.append(P18A)
        elif alias == "P18b":
            nodes.append(P18B)
        elif alias == "P19":
            nodes.append(P19)
        elif alias == "P20":
            nodes.extend(P20)
        else:
            raise AssertionError(f"unknown proof alias: {alias}")
    return tuple(dict.fromkeys(nodes))


def _expected_registry_records() -> tuple[dict[str, Any], ...]:
    """Return the exact VS11 totality registry required by the unified plan."""

    records: list[dict[str, Any]] = []
    finding_proofs = {
        "F01": ("P09", "P10", "P11-CI"),
        "F02": ("P01",),
        "F03": ("P02",),
        "F04": ("P05",),
        "F05": ("P04",),
        "F06": ("P03",),
        "F07": ("P01",),
        "F08": ("P01", "P05", "P06-H01"),
        "F09": ("P06-H01", "P06-H02"),
        "F10": ("P07", "P10"),
        "F11a": ("P08",),
        "F11b": ("P06-H03", "P06-H04"),
    }
    for finding_id, aliases in finding_proofs.items():
        records.append(
            _base_record(
                finding_id,
                source_locator="UNIFIED.md §6.1 Round-20 finding",
                real_node_ids=_expand_nodes(aliases),
                structural_node_ids=(T01,) if finding_id == "F02" else (),
                host="both" if "P10" in aliases or "P11-CI" in aliases else "linux",
                finding_ids=(finding_id,),
            )
        )

    clause_rows = {
        "R19-1": ("P01", "P09", "P10"),
        "R19-2.1": ("P11",),
        "R19-2.2": ("P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"),
        "R19-3.1": ("P01", "P04-E01", "P04-E06", "P04-E08", "P07"),
        "R19-3.2": ("P02", "P04"),
        "R19-3.3": ("P04-E03", "P09", "P10"),
        "R19-3.4": ("P02", "P03", "P04-E02", "P04-E04", "P04-E07", "P04-E08", "P04-E09"),
        "R19-3.5": ("P02", "P03"),
        "R19-3.6": ("P04-E11", "P04-E12", "P04-E13"),
        "R19-3.7": ("P01", "P07", "P09", "P10"),
        "R19-4": ("P01", "P09", "P10", "P13", "P19"),
        "R19-5.1": ("P09", "P10"),
        "R19-5.2": ("P09", "P10", "P11-CI"),
        "R19-6-VS1": ("P01", "P12", "P14", "P15"),
        "R19-6-VS2": ("P05",),
        "R19-6-VS3": ("P06", "P16"),
        "R19-6-VS4": ("P09", "P10", "P14", "P15"),
        "R19-6-VS5": ("P17",),
        "R19-6-VS6": ("P18a", "P18b"),
        "R19-6-VS7": ("P02", "P03", "P08"),
        "R19-6-EXIT": ("P11", "P11-CI"),
        "R19-7.1": ("P01", "P04", "P09", "P10"),
        "R19-7.2": ("P12", "P09", "P10"),
        "R19-7.3": ("P04",),
        "R19-7.4": ("P14", "P15", "P01"),
        "R19-7.5": ("P01", "P09", "P10", "P13", "P19"),
        "R19-8.1": ("P12", "P05", "P06-H01", "P18a"),
        "R19-8.2": ("P01",),
        "R19-8.3": ("P01",),
        "R19-8.4": ("P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"),
        "R19-10": ("P06-H01", "P06-H03", "P06-H04", "P16"),
        "R19-11": ("P09", "P10", "P14", "P15", "P11-CI"),
        "R19-12": ("P17",),
        "R19-13": ("P18a", "P18b"),
        "R19-14": ("P02", "P03", "P08"),
        "R19-16-P01": ("P12",),
        "R19-16-P02": ("P05-S08",),
        "R19-16-P03": ("P06-H01", "P06-H03", "P16"),
        "R19-16-P04": ("P20",),
        "R19-16-P05": ("P12", "P09"),
        "R19-16-P06": ("P01",),
        "R19-16-P07": ("P05-S06", "P05-S07"),
        "R19-16-P08": ("P12", "P06-H01", "P11"),
        "R19-17-D01": ("P01", "P04"),
        "R19-17-D02": ("P04-E03", "P09", "P10"),
        "R19-17-D03": ("P02", "P04-E02"),
        "R19-17-D04": ("P02", "P09", "P10"),
        "R19-17-D05": ("P04-E05", "P04-E06", "P04-E07", "P04-E08", "P04-E09", "P04-E10"),
        "R19-17-D06": ("P04-E11", "P04-E12", "P04-E13"),
        "R19-17-D07": ("P01", "P09", "P10", "P13", "P19"),
        "R19-17-D08": ("P09", "P10"),
        "R19-17-D09": ("P01",),
        "R19-17-D10": ("P10", "P15"),
        "R19-17-D11": ("P03",),
        "R19-17-D12": ("P09", "P10", "P01"),
        "R19-18-GATES": ("P09", "P10", "P11-CI"),
        "R19-18-EXIT": (
            "P01",
            "P02",
            "P03",
            "P04",
            "P05",
            "P06",
            "P07",
            "P08",
            "P09",
            "P10",
            "P11",
            "P11-CI",
        ),
    }
    for clause_id, aliases in clause_rows.items():
        records.append(
            _base_record(
                clause_id,
                source_locator="UNIFIED.md §6.2 Round-19 plan clause",
                real_node_ids=_expand_nodes(aliases),
                structural_node_ids=(T01, T02) if clause_id in {"R19-8.2", "R19-8.3"} else (),
                host="both"
                if any(alias in {"P10", "P11-CI", "P15"} for alias in aliases)
                else "linux",
                invariant_ids=tuple(sorted(LIVING_INVARIANT_IDS)),
            )
        )
    for cell_id in sorted(SHUTDOWN_IDS):
        records.append(
            _base_record(
                f"R19-9-{cell_id}",
                source_locator="UNIFIED.md §6.2 shutdown matrix row",
                real_node_ids=(_node_for_shutdown(cell_id),),
                host="linux",
            )
        )
    for index in range(1, 17):
        reversion_id = f"D{index:02d}" if index <= 11 else f"D{index + 3:02d}"
        records.append(
            _base_record(
                f"R19-18-REV{index:02d}",
                source_locator="UNIFIED.md §6.2 original reversion bullet",
                real_node_ids=(P11,),
                structural_node_ids=(T06,),
                host="linux",
                deliberate_reversion_ids=(reversion_id,),
            )
        )

    invariant_proofs = {
        "I15-01": ("P01", "P12", "P14", "P15"),
        "I15-02": ("P02", "P04-E02", "P04-E03", "P04-E04", "P04-E07", "P04-E08", "P04-E09"),
        "I15-03": ("P01", "P09", "P10", "P13", "P19"),
        "I15-04": ("P01", "P07", "P14", "P15", "P10"),
        "I15-05": ("P02", "P04-E11", "P09", "P10"),
        "I15-06": ("P04-E11", "P04-E12", "P04-E13"),
        "I15-07": ("P01", "P06-H01", "P06-H03"),
        "I15-08": ("P02", "P05", "P06", "P18a", "P18b"),
        "I15-09": ("P05", "P12"),
        "I15-10": ("P05-S08", "P05-S09", "P05-S10", "P05-S11", "P05-S12", "P05-S13"),
        "I15-11": ("P01", "P05", "P14", "P15"),
        "I15-12": ("P06-H01", "P06-H03", "P06-H04", "P16"),
        "I15-13": ("P17", "P20"),
        "I15-14": ("P01", "P04", "P07", "P10"),
        "I15-15": ("P05", "P02", "P04", "P18a"),
    }
    for invariant_id, aliases in invariant_proofs.items():
        records.append(
            _base_record(
                invariant_id,
                source_locator="UNIFIED.md §6.3 Round-19 §15 invariant",
                real_node_ids=_expand_nodes(aliases),
                structural_node_ids=(T01, T02, T06) if invariant_id == "I15-14" else (),
                host="both" if any(alias in {"P10", "P15"} for alias in aliases) else "linux",
                invariant_ids=(invariant_id,),
            )
        )

    for lp_id in sorted(LIVING_INVARIANT_IDS):
        records.append(
            _base_record(
                lp_id,
                source_locator="menagerie/crawler/PLAN.md §2 locked invariant",
                real_node_ids=_expand_nodes(("P01", "P12")),
                structural_node_ids=(T06,),
                invariant_ids=(lp_id,),
            )
        )
    living_clause_proofs = {
        "LP-5.0": ("P12",),
        "LP-5.0.1": ("P12", "P06-H01", "P05-S08"),
        "LP-6.1": ("P17", "P20"),
        "LP-6.2": ("P17",),
        "LP-6.3": ("P01", "P05", "P12"),
        "LP-8.1": ("P12",),
        "LP-8.2": ("P09", "P10"),
        "LP-8.3": ("P01", "P14", "P15"),
        "LP-8.4": ("P09", "P10"),
        "LP-8.5": ("P01",),
        "LP-11.1": ("P09", "P10"),
        "LP-11.2": ("P01", "P04"),
        "LP-11.3": ("P07", "P10"),
        "LP-11.4": ("P01", "P14", "P15"),
        "LP-12.1": ("P12",),
        "LP-12.2": ("P12",),
        "LP-12.3": ("P12",),
        "LP-12.4": ("P17", "P20"),
        "LP-12.5": ("P12",),
        "LP-13.1": ("P18a",),
        "LP-13.2": ("P18b",),
        "LP-13.3": ("P18a", "P18b"),
        "LP-17.1": ("P05",),
        "LP-17.2": ("P05-S06", "P05-S07"),
        "LP-17.3": ("P05-S06",),
        "LP-17.4": ("P05",),
        "LP-17.5": ("P05", "P18a"),
        "LP-17.6": ("P05",),
    }
    for clause_id, aliases in living_clause_proofs.items():
        records.append(
            _base_record(
                clause_id,
                source_locator="menagerie/crawler/PLAN.md crawler-relevant clause",
                real_node_ids=_expand_nodes(aliases),
                invariant_ids=tuple(sorted(LIVING_INVARIANT_IDS)),
                host="both" if any(alias in {"P10", "P15"} for alias in aliases) else "linux",
            )
        )
    for acceptance_id in sorted(ACCEPTANCE_IDS):
        records.append(
            _base_record(
                acceptance_id,
                source_locator="menagerie/crawler/PLAN.md §21 acceptance test item",
                real_node_ids=_expand_nodes(("P01", "P12")),
                structural_node_ids=(T06,),
                invariant_ids=tuple(sorted(LIVING_INVARIANT_IDS)),
            )
        )

    for cell_id in sorted(ENVIRONMENT_IDS):
        records.append(
            _base_record(
                cell_id,
                source_locator="UNIFIED.md VS4 environment-unit matrix",
                real_node_ids=(_node_for_environment(cell_id),),
                finding_ids=("F05",),
                deliberate_reversion_ids=tuple(
                    d_id
                    for d_id in ("D01", "D02", "D03", "D04", "D05", "D06")
                    if d_id in {"D01", "D02", "D03", "D04", "D05", "D06"}
                ),
            )
        )
    for cell_id in sorted(SHUTDOWN_IDS):
        records.append(
            _base_record(
                cell_id,
                source_locator="UNIFIED.md VS5 shutdown/admission/atomicity matrix",
                real_node_ids=(_node_for_shutdown(cell_id),),
                finding_ids=("F04",),
                deliberate_reversion_ids=("D15", "D16") if cell_id == "S11" else ("D15",),
            )
        )
    for cell_id in sorted(HANDOFF_IDS):
        records.append(
            _base_record(
                cell_id,
                source_locator="UNIFIED.md VS6 handoff authority matrix",
                real_node_ids=(_node_for_handoff(cell_id),),
                finding_ids=("F09", "F11b"),
                deliberate_reversion_ids=("D18", "D19"),
            )
        )
    for evasion_id in sorted(EVASION_IDS):
        records.append(
            _base_record(
                evasion_id,
                source_locator="UNIFIED.md VS1 §8.3 evasion class",
                real_node_ids=(P01,),
                structural_node_ids=(T01,),
                finding_ids=("F02",),
                deliberate_reversion_ids=("D09",),
            )
        )
    for preservation_id in sorted(PRESERVATION_IDS):
        mapped_aliases = {
            "PRES01": ("P12",),
            "PRES02": ("P05-S08",),
            "PRES03": ("P06-H01", "P06-H03", "P16"),
            "PRES04": ("P20",),
            "PRES05": ("P12", "P09"),
            "PRES06": ("P01",),
            "PRES07": ("P05-S06", "P05-S07"),
            "PRES08": ("P12", "P06-H01", "P11"),
        }[preservation_id]
        records.append(
            _base_record(
                preservation_id,
                source_locator="UNIFIED.md §6.2 R19-16 preservation item",
                real_node_ids=_expand_nodes(mapped_aliases),
                structural_node_ids=(T06,),
                deliberate_reversion_ids=("D29",),
            )
        )
    for decision_id in sorted(DECISION_IDS):
        records.append(
            _base_record(
                decision_id,
                source_locator="UNIFIED.md §6.2 R19-17 resolved decision",
                real_node_ids=_expand_nodes((f"P04-E{decision_id[-2:]}",))
                if decision_id <= "DEC06"
                else _expand_nodes(("P01", "P09", "P10")),
                structural_node_ids=(T06,),
                host="linux" if decision_id <= "DEC06" else "both",
            )
        )
    records.append(
        _base_record(
            "DISAGREEMENT-01",
            source_locator="UNIFIED.md §1.2 autonomous path versus optional fallback",
            real_node_ids=_expand_nodes(("P09", "P10", "P11-CI")),
            structural_node_ids=(T04_LINUX, T04_MACOS),
            host="both",
        )
    )
    reversion_proofs = {
        "D01": ("P12", "P04-E11"),
        "D02": ("P04-E02", "P04-E03"),
        "D03": ("P04-E02",),
        "D04": ("P04-E04", "P04-E08"),
        "D05": ("P04-E05", "P04-E06", "P04-E07", "P04-E08", "P04-E09", "P04-E10"),
        "D06": ("P04-E12", "P04-E13"),
        "D07": ("P01", "P09", "P10"),
        "D08": ("P13", "P19"),
        "D09": ("P01",),
        "D10": ("P01",),
        "D11": ("P01",),
        "D12": ("P02",),
        "D13": ("P02",),
        "D14": ("P03",),
        "D15": ("P05",),
        "D16": ("P05-S11",),
        "D17": ("P07", "P10"),
        "D18": ("P06-H02",),
        "D19": ("P06-H03", "P06-H04", "P06-H01"),
        "D20": ("P08",),
        "D21": ("P09", "P10", "P11-CI"),
        "D22": ("P11-CI",),
        "D23": ("P09", "P14", "P11-CI"),
        "D24": ("P10", "P15", "P11-CI"),
        "D25": ("P17", "P20"),
        "D26": ("P18a", "P18b"),
        "D27": ("P06-H01",),
        "D28": ("P11", "P11-CI"),
        "D29": ("P12", "P16", "P17"),
    }
    for reversion_id, aliases in reversion_proofs.items():
        records.append(
            _base_record(
                reversion_id,
                source_locator="UNIFIED.md T06 deliberate-reversion table",
                real_node_ids=_expand_nodes(aliases),
                structural_node_ids=(T06,),
                host="both"
                if any(alias in {"P10", "P11-CI", "P15"} for alias in aliases)
                else "linux",
                deliberate_reversion_ids=(reversion_id,),
            )
        )
    return tuple(sorted(records, key=lambda record: record["clause_id"]))


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    """Load one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must contain a JSON object"
    return payload


def _registry_payload() -> Mapping[str, Any]:
    """Load and minimally validate the public conformance registry envelope."""

    payload = _load_json_mapping(_REGISTRY_PATH)
    assert payload["schema_version"] == "menagerie.crawler.round21-conformance.v1"
    assert payload["status"] == "complete"
    assert payload["no_waivers"] is True
    records = payload["records"]
    assert isinstance(records, list) and records
    return payload


def _records_by_id(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return registry records keyed by unique clause ID."""

    records = payload["records"]
    assert isinstance(records, list)
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        assert isinstance(record, dict)
        clause_id = record.get("clause_id")
        assert isinstance(clause_id, str) and clause_id
        assert clause_id not in by_id, f"duplicate conformance clause_id: {clause_id}"
        by_id[clause_id] = record
    return by_id


def _load_release_nodes(path: Path, target: str) -> frozenset[str]:
    """Load one host release-node registry."""

    payload = _load_json_mapping(path)
    assert payload["target"] == target
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    return frozenset(str(node) for node in nodes)


def _collect_node_ids(nodes: Iterable[str]) -> frozenset[str]:
    """Collect exact pytest nodes without running the tests."""

    roots = sorted({node.split("::", 1)[0] for node in nodes})
    completed = subprocess.run(
        (sys.executable, "-m", "pytest", "--collect-only", "-q", *roots),
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    collected = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith(_PROOF_PREFIX) and "::" in line
    }
    return frozenset(collected)


def _function_fixture_names(node_id: str) -> frozenset[str]:
    """Return fixture-style parameter names declared by a node's root function."""

    source_path_text, function_part = node_id.split("::", 1)
    function_name = function_part.split("[", 1)[0]
    source_path = _REPOSITORY_ROOT / source_path_text
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1, f"missing or ambiguous test function for {node_id}"
    return frozenset(arg.arg for arg in functions[0].args.args)


def _assert_record_schema(record: Mapping[str, Any]) -> None:
    """Assert one conformance record has no escape hatch and every required field."""

    assert set(record) == {
        "clause_id",
        "source_locator",
        "invariant_ids",
        "finding_ids",
        "real_node_ids",
        "structural_node_ids",
        "host",
        "expected_outcome",
        "real_prefix",
        "shipped_compiler",
        "deliberate_reversion_ids",
    }
    assert record["host"] in {"linux", "macos", "both"}
    assert record["expected_outcome"] == "passed"
    assert record["real_prefix"] is True
    assert record["shipped_compiler"] is True
    assert record["real_node_ids"]
    forbidden = json.dumps(record, sort_keys=True).lower()
    assert "waiver" not in forbidden
    assert "planned" not in forbidden
    assert "non-proof" not in forbidden


def _assert_nodes_are_scanned_and_fixture_backed(nodes: Iterable[str]) -> None:
    """Assert every proof node is in the VS1 scan and uses a real fixture path."""

    composition_paths = {
        path.relative_to(_REPOSITORY_ROOT).as_posix()
        for path in structural._COMPOSITION_SOURCES  # noqa: SLF001
    }
    release_nodes = _load_release_nodes(_LINUX_NODES_PATH, "linux-64") | _load_release_nodes(
        _MACOS_NODES_PATH, "osx-arm64"
    )
    allowed_non_fixture_nodes = {
        P11,
        P11_CI,
        P16,
        T01,
        T02,
        T03,
        T04_LINUX,
        T04_MACOS,
        T05_LINUX,
        T05_MACOS,
    }
    for node in nodes:
        source_path = node.split("::", 1)[0]
        assert source_path in composition_paths, f"{node} is outside VS1 composition scan"
        if node in allowed_non_fixture_nodes or node in release_nodes:
            continue
        fixtures = _function_fixture_names(node)
        assert fixtures & {
            "real_environment_fixture",
            "isolated_real_environment_fixture",
            "isolated_real_environment_factory",
            "request",
        }, f"{node} does not declare a real fixture dependency"


def _attestation_paths() -> tuple[Path, Path, Path]:
    """Return the Linux, macOS, and reversion attestation paths from the environment."""

    linux = os.environ.get("MENAGERIE_LINUX_RELEASE_ATTESTATION")
    macos = os.environ.get("MENAGERIE_MACOS_RELEASE_ATTESTATION")
    reversion = os.environ.get("MENAGERIE_REVERSION_ATTESTATION")
    if not linux or not macos or not reversion:
        pytest.fail(
            "unmet-release-gate: MENAGERIE_LINUX_RELEASE_ATTESTATION, "
            "MENAGERIE_MACOS_RELEASE_ATTESTATION, and MENAGERIE_REVERSION_ATTESTATION are required"
        )
    return Path(linux), Path(macos), Path(reversion)


@pytest.mark.round21_linux_real
def test_round21_conformance_registry_is_total_and_executed(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The VS11 registry must be total, exact, collected, and real-composition backed."""

    expected_records = _expected_registry_records()
    expected_by_id = {record["clause_id"]: record for record in expected_records}
    payload = _registry_payload()
    observed_by_id = _records_by_id(payload)
    assert observed_by_id == expected_by_id

    sandbox = "bubblewrap" if sys.platform == "linux" else "sandbox-exec"
    _run_host_denial_composition(tmp_path, real_environment_fixture, expected_sandbox=sandbox)

    all_real_nodes = frozenset(
        node for record in observed_by_id.values() for node in record["real_node_ids"]
    )
    assert P11 in all_real_nodes
    assert P11_CI in all_real_nodes
    collected = _collect_node_ids(all_real_nodes)
    assert all_real_nodes <= collected
    _assert_nodes_are_scanned_and_fixture_backed(all_real_nodes)

    linux_nodes = _load_release_nodes(_LINUX_NODES_PATH, "linux-64")
    macos_nodes = _load_release_nodes(_MACOS_NODES_PATH, "osx-arm64")
    assert P11 in linux_nodes
    assert P11_CI not in linux_nodes
    assert P10 in macos_nodes
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "crawler-round21-deliberate-reversions:" in workflow
    assert "crawler-round21-conformance:" in workflow
    assert "round21-linux-release-attestation" in workflow
    assert "round21-macos-release-attestation" in workflow
    assert "round21-reversion-attestation" in workflow
    assert "test_round21_ci_attestations_cover_registry_without_skip" in workflow
    assert all(
        node in linux_nodes | macos_nodes | {P11, P11_CI}
        for node in all_real_nodes
        if not node.startswith("menagerie/crawler/tests/test_round17_structural_inventories.py")
    )

    clause_ids = frozenset(observed_by_id)
    assert ENVIRONMENT_IDS <= clause_ids
    assert SHUTDOWN_IDS <= clause_ids
    assert HANDOFF_IDS <= clause_ids
    assert EVASION_IDS <= clause_ids
    assert PRESERVATION_IDS <= clause_ids
    assert DECISION_IDS <= clause_ids
    assert FINDING_IDS <= clause_ids
    assert INVARIANT_IDS <= clause_ids
    assert LIVING_INVARIANT_IDS <= clause_ids
    assert ACCEPTANCE_IDS <= clause_ids
    assert REVERSIONS <= clause_ids
    assert "DISAGREEMENT-01" in clause_ids

    assert {
        record["clause_id"]
        for record in observed_by_id.values()
        if record["clause_id"].startswith("E")
    } == ENVIRONMENT_IDS | EVASION_IDS
    assert {
        record["clause_id"]
        for record in observed_by_id.values()
        if record["clause_id"].startswith("S")
    } == SHUTDOWN_IDS
    assert {
        record["clause_id"]
        for record in observed_by_id.values()
        if record["clause_id"].startswith("H")
    } == HANDOFF_IDS
    assert {
        finding_id for record in observed_by_id.values() for finding_id in record["finding_ids"]
    } == FINDING_IDS
    assert {
        invariant_id
        for record in observed_by_id.values()
        for invariant_id in record["invariant_ids"]
    } >= INVARIANT_IDS | LIVING_INVARIANT_IDS
    assert {
        reversion_id
        for record in observed_by_id.values()
        for reversion_id in record["deliberate_reversion_ids"]
    } == REVERSIONS

    reversion_matrix = _load_json_mapping(_REVERSIONS_TOOL_PATH.with_suffix(".json"))
    assert reversion_matrix["deliberate_reversion_ids"] == sorted(REVERSIONS)
    assert Counter(reversion_matrix["deliberate_reversion_ids"]) == Counter(REVERSIONS)

    for record in observed_by_id.values():
        _assert_record_schema(record)


def test_round21_ci_attestations_cover_registry_without_skip() -> None:
    """Both host attestations and the reversion result must cover VS11 without skips."""

    payload = _registry_payload()
    records = _records_by_id(payload)
    linux_path, macos_path, reversion_path = _attestation_paths()
    attestations = {
        "linux": _load_json_mapping(linux_path),
        "macos": _load_json_mapping(macos_path),
    }
    expected_targets = {"linux": "linux-64", "macos": "osx-arm64"}
    for host, attestation in attestations.items():
        assert attestation["schema_version"] == "menagerie.crawler.release-proof-attestation.v1"
        assert attestation["status"] == "passed"
        assert attestation["target"] == expected_targets[host]
        assert attestation["expected_nodes"] == attestation["collected_nodes"]
        assert attestation["expected_nodes"] == attestation["passed_nodes"]
        assert attestation["skipped_nodes"] == []
        assert attestation["xfailed_nodes"] == []
        assert attestation["failed_nodes"] == []
        assert attestation["environment_content_digest"]
        assert attestation["selected_interpreter"].endswith("/bin/python")

    linux_passed = frozenset(str(node) for node in attestations["linux"]["passed_nodes"])
    macos_passed = frozenset(str(node) for node in attestations["macos"]["passed_nodes"])
    for record in records.values():
        host = record["host"]
        required = frozenset(str(node) for node in record["real_node_ids"])
        if host == "linux":
            assert required <= linux_passed | {P11_CI}
        elif host == "macos":
            assert required <= macos_passed | {P11_CI}
        else:
            assert required <= linux_passed | macos_passed | {P11_CI}

    reversion = _load_json_mapping(reversion_path)
    assert reversion["schema_version"] == "menagerie.crawler.round21-reversion-result.v1"
    assert reversion["status"] == "passed"
    assert reversion["passed_reversions"] == sorted(REVERSIONS)
    assert reversion["skipped_reversions"] == []
    assert reversion["failed_reversions"] == []
