"""Deterministic Round-21 deliberate-reversion runner for disposable checkouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


_REVERSIONS = tuple(f"D{index:02d}" for index in range(1, 30))
_REGISTRY_PATH = Path("menagerie/crawler/conformance-round21.json")
_WORKFLOW_PATH = Path(".github/workflows/tests.yml")
_LINUX_LOCK_PATH = Path("menagerie/crawler/envs/locks/round19-linux-64.lock")
_AUTHORITY_PATH = Path("menagerie/crawler/authority.py")
_DRIVER_PATH = Path("menagerie/crawler/driver.py")
_POLICY_PATH = Path("menagerie/crawler/policy.py")
_SUPERVISOR_PATH = Path("menagerie/crawler/worker_supervisor.py")
_CLI_PATH = Path("menagerie/crawler/cli.py")
_STATUS_PATH = Path("menagerie/crawler/status.py")
_WORKER_PATH = Path("menagerie/crawler/worker.py")
_ENV_LIFECYCLE_PATH = Path("menagerie/crawler/env_lifecycle.py")
_CONFORMANCE_NODE = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_conformance_registry_is_total_and_executed"
)


def _node_for_environment(cell_id: str) -> str:
    """Return the expanded P04 environment-matrix pytest node."""

    return (
        "menagerie/crawler/tests/test_round21_environment_matrix_composition.py::"
        f"test_round21_environment_unit_matrix[{cell_id}]"
    )


def _node_for_shutdown(cell_id: str) -> str:
    """Return the expanded P05 shutdown-matrix pytest node."""

    return (
        "menagerie/crawler/tests/test_round21_shutdown_matrix_composition.py::"
        f"test_round21_shutdown_matrix[{cell_id}]"
    )


def _node_for_handoff(cell_id: str) -> str:
    """Return the expanded P06 handoff-matrix pytest node."""

    return (
        "menagerie/crawler/tests/test_round21_handoff_authority_composition.py::"
        f"test_round21_handoff_authority_identity_matrix[{cell_id}]"
    )


_P01 = (
    "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
    "test_round21_preclusion_real_v3_path_has_no_substitutable_fixture_edge"
)
_T01 = (
    "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
    "test_round21_tripwire_catches_python_evasion"
)
_P02 = (
    "menagerie/crawler/tests/test_round21_fingerprint_composition.py::"
    "test_round21_cheap_fingerprint_catches_stat_preserved_mutation_without_false_staling_clone"
)
_P03 = (
    "menagerie/crawler/tests/test_round21_scale_composition.py::"
    "test_round21_pass_and_spawn_validation_walks_are_constant_bounded"
)
_P07 = (
    "menagerie/crawler/tests/test_round21_transport_composition.py::"
    "test_round21_closed_transport_capability_awards_and_rejects_unlisted_library"
)
_P08 = (
    "menagerie/crawler/tests/test_round21_cache_rebind_composition.py::"
    "test_round21_mismatched_rebind_preserves_active_authority_and_awards"
)
_P09 = (
    "menagerie/crawler/tests/test_round21_ci_composition.py::"
    "test_round21_linux_committed_lock_provenance_awards_in_ci"
)
_P10 = (
    "menagerie/crawler/tests/test_round21_ci_composition.py::"
    "test_round21_macos_committed_lock_seatbelt_award_and_denial"
)
_P11_CI = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_ci_attestations_cover_registry_without_skip"
)
_P12_NONE = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[none]"
)
_P12_STAT = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[statistical]"
)
_P13 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_outside_selected_interpreter_is_rejected_at_binding"
)
_P14 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
_P15 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
_P16 = (
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes"
)
_P17 = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_unhashable_output_awards_runs_with_unverifiable_modes"
)
_P18A = (
    "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
    "test_documented_dry_run_and_resume_use_real_environment"
)
_P18B = (
    "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
    "test_dry_run_all_source_failure_is_acceptance_error"
)
_P19 = (
    "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
    "test_manifest_v3_rejects_changed_interpreter_association"
)


@dataclass(frozen=True)
class ReversionCase:
    """One deterministic disposable-checkout deliberate reversion."""

    reversion_id: str
    semantic_reversion: str
    proof_nodes: tuple[str, ...]
    expected_reason: str
    mutate: Callable[[Path, str], None]


def _load_registry(root: Path) -> dict[str, object]:
    """Load the conformance registry from a disposable checkout."""

    payload = json.loads((root / _REGISTRY_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("conformance registry must be a JSON object")
    return payload


def _write_registry(root: Path, payload: dict[str, object]) -> None:
    """Write the conformance registry atomically in a disposable checkout."""

    path = root / _REGISTRY_PATH
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _replace_once(root: Path, relative: Path, old: str, new: str) -> None:
    """Apply one exact textual source mutation to a disposable checkout file."""

    path = root / relative
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise ValueError(f"{relative} does not contain exactly one mutation anchor")
    path.write_text(source.replace(old, new), encoding="utf-8")


def _delete_linux_lock(root: Path, _reversion_id: str) -> None:
    """Delete the committed Linux explicit lock in a disposable checkout."""

    (root / _LINUX_LOCK_PATH).unlink()


def _drop_first_registry_record(root: Path, _reversion_id: str) -> None:
    """Delete the first conformance record to prove totality has teeth."""

    payload = _load_registry(root)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("conformance registry lacks records[]")
    payload["records"] = records[1:]
    _write_registry(root, payload)


def _ignore_content_staleness(root: Path, _reversion_id: str) -> None:
    """Accept changed sealed bytes during authority currentness verification."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """        if (
            current.content_manifest_sha256 != authority.content_manifest_sha256
            or current.selected_interpreter_digest != authority.selected_interpreter_digest
        ):
            self.invalidations += 1
            self._manifest = None
            self._authority = None
            raise AuthorityDerivationError("environment content seal changed; authority is stale")
""",
        """        if current.selected_interpreter_digest != authority.selected_interpreter_digest:
            self.invalidations += 1
            self._manifest = None
            self._authority = None
            raise AuthorityDerivationError("environment content seal changed; authority is stale")
""",
    )


def _omit_generation_content_seal(root: Path, reversion_id: str) -> None:
    """Omit content changes from authority generation and stale-byte enforcement."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """        final_generation = stable_hash(
            {
                "version": ENVIRONMENT_GENERATION_VERSION_V2,
                "base_environment_generation": base_generation,
                "environment_content_sha256": manifest.content_manifest_sha256,
            }
        )
""",
        """        final_generation = stable_hash(
            {
                "version": ENVIRONMENT_GENERATION_VERSION_V2,
                "base_environment_generation": base_generation,
            }
        )
""",
    )
    _ignore_content_staleness(root, reversion_id)


def _conda_meta_only_seal(root: Path, _reversion_id: str) -> None:
    """Trust package metadata while ignoring the declared lock/export package set."""

    _replace_once(
        root,
        _ENV_LIFECYCLE_PATH,
        """    if lock_receipts != package_receipts or len(package_receipts) != len(packages):
        raise EnvironmentExactnessError(
            "target lock artifacts differ from the created-prefix package inventory"
        )
""",
        """    if False and (lock_receipts != package_receipts or len(package_receipts) != len(packages)):
        raise EnvironmentExactnessError(
            "target lock artifacts differ from the created-prefix package inventory"
        )
""",
    )


def _grant_post_seal_bytes(root: Path, _reversion_id: str) -> None:
    """Let undeclared post-seal runtime files read through policy."""

    _replace_once(
        root,
        _POLICY_PATH,
        """        try:
            if candidate.is_dir():
                return True
        except OSError:
            pass
""",
        """        try:
            if candidate.exists():
                return True
        except OSError:
            pass
""",
    )


def _weaken_symlink_binding(root: Path, _reversion_id: str) -> None:
    """Stop recording symlink text and target identity in the content seal."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """        "link_text": entry.link_text,
        "resolved_target_relative_path": entry.resolved_target_relative_path,
""",
        """        "link_text": None,
        "resolved_target_relative_path": None,
""",
    )


def _treat_checkpoints_as_runtime(root: Path, _reversion_id: str) -> None:
    """Classify packaged model-data suffixes as ordinary runtime reads."""

    _replace_once(
        root,
        _POLICY_PATH,
        "        if not startup_pth:\n                return False\n",
        "        if False and not startup_pth:\n                return False\n",
    )


def _restore_parent_interpreter(root: Path, _reversion_id: str) -> None:
    """Run workers with the parent interpreter instead of the selected prefix interpreter."""

    _replace_once(
        root,
        _DRIVER_PATH,
        '    interpreter = prefix / "bin" / "python"\n',
        "    interpreter = Path(sys.executable)\n",
    )


def _allow_mismatched_interpreter(root: Path, _reversion_id: str) -> None:
    """Accept changed selected-interpreter associations."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """    if resolved_interpreter != expected_target:
        raise AuthorityDerivationError("selected interpreter association changed")
    if not resolved_interpreter.is_relative_to(authority.prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
""",
        """    if False and resolved_interpreter != expected_target:
        raise AuthorityDerivationError("selected interpreter association changed")
    if False and not resolved_interpreter.is_relative_to(authority.prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
""",
    )


def _restore_legacy_runtime_root(root: Path, _reversion_id: str) -> None:
    """Reintroduce a live legacy runtime-root token in production policy."""

    _replace_once(
        root,
        _POLICY_PATH,
        "_SAFE_INHERITED_KEYS = (",
        'LEGACY_RUNTIME_ROOT_KIND = "runtime-root"\n_SAFE_INHERITED_KEYS = (',
    )


def _remove_transitive_scope_symbol(root: Path, _reversion_id: str) -> None:
    """Rename the production compiler symbol required by the scope inventory."""

    _replace_once(
        root,
        _DRIVER_PATH,
        "def _compile_worker_read_manifest(",
        "def _compile_worker_read_manifest_removed(",
    )


def _omit_fingerprint_fields(root: Path, _reversion_id: str) -> None:
    """Drop cheap fingerprint change fields by accepting every fingerprint."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        "        if fingerprint == manifest.cheap_tree_fingerprint:\n            return\n",
        "        if True:\n            return\n",
    )


def _quarantine_hardlink_clone_churn(root: Path, _reversion_id: str) -> None:
    """Treat byte-identical hardlink-clone ctime churn as stale."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """        # Hardlink-clone creation can change source-inode ctime without changing bytes.
        # Keep the bound semantic identity and re-baseline only this authority's cheap fields.
        self._manifest = current
""",
        """        # Deliberate reversion: quarantine byte-identical hardlink-clone churn.
        self.invalidations += 1
        self._manifest = None
        self._authority = None
        raise AuthorityDerivationError("environment content seal changed; authority is stale")
""",
    )


def _reuse_currentness_for_spawn(root: Path, _reversion_id: str) -> None:
    """Permit a currentness-pass token to authorize a later worker spawn."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        '        token = self._issue_token(authority, "spawn")\n',
        '        token = self._active_currentness_token or self._issue_token(authority, "spawn")\n',
    )


def _delete_shutdown_guard(root: Path, _reversion_id: str) -> None:
    """Disable one production shutdown admission check."""

    _replace_once(
        root,
        _DRIVER_PATH,
        """        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
""",
        """        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
""",
    )


def _move_shutdown_inside_publication(root: Path, _reversion_id: str) -> None:
    """Insert a shutdown check inside the award-publication atomic section."""

    _replace_once(
        root,
        _DRIVER_PATH,
        """        if model.get("authored_metadata_state") == "accepted" and not isinstance(
            artifact, ActivatedHandoffArtifact
        ):
            self._authorize_and_publish_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
""",
        """        if model.get("authored_metadata_state") == "accepted" and not isinstance(
            artifact, ActivatedHandoffArtifact
        ):
            self._authorize_and_publish_artifact(artifact, model, gates, reducer)
        self._check_shutdown("post-award-commit")
        result = reducer.append_model(reducer.prepare_model(model))
""",
    )


def _restore_transport_suffix_grant(root: Path, _reversion_id: str) -> None:
    """Allow arbitrary shared-library suffixes as host transport."""

    _replace_once(
        root,
        _SUPERVISOR_PATH,
        "    return capability is not None and capability.allows(path)\n",
        '    return capability is not None and (capability.allows(path) or path.suffix == ".so" or ".so." in path.name)\n',
    )


def _retain_mismatched_artifact_transaction(root: Path, _reversion_id: str) -> None:
    """Retain prior artifact authority without checking the new transaction."""

    _replace_once(
        root,
        _DRIVER_PATH,
        "        if retain_prior_artifact_authority:\n",
        "        if False and retain_prior_artifact_authority:\n",
    )


def _accept_ambiguous_handoff_finals(root: Path, _reversion_id: str) -> None:
    """Accept zero or multiple selected handoff final events."""

    _replace_once(
        root,
        _DRIVER_PATH,
        """        if len(matching) != 1:
            raise DriverIntegrationError("handoff-authority-unavailable")
        if (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            raise DriverIntegrationError("handoff-authority-unavailable")
""",
        """        if not matching:
            continue
        if (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            continue
""",
    )


def _invalidate_on_rejected_rebind(root: Path, _reversion_id: str) -> None:
    """Invalidate the active cache when a mismatched rebind is rejected."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """                self.rejected_rebinds += 1
                raise AuthorityDerivationError(
                    "active environment authority cache cannot be rebound to different inputs"
                )
""",
        """                self.rejected_rebinds += 1
                self.invalidations += 1
                self._manifest = None
                self._authority = None
                raise AuthorityDerivationError(
                    "active environment authority cache cannot be rebound to different inputs"
                )
""",
    )


def _permit_ci_skip(root: Path, _reversion_id: str) -> None:
    """Remove the release gate from the final P11-CI workflow invocation."""

    _replace_once(
        root,
        _WORKFLOW_PATH,
        """    env:
      MENAGERIE_RELEASE_GATE: "1"
      MENAGERIE_LINUX_RELEASE_ATTESTATION: ${{ runner.temp }}/round21-attestations/linux/round21-linux-attestation.json
""",
        """    env:
      MENAGERIE_LINUX_RELEASE_ATTESTATION: ${{ runner.temp }}/round21-attestations/linux/round21-linux-attestation.json
""",
    )


def _remove_linux_denial_audit(root: Path, _reversion_id: str) -> None:
    """Remove Linux strace denial tooling from release workflow provisioning."""

    _replace_once(
        root,
        _WORKFLOW_PATH,
        """      - name: Require Linux lock, provenance, conda, bwrap, and strace
        shell: bash
        run: |
          sudo apt-get update
          sudo apt-get install -y bubblewrap strace
""",
        """      - name: Require Linux lock, provenance, conda, and bwrap
        shell: bash
        run: |
          sudo apt-get update
          sudo apt-get install -y bubblewrap
""",
    )
    _replace_once(
        root,
        _WORKFLOW_PATH,
        """          if ! command -v bwrap >/dev/null 2>&1 || ! command -v strace >/dev/null 2>&1; then
            echo "unmet-release-gate: Linux bwrap/strace enforcement is unavailable"
            exit 1
          fi
""",
        """          if ! command -v bwrap >/dev/null 2>&1; then
            echo "unmet-release-gate: Linux bwrap enforcement is unavailable"
            exit 1
          fi
""",
    )


def _replace_macos_denial_with_profile(root: Path, _reversion_id: str) -> None:
    """Remove macOS denial-audit command availability from the release workflow."""

    _replace_once(
        root,
        _WORKFLOW_PATH,
        "sandbox-exec >/dev/null 2>&1 || ! command -v log >/dev/null 2>&1",
        "sandbox-exec >/dev/null 2>&1",
    )


def _misclassify_unverifiable_mode(root: Path, _reversion_id: str) -> None:
    """Misclassify unverifiable outputs as ordinary mismatches."""

    _replace_once(
        root,
        _AUTHORITY_PATH,
        """        if not isinstance(train_digest, str) or not isinstance(eval_digest, str):
            comparison_state = "unverifiable"
            classification = "unverifiable"
            reason = "matching output signatures lack stable output value digests"
""",
        """        if not isinstance(train_digest, str) or not isinstance(eval_digest, str):
            comparison_state = "verified"
            classification = "none"
            reason = "matching output signatures lack stable output value digests"
""",
    )


def _restore_dry_run_false_complete(root: Path, _reversion_id: str) -> None:
    """Let dry-run acceptance proceed despite failed source checks."""

    _replace_once(
        root,
        _CLI_PATH,
        '    if bool(getattr(args, "dry_run", False)) and acceptance["status"] != "passed":\n',
        '    if False and bool(getattr(args, "dry_run", False)) and acceptance["status"] != "passed":\n',
    )


def _restore_source_fallback(root: Path, _reversion_id: str) -> None:
    """Allow handoff execution to continue without persisted artifact authority."""

    _replace_once(
        root,
        _DRIVER_PATH,
        """            if self.config.only_status is not None and inputs.handoff_execution is None:
                raise DriverIntegrationError("handoff-authority-unavailable")
""",
        """            if False and self.config.only_status is not None and inputs.handoff_execution is None:
                raise DriverIntegrationError("handoff-authority-unavailable")
""",
    )


def _regress_receipt_spawn_writer_schema(root: Path, _reversion_id: str) -> None:
    """Remove raw receipt digest binding from worker result preservation."""

    _replace_once(
        root,
        _WORKER_PATH,
        '"raw_award_receipt_sha256": (None if raw_receipt is None else stable_hash(raw_receipt)),',
        '"raw_award_receipt_sha256": None,',
    )


def _cases() -> tuple[ReversionCase, ...]:
    """Return the exact D01-D29 deliberate-reversion table."""

    return (
        ReversionCase(
            "D01",
            "restore hardlink or startup-.pth filtering",
            (_node_for_environment("E11"), _P12_NONE, _P12_STAT),
            "E11",
            _treat_checkpoints_as_runtime,
        ),
        ReversionCase(
            "D02",
            "omit content seal from generation",
            (_node_for_environment("E02"), _node_for_environment("E03")),
            "environment generation omits or rewrites the content seal",
            _omit_generation_content_seal,
        ),
        ReversionCase(
            "D03",
            "use conda-meta alone as seal",
            (_node_for_environment("E02"),),
            "content seal",
            _conda_meta_only_seal,
        ),
        ReversionCase(
            "D04",
            "grant post-seal bytes",
            (_node_for_environment("E04"), _node_for_environment("E08")),
            "DID NOT RAISE",
            _grant_post_seal_bytes,
        ),
        ReversionCase(
            "D05",
            "weaken internal/external symlink binding",
            tuple(
                _node_for_environment(cell) for cell in ("E05", "E06", "E07", "E08", "E09", "E10")
            ),
            "link_text",
            _weaken_symlink_binding,
        ),
        ReversionCase(
            "D06",
            "treat packaged checkpoints as runtime",
            (_node_for_environment("E12"), _node_for_environment("E13")),
            "checkpoint_or_weight_read_attempted",
            _treat_checkpoints_as_runtime,
        ),
        ReversionCase(
            "D07",
            "restore sys.executable or parallel interpreter",
            (_P01, _P09, _P10),
            "round19-selected-prefix",
            _restore_parent_interpreter,
        ),
        ReversionCase(
            "D08",
            "allow outside/mismatched selected interpreter",
            (_P13, _P19),
            "DID NOT RAISE",
            _allow_mismatched_interpreter,
        ),
        ReversionCase(
            "D09",
            "each substitution evasion",
            (_T01, _P01),
            "runtime-root",
            _restore_legacy_runtime_root,
        ),
        ReversionCase(
            "D10",
            "reintroduce live legacy root/fake result/hand binding",
            (_P01,),
            "runtime-root",
            _restore_legacy_runtime_root,
        ),
        ReversionCase(
            "D11",
            "remove transitive fixture/composition/CI scope",
            (_P01,),
            "_compile_worker_read_manifest",
            _remove_transitive_scope_symbol,
        ),
        ReversionCase(
            "D12",
            "omit ctime/inode/external fingerprint trigger",
            (_P02,),
            "DID NOT RAISE",
            _omit_fingerprint_fields,
        ),
        ReversionCase(
            "D13",
            "quarantine byte-identical hardlink clone churn",
            (_P02,),
            "content seal",
            _quarantine_hardlink_clone_churn,
        ),
        ReversionCase(
            "D14",
            "rewalk per model/consumer or reuse pass token for spawn",
            (_P03,),
            "stale spawn verification yielded",
            _reuse_currentness_for_spawn,
        ),
        ReversionCase(
            "D15",
            "delete shutdown guard or admit later work",
            tuple(_node_for_shutdown(f"S{index:02d}") for index in range(1, 14)),
            "post-award-commit",
            _delete_shutdown_guard,
        ),
        ReversionCase(
            "D16",
            "insert shutdown check inside atomic publication",
            (_node_for_shutdown("S11"),),
            "S11",
            _move_shutdown_inside_publication,
        ),
        ReversionCase(
            "D17",
            "restore root/suffix transport grant",
            (_P07, _P10),
            "assert not True",
            _restore_transport_suffix_grant,
        ),
        ReversionCase(
            "D18",
            "retain mismatched artifact transaction",
            (_node_for_handoff("H02"),),
            "DID NOT RAISE",
            _retain_mismatched_artifact_transaction,
        ),
        ReversionCase(
            "D19",
            "accept zero/multiple handoff finals or old-root fallback",
            (_node_for_handoff("H03"), _node_for_handoff("H04"), _node_for_handoff("H01")),
            "terminal partition invalid",
            _accept_ambiguous_handoff_finals,
        ),
        ReversionCase(
            "D20",
            "invalidate active cache on rejected rebind",
            (_P08,),
            "initial_counters",
            _invalidate_on_rejected_rebind,
        ),
        ReversionCase(
            "D21",
            "delete/corrupt either lock/export/probe contract",
            (_P09, _P10, _P11_CI),
            "round19-linux-64.lock",
            _delete_linux_lock,
        ),
        ReversionCase(
            "D22",
            "remove workflow proof node or permit skip/xfail",
            (_P11_CI,),
            "MENAGERIE_RELEASE_GATE",
            _permit_ci_skip,
        ),
        ReversionCase(
            "D23",
            "remove Linux bwrap/strace or denial audit",
            (_P09, _P14, _P11_CI),
            "strace",
            _remove_linux_denial_audit,
        ),
        ReversionCase(
            "D24",
            "replace macOS denial with profile-only proof",
            (_P10, _P15, _P11_CI),
            "seatbelt",
            _replace_macos_denial_with_profile,
        ),
        ReversionCase(
            "D25",
            "drop nullable evidence or misclassify mode",
            (_P17, _P12_NONE, _P12_STAT),
            "unverifiable",
            _misclassify_unverifiable_mode,
        ),
        ReversionCase(
            "D26",
            "restore dry-run false-complete/signature mismatch",
            (_P18A, _P18B),
            "dry_run",
            _restore_dry_run_false_complete,
        ),
        ReversionCase(
            "D27",
            "restore source/CAS/cache/fetch fallback",
            (_node_for_handoff("H01"),),
            "handoff-authority-unavailable",
            _restore_source_fallback,
        ),
        ReversionCase(
            "D28",
            "delete clause/finding/invariant/matrix registry record",
            (_CONFORMANCE_NODE, _P11_CI),
            "assert",
            _drop_first_registry_record,
        ),
        ReversionCase(
            "D29",
            "regress receipt/spawn/writer/schema preservation",
            (_P12_NONE, _P12_STAT, _P16, _P17),
            "raw_award_receipt_sha256",
            _regress_receipt_spawn_writer_schema,
        ),
    )


def _copy_checkout(source: Path, destination: Path) -> None:
    """Copy a repository tree into a disposable checkout destination."""

    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(
        ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"
    )
    shutil.copytree(source, destination, ignore=ignore)


def _run_pytest(root: Path, nodes: Sequence[str], python: str) -> subprocess.CompletedProcess[str]:
    """Run mapped proof nodes in a disposable checkout."""

    return subprocess.run(
        (python, "-m", "pytest", "-q", *nodes, "-x", "--tb=short"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )


def _run_case(
    case: ReversionCase,
    *,
    source: Path,
    work_root: Path,
    python: str,
) -> dict[str, object]:
    """Apply one reversion in a fresh disposable checkout and run its mapped proof."""

    checkout = work_root / case.reversion_id
    _copy_checkout(source, checkout)
    case.mutate(checkout, case.reversion_id)
    completed = _run_pytest(checkout, case.proof_nodes, python)
    combined = completed.stdout + "\n" + completed.stderr
    setup_failed = "ERROR at setup" in combined or "unmet-release-gate:" in combined
    passed = completed.returncode != 0 and not setup_failed and case.expected_reason in combined
    return {
        "reversion_id": case.reversion_id,
        "semantic_reversion": case.semantic_reversion,
        "proof_nodes": list(case.proof_nodes),
        "expected_reason": case.expected_reason,
        "exit_code": completed.returncode,
        "passed": passed,
        "setup_failed": setup_failed,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _write_matrix(path: Path) -> None:
    """Write the public exact D01-D29 matrix consumed by P11."""

    payload = {
        "schema_version": "menagerie.crawler.round21-reversion-matrix.v1",
        "deliberate_reversion_ids": list(_REVERSIONS),
        "cases": [
            {
                "reversion_id": case.reversion_id,
                "semantic_reversion": case.semantic_reversion,
                "proof_nodes": list(case.proof_nodes),
                "expected_reason": case.expected_reason,
            }
            for case in _cases()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ids", nargs="*", default=list(_REVERSIONS))
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--write-matrix", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested deliberate reversions and write an attestation."""

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.write_matrix is not None:
        _write_matrix(args.write_matrix)
        return 0
    source = args.source.resolve()
    work_root = args.work_root.resolve()
    if source == work_root or source in work_root.parents:
        raise SystemExit("work-root must be outside the source checkout")
    requested = tuple(args.ids)
    cases = {case.reversion_id: case for case in _cases()}
    if set(requested) - set(cases):
        raise SystemExit(f"unknown reversion id(s): {sorted(set(requested) - set(cases))}")
    work_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(cases[reversion_id], source=source, work_root=work_root, python=str(args.python))
        for reversion_id in requested
    ]
    passed = [str(result["reversion_id"]) for result in results if result["passed"] is True]
    failed = [str(result["reversion_id"]) for result in results if result["passed"] is not True]
    payload = {
        "schema_version": "menagerie.crawler.round21-reversion-result.v1",
        "status": "passed" if not failed and set(passed) == set(requested) else "failed",
        "requested_reversions": list(requested),
        "passed_reversions": sorted(passed),
        "skipped_reversions": [],
        "failed_reversions": sorted(failed),
        "results": results,
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
