"""Deterministic Round-21 deliberate-reversion runner for disposable checkouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Callable, Sequence


_REVERSIONS = tuple(f"D{index:02d}" for index in range(1, 30))
_REGISTRY_PATH = Path("menagerie/crawler/conformance-round21.json")
_WORKFLOW_PATH = Path(".github/workflows/tests.yml")
_LINUX_LOCK_PATH = Path("menagerie/crawler/envs/locks/round19-linux-64.lock")
_AUTHORITY_PATH = Path("menagerie/crawler/authority.py")
_DRIVER_PATH = Path("menagerie/crawler/driver.py")
_DRIVER_ADMISSION_PATH = Path("menagerie/crawler/driver_admission.py")
_DRIVER_RECEIPTS_PATH = Path("menagerie/crawler/driver_receipts.py")
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
_WORKFLOW_INVENTORY_NODE = (
    "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
    "test_conformance_workflow_and_reversion_inventory_is_exact"
)
_HOST_RELEASE_INVENTORY_NODE = (
    "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
    "test_supported_host_release_gate_inventory_is_exact"
)
_LINUX_RELEASE_ARTIFACTS_NODE = (
    "menagerie/crawler/tests/test_anti_substitution_inventories.py::"
    "test_linux_release_artifacts_and_provisioning_are_real"
)
_CASE_TIMEOUT_SECONDS = 900


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
_P12_NONE = (
    "menagerie/crawler/tests/test_award_worker_result_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[none]"
)
_P12_STAT = (
    "menagerie/crawler/tests/test_award_worker_result_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer[statistical]"
)
_P13 = (
    "menagerie/crawler/tests/test_environment_authority_composition.py::"
    "test_outside_selected_interpreter_is_rejected_at_binding"
)
_P14 = (
    "menagerie/crawler/tests/test_environment_authority_composition.py::"
    "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
_P15 = (
    "menagerie/crawler/tests/test_environment_authority_composition.py::"
    "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
)
_P16 = (
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes"
)
_P17 = (
    "menagerie/crawler/tests/test_award_worker_result_composition.py::"
    "test_real_unhashable_output_awards_runs_with_unverifiable_modes"
)
_P18A = (
    "menagerie/crawler/tests/test_execution_dry_run_composition.py::"
    "test_documented_dry_run_and_resume_use_real_environment"
)
_P18B = (
    "menagerie/crawler/tests/test_execution_dry_run_composition.py::"
    "test_dry_run_all_source_failure_is_acceptance_error"
)
_P19 = (
    "menagerie/crawler/tests/test_environment_authority_composition.py::"
    "test_manifest_v3_rejects_changed_interpreter_association"
)
_P27 = (
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_code_less_deferral_fails_visibly_without_failed_source"
)


@dataclass(frozen=True)
class ReversionCase:
    """One deterministic disposable-checkout deliberate reversion."""

    reversion_id: str
    semantic_reversion: str
    proof_nodes: tuple[str, ...]
    expected_reasons: tuple[str, ...]
    mutate: Callable[[Path, str], None]

    @property
    def expected_reason(self) -> str:
        """Return the legacy single expected-reason field for matrix readers."""

        return self.expected_reasons[0]


def _case(
    reversion_id: str,
    semantic_reversion: str,
    proof_nodes: tuple[str, ...],
    expected_reason: str | tuple[str, ...],
    mutate: Callable[[Path, str], None],
) -> ReversionCase:
    """Build one reversion case with one registered reason per mapped proof node."""

    expected_reasons = (
        (expected_reason,) * len(proof_nodes)
        if isinstance(expected_reason, str)
        else expected_reason
    )
    if len(expected_reasons) != len(proof_nodes):
        raise ValueError(f"{reversion_id} must register one reason per proof node")
    return ReversionCase(
        reversion_id,
        semantic_reversion,
        proof_nodes,
        expected_reasons,
        mutate,
    )


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
    _ignore_content_staleness(root, _reversion_id)


def _grant_post_seal_bytes(root: Path, _reversion_id: str) -> None:
    """Let undeclared post-seal runtime files read through policy."""

    _ignore_content_staleness(root, _reversion_id)
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
        """    suffix = path.suffix.lower()
    if suffix in _MODEL_DATA_SUFFIXES:
""",
        """    suffix = path.suffix.lower()
    if False and suffix in _MODEL_DATA_SUFFIXES:
""",
    )


def _block_startup_pth_filter(root: Path, _reversion_id: str) -> None:
    """Stop treating sealed textual startup ``.pth`` files as safe import metadata."""

    _replace_once(
        root,
        _POLICY_PATH,
        """    if path.suffix.lower() == ".pth" and path.parent.name in {
        "site-packages",
        "dist-packages",
    }:
        try:
            with _ORIGINAL_IO_OPEN(path, "rb") as handle:
                data = handle.read(1024**2 + 1)
        except OSError:
            return False
        return len(data) <= 1024**2 and b"\\x00" not in data
""",
        """    if False and path.suffix.lower() == ".pth" and path.parent.name in {
        "site-packages",
        "dist-packages",
    }:
        try:
            with _ORIGINAL_IO_OPEN(path, "rb") as handle:
                data = handle.read(1024**2 + 1)
        except OSError:
            return False
        return len(data) <= 1024**2 and b"\\x00" not in data
""",
    )


def _restore_parent_interpreter(root: Path, _reversion_id: str) -> None:
    """Run workers with the parent interpreter instead of the selected prefix interpreter."""

    _replace_once(
        root,
        _DRIVER_ADMISSION_PATH,
        "        interpreter = authority.selected_interpreter\n",
        "        interpreter = Path(sys.executable)\n",
    )


def _allow_mismatched_interpreter(root: Path, _reversion_id: str) -> None:
    """Accept changed selected-interpreter associations."""

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
        """        if False and (
            current.content_manifest_sha256 != authority.content_manifest_sha256
            or current.selected_interpreter_digest != authority.selected_interpreter_digest
        ):
            self.invalidations += 1
            self._manifest = None
            self._authority = None
            raise AuthorityDerivationError("environment content seal changed; authority is stale")
""",
    )
    _replace_once(
        root,
        _AUTHORITY_PATH,
        """    relative = path.relative_to(prefix).as_posix()
    normalized = unicodedata.normalize("NFC", relative)
    if relative != normalized or not relative or relative.startswith("/"):
        raise AuthorityDerivationError(f"environment member path is noncanonical: {relative!r}")
    return relative
""",
        """    try:
        relative = path.relative_to(prefix).as_posix()
    except ValueError:
        relative = path.as_posix().lstrip("/")
    normalized = unicodedata.normalize("NFC", relative)
    if relative != normalized or not relative or relative.startswith("/"):
        raise AuthorityDerivationError(f"environment member path is noncanonical: {relative!r}")
    return relative
""",
    )
    _replace_once(
        root,
        _AUTHORITY_PATH,
        """    if not lexical_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter is outside the canonical prefix")
""",
        """    if False and not lexical_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter is outside the canonical prefix")
""",
    )
    _replace_once(
        root,
        _AUTHORITY_PATH,
        """    if not resolved_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
""",
        """    if False and not resolved_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
""",
    )
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
        _DRIVER_RECEIPTS_PATH,
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
        _DRIVER_RECEIPTS_PATH,
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
        _DRIVER_RECEIPTS_PATH,
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
        _DRIVER_ADMISSION_PATH,
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
        _DRIVER_ADMISSION_PATH,
        """        if (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            raise DriverIntegrationError("handoff-authority-unavailable")
""",
        """        if False and (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            raise DriverIntegrationError("handoff-authority-unavailable")
""",
    )
    _replace_once(
        root,
        _DRIVER_ADMISSION_PATH,
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
        _case(
            "D01",
            "restore hardlink or startup-.pth filtering",
            (_node_for_environment("E11"), _P12_NONE, _P12_STAT),
            (
                # E11 directly asserts the sealed startup .pth stayed non-checkpoint;
                # blocking the filter flips checkpoint_or_weight_read_attempted to True.
                "checkpoint_or_weight_read_attempted",
                # P12 none/statistical award through the shared startup-.pth prefix; the
                # blocked .pth read fails the real forward so no run is awarded.
                "'awards_runs': False",
                "'awards_runs': False",
            ),
            _block_startup_pth_filter,
        ),
        _case(
            "D02",
            "omit content seal from generation",
            (_node_for_environment("E02"), _node_for_environment("E03")),
            (
                # E02 rebinds and re-verifies: the seal-less generation is caught by
                # verify_environment_authority.
                "environment generation omits or rewrites the content seal",
                # E03 expects a stale-byte refusal; with the content seal omitted the
                # changed prefix no longer stales, so cache.verify does not raise.
                "DID NOT RAISE",
            ),
            _omit_generation_content_seal,
        ),
        _case(
            "D03",
            "use conda-meta alone as seal",
            (_node_for_environment("E02"),),
            "content seal",
            _conda_meta_only_seal,
        ),
        _case(
            "D04",
            "grant post-seal bytes",
            (_node_for_environment("E04"), _node_for_environment("E08")),
            (
                # E04 adds a post-seal regular module; granting post-seal bytes stops the
                # stale-before-spawn refusal from firing.
                "DID NOT RAISE",
                # E08 adds a post-seal external symlink; with the content-seal staleness
                # granted the widened target is still caught by the static import closure.
                "static import is outside sealed environment and exact inventory",
            ),
            _grant_post_seal_bytes,
        ),
        _case(
            "D05",
            "weaken internal/external symlink binding",
            # Only E09 isolates the symlink text/target seal: it retargets an internal
            # symlink between two equal-length in-prefix members whose bytes are unchanged,
            # so dropping link_text/resolved_target_relative_path makes the retarget
            # invisible and the stale-before-spawn refusal no longer fires. E05/E06 are
            # positive awards; E07 is caught by the external-target content hash; E08 by
            # new-entry detection; E10 by the non-file-target check -- none exercise the
            # symlink-text binding this reversion weakens.
            (_node_for_environment("E09"),),
            "DID NOT RAISE",
            _weaken_symlink_binding,
        ),
        _case(
            "D06",
            "treat packaged checkpoints as runtime",
            (_node_for_environment("E12"), _node_for_environment("E13")),
            'assert attempt["result"] == "failed"',
            _treat_checkpoints_as_runtime,
        ),
        _case(
            "D07",
            "restore sys.executable or parallel interpreter",
            (_P01, _P09),
            (
                "selected interpreter is missing from worker argv",
                'assert fixture.binding.python_executable == fixture.prefix / "bin/python"',
            ),
            _restore_parent_interpreter,
        ),
        _case(
            "D08",
            "allow outside/mismatched selected interpreter",
            (_P13, _P19),
            (
                "selected interpreter is absent from the content seal",
                "DID NOT RAISE",
            ),
            _allow_mismatched_interpreter,
        ),
        _case(
            "D09",
            "each substitution evasion",
            # P01 is the real-path proof: it greps production sources for the legacy
            # runtime-root token. T01 exercises the AST substitution tripwire, which this
            # source-token reversion does not defeat, so it cannot red here (the registry
            # already scopes D09 to P01).
            (_P01,),
            "runtime-root",
            _restore_legacy_runtime_root,
        ),
        _case(
            "D10",
            "reintroduce live legacy root/fake result/hand binding",
            (_P01,),
            "runtime-root",
            _restore_legacy_runtime_root,
        ),
        _case(
            "D11",
            "remove transitive fixture/composition/CI scope",
            (_P01,),
            "_compile_worker_read_manifest",
            _remove_transitive_scope_symbol,
        ),
        _case(
            "D12",
            "omit ctime/inode/external fingerprint trigger",
            (_P02,),
            "DID NOT RAISE",
            _omit_fingerprint_fields,
        ),
        _case(
            "D13",
            "quarantine byte-identical hardlink clone churn",
            (_P02,),
            "content seal",
            _quarantine_hardlink_clone_churn,
        ),
        _case(
            "D14",
            "rewalk per model/consumer or reuse pass token for spawn",
            (_P03,),
            "failed:environment",
            _reuse_currentness_for_spawn,
        ),
        _case(
            "D15",
            "delete shutdown guard or admit later work",
            tuple(_node_for_shutdown(f"S{index:02d}") for index in range(1, 14)),
            "post-award-commit",
            _delete_shutdown_guard,
        ),
        _case(
            "D16",
            "insert shutdown check inside atomic publication",
            (_node_for_shutdown("S11"),),
            "assert publication < append < post_award_hook < post_award_guard",
            _move_shutdown_inside_publication,
        ),
        _case(
            "D17",
            "restore root/suffix transport grant",
            (_P07,),
            "assert not True",
            _restore_transport_suffix_grant,
        ),
        _case(
            "D18",
            "retain mismatched artifact transaction",
            (_node_for_handoff("H02"),),
            "DID NOT RAISE",
            _retain_mismatched_artifact_transaction,
        ),
        _case(
            "D19",
            "accept zero/multiple handoff finals or old-root fallback",
            # H03 (zero finals) and H04 (ambiguous finals) both exercise the accepted
            # handoff-final guard. H01 is the valid single-final positive control, which
            # this reversion leaves awarding, so it cannot red and is scoped out.
            (_node_for_handoff("H03"), _node_for_handoff("H04")),
            (
                # H03: accepting zero finals lets the selection continue, and the
                # unterminalized work item fails the downstream terminal partition.
                "terminal partition invalid",
                # H04: accepting the duplicate final proceeds into artifact-chain
                # validation, which rejects the duplicated event ID.
                "duplicate or invalid artifact event ID",
            ),
            _accept_ambiguous_handoff_finals,
        ),
        _case(
            "D20",
            "invalidate active cache on rejected rebind",
            (_P08,),
            "environment authority cache association is invalid",
            _invalidate_on_rejected_rebind,
        ),
        _case(
            "D21",
            "delete/corrupt either lock/export/probe contract",
            (_LINUX_RELEASE_ARTIFACTS_NODE, _P09),
            ("Extra items in the right set", "Extra items in the right set"),
            _delete_linux_lock,
        ),
        _case(
            "D22",
            "remove workflow proof node or permit skip/xfail",
            (_WORKFLOW_INVENTORY_NODE,),
            "MENAGERIE_RELEASE_GATE",
            _permit_ci_skip,
        ),
        _case(
            "D23",
            "remove Linux bwrap/strace or denial audit",
            (_HOST_RELEASE_INVENTORY_NODE,),
            "strace",
            _remove_linux_denial_audit,
        ),
        _case(
            "D24",
            "replace macOS denial with profile-only proof",
            (_HOST_RELEASE_INVENTORY_NODE,),
            "command -v log",
            _replace_macos_denial_with_profile,
        ),
        _case(
            "D25",
            "drop nullable evidence or misclassify mode",
            # Only P17 (digestless unhashable output) enters the unverifiable branch this
            # reversion misclassifies. P12 none/statistical carry real value digests and
            # take the verified branch, so the misclassification is a no-op for them.
            (_P17,),
            "unverifiable",
            _misclassify_unverifiable_mode,
        ),
        _case(
            "D26",
            "restore dry-run false-complete/signature mismatch",
            (_P18B,),
            "assert failed.returncode == EXIT_ERROR",
            _restore_dry_run_false_complete,
        ),
        _case(
            "D27",
            "restore source/CAS/cache/fetch fallback",
            (_P27,),
            "menagerie.crawler.status.PartitionError",
            _restore_source_fallback,
        ),
        _case(
            "D28",
            "delete clause/finding/invariant/matrix registry record",
            (_CONFORMANCE_NODE,),
            "conformance registry clause IDs changed",
            _drop_first_registry_record,
        ),
        _case(
            "D29",
            "regress receipt/spawn/writer/schema preservation",
            (_P12_NONE, _P12_STAT, _P16, _P17),
            (
                # P12 none/statistical and P17 award real workers whose receipts now carry
                # a null raw-award digest, tripping the supervisor's partial-receipt guard.
                "receipt_error",
                "receipt_error",
                # P16 supersedes a real deferred handoff; the regressed receipt digest
                # breaks the award so the run never reaches the terminal-partition-complete
                # status it asserts.
                "terminal-partition-complete",
                "receipt_error",
            ),
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


def _run_pytest_node(root: Path, node: str, python: str) -> subprocess.CompletedProcess[str]:
    """Run one mapped proof node in a disposable checkout."""

    process = subprocess.Popen(
        (python, "-m", "pytest", "-q", node, "--tb=short"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_CASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=exc.cmd,
            timeout=exc.timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(
        args=process.args,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
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
    node_results = []
    for node, expected_reason in zip(case.proof_nodes, case.expected_reasons, strict=True):
        timed_out = False
        try:
            completed = _run_pytest_node(checkout, node, python)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            completed = subprocess.CompletedProcess(
                args=exc.cmd,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\nTimeoutExpired after {_CASE_TIMEOUT_SECONDS}s",
            )
        combined = completed.stdout + "\n" + completed.stderr
        setup_failed = "ERROR at setup" in combined or "unmet-release-gate:" in combined
        node_selector = node.split("::", maxsplit=1)[-1]
        node_function = node_selector.split("[", maxsplit=1)[0]
        diagnostic = "\n".join(
            line
            for line in combined.splitlines()
            if node not in line and node_selector not in line and node_function not in line
        )
        node_passed = (
            completed.returncode != 0
            and not timed_out
            and not setup_failed
            and expected_reason in diagnostic
        )
        node_results.append(
            {
                "proof_node": node,
                "expected_reason": expected_reason,
                "exit_code": completed.returncode,
                "passed": node_passed,
                "setup_failed": setup_failed,
                "timed_out": timed_out,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
    passed = all(result["passed"] is True for result in node_results)
    setup_failed = any(result["setup_failed"] is True for result in node_results)
    timed_out = any(result["timed_out"] is True for result in node_results)
    return {
        "reversion_id": case.reversion_id,
        "semantic_reversion": case.semantic_reversion,
        "proof_nodes": list(case.proof_nodes),
        "expected_reason": case.expected_reason,
        "expected_reasons": list(case.expected_reasons),
        "exit_code": 1 if passed else 0,
        "passed": passed,
        "setup_failed": setup_failed,
        "timed_out": timed_out,
        "node_results": node_results,
        "stdout_tail": "\n".join(str(result["stdout_tail"]) for result in node_results)[-2000:],
        "stderr_tail": "\n".join(str(result["stderr_tail"]) for result in node_results)[-2000:],
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
                "expected_reasons": list(case.expected_reasons),
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
    setup_failed = [
        str(result["reversion_id"]) for result in results if result["setup_failed"] is True
    ]
    timed_out = [str(result["reversion_id"]) for result in results if result["timed_out"] is True]
    payload = {
        "schema_version": "menagerie.crawler.round21-reversion-result.v1",
        "status": "passed" if not failed and set(passed) == set(requested) else "failed",
        "requested_reversions": list(requested),
        "passed_reversions": sorted(passed),
        "skipped_reversions": [],
        "failed_reversions": sorted(failed),
        "setup_failed_reversions": sorted(setup_failed),
        "timed_out_reversions": sorted(timed_out),
        "results": results,
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
