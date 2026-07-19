"""Round-17 class-level prevention and anti-theater CI inventories."""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pytest

import menagerie.crawler.artifact_transactions as artifact_module
import menagerie.crawler.authority as authority_module
import menagerie.crawler.driver as driver_module
import menagerie.crawler.env_lifecycle as lifecycle_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.schema import (
    SchemaOwner,
    load_schema,
    owned_schema_leaves,
    schema_leaf_paths,
)
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_supervised_worker_result_v3,
    make_worker_result_v3_mapping,
)
from menagerie.crawler.tests.test_round17_vs1_v3_composition import VS1_LANDING_MANIFEST
from menagerie.crawler.tests.test_round17_vs2_shutdown_composition import VS2_LANDING_MANIFEST
from menagerie.crawler.tests.test_round17_vs3_authority_composition import VS3_LANDING_MANIFEST
from menagerie.crawler.worker_supervisor import SupervisorObservation


_CRAWLER_ROOT = Path(driver_module.__file__).parent
_REPOSITORY_ROOT = _CRAWLER_ROOT.parents[1]
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "tests.yml"
_PROTOCOL_VERSIONS = {
    "_WORKER_RESULT_VERSION": "menagerie.crawler.worker-result.v3",
    "_WORKER_DIAGNOSTIC_VERSION": "menagerie.crawler.worker-receipt.v1",
    "_RAW_AWARD_RECEIPT_VERSION": "menagerie.crawler.raw-award-receipt.v3",
}
_DEAD_SYMBOLS = {
    "_driver_deferral_attempt",
    "store_licensed_artifact",
    "_supervise_environment_worker",
    "_validate_terminal_evidence",
    "validate_reconstruction_source_binding",
}
_DEAD_OPTIONS = {"--scheduled-wake"}
_SENSITIVE_EDGE_COUNTS = Counter(
    {
        ("CrawlerDriver._ensure_authors", "dependencies.author.author"): 1,
        ("CrawlerDriver._repair_author", "dependencies.author.author"): 1,
        ("CrawlerDriver._repair_author_for_detected_modes", "dependencies.author.author"): 1,
        ("CrawlerDriver._ensure_gates", "dependencies.checker.check_metadata"): 2,
        ("CrawlerDriver._ensure_gates", "dependencies.checker.check_fidelity"): 1,
        ("CrawlerDriver._run_environment_work", "dependencies.environments.run"): 1,
        ("SupervisedForwardLane.forward", "open_worker_lease"): 1,
        ("SupervisedForwardLane.forward", "supervise_worker"): 1,
        ("CrawlerDriver._forward_and_reduce", "dependencies.forward.forward"): 2,
        ("CrawlerDriver._authorize_and_publish_artifact", "publish_authorized_artifact"): 1,
        ("CrawlerDriver._authorize_terminal_artifact", "publish_authorized_artifact"): 1,
        ("CrawlerDriver._forward_and_reduce", "append_model"): 1,
        ("CrawlerDriver._terminalize", "append_model"): 1,
    }
)
_SENSITIVE_SUFFIXES = frozenset(suffix for _owner, suffix in _SENSITIVE_EDGE_COUNTS)
_ROUND17_CI_NODES = (
    "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
    "test_real_v3_worker_result_awards_through_driver_and_reducer",
    "menagerie/crawler/tests/test_round17_vs2_shutdown_composition.py::"
    "test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes",
    "menagerie/crawler/tests/test_slice_f_driver.py::"
    "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes",
)
_ROUND19_RELEASE_NODE_INVENTORY = {
    "golden": _ROUND17_CI_NODES[0],
    "interpreter": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_outside_selected_interpreter_is_rejected_at_binding"
    ),
    "linux-denial": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_linux_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
    ),
    "shutdown": _ROUND17_CI_NODES[1],
    "clean-clone": _ROUND17_CI_NODES[2],
    "unverifiable": (
        "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
        "test_real_unhashable_output_awards_runs_with_unverifiable_modes"
    ),
    "dry-run-run-resume": (
        "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
        "test_documented_dry_run_and_resume_use_real_environment"
    ),
    "dry-run-false-complete": (
        "menagerie/crawler/tests/test_round19_vs6_dry_run_composition.py::"
        "test_dry_run_all_source_failure_is_acceptance_error"
    ),
    "cache": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_real_multi_model_cache_closes_currentness_and_quarantines_mutation"
    ),
    "structural": "menagerie/crawler/tests/test_round17_structural_inventories.py",
    "macos-positive-negative": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_macos_real_compiler_denies_caught_undeclared_repo_read_and_awards_package"
    ),
    "macos-profile": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_macos_v3_profile_has_one_fresh_literal_prefix_and_exact_outside_members"
    ),
}
_REQUIRED_CI_SELECTIONS = (
    "menagerie/crawler/tests -m round21_linux_real",
    _ROUND19_RELEASE_NODE_INVENTORY["macos-positive-negative"],
    _ROUND19_RELEASE_NODE_INVENTORY["macos-profile"],
    _ROUND19_RELEASE_NODE_INVENTORY["dry-run-run-resume"],
    _ROUND19_RELEASE_NODE_INVENTORY["dry-run-false-complete"],
)
_SUBSTITUTION_BOUNDARIES = frozenset(
    {
        "_compile_worker_read_manifest",
        "_collect_worker_executable_closure",
        "_execution_identity",
        "_attempts_from_supervised",
        "_verified_worker_result",
        "_read_verified_worker_receipt",
        "_seal_environment_content",
        "materialized_environment_generation",
        "verify_environment_authority",
        "bind_materialized_environment",
        "EnvironmentAuthorityCache",
        "compile_execution_read_manifest_v3",
        "compile_execution_read_manifest_v3_from_closure",
        "collect_executable_closure_v3",
        "environment_read_capability",
        "verify_execution_read_manifest",
        "verify_execution_read_manifest_v3",
        "supervise_worker",
        "run_isolated_subprocess",
        "verify_supervised_worker_result",
        "derive_parent_attestation",
        "derive_attempt_projection",
        "SupervisorObservation",
        "VerifiedWorkerResult",
        "PublicationAuthorization",
        "_assemble_run_model",
        "_authorize_and_publish_artifact",
        "_authorize_terminal_artifact",
        "append_attempt",
        "append_model",
        "prepare_model",
        "CanonicalReducer",
        "publish_authorized_artifact",
    }
)
_LEGACY_ALTERNATE_COMPILERS = frozenset(
    {"compile_execution_read_manifest", "compile_execution_read_manifest_v2"}
)
_FAKE_EXECUTION_TYPES = frozenset({"FakeEnvironments", "FakeForward", "SupervisedResult"})
_QUARANTINE_PATH = _CRAWLER_ROOT / "legacy_manifest_audit.py"


def _workflow_test_paths(workflow_source: str) -> tuple[Path, ...]:
    """Return every crawler test path selected textually by CI.

    Parameters
    ----------
    workflow_source:
        Complete workflow YAML text.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing selected crawler test modules in stable order.
    """

    relative_paths = {
        token.split("::", 1)[0]
        for token in re.findall(r"menagerie/crawler/tests/[A-Za-z0-9_./:-]+", workflow_source)
    }
    return tuple(
        sorted(
            path for relative in relative_paths if (path := _REPOSITORY_ROOT / relative).is_file()
        )
    )


def _registered_composition_roots() -> dict[Path, tuple[str, ...]]:
    """Return exact landing-manifest real nodes grouped by their source path.

    Returns
    -------
    dict[pathlib.Path, tuple[str, ...]]
        Registered source paths and exact top-level pytest function roots.
    """

    default_paths = (
        (VS1_LANDING_MANIFEST, _CRAWLER_ROOT / "tests" / "test_round17_vs1_v3_composition.py"),
        (
            VS2_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_round17_vs2_shutdown_composition.py",
        ),
        (
            VS3_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_round17_vs3_authority_composition.py",
        ),
    )
    grouped: dict[Path, set[str]] = {}
    for manifest, default_path in default_paths:
        for raw_node in manifest["real_composition_nodes"]:
            node_id = str(raw_node)
            if "::" in node_id:
                path_text, function_name = node_id.split("::", 1)
                path = _REPOSITORY_ROOT / path_text
            else:
                path = default_path
                function_name = node_id
            grouped.setdefault(path, set()).add(function_name.split("[", 1)[0])
    return {path: tuple(sorted(roots)) for path, roots in grouped.items()}


def _composition_source_paths(workflow_source: str | None = None) -> tuple[Path, ...]:
    """Discover the complete §8.3 composition and fixture-module scope.

    Parameters
    ----------
    workflow_source:
        Optional workflow mutation used by self-proofs.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Composition modules, transitive fixture/support modules, and CI selections.
    """

    workflow = (
        _WORKFLOW_PATH.read_text(encoding="utf-8") if workflow_source is None else workflow_source
    )
    test_root = _CRAWLER_ROOT / "tests"
    paths = {
        _CRAWLER_ROOT / "cli.py",
        test_root / "conftest.py",
        test_root / "dry_run_support.py",
        test_root / "test_slice_f_driver.py",
        *test_root.glob("test_*composition*.py"),
        *_workflow_test_paths(workflow),
        *_registered_composition_roots(),
    }
    return tuple(sorted(path for path in paths if path.is_file()))


_COMPOSITION_SOURCES = _composition_source_paths()

VS4_LANDING_MANIFEST: dict[str, Any] = {
    "findings": ("SOL-R16-07", "Fable-F5"),
    "production_symbols": {
        "driver": ("_SHUTDOWN_ADMISSION_REGISTRY", "_AWARD_CLOSURE_SYMBOLS"),
        "worker_supervisor": (
            "_SUBPROCESS_SPAWN_REGISTRY",
            "run_isolated_subprocess",
            "verify_supervised_worker_result",
        ),
        "artifact_transactions": (
            "_PUBLIC_WRITER_REGISTRY",
            "publish_authorized_artifact",
            "resolve_final_artifact_transaction",
        ),
    },
    "real_composition_nodes": _ROUND17_CI_NODES,
    "structural_nodes": (
        "test_spawn_inventory_is_closed_and_lease_bearing",
        "test_admission_boundary_inventory_is_closed",
        "test_public_writer_inventory_is_authorization_bearing",
        "test_dead_symbol_inventory_is_closed",
        "test_attempt_schema_consumers_and_producers_have_exact_parity",
        "test_protocol_literal_inventory_has_only_reviewed_comparison_owners",
        "test_synthetic_worker_result_factory_matches_production_protocol_shapes",
        "test_round17_real_compositions_are_explicitly_selected_in_ci",
    ),
}

ROUND21_VS1_PROOF_REGISTRY: dict[str, str] = {
    "P01": (
        "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
        "test_round21_preclusion_real_v3_path_has_no_substitutable_fixture_edge"
    ),
    "T01": (
        "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
        "test_round21_tripwire_catches_python_evasion"
    ),
    "T01-CI": (
        "menagerie/crawler/tests/test_round21_preclusion_composition.py::"
        "test_round21_tripwire_catches_deleted_ci_node"
    ),
    "T02": (
        "menagerie/crawler/tests/test_round17_structural_inventories.py::"
        "test_legacy_manifest_v1_is_quarantined_from_every_live_import_graph"
    ),
}

ROUND21_VS2_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS1_PROOF_REGISTRY,
    "P02": (
        "menagerie/crawler/tests/test_round21_fingerprint_composition.py::"
        "test_round21_cheap_fingerprint_catches_stat_preserved_mutation_without_false_staling_clone"
    ),
}

ROUND21_VS3_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS2_PROOF_REGISTRY,
    "P03": (
        "menagerie/crawler/tests/test_round21_scale_composition.py::"
        "test_round21_pass_and_spawn_validation_walks_are_constant_bounded"
    ),
    "T03": (
        "menagerie/crawler/tests/test_round17_structural_inventories.py::"
        "test_round21_verification_tree_walk_inventory_is_closed"
    ),
}

ROUND21_VS4_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS3_PROOF_REGISTRY,
    "P04": (
        "menagerie/crawler/tests/test_round21_environment_matrix_composition.py::"
        "test_round21_environment_unit_matrix"
    ),
    "P12": _ROUND17_CI_NODES[0],
    "P13": _ROUND19_RELEASE_NODE_INVENTORY["interpreter"],
    "P14": _ROUND19_RELEASE_NODE_INVENTORY["linux-denial"],
    "P17": _ROUND19_RELEASE_NODE_INVENTORY["unverifiable"],
    "P19": (
        "menagerie/crawler/tests/test_round19_environment_authority_composition.py::"
        "test_manifest_v3_rejects_changed_interpreter_association"
    ),
}

ROUND21_VS5_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS4_PROOF_REGISTRY,
    "P05": (
        "menagerie/crawler/tests/test_round21_shutdown_matrix_composition.py::"
        "test_round21_shutdown_matrix"
    ),
}

ROUND21_VS6_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS5_PROOF_REGISTRY,
    "P06": (
        "menagerie/crawler/tests/test_round21_handoff_authority_composition.py::"
        "test_round21_handoff_authority_identity_matrix"
    ),
}

ROUND21_VS7_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS6_PROOF_REGISTRY,
    "P07": (
        "menagerie/crawler/tests/test_round21_transport_composition.py::"
        "test_round21_closed_transport_capability_awards_and_rejects_unlisted_library"
    ),
}

ROUND21_VS8_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS7_PROOF_REGISTRY,
    "P08": (
        "menagerie/crawler/tests/test_round21_cache_rebind_composition.py::"
        "test_round21_mismatched_rebind_preserves_active_authority_and_awards"
    ),
}

ROUND21_VS9_PROOF_REGISTRY: dict[str, str] = {
    **ROUND21_VS8_PROOF_REGISTRY,
    "P09": (
        "menagerie/crawler/tests/test_round21_ci_composition.py::"
        "test_round21_linux_committed_lock_provenance_awards_in_ci"
    ),
    "T04": (
        "menagerie/crawler/tests/test_round17_structural_inventories.py::"
        "test_round21_linux_release_artifacts_and_provisioning_are_real"
    ),
    "T05": (
        "menagerie/crawler/tests/test_round17_structural_inventories.py::"
        "test_round21_linux_release_registry_is_exact"
    ),
}


def test_round21_verification_tree_walk_inventory_is_closed() -> None:
    """Every complete prefix walk and v3 reuse site stays explicitly registered."""

    authority_source = _source(authority_module)
    tree, parents, names = _tree_context(authority_source)
    observed_walks = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_scan_environment_tree"
    )
    assert observed_walks == Counter(authority_module._ENVIRONMENT_TREE_WALK_REGISTRY)  # noqa: SLF001

    required_token_sites = {
        "authority.py": {
            "collect_executable_closure_v3",
            "compile_execution_read_manifest_v3_from_closure",
            "environment_read_capability",
            "verify_execution_read_manifest_v3",
        },
        "driver.py": {
            "CrawlerDriver._forward_and_reduce",
            "CrawlerDriver._run_environment_work",
            "SupervisedForwardLane.forward",
            "_collect_worker_executable_closure",
            "_compile_worker_read_manifest",
        },
        "policy.py": {"generate_macos_sandbox_profile", "verify_execution_read_manifest"},
        "worker_supervisor.py": {
            "_request_allowed_read_paths",
            "run_isolated_subprocess",
            "supervise_worker",
        },
    }
    modules = {
        "authority.py": authority_module,
        "driver.py": driver_module,
        "policy.py": __import__("menagerie.crawler.policy", fromlist=["policy"]),
        "worker_supervisor.py": supervisor_module,
    }
    verification_calls = {
        "cache.verify",
        "_collect_worker_executable_closure",
        "_compile_worker_read_manifest",
        "_current_run_is_fresh",
        "_forward_and_reduce",
        "collect_executable_closure_v3",
        "compile_execution_read_manifest_v3",
        "compile_execution_read_manifest_v3_from_closure",
        "environment_read_capability",
        "supervise_worker",
        "verify_environment_authority",
        "verify_execution_read_manifest",
        "verify_execution_read_manifest_v3",
    }
    for filename, required_owners in required_token_sites.items():
        source = _source(modules[filename])
        module_tree, module_parents, module_names = _tree_context(source)
        token_owners = {
            _enclosing_definition(node, module_parents, module_names)
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Call)
            and any(_attribute_name(node.func).endswith(name) for name in verification_calls)
            and any(keyword.arg == "verification_token" for keyword in node.keywords)
        }
        assert required_owners <= token_owners


def _source(module: Any) -> str:
    """Return one imported production module's exact source text."""

    return Path(module.__file__).read_text(encoding="utf-8")


def _attribute_name(node: ast.expr) -> str:
    """Return a dotted name for a simple name or attribute expression."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _definition_names(tree: ast.Module) -> dict[ast.AST, str]:
    """Map every function node to its module- or class-qualified name."""

    names: dict[ast.AST, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names[node] = node.name
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names[child] = f"{node.name}.{child.name}"
    return names


def _enclosing_definition(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    names: Mapping[ast.AST, str],
) -> str:
    """Return the nearest named production definition containing an AST node."""

    current = node
    while current in parents:
        current = parents[current]
        if current in names:
            return names[current]
    return "<module>"


def _tree_context(source: str) -> tuple[ast.Module, dict[ast.AST, ast.AST], dict[ast.AST, str]]:
    """Parse source and return its tree, parent links, and qualified definitions."""

    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    return tree, parents, _definition_names(tree)


def _subprocess_spawn_inventory(source: str) -> Counter[str]:
    """Return every ``subprocess.Popen`` edge keyed by its enclosing function."""

    tree, parents, names = _tree_context(source)
    return Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "subprocess.Popen"
    )


def _sensitive_edge_inventory(source: str) -> Counter[tuple[str, str]]:
    """Return every classified admission, authority, and append call edge."""

    tree, parents, names = _tree_context(source)
    found: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _attribute_name(node.func)
        for suffix in _SENSITIVE_SUFFIXES:
            if call_name.endswith(suffix):
                found[(_enclosing_definition(node, parents, names), suffix)] += 1
    return found


def _function_node(source: str, qualified_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return one exact module or class function definition from source."""

    tree = ast.parse(source)
    parts = qualified_name.split(".")
    body: Sequence[ast.stmt] = tree.body
    for index, part in enumerate(parts):
        matching = [
            node
            for node in body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == part
        ]
        if len(matching) != 1:
            raise AssertionError(f"missing or ambiguous definition: {qualified_name}")
        selected = matching[0]
        if index == len(parts) - 1:
            if not isinstance(selected, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise AssertionError(f"qualified name is not a function: {qualified_name}")
            return selected
        if not isinstance(selected, ast.ClassDef):
            raise AssertionError(f"qualified owner is not a class: {qualified_name}")
        body = selected.body
    raise AssertionError(f"missing definition: {qualified_name}")


def _string_constants(node: ast.AST) -> set[str]:
    """Return every string literal below one AST node."""

    return {
        value.value
        for value in ast.walk(node)
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }


@dataclass(frozen=True)
class _SymbolicValue:
    """One statically resolved symbol and/or folded string value."""

    symbol: str | None = None
    text: str | None = None


class _SubstitutionAnalyzer:
    """Small fixed-point AST interpreter for §8.3 composition preclusion."""

    def __init__(self, source: str, source_path: Path) -> None:
        """Index one module's imports, constants, and local definitions.

        Parameters
        ----------
        source:
            Python source under analysis.
        source_path:
            Repository-relative diagnostic path.
        """

        self.tree = ast.parse(source, filename=str(source_path))
        self.source_path = source_path
        self.definitions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.module_values: dict[str, _SymbolicValue] = {}
        self.errors: set[str] = set()
        self._active_calls: set[tuple[str, tuple[str, ...]]] = set()
        self._index_module_bindings()

    def _index_module_bindings(self) -> None:
        """Resolve module imports, aliases, and foldable constant assignments."""

        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.module_values[alias.asname or alias.name.split(".")[0]] = _SymbolicValue(
                        symbol=alias.name
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.module_values[name] = _SymbolicValue(
                        symbol=".".join(part for part in (module, alias.name) if part)
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value_node = node.value
                if value_node is None:
                    continue
                value = self._value(value_node, self.module_values)
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.module_values[target.id] = value

    def _value(
        self,
        node: ast.AST,
        values: Mapping[str, _SymbolicValue],
    ) -> _SymbolicValue:
        """Resolve one expression into a symbol alias and folded string.

        Parameters
        ----------
        node:
            Expression to resolve.
        values:
            Current lexical bindings.

        Returns
        -------
        _SymbolicValue
            Best-effort static value; unknown fields remain ``None``.
        """

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _SymbolicValue(text=node.value)
        if isinstance(node, ast.Name):
            return values.get(node.id, _SymbolicValue(symbol=node.id))
        if isinstance(node, ast.Attribute):
            owner = self._value(node.value, values).symbol
            return _SymbolicValue(symbol=f"{owner}.{node.attr}" if owner else node.attr)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._value(node.left, values).text
            right = self._value(node.right, values).text
            return _SymbolicValue(
                text=left + right if left is not None and right is not None else None
            )
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    folded = self._value(value.value, values).text
                    if folded is None:
                        return _SymbolicValue()
                    parts.append(folded)
                else:
                    return _SymbolicValue()
            return _SymbolicValue(text="".join(parts))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
        ):
            separator = self._value(node.func.value, values).text
            items = node.args[0]
            if separator is not None and isinstance(items, (ast.List, ast.Tuple)):
                folded_items = [self._value(item, values).text for item in items.elts]
                if all(item is not None for item in folded_items):
                    return _SymbolicValue(text=separator.join(str(item) for item in folded_items))
        if isinstance(node, ast.Call):
            return _SymbolicValue(symbol=self._value(node.func, values).symbol)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for candidate in node.values:
                symbolic_value = self._value(candidate, values)
                if symbolic_value.symbol is not None or symbolic_value.text is not None:
                    return symbolic_value
        return _SymbolicValue()

    def _definition_label(self, stack: Sequence[str]) -> str:
        """Return a stable caller-to-helper definition chain."""

        return "->".join(stack) if stack else "<module>"

    def _add_error(self, stack: Sequence[str], boundary: str, evasion_class: str) -> None:
        """Record one fully located substitution diagnostic."""

        self.errors.add(
            f"{self.source_path.as_posix()}:{self._definition_label(stack)}:"
            f"{boundary}:{evasion_class}"
        )

    def _boundary_from_call(
        self,
        node: ast.Call,
        resolved_call: str,
        values: Mapping[str, _SymbolicValue],
    ) -> str | None:
        """Resolve the registered boundary targeted by a mutation or lookup call."""

        name_index = 0 if resolved_call.endswith("patch") else 1
        if len(node.args) <= name_index:
            return None
        candidate = self._value(node.args[name_index], values).text
        if candidate is not None:
            tail = candidate.rsplit(".", 1)[-1]
            if tail in _SUBSTITUTION_BOUNDARIES:
                return tail
        if name_index == 1:
            owner_symbol = self._value(node.args[0], values).symbol
            if owner_symbol:
                tail = owner_symbol.rsplit(".", 1)[-1]
                if tail in _SUBSTITUTION_BOUNDARIES:
                    return tail
        return None

    def _analyze_call(
        self,
        node: ast.Call,
        values: dict[str, _SymbolicValue],
        stack: tuple[str, ...],
        *,
        decorator: bool = False,
    ) -> None:
        """Analyze one call, including aliases and local-helper argument flow."""

        syntactic_call = _attribute_name(node.func)
        resolved_call = self._value(node.func, values).symbol or syntactic_call
        call_tail = resolved_call.rsplit(".", 1)[-1]
        mutation = bool(
            call_tail in {"patch", "setattr", "delattr"} or resolved_call.endswith("patch.object")
        )
        boundary = self._boundary_from_call(node, resolved_call, values) if mutation else None
        if mutation and boundary is not None:
            if decorator:
                evasion_class = "decorator"
            elif len(stack) > 1:
                evasion_class = "helper-indirection"
            elif syntactic_call != resolved_call and "." not in syntactic_call:
                evasion_class = "alias-patch"
            elif call_tail in {"setattr", "delattr"} and not isinstance(
                node.args[1] if len(node.args) > 1 else None, ast.Constant
            ):
                evasion_class = "dynamic-lookup"
            else:
                evasion_class = "direct-patch"
            self._add_error(stack, boundary, evasion_class)
        elif mutation and boundary is None:
            target_known = (
                len(node.args) > 1 and self._value(node.args[1], values).text is not None
            ) or (bool(node.args) and self._value(node.args[0], values).text is not None)
            if not target_known:
                self._add_error(stack, "<unresolved-mutation-target>", "dynamic-lookup")

        if call_tail == "getattr" and len(node.args) >= 2:
            dynamic_boundary = self._value(node.args[1], values).text
            if dynamic_boundary in _SUBSTITUTION_BOUNDARIES:
                self._add_error(stack, str(dynamic_boundary), "dynamic-lookup")
        if call_tail in _FAKE_EXECUTION_TYPES:
            self._add_error(stack, call_tail, "fake-environment-result")
        if call_tail in _LEGACY_ALTERNATE_COMPILERS:
            self._add_error(stack, call_tail, "alternate-compiler")
        if call_tail in {"wraps", "spy", "Mock", "MagicMock"}:
            candidates = [*node.args, *(keyword.value for keyword in node.keywords)]
            for candidate in candidates:
                symbol = self._value(candidate, values).symbol
                if symbol and symbol.rsplit(".", 1)[-1] in _SUBSTITUTION_BOUNDARIES:
                    self._add_error(stack, symbol.rsplit(".", 1)[-1], "wrapper-spy")

        helper_name = call_tail if call_tail in self.definitions else None
        helper = self.definitions.get(helper_name) if helper_name is not None else None
        if (
            helper_name is not None
            and helper_name not in stack
            and self.source_path.name != "test_round17_structural_inventories.py"
            and isinstance(helper, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            bound = dict(self.module_values)
            parameters = [*helper.args.posonlyargs, *helper.args.args]
            for parameter, argument in zip(parameters, node.args):
                bound[parameter.arg] = self._value(argument, values)
            for keyword in node.keywords:
                if keyword.arg is not None:
                    bound[keyword.arg] = self._value(keyword.value, values)
            self._analyze_function(helper, bound, (*stack, helper_name))

        for child in (*node.args, *(keyword.value for keyword in node.keywords)):
            self._analyze_expression(child, values, stack)

    def _analyze_expression(
        self,
        node: ast.AST,
        values: dict[str, _SymbolicValue],
        stack: tuple[str, ...],
        *,
        decorator: bool = False,
    ) -> None:
        """Recursively analyze calls and folded forbidden values in an expression."""

        folded = self._value(node, values).text
        if (
            folded == "runtime-root"
            and self.source_path.name != "test_round17_structural_inventories.py"
        ):
            self._add_error(stack, "runtime-root", "legacy-root")
        if isinstance(node, ast.Call):
            self._analyze_call(node, values, stack, decorator=decorator)
            self._analyze_expression(node.func, values, stack)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for candidate in node.values:
                self._analyze_expression(candidate, values, stack)
                value = self._value(candidate, values)
                if value.symbol is not None or value.text is not None:
                    break
            return
        for child in ast.iter_child_nodes(node):
            self._analyze_expression(child, values, stack)

    def _analyze_target(
        self,
        target: ast.AST,
        values: Mapping[str, _SymbolicValue],
        stack: tuple[str, ...],
    ) -> None:
        """Reject direct attribute and mapping assignment to a registered boundary."""

        if isinstance(target, ast.Attribute) and target.attr in _SUBSTITUTION_BOUNDARIES:
            self._add_error(stack, target.attr, "assignment")
        elif isinstance(target, ast.Subscript):
            key = self._value(target.slice, values).text
            if key in _SUBSTITUTION_BOUNDARIES:
                self._add_error(stack, str(key), "assignment")

    def _analyze_statements(
        self,
        statements: Sequence[ast.stmt],
        values: dict[str, _SymbolicValue],
        stack: tuple[str, ...],
    ) -> None:
        """Interpret one statement list with local alias and constant propagation."""

        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in statement.decorator_list:
                    self._analyze_expression(decorator, values, stack, decorator=True)
                self._analyze_function(statement, dict(values), (*stack, statement.name))
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                value_node = statement.value
                if value_node is None:
                    continue
                self._analyze_expression(value_node, values, stack)
                value = self._value(value_node, values)
                for target in targets:
                    self._analyze_target(target, values, stack)
                    if isinstance(target, ast.Name):
                        values[target.id] = value
                continue
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    self._analyze_expression(child, values, stack)
                elif isinstance(child, ast.stmt):
                    self._analyze_statements((child,), dict(values), stack)

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        values: dict[str, _SymbolicValue],
        stack: tuple[str, ...],
    ) -> None:
        """Analyze one local function once per symbolic caller binding."""

        signature = tuple(
            f"{name}={value.symbol or value.text or '?'}" for name, value in sorted(values.items())
        )
        key = (node.name, signature)
        if key in self._active_calls:
            return
        self._active_calls.add(key)
        base_worker_argv = any(
            "menagerie.crawler.worker" in _string_constants(container)
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == "executable"
                and isinstance(child.value, ast.Name)
                and child.value.id == "sys"
                for child in ast.walk(container)
            )
            for container in ast.walk(node)
            if isinstance(container, (ast.List, ast.Tuple))
        )
        if base_worker_argv:
            self._add_error(stack, "selected-interpreter-argv", "base-interpreter-argv")
        self._analyze_statements(node.body, values, stack)
        self._active_calls.remove(key)

    def analyze(self, root_definitions: Sequence[str] | None = None) -> tuple[str, ...]:
        """Return all diagnostics reachable from the selected module definitions.

        Parameters
        ----------
        root_definitions:
            Exact top-level roots.  By default every test function is analyzed.

        Returns
        -------
        tuple[str, ...]
            Sorted fully located diagnostics.
        """

        roots = tuple(root_definitions or ())
        if not roots:
            roots = tuple(name for name in self.definitions if name.startswith("test_"))
        if not roots:
            roots = tuple(self.definitions)
        for root in roots:
            definition = self.definitions.get(root)
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                values = dict(self.module_values)
                for decorator in definition.decorator_list:
                    self._analyze_expression(decorator, values, (root,), decorator=True)
                self._analyze_function(definition, values, (root,))
            elif isinstance(definition, ast.ClassDef):
                for child in definition.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._analyze_function(
                            child,
                            dict(self.module_values),
                            (root, child.name),
                        )
        return tuple(sorted(self.errors))


def _substitution_boundary_errors(
    source: str,
    *,
    source_path: Path = Path("<memory>"),
    root_definitions: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return transitive AST-detected substitutions in one composition module.

    Parameters
    ----------
    source:
        Python module source.
    source_path:
        Repository-relative diagnostic path.
    root_definitions:
        Optional exact composition/fixture roots.

    Returns
    -------
    tuple[str, ...]
        Fully located substitution diagnostics.
    """

    return _SubstitutionAnalyzer(source, source_path).analyze(root_definitions)


def _composition_roots(path: Path, source: str) -> tuple[str, ...] | None:
    """Return exact roots for support modules that also contain unit-only fixtures."""

    if path.name == "conftest.py":
        raise AssertionError("conftest roots require the transitive fixture graph")
    if path.name == "test_slice_f_driver.py":
        return ("test_linux_handoff_attempts_both_deferred_statuses_and_supersedes",)
    if path.name in {"cli.py", "dry_run_support.py"}:
        tree = ast.parse(source, filename=str(path))
        return tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    registered = _registered_composition_roots().get(path)
    if registered is not None and "composition" not in path.stem:
        return registered
    return None


def _transitive_fixture_roots(paths: Sequence[Path]) -> tuple[str, ...]:
    """Resolve every applicable conftest fixture dependency to a fixed point.

    Parameters
    ----------
    paths:
        Complete discovered composition/registered/CI source scope.

    Returns
    -------
    tuple[str, ...]
        Exact conftest fixture roots plus the real fixture data/lane classes.
    """

    conftest_path = _CRAWLER_ROOT / "tests" / "conftest.py"
    conftest_tree = ast.parse(
        conftest_path.read_text(encoding="utf-8"), filename=str(conftest_path)
    )
    fixtures = {
        node.name: node
        for node in conftest_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            _attribute_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            ).endswith("fixture")
            for decorator in node.decorator_list
        )
    }
    requested: set[str] = set()
    for path in paths:
        if path == conftest_path:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        selected_roots = _composition_roots(path, source)
        roots = set(selected_roots or ())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if roots and node.name not in roots:
                continue
            if not roots and not node.name.startswith("test_"):
                continue
            requested.update(
                parameter.arg for parameter in (*node.args.posonlyargs, *node.args.args)
            )

    resolved: set[str] = set()
    pending = requested & fixtures.keys()
    while pending:
        fixture_name = pending.pop()
        if fixture_name in resolved:
            continue
        resolved.add(fixture_name)
        fixture = fixtures[fixture_name]
        dependencies = {
            parameter.arg for parameter in (*fixture.args.posonlyargs, *fixture.args.args)
        }
        pending.update((dependencies & fixtures.keys()) - resolved)
    return tuple(
        sorted(
            {
                "RealEnvironmentFixture",
                "RealEnvironmentLane",
                "real_environment_registry",
                *resolved,
            }
        )
    )


def _composition_scope_errors() -> tuple[str, ...]:
    """Return substitution diagnostics across every discovered composition edge."""

    errors: list[str] = []
    paths = _composition_source_paths()
    fixture_roots = _transitive_fixture_roots(paths)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        roots = fixture_roots if path.name == "conftest.py" else _composition_roots(path, source)
        errors.extend(
            _substitution_boundary_errors(
                source,
                source_path=path.relative_to(_REPOSITORY_ROOT),
                root_definitions=roots,
            )
        )
    return tuple(sorted(set(errors)))


def _required_ci_selection_errors(workflow_source: str) -> tuple[str, ...]:
    """Return fully located diagnostics for missing required real CI nodes."""

    return tuple(
        f".github/workflows/tests.yml:<workflow>:{node}:deleted-ci-node"
        for node in sorted(_REQUIRED_CI_SELECTIONS)
        if node not in workflow_source
    )


def _production_python_paths() -> tuple[Path, ...]:
    """Return every crawler production Python path, excluding tests and tools."""

    return tuple(
        path
        for path in sorted(_CRAWLER_ROOT.rglob("*.py"))
        if "tests" not in path.parts and "tools" not in path.parts
    )


def _dead_contract_occurrences(source: str) -> set[str]:
    """Return forbidden dead identifiers and exact retired option literals in source."""

    tree = ast.parse(source)
    found = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in _DEAD_SYMBOLS
    }
    found.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in _DEAD_SYMBOLS
    )
    found.update(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in _DEAD_SYMBOLS
    )
    found.update(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in {*_DEAD_SYMBOLS, *_DEAD_OPTIONS}
    )
    return found


def _comparison_owner_inventory(source: str) -> set[tuple[str, str]]:
    """Return protocol-version constant comparisons and their enclosing owner."""

    tree, parents, names = _tree_context(source)
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            if isinstance(operand, ast.Name) and operand.id in _PROTOCOL_VERSIONS:
                found.add((_enclosing_definition(node, parents, names), operand.id))
    return found


def _schema_parity_errors(schema: Mapping[str, Any], sources: Mapping[str, str]) -> tuple[str, ...]:
    """Return Round-17 attempt schema/consumer/producer parity violations."""

    leaves = schema_leaf_paths(schema)
    expected_leaves = {
        "$.capability_observation.claim",
        "$.capability_observation.supported",
        "$.raw_award_receipt.observation.output_value_sha256",
        "$.worker_receipt.output_value_sha256",
    }
    errors = [f"schema-missing:{path}" for path in sorted(expected_leaves - leaves)]
    required_literals = {
        "authority.py:derive_mode_summary": {"output_value_sha256"},
        "authority.py:derive_terminal_proof": {"capability_observation", "claim", "supported"},
        "worker.py:_mode_receipt": {"output_value_sha256"},
        "worker.py:_raw_award_receipt": {"output_value_sha256"},
        "driver.py:_attempts_from_supervised": {
            "capability_observation",
            "output_value_sha256",
        },
        "driver.py:_driver_failure_attempt": {
            "capability_observation",
            "output_value_sha256",
        },
    }
    for owner, required in required_literals.items():
        filename, function_name = owner.split(":", 1)
        function = _function_node(sources[filename], function_name)
        missing = required - _string_constants(function)
        errors.extend(f"consumer-or-producer-missing:{owner}:{key}" for key in sorted(missing))
    return tuple(errors)


def _assert_symbol(module: Any, qualified_name: str) -> None:
    """Assert that one dot-qualified production symbol exists."""

    current = module
    for part in qualified_name.split("."):
        current = getattr(current, part)


def _assert_test_node_exists(node_id: str, default_path: Path) -> None:
    """Assert that one manifest node names a pytest-collectable top-level function."""

    if "::" in node_id:
        raw_path, function_name = node_id.split("::", 1)
        path = _REPOSITORY_ROOT / raw_path
    else:
        path = default_path
        function_name = node_id
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        and node.name.startswith("test_")
    ]
    assert len(matches) == 1, f"manifest node is not pytest-collectable: {node_id}"


def test_receipt_consumers_are_confined_to_the_typed_projection() -> None:
    """No production semantic consumer may read the opaque wrapper directly."""

    observed: set[tuple[str, str]] = set()
    for path in _production_python_paths():
        tree, parents, names = _tree_context(path.read_text(encoding="utf-8"))
        observed.update(
            (path.name, _enclosing_definition(node, parents, names))
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "worker_receipt"
        )
    assert observed == {
        ("worker_supervisor.py", "verify_supervised_worker_result"),
        ("worker_supervisor.py", "worker_result_outer_for_diagnostics"),
    }


def test_spawn_inventory_is_closed_and_lease_bearing() -> None:
    """Every process spawn must keep its reviewed role and v3 lease inheritance."""

    source = _source(supervisor_module)
    assert supervisor_module._SUBPROCESS_SPAWN_REGISTRY == {  # noqa: SLF001
        "_emit_macos_audit_sentinel": "audit-sentinel:no-model-work",
        "_start_macos_denial_audit": "audit-collector:no-model-work",
        "run_isolated_subprocess": "model-worker:inherited-live-lease",
    }
    assert _subprocess_spawn_inventory(source) == Counter(
        {name: 1 for name in supervisor_module._SUBPROCESS_SPAWN_REGISTRY}  # noqa: SLF001
    )
    spawn = _function_node(source, "run_isolated_subprocess")
    calls = [
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "subprocess.Popen"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert keywords["shell"] == "False"
    assert keywords["start_new_session"] == "True"
    assert keywords["close_fds"] == "True"
    assert keywords["pass_fds"] == "inherited_fds"
    supervisor = _function_node(source, "supervise_worker")
    assert "v3 model worker spawn requires an inherited live worker lease" in _string_constants(
        supervisor
    )

    mutated = f"{source}\n\ndef rogue_model_spawn():\n    subprocess.Popen(['python'])\n"
    assert _subprocess_spawn_inventory(mutated) != _subprocess_spawn_inventory(source)


def test_admission_boundary_inventory_is_closed() -> None:
    """All authority-bearing admissions must remain classified and guarded."""

    source = _source(driver_module)
    assert _sensitive_edge_inventory(source) == _SENSITIVE_EDGE_COUNTS
    assert driver_module._SHUTDOWN_ADMISSION_REGISTRY == {  # noqa: SLF001
        "author": "guard:author-admission",
        "checker": "guard:checker-admission",
        "environment-create": "guard:environment-create-admission",
        "environment-use": "guard:environment-use-admission",
        "model": "guard:model-admission",
        "lease": "guard:forward-admission|pre-slot-resolution",
        "spawn": "guard:forward-admission|pre-slot-resolution",
        "run-model-assembly": "guard:post-attempt-pre-award",
        "publication-admission": "guard:pre-publication-admission",
        "publication": "atomic:award-commit",
        "terminal-publication": "atomic:award-commit",
        "model-append": "atomic:award-commit",
        "post-award-observation": "guard:post-award-commit",
    }
    forward = inspect.getsource(driver_module.CrawlerDriver._forward_and_reduce)
    terminal = inspect.getsource(driver_module.CrawlerDriver._terminalize)
    assert forward.index('"post-attempt-pre-award"') < forward.index("_assemble_run_model(")
    assert forward.index('"pre-publication-admission"') < forward.index('"pre-award-commit"')
    assert forward.index('"pre-award-commit"') < forward.index("reducer.append_model(")
    assert forward.index("reducer.append_model(") < forward.index('"post-award-commit"')
    assert terminal.index('"pre-publication-admission"') < terminal.index('"pre-award-commit"')
    assert terminal.index('"pre-award-commit"') < terminal.index("reducer.append_model(")
    assert terminal.index("reducer.append_model(") < terminal.index('"post-award-commit"')

    mutated = (
        f"{source}\n\ndef rogue_admission(self):\n"
        "    self.dependencies.author.author(None, None, None, None)\n"
    )
    assert _sensitive_edge_inventory(mutated) != _SENSITIVE_EDGE_COUNTS


def test_public_writer_inventory_is_authorization_bearing() -> None:
    """Public placement must have one low-level writer and one authorized caller."""

    source = _source(artifact_module)
    assert artifact_module._PUBLIC_WRITER_REGISTRY == {  # noqa: SLF001
        "_materialize_public_claims": "publish_authorized_artifact:committed-authorization"
    }
    tree, parents, names = _tree_context(source)
    materializer_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_materialize_public_claims"
    )
    assert materializer_callers == Counter({"publish_authorized_artifact": 1})
    public_put_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith(".put")
        and any(
            keyword.arg == "mirror_class" and ast.unparse(keyword.value) == "MirrorClass.PUBLIC"
            for keyword in node.keywords
        )
    )
    assert public_put_callers == Counter({"_materialize_public_claims": 1})
    publisher = _function_node(source, "publish_authorized_artifact")
    assert "public/private commitment requires one exact prior authorization event" in source
    publisher_calls = [
        _attribute_name(node.func) for node in ast.walk(publisher) if isinstance(node, ast.Call)
    ]
    assert publisher_calls.index("_authorization_event") < publisher_calls.index(
        "_materialize_public_claims"
    )

    mutated = (
        f"{source}\n\ndef rogue_public_writer(*args):\n    _materialize_public_claims(*args)\n"
    )
    mutated_tree, mutated_parents, mutated_names = _tree_context(mutated)
    mutated_callers = Counter(
        _enclosing_definition(node, mutated_parents, mutated_names)
        for node in ast.walk(mutated_tree)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "_materialize_public_claims"
    )
    assert mutated_callers != materializer_callers


def test_dead_symbol_inventory_is_closed() -> None:
    """Retired producers, validators, writers, and CLI options must stay absent."""

    observed = {
        (path.relative_to(_CRAWLER_ROOT).as_posix(), occurrence)
        for path in _production_python_paths()
        for occurrence in _dead_contract_occurrences(path.read_text(encoding="utf-8"))
    }
    assert observed == set()

    for symbol in sorted(_DEAD_SYMBOLS):
        assert _dead_contract_occurrences(f"def {symbol}():\n    return None\n") == {symbol}
    for option in sorted(_DEAD_OPTIONS):
        assert _dead_contract_occurrences(f"RETIRED = {option!r}\n") == {option}


def test_protocol_literal_inventory_has_only_reviewed_comparison_owners() -> None:
    """Live wrapper version comparisons must remain in protocol projections."""

    supervisor_source = _source(supervisor_module)
    authority_source = _source(authority_module)
    assert {
        name: getattr(supervisor_module, name) for name in _PROTOCOL_VERSIONS
    } == _PROTOCOL_VERSIONS
    assert _comparison_owner_inventory(supervisor_source) == {
        ("_load_worker_result_value", "_WORKER_RESULT_VERSION"),
        ("_load_worker_result_value", "_WORKER_DIAGNOSTIC_VERSION"),
        ("_load_worker_result_value", "_RAW_AWARD_RECEIPT_VERSION"),
    }
    assert _comparison_owner_inventory(authority_source) == {
        ("_validate_raw_receipt", "_RAW_AWARD_RECEIPT_VERSION")
    }
    for path in _production_python_paths():
        if path.name in {"worker_supervisor.py", "authority.py"}:
            continue
        assert _comparison_owner_inventory(path.read_text(encoding="utf-8")) == set()

    mutated = (
        f"{supervisor_source}\n\ndef rogue_parser(value):\n"
        "    return value == _WORKER_RESULT_VERSION\n"
    )
    assert ("rogue_parser", "_WORKER_RESULT_VERSION") in _comparison_owner_inventory(mutated)


def test_attempt_schema_consumers_and_producers_have_exact_parity() -> None:
    """Attempt-v3 evidence leaves must exist, be produced, consumed, owned, and retained."""

    schema = load_schema(ATTEMPT_SCHEMA_VERSION_V3)
    sources = {
        filename: (_CRAWLER_ROOT / filename).read_text(encoding="utf-8")
        for filename in ("authority.py", "driver.py", "worker.py")
    }
    assert _schema_parity_errors(schema, sources) == ()
    ownership = {leaf.path: leaf.owner for leaf in owned_schema_leaves(ATTEMPT_SCHEMA_VERSION_V3)}
    assert ownership["$.capability_observation.claim"] is SchemaOwner.PARENT_OBSERVED
    assert ownership["$.capability_observation.supported"] is SchemaOwner.PARENT_OBSERVED
    assert (
        ownership["$.raw_award_receipt.observation.output_value_sha256"]
        is SchemaOwner.WORKER_OBSERVED
    )
    assert ownership["$.worker_receipt.output_value_sha256"] is SchemaOwner.WORKER_OBSERVED
    assert {
        "capability_observation",
        "output_value_sha256",
    }.isdisjoint(driver_module._EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS)  # noqa: SLF001

    forbidden_schema = deepcopy(schema)
    del forbidden_schema["properties"]["capability_observation"]
    assert any(
        error.startswith("schema-missing:$.capability_observation")
        for error in _schema_parity_errors(forbidden_schema, sources)
    )
    missing_consumer = dict(sources)
    missing_consumer["authority.py"] = sources["authority.py"].replace(
        'probe.get("capability_observation")', 'probe.get("retired_capability")', 1
    )
    assert (
        "consumer-or-producer-missing:authority.py:derive_terminal_proof:capability_observation"
        in (_schema_parity_errors(schema, missing_consumer))
    )


def test_synthetic_worker_result_factory_matches_production_protocol_shapes() -> None:
    """The sole synthetic live fixture must round-trip through production's loader shape."""

    attempt = make_attempt()
    raw = deepcopy(attempt["raw_award_receipt"])
    synthetic = make_worker_result_v3_mapping({}, raw_award_receipt=raw)
    loaded, error = supervisor_module._load_worker_result_value(synthetic)  # noqa: SLF001
    assert error is None
    assert loaded == synthetic
    assert set(synthetic) == supervisor_module._WORKER_RESULT_KEYS  # noqa: SLF001
    assert set(synthetic["diagnostic"]) == supervisor_module._WORKER_DIAGNOSTIC_KEYS  # noqa: SLF001
    assert set(synthetic["raw_award_receipt"]) == supervisor_module._RAW_AWARD_RECEIPT_KEYS  # noqa: SLF001

    observation = SupervisorObservation(
        argv=("python",),
        cwd=str(Path.cwd()),
        exit_code=1,
        signal_number=None,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        peak_rss_bytes=0,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256="sha256:" + "a" * 64,
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256="sha256:" + "b" * 64,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
    )
    supervised = make_supervised_worker_result_v3(observation, {})
    assert supervised.worker_receipt is not None
    assert set(supervised.worker_receipt) == set(synthetic)

    constructed: list[tuple[str, str]] = []
    for path in sorted((_CRAWLER_ROOT / "tests").glob("*.py")):
        tree, parents, names = _tree_context(path.read_text(encoding="utf-8"))
        constructed.extend(
            (path.name, _enclosing_definition(node, parents, names))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _attribute_name(node.func) == "SupervisedResult"
        )
    assert constructed == [("conftest.py", "make_supervised_worker_result_v3")]


def test_executable_and_artifact_authority_have_one_final_projection() -> None:
    """Exact members and one normalized final transaction must feed all consumers."""

    read_projection = inspect.getsource(supervisor_module._request_allowed_read_paths)  # noqa: SLF001
    assert "exact_read_capability(manifest).member_paths" in read_projection
    assert "allowed_read_paths" in inspect.getsource(supervisor_module.run_isolated_subprocess)
    assert "runtime_read_roots" not in inspect.getsource(
        authority_module.compile_execution_read_manifest_v2
    )

    driver_source = _source(driver_module)
    tree, parents, names = _tree_context(driver_source)
    resolver_callers = Counter(
        _enclosing_definition(node, parents, names)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _attribute_name(node.func).endswith("resolve_final_artifact_transaction")
    )
    assert resolver_callers == Counter({"CrawlerDriver._rehydrate_final_authority": 1})
    reconstruction = inspect.getsource(driver_module.CrawlerDriver._rehydrate_final_authority)
    assert (
        reconstruction.index("validate_artifact_checkpoint(")
        < reconstruction.index("resolve_final_artifact_transaction(")
        < reconstruction.index("rehydrate_artifact_transaction(")
    )
    resolver = _function_node(_source(artifact_module), "resolve_final_artifact_transaction")
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.UnaryOp)
        and isinstance(node.slice.op, ast.USub)
        for node in ast.walk(resolver)
    )


def test_award_identity_closure_names_live_symbols_and_no_dead_symbols() -> None:
    """Award closure roots must resolve and must not retain deleted authority."""

    for relative, symbols in driver_module._AWARD_CLOSURE_SYMBOLS.items():  # noqa: SLF001
        path = _CRAWLER_ROOT / relative
        for symbol in symbols:
            assert driver_module._source_symbol_bytes(path, symbol)  # noqa: SLF001
            assert symbol not in _DEAD_SYMBOLS


def test_round17_real_composition_sources_are_not_fake_substitutes() -> None:
    """Permanent adjudication probes must retain their production composition seams."""

    import menagerie.crawler.tests.test_round17_vs1_v3_composition as vs1_module
    import menagerie.crawler.tests.test_round17_vs2_shutdown_composition as vs2_module
    import menagerie.crawler.tests.test_round19_vs6_dry_run_composition as vs6_module
    import menagerie.crawler.tests.test_slice_f_driver as driver_tests

    golden = inspect.getsource(
        vs1_module.test_real_v3_worker_result_awards_through_driver_and_reducer
    )
    shutdown = inspect.getsource(
        vs2_module.test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes
    )
    handoff = inspect.getsource(
        driver_tests.test_linux_handoff_attempts_both_deferred_statuses_and_supersedes
    )
    dry_run = inspect.getsource(vs6_module.test_documented_dry_run_and_resume_use_real_environment)
    dry_run_support = (_CRAWLER_ROOT / "tests" / "dry_run_support.py").read_text(encoding="utf-8")
    module_source = _source(vs1_module)
    shutdown_module_source = _source(vs2_module)
    assert "SupervisedForwardLane" in module_source
    assert "CanonicalReducer" in module_source
    assert "runs" in _string_constants(ast.parse(module_source))
    assert "FakeForward" not in golden
    assert "SupervisedResult(" not in golden
    assert "multiprocessing.get_context" in shutdown
    assert "RealEnvironmentLane" in shutdown_module_source
    assert 'observation["models"] == 0' in shutdown
    assert "DisabledAuthor" in handoff
    assert "SupervisedForwardLane" in handoff
    assert "RealEnvironmentLane" in handoff
    assert "FakeEnvironments" not in handoff
    assert 'record["status"]["code"] for record in superseding' in handoff
    assert "_dry_run_command" in dry_run
    assert "scan_jsonl" in dry_run
    assert "MaterializedDryRunEnvironment" in dry_run_support
    assert "SupervisedForwardLane" in dry_run_support
    assert "FakeEnvironments" not in dry_run_support


def test_environment_cache_has_one_lifecycle_owner_and_no_default_collector() -> None:
    """Live scheduling and currentness must share one lifecycle-owned cache."""

    lifecycle_source = _source(lifecycle_module)
    driver_source = _source(driver_module)
    binder_signature = inspect.signature(driver_module.bind_materialized_environment)
    assert lifecycle_source.count("EnvironmentAuthorityCache()") == 1
    assert "EnvironmentAuthorityCache()" not in driver_source
    assert binder_signature.parameters["authority_cache"].default is inspect.Signature.empty
    assert "active_authority_cache(" in inspect.getsource(
        driver_module.CrawlerDriver._run_environment_work
    )
    assert "cache.verify(authority, verification_token=verification_token)" in inspect.getsource(
        driver_module._current_run_is_fresh
    )


def test_real_compositions_cannot_substitute_execution_boundaries() -> None:
    """Transitive AST tripwire covers fixtures, compositions, helpers, and CI."""

    assert _COMPOSITION_SOURCES == _composition_source_paths()
    assert _CRAWLER_ROOT / "tests" / "conftest.py" in _COMPOSITION_SOURCES
    assert (
        _CRAWLER_ROOT / "tests" / "test_round17_vs3_authority_composition.py"
        in _COMPOSITION_SOURCES
    )
    assert _composition_scope_errors() == ()

    mutated = (
        _COMPOSITION_SOURCES[0].read_text(encoding="utf-8")
        + "\n"
        + "def test_reintroduced_patch(monkeypatch):\n"
        + "    monkeypatch.setattr(driver_module, '_compile_worker_read_manifest', object())\n"
    )
    assert _substitution_boundary_errors(
        mutated,
        source_path=Path("tests/test_reintroduced_composition.py"),
        root_definitions=("test_reintroduced_patch",),
    ) == (
        "tests/test_reintroduced_composition.py:test_reintroduced_patch:"
        "_compile_worker_read_manifest:direct-patch",
    )


def test_legacy_manifest_v1_is_quarantined_from_every_live_import_graph() -> None:
    """Legacy root-grant parsing exists only in the audit module and cannot spawn."""

    production_paths = _production_python_paths()
    assert _QUARANTINE_PATH in production_paths
    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        if path == _QUARANTINE_PATH:
            assert "class ExecutionReadManifest" in source
            assert "def compile_execution_read_manifest" in source
            assert "def audit_runtime_root_grants" in source
            assert "runtime-root" in source
            assert "supervise_worker" not in source
            assert "run_isolated_subprocess" not in source
            continue
        assert "legacy_manifest_audit" not in source, path
        assert "runtime-root" not in source, path
        assert re.search(r"^class ExecutionReadManifest(?:\(|:)", source, re.MULTILINE) is None, (
            path
        )
        assert "def compile_execution_read_manifest(" not in source, path

    supervisor_source = _source(supervisor_module)
    assert "live v3 model worker spawn requires execution-read-manifest.v3" in supervisor_source
    assert "ExecutionReadManifest |" not in supervisor_source
    assert 'runtime_support if kind == "runtime-root"' not in supervisor_source


def test_round17_real_compositions_are_explicitly_selected_in_ci() -> None:
    """The always-on Linux CI lane must select the exact release marker."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "name: Crawler Round 21 Linux committed-lock release proofs" in workflow
    crawler_job = workflow.split("crawler-round21-linux-release:", 1)[1]
    crawler_job = crawler_job.split("\n  crawler-round19-macos-release:", 1)[0]
    assert "menagerie/crawler/tests -m round21_linux_real" in crawler_job
    assert 'MENAGERIE_RELEASE_GATE: "1"' in crawler_job
    assert "MENAGERIE_RELEASE_ATTESTATION" in crawler_job
    assert "--ignore" not in crawler_job
    assert "not heavy" not in crawler_job
    assert "not slow" not in crawler_job


def test_round19_supported_host_release_gate_inventory_is_exact() -> None:
    """Both mandatory host jobs select the closed permanent proof inventory."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert set(_ROUND19_RELEASE_NODE_INVENTORY) == {
        "golden",
        "interpreter",
        "linux-denial",
        "shutdown",
        "clean-clone",
        "unverifiable",
        "dry-run-run-resume",
        "dry-run-false-complete",
        "cache",
        "structural",
        "macos-positive-negative",
        "macos-profile",
    }
    assert "crawler-round21-linux-release:" in workflow
    assert "crawler-round19-macos-release:" in workflow
    release_jobs = workflow.split("crawler-round21-linux-release:", 1)[1]
    linux_job, macos_job = release_jobs.split("\n  crawler-round19-macos-release:", 1)
    macos_selected = set(re.findall(r"menagerie/crawler/tests/[A-Za-z0-9_./:-]+", macos_job))
    assert macos_selected == {
        _ROUND19_RELEASE_NODE_INVENTORY["macos-positive-negative"],
        _ROUND19_RELEASE_NODE_INVENTORY["macos-profile"],
        _ROUND19_RELEASE_NODE_INVENTORY["dry-run-run-resume"],
        _ROUND19_RELEASE_NODE_INVENTORY["dry-run-false-complete"],
    }
    assert "-m round21_linux_real" in linux_job
    assert "runs-on: macos-14-xlarge" in macos_job
    for job in (linux_job, macos_job):
        assert 'MENAGERIE_RELEASE_GATE: "1"' in job
        assert "unmet-release-gate" in job
        assert "pytest.skip" not in job


def test_round21_linux_release_artifacts_and_provisioning_are_real() -> None:
    """T04 requires every workflow-named Linux lock-family artifact to exist and parse."""

    lock_path = _CRAWLER_ROOT / "envs" / "locks" / "round19-linux-64.lock"
    family = {
        "lock": lock_path,
        "export": lock_path.with_suffix(".resolved.json"),
        "export-hash": lock_path.with_suffix(".resolved.sha256"),
        "provenance": lock_path.with_suffix(".provenance.json"),
        "probes": lock_path.with_suffix(".probes.json"),
    }
    assert {name for name, path in family.items() if path.is_file()} == set(family)
    lock_bytes = family["lock"].read_bytes()
    export_bytes = family["export"].read_bytes()
    assert lifecycle_module.parse_exact_lock(lock_bytes)
    assert lifecycle_module.parse_resolved_export(export_bytes) == export_bytes
    assert family["export-hash"].read_text(encoding="utf-8").strip() == hash_bytes(export_bytes)
    provenance = json.loads(family["provenance"].read_bytes())
    assert provenance["target"] == "linux-64"
    assert provenance["lock_sha256"] == hash_bytes(lock_bytes)
    assert provenance["resolved_export_sha256"] == hash_bytes(export_bytes)
    assert provenance["clean_create"]["validated"] is True

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    linux_job = workflow.split("crawler-round21-linux-release:", 1)[1].split(
        "\n  crawler-round19-macos-release:", 1
    )[0]
    assert family["lock"].relative_to(_REPOSITORY_ROOT).as_posix() in linux_job
    for suffix in (".resolved.json", ".resolved.sha256", ".provenance.json", ".probes.json"):
        assert suffix in linux_job
    assert "menagerie.crawler.tools.release_lock" in linux_job
    assert '"$MENAGERIE_REAL_ENV_PREFIX/bin/python" -m pytest' in linux_job
    assert "if-no-files-found: error" in linux_job


def test_round21_linux_release_registry_is_exact() -> None:
    """T05 requires the committed marker registry to name expanded existing nodes."""

    registry_path = _CRAWLER_ROOT / "tests" / "round21_linux_real_nodes.json"
    payload = json.loads(registry_path.read_bytes())
    nodes = payload["nodes"]
    assert payload["target"] == "linux-64"
    assert len(nodes) == 45
    assert len(nodes) == len(set(nodes))
    assert set(ROUND21_VS9_PROOF_REGISTRY) == {
        "P01",
        "T01",
        "T01-CI",
        "T02",
        "P02",
        "P03",
        "T03",
        "P04",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "T04",
        "T05",
    }
    assert ROUND21_VS9_PROOF_REGISTRY["P09"] in nodes
    assert all(node.startswith("menagerie/crawler/tests/") and "::test_" in node for node in nodes)


@pytest.mark.parametrize(
    ("manifest", "default_path"),
    (
        (
            VS1_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_round17_vs1_v3_composition.py",
        ),
        (
            VS2_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_round17_vs2_shutdown_composition.py",
        ),
        (
            VS3_LANDING_MANIFEST,
            _CRAWLER_ROOT / "tests" / "test_round17_vs3_authority_composition.py",
        ),
        (VS4_LANDING_MANIFEST, Path(__file__)),
    ),
)
def test_slice_landing_manifests_name_existing_seams_and_nodes(
    manifest: Mapping[str, Any],
    default_path: Path,
) -> None:
    """Each prevention landing manifest must resolve production symbols and tests."""

    modules = {
        "driver": driver_module,
        "worker_supervisor": supervisor_module,
        "artifact_transactions": artifact_module,
        "authority": authority_module,
    }
    for module_name, symbols in manifest["production_symbols"].items():
        for symbol in symbols:
            _assert_symbol(modules[module_name], symbol)
    assert manifest["real_composition_nodes"]
    assert manifest["structural_nodes"]
    for node_id in (*manifest["real_composition_nodes"], *manifest["structural_nodes"]):
        _assert_test_node_exists(str(node_id), default_path)
