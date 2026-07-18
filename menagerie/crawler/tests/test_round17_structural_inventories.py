"""Round-17 class-level prevention and anti-theater CI inventories."""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import inspect
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pytest

import menagerie.crawler.artifact_transactions as artifact_module
import menagerie.crawler.authority as authority_module
import menagerie.crawler.driver as driver_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
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
        "test_hardlinked_prefix_is_one_sealed_authority_and_mutation_stales"
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
_SUBSTITUTION_BOUNDARIES = frozenset(
    {
        "_compile_worker_read_manifest",
        "_collect_worker_executable_closure",
        "bind_materialized_environment",
        "EnvironmentAuthorityCache",
        "compile_execution_read_manifest_v3",
        "compile_execution_read_manifest_v3_from_closure",
        "environment_read_capability",
        "verify_execution_read_manifest",
        "verify_execution_read_manifest_v3",
        "supervise_worker",
        "run_isolated_subprocess",
    }
)
_COMPOSITION_SOURCES = (
    _CRAWLER_ROOT / "cli.py",
    _CRAWLER_ROOT / "tests" / "dry_run_support.py",
    _CRAWLER_ROOT / "tests" / "test_round17_vs1_v3_composition.py",
    _CRAWLER_ROOT / "tests" / "test_round17_vs2_shutdown_composition.py",
    _CRAWLER_ROOT / "tests" / "test_slice_f_driver.py",
    _CRAWLER_ROOT / "tests" / "test_round19_environment_authority_composition.py",
    _CRAWLER_ROOT / "tests" / "test_round19_vs6_dry_run_composition.py",
)

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


def _substitution_boundary_errors(source: str) -> tuple[str, ...]:
    """Return AST-detected composition substitutions of live execution boundaries."""

    tree, parents, names = _tree_context(source)
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _attribute_name(node.func)
            if (
                call_name in {"patch", "patch.object"}
                or call_name.endswith(".monkeypatch.setattr")
                or call_name.endswith("monkeypatch.setattr")
            ):
                rendered = " ".join(ast.unparse(arg) for arg in node.args)
                rendered = f"{rendered} {' '.join(_string_constants(node))}"
                for boundary in sorted(_SUBSTITUTION_BOUNDARIES):
                    if boundary in rendered:
                        errors.append(f"{_enclosing_definition(node, parents, names)}:{boundary}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            rendered_targets = " ".join(ast.unparse(target) for target in targets)
            for boundary in sorted(_SUBSTITUTION_BOUNDARIES):
                if boundary in rendered_targets:
                    errors.append(f"{_enclosing_definition(node, parents, names)}:{boundary}")
    return tuple(sorted(set(errors)))


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


def test_real_compositions_cannot_substitute_execution_boundaries() -> None:
    """AST tripwire forbids compiler, authority, and supervisor replacement."""

    assert {
        path.name: _substitution_boundary_errors(path.read_text(encoding="utf-8"))
        for path in _COMPOSITION_SOURCES
    } == {path.name: () for path in _COMPOSITION_SOURCES}

    mutated = (
        _COMPOSITION_SOURCES[0].read_text(encoding="utf-8")
        + "\n"
        + "def test_reintroduced_patch(monkeypatch):\n"
        + "    monkeypatch.setattr(driver_module, '_compile_worker_read_manifest', object())\n"
    )
    assert _substitution_boundary_errors(mutated) == (
        "test_reintroduced_patch:_compile_worker_read_manifest",
    )


def test_round17_real_compositions_are_explicitly_selected_in_ci() -> None:
    """CI must select every real probe and the structural module by exact path."""

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "name: Crawler Round 17 real compositions" in workflow
    assert "test_round17_structural_inventories.py" in workflow
    for node_id in _ROUND17_CI_NODES:
        assert node_id in workflow
    crawler_job = workflow.split("crawler-round17:", 1)[1]
    assert "--ignore" not in crawler_job
    assert "not heavy" not in crawler_job
    assert "not slow" not in crawler_job

    mutated = workflow.replace(_ROUND17_CI_NODES[0], "")
    assert _ROUND17_CI_NODES[0] not in mutated


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
    assert "crawler-round19-linux-release:" in workflow
    assert "crawler-round19-macos-release:" in workflow
    release_jobs = workflow.split("crawler-round19-linux-release:", 1)[1]
    selected = set(re.findall(r"menagerie/crawler/tests/[A-Za-z0-9_./:-]+", release_jobs))
    assert selected == set(_ROUND19_RELEASE_NODE_INVENTORY.values())
    linux_job, macos_job = release_jobs.split("\n  crawler-round19-macos-release:", 1)
    assert "runs-on: macos-14-xlarge" in macos_job
    for job in (linux_job, macos_job):
        assert 'MENAGERIE_RELEASE_GATE: "1"' in job
        assert "unmet-release-gate" in job
        assert "pytest.skip" not in job


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
