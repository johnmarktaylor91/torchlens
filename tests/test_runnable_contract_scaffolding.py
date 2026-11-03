"""Import and construction tripwires for runnable contract scaffolding."""

from __future__ import annotations

from torchlens._runnable import (
    RunnableErrorCode,
)
from torchlens.errors import ReattachError, RunnableTLSPECError, StateBindingError


# Kept: sole branch coverage of errors/_base.py.
def test_runnable_errors_import_and_instantiate() -> None:
    """Ensure public runnable errors retain the shared TorchLens payload style."""

    error = ReattachError(registry_id="callable:1")
    assert isinstance(error, RunnableTLSPECError)
    assert error.fields == {"registry_id": "callable:1"}
    assert isinstance(StateBindingError("bad state"), ValueError)


# The S1 seam contract (docs/reference/runnable_model.md "Extension points"):
# every extension point resolves on its contract module, the protocol stays
# the stable typed minimum the concrete door satisfies, and the run-door
# conflict matrix is bound by CODE LIST, never by count.

_S1_EXTENSION_POINTS: dict[str, tuple[str, ...]] = {
    "torchlens._runnable": (
        "WITNESS_FAMILY_REGISTRY",
        "WITNESS_GAP_REGISTRY",
        "CANONICAL_INITIALIZER_BY_ROLE",
        "SparseRunDescriptor",
        "RunnableErrorCode",
        "RunProvider",
        "RunnableTraceProtocol",
        "RunResult",
        "mark_trace_path_status",
        "refuse_poisoned_trace",
    ),
    "torchlens._runnable.seam": (
        "RUNNABLE_TRACE_PUBLIC_MEMBERS",
        "RunnableCoordinator",
        "RunnableTraceState",
        "runnable_trace_state",
    ),
    "torchlens._runnable.execution": (
        "run_loaded_sparse_trace",
        "run_live_trace",
    ),
    "torchlens._runnable.providers": ("_finalize_provider_run",),
    "torchlens._runnable.state": ("load_trace_state_dict",),
    # Coordinator verbs reached through the RunnableCoordinator boundary keep
    # their transport-side homes (contract E2).
    "torchlens._io.runnable": ("build_sparse_run_descriptor",),
    "torchlens._io.runnable_load": (
        "parse_sparse_run_descriptor",
        "attach_sparse_run_readiness",
    ),
    "torchlens.intervention._trace_run": ("run_fast_loaded_trace", "run_fast_live_trace"),
}

# Trace.run keyword-conflict matrix, bound as the CODE LIST (S1 contract:
# totality binds to codes, not counts). Every new run keyword lands with its
# conflict-matrix row here in the same change.
RUN_DOOR_CONFLICT_CODES: frozenset[str] = frozenset(
    {
        "run_legacy_arguments_conflict",
        "run_legacy_options_conflict",
        "run_fast_divergence_policy_invalid",
        "run_fast_requires_inputs",
        "run_source_model_collected",
        "run_capability_unavailable",
    }
)


# Kept: sole provocation of an error code.
def test_run_door_conflict_matrix_codes_present_in_door_source() -> None:
    """The conflict-matrix code list stays raised on the run-door path.

    Literal codes must appear in the door source itself; enum-spelled codes
    (``run_capability_unavailable``) must stay declared as ``RunnableErrorCode``
    members whose member name is referenced by runnable machinery the door
    delegates to.
    """

    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent / "torchlens"
    door_source = (package_root / "intervention" / "_trace_run.py").read_text(encoding="utf-8")
    enum_values = {member.value for member in RunnableErrorCode}
    for code in sorted(RUN_DOOR_CONFLICT_CODES):
        if f'"{code}"' in door_source:
            continue
        assert code in enum_values, (
            f"run-door conflict code {code} is neither a door literal nor an enum member"
        )
        member_name = RunnableErrorCode(code).name
        referenced = any(
            member_name in path.read_text(encoding="utf-8")
            for path in (package_root / "_runnable").glob("*.py")
            if path.name != "__init__.py"
        )
        assert referenced, (
            f"run-door conflict code {code} has no enum-member raise site in runnable machinery"
        )


# Import purity of the vocabulary facade (docs/reference/runnable_model.md: the
# vocabulary "remains behavior-free and must not import I/O, capture, state
# preparation, or execution modules"). The package ``__init__.py`` is that
# vocabulary, so it may not import its own submodules, and its only eager
# TorchLens import edges stay absent so it stays outside the import cycle.
_RUNNABLE_FACADE_FORBIDDEN_PREFIXES = (
    "torchlens._io",
    "torchlens._capture",
    "torchlens.backends",
)


def _runnable_facade_imports() -> list[tuple[str, int, bool]]:
    """Return ``(absolute target, line, eager)`` for every import in the facade."""

    import ast
    from pathlib import Path

    package = "torchlens._runnable"
    init_path = Path(__file__).resolve().parent.parent / "torchlens" / "_runnable" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))

    def is_type_checking(test: ast.expr) -> bool:
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    found: list[tuple[str, int, bool]] = []

    def visit(node: ast.AST, eager: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and is_type_checking(child.test):
                for stmt in child.body:
                    visit_stmt(stmt, False)
                for stmt in child.orelse:
                    visit_stmt(stmt, eager)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, False)
            else:
                visit_stmt(child, eager)

    def visit_stmt(node: ast.AST, eager: bool) -> None:
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno, eager) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.append((module, node.lineno, eager))
            if module == package:
                found.extend(
                    (f"{package}.{alias.name}", node.lineno, eager) for alias in node.names
                )
        else:
            visit(node, eager)

    visit(tree, True)
    return found
