"""Reusable AST inventories for crawler anti-substitution guards."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path


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


def _sensitive_edge_inventory(
    source: str, sensitive_suffixes: Set[str]
) -> Counter[tuple[str, str]]:
    """Return every classified admission, authority, and append call edge.

    Parameters
    ----------
    source:
        Complete Python source to inspect.
    sensitive_suffixes:
        Closed call-name suffixes to inventory.

    Returns
    -------
    collections.Counter[tuple[str, str]]
        Exact enclosing-owner and call-suffix counts.
    """

    tree, parents, names = _tree_context(source)
    found: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _attribute_name(node.func)
        for suffix in sensitive_suffixes:
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
            and self.source_path.name != "test_anti_substitution_inventories.py"
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
            and self.source_path.name != "test_anti_substitution_inventories.py"
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


def _dead_contract_occurrences(
    source: str, dead_symbols: Set[str], dead_options: Set[str]
) -> set[str]:
    """Return forbidden identifiers and exact retired option literals.

    Parameters
    ----------
    source:
        Complete Python source to inspect.
    dead_symbols:
        Closed forbidden identifier set.
    dead_options:
        Closed forbidden string-literal set.

    Returns
    -------
    set[str]
        Forbidden names and literals found in the source.
    """

    tree = ast.parse(source)
    found = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in dead_symbols
    }
    found.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in dead_symbols
    )
    found.update(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in dead_symbols
    )
    found.update(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in {*dead_symbols, *dead_options}
    )
    return found


def _comparison_owner_inventory(
    source: str, protocol_versions: Mapping[str, str]
) -> set[tuple[str, str]]:
    """Return protocol-version comparisons and their enclosing owner.

    Parameters
    ----------
    source:
        Complete Python source to inspect.
    protocol_versions:
        Closed protocol constant mapping whose keys are inventoried.

    Returns
    -------
    set[tuple[str, str]]
        Exact enclosing-owner and protocol-constant pairs.
    """

    tree, parents, names = _tree_context(source)
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            if isinstance(operand, ast.Name) and operand.id in protocol_versions:
                found.add((_enclosing_definition(node, parents, names), operand.id))
    return found
