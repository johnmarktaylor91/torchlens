"""Code-derived universe censuses for the composition harness (compo rows 0.2-0.4).

Every derivation here consumes the ONE shared source parse
(``tests/_source_corpus.py``) -- independent AST walks are forbidden by the
compo memo (section 3.2, "one parse, many universes").

The two diagnostic censuses:

* **S-17 refusal-site census** (:func:`raise_sites`): every ``raise`` of an
  ``*Error``-named class, INCLUDING factory-call closure -- ``raise f(...)``
  where ``f`` is a package function whose every return constructs an
  ``*Error`` (fixpoint over factories). Without the closure the largest
  class was undercounted 27% (memo 3.5). Declared residual (descriptor
  ``filters``): call sites of always-raising helpers OUTSIDE raise position
  are not yet derived (review trigger: wave A).
* **S-18 warning census** (:func:`warn_sites`): every ``warnings.warn`` call,
  ALIAS-RESOLVING -- ``import warnings as _w; _w.warn(...)`` and
  ``from warnings import warn as w; w(...)`` count (the literal
  ``warnings.warn`` scan provably misses live sites; the adequacy plant is an
  aliased site).

Site keys are line-number-free (module, scope, class, ordinal) so ordinary
code motion does not churn the classification ledger.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from _source_corpus import PACKAGE_ROOT, package_ast, package_files

__all__ = [
    "CAPTURE_OPTIONS_PRIVATE_FIELDS",
    "ENV_READER_CALLEES",
    "RaiseSite",
    "WarnSite",
    "capture_option_fields",
    "env_var_reads",
    "env_var_reads_in_tree",
    "package_error_classes",
    "public_option_fields",
    "raise_sites",
    "raise_sites_in_tree",
    "warn_sites",
    "warn_sites_in_tree",
]


def _module_name(path: Path) -> str:
    """Dotted module name for one package file."""

    parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _terminal_name(node: ast.expr) -> str:
    """Rightmost identifier of a Name/Attribute expression ('' otherwise)."""

    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_error_name(name: str) -> bool:
    """The universe's class filter: ``*Error``-named exception classes."""

    return name.endswith("Error")


def _call_has_code_kwarg(call: ast.Call) -> bool:
    """Whether a constructor call passes a ``code=`` keyword."""

    return any(keyword.arg == "code" for keyword in call.keywords)


@dataclass(frozen=True)
class RaiseSite:
    """One raise site of an ``*Error`` class (S-17 universe member).

    Parameters
    ----------
    module:
        Dotted module containing the site.
    scope:
        Enclosing function/class qualname (``<module>`` at module level).
    error_class:
        The ``*Error`` class name raised (factory sites report the factory's
        constructed class).
    via_factory:
        Factory function name for closure-derived sites; ``""`` for direct.
    ordinal:
        Per-(module, scope, error_class, via_factory) occurrence ordinal.
    has_code:
        Whether the construction passes a ``code=`` keyword.
    """

    module: str
    scope: str
    error_class: str
    via_factory: str
    ordinal: int
    has_code: bool

    @property
    def site_key(self) -> str:
        """Stable, line-number-free ledger key."""

        return f"{self.module}:{self.scope}:{self.error_class}:{self.via_factory}:{self.ordinal}"


@dataclass(frozen=True)
class WarnSite:
    """One ``warnings.warn`` call site (S-18 universe member).

    Parameters
    ----------
    module:
        Dotted module containing the site.
    scope:
        Enclosing function/class qualname (``<module>`` at module level).
    aliased:
        Whether the call reaches ``warn`` through a module alias or bare-name
        import (the shapes a literal ``warnings.warn`` scan misses).
    ordinal:
        Per-(module, scope) occurrence ordinal.
    has_code:
        Whether the warned instance/category construction passes ``code=``.
    category:
        Best-effort category/instance class name (``""`` when not statically
        visible).
    """

    module: str
    scope: str
    aliased: bool
    ordinal: int
    has_code: bool
    category: str
    code: str = ""

    @property
    def site_key(self) -> str:
        """Stable, line-number-free ledger key."""

        return f"{self.module}:{self.scope}:{self.ordinal}"


class _ScopedVisitor(ast.NodeVisitor):
    """Base visitor tracking the enclosing function/class qualname."""

    def __init__(self) -> None:
        self._scope: list[str] = []

    @property
    def scope(self) -> str:
        return ".".join(self._scope) if self._scope else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def package_error_classes(trees: dict[str, ast.Module]) -> frozenset[str]:
    """Every ``*Error`` class DEFINED in the package (the S-17 class filter).

    The universe is raise sites of TorchLens's own error classes (the memo's
    159-class boundary), not builtin ``ValueError``/``TypeError`` raises --
    those belong to the wider refusal surface reviewed at wave A.
    """

    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_error_name(node.name):
                names.add(node.name)
    return frozenset(names)


def _error_factories(trees: dict[str, ast.Module]) -> dict[str, str]:
    """Package functions whose EVERY return constructs an ``*Error``.

    Fixpoint over factory names, so a factory may return another factory's
    call. Name-keyed across modules (a redefinition collision would merge
    rows; acceptable at wave 0 and disclosed in the universe descriptor).

    Returns
    -------
    dict[str, str]
        Factory function name -> constructed error class name.
    """

    def _own_return_callees(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str] | None:
        """Terminal callee names of the function's own returns.

        ``None`` when the function cannot be a factory (no returns, or a
        return that is not a bare constructor/factory call).
        """

        callees: list[str] = []
        stack: list[ast.AST] = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Return):
                if not isinstance(node.value, ast.Call):
                    return None
                callees.append(_terminal_name(node.value.func))
            stack.extend(ast.iter_child_nodes(node))
        return callees or None

    # One walk collects candidate return-callee lists; the fixpoint then runs
    # over lightweight name data (never re-walking trees -- the naive form
    # cost ~6s per session on the 546-module corpus).
    candidates: dict[str, list[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callees = _own_return_callees(node)
                if callees is not None:
                    candidates[node.name] = callees

    factories: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, callees in candidates.items():
            if name in factories:
                continue
            constructed: list[str] = []
            for callee in callees:
                if _is_error_name(callee):
                    constructed.append(callee)
                elif callee in factories:
                    constructed.append(factories[callee])
                else:
                    break
            else:
                factories[name] = constructed[0]
                changed = True
    return factories


def raise_sites_in_tree(
    trees: dict[str, ast.Module], *, factory_closure: bool = True
) -> tuple[RaiseSite, ...]:
    """Derive S-17 raise sites from parsed module trees.

    Parameters
    ----------
    trees:
        Dotted module name -> parsed tree.
    factory_closure:
        Whether ``raise f(...)`` through error factories counts. ``False``
        exists ONLY for the adequacy plant (proving the closure finds members
        the direct scan misses).

    Returns
    -------
    tuple[RaiseSite, ...]
        Every derived site, ordinal-keyed.
    """

    package_classes = package_error_classes(trees)
    all_factories = _error_factories(trees)
    factories = (
        {name: cls for name, cls in all_factories.items() if cls in package_classes}
        if factory_closure
        else {}
    )
    sites: list[RaiseSite] = []

    class _Visitor(_ScopedVisitor):
        def __init__(self, module: str) -> None:
            super().__init__()
            self._module = module
            self._ordinals: dict[tuple[str, str, str, str], int] = defaultdict(int)

        def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
            exc = node.exc
            error_class = ""
            via_factory = ""
            has_code = False
            if isinstance(exc, ast.Call):
                callee = _terminal_name(exc.func)
                if callee in package_classes:
                    error_class = callee
                    has_code = _call_has_code_kwarg(exc)
                elif callee in factories:
                    error_class = factories[callee]
                    via_factory = callee
                    has_code = _call_has_code_kwarg(exc)
            elif exc is not None and _terminal_name(exc) in package_classes:
                error_class = _terminal_name(exc)
            if error_class:
                key = (self._module, self.scope, error_class, via_factory)
                self._ordinals[key] += 1
                sites.append(
                    RaiseSite(
                        module=self._module,
                        scope=self.scope,
                        error_class=error_class,
                        via_factory=via_factory,
                        ordinal=self._ordinals[key],
                        has_code=has_code,
                    )
                )
            self.generic_visit(node)

    for module, tree in sorted(trees.items()):
        _Visitor(module).visit(tree)
    return tuple(sites)


def _warn_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Names bound to the warnings module / to ``warnings.warn`` in one module.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(module_aliases, bare_warn_names)``.
    """

    module_aliases: set[str] = set()
    bare_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "warnings":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "warnings":
            for alias in node.names:
                if alias.name == "warn":
                    bare_names.add(alias.asname or alias.name)
    return module_aliases, bare_names


def warn_sites_in_tree(
    trees: dict[str, ast.Module], *, alias_resolving: bool = True
) -> tuple[WarnSite, ...]:
    """Derive S-18 warn sites from parsed module trees.

    Parameters
    ----------
    trees:
        Dotted module name -> parsed tree.
    alias_resolving:
        Whether module-alias and bare-name imports of ``warn`` count.
        ``False`` is the literal ``warnings.warn`` scan, kept ONLY for the
        adequacy plant.

    Returns
    -------
    tuple[WarnSite, ...]
        Every derived site, ordinal-keyed.
    """

    sites: list[WarnSite] = []

    class _Visitor(_ScopedVisitor):
        def __init__(self, module: str, aliases: set[str], bare: set[str]) -> None:
            super().__init__()
            self._module = module
            self._aliases = aliases
            self._bare = bare
            self._ordinals: dict[tuple[str, str], int] = defaultdict(int)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            aliased: bool | None = None
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "warn"
                and isinstance(func.value, ast.Name)
                and func.value.id in self._aliases
            ):
                aliased = func.value.id != "warnings"
            elif isinstance(func, ast.Name) and func.id in self._bare:
                aliased = True
            if aliased is not None and (alias_resolving or not aliased):
                first = node.args[0] if node.args else None
                has_code = isinstance(first, ast.Call) and _call_has_code_kwarg(first)
                code = ""
                category = ""
                if isinstance(first, ast.Call):
                    category = _terminal_name(first.func)
                    for keyword in first.keywords:
                        if (
                            keyword.arg == "code"
                            and isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)
                        ):
                            code = keyword.value.value
                elif len(node.args) > 1:
                    category = _terminal_name(node.args[1])
                else:
                    for keyword in node.keywords:
                        if keyword.arg == "category":
                            category = _terminal_name(keyword.value)
                key = (self._module, self.scope)
                self._ordinals[key] += 1
                sites.append(
                    WarnSite(
                        module=self._module,
                        scope=self.scope,
                        aliased=aliased,
                        ordinal=self._ordinals[key],
                        has_code=has_code,
                        category=category,
                        code=code,
                    )
                )
            self.generic_visit(node)

    for module, tree in sorted(trees.items()):
        aliases, bare = _warn_bindings(tree)
        _Visitor(module, aliases, bare).visit(tree)
    return tuple(sites)


@lru_cache(maxsize=1)
def _package_trees() -> dict[str, ast.Module]:
    """Dotted-name -> tree over the SHARED corpus parse (one walk)."""

    return {_module_name(path): package_ast(path) for path in package_files()}


@lru_cache(maxsize=1)
def raise_sites() -> tuple[RaiseSite, ...]:
    """S-17: every raise site of an ``*Error`` class in the live package."""

    return raise_sites_in_tree(_package_trees())


@lru_cache(maxsize=1)
def warn_sites() -> tuple[WarnSite, ...]:
    """S-18: every ``warnings.warn`` site in the live package, alias-resolved."""

    return warn_sites_in_tree(_package_trees())


#: The one declared private filter on the capture-options universe
#: (red-capability-proved BOTH directions by the registry tests).
CAPTURE_OPTIONS_PRIVATE_FIELDS = frozenset({"_specified_fields"})


def public_option_fields(options_class: type, declared_private: frozenset[str]) -> tuple[str, ...]:
    """Public fields of an options dataclass under an EXPLICIT private filter.

    Fail-closed BOTH directions (memo 3.2): a private-named field outside the
    declared filter refuses (the census goes red rather than silently
    narrowing), and a declared filter row the class lacks refuses (stale
    filters cannot hide).
    """

    import dataclasses

    names = tuple(field.name for field in dataclasses.fields(options_class))
    undeclared_private = {name for name in names if name.startswith("_")} - declared_private
    if undeclared_private:
        raise AssertionError(
            f"{options_class.__name__} grew private fields outside the declared filter "
            f"(extend the filter deliberately): {sorted(undeclared_private)}"
        )
    stale_filter = declared_private - set(names)
    if stale_filter:
        raise AssertionError(
            f"declared private-filter rows do not exist on {options_class.__name__} "
            f"(remove the stale rows): {sorted(stale_filter)}"
        )
    return tuple(name for name in names if name not in declared_private)


def capture_option_fields() -> tuple[str, ...]:
    """The 47 public ``CaptureOptions`` fields (declared private filter applied)."""

    from torchlens.options import CaptureOptions

    return public_option_fields(CaptureOptions, CAPTURE_OPTIONS_PRIVATE_FIELDS)


#: Declared environment-reader spellings (a new reader helper joins here or
#: the census undercounts -- disclosed in the universe descriptor's filters).
ENV_READER_CALLEES = frozenset({"getenv", "closed_bool_env"})


def env_var_reads() -> tuple[str, ...]:
    """Every ``TORCHLENS_*`` environment variable the package READS."""

    return env_var_reads_in_tree(_package_trees())


def env_var_reads_in_tree(trees: dict[str, ast.Module]) -> tuple[str, ...]:
    """Env-var read census over parsed trees.

    Three derivation shapes (memo 3.2): (1) ``os.environ`` method calls /
    subscripts carrying a literal key, (2) declared reader helpers
    (:data:`ENV_READER_CALLEES`) called with a literal first argument,
    (3) module-level ``*_ENV = "TORCHLENS_..."`` constant bindings (names
    later read through variables). Non-literal keys are a declared residual.
    """

    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = _terminal_name(node.func)
                literal_args = [
                    child.value
                    for child in node.args[:1]
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and child.value.startswith("TORCHLENS_")
                ]
                is_environ_method = (
                    isinstance(node.func, ast.Attribute)
                    and _terminal_name(node.func.value) == "environ"
                    and callee in {"get", "pop", "setdefault"}
                )
                if literal_args and (callee in ENV_READER_CALLEES or is_environ_method):
                    names.update(literal_args)
            elif isinstance(node, ast.Subscript):
                if (
                    _terminal_name(node.value) == "environ"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and node.slice.value.startswith("TORCHLENS_")
                ):
                    names.add(node.slice.value)
        for statement in tree.body:
            for target_name in _assigned_env_constant(statement):
                names.add(target_name)
    return tuple(sorted(names))


def _assigned_env_constant(statement: ast.stmt) -> list[str]:
    """``TORCHLENS_*`` values of module-level ``*_ENV = "..."`` assignments."""

    if isinstance(statement, ast.Assign):
        targets = statement.targets
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
        value = statement.value
    else:
        return []
    if not (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.startswith("TORCHLENS_")
    ):
        return []
    named_env = any(
        isinstance(target, ast.Name) and target.id.endswith("_ENV") for target in targets
    )
    return [value.value] if named_env else []
