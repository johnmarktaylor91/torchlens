"""What a call's keyword arguments may belong to, for the stale ``kw=`` check.

A removed TorchLens keyword (``format=``, which only ``export_extraction`` took) is
still spelled by a call that passes it to a TorchLens callable, and not by one that
passes a same-named keyword to a third-party callable (``fig.savefig(svg,
format="svg")`` on a matplotlib figure). The gate cannot import the code it reads, so
it resolves each call's callee syntactically and answers with the leaf names the call
may resolve to. A leaf of ``""`` means "cannot be told from TorchLens": no use leaf
equals it, so it never lets a hit be explained away.

A call's leaves are:

- the callee's own leaf (``savefig`` for ``fig.savefig``);
- for a bare name bound by an alias (``import a.b as c``, ``from a import b as c``,
  ``c = a.b``), the aliased callable's leaf too, so an aliased TorchLens callable is
  never taken for a foreign one;
- ``""`` when the callee's receiver traces back to TorchLens (``tl.record(...)``,
  ``trace.draw(...)`` after ``trace = tl.trace(...)``): a TorchLens callable may
  forward keywords it does not list, so its keywords are never foreign;
- ``""`` when the callee is a function defined in the same file or notebook that does
  not name the keyword itself (a local ``render_png(trace, **kw)`` wrapper that hands
  ``**kw`` to ``tl.draw``);
- ``""`` when the callee is neither a name nor an attribute (a call result, a
  subscript).

Every rule only adds leaves, so a call is foreign only when its leaf is outside the
gone keyword's uses and no rule ties it to TorchLens.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterable, Iterator

#: Leaf meaning "cannot be told from a TorchLens callable".
UNKNOWN = ""


def _leaf(func: ast.expr) -> str:
    """Return a callee's leaf name: ``savefig`` for ``fig.savefig``, ``""`` otherwise."""

    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else UNKNOWN


def _root(node: ast.expr) -> str:
    """Return the name an attribute, subscript or call chain starts from (``fig`` for
    ``fig.axes[0].plot``), or ``""`` when it starts from anything else."""

    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else UNKNOWN


def _roots(node: ast.expr) -> set[str]:
    """Return the names a bound value may start from, looking inside literals.

    ``bundle, lengths = tl.Bundle(), {}`` and ``for fn, kw in [(tl.x.f, {}), ...]``
    bind TorchLens objects through a tuple or list literal; their elements' roots
    count, as do both arms of ``a if c else b`` and ``a or b``.
    """

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_roots(elt) for elt in node.elts))
    if isinstance(node, ast.Dict):
        parts = [*(key for key in node.keys if key is not None), *node.values]
        return set().union(*(_roots(part) for part in parts))
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _roots(node.elt) | set().union(*(_roots(g.iter) for g in node.generators))
    if isinstance(node, ast.DictComp):
        return _roots(node.key) | _roots(node.value)
    if isinstance(node, ast.IfExp):
        return _roots(node.body) | _roots(node.orelse)
    if isinstance(node, ast.BoolOp):
        return set().union(*(_roots(value) for value in node.values))
    if isinstance(node, ast.Starred):
        return _roots(node.value)
    return {_root(node)}


def _target_names(target: ast.expr) -> Iterator[str]:
    """Yield the plain names an assignment, loop or ``with`` target binds."""

    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            yield node.id


def _bindings(tree: ast.AST) -> Iterator[tuple[list[ast.expr], ast.expr]]:
    """Yield ``(targets, value)`` for every assignment-like binding in a tree."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            yield node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            yield [node.target], node.value
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            yield [node.target], node.iter
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    yield [item.optional_vars], item.context_expr


def _is_torchlens(node: ast.Import | ast.ImportFrom, module: str) -> bool:
    """Return whether an import statement's module is TorchLens (or relative)."""

    if isinstance(node, ast.ImportFrom) and node.level:
        return True
    return module == "torchlens" or module.startswith("torchlens.")


@dataclasses.dataclass(frozen=True)
class Receivers:
    """Name facts of one file or notebook, read from every parsed tree in it.

    Parameters
    ----------
    aliases
        Bound name to the leaf of the callable it aliases.
    torchlens
        Names that trace back to a TorchLens import.
    local_defs
        Function or class name defined here to the keyword names every definition of
        it declares explicitly (``None`` when some definition takes ``**kwargs``).
    """

    aliases: dict[str, str]
    torchlens: frozenset[str]
    local_defs: dict[str, frozenset[str] | None]

    @classmethod
    def of(cls, trees: Iterable[ast.AST]) -> Receivers:
        """Read the name facts of a file (one tree) or a notebook (one tree per cell).

        Parameters
        ----------
        trees
            Parsed modules; cells of one notebook share their names.

        Returns
        -------
        Receivers
            The facts.
        """

        trees = list(trees)
        aliases: dict[str, str] = {}
        tainted: set[str] = set()
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    _read_import(node, aliases, tainted)
        for tree in trees:
            for targets, value in _bindings(tree):
                if isinstance(value, (ast.Name, ast.Attribute)) and _leaf(value):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = _leaf(value)
        return cls(aliases, _spread(trees, tainted), _local_defs(trees))

    def leaves(self, call: ast.Call, keyword: str) -> frozenset[str]:
        """Return the leaf names a call passing ``keyword=`` may resolve to.

        Parameters
        ----------
        call
            The call node.
        keyword
            The keyword argument's name.

        Returns
        -------
        frozenset[str]
            The leaves; ``""`` among them when the call cannot be told from TorchLens.
        """

        leaf = _leaf(call.func)
        found = {leaf}
        if isinstance(call.func, ast.Name) and call.func.id in self.aliases:
            found.add(self.aliases[call.func.id])
        if _root(call.func) in self.torchlens:
            found.add(UNKNOWN)
        if leaf in self.local_defs:
            declared = self.local_defs[leaf]
            if declared is None or keyword not in declared:
                found.add(UNKNOWN)
        return frozenset(found)

    def keyword_calls(self, tree: ast.AST) -> Iterator[tuple[ast.keyword, frozenset[str]]]:
        """Yield every named keyword argument in a tree with its call's leaves.

        Parameters
        ----------
        tree
            A parsed module (one of the trees the facts were read from).

        Yields
        ------
        tuple[ast.keyword, frozenset[str]]
            The keyword node and :meth:`leaves` of its call.
        """

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        yield keyword, self.leaves(node, keyword.arg)


def _read_import(
    node: ast.Import | ast.ImportFrom, aliases: dict[str, str], tainted: set[str]
) -> None:
    """Record an import's aliases and the names it binds to TorchLens."""

    for alias in node.names:
        bound = alias.asname or alias.name.split(".", 1)[0]
        if alias.asname:
            aliases[alias.asname] = alias.name.rsplit(".", 1)[-1]
        module = (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        if _is_torchlens(node, module):
            tainted.add(bound)


def _spread(trees: list[ast.AST], tainted: set[str]) -> frozenset[str]:
    """Close the TorchLens names over bindings whose value starts from one of them."""

    pairs = [
        (list(_names_of(targets)), _roots(value))
        for tree in trees
        for targets, value in _bindings(tree)
    ]
    changed = True
    while changed:
        changed = False
        for names, roots in pairs:
            if not roots.isdisjoint(tainted) and not tainted.issuperset(names):
                tainted.update(names)
                changed = True
    return frozenset(tainted)


def _names_of(targets: list[ast.expr]) -> Iterator[str]:
    """Yield every plain name a list of targets binds."""

    for target in targets:
        yield from _target_names(target)


def _local_defs(trees: list[ast.AST]) -> dict[str, frozenset[str] | None]:
    """Map each function or class defined in the trees to the keywords it declares."""

    found: dict[str, frozenset[str] | None] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                declared = None if args.kwarg else _declared(args)
            elif isinstance(node, ast.ClassDef):
                declared = None
            else:
                continue
            if node.name not in found:
                found[node.name] = declared
                continue
            previous = found[node.name]
            both = previous is not None and declared is not None
            found[node.name] = previous & declared if both else None  # type: ignore[operator]
    return found


def _declared(args: ast.arguments) -> frozenset[str]:
    """Return the keyword names a signature accepts explicitly."""

    return frozenset(a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))
