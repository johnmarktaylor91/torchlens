"""Repo-wide docstring coverage gate for ``torchlens/`` (grind r3, R69).

The exemption policy is the orchestrator's D3 ruling: dunder methods and
trivial pass-through properties are exempt; EVERY other function, method, and
class in the package -- public or load-bearing internal -- requires a
docstring. Both exemptions are detected STRUCTURALLY (name shape / body shape),
not from a hand-maintained name list, so a newly added dunder is auto-exempt
while a newly added helper is not.

The only per-site escape hatch is :data:`DEFERRED`, an exact ledger. It is
checked both ways: an unledgered undocumented def fails, and a ledger entry
that has since been documented (or has disappeared) ALSO fails, so the ledger
cannot rot into a permanent exemption.
"""

from __future__ import annotations

import ast
import functools
import pathlib

from _source_corpus import module_ast, package_files

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "torchlens"

#: Sites deliberately left undocumented for now, each with the reason it is
#: deferred rather than exempt. Entries are ``(relative path, qualname)``.
#:
#: EMPTY, and that is the intended steady state: the ledger existed only to
#: park the nine sites inside the giant files being SPLIT in grind round 3,
#: and the split landed, so each was documented in its new module instead.
#: Adding an entry is a last resort -- write the docstring. The staleness test
#: keeps the ledger from rotting into a permanent exemption in either
#: direction.
#: Shared reason for the 35 call-rooted OpRecord facet properties surfaced
#: when the trivial-property exemption stopped laundering call-rooted chains
#: (b9 R69-1 round 3): each is ``return self._facet_or_default(...).x``, which
#: carries real missing-facet defaulting semantics. torchlens/ is fenced for
#: this lane; the docstrings ride the ir/ source lane.
DEFERRED: dict[tuple[str, str], str] = {
    # EMPTY is the intended steady state (this ledger's original header).
    # 2026-08-16 (fixwave-7, R69): the 39 rows parked here for three fix
    # waves -- 35 OpRecord facet-defaulting properties and 4 identity shims
    # whose "owning lanes" had long since shipped -- are DOCUMENTED and the
    # ledger is emptied. A new row needs a live owner and a dated reason;
    # parking on a lane that already settled is how these went unowned.
}

_PROPERTY_DECORATORS = frozenset({"property", "cached_property", "functools.cached_property"})


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Render each decorator as a dotted name, ignoring call arguments."""

    names: list[str] = []
    for decorator in node.decorator_list:
        current: ast.expr = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        names.append(".".join(reversed(parts)))
    return names


def _is_dunder(name: str) -> bool:
    """Whether ``name`` is a dunder (``__x__``) rather than a normal identifier."""

    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _is_trivial_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``node`` is a property whose whole body returns one plain value.

    A trivial property is a pass-through: ``return self._x``, ``return X``, or
    ``return <literal>``. Anything that computes -- a comprehension, a call, a
    conditional, more than one statement -- is NOT trivial and needs a
    docstring stating what it derives.
    """

    if not any(name in _PROPERTY_DECORATORS for name in _decorator_names(node)):
        return False
    if len(node.body) != 1:
        return False
    statement = node.body[0]
    if not isinstance(statement, ast.Return):
        return False
    # The returned expression must be a PLAIN attribute chain rooted at a name
    # (or a bare name/literal). A chain rooted at a CALL — e.g.
    # ``return self._facet_or_default("graph").x`` — computes (missing-facet
    # defaulting semantics) and is NOT trivial; the old isinstance-on-the-tip
    # check laundered 35 such computing properties through the exemption
    # (b9 R69-1 round 3).
    value = statement.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return isinstance(value, (ast.Name, ast.Constant))


def _undocumented(path: pathlib.Path) -> list[tuple[str, int]]:
    """Return ``(qualname, lineno)`` for every required-but-undocumented def."""

    tree = module_ast(path)
    found: list[tuple[str, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qualname = f"{prefix}{child.name}"
                if ast.get_docstring(child) is None:
                    found.append((qualname, child.lineno))
                walk(child, f"{qualname}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{child.name}"
                decorators = _decorator_names(child)
                exempt = (
                    "overload" in decorators
                    or "typing.overload" in decorators
                    or _is_dunder(child.name)
                    or _is_trivial_property(child)
                )
                if not exempt and ast.get_docstring(child) is None:
                    found.append((qualname, child.lineno))
                # Nested defs keep the ENCLOSING CLASS prefix, not the enclosing
                # function's: closure names are not stable identifiers, so the
                # ledger keys on module + name and covers every occurrence.
                walk(child, prefix)
            elif isinstance(
                child,
                (
                    ast.If,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Match,
                    # INTERMEDIATE nodes (b9-sol R69-1 round 3): iter_child_nodes
                    # of a Try yields ExceptHandler nodes (not their bodies) and
                    # of a Match yields match_case nodes — neither matched any
                    # branch, so a def in an except arm or a case body evaded
                    # the claimed repo-wide gate entirely.
                    ast.ExceptHandler,
                    ast.match_case,
                ),
            ):
                # Loop and match bodies hold real defs too (a def under a
                # ``for`` in TraceCore.transaction() shipped undocumented while
                # its sibling outside the loop was caught -- the gate's exact
                # blind spot).
                walk(child, prefix)

    walk(tree, "")
    return found


@functools.lru_cache(maxsize=1)
def _scan() -> dict[tuple[str, str], list[int]]:
    """Map ``(relative path, qualname)`` to every undocumented line for the package.

    The value is a LIST because closure names are not unique within a module;
    one ledger key therefore covers every occurrence of that name in that file.
    """

    result: dict[tuple[str, str], list[int]] = {}
    for path in package_files():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        for qualname, lineno in _undocumented(path):
            result.setdefault((relative, qualname), []).append(lineno)
    return result


def test_every_function_and_class_has_a_docstring() -> None:
    """No undocumented def outside the ``DEFERRED`` ledger (D3 exemption policy)."""

    undocumented = _scan()
    unledgered = sorted(
        f"{path}:{lineno} {qualname}"
        for (path, qualname), linenos in undocumented.items()
        if (path, qualname) not in DEFERRED
        for lineno in linenos
    )
    assert not unledgered, (
        f"{len(unledgered)} def(s) need a docstring (D3: only dunders and trivial "
        f"pass-through properties are exempt):\n  " + "\n  ".join(unledgered)
    )


def test_deferred_ledger_has_no_stale_entries() -> None:
    """Every ``DEFERRED`` entry still names a real undocumented def.

    A one-way gate would let the ledger rot: once the split lane documents one
    of these, its exemption would silently survive forever. This is the other
    direction -- document it (or delete the def) and the ledger must shrink.
    """

    undocumented = _scan()
    stale = sorted(
        f"{path} {qualname}" for path, qualname in DEFERRED if (path, qualname) not in undocumented
    )
    assert not stale, (
        "DEFERRED entries are documented or gone -- delete them from the ledger:\n  "
        + "\n  ".join(stale)
    )


def test_exemption_policy_is_structural() -> None:
    """The two exemptions match body/name SHAPE, so they cannot be gamed by naming."""

    module = ast.parse(
        "class C:\n"
        "    def __repr__(self): return ''\n"
        "    @property\n"
        "    def passthrough(self): return self._x\n"
        "    @property\n"
        "    def deep_passthrough(self): return self.core.inner.x\n"
        "    @property\n"
        "    def derived(self): return [op for op in self._ops if op]\n"
        "    @property\n"
        "    def call_rooted(self): return self._facet_or_default('graph').x\n"
        "    def helper(self): return 1\n"
        "    def _x(self): return 1\n"
    )
    class_node = module.body[0]
    assert isinstance(class_node, ast.ClassDef)
    methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}

    assert _is_dunder("__repr__")
    assert not _is_dunder("_x")
    assert not _is_dunder("__x")
    assert _is_trivial_property(methods["passthrough"])
    assert _is_trivial_property(methods["deep_passthrough"])
    assert not _is_trivial_property(methods["derived"]), "a computing property is not trivial"
    assert not _is_trivial_property(methods["call_rooted"]), (
        "a chain rooted at a CALL computes (missing-facet defaulting) and must "
        "not launder through the pass-through exemption (b9 R69-1)"
    )
    assert not _is_trivial_property(methods["helper"]), "a plain method is not a property"


def test_walker_descends_except_and_match_arms(tmp_path: pathlib.Path) -> None:
    """Defs hidden in except handlers and match cases are visible (b9-sol R69-1).

    ``ast.iter_child_nodes`` of a ``Try`` yields ``ExceptHandler`` nodes (not
    their bodies) and of a ``Match`` yields ``match_case`` nodes; neither used
    to match any walker branch, so a def in either arm evaded the claimed
    repo-wide gate entirely.
    """

    planted = tmp_path / "planted.py"
    planted.write_text(
        "try:\n"
        "    import missing\n"
        "except ImportError:\n"
        "    def hidden_in_except():\n"
        "        return 1\n"
        "\n"
        "match 1:\n"
        "    case 1:\n"
        "        def hidden_in_match():\n"
        "            return 2\n"
    )
    names = {qualname for qualname, _lineno in _undocumented(planted)}
    assert names == {"hidden_in_except", "hidden_in_match"}
