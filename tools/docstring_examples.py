"""Census and runner for the runnable example every public function and class carries.

The docstring standard (docs/agent-reference/conventions.md, "Docstrings") asks for a
NumPy-format docstring with an ``Examples`` section whose ``>>>`` lines run. This
module is the one place that knows which objects owe an example and how one is run:

* :func:`public_symbols` lists the public functions and classes from the lockstep
  inventory (``tools/name_lockstep``): every root ``tl.X`` name and every export of a
  public namespace, each object once under its shortest public path. Public members of
  those classes are listed by :func:`public_members` for the census; the example
  obligation sits on the function or class.
* :func:`example_of` reads the ``Examples`` section of a docstring as doctest examples.
* :func:`run_example` executes those examples in the ambient namespace the docs pages
  also assume: ``tl`` (torchlens), ``torch``, ``nn``, ``model`` and ``x``.

An example whose first line is ``>>> # requires: <module>`` is skipped when ``<module>``
does not import; that is how an optional-dependency bridge keeps a runnable example.

Command line: ``python tools/docstring_examples.py census [--out PATH]`` writes the
per-package census table (missing examples per package); ``--list`` prints one line per
public symbol with its status.
"""

from __future__ import annotations

import argparse
import dataclasses
import doctest
import importlib
import inspect
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from name_lockstep.inventory import build_inventory  # noqa: E402

PACKAGE = "torchlens"
#: Row kinds that carry the example obligation.
OBLIGATED_SUBKINDS = frozenset({"function", "class"})
#: Member subkinds that are callables users read documentation for.
MEMBER_SUBKINDS = frozenset({"method", "classmethod", "staticmethod", "property"})
#: Comment prefix on an example's first line naming an optional import it needs.
REQUIRES_PREFIX = "# requires:"
#: Doctest option flags every example runs under.
OPTION_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
#: Default home of the committed census.
CENSUS_PATH = REPO / "docs" / "maintainers" / "docstring_examples_census.md"


@dataclasses.dataclass(frozen=True)
class PublicSymbol:
    """One public function or class owing a runnable example.

    Attributes
    ----------
    path
        Shortest public dotted path (``tl.``-prefixed).
    module
        Defining module.
    subkind
        ``function`` or ``class``.
    obj
        The object itself.

    Examples
    --------
    >>> symbol = PublicSymbol("tl.log_forward_pass", "torchlens.user_funcs", "function", print)
    >>> symbol.package
    'user_funcs'
    """

    path: str
    module: str
    subkind: str
    obj: Any

    @property
    def package(self) -> str:
        """Return the top-level package name below ``torchlens`` (``torchlens`` for the root)."""

        parts = self.module.split(".")
        return parts[1] if len(parts) > 1 else parts[0]


@dataclasses.dataclass(frozen=True)
class ExampleResult:
    """Outcome of running one symbol's example.

    Attributes
    ----------
    path
        The symbol's public path.
    status
        ``ok``, ``missing``, ``skipped`` or ``failed``.
    detail
        The doctest report on failure, or the module a skip waited on.

    Examples
    --------
    >>> ExampleResult("tl.x", "ok", "").status
    'ok'
    """

    path: str
    status: str
    detail: str


def _resolve(path: str) -> Any:
    """Return the object a ``tl.``-prefixed public path names.

    Parameters
    ----------
    path
        Public dotted path.

    Returns
    -------
    Any
        The attribute chain's value, or None when a step is unreachable.
    """

    obj: Any = importlib.import_module(PACKAGE)
    for part in path.split(".")[1:]:
        try:
            obj = getattr(obj, part)
        except (AttributeError, ImportError):
            return None
    return obj


def _identity(obj: Any) -> int:
    """Return the identity key an object is deduplicated under across public paths."""

    target = inspect.unwrap(obj) if callable(obj) and not inspect.isclass(obj) else obj
    return id(target)


def public_symbols(inventory: dict[str, Any] | None = None) -> list[PublicSymbol]:
    """List every public function and class once, under its shortest public path.

    Parameters
    ----------
    inventory
        A lockstep inventory (``build_inventory()``); built when omitted.

    Returns
    -------
    list[PublicSymbol]
        Sorted by path.

    Examples
    --------
    >>> symbols = public_symbols()
    >>> any(s.path == "tl.trace" for s in symbols)
    True
    """

    rows = (inventory or build_inventory(PACKAGE))["rows"]
    seen: dict[int, PublicSymbol] = {}
    candidates = [
        row
        for row in rows
        if row["kind"] in ("root", "export") and row["subkind"] in OBLIGATED_SUBKINDS
    ]
    for row in sorted(candidates, key=lambda r: (r["path"].count("."), r["path"])):
        obj = _resolve(row["path"])
        if obj is None or not str(row["module"]).startswith(PACKAGE):
            continue
        key = _identity(obj)
        if key not in seen:
            seen[key] = PublicSymbol(row["path"], row["module"], row["subkind"], obj)
    return sorted(seen.values(), key=lambda s: s.path)


def public_members(inventory: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the inventory's public callable member rows (methods and properties).

    Parameters
    ----------
    inventory
        A lockstep inventory; built when omitted.

    Returns
    -------
    list[dict[str, Any]]
        Member rows whose subkind is a method or property, sorted by id.

    Examples
    --------
    >>> rows = public_members()
    >>> all(row["kind"] == "member" for row in rows)
    True
    """

    rows = (inventory or build_inventory(PACKAGE))["rows"]
    return [row for row in rows if row["kind"] == "member" and row["subkind"] in MEMBER_SUBKINDS]


def example_of(obj: Any) -> list[doctest.Example]:
    """Return the doctest examples in ``obj``'s docstring.

    Parameters
    ----------
    obj
        A function or class.

    Returns
    -------
    list[doctest.Example]
        Empty when the docstring has no ``>>>`` line.

    Examples
    --------
    >>> len(example_of(example_of)) > 0
    True
    """

    doc = inspect.getdoc(obj) or ""
    if ">>>" not in doc:
        return []
    return doctest.DocTestParser().get_examples(doc)


def _required_module(examples: list[doctest.Example]) -> str | None:
    """Return the optional module the first example line declares, if any."""

    first = examples[0].source.strip()
    if first.startswith(REQUIRES_PREFIX):
        return first[len(REQUIRES_PREFIX) :].strip()
    return None


def ambient_namespace() -> dict[str, Any]:
    """Return the namespace every example runs in.

    Returns
    -------
    dict[str, Any]
        ``tl``, ``torch``, ``nn``, a small eval-mode ``model`` and a matching input ``x``;
        the same application-supplied names the docs pages assume. :func:`run_example`
        adds the documented object under its own name.

    Examples
    --------
    >>> sorted(ambient_namespace())
    ['model', 'nn', 'tl', 'torch', 'x']
    """

    import torch
    from torch import nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 2, 3, padding=1))
    return {
        "tl": importlib.import_module(PACKAGE),
        "torch": torch,
        "nn": nn,
        "model": model.eval(),
        "x": torch.randn(1, 3, 8, 8),
    }


def run_example(symbol: PublicSymbol, namespace: dict[str, Any] | None = None) -> ExampleResult:
    """Execute one symbol's example and report the outcome.

    Parameters
    ----------
    symbol
        The public symbol.
    namespace
        Ambient names; a fresh :func:`ambient_namespace` copy when omitted.

    Returns
    -------
    ExampleResult
        ``missing`` when there is no example, ``skipped`` when its declared optional
        module does not import, ``failed`` with the doctest report, else ``ok``.

    Examples
    --------
    >>> run_example(PublicSymbol("tl.example_of", __name__, "function", example_of)).status
    'ok'
    """

    examples = example_of(symbol.obj)
    if not examples:
        return ExampleResult(symbol.path, "missing", "")
    required = _required_module(examples)
    if required is not None:
        try:
            importlib.import_module(required)
        except ImportError:
            return ExampleResult(symbol.path, "skipped", required)
    globs = dict(namespace if namespace is not None else ambient_namespace())
    # The documented object is in scope under its own name, as a reader expects.
    globs.setdefault(symbol.path.rsplit(".", 1)[-1], symbol.obj)
    test = doctest.DocTest(examples, globs, symbol.path, None, None, None)
    report = io.StringIO()
    runner = doctest.DocTestRunner(optionflags=OPTION_FLAGS, verbose=False)
    runner.run(test, out=report.write, clear_globs=True)
    if runner.failures:
        return ExampleResult(symbol.path, "failed", report.getvalue())
    return ExampleResult(symbol.path, "ok", "")


def census_rows(symbols: list[PublicSymbol]) -> list[tuple[str, int, int]]:
    """Count public symbols and missing examples per package.

    Parameters
    ----------
    symbols
        The public symbols.

    Returns
    -------
    list[tuple[str, int, int]]
        ``(package, public symbols, missing examples)`` sorted by package.

    Examples
    --------
    >>> census_rows([PublicSymbol("tl.f", "torchlens.a", "function", example_of)])
    [('a', 1, 0)]
    """

    counts: dict[str, list[int]] = {}
    for symbol in symbols:
        entry = counts.setdefault(symbol.package, [0, 0])
        entry[0] += 1
        if not example_of(symbol.obj):
            entry[1] += 1
    return [(package, total, missing) for package, (total, missing) in sorted(counts.items())]


def render_census(symbols: list[PublicSymbol], members: list[dict[str, Any]]) -> str:
    """Render the census as the committed Markdown table.

    Parameters
    ----------
    symbols
        Public functions and classes.
    members
        Public callable member rows (counted, not gated).

    Returns
    -------
    str
        Markdown with one row per package and a totals row.

    Examples
    --------
    >>> text = render_census([PublicSymbol("tl.f", "torchlens.a", "function", example_of)], [])
    >>> "| a | 1 | 0 |" in text
    True
    """

    rows = census_rows(symbols)
    total = sum(row[1] for row in rows)
    missing = sum(row[2] for row in rows)
    lines = [
        "# Docstring example census",
        "",
        "Public functions and classes (root names and namespace exports, each object once)",
        "and how many lack a runnable `Examples` section. Regenerate with",
        "`python tools/docstring_examples.py census`; the gate is",
        "`tests/test_docstring_examples.py`.",
        "",
        "| package | public functions and classes | missing examples |",
        "|---|---|---|",
    ]
    lines.extend(f"| {package} | {count} | {gap} |" for package, count, gap in rows)
    lines.append(f"| **total** | **{total}** | **{missing}** |")
    lines.append("")
    lines.append(
        f"Public callable members of those classes (methods and properties): {len(members)}."
    )
    lines.append("")
    return "\n".join(lines)


def census_packages(path: Path = CENSUS_PATH) -> list[str]:
    """Read the package names from the committed census table.

    Parameters
    ----------
    path
        The census Markdown file.

    Returns
    -------
    list[str]
        Package names in table order, the totals row excluded; empty when the file is
        missing.

    Examples
    --------
    >>> "intervention" in census_packages()
    True
    """

    if not path.exists():
        return []
    names = []
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[1].isdigit() and not cells[0].startswith("**"):
            names.append(cells[0])
    return names


def iter_status(symbols: list[PublicSymbol]) -> Iterator[str]:
    """Yield one ``<status>\\t<path>\\t<module>`` line per symbol without running examples."""

    for symbol in symbols:
        status = "has" if example_of(symbol.obj) else "missing"
        yield f"{status}\t{symbol.subkind}\t{symbol.path}\t{symbol.module}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        Exit status.

    Examples
    --------
    >>> import tempfile
    >>> out = Path(tempfile.mkdtemp()) / "census.md"
    >>> main(["census", "--out", str(out)])
    wrote ...
    0
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    census = sub.add_parser("census", help="write the per-package census")
    census.add_argument("--out", type=Path, default=CENSUS_PATH)
    census.add_argument("--list", action="store_true", help="print one status line per symbol")
    args = parser.parse_args(argv)
    inventory = build_inventory(PACKAGE)
    symbols = public_symbols(inventory)
    if args.list:
        for line in iter_status(symbols):
            print(line)
    text = render_census(symbols, public_members(inventory))
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(symbols)} public functions and classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
