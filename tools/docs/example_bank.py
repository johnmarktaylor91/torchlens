"""Run, index and measure the example bank (``torchlens/_docs/example_bank/``).

The bank is Markdown shipped in the wheel; :mod:`torchlens.agent._example_bank` parses it.
This tool is the repository side:

* :func:`run_example` executes one example after its fixture, in a scratch directory,
  and compares what it prints with the ``# Output:`` comment (doctest ``ELLIPSIS`` and
  ``NORMALIZE_WHITESPACE``); a ``raises`` example must end in that error class.
* :func:`cover_problems` checks statically that each name an example claims to cover is
  spelled in its code.
* :func:`render_index` writes ``INDEX.md`` (by task, then by name); torch-free.
* :func:`coverage_rows` lists the public names the bank owes an example (root ``tl.*``
  names and the members of the core objects) and which examples cover each; it imports
  torchlens through the lockstep inventory.

Command line::

    python tools/docs/example_bank.py index [--check]
    python tools/docs/example_bank.py run [--group G ...] [--id ID ...] [--bless] [--json OUT]
    python tools/docs/example_bank.py coverage [--out PATH]
    python tools/docs/example_bank.py bless --from RUN.json

``bless --from`` applies the printed output a ``run --json`` recorded (on a machine with
torch) to the bank files of this checkout, torch-free.

``--bless`` rewrites the ``# Output:`` comment of every example that ran without raising
from what it printed; review the diff before committing.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import doctest
import importlib
import io
import json
import os
import re
import runpy
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from torchlens.agent import _example_bank as bank  # noqa: E402

BANK = REPO / "torchlens" / "_docs" / "example_bank"
INDEX = BANK / "INDEX.md"
#: Classes whose public members the bank owes an example, by short name.
CORE_CLASSES = (
    "Trace",
    "Layer",
    "Op",
    "Module",
    "ModuleCall",
    "Param",
    "Buffer",
    "GradFn",
    "Bundle",
    "Selection",
    "CaptureOptions",
    "VisualizationOptions",
)
#: Defining modules of the core classes (the inventory owner is ``module:Class``).
CORE_OWNERS = frozenset(
    {
        "torchlens._data_classes.trace:Trace",
        "torchlens._data_classes.layer:Layer",
        "torchlens._data_classes.op:Op",
        "torchlens._data_classes.module:Module",
        "torchlens._data_classes.module:ModuleCall",
        "torchlens._data_classes.param:Param",
        "torchlens._data_classes.buffer:Buffer",
        "torchlens._data_classes.grad_fn:GradFn",
        "torchlens.bundle:Bundle",
        "torchlens._selection.selection:Selection",
        "torchlens.options:CaptureOptions",
        "torchlens.options:VisualizationOptions",
    }
)
#: How a protocol dunder is spelled in code.
DUNDER_SPELLING = {
    "__len__": r"\blen\(",
    "__iter__": r"\bfor\b|\blist\(|\bsorted\(|\btuple\(",
    "__getitem__": r"\w\[",
    "__contains__": r"\bin\b",
    "__enter__": r"\bwith\b",
    "__exit__": r"\bwith\b",
    "__bool__": r"\bif\b|\bbool\(|\bnot\b",
}
#: Doctest flags the output comparison uses.
#: Most code lines an example may have (the ``# Output:`` comment excluded).
MAX_CODE_LINES = 15
OPTION_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Outcome of running one example.

    Attributes
    ----------
    id:
        Example id.
    status:
        ``ok``, ``skipped`` (its optional module is absent), ``mismatch`` (ran, printed
        something else), ``error`` (raised, or did not raise its declared error).
    printed:
        What the example printed.
    detail:
        The error or the comparison report.
    seconds:
        Wall time of the example and its fixture.
    """

    id: str
    status: str
    printed: str
    detail: str
    seconds: float


def load_examples(bank_dir: Path = BANK) -> list[bank.BankExample]:
    """Parse every group file of a bank checkout.

    Parameters
    ----------
    bank_dir:
        The bank folder.

    Returns
    -------
    list[bank.BankExample]
        Examples ordered by group file, then file order.
    """

    examples: list[bank.BankExample] = []
    for path in sorted(bank_dir.glob("*.md")):
        if path.name not in {"README.md", "INDEX.md"}:
            examples.extend(bank.parse_group(path.stem, path.read_text(encoding="utf-8")))
    return examples


def load_fixtures(bank_dir: Path = BANK) -> dict[str, str]:
    """Return fixture name to setup code from a bank checkout's README."""

    readme = (bank_dir / "README.md").read_text(encoding="utf-8")
    fixtures: dict[str, str] = {}
    for match in re.finditer(r"^### fixture: ([a-z0-9-]+)\s*$", readme, re.MULTILINE):
        fence = re.compile(r"^```python\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
        found = fence.search(readme, match.end())
        if found is not None:
            fixtures[match.group(1)] = found.group(1).rstrip("\n")
    return fixtures


def _outputs_match(expected: str, printed: str) -> bool:
    """Compare printed text with the expected output under the doctest flags."""

    want = expected.rstrip("\n") + "\n" if expected else ""
    return doctest.OutputChecker().check_output(want, printed, OPTION_FLAGS)


def _raised_matches(error: BaseException, name: str) -> bool:
    """Return whether ``error`` is an instance of a class spelled ``name``."""

    return any(cls.__name__ == name for cls in type(error).__mro__)


def run_example(
    example: bank.BankExample, fixtures: dict[str, str], workdir: Path | None = None
) -> RunResult:
    """Run one example after its fixture and compare what it prints.

    Parameters
    ----------
    example:
        The example.
    fixtures:
        Fixture setup code by name.
    workdir:
        Scratch directory the example runs in (a fresh temporary one when omitted). The
        script file is written there so source-reading features see real source.

    Returns
    -------
    RunResult
        The outcome.
    """

    if example.requires is not None:
        try:
            importlib.import_module(example.requires)
        except ImportError:
            return RunResult(example.id, "skipped", "", example.requires, 0.0)
    with contextlib.ExitStack() as stack:
        if workdir is None:
            workdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        script = workdir / f"bank_{example.id.replace('-', '_')}.py"
        script.write_text(f"{fixtures[example.fixture]}\n\n{example.code}\n", encoding="utf-8")
        printed = io.StringIO()
        started = time.perf_counter()
        previous = Path.cwd()
        error: BaseException | None = None
        try:
            os.chdir(workdir)
            with contextlib.redirect_stdout(printed):
                runpy.run_path(str(script), run_name="__main__")
        except Exception as raised:  # the example's own failure is the result
            error = raised
        finally:
            os.chdir(previous)
        seconds = time.perf_counter() - started
    text = printed.getvalue()
    if example.raises is not None:
        if error is None or not _raised_matches(error, example.raises):
            detail = (
                f"expected {example.raises}, got {type(error).__name__ if error else 'no error'}"
            )
            return RunResult(example.id, "error", text, detail, seconds)
    elif error is not None:
        return RunResult(example.id, "error", text, f"{type(error).__name__}: {error}", seconds)
    if not _outputs_match(example.expected, text):
        detail = f"expected:\n{example.expected}\nprinted:\n{text}"
        return RunResult(example.id, "mismatch", text, detail, seconds)
    return RunResult(example.id, "ok", text, "", seconds)


def cover_problems(example: bank.BankExample) -> list[str]:
    """Return the example's static problems: unspelled covers, size, empty output.

    A covered name must be spelled in the code; the code must be 1 to 15 lines; and an
    example that does not raise must show some output other than ``...``.

    Parameters
    ----------
    example:
        The example.

    Returns
    -------
    list[str]
        One message per problem.
    """

    problems = []
    code_lines = [line for line in example.code.splitlines() if line.strip()]
    if not 1 <= len(code_lines) <= MAX_CODE_LINES:
        problems.append(f"{example.id}: {len(code_lines)} code lines, want 1 to {MAX_CODE_LINES}")
    shown = [line for line in example.expected.splitlines() if line.strip() not in ("", "...")]
    if not shown and not example.raises:
        problems.append(f"{example.id}: output shows nothing but '...'")
    for name in example.covers:
        if name.startswith("tl."):
            pattern = re.escape(name) + r"\b"
        elif "." in name:
            member = name.rsplit(".", 1)[1]
            pattern = DUNDER_SPELLING.get(
                member, rf"\.{re.escape(member)}\b|\b{re.escape(member)}="
            )
        else:
            problems.append(f"{example.id}: covered name {name!r} is not tl.X or Class.member")
            continue
        if re.search(pattern, example.code) is None:
            problems.append(f"{example.id}: covers {name!r} but the code does not spell it")
    return problems


def render_index(examples: list[bank.BankExample]) -> str:
    """Render ``INDEX.md``: examples by task, then by public name.

    Parameters
    ----------
    examples:
        Every bank example.

    Returns
    -------
    str
        The index text.
    """

    def link(example: bank.BankExample) -> str:
        return f"[{example.id}]({example.group}.md#{example.id})"

    by_task: dict[str, list[bank.BankExample]] = {}
    by_name: dict[str, list[bank.BankExample]] = {}
    for example in examples:
        for task in example.tasks:
            by_task.setdefault(task, []).append(example)
        for name in example.covers:
            by_name.setdefault(name, []).append(example)
    lines = [
        "# Example bank index",
        "",
        "Generated by `tools/docs/example_bank.py index`; do not edit. Fixtures and the",
        "format are in `README.md`.",
        "",
        f"{len(examples)} examples, {len(by_name)} public names, {len(by_task)} tasks.",
        "",
        "## By task",
        "",
        "| Task | Examples |",
        "|---|---|",
    ]
    for task in sorted(by_task):
        lines.append(f"| {task} | {', '.join(link(e) for e in by_task[task])} |")
    lines += ["", "## By name", "", "| Name | Examples |", "|---|---|"]
    for name in sorted(by_name, key=str.lower):
        lines.append(f"| `{name}` | {', '.join(link(e) for e in by_name[name])} |")
    return "\n".join(lines) + "\n"


def bless(results: Iterable[RunResult], bank_dir: Path = BANK) -> list[str]:
    """Rewrite the ``# Output:`` comment of examples from what they printed.

    Parameters
    ----------
    results:
        Run results; only ``ok`` and ``mismatch`` ones are written.
    bank_dir:
        The bank folder.

    Returns
    -------
    list[str]
        Ids whose output comment changed.
    """

    printed = {r.id: r.printed for r in results if r.status in {"ok", "mismatch"}}
    changed: list[str] = []
    for example in load_examples(bank_dir):
        new_output = printed.get(example.id, example.expected).rstrip("\n")
        if example.id not in printed or new_output == example.expected:
            continue
        path = bank_dir / f"{example.group}.md"
        text = path.read_text(encoding="utf-8")
        head = text.index(f"### {example.id}\n")
        start = text.index("```python\n", head) + len("```python\n")
        end = text.index("\n```", start)
        renewed = dataclasses.replace(example, expected=new_output).text()
        path.write_text(text[:start] + renewed + text[end:], encoding="utf-8")
        changed.append(example.id)
    return changed


def _inventory_rows() -> list[dict[str, Any]]:
    """Return the lockstep inventory rows of the importable surface (imports torchlens)."""

    sys.path.insert(0, str(REPO / "tools"))
    from name_lockstep.inventory import build_inventory

    rows: list[dict[str, Any]] = build_inventory("torchlens")["rows"]
    return rows


def owed_names(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Return the public names the bank owes an example, with their kind.

    Parameters
    ----------
    rows:
        Lockstep inventory rows.

    Returns
    -------
    dict[str, str]
        ``tl.X`` root names (``root function``, ``root class``, ``root module``) and
        ``Class.member`` names of the core classes (``member <subkind>``).
    """

    owed: dict[str, str] = {}
    for row in rows:
        if row["kind"] == "root":
            owed[row["path"]] = f"root {row['subkind']}"
        elif row["kind"] == "member" and row["owner"] in CORE_OWNERS:
            owed[f"{row['owner'].rsplit(':', 1)[1]}.{row['name']}"] = f"member {row['subkind']}"
    return owed


def known_names(rows: list[dict[str, Any]]) -> set[str]:
    """Return every name a ``covers`` field may spell (root, export, Class.member)."""

    names = {row["path"] for row in rows if row["kind"] in {"root", "export"}}
    names |= {
        f"{row['owner'].rsplit(':', 1)[1]}.{row['name']}" for row in rows if row["kind"] == "member"
    }
    return names


def _module_covered(name: str, covered: dict[str, list[str]]) -> list[str]:
    """Return examples covering a root module: the module itself or any name inside it."""

    prefix = name + "."
    return sorted(
        {e for key, ids in covered.items() if key == name or key.startswith(prefix) for e in ids}
    )


def coverage_rows(
    examples: list[bank.BankExample], rows: list[dict[str, Any]]
) -> list[tuple[str, str, list[str]]]:
    """Return ``(name, kind, example ids)`` for every owed public name.

    Parameters
    ----------
    examples:
        Every bank example.
    rows:
        Lockstep inventory rows.

    Returns
    -------
    list[tuple[str, str, list[str]]]
        Sorted by name; an empty id list means the bank has no example for it.
    """

    covered: dict[str, list[str]] = {}
    for example in examples:
        for name in example.covers:
            covered.setdefault(name, []).append(example.id)
    result = []
    for name, kind in sorted(owed_names(rows).items()):
        ids = _module_covered(name, covered) if kind == "root module" else covered.get(name, [])
        result.append((name, kind, ids))
    return result


def _cmd_run(args: argparse.Namespace) -> int:
    """Run selected examples, print one line each, optionally bless and write JSON."""

    fixtures = load_fixtures()
    chosen = [
        e
        for e in load_examples()
        if (not args.group or e.group in args.group) and (not args.id or e.id in args.id)
    ]
    results = []
    for example in chosen:
        result = run_example(example, fixtures)
        results.append(result)
        print(f"{result.status:8} {result.seconds:6.2f}s {example.group}/{result.id}", flush=True)
        if result.status in {"error", "mismatch"}:
            print("    " + result.detail.replace("\n", "\n    ")[:3000], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=1))
    if args.bless:
        for changed in bless(results):
            print(f"blessed  {changed}")
    bad = [r for r in results if r.status in {"error", "mismatch"}]
    print(f"examples {len(results)} ok {sum(r.status == 'ok' for r in results)} bad {len(bad)}")
    return 1 if bad and not args.bless else 0


def _cmd_index(args: argparse.Namespace) -> int:
    """Write INDEX.md, or with --check report whether it is current."""

    text = render_index(load_examples())
    if args.check:
        current = INDEX.exists() and INDEX.read_text(encoding="utf-8") == text
        print("INDEX.md current" if current else "INDEX.md stale: run example_bank.py index")
        return 0 if current else 1
    INDEX.write_text(text, encoding="utf-8")
    print(f"wrote {INDEX.relative_to(REPO)}")
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    """Write the coverage table of owed names (imports torchlens)."""

    table = coverage_rows(load_examples(), _inventory_rows())
    done = sum(1 for _, _, ids in table if ids)
    lines = [f"owed {len(table)} covered {done} missing {len(table) - done}", ""]
    lines += [f"{name}\t{kind}\t{','.join(ids) or '-'}" for name, kind, ids in table]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(lines[0])
    return 0


def _cmd_bless(args: argparse.Namespace) -> int:
    """Apply a recorded run's printed output to the bank files."""

    raw = json.loads(Path(args.source).read_text(encoding="utf-8"))
    for changed in bless(RunResult(**item) for item in raw):
        print(f"blessed  {changed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run examples and compare output")
    run.add_argument("--group", action="append", default=[])
    run.add_argument("--id", action="append", default=[])
    run.add_argument("--bless", action="store_true")
    run.add_argument("--json")
    index = sub.add_parser("index", help="write INDEX.md")
    index.add_argument("--check", action="store_true")
    coverage = sub.add_parser("coverage", help="list owed names and their examples")
    coverage.add_argument("--out")
    blessing = sub.add_parser("bless", help="apply a run's recorded output")
    blessing.add_argument("--from", dest="source", required=True)
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "index": _cmd_index,
        "coverage": _cmd_coverage,
        "bless": _cmd_bless,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
