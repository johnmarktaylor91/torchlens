"""First-call evaluation: does a fresh agent reach for the canonical TorchLens call first?

Usage (from the repository root)::

    python tools/agent_evals/first_call.py selfcheck
    python tools/agent_evals/first_call.py prepare --out DIR [--mode index|bare] [--cards ID,...]
    python tools/agent_evals/first_call.py live --agent-cmd TEMPLATE --out DIR --label TIER
        [--runs N] [--jobs N] [--mode index|bare] [--cards ID,...] [--keep-env NAME ...]
    python tools/agent_evals/first_call.py grade --dir DIR
    python tools/agent_evals/first_call.py summary DIR [DIR ...]

Each task card in ``docs/field-guide/task_cards.yaml`` carries a test prompt in a user's words
that names no TorchLens API. ``prepare`` writes one directory per card: ``prompt.md`` and, in
``index`` mode, an ``AGENTS.md`` holding the block ``torchlens agent init`` writes, pointing at the
agent index inside a shared copy of the package. ``bare`` mode leaves both out, as a baseline. ``live`` runs the agent
command once per card and run, then grades. The template may use ``{prompt}`` (prompt file),
``{instructions}`` (the ``AGENTS.md`` path, empty in ``bare`` mode), ``{package}`` (the vendored
package directory, which plays the installed package: grant the agent read access to it),
``{workdir}`` and ``{solution}``; it runs in the task directory. Live mode calls a model and never runs in CI.
The agent gets a clean environment: only :data:`SUBJECT_ENV` and the names given with
``--keep-env`` (say, the variable holding its credential) pass through, so nothing from the
caller's session (instruction-file switches, hook triggers, tokens) reaches the agent under test.

Grading reads ``solution.py`` without running it. The *entry call* is the first TorchLens call in
source order, skipping helper constructors (selectors such as ``tl.module`` and edits such as
``tl.steer``, which are only ever arguments in a canonical call). A call whose result is stored
in a name that a later TorchLens call takes as an argument hands the entry to that later call
(``edit = tl.when(...)`` then ``tl.trace(model, x, intervene=edit)`` enters through
``tl.trace``). A run passes when the entry
call is the card's canonical entry call or one its ``accept`` list allows. A card marked
``hard_fail`` also fails when any of its ``avoid_calls`` appears anywhere (the per-step capture
trap). A card whose canonical is ``none`` passes when none of its ``avoid_calls`` appears.

The launch bars are the pass rates in ``index`` mode: :data:`BARS`.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "tools" / "docs" / "build_agent_index.py"
INIT_BLOCK_PATH = REPO_ROOT / "torchlens" / "agent" / "_init_block.py"
#: Launch bars: share of runs whose entry call is canonical, per subject tier, index mode.
BARS = {"mid": 0.90, "small": 0.75}
#: Wall-clock limit for one live agent invocation.
AGENT_TIMEOUT_S = 600
#: Environment variables a live agent inherits; ``--keep-env`` adds more by name.
SUBJECT_ENV = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL")
#: Names that are never an entry call even if no canonical uses them as an argument.
EXTRA_HELPER_PREFIXES = ("tl.options.", "tl.label", "tl.op")
#: What each fixture puts in the script's namespace, as the prompt describes it.
FIXTURES = {
    "vision": (
        "`from fixture import *` gives you: `model`, a torchvision ResNet-18 in eval mode "
        "(modules include `layer1` to `layer4` and `avgpool`); `x`, `x_clean` and `x_corrupt`, "
        "image batches of shape (2, 3, 224, 224); `stimuli`, an (N, 3, 224, 224) image tensor, "
        "and `labels`, its class labels; `direction`, a tensor of shape (256,); `batches`, an "
        "iterable of image batches; and `ablated` where the task mentions it."
    ),
    "lm": (
        "`from fixture import *` gives you: `lm`, a Hugging Face causal language model whose "
        "decoder layers are at `model.layers.N`; `tokenizer`; `ids`, a tensor of input token "
        "ids; and `direction`, a tensor with the model's hidden size."
    ),
    "none": "`from fixture import *` gives you every variable the task names.",
}
PROMPT_TEMPLATE = """# Task

This Python project uses TorchLens, which is installed (`import torchlens as tl`); the installed
package is at `{package}`.
{fixture}

{prompt}

Write the complete script to `solution.py` in this directory. Do not try to run it: there is no
model or GPU here. If TorchLens is the wrong tool for this, write a short comment in
`solution.py` saying so and what to use instead.
"""


def _load_by_path(name: str, path: Path) -> ModuleType:
    """Import a module from a file path (``tools/`` is not a package)."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_cards() -> tuple[Any, ...]:
    """Return the task cards, validated by the agent index generator.

    Returns
    -------
    tuple
        The generator's ``Card`` objects, in file order.
    """

    generator = _load_by_path("_tl_build_agent_index", GENERATOR_PATH)
    return tuple(generator.load_card_set().cards)


def _dotted(node: ast.expr) -> str | None:
    """Return ``a.b.c`` for a Name/Attribute chain, else ``None``."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Map each local name bound to torchlens (or a torchlens member) to its ``tl.`` spelling."""

    aliases: dict[str, str] = {"tl": "tl"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "torchlens" or item.name.startswith("torchlens."):
                    local = item.asname or item.name.split(".")[0]
                    target = item.name if item.asname else "torchlens"
                    aliases[local] = "tl" + target[len("torchlens") :]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "torchlens" or node.module.startswith("torchlens."))
        ):
            base = "tl" + node.module[len("torchlens") :]
            for item in node.names:
                aliases[item.asname or item.name] = f"{base}.{item.name}"
    return aliases


def torchlens_calls(source: str) -> list[str]:
    """Return every TorchLens call in ``source`` as a ``tl.`` dotted name, in source order.

    Parameters
    ----------
    source:
        Python source text.

    Returns
    -------
    list[str]
        Normalised call names ordered by line, then column.

    Raises
    ------
    SyntaxError
        When ``source`` does not parse.

    Examples
    --------
    >>> torchlens_calls("import torchlens as lens\\nt = lens.trace(m, x, save=lens.func('relu'))")
    ['tl.trace', 'tl.func']
    """

    return [name for name, _ in call_sites(source)]


def call_sites(source: str) -> list[tuple[str, ast.Call]]:
    """Return every TorchLens call in ``source`` with its node, in source order.

    Parameters
    ----------
    source:
        Python source text.

    Returns
    -------
    list[tuple[str, ast.Call]]
        ``(normalised tl. name, call node)`` ordered by line, then column.

    Raises
    ------
    SyntaxError
        When ``source`` does not parse.
    """

    tree = ast.parse(source)
    aliases = _aliases(tree)
    found: list[tuple[int, int, str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name is None:
            continue
        root, _, rest = name.partition(".")
        if root not in aliases:
            continue
        normalised = aliases[root] + (f".{rest}" if rest else "")
        found.append((node.lineno, node.col_offset, normalised, node))
    found.sort(key=lambda row: (row[0], row[1]))
    return [(name, node) for _, _, name, node in found]


def _assigned_name(tree: ast.AST, call: ast.Call) -> str | None:
    """Return ``N`` when ``call`` is the whole value of an assignment ``N = call``.

    ``call`` may come from another parse of the same source, so it is matched by position.
    """

    span = (call.lineno, call.col_offset, call.end_lineno, call.end_col_offset)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        value, target = node.value, node.targets[0]
        if not isinstance(value, ast.Call) or not isinstance(target, ast.Name):
            continue
        if (value.lineno, value.col_offset, value.end_lineno, value.end_col_offset) == span:
            return target.id
    return None


def _passes_name(call: ast.Call, name: str) -> bool:
    """Return whether ``name`` is passed directly as one of ``call``'s arguments."""

    values = [*call.args, *(keyword.value for keyword in call.keywords)]
    return any(isinstance(value, ast.Name) and value.id == name for value in values)


def _primary_call(source: str) -> str | None:
    """Return the outermost TorchLens call of a one-statement canonical: the first in source order."""

    calls = torchlens_calls(source)
    return calls[0] if calls else None


@dataclass(frozen=True)
class Grader:
    """Entry-call grading rules derived from the whole card set.

    Attributes
    ----------
    helpers:
        Calls only ever used as arguments in the canonicals; never an entry call.
    """

    helpers: frozenset[str]

    @classmethod
    def from_cards(cls, cards: Iterable[Any]) -> Grader:
        """Derive the helper set: every canonical's non-primary calls minus all primaries."""

        primaries: set[str] = set()
        inner: set[str] = set()
        for card in cards:
            if card.canonical == "none":
                continue
            primary = _primary_call(card.canonical)
            calls = torchlens_calls(card.canonical)
            if primary:
                primaries.add(primary)
            inner.update(call for call in calls if call != primary)
        return cls(frozenset(inner - primaries))

    def is_helper(self, name: str) -> bool:
        """Return whether ``name`` is a selector or edit constructor, not an entry call."""

        return name in self.helpers or name.startswith(EXTRA_HELPER_PREFIXES)

    def entry_call(self, calls: list[str]) -> str | None:
        """Return the first non-helper call, or ``None``."""

        return next((call for call in calls if not self.is_helper(call)), None)

    def entry_from_source(self, source: str) -> str | None:
        """Return the entry call, following a built spec into the call that consumes it.

        The entry is the first non-helper call in source order. When that call only builds a
        value stored in a name (``edit = tl.when(...)``) and a later TorchLens call takes that
        name as an argument (``tl.trace(model, x, intervene=edit)``), the later call is the
        entry: the value was an argument written on its own line. A spec used any other way
        (``edit.bind(model)``) keeps its own call as the entry.

        Parameters
        ----------
        source:
            Python source text.

        Returns
        -------
        str | None
            The entry call's ``tl.`` name, or ``None`` when there is none.

        Raises
        ------
        SyntaxError
            When ``source`` does not parse.
        """

        sites = [(name, node) for name, node in call_sites(source) if not self.is_helper(name)]
        if not sites:
            return None
        tree = ast.parse(source)
        index = 0
        for _ in range(len(sites)):
            bound = _assigned_name(tree, sites[index][1])
            consumer = next(
                (
                    later
                    for later in range(index + 1, len(sites))
                    if bound is not None and _passes_name(sites[later][1], bound)
                ),
                None,
            )
            if consumer is None:
                break
            index = consumer
        return sites[index][0]

    def grade(self, card: Any, source: str | None) -> dict[str, Any]:
        """Grade one solution against one card.

        Parameters
        ----------
        card:
            The task card.
        source:
            The solution text, or ``None`` when the agent wrote no file.

        Returns
        -------
        dict
            ``card``, ``passed``, ``entry``, ``expected``, ``avoid_hit``, ``names_covered``
            and ``reason``.
        """

        expected = None if card.canonical == "none" else _primary_call(card.canonical)
        row: dict[str, Any] = {"card": card.slug, "expected": expected, "entry": None}
        if source is None:
            return {**row, "passed": False, "avoid_hit": [], "names_covered": False,
                    "reason": "no solution.py"}  # fmt: skip
        try:
            calls = torchlens_calls(source)
        except SyntaxError as exc:
            return {**row, "passed": False, "avoid_hit": [], "names_covered": False,
                    "reason": f"syntax error: {exc.msg}"}  # fmt: skip
        entry = self.entry_from_source(source)
        avoid_hit = sorted(set(card.avoid_calls) & set(calls))
        wanted = set() if expected is None else set(torchlens_calls(card.canonical))
        row.update(entry=entry, avoid_hit=avoid_hit, names_covered=wanted <= set(calls))
        if expected is None:
            passed, reason = not avoid_hit, "avoid call used" if avoid_hit else ""
        elif entry != expected and entry not in card.accept:
            passed, reason = False, f"entry {entry} is not {expected}"
        elif card.hard_fail and avoid_hit:
            passed, reason = False, f"hard-fail call used: {', '.join(avoid_hit)}"
        else:
            passed, reason = True, ""
        return {**row, "passed": passed, "reason": reason}


def render_prompt(card: Any, package_dir: Path) -> str:
    """Return the prompt file text for one card."""

    return PROMPT_TEMPLATE.format(
        fixture=FIXTURES[card.fixture], prompt=card.prompt, package=package_dir
    )


def vendor_package(dest: Path) -> Path:
    """Copy the torchlens package as a wheel installs it (no agent notes, no bytecode).

    Parameters
    ----------
    dest:
        Directory to receive ``torchlens/`` (replaced if present).

    Returns
    -------
    Path
        The copied package directory, which plays the installed package.
    """

    package_dir = dest / "torchlens"
    shutil.rmtree(package_dir, ignore_errors=True)
    shutil.copytree(
        REPO_ROOT / "torchlens",
        package_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "AGENTS.md", "CLAUDE.md"),
    )
    return package_dir


def prepare_card(card: Any, task_dir: Path, mode: str, package_dir: Path) -> Path:
    """Write one card's task directory and return it.

    Parameters
    ----------
    card:
        The task card.
    task_dir:
        Directory to create (replaced if present).
    mode:
        ``index`` (an instruction file naming the package's index) or ``bare`` (prompt only).
    package_dir:
        The vendored package (:func:`vendor_package`), shared by every task directory.

    Returns
    -------
    Path
        ``task_dir``.
    """

    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True)
    (task_dir / "prompt.md").write_text(render_prompt(card, package_dir), encoding="utf-8")
    if mode == "index":
        init_block = _load_by_path("_tl_init_block", INIT_BLOCK_PATH)
        block = init_block.render_block(package_dir / "_docs" / "AGENT_INDEX.md", "unknown")
        (task_dir / "AGENTS.md").write_text(block, encoding="utf-8")
    return task_dir


def subject_env(keep: Iterable[str] = (), source: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment a live agent runs with: :data:`SUBJECT_ENV` plus ``keep``.

    Parameters
    ----------
    keep:
        Extra variable names to pass through (for example the agent's credential variable).
    source:
        Environment to copy from (default: this process's).

    Returns
    -------
    dict[str, str]
        The allowed variables that are set in ``source``.
    """

    env = os.environ if source is None else source
    return {name: env[name] for name in (*SUBJECT_ENV, *keep) if name in env}


def _run_one(
    card: Any,
    task_dir: Path,
    agent_cmd: str,
    grader: Grader,
    package_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run the agent command in one prepared directory with ``env``, then grade it."""

    instructions = task_dir / "AGENTS.md"
    command = agent_cmd.format(
        package=shlex.quote(str(package_dir)),
        prompt=shlex.quote(str(task_dir / "prompt.md")),
        instructions=shlex.quote(str(instructions)) if instructions.exists() else "''",
        workdir=shlex.quote(str(task_dir)),
        solution=shlex.quote(str(task_dir / "solution.py")),
    )
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command, shell=True, cwd=task_dir, env=env, timeout=AGENT_TIMEOUT_S, check=False
        )
        agent_rc: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        agent_rc = None
    result = grade_dir(card, task_dir, grader)
    result.update(agent_returncode=agent_rc, seconds=round(time.monotonic() - started, 1))
    return result


def grade_dir(card: Any, task_dir: Path, grader: Grader) -> dict[str, Any]:
    """Grade the ``solution.py`` in one task directory."""

    solution = task_dir / "solution.py"
    source = solution.read_text(encoding="utf-8") if solution.exists() else None
    return {**grader.grade(card, source), "dir": str(task_dir)}


def run_live(
    cards: list[Any],
    out: Path,
    agent_cmd: str,
    *,
    label: str,
    runs: int,
    jobs: int,
    mode: str,
    keep_env: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Prepare and run every card ``runs`` times with ``jobs`` agents at once."""

    grader = Grader.from_cards(load_cards())
    package_dir = vendor_package(out / "site-packages")
    plan = [
        (card, prepare_card(card, out / f"run{run}" / card.slug, mode, package_dir))
        for run in range(runs)
        for card in cards
    ]
    env = subject_env(keep_env)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(
            pool.map(lambda item: _run_one(*item, agent_cmd, grader, package_dir, env), plan)
        )
    for result in results:
        result.update(label=label, mode=mode)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return pass rates per label and mode, with the bar verdict for index mode.

    Parameters
    ----------
    results:
        Graded rows carrying ``label`` and ``mode``.

    Returns
    -------
    dict
        ``{"<label>/<mode>": {"runs", "passed", "rate", "bar", "meets_bar", "failures"}}``.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(f"{row.get('label', '?')}/{row.get('mode', '?')}", []).append(row)
    summary: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        label, mode = key.split("/", 1)
        passed = sum(1 for row in rows if row["passed"])
        bar = BARS.get(label) if mode == "index" else None
        rate = passed / len(rows)
        summary[key] = {
            "runs": len(rows),
            "passed": passed,
            "rate": round(rate, 3),
            "bar": bar,
            "meets_bar": None if bar is None else rate >= bar,
            "failures": sorted(
                f"{row['card']}: {row['reason']}" for row in rows if not row["passed"]
            ),
        }
    return summary


def selfcheck() -> list[str]:
    """Grade every card's canonical as a solution; return the problems (empty when sound).

    Returns
    -------
    list[str]
        One line per card whose own canonical fails its grader.
    """

    cards = load_cards()
    grader = Grader.from_cards(cards)
    problems = []
    for card in cards:
        source = f"import torchlens as tl\n{card.canonical}\n" if card.canonical != "none" else ""
        result = grader.grade(card, source)
        if not result["passed"]:
            problems.append(f"{card.slug}: {result['reason']}")
        if card.fixture not in FIXTURES:
            problems.append(f"{card.slug}: unknown fixture {card.fixture}")
        if not card.prompt:
            problems.append(f"{card.slug}: no prompt")
    return problems


def _select(cards: tuple[Any, ...], ids: str | None) -> list[Any]:
    """Return the cards named in a comma list (all when ``None``)."""

    if not ids:
        return list(cards)
    wanted = ids.split(",")
    by_id = {card.slug: card for card in cards}
    missing = [card_id for card_id in wanted if card_id not in by_id]
    if missing:
        raise SystemExit(f"unknown card id(s): {', '.join(missing)}")
    return [by_id[card_id] for card_id in wanted]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck")
    for name in ("prepare", "live"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--mode", choices=("index", "bare"), default="index")
        sp.add_argument("--cards", default=None, help="comma-separated card ids (default: all)")
        if name == "live":
            sp.add_argument("--agent-cmd", required=True)
            sp.add_argument("--label", required=True, help="subject tier, e.g. small or mid")
            sp.add_argument("--runs", type=int, default=1)
            sp.add_argument("--jobs", type=int, default=1)
            sp.add_argument(
                "--keep-env",
                action="append",
                default=[],
                help="environment variable the agent also inherits (repeatable)",
            )
    grade_parser = sub.add_parser("grade")
    grade_parser.add_argument("--dir", type=Path, required=True)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "selfcheck":
        problems = selfcheck()
        print("\n".join(problems) or "selfcheck ok")
        return 1 if problems else 0
    cards = load_cards()
    if args.command == "prepare":
        package_dir = vendor_package(args.out / "site-packages")
        for card in _select(cards, args.cards):
            prepare_card(card, args.out / card.slug, args.mode, package_dir)
        print(f"prepared under {args.out}")
        return 0
    if args.command == "live":
        selected = _select(cards, args.cards)
        results = run_live(
            selected, args.out, args.agent_cmd,
            label=args.label, runs=args.runs, jobs=args.jobs, mode=args.mode,
            keep_env=args.keep_env,
        )  # fmt: skip
        (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(summarize(results), indent=2))
        return 0
    if args.command == "grade":
        grader = Grader.from_cards(cards)
        by_id = {card.slug: card for card in cards}
        rows = [
            grade_dir(by_id[path.parent.name], path.parent, grader)
            for path in sorted(args.dir.rglob("prompt.md"))
            if path.parent.name in by_id
        ]
        print(json.dumps(rows, indent=2))
        return 0
    rows = []
    for directory in args.dirs:
        rows.extend(json.loads((directory / "results.json").read_text(encoding="utf-8")))
    print(json.dumps(summarize(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
