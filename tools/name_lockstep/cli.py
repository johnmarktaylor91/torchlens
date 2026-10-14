"""Command line for the name lockstep tooling (``python tools/name_lockstep --help``)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from name_lockstep import check, codemod, gate_input
from name_lockstep.diff import diff_inventories, proposals_to_map
from name_lockstep.inventory import build_inventory, load_inventory

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]


def _inventory_at_ref(ref: str, repo: Path) -> dict[str, Any]:
    """Build an inventory of the package as it is at a git ref.

    The ref's tree is exported to a temporary directory and walked in a fresh
    interpreter, so the working tree and this process stay untouched.

    Parameters
    ----------
    ref
        Any git revision.
    repo
        Repository to read the ref from.

    Returns
    -------
    dict[str, Any]
        The inventory.
    """

    with tempfile.TemporaryDirectory(prefix="name-lockstep-") as tmp:
        archive = Path(tmp) / "tree.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                ref,
                "torchlens",
            ],
            check=True,
        )
        with tarfile.open(archive) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:  # Python without extraction filters; the archive is our own tree
                tar.extractall(tmp)  # noqa: S202
        code = (
            "import json, sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
            "from name_lockstep.inventory import build_inventory; "
            "print(json.dumps(build_inventory()))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code, tmp, str(TOOLS)],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp,
        )
    inventory = json.loads(done.stdout)
    inventory["ref"] = ref
    return inventory


def _rows(source: str, repo: Path) -> dict[str, dict[str, Any]]:
    """Load rows from an inventory file, or build them at a git ref.

    Parameters
    ----------
    source
        A JSON path, ``ref:<rev>``, or ``.`` for the importable package.
    repo
        Repository for ``ref:`` sources.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rows keyed on id.
    """

    if source == ".":
        inventory = build_inventory()
    elif source.startswith("ref:"):
        inventory = _inventory_at_ref(source[4:], repo)
    else:
        return load_inventory(Path(source))
    return {row["id"]: row for row in inventory["rows"]}


def _diff(args: argparse.Namespace):
    """Diff the ``--base`` and ``--head`` sources named on the command line.

    Parameters
    ----------
    args
        Parsed arguments.

    Returns
    -------
    tuple
        Base rows, head rows and the diff.
    """

    base, head = _rows(args.base, args.repo), _rows(args.head, args.repo)
    declared = proposals_to_map(load_inventory(Path(args.renames))) if args.renames else {}
    return base, head, diff_inventories(base, head, declared)


def cmd_inventory(args: argparse.Namespace) -> int:
    """Write an inventory of the package (working tree, or ``--ref``)."""

    inventory = _inventory_at_ref(args.ref, args.repo) if args.ref else build_inventory()
    Path(args.out).write_text(json.dumps(inventory, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(inventory['rows'])} rows)")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Print new, removed and renamed public names."""

    _, _, surface_diff = _diff(args)
    print(json.dumps(surface_diff.to_json(), indent=1))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check the lockstep rule; exit 1 and list the missing places on any finding."""

    _, head, surface_diff = _diff(args)
    corpus = check.Corpus.read(args.repo, [Path(g) for g in args.glossary])
    findings = check.check_lockstep(surface_diff, corpus, head)
    counts = surface_diff.to_json()["counts"]
    for finding in findings:
        print(finding.render())
    print(f"lockstep: {counts} names changed, {len(findings)} findings")
    return 1 if findings else 0


def cmd_codemod(args: argparse.Namespace) -> int:
    """Plan (default) or apply proposed renames."""

    rows = load_inventory(Path(args.proposals)).values()
    edits = codemod.plan(rows, args.repo)
    for edit in edits:
        print(
            f"{edit.certainty:9} {edit.path}:{edit.line}:{edit.start + 1} {edit.old} -> {edit.new}"
        )
    definite = sum(edit.certainty == codemod.DEFINITE for edit in edits)
    print(f"codemod: {len(edits)} edits ({definite} definite, {len(edits) - definite} ambiguous)")
    if not args.apply:
        print("dry run: nothing written (pass --apply to write)")
        return 0
    written = codemod.apply(edits, args.repo, include_ambiguous=args.include_ambiguous)
    print(f"wrote {written} edits")
    return 0


def cmd_gate_input(args: argparse.Namespace) -> int:
    """Write the legibility-check input for every new or renamed name."""

    _, head, surface_diff = _diff(args)
    corpus = check.Corpus.read(args.repo, [Path(g) for g in args.glossary], snippets=True)
    items = gate_input.gate_items(surface_diff, head, corpus)
    Path(args.out).write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(items)} names)")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """

    parser = argparse.ArgumentParser(prog="name_lockstep", description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO, help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="write a surface inventory")
    inventory.add_argument("--out", required=True)
    inventory.add_argument("--ref", help="git ref to inventory instead of the working tree")
    inventory.set_defaults(func=cmd_inventory)
    for name, func, help_text in (
        ("diff", cmd_diff, "list new, removed and renamed names"),
        ("check", cmd_check, "check the lockstep rule over a diff"),
        ("gate-input", cmd_gate_input, "write legibility-check input for a diff"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--base", required=True, help="inventory JSON, ref:<rev> or .")
        command.add_argument("--head", default=".", help="inventory JSON, ref:<rev> or .")
        command.add_argument("--renames", help="rows with id and proposed (declared renames)")
        command.add_argument("--glossary", action="append", default=list(check.DEFAULT_GLOSSARY))
        if name == "gate-input":
            command.add_argument("--out", required=True)
        command.set_defaults(func=func)
    mod = sub.add_parser("codemod", help="apply proposed renames (dry run by default)")
    mod.add_argument("--proposals", required=True, help="rows with id, kind, name, proposed")
    mod.add_argument("--apply", action="store_true", help="write the definite edits")
    mod.add_argument("--include-ambiguous", action="store_true", help="also write ambiguous edits")
    mod.set_defaults(func=cmd_codemod)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status.
    """

    args = _parser().parse_args(argv)
    return int(args.func(args))
