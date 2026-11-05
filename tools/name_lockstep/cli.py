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

from name_lockstep import baseguard, check, codemod, gate_input
from name_lockstep.baseguard import BaseError
from name_lockstep.diff import diff_inventories, proposals_to_map
from name_lockstep.inventory import build_inventory, load_inventory

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
#: Exit status for a base or ref the diff must not run against (never 0 or 1).
BAD_BASE = 2


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
        The inventory, with ``ref`` and the resolved ``commit``.

    Raises
    ------
    BaseError
        When the repository does not have the ref.
    """

    commit = baseguard.resolve(repo, ref, "ref")
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
    inventory["commit"] = commit
    return inventory


def _working_inventory(repo: Path) -> dict[str, Any]:
    """Build the working tree's inventory, recording the checked-out commit."""

    inventory = build_inventory()
    inventory["commit"] = baseguard.rev(repo, "HEAD")
    return inventory


def _rows(source: str, repo: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Load rows from an inventory file, or build them at a git ref.

    Parameters
    ----------
    source
        A JSON path, ``ref:<rev>``, or ``.`` for the importable package.
    repo
        Repository for ``ref:`` sources.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], str | None]
        Rows keyed on id, and the commit they describe (None when a JSON inventory
        records none).
    """

    if source == ".":
        inventory = _working_inventory(repo)
    elif source.startswith("ref:"):
        inventory = _inventory_at_ref(source[4:], repo)
    else:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        commit = data.get("commit") if isinstance(data, dict) else None
        return load_inventory(path), commit
    return {row["id"]: row for row in inventory["rows"]}, inventory.get("commit")


def _guard_base(args: argparse.Namespace, base_commit: str | None, head_commit: str | None) -> None:
    """Refuse a base the diff cannot be trusted against; log the base summary otherwise.

    The summary goes to stderr: ``check`` output is parsed line by line as findings.

    Parameters
    ----------
    args
        Parsed arguments (``base``, ``head``, ``target``, ``repo``).
    base_commit, head_commit
        Commits the two inventories describe, when known.

    Raises
    ------
    BaseError
        A missing commit (unless ``--target none``), a base outside the head's history, or
        a base no merge target reaches.
    """

    for label, source, commit in (
        ("base", args.base, base_commit),
        ("head", args.head, head_commit),
    ):
        if commit is None and args.target != baseguard.NONE:
            raise BaseError(
                f"{label} inventory {source} records no commit, so the base cannot be"
                " verified: rebuild it with this tool, or pass --target none"
            )
    if base_commit is None or head_commit is None:
        print("base: unverified (an inventory records no commit; --target none)", file=sys.stderr)
        return
    print(baseguard.verify(args.repo, base_commit, head_commit, args.target), file=sys.stderr)


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

    (base, base_commit), (head, head_commit) = (
        _rows(args.base, args.repo),
        _rows(args.head, args.repo),
    )
    _guard_base(args, base_commit, head_commit)
    declared = proposals_to_map(load_inventory(Path(args.renames))) if args.renames else {}
    return base, head, diff_inventories(base, head, declared)


def cmd_inventory(args: argparse.Namespace) -> int:
    """Write an inventory of the package (working tree, or ``--ref``)."""

    inventory = (
        _inventory_at_ref(args.ref, args.repo) if args.ref else _working_inventory(args.repo)
    )
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

    rows = list(load_inventory(Path(args.proposals)).values())
    skipped = codemod.skipped_proposals(rows)
    edits = codemod.plan(rows, args.repo)
    for edit in edits:
        print(
            f"{edit.certainty:9} {edit.path}:{edit.line}:{edit.start + 1} "
            f"{edit.old} -> {edit.new} [{edit.row_id}]"
        )
    for row in skipped:
        print(f"{'skipped':9} {row['id']}: {row['proposed']!r} {row['reason']}")
    definite = sum(edit.certainty == codemod.DEFINITE for edit in edits)
    print(
        f"codemod: {len(edits)} edits ({definite} definite, {len(edits) - definite} ambiguous); "
        f"skipped rows: {len(skipped)}"
    )
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
        command.add_argument(
            "--target",
            help="ref the base must be reachable from (default: main or a *-train branch);"
            " 'none' skips that check",
        )
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
    try:
        return int(args.func(args))
    except BaseError as error:
        print(f"name_lockstep: {error}", file=sys.stderr)
        return BAD_BASE
