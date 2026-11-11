"""Field differential: compare every public field of a trace between two commits.

The tool dumps every public field of every ``Op``, ``Layer``, ``Module``,
module call, ``Param``, ``Buffer``, grad function and the ``Trace`` itself,
plus ``repr`` and ``to_pandas()``, in a canonical form, for a fixed case list
(``field_diff_lib/cases.py``). In two-commit mode it runs the same dump code
against two source trees, each in its own fresh process, and prints every
difference outside the fixed mask (``field_diff_lib/mask.py``). It also runs
the behavioural sequences (``field_diff_lib/sequences.py``) on both sides and
compares their outcomes.

CLI:
    python tools/field_differential.py dump --out DIR [--cases a,b] [--no-mask]
    python tools/field_differential.py compare --base-tree PATH --work DIR
    python tools/field_differential.py compare --base REV --work DIR
    python tools/field_differential.py selfcheck --work DIR
    python tools/field_differential.py compare --base-tree . --plant value --work DIR
    python tools/field_differential.py cases

``compare`` exits 0 when there are no differences and 1 when there are.
``selfcheck`` compares the current tree against itself (two processes).
``--plant`` plants one fault on the head side; the comparison must then
report it, so with ``--plant`` the exit status is 0 when the fault is
reported and 1 when it is missed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

#: Environment pinned before torch loads, so content hashes do not follow the host's CPU.
PINNED_ENV = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "COMPATIBLE"}


def _pin_torch() -> None:
    """Pin threads and kernel selection: both steer float reduction order."""

    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.mkldnn.enabled = False


def _roots(scratch: str) -> list[tuple[str, str]]:
    """Return the path prefixes replaced in strings, longest first."""

    import torchlens

    tree = str(Path(torchlens.__file__).resolve().parent.parent)
    roots = [(scratch, "<scratch>"), (str(TOOLS_DIR), "<tools>"), (tree, "<repo>")]
    return sorted(roots, key=lambda pair: -len(pair[0]))


def _dump_case(case: Any, built: Any, scratch: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one case and return its canonical dump."""

    from field_diff_lib import faults
    from field_diff_lib.canonical import Canonicalizer, error_record
    from field_diff_lib.mask import GEOMETRY_READ_MODELS, Mask
    from field_diff_lib.surface import snapshot_trace

    import torchlens as tl

    roots = _roots(scratch)
    plain = Canonicalizer(roots, frozenset())
    if isinstance(built, BaseException):
        return {"case_raises": error_record(built, plain.string), "stage": "build"}
    model, model_input = built
    result: dict[str, Any] = {"traces": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            traces = case.run(tl, model, model_input, scratch)
        except Exception as error:  # noqa: BLE001 - a refusing capture is recorded
            return {"case_raises": error_record(error, plain.string), "stage": "capture"}
    result["capture_warnings"] = [
        {"category": w.category.__name__, "message": plain.string(str(w.message))[:300]}
        for w in caught
    ]
    skip: frozenset[str] = frozenset()
    if args.plant and case.name == faults.PLANT_CASE:
        skip = faults.plant(args.plant, next(iter(traces.values())))
    for name, trace in traces.items():
        dump, canon = snapshot_trace(
            trace,
            roots,
            Mask(enabled=not args.no_mask),
            skip,
            read_geometry=case.model in GEOMETRY_READ_MODELS,
        )
        result["traces"][name] = dump
        if case.mutation_probe:
            result.setdefault("mutation_probe", {})[name] = faults.mutation_probe(
                canon.tensor_sites
            )
    return result


def cmd_dump(args: argparse.Namespace) -> int:
    """Dump the selected cases, and the sequences, into ``args.out``."""

    os.environ.update(PINNED_ENV)
    _pin_torch()
    import torch
    from field_diff_lib.cases import build_model, case_list

    import torchlens

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.cases.split(",")) if args.cases else None
    cases = [case for case in case_list() if wanted is None or case.name in wanted]
    built: dict[str, Any] = {}
    for case in cases:
        try:
            built[case.name] = build_model(case.model)
        except Exception as error:  # noqa: BLE001 - a missing optional model is recorded
            built[case.name] = error
    scratch = tempfile.mkdtemp(prefix="field-diff-")
    try:
        if "fields" in args.what:
            for case in cases:
                dump = _dump_case(case, built.pop(case.name), scratch, args)
                (out / f"case-{case.name}.json").write_text(json.dumps(dump))
                print(f"dumped {case.name}", flush=True)
        if "sequences" in args.what:
            from field_diff_lib.sequences import run_sequences

            names = set(args.sequences.split(",")) if args.sequences else None
            for name, outcome in run_sequences(scratch, _roots(scratch), names).items():
                (out / f"sequence-{name}.json").write_text(json.dumps(outcome))
                print(f"ran sequence {name}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    meta = {
        "torchlens": str(Path(torchlens.__file__).resolve()),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "plant": args.plant,
        "mask": not args.no_mask,
    }
    (out / "_meta.json").write_text(json.dumps(meta, indent=1))
    return 0


def _export_tree(rev: str, work: Path) -> Path:
    """Export a revision's ``torchlens`` package into ``work`` (needs Git history)."""

    tree = work / f"tree-{rev[:12]}"
    if not tree.exists():
        tree.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "archive", rev, "torchlens"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    return tree


def _run_side(tree: Path, out: Path, args: argparse.Namespace, plant: str | None) -> None:
    """Run the dump against one source tree in a fresh process."""

    command = [sys.executable, str(Path(__file__).resolve()), "dump", "--out", str(out)]
    command += ["--what", ",".join(args.what)]
    for flag, value in (("--cases", args.cases), ("--sequences", args.sequences)):
        if value:
            command += [flag, value]
    if args.no_mask:
        command.append("--no-mask")
    if plant:
        command += ["--plant", plant]
    env = {**os.environ, **PINNED_ENV, "PYTHONPATH": str(tree), "PYTHONHASHSEED": "0"}
    subprocess.run(command, check=True, env=env, cwd=str(out.parent), stdout=subprocess.DEVNULL)
    meta = json.loads((out / "_meta.json").read_text())
    if not Path(meta["torchlens"]).is_relative_to(tree.resolve()):
        raise SystemExit(
            f"field differential: side {out.name} imported {meta['torchlens']}, not {tree}"
        )


def cmd_compare(args: argparse.Namespace) -> int:
    """Dump base and head in two fresh processes and print the differences."""

    from field_diff_lib.compare import diff, report
    from field_diff_lib.count_only import accept_count_only
    from field_diff_lib.mask import accept_release_moves

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    head_tree = Path(args.head_tree).resolve() if args.head_tree else REPO_ROOT
    if args.base:
        base_tree = _export_tree(args.base, work)
    else:
        base_tree = Path(args.base_tree).resolve() if args.base_tree else head_tree
    for side, tree, plant in (("base", base_tree, None), ("head", head_tree, args.plant)):
        shutil.rmtree(work / side, ignore_errors=True)
        _run_side(tree, work / side, args, plant)
    names = sorted(
        {path.name for side in ("base", "head") for path in (work / side).glob("*-*.json")}
    )
    differences: dict[str, list[tuple[str, Any, Any]]] = {}
    for name in names:
        sides = []
        for side in ("base", "head"):
            path = work / side / name
            sides.append(json.loads(path.read_text()) if path.exists() else "<absent>")
        dump = name.removesuffix(".json")
        if not args.no_mask:
            sides[1] = accept_count_only(sides[0], accept_release_moves(dump, sides[0], sides[1]))
        differences[dump] = diff(sides[0], sides[1])
    total, lines = report(differences, limit=args.limit)
    lines.insert(1, f"base tree {base_tree}; head tree {head_tree}; plant {args.plant}")
    (work / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:400]))
    if args.plant:
        print(f"planted fault {args.plant!r}: {'reported' if total else 'MISSED'}")
        return 0 if total else 1
    return 1 if total else 0


def cmd_cases(args: argparse.Namespace) -> int:
    """Print the case list, the sequences and the mask."""

    from field_diff_lib import mask
    from field_diff_lib.cases import case_list
    from field_diff_lib.faults import FAULTS

    for case in case_list():
        print(f"case {case.name} (model {case.model})")
    for rule in mask.MASK:
        print(f"mask {rule.field} [{rule.rule}]: {rule.reason}")
    for text_rule in mask.TEXT_RULES:
        print(f"text {text_rule.pattern!r}: {text_rule.reason}")
    for name, reason in mask.NOT_READ.items():
        print(f"not read {name}: {reason}")
    for name, names in mask.TYPE_TAG_ONLY.items():
        print(f"type tag only {name} {names}: {mask.TYPE_TAG_ONLY_REASON}")
    print(f"release order {mask.RELEASE_ORDER}: {mask.RELEASE_ORDER_REASON}")
    for name, bucket in mask.COUNT_ONLY.items():
        print(f"count only {name} {bucket!r}: {mask.COUNT_ONLY_REASON}")
    for name, steps in mask.RELEASE_LATER_CLASSES.items():
        print(f"release later {name} in steps {sorted(steps)}: {mask.RELEASE_LATER_REASON}")
    for name, what in FAULTS.items():
        print(f"fault {name}: {what}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def _common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--cases", help="comma-separated case names (default: all)")
        sub.add_argument("--sequences", help="comma-separated sequence names (default: all)")
        sub.add_argument(
            "--what",
            type=lambda text: text.split(","),
            default=["fields", "sequences"],
            help="fields, sequences or both (comma-separated)",
        )
        sub.add_argument("--no-mask", action="store_true", help="turn the mask off")
        sub.add_argument("--plant", help="plant one fault on the head side")

    dump = commands.add_parser("dump", help="dump the tree on sys.path")
    dump.add_argument("--out", required=True)
    _common(dump)
    dump.set_defaults(run=cmd_dump)
    for name in ("compare", "selfcheck"):
        compare = commands.add_parser(name, help="compare two trees")
        compare.add_argument("--work", required=True, help="scratch folder for dumps and report")
        compare.add_argument("--base", help="base revision (needs Git history)")
        compare.add_argument("--base-tree", help="base source tree (default: the head tree)")
        compare.add_argument("--head-tree", help="head source tree (default: this checkout)")
        compare.add_argument("--limit", type=int, default=12, help="differences shown per case")
        _common(compare)
        compare.set_defaults(run=cmd_compare)
    commands.add_parser("cases", help="list cases, mask and faults").set_defaults(run=cmd_cases)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""

    args = _parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
