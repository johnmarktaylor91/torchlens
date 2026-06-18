"""Plan or run menagerie rendering and validation across conda environments.

The environment map below is intentionally non-exhaustive. Extend ``ENV_SPECS``
when new catalog zoos require mutually incompatible dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.catalog import CatalogRow, load_rows
from menagerie.generate_menagerie import log_event


DEFAULT_RENDER_OUT_DIR = Path("/tmp/torchlens_menagerie_gallery")
DEFAULT_VALIDATE_OUT_DIR = Path("/tmp/torchlens_menagerie_validation")
ENV_PREFIX = "tlmenagerie_"
ENV_SPECS: dict[str, dict[str, Any]] = {
    "mmlab": {
        "pip_packages": [
            "openmim",
            "mmcv",
            "mmengine",
            "mmdet",
            "mmsegmentation",
            "mmpretrain",
            "mmpose",
        ],
        "zoo_patterns": [r"mm", r"openmmlab", r"mmdet", r"mmseg", r"mmpose", r"mmpretrain"],
        "catch_all": False,
    },
    "recbole": {
        "pip_packages": ["recbole"],
        "zoo_patterns": [r"recbole"],
        "catch_all": False,
    },
    "paddle": {
        "pip_packages": ["paddlepaddle", "paddleseg", "paddleclas", "paddledet"],
        "zoo_patterns": [r"paddle"],
        "catch_all": False,
    },
    "detectron2": {
        "pip_packages": ["detectron2"],
        "zoo_patterns": [r"detectron2"],
        "catch_all": False,
    },
    "fla": {
        "pip_packages": ["flash-linear-attention"],
        "zoo_patterns": [r"flash-linear-attention", r"fla"],
        "catch_all": False,
    },
    "core": {
        "pip_packages": [],
        "zoo_patterns": [
            r"classics-pytorch",
            r"timm",
            r"torchvision",
            r"huggingface",
            r"transformers",
            r"diffusers",
            r"segmentation-models-pytorch",
            r"segmentation_models_pytorch",
            r"smp",
            r"pyg",
            r"torch-geometric",
            r"ultralytics",
            r"monai",
            r"open_clip",
            r"open-clip",
        ],
        "catch_all": True,
    },
}


@dataclass(frozen=True)
class PlannedCommand:
    """A command planned for one environment.

    Parameters
    ----------
    env_key:
        Logical environment key from ``ENV_SPECS``.
    kind:
        Command kind.
    command:
        Command arguments.
    """

    env_key: str
    kind: str
    command: tuple[str, ...]


def _shell_join(command: Sequence[str]) -> str:
    """Return a shell-escaped command string.

    Parameters
    ----------
    command:
        Command arguments.

    Returns
    -------
    str
        Shell-escaped command.
    """

    return shlex.join(str(part) for part in command)


def _env_name(env_key: str) -> str:
    """Return the concrete conda environment name.

    Parameters
    ----------
    env_key:
        Logical environment key.

    Returns
    -------
    str
        Conda environment name.
    """

    return f"{ENV_PREFIX}{env_key}"


def _module_for_task(task: str) -> str:
    """Return the menagerie module for a task.

    Parameters
    ----------
    task:
        ``"render"`` or ``"validate"``.

    Returns
    -------
    str
        Module name under ``menagerie``.
    """

    if task == "render":
        return "generate_menagerie"
    return "validate_menagerie"


def _default_out_dir(task: str) -> Path:
    """Return the default shared output directory for a task.

    Parameters
    ----------
    task:
        ``"render"`` or ``"validate"``.

    Returns
    -------
    pathlib.Path
        Default output directory.
    """

    if task == "render":
        return DEFAULT_RENDER_OUT_DIR
    return DEFAULT_VALIDATE_OUT_DIR


def _pattern_matches(pattern: str, value: str) -> bool:
    """Return whether an environment pattern matches a zoo string.

    Parameters
    ----------
    pattern:
        Regex or substring pattern.
    value:
        Zoo string.

    Returns
    -------
    bool
        Whether the pattern matches.
    """

    try:
        return re.search(pattern, value, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in value.lower()


def _claimed_by_non_core(row: CatalogRow) -> bool:
    """Return whether a row is claimed by a non-core environment.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    bool
        Whether a specialized environment claims the row.
    """

    for env_key, spec in ENV_SPECS.items():
        if env_key == "core":
            continue
        if any(_pattern_matches(pattern, row.zoo) for pattern in spec["zoo_patterns"]):
            return True
    return False


def _row_counts_for_env(env_key: str, rows: Sequence[CatalogRow]) -> int:
    """Count rows selected by one environment specification.

    Parameters
    ----------
    env_key:
        Logical environment key.
    rows:
        Catalog rows.

    Returns
    -------
    int
        Matching row count.
    """

    spec = ENV_SPECS[env_key]
    count = 0
    for row in rows:
        matches_named = any(_pattern_matches(pattern, row.zoo) for pattern in spec["zoo_patterns"])
        if matches_named or (spec.get("catch_all") and not _claimed_by_non_core(row)):
            count += 1
    return count


def _ordered_envs(requested: Sequence[str] | None) -> list[str]:
    """Return execution order with specialized environments before core.

    Parameters
    ----------
    requested:
        Optional requested environment keys.

    Returns
    -------
    list[str]
        Ordered environment keys.
    """

    envs = list(requested) if requested else list(ENV_SPECS)
    unknown = [env for env in envs if env not in ENV_SPECS]
    if unknown:
        raise ValueError(f"unknown env(s): {', '.join(unknown)}")
    return sorted(envs, key=lambda env: env == "core")


def _run_command_for_pattern(
    env_key: str,
    task: str,
    out_dir: Path,
    pattern: str | None,
    extra_args: Sequence[str],
) -> PlannedCommand:
    """Build one conda-run task command.

    Parameters
    ----------
    env_key:
        Logical environment key.
    task:
        ``"render"`` or ``"validate"``.
    out_dir:
        Shared output directory.
    pattern:
        Zoo substring filter, or ``None`` for the core catch-all pass.
    extra_args:
        Extra arguments forwarded to the task module.

    Returns
    -------
    PlannedCommand
        Planned run command.
    """

    module_name = _module_for_task(task)
    command = [
        "conda",
        "run",
        "-n",
        _env_name(env_key),
        "python",
        "-m",
        f"menagerie.{module_name}",
        "--no-install-deps",
        "--out-dir",
        str(out_dir),
    ]
    if pattern is not None:
        command.extend(["--zoo", pattern])
    command.extend(extra_args)
    return PlannedCommand(env_key, "run", tuple(command))


def build_plan(args: argparse.Namespace) -> list[PlannedCommand]:
    """Build all commands needed for the selected environments.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    list[PlannedCommand]
        Planned commands.
    """

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (args.out_dir or _default_out_dir(args.task)).resolve()
    plan: list[PlannedCommand] = []
    for env_key in _ordered_envs(args.envs):
        env_name = _env_name(env_key)
        packages = ENV_SPECS[env_key]["pip_packages"]
        plan.append(
            PlannedCommand(
                env_key,
                "create",
                (
                    "conda",
                    "create",
                    "-y",
                    "-n",
                    env_name,
                    f"python={args.python_version}",
                ),
            )
        )
        plan.append(
            PlannedCommand(
                env_key,
                "install",
                (
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    str(repo_root),
                    *packages,
                ),
            )
        )
        for pattern in ENV_SPECS[env_key]["zoo_patterns"]:
            plan.append(
                _run_command_for_pattern(env_key, args.task, out_dir, pattern, args.extra_args)
            )
        if ENV_SPECS[env_key].get("catch_all"):
            plan.append(
                _run_command_for_pattern(env_key, args.task, out_dir, None, args.extra_args)
            )
    return plan


def _conda_envs() -> set[str]:
    """Return existing conda environment names.

    Returns
    -------
    set[str]
        Existing environment names.
    """

    completed = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return set()
    payload = json.loads(completed.stdout)
    envs = payload.get("envs", [])
    return {Path(env).name for env in envs}


def _run_logged(command: Sequence[str], log_path: Path) -> int:
    """Run a command while streaming output to console and a log file.

    Parameters
    ----------
    command:
        Command arguments.
    log_path:
        File to append combined stdout and stderr.

    Returns
    -------
    int
        Process return code.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {_shell_join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
        return_code = process.wait()
        handle.write(f"[exit {return_code}]\n")
    return return_code


def _print_manual_recipe(plan: Sequence[PlannedCommand]) -> None:
    """Print a manual recipe when conda is unavailable.

    Parameters
    ----------
    plan:
        Planned commands.
    """

    print("conda was not found on PATH; review/run the per-environment commands manually.")
    for item in plan:
        print(_shell_join(item.command))


def print_plan(plan: Sequence[PlannedCommand], rows: Sequence[CatalogRow]) -> None:
    """Print planned commands without executing them.

    Parameters
    ----------
    plan:
        Planned commands.
    rows:
        Catalog rows used for row-count context.
    """

    printed_envs: set[str] = set()
    for item in plan:
        if item.env_key not in printed_envs:
            count = _row_counts_for_env(item.env_key, rows)
            print(f"# {item.env_key}: approximately {count} catalog rows")
            printed_envs.add(item.env_key)
        print(_shell_join(item.command))


def execute_plan(plan: Sequence[PlannedCommand], out_dir: Path) -> int:
    """Execute a command plan, continuing across per-env failures.

    Parameters
    ----------
    plan:
        Planned commands.
    out_dir:
        Shared output directory.

    Returns
    -------
    int
        Zero if all selected environments succeeded, otherwise one.
    """

    existing_envs = _conda_envs()
    failed_envs: set[str] = set()
    ok = True
    for item in plan:
        if item.env_key in failed_envs:
            continue
        env_name = _env_name(item.env_key)
        if item.kind == "create" and env_name in existing_envs:
            log_event("env_reuse", env=item.env_key, conda_env=env_name)
            continue
        log_path = out_dir / "logs" / f"{item.env_key}.log"
        log_event("command_start", env=item.env_key, kind=item.kind, command=list(item.command))
        return_code = _run_logged(item.command, log_path)
        if return_code != 0:
            ok = False
            log_event(
                "command_failed",
                env=item.env_key,
                kind=item.kind,
                return_code=return_code,
                log=str(log_path),
            )
            if item.kind in {"create", "install"}:
                failed_envs.add(item.env_key)
            continue
        if item.kind == "create":
            existing_envs.add(env_name)
        log_event("command_done", env=item.env_key, kind=item.kind)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the cross-environment runner CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("render", "validate"), required=True)
    parser.add_argument("--envs", nargs="+", choices=tuple(ENV_SPECS), help="env keys to run")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="create/install/run commands")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are forwarded to the render/validate module",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cross-environment CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    try:
        plan = build_plan(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    rows = load_rows()
    if not args.execute:
        if shutil.which("conda") is None:
            print("conda was not found on PATH; this is still a dry-run command plan.")
        print_plan(plan, rows)
        return 0
    out_dir = (args.out_dir or _default_out_dir(args.task)).resolve()
    if shutil.which("conda") is None:
        _print_manual_recipe(plan)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    return execute_plan(plan, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
