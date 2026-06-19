"""Plan or run menagerie rendering and validation across conda environments.

Environment recipes live in ``menagerie/data/env_specs.json`` so hard-won
install recipes can be captured once and reused without code changes. The
runner is dry-run by default and only creates or mutates environments with
``--execute``.
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
from typing import Any, Sequence

from menagerie.catalog import CatalogRow, load_rows
from menagerie.generate_menagerie import log_event


DEFAULT_RENDER_OUT_DIR = Path("/tmp/torchlens_menagerie_gallery")
DEFAULT_VALIDATE_OUT_DIR = Path("/tmp/torchlens_menagerie_validation")
ENV_PREFIX = "tlmenagerie_"
ENV_SPECS_PATH = Path(__file__).resolve().parent / "data" / "env_specs.json"
DEFAULT_CONDA_PYTHON = "3.11"
DEFAULT_MIN_FREE_GB = 20.0
ENV_LOG_DIRNAME = "env_logs"

DEFAULT_ENV_SPECS: dict[str, dict[str, Any]] = {
    "core": {
        "name": "core",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": [],
        "extra_index_url": "https://download.pytorch.org/whl/cu121",
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
        "post_install_check": "import torchlens, torch",
        "status": "untested",
        "notes": "Assumes the base PyTorch stack is provided by the editable TorchLens install.",
    },
    "mmlab": {
        "name": "mmlab",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": [
            "openmim",
            "mim:mmengine",
            "mim:mmcv",
            "mim:mmdet",
            "mim:mmsegmentation",
            "mim:mmpretrain",
            "mim:mmpose",
        ],
        "extra_index_url": "https://download.pytorch.org/whl/cu121",
        "zoo_patterns": [r"mm", r"openmmlab", r"mmdet", r"mmseg", r"mmpose", r"mmpretrain"],
        "post_install_check": "import torchlens, mmengine, mmcv, mmdet, mmseg, mmpretrain, mmpose",
        "status": "untested",
        "notes": "Install openmim with pip, then install mim:* packages via `mim install`; mmcv pins the compatible torch stack.",
    },
    "detectron2": {
        "name": "detectron2",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": ["git+https://github.com/facebookresearch/detectron2.git"],
        "extra_index_url": "https://download.pytorch.org/whl/cu121",
        "zoo_patterns": [r"detectron2"],
        "post_install_check": "import torchlens, detectron2",
        "status": "untested",
        "notes": "Built from the upstream git repository against the environment torch build.",
    },
    "recbole": {
        "name": "recbole",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": ["recbole"],
        "extra_index_url": "https://download.pytorch.org/whl/cu121",
        "zoo_patterns": [r"recbole"],
        "post_install_check": "import torchlens, recbole",
        "status": "untested",
        "notes": "RecBole recommender models.",
    },
    "paddle": {
        "name": "paddle",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": ["paddlepaddle-gpu", "paddleseg", "paddleclas"],
        "extra_index_url": None,
        "zoo_patterns": [r"paddle"],
        "post_install_check": "import torchlens, paddle, paddleseg, paddleclas",
        "status": "untested",
        "notes": "GPU Paddle stack for PaddleSeg and PaddleClas zoos.",
    },
    "fla": {
        "name": "fla",
        "conda_python": DEFAULT_CONDA_PYTHON,
        "pip_packages": ["flash-linear-attention", "triton"],
        "extra_index_url": "https://download.pytorch.org/whl/cu121",
        "zoo_patterns": [r"flash-linear-attention", r"fla"],
        "post_install_check": "import torchlens, fla, triton",
        "status": "untested",
        "notes": "GPU recipe; requires CUDA and Triton support.",
    },
}


@dataclass(frozen=True)
class EnvRecipe:
    """A normalized environment recipe.

    Parameters
    ----------
    key:
        Logical recipe key from the JSON object.
    name:
        Human-readable recipe name.
    conda_python:
        Python version requested at environment creation.
    pip_packages:
        Ordered package recipe. Plain strings install with pip; ``mim:pkg`` installs with MIM.
    extra_index_url:
        Optional pip extra index URL.
    zoo_patterns:
        Regex or substring filters matched against catalog ``zoo`` values.
    post_install_check:
        Python code that must import successfully for an environment to be reusable.
    status:
        Recipe status stored in JSON.
    notes:
        Human-readable notes.
    catch_all:
        Whether unmatched rows should run in this environment.
    resolved_packages:
        Captured ``pip freeze`` output from a working environment.
    torch_pin:
        Optional torch/torchvision pins installed FIRST via ``--index-url`` so the
        editable TorchLens install does not pull an incompatible default torch
        (e.g. ``("torch==2.4.1", "torchvision==0.19.1")``). TorchLens needs
        ``torch>=2.4``; many frameworks only ship wheels for a specific torch+CUDA.
    torch_index_url:
        Index URL used with ``--index-url`` for ``torch_pin`` (forces the right
        CUDA wheels, unlike ``extra_index_url`` which only augments PyPI).
    """

    key: str
    name: str
    conda_python: str
    pip_packages: tuple[str, ...]
    extra_index_url: str | None
    zoo_patterns: tuple[str, ...]
    post_install_check: str
    status: str
    notes: str
    catch_all: bool
    resolved_packages: tuple[str, ...]
    torch_pin: tuple[str, ...] = ()
    torch_index_url: str | None = None
    post_install_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedCommand:
    """A command planned for one environment.

    Parameters
    ----------
    env_key:
        Logical environment key from ``env_specs.json``.
    kind:
        Command kind.
    command:
        Command arguments.
    """

    env_key: str
    kind: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class EnvPlan:
    """The sequential workflow for one environment.

    Parameters
    ----------
    recipe:
        Environment recipe.
    create:
        Environment creation command.
    install:
        Installation commands.
    checks:
        Post-install verification commands.
    runs:
        Render or validation commands.
    cleanup:
        Optional cleanup command.
    """

    recipe: EnvRecipe
    create: PlannedCommand
    install: tuple[PlannedCommand, ...]
    checks: tuple[PlannedCommand, ...]
    runs: tuple[PlannedCommand, ...]
    cleanup: PlannedCommand | None


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


def _load_raw_env_specs(path: Path = ENV_SPECS_PATH) -> dict[str, dict[str, Any]]:
    """Load raw environment specs from JSON or built-in defaults.

    Parameters
    ----------
    path:
        Environment spec JSON path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Raw environment spec mapping.
    """

    if not path.exists():
        return json.loads(json.dumps(DEFAULT_ENV_SPECS))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by environment name")
    return payload


def _normalize_recipe(key: str, payload: dict[str, Any]) -> EnvRecipe:
    """Normalize a JSON environment recipe.

    Parameters
    ----------
    key:
        Environment key.
    payload:
        Raw JSON payload.

    Returns
    -------
    EnvRecipe
        Normalized recipe.
    """

    return EnvRecipe(
        key=key,
        name=str(payload.get("name", key)),
        conda_python=str(payload.get("conda_python", DEFAULT_CONDA_PYTHON)),
        pip_packages=tuple(str(package) for package in payload.get("pip_packages", [])),
        extra_index_url=payload.get("extra_index_url"),
        zoo_patterns=tuple(str(pattern) for pattern in payload.get("zoo_patterns", [])),
        post_install_check=str(payload.get("post_install_check", "import torchlens")),
        status=str(payload.get("status", "untested")),
        notes=str(payload.get("notes", "")),
        catch_all=bool(payload.get("catch_all", False)),
        resolved_packages=tuple(str(package) for package in payload.get("resolved_packages", [])),
        torch_pin=tuple(str(package) for package in payload.get("torch_pin", [])),
        torch_index_url=payload.get("torch_index_url"),
        post_install_commands=tuple(
            str(command) for command in payload.get("post_install_commands", [])
        ),
    )


def load_env_specs(path: Path = ENV_SPECS_PATH) -> dict[str, EnvRecipe]:
    """Load normalized environment specs.

    Parameters
    ----------
    path:
        Environment spec JSON path.

    Returns
    -------
    dict[str, EnvRecipe]
        Normalized recipe mapping.
    """

    raw_specs = _load_raw_env_specs(path)
    return {key: _normalize_recipe(key, payload) for key, payload in raw_specs.items()}


def _write_env_specs(raw_specs: dict[str, dict[str, Any]], path: Path = ENV_SPECS_PATH) -> None:
    """Write environment specs in stable JSON form.

    Parameters
    ----------
    raw_specs:
        Raw JSON-compatible spec mapping.
    path:
        Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(raw_specs, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _update_env_status(
    env_key: str,
    status: str,
    resolved_packages: Sequence[str] | None = None,
    path: Path = ENV_SPECS_PATH,
) -> None:
    """Update one environment status and optional resolved package capture.

    Parameters
    ----------
    env_key:
        Environment key.
    status:
        New status value.
    resolved_packages:
        Optional frozen package versions.
    path:
        Environment spec JSON path.
    """

    raw_specs = _load_raw_env_specs(path)
    raw_specs.setdefault(env_key, {"name": env_key})["status"] = status
    if resolved_packages is not None:
        raw_specs[env_key]["resolved_packages"] = list(resolved_packages)
    _write_env_specs(raw_specs, path)


def _claimed_by_non_core(row: CatalogRow, specs: dict[str, EnvRecipe]) -> bool:
    """Return whether a row is claimed by a non-core environment.

    Parameters
    ----------
    row:
        Catalog row.
    specs:
        Environment specs.

    Returns
    -------
    bool
        Whether a specialized environment claims the row.
    """

    for env_key, recipe in specs.items():
        if env_key == "core":
            continue
        if any(_pattern_matches(pattern, row.zoo) for pattern in recipe.zoo_patterns):
            return True
    return False


def _row_counts_for_env(
    env_key: str, rows: Sequence[CatalogRow], specs: dict[str, EnvRecipe]
) -> int:
    """Count rows selected by one environment specification.

    Parameters
    ----------
    env_key:
        Logical environment key.
    rows:
        Catalog rows.
    specs:
        Environment specs.

    Returns
    -------
    int
        Matching row count.
    """

    recipe = specs[env_key]
    count = 0
    for row in rows:
        matches_named = any(_pattern_matches(pattern, row.zoo) for pattern in recipe.zoo_patterns)
        if matches_named or (recipe.catch_all and not _claimed_by_non_core(row, specs)):
            count += 1
    return count


def _ordered_envs(requested: Sequence[str] | None, specs: dict[str, EnvRecipe]) -> list[str]:
    """Return execution order with specialized environments before core.

    Parameters
    ----------
    requested:
        Optional requested environment keys.
    specs:
        Environment specs.

    Returns
    -------
    list[str]
        Ordered environment keys.
    """

    envs = list(requested) if requested else list(specs)
    unknown = [env for env in envs if env not in specs]
    if unknown:
        raise ValueError(f"unknown env(s): {', '.join(unknown)}")
    return sorted(envs, key=lambda env: env == "core")


def _command_with_extra_index(command: list[str], extra_index_url: str | None) -> list[str]:
    """Append a pip extra index URL when configured.

    Parameters
    ----------
    command:
        Pip command under construction.
    extra_index_url:
        Optional extra index URL.

    Returns
    -------
    list[str]
        Updated command.
    """

    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    return command


def _install_commands_for_recipe(recipe: EnvRecipe, repo_root: Path) -> tuple[PlannedCommand, ...]:
    """Build install commands for one recipe.

    Parameters
    ----------
    recipe:
        Environment recipe.
    repo_root:
        Repository root for editable TorchLens install.

    Returns
    -------
    tuple[PlannedCommand, ...]
        Install commands.
    """

    env_name = _env_name(recipe.key)
    commands: list[PlannedCommand] = []
    base_pip = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "-m",
        "pip",
        "install",
    ]
    pip_packages = [package for package in recipe.pip_packages if not package.startswith("mim:")]
    mim_packages = [
        package.removeprefix("mim:")
        for package in recipe.pip_packages
        if package.startswith("mim:")
    ]
    # Pin torch FIRST (via --index-url, which forces the chosen CUDA wheels) so the
    # editable TorchLens install below sees a satisfying torch>=2.4 and does not pull
    # an incompatible default build. --extra-index-url is insufficient here because it
    # only augments PyPI and pip may still prefer a newer default-CUDA torch.
    if recipe.torch_pin:
        torch_command = [*base_pip, *recipe.torch_pin]
        if recipe.torch_index_url:
            torch_command.extend(["--index-url", recipe.torch_index_url])
        commands.append(PlannedCommand(recipe.key, "install", tuple(torch_command)))
    editable_command = _command_with_extra_index(
        [*base_pip, "-e", str(repo_root)], recipe.extra_index_url
    )
    commands.append(PlannedCommand(recipe.key, "install", tuple(editable_command)))
    if pip_packages:
        pip_command = _command_with_extra_index(
            [*base_pip, "-U", *pip_packages], recipe.extra_index_url
        )
        commands.append(PlannedCommand(recipe.key, "install", tuple(pip_command)))
    if mim_packages:
        commands.append(
            PlannedCommand(
                recipe.key,
                "install",
                ("conda", "run", "-n", env_name, "mim", "install", *mim_packages),
            )
        )
    # Arbitrary post-install shell steps (run via bash -c in the env), e.g. writing a
    # sitecustomize.py shim. Used by mmlab to spoof mmcv.__version__ so mmdet/mmseg's
    # conservative <2.2.0 cap passes (mmcv 2.2.0 is additive-only over 2.1.0).
    for shell_command in recipe.post_install_commands:
        commands.append(
            PlannedCommand(
                recipe.key,
                "post_install",
                ("conda", "run", "-n", env_name, "bash", "-c", shell_command),
            )
        )
    return tuple(commands)


def _check_command_for_recipe(recipe: EnvRecipe) -> PlannedCommand:
    """Build the post-install import check command.

    Parameters
    ----------
    recipe:
        Environment recipe.

    Returns
    -------
    PlannedCommand
        Post-install check command.
    """

    return PlannedCommand(
        recipe.key,
        "check",
        (
            "conda",
            "run",
            "-n",
            _env_name(recipe.key),
            "python",
            "-c",
            recipe.post_install_check,
        ),
    )


def _run_command_for_pattern(
    recipe: EnvRecipe,
    task: str,
    out_dir: Path,
    pattern: str | None,
    extra_args: Sequence[str],
) -> PlannedCommand:
    """Build one conda-run task command.

    Parameters
    ----------
    recipe:
        Environment recipe.
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
        _env_name(recipe.key),
        "python",
        "-m",
        f"menagerie.{module_name}",
        "--no-install-deps",
        "--device",
        "auto",
        "--out-dir",
        str(out_dir),
    ]
    if task == "render":
        # Rows previously marked skipped:dependency_unavailable in the base env are
        # present in the shared manifest; without this they would be treated as
        # completed and skipped. Retry them so the freshly-installed env can render them.
        command.append("--retry-failed")
    if pattern is not None:
        command.extend(["--zoo", pattern])
    command.extend(extra_args)
    return PlannedCommand(recipe.key, "run", tuple(command))


def build_plan(args: argparse.Namespace, specs: dict[str, EnvRecipe]) -> list[EnvPlan]:
    """Build all per-environment plans.

    Parameters
    ----------
    args:
        Parsed CLI args.
    specs:
        Environment specs.

    Returns
    -------
    list[EnvPlan]
        Planned per-environment workflows.
    """

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (args.out_dir or _default_out_dir(args.task)).resolve()
    plans: list[EnvPlan] = []
    for env_key in _ordered_envs(args.envs, specs):
        recipe = specs[env_key]
        env_name = _env_name(env_key)
        create = PlannedCommand(
            env_key,
            "create",
            ("conda", "create", "-y", "-n", env_name, f"python={recipe.conda_python}"),
        )
        runs: list[PlannedCommand] = []
        if not args.setup_only:
            for pattern in recipe.zoo_patterns:
                runs.append(
                    _run_command_for_pattern(recipe, args.task, out_dir, pattern, args.extra_args)
                )
            if recipe.catch_all:
                runs.append(
                    _run_command_for_pattern(recipe, args.task, out_dir, None, args.extra_args)
                )
        cleanup = None
        if args.cleanup_env:
            cleanup = PlannedCommand(
                env_key, "cleanup", ("conda", "env", "remove", "-y", "-n", env_name)
            )
        plans.append(
            EnvPlan(
                recipe=recipe,
                create=create,
                install=_install_commands_for_recipe(recipe, repo_root),
                checks=(_check_command_for_recipe(recipe),),
                runs=tuple(runs),
                cleanup=cleanup,
            )
        )
    return plans


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


def _has_enough_disk(path: Path, min_free_gb: float) -> tuple[bool, float]:
    """Return whether a filesystem has enough free space.

    Parameters
    ----------
    path:
        Path whose filesystem should be checked.
    min_free_gb:
        Required free GiB.

    Returns
    -------
    tuple[bool, float]
        Whether the check passed and current free GiB.
    """

    check_path = path if path.exists() else path.parent
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent
    free_gb = shutil.disk_usage(check_path).free / (1024**3)
    return free_gb >= min_free_gb, free_gb


def _print_manual_recipe(plans: Sequence[EnvPlan]) -> None:
    """Print a manual recipe when conda is unavailable.

    Parameters
    ----------
    plans:
        Planned environment workflows.
    """

    print("conda was not found on PATH; review/run the per-environment commands manually.")
    for plan in plans:
        for item in _all_planned_commands(plan):
            print(_shell_join(item.command))


def _all_planned_commands(plan: EnvPlan) -> tuple[PlannedCommand, ...]:
    """Return all commands from an environment plan in execution order.

    Parameters
    ----------
    plan:
        Environment plan.

    Returns
    -------
    tuple[PlannedCommand, ...]
        Commands in execution order.
    """

    cleanup = (plan.cleanup,) if plan.cleanup is not None else ()
    return (plan.create, *plan.install, *plan.checks, *plan.runs, *cleanup)


def print_plan(
    plans: Sequence[EnvPlan], rows: Sequence[CatalogRow], specs: dict[str, EnvRecipe]
) -> None:
    """Print planned commands without executing them.

    Parameters
    ----------
    plans:
        Per-environment plans.
    rows:
        Catalog rows used for row-count context.
    specs:
        Environment specs.
    """

    for plan in plans:
        count = _row_counts_for_env(plan.recipe.key, rows, specs)
        print(f"# {plan.recipe.key}: approximately {count} catalog rows")
        if plan.recipe.status == "working":
            print(
                f"# existing env {_env_name(plan.recipe.key)} will be reused if import checks pass"
            )
        for item in _all_planned_commands(plan):
            print(_shell_join(item.command))


def _env_import_check_passes(recipe: EnvRecipe, log_path: Path) -> bool:
    """Return whether an existing environment passes its import check.

    Parameters
    ----------
    recipe:
        Environment recipe.
    log_path:
        Log path for check output.

    Returns
    -------
    bool
        Whether the check succeeds.
    """

    command = _check_command_for_recipe(recipe).command
    return _run_logged(command, log_path) == 0


def _capture_resolved_packages(recipe: EnvRecipe, log_path: Path) -> list[str] | None:
    """Capture installed package versions with ``pip freeze``.

    Parameters
    ----------
    recipe:
        Environment recipe.
    log_path:
        Log path for command output.

    Returns
    -------
    list[str] | None
        Sorted ``pip freeze`` lines, or ``None`` on failure.
    """

    command = (
        "conda",
        "run",
        "-n",
        _env_name(recipe.key),
        "python",
        "-m",
        "pip",
        "freeze",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {_shell_join(command)}\n")
        handle.write(completed.stdout)
        handle.write(completed.stderr)
        handle.write(f"[exit {completed.returncode}]\n")
    if completed.returncode != 0:
        return None
    return sorted(line for line in completed.stdout.splitlines() if line.strip())


def _remove_existing_env(recipe: EnvRecipe, log_path: Path, existing_envs: set[str]) -> bool:
    """Remove an existing environment before recreating it.

    Parameters
    ----------
    recipe:
        Environment recipe.
    log_path:
        Log path for command output.
    existing_envs:
        Mutable existing environment name set.

    Returns
    -------
    bool
        Whether the environment is now absent.
    """

    env_name = _env_name(recipe.key)
    if env_name not in existing_envs:
        return True
    command = ("conda", "env", "remove", "-y", "-n", env_name)
    if _run_logged(command, log_path) != 0:
        return False
    existing_envs.discard(env_name)
    return True


def _run_setup_for_plan(
    plan: EnvPlan, out_dir: Path, min_free_gb: float, existing_envs: set[str]
) -> bool:
    """Create or reuse one environment and capture its working recipe.

    Parameters
    ----------
    plan:
        Environment plan.
    out_dir:
        Shared output directory.
    min_free_gb:
        Required free GiB before create/install.
    existing_envs:
        Mutable existing environment name set.

    Returns
    -------
    bool
        Whether setup or reuse succeeded.
    """

    recipe = plan.recipe
    env_name = _env_name(recipe.key)
    log_path = out_dir / ENV_LOG_DIRNAME / f"{recipe.key}.log"
    if env_name in existing_envs and _env_import_check_passes(recipe, log_path):
        log_event("env_reuse", env=recipe.key, conda_env=env_name)
        resolved = _capture_resolved_packages(recipe, log_path)
        _update_env_status(recipe.key, "working", resolved)
        return True
    if env_name in existing_envs:
        log_event("env_recreate", env=recipe.key, conda_env=env_name)

    enough_disk, free_gb = _has_enough_disk(out_dir, min_free_gb)
    if not enough_disk:
        message = (
            f"Skipping {recipe.key}: {free_gb:.1f} GiB free under {out_dir}, "
            f"below --min-free-gb {min_free_gb:.1f}."
        )
        print(message, file=sys.stderr)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
        _update_env_status(recipe.key, "failed")
        return False

    if not _remove_existing_env(recipe, log_path, existing_envs):
        _update_env_status(recipe.key, "failed")
        return False
    if env_name not in existing_envs:
        if _run_logged(plan.create.command, log_path) != 0:
            _update_env_status(recipe.key, "failed")
            return False
        existing_envs.add(env_name)
    for item in plan.install:
        if _run_logged(item.command, log_path) != 0:
            _update_env_status(recipe.key, "failed")
            return False
    for item in plan.checks:
        if _run_logged(item.command, log_path) != 0:
            _update_env_status(recipe.key, "failed")
            return False
    resolved = _capture_resolved_packages(recipe, log_path)
    _update_env_status(recipe.key, "working", resolved)
    return True


def execute_plan(plans: Sequence[EnvPlan], out_dir: Path, min_free_gb: float) -> int:
    """Execute per-env plans, continuing across environment failures.

    Parameters
    ----------
    plans:
        Planned per-environment workflows.
    out_dir:
        Shared output directory.
    min_free_gb:
        Required free GiB before create/install.

    Returns
    -------
    int
        Zero if all selected environments succeeded, otherwise one.
    """

    existing_envs = _conda_envs()
    ok = True
    for plan in plans:
        recipe = plan.recipe
        log_path = out_dir / ENV_LOG_DIRNAME / f"{recipe.key}.log"
        setup_ok = _run_setup_for_plan(plan, out_dir, min_free_gb, existing_envs)
        if not setup_ok:
            ok = False
            continue
        for item in plan.runs:
            log_event("command_start", env=recipe.key, kind=item.kind, command=list(item.command))
            if _run_logged(item.command, log_path) != 0:
                ok = False
                log_event("command_failed", env=recipe.key, kind=item.kind, log=str(log_path))
                break
            log_event("command_done", env=recipe.key, kind=item.kind)
        if plan.cleanup is not None:
            if _run_logged(plan.cleanup.command, log_path) == 0:
                existing_envs.discard(_env_name(recipe.key))
            else:
                ok = False
    return 0 if ok else 1


def build_parser(specs: dict[str, EnvRecipe]) -> argparse.ArgumentParser:
    """Build the cross-environment runner CLI parser.

    Parameters
    ----------
    specs:
        Environment specs used for choices.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("render", "validate"), default="render")
    parser.add_argument("--envs", nargs="+", choices=tuple(specs), help="env keys to run")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--setup-only", action="store_true", help="create/install/check envs only")
    parser.add_argument(
        "--cleanup-env", action="store_true", help="remove each env after its task finishes"
    )
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="create/install/run commands")
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

    try:
        specs = load_env_specs()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    parser = build_parser(specs)
    args = parser.parse_args(argv)
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    try:
        plans = build_plan(args, specs)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    rows = load_rows()
    if not args.execute:
        if shutil.which("conda") is None:
            print("conda was not found on PATH; this is still a dry-run command plan.")
        print_plan(plans, rows, specs)
        return 0
    out_dir = (args.out_dir or _default_out_dir(args.task)).resolve()
    if shutil.which("conda") is None:
        _print_manual_recipe(plans)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    return execute_plan(plans, out_dir, args.min_free_gb)


if __name__ == "__main__":
    raise SystemExit(main())
