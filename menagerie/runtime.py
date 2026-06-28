"""Impure menagerie runtime policy for dependencies, devices, caches, and workers."""

from __future__ import annotations

import argparse
import ast
import gc
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from menagerie.catalog import CatalogRow, load_rows

CACHE_ROOTS = (
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    Path.home() / ".cache" / "torch" / "hub",
    Path.home() / ".cache" / "timm",
)
CORE_FEATURED_ZOOS = {
    "torchvision",
    "torchvision.models",
    "timm-core",
    "hf-core",
    "huggingface_transformers",
    "transformers",
    "transformers:automodel",
}
CANONICAL_FEATURE_PATTERNS = (
    "alexnet",
    "lenet",
    "vgg",
    "resnet",
    "resnext",
    "densenet",
    "inception",
    "googlenet",
    "mobilenet",
    "efficientnet",
    "convnext",
    "transformer",
    "bert",
    "roberta",
    "gpt2",
    "gpt",
    "t5",
    "vit",
    "deit",
    "swin",
    "clip",
    "unet",
    "u-net",
    "yolo",
    "mask-rcnn",
    "faster-rcnn",
    "stable-diffusion",
    "mamba",
    "gan",
    "llama",
    "whisper",
    "dust3r",
    "nerf",
    "sam",
    "detr",
    "diffusion",
    "mistral",
    "mixtral",
)
MODULE_PACKAGE_MAP = {
    "clip": "clip",
    "diffusers": "diffusers",
    "effdet": "effdet",
    "mmcv": "mmcv",
    "mmdet": "mmdet",
    "mmengine": "mmengine",
    "mmseg": "mmsegmentation",
    "monai": "monai",
    "norse": "norse",
    "open_clip": "open-clip-torch",
    "recbole": "recbole",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "sentence_transformers": "sentence-transformers",
    "snntorch": "snntorch",
    "speechbrain": "speechbrain",
    "super_image": "super-image",
    "timm": "timm",
    "torch_geometric": "torch-geometric",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "ultralytics": "ultralytics",
}
# Framework module names always treated as real external dependencies even when they have no
# MODULE_PACKAGE_MAP entry (their pip name == module name, or they are vendored framework extras).
KNOWN_FRAMEWORK_MODULES = {
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    *MODULE_PACKAGE_MAP,
}
# Generic, repo-internal source-layout package names. A recipe that does
# `from models.foo import X` / `from src.bar import Y` is importing the ORIGINAL repo's local
# package tree, NOT a PyPI distribution. These are never pip-installable and must NOT be reported
# as "dependency missing" (that wrongly skipped ~350 base-env-renderable rows and produced bogus
# cluster keys like `models` (226), `src` (48), `lib` (14), ...). Recipes that genuinely need the
# upstream source are surfaced honestly via `unrenderable_reason` as `local_source_unavailable`,
# and self-contained recipes that merely *mention* such a name render normally.
LOCAL_SOURCE_NAMES = {
    "models",
    "model",
    "model_zoo",
    "modeling",
    "modelling",
    "src",
    "lib",
    "libs",
    "networks",
    "network",
    "net",
    "nets",
    "nn_modules",
    "implementations",
    "impl",
    "training",
    "train",
    "core",
    "architectures",
    "architecture",
    "arch",
    "source",
    "sources",
    "types",
    "utils",
    "util",
    "modules",
    "module",
    "backbone",
    "backbones",
    "layers",
    "layer",
    "common",
    "components",
    "component",
    "scripts",
    "script",
    "code",
    "main",
    "config",
    "configs",
    "cfg",
}
# Names that are stdlib or recipe-local and never count as an external dependency.
STDLIB_OR_LOCAL = {
    "model",
    "cfg",
    "os",
    "sys",
    "math",
    "typing",
    "functools",
    "collections",
    "re",
    "itertools",
    "json",
    "copy",
    "abc",
    "dataclasses",
    "warnings",
    "builtins",
    "random",
}
ZOO_PACKAGE_HINTS = (
    (re.compile(r"torchvision", re.I), ("torchvision",)),
    (re.compile(r"\btimm\b", re.I), ("timm",)),
    (re.compile(r"transformers|huggingface", re.I), ("transformers",)),
    (re.compile(r"diffusers", re.I), ("diffusers", "transformers")),
    (re.compile(r"segmentation_models_pytorch|smp", re.I), ("segmentation-models-pytorch",)),
    (re.compile(r"ultralytics", re.I), ("ultralytics",)),
    (re.compile(r"torch_geometric|pyg", re.I), ("torch-geometric",)),
    (re.compile(r"open.?mmlab|mmdet|mmseg|mmpose|mmaction|mmagic", re.I), ("mmengine",)),
    (re.compile(r"recbole", re.I), ("recbole",)),
    (re.compile(r"open_clip", re.I), ("open-clip-torch",)),
)
UNRENDERABLE_MARKERS = (
    ("verification_expectation=deferred", "deferred"),
    ("jax", "jax_native"),
    ("flax", "jax_native"),
    ("web-only", "web_only_recipe"),
    ("web only", "web_only_recipe"),
    ("source-only", "source_only_recipe"),
    ("source only", "source_only_recipe"),
    ("no public code", "no_public_code"),
    # NOTE: "gated"/"weights-gated"/"not random-init" markers REMOVED 2026-06-19 — the menagerie
    # builds every model RANDOM-INIT, so gated/missing trained weights are irrelevant; those markers
    # caught architectural terms ("gated activation", "GatedGenerator") and wrongly skipped buildable
    # models (cornet_rt, WaveNet, Deep Kalman Filter, ...). Such rows now render or surface a real
    # recipe bug (-> normal repair), never a fake "ceiling".
    ("metadata-only", "metadata_only"),
    ("catalog-only", "metadata_only"),
)
DEVICE_ERROR_MARKERS = (
    "triton",
    "cpu tensor",
    "expected all tensors to be on the same device",
    "cuda",
    "must be on",
    "device",
)


@dataclass(frozen=True)
class CacheSnapshot:
    """A filesystem snapshot for cache cleanup.

    Parameters
    ----------
    root:
        Cache root path.
    paths:
        Relative paths that existed before a model ran.
    """

    root: Path
    paths: frozenset[Path]


@dataclass(frozen=True)
class DependencyPlan:
    """Dependency resolution plan for one catalog row.

    Parameters
    ----------
    cluster_key:
        Stable dependency-cluster label used for grouping.
    packages:
        Pip packages to install before processing the cluster.
    top_modules:
        Top-level import modules expected after installation.
    environment:
        Current Python environment label.
    recipe_parse_error:
        Non-empty when the recipe source is malformed and could not be parsed. When set, the row
        is recorded as a clean ``recipe_error`` rather than producing a garbled cluster key from a
        regex split of the (unparseable) recipe body.
    """

    cluster_key: str
    packages: tuple[str, ...]
    top_modules: tuple[str, ...]
    environment: str
    recipe_parse_error: str = ""


def disk_free_gb(path: Path) -> float:
    """Return free disk space for a path in GiB.

    Parameters
    ----------
    path:
        Path whose filesystem should be checked.

    Returns
    -------
    float
        Free GiB.
    """

    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def assert_min_free(path: Path, min_free_gb: float) -> None:
    """Raise if free disk is below the configured threshold.

    Parameters
    ----------
    path:
        Path whose filesystem should be checked.
    min_free_gb:
        Minimum free GiB.
    """

    free_gb = disk_free_gb(path)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"free disk below threshold: {free_gb:.2f} GiB < {min_free_gb:.2f} GiB at {path}"
        )


def snapshot_cache(root: Path) -> CacheSnapshot:
    """Snapshot a cache tree before running one model.

    Parameters
    ----------
    root:
        Cache root.

    Returns
    -------
    CacheSnapshot
        Snapshot with relative paths.
    """

    if not root.exists():
        return CacheSnapshot(root=root, paths=frozenset())
    paths = frozenset(path.relative_to(root) for path in root.rglob("*"))
    return CacheSnapshot(root=root, paths=paths)


def purge_new_cache_entries(snapshot: CacheSnapshot) -> int:
    """Remove cache paths created after a snapshot.

    Parameters
    ----------
    snapshot:
        Cache snapshot.

    Returns
    -------
    int
        Number of new paths removed or already absent.
    """

    root = snapshot.root
    if not root.exists():
        return 0
    current = sorted(
        (path.relative_to(root) for path in root.rglob("*")),
        key=lambda relpath: len(relpath.parts),
        reverse=True,
    )
    removed = 0
    for relpath in current:
        if relpath in snapshot.paths:
            continue
        path = root / relpath
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        except FileNotFoundError:
            removed += 1
    return removed


def cleanup_runtime(cache_snapshots: Sequence[CacheSnapshot], tmp_dir: Path) -> int:
    """Clear accelerator memory, new caches, and temporary files.

    Parameters
    ----------
    cache_snapshots:
        Cache snapshots captured before model execution.
    tmp_dir:
        Per-model temporary directory.

    Returns
    -------
    int
        Number of new cache entries removed.
    """

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    removed = sum(purge_new_cache_entries(snapshot) for snapshot in cache_snapshots)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return removed


def is_device_related_error(error: BaseException) -> bool:
    """Return whether an exception looks like a device placement failure.

    Parameters
    ----------
    error:
        Exception raised while running a model.

    Returns
    -------
    bool
        Whether the error message matches known device-related failures.
    """

    message = str(error).lower()
    return any(marker in message for marker in DEVICE_ERROR_MARKERS)


def move_input_to_device(value: Any, device: str) -> Any:
    """Move nested tensor inputs to a target device.

    Parameters
    ----------
    value:
        Tensor, sequence, mapping, or scalar input object.
    device:
        Target torch device string.

    Returns
    -------
    Any
        Input object with tensor leaves moved to ``device``.
    """

    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_input_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_input_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_input_to_device(item, device) for key, item in value.items()}
    return value


def move_model_and_input_to_device(model: Any, input_value: Any, device: str) -> tuple[Any, Any]:
    """Move a model and its example input to a target device.

    Parameters
    ----------
    model:
        Instantiated model.
    input_value:
        Example model input.
    device:
        Target torch device string.

    Returns
    -------
    tuple[Any, Any]
        Moved ``(model, input_value)`` pair.
    """

    return model.to(device), move_input_to_device(input_value, device)


def cuda_is_available() -> bool:
    """Return whether PyTorch reports an available CUDA device.

    Returns
    -------
    bool
        Whether CUDA is available.
    """

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_note(requested_device: str, actual_device: str) -> str:
    """Build a manifest note for non-default device execution.

    Parameters
    ----------
    requested_device:
        CLI device mode.
    actual_device:
        Device used for the successful attempt.

    Returns
    -------
    str
        Manifest note, or an empty string for default-compatible CPU runs.
    """

    if requested_device == "cpu" and actual_device == "cpu":
        return ""
    return f"device={actual_device}"


def combine_notes(*notes: str) -> str:
    """Join non-empty manifest notes.

    Parameters
    ----------
    *notes:
        Candidate note strings.

    Returns
    -------
    str
        Combined manifest note.
    """

    return "; ".join(note for note in notes if note)


def safe_path_part(value: str) -> str:
    """Return a conservative filesystem path component.

    Parameters
    ----------
    value:
        Source value.

    Returns
    -------
    str
        Filesystem-safe path component.
    """

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:180] or "unknown"


def featured_reason(row: CatalogRow) -> str:
    """Return a feature-gallery reason for a row, or an empty string.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Reason string or empty string.
    """

    zoo_lower = row.zoo.lower()
    haystack = f"{row.name} {row.family_normalized}".lower().replace("_", "-")
    if zoo_lower in CORE_FEATURED_ZOOS:
        return f"core zoo: {row.zoo}"
    for pattern in CANONICAL_FEATURE_PATTERNS:
        if pattern in haystack:
            return f"canonical family: {pattern}"
    return ""


def is_featured(row: CatalogRow) -> bool:
    """Return whether a row belongs in the curated featured tier.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    bool
        Whether the row is featured.
    """

    return bool(featured_reason(row))


def _collect_recipe_names(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Collect (imported-module-roots, attribute-roots, locally-bound-names) from a parsed recipe.

    ``imported`` are the top-level package of every real absolute ``import``/``from`` statement --
    these are the only reliable EXTERNAL-dependency signal. ``attribute_roots`` are the root names of
    dotted attribute access (``models.networks.X`` -> ``models``); they are used ONLY to detect a
    bare reference to the upstream repo's local source layout (e.g. ``models.networks.X(...)`` with
    no import), never to invent a pip dependency -- a bare attribute root is just as likely a
    namespace alias (``np.array``) or pseudo-code (``nn.GRUCell``) as a module. ``bound`` are import
    aliases, assignment targets, comprehension/lambda/def/class params -- names LOCAL to the recipe.

    Parameters
    ----------
    tree:
        Parsed recipe AST.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        ``(imported_modules, attribute_roots, bound_names)``.
    """

    imported: set[str] = set()
    bound: set[str] = set()
    attr_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Only absolute imports name an external module; relative (level>0) imports are local.
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        bound.add(name.id)
        elif isinstance(node, ast.comprehension):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    bound.add(name.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                bound.add(arg.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Attribute):
            value = node.value
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                attr_roots.add(value.id)
    return imported, attr_roots, bound


def _required_modules_with_error(constructor_call: str, zoo: str) -> tuple[tuple[str, ...], str]:
    """Infer top-level EXTERNAL modules required by a constructor, plus a recipe parse-error string.

    Uses a real AST parse instead of a regex split. The old regex split on ``[;\\n]`` shredded
    inline ``exec``/``code=`` recipe bodies, mistaking fragments of the quoted source for imports
    and producing garbled cluster keys. AST parsing ignores string-literal contents entirely, so
    only genuine outer imports/attribute roots count; an unparseable recipe yields a clean error
    string instead of garbage. Generic repo-internal source names (``models``/``src``/``lib``/...)
    are local layout, NOT pip distributions, so they are excluded from the external-dep set.

    Parameters
    ----------
    constructor_call:
        Catalog constructor expression.
    zoo:
        Source model zoo.

    Returns
    -------
    tuple[tuple[str, ...], str]
        ``(external_module_names, parse_error)`` -- ``parse_error`` is empty on success.
    """

    modules: set[str] = set()
    bound: set[str] = set()
    parse_error = ""
    source = constructor_call.strip()
    try:
        tree: ast.AST | None = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        tree = None
        parse_error = f"recipe parse error: {exc.msg} (line {exc.lineno})"
    if tree is not None:
        # External-dep detection uses ONLY real import statements -- never bare attribute roots,
        # which are just as likely a namespace alias (`np.array`) or pseudo-code (`nn.GRUCell`) as a
        # module and would wrongly gate valid recipes. Local-source attribute refs are handled
        # separately by `unrenderable_reason`.
        modules, _attr_roots, bound = _collect_recipe_names(tree)
    else:
        # Malformed recipe: never regex-split the body (that is what produced garbled cluster keys).
        # Scan ONLY for known framework tokens so a recognizable real dep is still detected.
        for token in KNOWN_FRAMEWORK_MODULES:
            if re.search(rf"\b{re.escape(token)}\b", source):
                modules.add(token)
    zoo_lower = zoo.lower()
    if "torchvision" in zoo_lower:
        modules.add("torchvision")
    if "timm" in zoo_lower:
        modules.add("timm")
    if "transformers" in zoo_lower or "huggingface" in zoo_lower:
        modules.add("transformers")
    if "diffusers" in zoo_lower:
        modules.add("diffusers")
    if "segmentation_models_pytorch" in zoo_lower or "smp" in zoo_lower:
        modules.add("segmentation_models_pytorch")
    if "ultralytics" in zoo_lower:
        modules.add("ultralytics")

    def _is_external_dep(name: str) -> bool:
        if not name or name in STDLIB_OR_LOCAL:
            return False
        if name[0].isupper():
            return False  # a Class/symbol, not a module
        if name in LOCAL_SOURCE_NAMES:
            return False  # repo-internal source layout, not a PyPI distribution
        if name in bound and name not in KNOWN_FRAMEWORK_MODULES:
            return False  # locally bound alias / variable, not an imported module
        return True

    return tuple(sorted(m for m in modules if _is_external_dep(m))), parse_error


def required_modules(constructor_call: str, zoo: str) -> tuple[str, ...]:
    """Infer top-level EXTERNAL modules required by a constructor expression.

    Parameters
    ----------
    constructor_call:
        Catalog constructor expression.
    zoo:
        Source model zoo.

    Returns
    -------
    tuple[str, ...]
        Required top-level import names.
    """

    modules, _error = _required_modules_with_error(constructor_call, zoo)
    return modules


def dependency_plan(row: CatalogRow) -> DependencyPlan:
    """Build the dependency plan for one catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    DependencyPlan
        Dependency plan.
    """

    top_modules, parse_error = _required_modules_with_error(row.constructor_call, row.zoo)
    packages: set[str] = set()
    pip_cluster_package = _pip_cluster_package(row.zoo)
    for module in top_modules:
        package = MODULE_PACKAGE_MAP.get(module)
        if (
            package is None
            and pip_cluster_package is not None
            and module == pip_cluster_package.replace("-", "_")
        ):
            package = pip_cluster_package
        if package:
            packages.add(package)
    for pattern, hints in ZOO_PACKAGE_HINTS:
        if pattern.search(row.zoo):
            packages.update(hints)
    environment = f"python-{sys.version_info.major}.{sys.version_info.minor}:{sys.prefix}"
    if parse_error:
        # Malformed recipe: a stable, non-garbled cluster key (never the shredded recipe body).
        cluster_key = "recipe_error"
    else:
        cluster_parts = sorted(packages) or sorted(top_modules) or [safe_path_part(row.zoo.lower())]
        cluster_key = "+".join(cluster_parts)
    return DependencyPlan(
        cluster_key=cluster_key,
        packages=tuple(sorted(packages)),
        top_modules=top_modules,
        environment=environment,
        recipe_parse_error=parse_error,
    )


def _pip_cluster_package(zoo: str) -> str | None:
    """Return the package implied by an explicit ``pip:`` dependency cluster.

    Parameters
    ----------
    zoo:
        Catalog zoo/dependency label.

    Returns
    -------
    str | None
        Package name, or ``None`` when the row does not explicitly
        declare a pip-installable dependency cluster.
    """

    prefix = "pip:"
    if not zoo.startswith(prefix):
        return None
    package = zoo[len(prefix) :].strip()
    return package if package else None


def module_importable(module_name: str) -> bool:
    """Return whether a top-level module is importable.

    Parameters
    ----------
    module_name:
        Module name.

    Returns
    -------
    bool
        Whether importlib can find the module.
    """

    return importlib.util.find_spec(module_name) is not None


def missing_modules(plan: DependencyPlan) -> tuple[str, ...]:
    """Return missing import modules for a dependency plan.

    Parameters
    ----------
    plan:
        Dependency plan.

    Returns
    -------
    tuple[str, ...]
        Missing module names.
    """

    return tuple(module for module in plan.top_modules if not module_importable(module))


def install_dependency_plan(plan: DependencyPlan, args: argparse.Namespace) -> str | None:
    """Install dependency packages for a cluster when requested.

    Parameters
    ----------
    plan:
        Dependency plan.
    args:
        Parsed CLI args.

    Returns
    -------
    str | None
        Error message on failure, otherwise ``None``.
    """

    if not args.install_deps:
        missing = missing_modules(plan)
        if missing:
            return f"dependency missing and --no-install-deps set: {', '.join(missing)}"
        return None
    missing = missing_modules(plan)
    if not missing:
        return None
    if not plan.packages:
        return f"dependency missing with no package mapping: {', '.join(missing)}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *args.pip_args,
        *plan.packages,
    ]
    log_event("install_start", cluster=plan.cluster_key, packages=list(plan.packages))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.install_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"dependency install timed out after {args.install_timeout}s: {plan.packages}"
    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()[-5:]
        return "dependency install failed: " + " | ".join(stderr)
    still_missing = missing_modules(plan)
    if still_missing:
        return f"dependency installed but modules still missing: {', '.join(still_missing)}"
    log_event("install_done", cluster=plan.cluster_key, packages=list(plan.packages))
    return None


def unrenderable_reason(row: CatalogRow) -> str | None:
    """Return an honest skip reason for rows that are not runnable PyTorch recipes.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str | None
        Skip reason or ``None``.
    """

    joined = f"{row.zoo} {row.constructor_call} {row.notes}".lower()
    if "jax" in row.zoo.lower() or "jax" in row.constructor_call.lower():
        return "jax_native"
    # Recipes that import the upstream repo's local source layout (`from models...`, `from src...`,
    # `models.networks.X(...)`) need that repo on sys.path -- not a PyPI install. Surface that
    # honestly instead of letting the dependency stage mislabel them "dependency missing". A recipe
    # that is self-contained (its only `models`/`src` mention is a defined symbol, or it `exec`s an
    # inline reimplementation) does NOT trip this, because such names are locally bound.
    try:
        local_tree: ast.AST | None = ast.parse(row.constructor_call.strip(), mode="exec")
    except SyntaxError:
        local_tree = None
    if local_tree is not None:
        imported, attr_roots, bound = _collect_recipe_names(local_tree)
        local_refs = {
            r for r in (imported | attr_roots) if r in LOCAL_SOURCE_NAMES and r not in bound
        }
        if local_refs:
            return "local_source_unavailable"
    if "<model_config>" in row.constructor_call:
        return "web_or_config_recipe_sketch"
    # A real resolved OpenMMLab .mim config build -- Config.fromfile(<pkg>/.mim/configs/<real>.py) -- is a
    # genuine buildable recipe, NOT a sketch. Only flag config.fromfile when it lacks a resolved .mim path
    # (a placeholder / web-config sketch). Fixed 2026-06-19: the blanket flag wrongly pre-skipped 838
    # mm-family recipes that build the real published architecture from the installed .mim config.
    if "config.fromfile" in joined and ".mim/configs/" not in row.constructor_call.lower():
        return "web_or_config_recipe_sketch"
    for marker, reason in UNRENDERABLE_MARKERS:
        if marker in joined:
            return reason
    return None


def _importable_attr_module_roots(constructor_call: str) -> set[str]:
    """Attribute roots that resolve to an installed importable module.

    Many catalog recipes call a model via a dotted attribute path with no explicit
    ``import`` statement -- e.g. ``torchaudio.models.DeepSpeech(...)`` or
    ``monai.networks.nets.AHNet(...)``. ``required_modules`` deliberately ignores bare
    attribute roots for *dependency-cluster* gating (an attribute root is just as likely a
    namespace alias like ``np`` or pseudo-code like ``nn`` as a real package). For
    *namespace building*, though, an attribute root that genuinely resolves to an installed
    top-level module must be bound, or the recipe raises ``NameError`` even though the
    package is present. This recovers the whole ``module.Class(...)``-with-no-import class of
    recipes (torchaudio / torchsr / monai / torch_geometric / cornet / ...) without inventing
    any pip dependency: only roots that ``importlib`` can actually find a spec for are added.
    """

    try:
        tree = ast.parse(constructor_call.strip(), mode="exec")
    except SyntaxError:
        return set()
    _imported, attr_roots, bound = _collect_recipe_names(tree)
    roots: set[str] = set()
    for root in attr_roots:
        if not root or root[0].isupper():
            continue  # a Class/symbol, not a module
        if root in bound or root in STDLIB_OR_LOCAL or root in LOCAL_SOURCE_NAMES:
            continue
        try:
            if importlib.util.find_spec(root) is not None:
                roots.add(root)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return roots


def import_namespace(row: CatalogRow) -> dict[str, Any]:
    """Build the namespace used to instantiate a model recipe.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    dict[str, Any]
        Evaluation namespace.
    """

    namespace: dict[str, Any] = {}
    module_names = {
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "diffusers",
        "segmentation_models_pytorch",
        *required_modules(row.constructor_call, row.zoo),
        *_importable_attr_module_roots(row.constructor_call),
    }
    for module_name in sorted(module_names):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        namespace[module_name] = module
        if module_name == "segmentation_models_pytorch":
            namespace["smp"] = module
        if module_name == "transformers":
            for attr in (
                "AutoConfig",
                "AutoModel",
                "AutoModelForCausalLM",
                "AutoModelForMaskedLM",
                "AutoModelForSeq2SeqLM",
                "AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForSemanticSegmentation",
                "AutoModelForTextToSpectrogram",
                "AutoModelForTextToWaveform",
                "AutoModelForZeroShotImageClassification",
                "BertConfig",
                "GPT2Config",
                "T5Config",
            ):
                if hasattr(module, attr):
                    namespace[attr] = getattr(module, attr)
    return namespace


@contextmanager
def isolated_tmp_env(tmp_dir: Path) -> Iterator[None]:
    """Temporarily route common temporary-file variables to a per-model directory.

    Parameters
    ----------
    tmp_dir:
        Temporary directory for this model.

    Yields
    ------
    None
        Context body.
    """

    old_values = {key: os.environ.get(key) for key in ("TMPDIR", "TEMP", "TMP")}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key in old_values:
            os.environ[key] = str(tmp_dir)
        tempfile.tempdir = str(tmp_dir)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = None


def select_rows(args: argparse.Namespace) -> list[CatalogRow]:
    """Select catalog rows from CLI filters.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    list[CatalogRow]
        Selected rows.
    """

    rows = load_rows(
        family=args.family,
        domain=args.domain,
        zoo=args.zoo,
        verified=args.verified_only,
        limit=None,
        db_path=args.db,
    )
    if args.name:
        terms = [term.lower() for term in args.name]
        rows = [row for row in rows if any(term in row.name.lower() for term in terms)]
    if getattr(args, "names_file", None):
        with open(args.names_file) as handle:
            wanted = {line.strip().lower() for line in handle if line.strip()}
        rows = [row for row in rows if row.name.lower() in wanted]
    if args.model_id:
        model_ids = set(args.model_id)
        rows = [row for row in rows if row.model_id in model_ids]
    if getattr(args, "stable_ids", None):
        stable_ids = set(args.stable_ids)
        rows = [row for row in rows if row.stable_id in stable_ids]
    if args.featured_only:
        rows = [row for row in rows if is_featured(row)]
    if args.since is not None:
        rows = [row for row in rows if row.model_id > args.since]
    if args.subset is not None:
        rows = rows[: args.subset]
    if args.max_models is not None:
        rows = rows[: args.max_models]
    return rows


def group_by_dependency(
    rows: Sequence[CatalogRow],
) -> list[tuple[DependencyPlan, list[CatalogRow]]]:
    """Group rows by dependency plan so installs are amortized.

    Parameters
    ----------
    rows:
        Catalog rows.

    Returns
    -------
    list[tuple[DependencyPlan, list[CatalogRow]]]
        Dependency groups.
    """

    groups: dict[str, list[CatalogRow]] = defaultdict(list)
    plans: dict[str, DependencyPlan] = {}
    for row in rows:
        plan = dependency_plan(row)
        plans[plan.cluster_key] = plan
        groups[plan.cluster_key].append(row)
    return [(plans[key], groups[key]) for key in sorted(groups)]


def log_event(event: str, **payload: Any) -> None:
    """Write one structured log event to stdout.

    Parameters
    ----------
    event:
        Event name.
    **payload:
        JSON payload.
    """

    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def default_jobs() -> int:
    """Return the default concurrency level for the render/validate engines.

    Returns
    -------
    int
        ``min(8, os.cpu_count() - 2)`` clamped to at least one worker.
    """

    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 2))
