"""Dependency-aware, disk-safe renderer for the TorchLens model menagerie."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import html
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Iterator, Mapping, Sequence

from menagerie.catalog import CatalogRow, load_rows


DEFAULT_OUT_DIR = Path("/tmp/torchlens_menagerie_gallery")
MANIFEST_COLUMNS = (
    "name",
    "model_id",
    "status",
    "n_nodes",
    "render_path",
    "elapsed",
    "dependency_cluster",
    "error",
    "graph_shape_hash",
)
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
    """

    cluster_key: str
    packages: tuple[str, ...]
    top_modules: tuple[str, ...]
    environment: str


@dataclass(frozen=True)
class RenderResult:
    """One model render result.

    Parameters
    ----------
    name:
        Catalog model name.
    model_id:
        Catalog model identifier.
    status:
        Result status, such as ``rendered`` or ``skipped:dependency_missing``.
    n_nodes:
        Number of traced graph nodes.
    render_path:
        Rendered artifact path, if produced.
    elapsed:
        Elapsed seconds.
    dependency_cluster:
        Dependency cluster used for this row.
    error:
        Error or skip note.
    graph_shape_hash:
        TorchLens architecture hash for deduplication.
    """

    name: str
    model_id: int
    status: str
    n_nodes: int
    render_path: str
    elapsed: float
    dependency_cluster: str
    error: str
    graph_shape_hash: str = ""


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


def parse_vis_option_value(value: str) -> Any:
    """Coerce a CLI ``KEY=VALUE`` value to a bool, int, float, or string.

    Parameters
    ----------
    value:
        Raw string value from a ``--vis-option KEY=VALUE`` argument.

    Returns
    -------
    Any
        Parsed scalar: ``True``/``False`` for boolean literals, an ``int`` or
        ``float`` for numeric literals, otherwise the original string.
    """

    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_vis_options(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` strings into draw() keyword arguments.

    Parameters
    ----------
    pairs:
        Sequence of ``KEY=VALUE`` strings from ``--vis-option``.

    Returns
    -------
    dict[str, Any]
        Mapping of draw() keyword names to coerced values.
    """

    options: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected KEY=VALUE for --vis-option, got {pair!r}")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in --vis-option {pair!r}")
        options[key] = parse_vis_option_value(raw_value)
    return options


def manifest_records(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read the latest manifest record for each model name.

    Parameters
    ----------
    manifest_path:
        Manifest TSV path.

    Returns
    -------
    dict[str, dict[str, str]]
        Latest manifest rows keyed by model name.
    """

    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["name"]: row for row in reader if row.get("name")}


def completed_names(manifest_path: Path, retry_failed: bool) -> set[str]:
    """Read names already completed in an append-only manifest.

    Parameters
    ----------
    manifest_path:
        Manifest TSV path.
    retry_failed:
        Whether failed and skipped rows should be retried.

    Returns
    -------
    set[str]
        Completed model names.
    """

    records = manifest_records(manifest_path)
    if not retry_failed:
        return set(records)
    return {name for name, row in records.items() if row.get("status") == "rendered"}


def append_manifest(manifest_path: Path, result: RenderResult) -> None:
    """Append one result row to the manifest.

    Parameters
    ----------
    manifest_path:
        Manifest TSV path.
    result:
        Render result.
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(
            (
                result.name,
                result.model_id,
                result.status,
                result.n_nodes,
                result.render_path,
                f"{result.elapsed:.3f}",
                result.dependency_cluster,
                result.error,
                result.graph_shape_hash,
            )
        )


def parse_shape(shape: str) -> tuple[int, ...] | list[tuple[int, ...]]:
    """Parse a concrete tensor shape from the catalog.

    Parameters
    ----------
    shape:
        Catalog shape string.

    Returns
    -------
    tuple[int, ...] | list[tuple[int, ...]]
        Parsed input shape or list of shapes.
    """

    shape_text = shape.strip()
    parsed_text = shape_text
    if not shape_text.startswith(("(", "[")):
        match = re.search(r"\(([0-9,\s]+)\)", shape_text)
        if match is None:
            raise ValueError(f"expected concrete tuple shape, got {shape!r}")
        parsed_text = f"({match.group(1)})"
    parsed = ast.literal_eval(parsed_text)
    if isinstance(parsed, tuple) and all(isinstance(value, int) for value in parsed):
        return parsed
    if isinstance(parsed, list) and all(
        isinstance(item, tuple) and all(isinstance(value, int) for value in item) for item in parsed
    ):
        return parsed
    raise ValueError(f"expected tuple[int, ...] or list[tuple[int, ...]], got {shape!r}")


def tensor_for_recipe(shape: str, dtype: str) -> Any:
    """Create a synthetic input tensor or input list for a catalog recipe.

    Parameters
    ----------
    shape:
        Catalog input shape.
    dtype:
        Catalog dtype string.

    Returns
    -------
    Any
        Torch tensor or list of tensors.
    """

    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"unsupported input_dtype={dtype!r}")
    parsed_shape = parse_shape(shape)

    def make_tensor(parsed: tuple[int, ...]) -> Any:
        """Create one tensor for an already parsed shape."""

        if torch_dtype.is_floating_point:
            return torch.randn(parsed, dtype=torch_dtype)
        if torch_dtype is torch.bool:
            return torch.zeros(parsed, dtype=torch_dtype)
        return torch.zeros(parsed, dtype=torch_dtype)

    if isinstance(parsed_shape, list):
        return [make_tensor(item) for item in parsed_shape]
    return make_tensor(parsed_shape)


def is_classics_row(row: CatalogRow) -> bool:
    """Return whether a catalog row is a local historical classic.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    bool
        Whether the row is provided by ``menagerie.classics``.
    """

    return row.zoo == "classics-pytorch" and row.constructor_call.startswith("menagerie.classics.")


def classics_module_name(row: CatalogRow) -> str:
    """Extract the classics module name from a constructor expression.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Module name under ``menagerie.classics``.
    """

    match = re.fullmatch(
        r"menagerie\.classics\.([A-Za-z_][A-Za-z0-9_]*)\.build\(\)", row.constructor_call
    )
    if match is None:
        raise ValueError(f"unsupported classics constructor={row.constructor_call!r}")
    return match.group(1)


def classics_example_input(row: CatalogRow) -> Any:
    """Return the registered example input for a local historical classic.

    Resolves through the ``menagerie.classics`` registry by canonical name, so
    both singleton modules (``example_input``) and grouped family modules
    (``example_input_<variant>``) are handled uniformly.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Example input object from the classics registry.
    """

    from menagerie.classics import CLASSICS

    return CLASSICS[row.name]["example_input"]()


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


def model_render_stem(row: CatalogRow, out_dir: Path) -> Path:
    """Return the organized render stem for one model.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Gallery output directory.

    Returns
    -------
    Path
        Render stem without file extension.
    """

    if is_classics_row(row):
        return (
            out_dir
            / "history"
            / safe_path_part(row.era)
            / safe_path_part(row.family_normalized)
            / f"{row.model_id:05d}_{safe_path_part(row.name)}"
        )
    return (
        out_dir
        / safe_path_part(row.domain)
        / safe_path_part(row.family_normalized)
        / f"{row.model_id:05d}_{safe_path_part(row.name)}"
    )


def model_render_path(row: CatalogRow, out_dir: Path, file_format: str) -> Path:
    """Return the organized render file path for one model.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Gallery output directory.
    file_format:
        Render file format.

    Returns
    -------
    Path
        Render path.
    """

    return Path(f"{model_render_stem(row, out_dir)}.{file_format}")


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


def link_featured_copy(row: CatalogRow, render_path: Path, out_dir: Path) -> Path:
    """Create or refresh a featured symlink or copy for one rendered model.

    Parameters
    ----------
    row:
        Catalog row.
    render_path:
        Rendered file path.
    out_dir:
        Gallery output directory.

    Returns
    -------
    Path
        Featured path.
    """

    featured_dir = out_dir / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)
    target = featured_dir / f"{row.model_id:05d}_{safe_path_part(row.name)}{render_path.suffix}"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(os.path.relpath(render_path, target.parent))
    except OSError:
        shutil.copy2(render_path, target)
    return target


def required_modules(constructor_call: str, zoo: str) -> tuple[str, ...]:
    """Infer top-level modules required by a constructor expression.

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

    modules: set[str] = set()
    for match in re.finditer(r"\b(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", constructor_call):
        module_name = match.group(1)
        if not module_name[0].isupper():
            modules.add(module_name)
    dotted = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\.", constructor_call)
    if dotted and not dotted.group(1)[0].isupper():
        modules.add(dotted.group(1))
    for token in ("timm", "torch", "torchvision", "transformers", "diffusers", *MODULE_PACKAGE_MAP):
        if re.search(rf"\b{re.escape(token)}\b", constructor_call):
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
    return tuple(sorted(module for module in modules if module not in {"model", "cfg"}))


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

    top_modules = required_modules(row.constructor_call, row.zoo)
    packages: set[str] = set()
    for module in top_modules:
        package = MODULE_PACKAGE_MAP.get(module)
        if package:
            packages.add(package)
    for pattern, hints in ZOO_PACKAGE_HINTS:
        if pattern.search(row.zoo):
            packages.update(hints)
    environment = f"python-{sys.version_info.major}.{sys.version_info.minor}:{sys.prefix}"
    cluster_parts = sorted(packages) or sorted(top_modules) or [safe_path_part(row.zoo.lower())]
    return DependencyPlan(
        cluster_key="+".join(cluster_parts),
        packages=tuple(sorted(packages)),
        top_modules=top_modules,
        environment=environment,
    )


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
    if "<model_config>" in row.constructor_call or "config.fromfile" in joined:
        return "web_or_config_recipe_sketch"
    for marker, reason in UNRENDERABLE_MARKERS:
        if marker in joined:
            return reason
    return None


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


def instantiate_model(row: CatalogRow) -> Any:
    """Instantiate a model from a guarded constructor expression.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Instantiated model.
    """

    if is_classics_row(row):
        from menagerie.classics import CLASSICS

        return CLASSICS[row.name]["build"]()

    namespace = import_namespace(row)
    builtins = {
        "__import__": __import__,
        "dict": dict,
        "getattr": getattr,
        "len": len,
        "list": list,
        "range": range,
        "set": set,
        "tuple": tuple,
    }
    globals_dict = {"__builtins__": builtins, **namespace}
    constructor_call = row.constructor_call.strip()
    if ";" in constructor_call or constructor_call.startswith(("import ", "from ")):
        locals_dict = dict(namespace)
        exec(constructor_call, globals_dict, locals_dict)  # noqa: S102
        for output_name in ("model", "net", "module"):
            if output_name in locals_dict:
                return locals_dict[output_name]
        raise ValueError("statement recipe did not assign a `model`, `net`, or `module` variable")
    return eval(constructor_call, globals_dict, namespace)  # noqa: S307


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


def render_one(
    row: CatalogRow,
    out_dir: Path,
    dry_run: bool,
    file_format: str,
    device: str,
    vis_options: Mapping[str, Any] | None = None,
) -> RenderResult:
    """Instantiate, trace, render, and summarize one model.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Output directory.
    dry_run:
        Validate only when true.
    file_format:
        TorchLens render file format.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    vis_options:
        Extra keyword arguments forwarded to ``Trace.draw`` so reruns can
        restyle the gallery. Explicit visualization defaults (legend, save-only,
        output path, and file format) always win over passthrough values.

    Returns
    -------
    RenderResult
        Model result.
    """

    draw_kwargs = dict(vis_options or {})

    start = time.monotonic()
    plan = dependency_plan(row)
    skip_reason = unrenderable_reason(row)
    if skip_reason is not None:
        return RenderResult(
            row.name,
            row.model_id,
            f"skipped:{skip_reason}",
            0,
            "",
            time.monotonic() - start,
            plan.cluster_key,
            skip_reason,
        )
    try:
        input_tensor = (
            classics_example_input(row)
            if is_classics_row(row)
            else tensor_for_recipe(row.input_shape, row.input_dtype)
        )
    except Exception as error:
        return RenderResult(
            row.name,
            row.model_id,
            "skipped:unsupported_input_recipe",
            0,
            "",
            time.monotonic() - start,
            plan.cluster_key,
            str(error),
        )
    if dry_run:
        return RenderResult(
            row.name,
            row.model_id,
            "skipped:dry_run",
            0,
            "",
            time.monotonic() - start,
            plan.cluster_key,
            "validated recipe",
        )

    import torch
    import torchlens as tl

    torch.set_num_threads(1)
    model = instantiate_model(row)
    model.eval()
    render_stem = model_render_stem(row, out_dir)
    render_stem.parent.mkdir(parents=True, exist_ok=True)

    def attempt_render(attempt_model: Any, attempt_input: Any, actual_device: str) -> RenderResult:
        """Trace and render the model on one resolved device.

        Parameters
        ----------
        attempt_model:
            Model prepared for the attempt device.
        attempt_input:
            Example input prepared for the attempt device.
        actual_device:
            Device used by this attempt.

        Returns
        -------
        RenderResult
            Successful render result.
        """

        with torch.no_grad():
            trace = tl.trace(
                attempt_model,
                attempt_input,
                layers_to_save=None,
                save=None,
                save_rng_states=False,
                inference_only=True,
            )
        graph_shape_hash = str(getattr(trace, "graph_shape_hash", "") or "")
        n_nodes = len(getattr(trace, "layer_logs", {}) or {})
        draw_call_kwargs = {
            "vis_save_only": True,
            "show_legend": False,
            **draw_kwargs,
            "vis_outpath": str(render_stem),
            "vis_fileformat": file_format,
        }
        trace.draw(**draw_call_kwargs)
        render_path = model_render_path(row, out_dir, file_format)
        if is_featured(row):
            link_featured_copy(row, render_path, out_dir)
        del trace
        return RenderResult(
            row.name,
            row.model_id,
            "rendered",
            n_nodes,
            str(render_path),
            time.monotonic() - start,
            plan.cluster_key,
            device_note(device, actual_device),
            graph_shape_hash,
        )

    if device == "cuda":
        try:
            model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
            return attempt_render(model, input_tensor, "cuda")
        except Exception as error:
            raise RuntimeError(f"device=cuda; {error!r}") from error
    if device == "auto":
        try:
            return attempt_render(model, input_tensor, "cpu")
        except Exception as error:
            if not is_device_related_error(error) or not cuda_is_available():
                raise RuntimeError(f"device=cpu; {error!r}") from error
            try:
                model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
                return attempt_render(model, input_tensor, "cuda")
            except Exception as cuda_error:
                raise RuntimeError(f"device=cuda; {cuda_error!r}") from cuda_error
    return attempt_render(model, input_tensor, "cpu")


def catalog_row_from_payload(payload: Mapping[str, Any]) -> CatalogRow:
    """Build a catalog row from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible row payload.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=int(payload["model_id"]),
        name=str(payload["name"]),
        family=str(payload["family"]),
        family_normalized=str(payload["family_normalized"]),
        domain=str(payload["domain"]),
        zoo=str(payload["zoo"]),
        constructor_call=str(payload["constructor_call"]),
        input_shape=str(payload["input_shape"]),
        input_dtype=str(payload["input_dtype"]),
        era=str(payload["era"]),
        verified=bool(payload["verified"]),
        notes=str(payload["notes"]),
    )


def render_result_from_payload(payload: Mapping[str, Any]) -> RenderResult:
    """Build a render result from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible result payload.

    Returns
    -------
    RenderResult
        Render result.
    """

    return RenderResult(
        name=str(payload["name"]),
        model_id=int(payload["model_id"]),
        status=str(payload["status"]),
        n_nodes=int(payload["n_nodes"]),
        render_path=str(payload["render_path"]),
        elapsed=float(payload["elapsed"]),
        dependency_cluster=str(payload["dependency_cluster"]),
        error=str(payload["error"]),
        graph_shape_hash=str(payload.get("graph_shape_hash", "")),
    )


def render_with_timeout(
    row: CatalogRow,
    out_dir: Path,
    dry_run: bool,
    file_format: str,
    device: str,
    timeout_sec: float,
    vis_options: Sequence[str] = (),
    tmp_dir: Path | None = None,
) -> RenderResult:
    """Run one render in an isolated child process with a timeout.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Output directory.
    dry_run:
        Validate only when true.
    file_format:
        TorchLens render file format.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    timeout_sec:
        Maximum wall time in seconds.
    vis_options:
        Raw ``KEY=VALUE`` strings forwarded to the worker as repeated
        ``--vis-option`` arguments so reruns can restyle the gallery.
    tmp_dir:
        Optional per-model temporary directory routed to the worker via the
        ``TMPDIR``/``TEMP``/``TMP`` environment variables. Passed through the
        subprocess environment (not process globals) so concurrent workers each
        get an isolated scratch directory without mutating shared state.

    Returns
    -------
    RenderResult
        Render result.
    """

    plan = dependency_plan(row)
    command = [
        sys.executable,
        "-m",
        "menagerie.generate_menagerie",
        "--worker-row-json",
        json.dumps(row.__dict__),
        "--out-dir",
        str(out_dir),
        "--file-format",
        file_format,
        "--device",
        device,
    ]
    for pair in vis_options:
        command.extend(("--vis-option", pair))
    if dry_run:
        command.append("--dry-run")
    child_env = None
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ)
        for key in ("TMPDIR", "TEMP", "TMP"):
            child_env[key] = str(tmp_dir)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return RenderResult(
            row.name,
            row.model_id,
            "failed:timeout",
            0,
            "",
            timeout_sec,
            plan.cluster_key,
            f"timed out after {timeout_sec:.1f}s",
        )
    if completed.returncode != 0:
        stderr_tail = " | ".join(completed.stderr.strip().splitlines()[-5:])
        return RenderResult(
            row.name,
            row.model_id,
            "failed:worker_exit",
            0,
            "",
            0.0,
            plan.cluster_key,
            stderr_tail or f"worker exited with code {completed.returncode}",
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "worker_result":
            return render_result_from_payload(payload["result"])
    return RenderResult(
        row.name,
        row.model_id,
        "failed:worker_protocol",
        0,
        "",
        0.0,
        plan.cluster_key,
        "worker did not emit a worker_result event",
    )


def selected_render_exists(row: CatalogRow, out_dir: Path, file_format: str) -> bool:
    """Return whether the organized render file exists for one row.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Output directory.
    file_format:
        Render file format.

    Returns
    -------
    bool
        Whether the render exists.
    """

    return model_render_path(row, out_dir, file_format).exists()


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
    if args.model_id:
        model_ids = set(args.model_id)
        rows = [row for row in rows if row.model_id in model_ids]
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


def relative_markdown_link(from_path: Path, target: Path, label: str) -> str:
    """Build a relative Markdown link.

    Parameters
    ----------
    from_path:
        Markdown file path.
    target:
        Link target path.
    label:
        Link label.

    Returns
    -------
    str
        Markdown link.
    """

    relpath = os.path.relpath(target, from_path.parent)
    return f"[{label}]({relpath})"


def era_year(era: str) -> int | None:
    """Extract the earliest plausible year from an era string.

    Parameters
    ----------
    era:
        Catalog era string.

    Returns
    -------
    int | None
        Year or ``None``.
    """

    years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", era)]
    return min(years) if years else None


def domain_index_path(out_dir: Path, domain: str) -> Path:
    """Return the index path for a domain.

    Parameters
    ----------
    out_dir:
        Output directory.
    domain:
        Domain name.

    Returns
    -------
    Path
        Domain index path.
    """

    return out_dir / safe_path_part(domain) / "INDEX.md"


def family_index_path(out_dir: Path, domain: str, family: str) -> Path:
    """Return the index path for a domain/family pair.

    Parameters
    ----------
    out_dir:
        Output directory.
    domain:
        Domain name.
    family:
        Family name.

    Returns
    -------
    Path
        Family index path.
    """

    return out_dir / safe_path_part(domain) / safe_path_part(family) / "INDEX.md"


def write_text(path: Path, text: str) -> None:
    """Write text to a path, creating parent directories.

    Parameters
    ----------
    path:
        Output path.
    text:
        Text body.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_link_for_row(
    row: CatalogRow,
    out_dir: Path,
    file_format: str,
    from_path: Path,
    records: Mapping[str, Mapping[str, str]],
) -> str:
    """Return a render link or status label for one row.

    Parameters
    ----------
    row:
        Catalog row.
    out_dir:
        Output directory.
    file_format:
        Render file format.
    from_path:
        Markdown file path.
    records:
        Manifest records keyed by model name.

    Returns
    -------
    str
        Render link or status label.
    """

    render_path = model_render_path(row, out_dir, file_format)
    if render_path.exists():
        return relative_markdown_link(from_path, render_path, "graph")
    status = records.get(row.name, {}).get("status", "not rendered")
    return status


def write_family_index(
    out_dir: Path,
    domain: str,
    family: str,
    rows: Sequence[CatalogRow],
    file_format: str,
    records: Mapping[str, Mapping[str, str]],
) -> None:
    """Write one family member index.

    Parameters
    ----------
    out_dir:
        Output directory.
    domain:
        Domain name.
    family:
        Family name.
    rows:
        Family rows.
    file_format:
        Render file format.
    records:
        Manifest records keyed by model name.
    """

    index_path = family_index_path(out_dir, domain, family)
    rendered = sum(model_render_path(row, out_dir, file_format).exists() for row in rows)
    lines = [
        f"# {family}",
        "",
        f"Domain: `{domain}`",
        f"Models: {len(rows)} catalog rows, {rendered} rendered.",
        "",
        "| Model | Zoo | Era | Status |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: (item.name.lower(), item.zoo.lower())):
        status = render_link_for_row(row, out_dir, file_format, index_path, records)
        lines.append(f"| {row.name} | {row.zoo} | {row.era} | {status} |")
    write_text(index_path, "\n".join(lines) + "\n")


def write_domain_index(
    out_dir: Path,
    domain: str,
    family_rows: Mapping[str, Sequence[CatalogRow]],
    file_format: str,
) -> None:
    """Write one domain index linking family indexes.

    Parameters
    ----------
    out_dir:
        Output directory.
    domain:
        Domain name.
    family_rows:
        Rows grouped by family.
    file_format:
        Render file format.
    """

    index_path = domain_index_path(out_dir, domain)
    lines = [
        f"# {domain}",
        "",
        f"Families: {len(family_rows)}.",
        "",
        "| Family | Models | Rendered |",
        "| --- | ---: | ---: |",
    ]
    for family, rows in sorted(family_rows.items(), key=lambda item: item[0].lower()):
        rendered = sum(model_render_path(row, out_dir, file_format).exists() for row in rows)
        link = relative_markdown_link(
            index_path, family_index_path(out_dir, domain, family), family
        )
        lines.append(f"| {link} | {len(rows)} | {rendered} |")
    write_text(index_path, "\n".join(lines) + "\n")


def write_featured_index(
    out_dir: Path,
    rows: Sequence[CatalogRow],
    file_format: str,
    records: Mapping[str, Mapping[str, str]],
) -> None:
    """Write the featured hall-of-fame index.

    Parameters
    ----------
    out_dir:
        Output directory.
    rows:
        Catalog rows.
    file_format:
        Render file format.
    records:
        Manifest records keyed by model name.
    """

    index_path = out_dir / "FEATURED.md"
    featured = [row for row in rows if is_featured(row)]
    rendered = [row for row in featured if model_render_path(row, out_dir, file_format).exists()]
    lines = [
        "# Featured Models",
        "",
        "A one-click tier for canonical architectures and core zoo entries.",
        "",
        f"Featured catalog rows: {len(featured)}. Rendered featured rows: {len(rendered)}.",
        "",
        "| Model | Family | Domain | Zoo | Reason | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(featured, key=lambda item: (item.name.lower(), item.zoo.lower()))
    for row in ordered:
        status = render_link_for_row(row, out_dir, file_format, index_path, records)
        lines.append(
            f"| {row.name} | {row.family_normalized} | {row.domain} | {row.zoo} | "
            f"{featured_reason(row)} | {status} |"
        )
    write_text(index_path, "\n".join(lines) + "\n")


def write_html_index(out_dir: Path, domain_counts: Counter[str]) -> None:
    """Write a small optional HTML landing page.

    Parameters
    ----------
    out_dir:
        Output directory.
    domain_counts:
        Catalog row counts by domain.
    """

    links = "\n".join(
        f'<li><a href="{html.escape(safe_path_part(domain))}/INDEX.md">'
        f"{html.escape(domain)}</a> <span>{count}</span></li>"
        for domain, count in sorted(domain_counts.items(), key=lambda item: item[0].lower())
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TorchLens Menagerie</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; margin: 2rem; max-width: 920px; }}
    a {{ color: #0645ad; }}
    li {{ margin: .35rem 0; }}
    span {{ color: #666; }}
  </style>
</head>
<body>
  <h1>TorchLens Menagerie</h1>
  <p><a href="FEATURED.md">Featured hall of fame</a></p>
  <ul>{links}</ul>
</body>
</html>
"""
    write_text(out_dir / "index.html", body)


def build_indexes(
    rows: Sequence[CatalogRow], out_dir: Path, manifest_path: Path, file_format: str
) -> None:
    """Build all browsable gallery indexes.

    Parameters
    ----------
    rows:
        Catalog rows to index.
    out_dir:
        Output directory.
    manifest_path:
        Manifest path.
    file_format:
        Render file format.
    """

    records = manifest_records(manifest_path)
    by_domain: dict[str, dict[str, list[CatalogRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_domain[row.domain][row.family_normalized].append(row)

    for domain, family_rows in by_domain.items():
        for family, family_members in family_rows.items():
            write_family_index(out_dir, domain, family, family_members, file_format, records)
        write_domain_index(out_dir, domain, family_rows, file_format)

    total_rendered = sum(model_render_path(row, out_dir, file_format).exists() for row in rows)
    status_counts = Counter(row.get("status", "unknown") for row in records.values())
    domain_counts = Counter(row.domain for row in rows)
    family_counts = Counter(row.family_normalized for row in rows)
    years = Counter(year for row in rows if (year := era_year(row.era)) is not None)
    index_path = out_dir / "INDEX.md"
    lines = [
        "# TorchLens Menagerie",
        "",
        "A browsable atlas of model graphs rendered with TorchLens.",
        "",
        f"Catalog rows indexed: {len(rows)}.",
        f"Distinct families indexed: {len(family_counts)}.",
        f"Rendered graphs present: {total_rendered}.",
        "",
        f"Featured hall of fame: {relative_markdown_link(index_path, out_dir / 'FEATURED.md', 'FEATURED.md')}",
        "",
        "## Domains",
        "",
        "| Domain | Families | Models | Rendered |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain, count in sorted(domain_counts.items(), key=lambda item: item[0].lower()):
        families = len(by_domain[domain])
        rendered = sum(
            model_render_path(row, out_dir, file_format).exists()
            for family_rows in by_domain[domain].values()
            for row in family_rows
        )
        link = relative_markdown_link(index_path, domain_index_path(out_dir, domain), domain)
        lines.append(f"| {link} | {families} | {count} | {rendered} |")
    if status_counts:
        lines.extend(["", "## Manifest Status", "", "| Status | Count |", "| --- | ---: |"])
        for status, count in status_counts.most_common():
            lines.append(f"| {status} | {count} |")
    if years:
        lines.extend(["", "## Era Timeline", "", "| Year | Models |", "| ---: | ---: |"])
        for year in sorted(years):
            lines.append(f"| {year} | {years[year]} |")
    write_text(index_path, "\n".join(lines) + "\n")
    write_featured_index(out_dir, rows, file_format, records)
    write_html_index(out_dir, domain_counts)


def run(args: argparse.Namespace) -> int:
    """Run the dependency-aware disk-safe renderer.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    out_dir = args.out_dir.resolve()
    manifest_path = (args.manifest or out_dir / "manifest.tsv").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_rows(args)
    if args.index_only:
        build_indexes(selected, out_dir, manifest_path, args.file_format)
        log_event("index_done", rows=len(selected), out_dir=str(out_dir))
        return 0

    run_cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
    start_free_gb = disk_free_gb(out_dir)
    log_event("run_start", out_dir=str(out_dir), free_gb=round(start_free_gb, 3))
    assert_min_free(out_dir, args.min_free_gb)

    done = set() if args.force else completed_names(manifest_path, args.retry_failed)
    rows = [row for row in selected if row.name not in done]
    if args.only_new and not args.force:
        rows = [
            row
            for row in rows
            if not selected_render_exists(row, out_dir, args.file_format)
            and manifest_records(manifest_path).get(row.name, {}).get("status") != "rendered"
        ]
    log_event("selected", count=len(rows), skipped_existing=len(selected) - len(rows))

    vis_options = parse_vis_options(args.vis_option)
    if vis_options:
        log_event("vis_options", options={key: str(value) for key, value in vis_options.items()})

    # Phase 1: install dependencies per cluster (serial -- installs mutate the
    # shared interpreter/site-packages and must precede their rows). Clusters
    # whose dependencies are unavailable are recorded directly to the manifest.
    runnable: list[tuple[DependencyPlan, CatalogRow]] = []
    for plan, cluster_rows in group_by_dependency(rows):
        install_error = install_dependency_plan(plan, args)
        if install_error is not None:
            for row in cluster_rows:
                append_manifest(
                    manifest_path,
                    RenderResult(
                        row.name,
                        row.model_id,
                        "skipped:dependency_unavailable",
                        0,
                        "",
                        0.0,
                        plan.cluster_key,
                        install_error,
                    ),
                )
            log_event(
                "cluster_skipped",
                cluster=plan.cluster_key,
                count=len(cluster_rows),
                error=install_error,
            )
            continue
        runnable.extend((plan, row) for row in cluster_rows)

    # Phase 2: render runnable rows concurrently. Each model already runs in an
    # isolated child process (``render_with_timeout``); threads here just dispatch
    # and await those subprocesses. The GPU semaphore caps in-flight jobs when a
    # device that may use CUDA is selected. The main thread does ALL manifest
    # appends and disk bookkeeping single-threaded as futures complete.
    jobs = max(1, args.jobs)
    use_gpu_cap = args.device in {"cuda", "auto"}
    gpu_jobs = max(1, args.gpu_jobs)
    effective_jobs = min(jobs, gpu_jobs) if use_gpu_cap else jobs
    gpu_semaphore = threading.Semaphore(gpu_jobs) if use_gpu_cap else None

    def process_one(plan: DependencyPlan, row: CatalogRow) -> tuple[RenderResult, int]:
        """Render one row in a worker thread and clean up its scratch state.

        Parameters
        ----------
        plan:
            Dependency plan for the row's cluster.
        row:
            Catalog row to render.

        Returns
        -------
        tuple[RenderResult, int]
            Render result and the number of new cache entries removed.
        """

        cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
        tmp_dir = out_dir / "_tmp" / f"{row.model_id:05d}_{safe_path_part(row.name)}"
        gate: ContextManager[Any] = gpu_semaphore if gpu_semaphore is not None else nullcontext()
        with gate:
            result = render_with_timeout(
                row,
                out_dir,
                args.dry_run,
                args.file_format,
                args.device,
                args.timeout_sec,
                vis_options=args.vis_option,
                tmp_dir=tmp_dir,
            )
            removed = 0 if args.keep_cache else cleanup_runtime(cache_snapshots, tmp_dir)
        return result, removed

    previous_free_gb = start_free_gb
    downward_steps = 0
    processed = 0
    total = len(runnable)
    log_event(
        "parallel_start",
        jobs=jobs,
        effective_jobs=effective_jobs,
        gpu_jobs=gpu_jobs if use_gpu_cap else None,
        device=args.device,
        rows=total,
    )

    if total:
        try:
            assert_min_free(out_dir, args.min_free_gb)
        except RuntimeError:
            for snapshot in run_cache_snapshots:
                purge_new_cache_entries(snapshot)
            assert_min_free(out_dir, args.min_free_gb)

    with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
        futures: dict[Future[tuple[RenderResult, int]], tuple[DependencyPlan, CatalogRow]] = {}
        for plan, row in runnable:
            before_free_gb = disk_free_gb(out_dir)
            log_event(
                "model_start",
                name=row.name,
                cluster=plan.cluster_key,
                free_gb=round(before_free_gb, 3),
            )
            futures[executor.submit(process_one, plan, row)] = (plan, row)

        for future in as_completed(futures):
            plan, row = futures[future]
            processed += 1
            result, removed = future.result()
            append_manifest(manifest_path, result)
            after_free_gb = disk_free_gb(out_dir)
            if after_free_gb < previous_free_gb - args.drift_tolerance_gb:
                downward_steps += 1
            else:
                downward_steps = 0
            previous_free_gb = after_free_gb
            log_event(
                "model_done",
                index=processed,
                total=total,
                name=row.name,
                status=result.status,
                n_nodes=result.n_nodes,
                cache_entries_removed=removed,
                after_free_gb=round(after_free_gb, 3),
                elapsed=round(result.elapsed, 3),
                error=result.error,
            )
            if processed % args.disk_log_every == 0:
                log_event(
                    "disk_delta",
                    index=processed,
                    start_free_gb=round(start_free_gb, 3),
                    current_free_gb=round(after_free_gb, 3),
                    delta_gb=round(after_free_gb - start_free_gb, 3),
                )
            # Periodic disk-safety check: free space should not run dry as the
            # batch progresses, and a sustained monotonic decline aborts the run.
            try:
                assert_min_free(out_dir, args.min_free_gb)
            except RuntimeError:
                for snapshot in run_cache_snapshots:
                    purge_new_cache_entries(snapshot)
                assert_min_free(out_dir, args.min_free_gb)
            if downward_steps >= args.max_monotonic_down_steps:
                raise RuntimeError(
                    "free disk is drifting downward monotonically; aborting to protect disk "
                    f"(steps={downward_steps}, tolerance={args.drift_tolerance_gb} GiB)"
                )

    if not args.skip_index:
        build_indexes(selected, out_dir, manifest_path, args.file_format)
    final_free_gb = disk_free_gb(out_dir)
    log_event(
        "run_done",
        processed=processed,
        start_free_gb=round(start_free_gb, 3),
        final_free_gb=round(final_free_gb, 3),
        delta_gb=round(final_free_gb - start_free_gb, 3),
        manifest=str(manifest_path),
    )
    return 0


def default_jobs() -> int:
    """Return the default concurrency level for the render/validate engines.

    Returns
    -------
    int
        ``min(8, os.cpu_count() - 2)`` clamped to at least one worker.
    """

    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 2))


def build_parser() -> argparse.ArgumentParser:
    """Build the generator CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, help="process the first N rows after filters")
    parser.add_argument("--family")
    parser.add_argument("--domain")
    parser.add_argument("--zoo")
    parser.add_argument("--name", action="append", help="case-insensitive model-name substring")
    parser.add_argument("--model-id", action="append", type=int, help="exact catalog model id")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--featured-only", action="store_true")
    parser.add_argument(
        "--since", type=int, help="only process rows with model_id greater than this"
    )
    parser.add_argument("--only-new", action="store_true", help="skip rows with rendered files")
    parser.add_argument(
        "--retry-failed", action="store_true", help="retry non-rendered manifest rows"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render even when an output file already exists (overwrite); "
        "needed to regenerate the whole gallery with new aesthetics",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="number of models to render concurrently (each in its own subprocess)",
    )
    parser.add_argument(
        "--gpu-jobs",
        type=int,
        default=4,
        help="max concurrent in-flight jobs when --device is cuda/auto (GPU OOM guard)",
    )
    parser.add_argument(
        "--vis-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Trace.draw() keyword argument (repeatable); VALUE parsed as "
        "bool/int/float/str, e.g. --vis-option order_siblings=True",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--db", type=Path, default=Path(__file__).resolve().parent / "data" / "catalog.db"
    )
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--file-format", default="svg")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--install-timeout", type=float, default=600.0)
    parser.add_argument(
        "--pip-args", action="append", default=[], help="extra argument for pip install"
    )
    parser.add_argument("--install-deps", dest="install_deps", action="store_true", default=True)
    parser.add_argument("--no-install-deps", dest="install_deps", action="store_false")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--disk-log-every", type=int, default=10)
    parser.add_argument("--drift-tolerance-gb", type=float, default=0.25)
    parser.add_argument("--max-monotonic-down-steps", type=int, default=10)
    parser.add_argument("--worker-row-json", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator CLI.

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
    # Pin BLAS/OMP threads to 1: many render workers run concurrently, the trace runs on GPU, and graphviz
    # layout is single-threaded, so per-worker multi-threaded BLAS only oversubscribes the CPU. Must precede torch.
    for _thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_var, "1")
    if args.worker_row_json:
        row = catalog_row_from_payload(json.loads(args.worker_row_json))
        try:
            result = render_one(
                row,
                args.out_dir.resolve(),
                args.dry_run,
                args.file_format,
                args.device,
                vis_options=parse_vis_options(args.vis_option),
            )
        except Exception as error:
            plan = dependency_plan(row)
            result = RenderResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                "",
                0.0,
                plan.cluster_key,
                repr(error),
            )
        print(json.dumps({"event": "worker_result", "result": result.__dict__}), flush=True)
        return 0
    try:
        return run(args)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
