"""Dependency-aware, disk-safe renderer for the TorchLens model menagerie."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, Mapping, Sequence

from menagerie.catalog import CatalogRow, catalog_row_from_payload
from menagerie.recipe import (
    build_input_for_row,
    instantiate_model,
    is_classics_row,
)
from menagerie.runtime import (
    CACHE_ROOTS,
    DependencyPlan,
    assert_min_free,
    cleanup_runtime,
    combine_notes,  # noqa: F401  # legacy re-export (module-split import surface)
    cuda_is_available,
    default_jobs,
    dependency_plan,
    device_note,
    disk_free_gb,
    group_by_dependency,
    install_dependency_plan,
    is_device_related_error,
    featured_reason,
    is_featured,
    log_event,
    move_model_and_input_to_device,
    purge_new_cache_entries,
    safe_path_part,
    select_rows,
    snapshot_cache,
    unrenderable_reason,
)
from menagerie.worker_subprocess import run_worker_subprocess


DEFAULT_OUT_DIR = Path("/tmp/torchlens_menagerie_gallery")
MANIFEST_COLUMNS = (
    "name",
    "model_id",
    "stable_id",
    "recipe_revision_sha256",
    "status",
    "n_nodes",
    "render_path",
    "elapsed",
    "dependency_cluster",
    "error",
    "graph_shape_hash",
    "visual_mode",
)


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
    stable_id:
        Opaque durable model identity.
    recipe_revision_sha256:
        Frozen recipe fingerprint for the row's current construction recipe.
    visual_mode:
        Compact rendering mode derived from forwarded visualization options.
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
    stable_id: str = ""
    recipe_revision_sha256: str = ""
    visual_mode: str = "default"


def visual_mode_from_options(options: Mapping[str, Any] | None) -> str:
    """Return a compact visual-mode label for render manifests.

    Parameters
    ----------
    options:
        Trace.draw keyword arguments.

    Returns
    -------
    str
        Visual mode label.
    """

    draw_options = dict(options or {})
    collapse = draw_options.get("collapse")
    if collapse not in (None, False, "", "none"):
        return f"collapsed:{collapse}"
    view = draw_options.get("view")
    if view not in (None, False, "", "default"):
        return str(view)
    if draw_options.get("roll") is True or draw_options.get("rolled") is True:
        return "rolled"
    if draw_options.get("unroll") is True or draw_options.get("unrolled") is True:
        return "unrolled"
    return "default"


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


def load_smoke_vis_options(smoke_manifest: Path | None) -> dict[str, list[str]]:
    """Return per-stable-id ``--vis-option`` strings from a smoke manifest.

    Each smoke manifest row may carry a ``vis_option`` object of draw() keyword
    arguments specific to that case (e.g. ``{"view": "rolled"}``). They are
    converted to ``KEY=VALUE`` strings so the render worker applies the right
    visual mode per model rather than one global mode for the whole batch.

    Parameters
    ----------
    smoke_manifest:
        Optional smoke-manifest JSONL path.

    Returns
    -------
    dict[str, list[str]]
        Mapping of stable ID to ``KEY=VALUE`` vis-option strings.
    """

    if smoke_manifest is None or not Path(smoke_manifest).exists():
        return {}
    per_row: dict[str, list[str]] = {}
    with Path(smoke_manifest).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            stable_id = row.get("stable_id")
            vis_option = row.get("vis_option")
            if not stable_id or not isinstance(vis_option, dict) or not vis_option:
                continue
            per_row[str(stable_id)] = [f"{key}={value}" for key, value in vis_option.items()]
    return per_row


def manifest_records(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read the latest manifest record for each stable model identity.

    Parameters
    ----------
    manifest_path:
        Manifest TSV path.

    Returns
    -------
    dict[str, dict[str, str]]
        Latest manifest rows keyed by stable ID.
    """

    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["stable_id"]: row for row in reader if row.get("stable_id")}


def completed_stable_ids(manifest_path: Path, retry_failed: bool) -> set[str]:
    """Read stable IDs already completed in an append-only manifest.

    Parameters
    ----------
    manifest_path:
        Manifest TSV path.
    retry_failed:
        Whether failed and skipped rows should be retried.

    Returns
    -------
    set[str]
        Completed stable IDs.
    """

    records = manifest_records(manifest_path)
    if not retry_failed:
        return set(records)
    return {stable_id for stable_id, row in records.items() if row.get("status") == "rendered"}


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
                result.stable_id,
                result.recipe_revision_sha256,
                result.status,
                result.n_nodes,
                result.render_path,
                f"{result.elapsed:.3f}",
                result.dependency_cluster,
                result.error,
                result.graph_shape_hash,
                result.visual_mode,
            )
        )


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
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    try:
        input_tensor = build_input_for_row(row)
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
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
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
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
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
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
            visual_mode=visual_mode_from_options(draw_kwargs),
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
        stable_id=str(payload.get("stable_id", "")),
        recipe_revision_sha256=str(payload.get("recipe_revision_sha256", "")),
        visual_mode=str(payload.get("visual_mode", "default")),
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
        json.dumps(asdict(row)),
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
    # Run the worker in its OWN session/process group so that on timeout we can
    # kill the entire group. The worker spawns graphviz children (``dot`` /
    # ``neato``) for layout; killing only the direct child can orphan those
    # grandchildren on pathologically dense graphs.
    completed = run_worker_subprocess(
        command,
        env=child_env,
        timeout_sec=timeout_sec,
    )
    if completed.timed_out:
        return RenderResult(
            row.name,
            row.model_id,
            "failed:timeout",
            0,
            "",
            timeout_sec,
            plan.cluster_key,
            f"timed out after {timeout_sec:.1f}s",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
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
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
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
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
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
        Manifest records keyed by stable ID.

    Returns
    -------
    str
        Render link or status label.
    """

    render_path = model_render_path(row, out_dir, file_format)
    if render_path.exists():
        return relative_markdown_link(from_path, render_path, "graph")
    status = records.get(row.stable_id, {}).get("status", "not rendered")
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

    done = set() if args.force else completed_stable_ids(manifest_path, args.retry_failed)
    rows = [row for row in selected if row.stable_id not in done]
    if args.only_new and not args.force:
        rows = [
            row
            for row in rows
            if not selected_render_exists(row, out_dir, args.file_format)
            and manifest_records(manifest_path).get(row.stable_id, {}).get("status") != "rendered"
        ]
    log_event("selected", count=len(rows), skipped_existing=len(selected) - len(rows))

    vis_options = parse_vis_options(args.vis_option)
    if vis_options:
        log_event("vis_options", options={key: str(value) for key, value in vis_options.items()})
    smoke_vis_options = load_smoke_vis_options(getattr(args, "smoke_manifest", None))

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
                        stable_id=row.stable_id,
                        recipe_revision_sha256=row.recipe_revision_sha256,
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
        # Per-row vis-options (e.g. a recurrent model rendered rolled) override the
        # global ones for that stable id; later KEY=VALUE entries win in the worker.
        row_vis_options = list(args.vis_option) + smoke_vis_options.get(row.stable_id, [])
        with gate:
            result = render_with_timeout(
                row,
                out_dir,
                args.dry_run,
                args.file_format,
                args.device,
                args.timeout_sec,
                vis_options=row_vis_options,
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
    parser.add_argument(
        "--names-file", help="render only models whose exact name (one per line) is in this file"
    )
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
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        help="optional smoke-manifest JSONL providing per-stable-id vis_option overrides",
    )
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
        # Per-worker address-space backstop: a pathological model that tries to
        # allocate a runaway amount fails its OWN worker with a clean MemoryError
        # (-> failed:exception) instead of triggering a global OOM that the kernel
        # resolves by SIGKILLing an innocent sibling worker. Set generously (0.8x
        # total RAM) so normal models -- even large ones -- never false-trip; it
        # only catches genuine runaway allocation.
        try:
            import resource as _resource

            _total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            _cap = int(_total_ram * 0.8)
            _soft, _hard = _resource.getrlimit(_resource.RLIMIT_AS)
            _new_hard = _cap if _hard == _resource.RLIM_INFINITY else min(_cap, _hard)
            _resource.setrlimit(_resource.RLIMIT_AS, (_cap, _new_hard))
        except (ValueError, OSError, AttributeError):
            pass
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
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
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
