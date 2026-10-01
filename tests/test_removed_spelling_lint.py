"""Repo-wide tripwire: no removed spelling creeps back in as a taught spelling.

The 2026-08 shim-removal pass (RULINGS Batch-8: FULL deletion, clean v2)
deleted every runtime deprecation shim, warned alias, alias property, and
moved-name redirect. ``tests/test_deprecation_inventory.py`` pins the PACKAGE
deprecation-free (AST scanners over shipped code); THIS module pins the REPO
free of the deleted spellings themselves -- code, docs, notebooks, examples,
benchmarks, scripts -- so a stale example or a resurrected alias cannot creep
back after the sweep (the S02 review found 20 such residues; its closure
greps are the method this lint freezes).

Scope boundary (mirrors the removal ruling):

- Torch-version compatibility (``HAS_*`` flags) and artifact-format
  compatibility (tlspec floors, legacy-save loading, pickle rename maps such
  as the ``TensorLog`` module attribute) are NOT shims and are not matched.
- Method-level ``vis_*`` implementation params on ``draw``/``draw_backward``/
  ``draw_combined`` (``vis_mode``, ``vis_outpath``, ``vis_node_mode``,
  ``vis_buffers``, ...) are the KEPT non-warning spellings owned by the
  naming session -- deliberately absent from the forbidden set.
- The allowlist below is a LEDGER: every entry names an audited legitimate
  reference (history record, removal-record prose, a ``pytest.raises``
  absence pin, or the deprecation scanners themselves). An entry whose file
  no longer matches goes STALE and fails the lint until the row is pruned.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SCAN_SUFFIXES = (".py", ".md", ".ipynb", ".rst", ".txt")

#: Forbidden-spelling groups. Every pattern targets a spelling the shim
#: removal DELETED, written to miss the surviving canonical spellings
#: (``torchlens.validation.validate_forward_pass`` does not match the
#: ``tl_moved`` row for ``validate_forward_pass`` because the qualifier
#: requires ``tl.``/``torchlens.`` immediately before the name).
_FORBIDDEN: dict[str, re.Pattern[str]] = {
    # The deleted deprecation machinery itself.
    "machinery": re.compile(r"\bwarn_deprecated_alias\b|\bTorchLensDeprecationWarning\b"),
    # Paper-era 1.x API names (fully removed; no canonical home keeps them).
    "paper_era": re.compile(
        r"\b(?:log_forward_pass|validate_model_activations|validate_saved_activations"
        r"|draw_model_graph|render_model_graph|ModelHistory|get_model_structure"
        r"|show_model_structure)\b"
    ),
    # Removed top-level spellings: the name is only illegal QUALIFIED at the
    # root (its canonical home is a submodule, or it was renamed outright).
    "tl_moved": re.compile(
        r"\b(?:tl|torchlens)\.(?:peek|batched_extract|record_span|intervening|replay"
        r"|replay_from|rerun|summary|show_model_graph|draw_backward|draw_combined"
        r"|validate_forward_pass|validate_backward_pass|validate_saved_outs"
        r"|load_intervention_spec|StreamingOptions|VisualizationOptions|wrap_torch"
        r"|unwrap_torch|wrapped|get_model_metadata|check_spec_compat"
        r"|check_metadata_invariants|resolve_sites|preview_fastlog"
        r"|ModuleInputSnapshot|PreHookEffect|TensorInputObservation|TraceState"
        r"|SaveLevel)\b"
    ),
    # Removed Trace/Bundle method spellings (canonical: push/push_from/run/
    # span/conditional_arm_entry_edges). ``validate_saved_outs`` is absent on
    # purpose: the INTERNAL ``validation.core.validate_saved_outs`` keeps the
    # name (its zero-reference ``validate_trace_saved_outs`` rename re-export
    # was deleted by the F38 punch list); only the top-level spelling was
    # removed, and the ``tl_moved`` row covers it.
    "methods": re.compile(
        r"\.(?:replay|replay_from|rerun|record_span"
        r"|conditional_then_entry_edges|conditional_elif_entry_edges"
        r"|conditional_else_entry_edges)\s*\("
    ),
    # Flat capture/save kwargs on a single-line trace() call. Grouped-option
    # reuse of the same field names is legal, so a line that spells a grouped
    # bundle (``...Options(``, ``options.``) is exempt; multi-line calls are
    # out of reach for a line lint and are covered by the runtime TypeError.
    "trace_flat": re.compile(
        r"(?:tl|torchlens)\.trace\s*\([^)\n]*\b(?:layers_to_save|vis_mode|vis_opt"
        r"|intervention_ready|backward_ready|output_style|output_head|random_seed"
        r"|save_grads|structure_only|save_outs_to|out_sink|keep_outs_in_memory"
        r"|activation_transform|save_arg_values|capture_container_structure)\s*="
    ),
    # Removed record()/dry_run()/Recorder() predicate aliases.
    "keep_op": re.compile(r"\b(?:record|dry_run|Recorder)\s*\([^)\n]*\bkeep_(?:op|module)\s*="),
    # Crawler-era no-op wrap kwargs and stub functions.
    "patch": re.compile(
        r"\bpatch_detached_references\b"
        r"|wrap_torch\s*\([^)\n]*\b(?:patch_policy|patch_modules)\s*="
    ),
    # Inert backward-perturbation flag.
    "perturb": re.compile(r"\bperturb_saved_grads\b"),
    # Removed draw()/show() warned alias (vis_mode is the kept method param).
    "vis_opt": re.compile(r"\bvis_opt\s*="),
    # Removed legacy bool VALUES for buffer visibility (tri-state only).
    "buffers_bool": re.compile(r"\bshow_buffers\s*=\s*(?:True|False)\b"),
    # Removed 'vision'/'attention' node-style presets (canonical:
    # node_spec_fn=torchlens.experimental.node_styles.<style>_node_mode).
    "node_style_domain": re.compile(r"\bnode_(?:style|mode)\s*=\s*['\"](?:vision|attention)['\"]"),
    # Removed inert/alias option-constructor kwargs.
    "inert_opts": re.compile(
        r"SaveOptions\s*\([^)\n]*\b(?:output_dir|save_level|bundle_format)\s*="
        r"|ReplayOptions\s*\([^)\n]*\b(?:is_appended|device_override)\s*="
        r"|InterventionOptions\s*\([^)\n]*\b(?:helper_validation|auto_promote"
        r"|cohort_migration|error_severity_threshold)\s*="
        r"|StreamingOptions\s*\([^)\n]*\b(?:save_outs_to|keep_outs_in_memory|out_sink)\s*="
        r"|VisualizationOptions\s*\([^)\n]*\b(?:mode|max_module_depth|layout_engine"
        r"|node_mode)\s*="
    ),
}

#: Lines spelling a grouped-option bundle are exempt from ``trace_flat``.
_GROUPED_OPTION_LINE = re.compile(r"Options\s*\(|options\.")

_TL_MOVED_NEEDLE_NAMES = (
    "peek",
    "batched_extract",
    "record_span",
    "intervening",
    "replay",  # prefix also covers replay_from
    "rerun",
    # "summary" left this needle list 2026-08-26 (megasprint lane A07, summary
    # memo A8): tl.summary is UN-removed as the first-class one-call front
    # door, so the spelling is canonical again.
    "show_model_graph",
    "draw_backward",
    "draw_combined",
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_saved_outs",
    "load_intervention_spec",
    "StreamingOptions",
    "VisualizationOptions",
    "wrap_torch",
    "unwrap_torch",
    "wrapped",
    "get_model_metadata",
    "check_spec_compat",
    "check_metadata_invariants",
    "resolve_sites",
    "preview_fastlog",
    "ModuleInputSnapshot",
    "PreHookEffect",
    "TensorInputObservation",
    "TraceState",
    "SaveLevel",
)

#: Plain-substring prefilter needles per group: a group's regex can only
#: match text containing at least one of its needles (C-level ``in`` scans
#: keep the lint inside the smoke runtime budget; the red-capability test
#: routes through this same filter, so needle/regex drift fails loudly).
_NEEDLES: dict[str, tuple[str, ...]] = {
    "machinery": ("warn_deprecated_alias", "TorchLensDeprecationWarning"),
    "paper_era": (
        "log_forward_pass",
        "validate_model_activations",
        "validate_saved_activations",
        "draw_model_graph",
        "render_model_graph",
        "ModelHistory",
        "get_model_structure",
        "show_model_structure",
    ),
    "tl_moved": tuple(
        f"{qualifier}.{name}"
        for qualifier in ("tl", "torchlens")
        for name in _TL_MOVED_NEEDLE_NAMES
    ),
    "methods": (
        ".replay",
        ".rerun",
        ".record_span",
        ".conditional_then_entry_edges",
        ".conditional_elif_entry_edges",
        ".conditional_else_entry_edges",
    ),
    "trace_flat": ("tl.trace", "torchlens.trace"),
    "keep_op": ("keep_op", "keep_module"),
    "patch": ("patch_detached_references", "patch_policy", "patch_modules"),
    "perturb": ("perturb_saved_grads",),
    "vis_opt": ("vis_opt",),
    "buffers_bool": ("show_buffers",),
    "node_style_domain": ("node_style", "node_mode"),
    "inert_opts": (
        "SaveOptions",
        "ReplayOptions",
        "InterventionOptions",
        "StreamingOptions",
        "VisualizationOptions",
    ),
}

#: The audited-legitimate ledger: path -> (groups | {"ALL"}, reason).
#: Every row must still MATCH (a stale row fails the lint until pruned).
_ALLOWED: dict[str, tuple[frozenset[str], str]] = {
    "CHANGELOG.md": (
        frozenset({"ALL"}),
        "release history; records the old spellings as history by design",
    ),
    "RESULTS.md": (
        frozenset({"ALL"}),
        "self-declared HISTORICAL SNAPSHOT benchmark record",
    ),
    "benchmarks/perf_results_2026-05-14.md": (
        frozenset({"ALL"}),
        "dated benchmark results record (history exemption, S02-fix C3b)",
    ),
    "benchmarks/perf_results_provisional.md": (
        frozenset({"ALL"}),
        "dated benchmark results record (history exemption, S02-fix C3b)",
    ),
    "benchmarks/intervention_overhead_results.md": (
        frozenset({"ALL"}),
        "generated results record with an explicit history note (S02-fix C3b)",
    ),
    "docs/reference/deprecations.md": (
        frozenset({"ALL"}),
        "THE removed-spellings ledger; naming old spellings is its job",
    ),
    "docs/migration/v2.0_api_changes.md": (
        frozenset({"ALL"}),
        "migration ledger; maps old spellings to canonical ones",
    ),
    "docs/migration/scoped_detached_patching.md": (
        frozenset({"patch", "tl_moved"}),
        "removal-record prose for the crawler-era patch surface",
    ),
    "docs/backward.md": (
        frozenset({"tl_moved"}),
        "removal-record prose: names the former tl.intervening alias as removed",
    ),
    "docs/performance.md": (
        frozenset({"tl_moved"}),
        "removal-record prose: names the former tl.batched_extract alias as removed",
    ),
    "notebooks/audit/07_intervention.ipynb": (
        frozenset({"tl_moved"}),
        "audit checklist/prose records the former replay-family spellings as removed",
    ),
    "notebooks/audit/12_validation_stats_reporting.ipynb": (
        frozenset({"tl_moved"}),
        "audit checklist/prose records the former record_span spelling as removed",
    ),
    "tests/release_goldens/generators/write216.py": (
        frozenset({"paper_era"}),
        "harvest-time provenance script committed AS RUN: it executes under the "
        "harvested torchlens 2.16 wheel (G1 golden corpus), whose API IS the "
        "paper-era spelling; it never runs against current torchlens",
    ),
    "tests/test_deprecation_inventory.py": (
        frozenset({"ALL"}),
        "the package deprecation scanners recognize the deleted helper names",
    ),
    "tests/test_api_surface_deprecation.py": (
        frozenset({"ALL"}),
        "absence pins enumerate the removed spellings to assert they raise",
    ),
    "tests/test_capture_unification_p7.py": (
        frozenset({"keep_op"}),
        "pytest.raises absence pin: removed keep_op/keep_module fail loudly",
    ),
    "tests/test_backward.py": (
        frozenset({"perturb"}),
        "pytest.raises absence pin: perturb_saved_grads kwarg is deleted",
    ),
    "tests/test_node_modes.py": (
        frozenset({"node_style_domain"}),
        "pytest.raises absence pin: 'vision'/'attention' presets refuse typed",
    ),
    "tests/test_trace_autoroute_kwarg_forwarding.py": (
        frozenset({"trace_flat"}),
        "docstring records the 2026-08-19 incident in its then-live spelling",
    ),
    "notebooks/audit/09_fastlog_record.ipynb": (
        frozenset({"keep_op"}),
        "audit cell demos the removal: try/except TypeError on keep_op=",
    ),
    "torchlens/_deprecations.py": (
        frozenset({"machinery"}),
        "module docstring's historical note names the deleted helpers",
    ),
    "torchlens/AGENTS.md": (
        frozenset({"keep_op"}),
        "removal-record prose: keep_op/keep_module raise TypeError",
    ),
    "AGENTS.md": (
        frozenset({"keep_op", "patch", "paper_era"}),
        "removal-record prose (keep_op/keep_module raise TypeError) + the "
        "docs-lockstep incident's historical reference; CLAUDE.md only imports this file",
    ),
    "tests/test_removed_spelling_lint.py": (
        frozenset({"ALL"}),
        "this lint's own ledger",
    ),
    "torchlens/__init__.py": (
        frozenset({"paper_era"}),
        "teaching redirect table: the three paper-era rows raise a typed "
        "facade_redirect AttributeError naming the canonical home (AUD-CODE "
        "3.14; GATE-FIX row 1) -- a redirect for a removed name is the table's "
        "purpose, never a resurrection; tests/test_w051_capt3_paper_era_redirects.py pins them",
    ),
    "tests/test_w051_capt3_paper_era_redirects.py": (
        frozenset({"paper_era"}),
        "pytest.raises teaching pin: the three paper-era redirect rows refuse "
        "typed and name their canonical spelling",
    ),
    "tests/test_packaging_docs_a12.py": (
        frozenset({"tl_moved"}),
        "sibling removed-spelling scanner over docs/examples cells; naming "
        "old spellings (tl.StreamingOptions needle) is its job",
    ),
}


def _in_scan_scope(rel: str) -> bool:
    """Return whether a tracked path is inside the lint's sweep boundary.

    ``menagerie/`` vendored subtrees (classics, data, per-model sources) are
    the ONE exempt region -- the removal ruling's sweeps never rewrote
    vendored model code (repo lint-exclude respected); the menagerie
    TOP-LEVEL tooling and ``menagerie/tools/`` stay in scope because they
    call the live TorchLens surface.
    """

    if not rel.endswith(_SCAN_SUFFIXES):
        return False
    if rel.startswith("menagerie/"):
        remainder = rel[len("menagerie/") :]
        return "/" not in remainder or rel.startswith("menagerie/tools/")
    return True


def _tracked_scan_files() -> list[Path]:
    """Return the tracked text files in scan scope."""

    try:
        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - non-checkout run
        pytest.skip("removed-spelling lint requires a git checkout")
    return [_REPO_ROOT / rel for rel in listing.splitlines() if _in_scan_scope(rel)]


def _scan_file(path: Path, text: str) -> list[tuple[str, int, str]]:
    """Return (group, line_number, line) violations for one file's text."""

    violations: list[tuple[str, int, str]] = []
    for group, pattern in _FORBIDDEN.items():
        if not any(needle in text for needle in _NEEDLES[group]):
            continue
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.start())
            line = text[line_start : line_end if line_end != -1 else len(text)]
            if group == "trace_flat" and _GROUPED_OPTION_LINE.search(line):
                continue
            line_number = text.count("\n", 0, match.start()) + 1
            violations.append((group, line_number, line.strip()))
    return violations


def _relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


@pytest.mark.smoke
def test_no_removed_spelling_creeps_back() -> None:
    """Every hit outside the audited ledger is a resurrection defect."""

    offenders: list[str] = []
    matched_allowlist_groups: dict[str, set[str]] = {}
    for path in _tracked_scan_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, IsADirectoryError):  # pragma: no cover - racing deletion
            continue
        hits = _scan_file(path, text)
        if not hits:
            continue
        rel = _relative(path)
        allowed_groups, _reason = _ALLOWED.get(rel, (frozenset(), ""))
        for group, line_number, line in hits:
            if "ALL" in allowed_groups or group in allowed_groups:
                matched_allowlist_groups.setdefault(rel, set()).add(group)
                continue
            offenders.append(f"{rel}:{line_number} [{group}] {line[:160]}")
    assert not offenders, (
        "Removed spellings crept back (Batch-8 full shim deletion; "
        "docs/reference/deprecations.md names each canonical replacement). "
        "Migrate the site to the canonical spelling -- or, ONLY for an audited "
        "history record / removal-record prose / raises-pin, add a ledger row "
        "here with its reason:\n    " + "\n    ".join(offenders)
    )

    stale = [
        f"{rel} (reason: {_ALLOWED[rel][1]})"
        for rel in _ALLOWED
        if rel not in matched_allowlist_groups and rel != "tests/test_removed_spelling_lint.py"
    ]
    assert not stale, (
        "Stale allowlist rows (file no longer matches any forbidden pattern); "
        "prune them so the ledger stays honest:\n    " + "\n    ".join(stale)
    )


@pytest.mark.smoke
def test_lint_is_red_capable(tmp_path: Path) -> None:
    """The scanner actually detects a planted resurrection per group shape."""

    planted = {
        "machinery": "from torchlens._deprecations import warn_deprecated_alias\n",
        "paper_era": "log = tl.log_forward_pass(model, x)\n",
        "tl_moved": "spec = tl.load_intervention_spec(path)\n",
        "methods": "trace.replay(strict=True)\n",
        "trace_flat": 'log = tl.trace(model, x, layers_to_save="all")\n',
        "keep_op": "rec = tl.record(model, x, keep_op=tl.func('relu'))\n",
        "patch": "tl.wrap_torch(patch_policy='detached')\n",
        "perturb": "validate_backward_pass(model, x, perturb_saved_grads=True)\n",
        "vis_opt": "log.draw(vis_opt='rolled')\n",
        "buffers_bool": "log.draw(show_buffers=True)\n",
        "node_style_domain": "log.draw(node_style='vision')\n",
        "inert_opts": "opts = SaveOptions(output_dir='out')\n",
    }
    assert set(planted) == set(_FORBIDDEN)
    assert set(_NEEDLES) == set(_FORBIDDEN), "every group needs a needle row"
    for group, snippet in planted.items():
        target = tmp_path / f"planted_{group}.py"
        target.write_text(snippet, encoding="utf-8")
        hits = _scan_file(target, snippet)
        assert any(hit_group == group for hit_group, _, _ in hits), (
            f"scanner failed to flag the planted {group} resurrection: {snippet!r}"
        )

    clean = 'log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))\n'
    assert not _scan_file(tmp_path / "clean.py", clean), (
        "grouped-option spelling must not false-positive"
    )
