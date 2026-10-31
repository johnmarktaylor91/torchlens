"""Field-catalog lockstep ENFORCEMENT, repository-policy half (grind r2 row 20 / matrix R11).

The recurring incident class this module exists to end: a field catalog, its
owning record class, the generated artifacts derived from it, and the version
authorities that gate its persistence drift apart between efforts. Round 81
found ~52 such drifts by hand. Hand passes do not scale and do not stay found,
so this module is the MECHANISM instead:

1. **Closure.** ``check_every_field_catalog_is_registered`` derives the catalog
   universe from :mod:`torchlens._constants` at runtime and demands every member
   be registered here. A new ``*_FIELD_ORDER`` cannot be added without a
   lockstep entry, so the mechanism cannot be bypassed by omission -- the exact
   hole that left ``FUNC_CALL_LOCATION_FIELD_ORDER`` outside
   ``tests/test_record_field_policy.py::RECORD_CASES``.
2. **Generated == declared.** Every checked-in generated artifact is
   regenerated in-process and diffed, and the artifact universe is itself
   derived from the tree (by generated-file header marker), so a new generated
   file also cannot skip registration.
3. **Red-capability.** Each checker is a plain function over its two sides, and
   the ``check_*_detects_*`` functions plant drift into each one and prove the
   checker reports it. A lockstep gate nobody has proved can fail is not a
   gate.

The live-record half (every attribute a captured record actually carries must
be declared in its ``FIELD_POLICY``) captures a model and stays in
``tests/test_schema_lockstep.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _harness import REPO_ROOT, run_checks

if str(REPO_ROOT) not in sys.path:
    # pytest puts the repo root on sys.path (pyproject ``pythonpath``); the
    # generators (``tools``, ``benchmarks``) are imported by name below.
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens import _constants as constants  # noqa: E402
from torchlens._data_classes.aten_op import AtenOp  # noqa: E402
from torchlens._data_classes.backward_pass import BackwardPass  # noqa: E402
from torchlens._data_classes.buffer import Buffer  # noqa: E402
from torchlens._data_classes.field_policy import (  # noqa: E402
    RecordFieldPolicy,
    field_order_from_policy,
)
from torchlens._data_classes.func_call_location import FuncCallLocation  # noqa: E402
from torchlens._data_classes.grad_fn import GradFn  # noqa: E402
from torchlens._data_classes.grad_fn_call import GradFnCall  # noqa: E402
from torchlens._data_classes.layer import Layer  # noqa: E402
from torchlens._data_classes.module import Module, ModuleCall  # noqa: E402
from torchlens._data_classes.op import Op  # noqa: E402
from torchlens._data_classes.param import Param  # noqa: E402
from torchlens._data_classes.trace import Trace  # noqa: E402
from torchlens._io import FieldPolicy  # noqa: E402


def _for_each_case(
    cases: Iterable[Any], label: Callable[[Any], str], body: Callable[[Any], None]
) -> None:
    """Run ``body`` on every case (the former ``parametrize``), naming the failing case."""

    for case in cases:
        try:
            body(case)
        except AssertionError as error:
            raise AssertionError(f"[{label(case)}] {error}") from error


# ---------------------------------------------------------------------------
# Authority 1: field catalogs (``*_FIELD_ORDER``) vs their owning FIELD_POLICY.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Catalog:
    """One registered field catalog and the authority it must track.

    Parameters
    ----------
    constant:
        Name of the catalog list in :mod:`torchlens._constants`.
    owner:
        Record class whose ``FIELD_POLICY`` generates the catalog, or ``None``
        for a pure alias.
    alias_of:
        Name of the catalog this one is a compatibility alias for, or ``None``
        for a primary catalog. An alias must be the SAME list object, not a
        copy, so the two spellings cannot diverge.
    """

    constant: str
    owner: type[Any] | None = None
    alias_of: str | None = None


#: Every field catalog in ``torchlens._constants``, with its authority.
#: ``check_every_field_catalog_is_registered`` keeps this exhaustive.
CATALOGS: tuple[Catalog, ...] = (
    Catalog("MODEL_LOG_FIELD_ORDER", owner=Trace),
    Catalog("LAYER_PASS_LOG_FIELD_ORDER", owner=Op),
    Catalog("LAYER_LOG_FIELD_ORDER", owner=Layer),
    Catalog("PARAM_LOG_FIELD_ORDER", owner=Param),
    Catalog("BUFFER_LOG_FIELD_ORDER", owner=Buffer),
    Catalog("GRAD_FN_LOG_FIELD_ORDER", owner=GradFn),
    Catalog("GRAD_FN_PASS_LOG_FIELD_ORDER", owner=GradFnCall),
    Catalog("MODULE_PASS_LOG_FIELD_ORDER", owner=ModuleCall),
    Catalog("MODULE_LOG_FIELD_ORDER", owner=Module),
    Catalog("BACKWARD_PASS_FIELD_ORDER", owner=BackwardPass),
    Catalog("FUNC_CALL_LOCATION_FIELD_ORDER", owner=FuncCallLocation),
    Catalog("PRIMITIVE_OP_FIELD_ORDER", owner=AtenOp),
    # Historical spellings retained for callers; same object by construction.
    Catalog("OP_LOG_FIELD_ORDER", alias_of="LAYER_PASS_LOG_FIELD_ORDER"),
    Catalog("TENSOR_LOG_FIELD_ORDER", alias_of="LAYER_PASS_LOG_FIELD_ORDER"),
)

_CATALOG_SUFFIX = "FIELD_ORDER"


def declared_catalog_names(module: Any = constants) -> set[str]:
    """Return the field-catalog universe derived from a constants module.

    Parameters
    ----------
    module:
        Module to scan (injectable so the closure checker can be drift-planted).

    Returns
    -------
    set[str]
        Names of every public list constant whose name ends in ``FIELD_ORDER``.
    """

    return {
        name
        for name, value in vars(module).items()
        if name.endswith(_CATALOG_SUFFIX) and isinstance(value, list)
    }


def catalog_registration_gaps(
    declared: set[str],
    registered: set[str],
) -> tuple[set[str], set[str]]:
    """Return catalogs missing from, and phantom in, the lockstep registry.

    Parameters
    ----------
    declared:
        Catalog names derived from the code.
    registered:
        Catalog names carried by :data:`CATALOGS`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unregistered, phantom)``.
    """

    return declared - registered, registered - declared


def check_every_field_catalog_is_registered() -> None:
    """Every ``*_FIELD_ORDER`` in constants has a lockstep registry entry."""

    unregistered, phantom = catalog_registration_gaps(
        declared_catalog_names(), {catalog.constant for catalog in CATALOGS}
    )
    assert not unregistered, (
        "field catalogs with no lockstep entry (add them to CATALOGS in "
        f"scripts/checks/schema_lockstep.py): {sorted(unregistered)}"
    )
    assert not phantom, f"registered catalogs that no longer exist: {sorted(phantom)}"


def policy_catalog_diff(
    policy: dict[str, RecordFieldPolicy],
    catalog: list[str],
) -> tuple[list[str], list[str]]:
    """Return the generated-vs-declared difference for one catalog.

    Parameters
    ----------
    policy:
        Owning record's ``FIELD_POLICY`` table.
    catalog:
        Checked-in ordered field list.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(generated, declared)`` -- equal when the catalog is in lockstep.
    """

    return field_order_from_policy(policy), list(catalog)


_PRIMARY_CATALOGS = tuple(c for c in CATALOGS if c.owner is not None)
_ALIAS_CATALOGS = tuple(c for c in CATALOGS if c.alias_of is not None)


def _check_catalog_is_generated_from_its_owning_field_policy(catalog: Catalog) -> None:
    """The checked-in catalog equals the view generated from ``FIELD_POLICY``."""

    assert catalog.owner is not None
    generated, declared = policy_catalog_diff(
        catalog.owner.FIELD_POLICY, getattr(constants, catalog.constant)
    )
    assert generated == declared, (
        f"{catalog.constant} drifted from {catalog.owner.__name__}.FIELD_POLICY; "
        f"only in policy: {sorted(set(generated) - set(declared))}; "
        f"only in catalog: {sorted(set(declared) - set(generated))}"
    )
    assert len(declared) == len(set(declared)), f"{catalog.constant} has duplicates"


def check_catalog_is_generated_from_its_owning_field_policy() -> None:
    """Run the check body over every registered case."""

    _for_each_case(
        _PRIMARY_CATALOGS,
        lambda catalog: catalog.constant,
        _check_catalog_is_generated_from_its_owning_field_policy,
    )


def _check_alias_catalogs_share_the_primary_object(catalog: Catalog) -> None:
    """An alias catalog IS its primary, so the spellings cannot diverge."""

    assert catalog.alias_of is not None
    assert getattr(constants, catalog.constant) is getattr(constants, catalog.alias_of)


def check_alias_catalogs_share_the_primary_object() -> None:
    """Run the check body over every registered case."""

    _for_each_case(
        _ALIAS_CATALOGS,
        lambda catalog: catalog.constant,
        _check_alias_catalogs_share_the_primary_object,
    )


# ---------------------------------------------------------------------------
# Authority 2: private-named ordered fields (the OL#47 disclosure class).
# ---------------------------------------------------------------------------

#: Tier A -- private-named fields that sit in a PUBLIC ``FIELD_ORDER`` while
#: being ``FieldPolicy.DROP`` (ordered, yet deliberately non-portable and
#: session-only). This is the most confusing combination on the record surface
#: and the one the package docs claim to enumerate, so each entry states WHY.
#: OL#47 -- ``_fast_run_session`` landing here undocumented -- is the incident
#: this ledger closes as a class.
PRIVATE_ORDERED_DROP_FIELDS: dict[str, dict[str, str]] = {
    "Trace": {
        "_runnable": "sparse-runnable state container; rebuilt at load, never portable itself",
        "_fast_run_session": "session-time guarded-static-loop handle (tl.Trace.run(fast=True))",
        "_distributed_plane_p": "session-time plane-P dispatch journal for armed captures (merge-ranks C2 census evidence)",
        "_transform": "capture-time input transform callable; opaque, session-only",
        "_output_transform": "capture-time output transform callable; opaque, session-only",
        "_visualizer_dir": "per-session visualizer scratch directory path",
        "_out_dedup_mode": "session dedup strategy for retained activation payloads",
        "_out_identity_cache": "session identity cache backing out dedup",
        "_out_hash_cache": "session hash cache backing out dedup",
        "_code_context_cache": "session source-context cache; rebuilt from source on demand",
        "_source_model_ref": "weak reference to the captured model; never portable",
        "_intervention_spec": "live intervention spec object; persisted by its own saver",
        "_warned_direct_write": "once-per-trace warn sentinel; a fresh load must warn again",
        "_warned_mutate_in_place": "once-per-trace warn sentinel; a fresh load must warn again",
        "_last_hook_handle_ids": "session hook-handle bookkeeping for teardown",
    },
    "Op": {
        "_construction_done": "construction-phase latch read by the direct-write guard",
    },
}

#: Tier B -- every other private-named ordered field. These are ordinary
#: persisted internals (KEEP/BLOB policy), so they need a reviewed one-line
#: registration rather than prose. Pinning the set still makes a newly ordered
#: private field a deliberate diff instead of a silent surface change.
PRIVATE_ORDERED_PERSISTED_FIELDS: frozenset[str] = frozenset(
    {
        "Trace._tracing_finished",
        "Trace._capture_outcome",
        "Trace._layers_logged",
        "Trace._layers_saved",
        "Trace._replay_arg_version_data_complete",
        "Trace._grad_op_nums_to_save",
        "Trace._activation_transform_repr",
        "Trace._source_code_blob",
        "Trace._has_direct_writes",
        "Trace._spec_revision",
        "Trace._out_recipe_revision",
        "Trace._append_sequence_id",
        "Trace._layer_nums_to_save",
        "Trace._raw_to_final_layer_labels",
        "Trace._raw_to_final_parent_layer_labels",
        "Trace._raw_to_final_op_labels",
        "Trace._final_to_raw_layer_labels",
        "Trace._lookup_keys_to_layer_num_dict",
        "Trace._layer_num_to_lookup_keys_dict",
        "Trace._ambiguous_lookup_keys",
        "Trace._containers",
        "Trace._annotation_blobs",
        "Trace._buffer_persistence",
        "Trace._orphan_labels",
        "Trace._orphan_logs",
        "Trace._phase_timings",
        "Trace._grad_fn_param_refs",
        "Op._label_raw",
        "Op._layer_label_raw",
        "Op._tracing_finished",
        "Op._param_barcodes",
        "Op._param_logs",
        "Op._edge_uses",
        "Op._address_normalized",
        "Layer._param_barcodes",
        "Layer._param_logs",
        "Param._derived_grad_record_path",
        "GradFnCall._time_started",
        "GradFnCall._time_finished",
    }
)


def _private_ordered_fields_by_tier(
    catalogs: tuple[Catalog, ...],
) -> tuple[set[str], set[str]]:
    """Return live private-named ordered fields split by portable policy.

    Parameters
    ----------
    catalogs:
        Registered primary catalogs.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(drop_fields, persisted_fields)`` as ``"Class._field"`` strings.
    """

    drop: set[str] = set()
    persisted: set[str] = set()
    for catalog in catalogs:
        if catalog.owner is None:
            continue
        policy = catalog.owner.FIELD_POLICY
        for name in getattr(constants, catalog.constant):
            if not name.startswith("_"):
                continue
            key = f"{catalog.owner.__name__}.{name}"
            if policy[name].portable_policy is FieldPolicy.DROP:
                drop.add(key)
            else:
                persisted.add(key)
    return drop, persisted


def private_ordered_field_gaps(
    catalogs: tuple[Catalog, ...],
    ledger: dict[str, dict[str, str]],
) -> tuple[set[str], set[str]]:
    """Return unledgered and phantom private-named ordered DROP fields.

    Parameters
    ----------
    catalogs:
        Registered primary catalogs.
    ledger:
        Declared tier-A ledger keyed by record class name.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unledgered, phantom)`` as ``"Class._field"`` strings.
    """

    live, _ = _private_ordered_fields_by_tier(catalogs)
    declared = {f"{cls_name}.{field}" for cls_name, fields in ledger.items() for field in fields}
    return live - declared, declared - live


def check_private_named_ordered_drop_fields_are_ledgered() -> None:
    """An ordered private DROP field must state why it is on the surface."""

    unledgered, phantom = private_ordered_field_gaps(_PRIMARY_CATALOGS, PRIVATE_ORDERED_DROP_FIELDS)
    assert not unledgered, (
        "private-named FieldPolicy.DROP fields in a public FIELD_ORDER with no ledger "
        f"entry (document them in PRIVATE_ORDERED_DROP_FIELDS): {sorted(unledgered)}"
    )
    assert not phantom, f"ledgered private ordered fields that no longer exist: {sorted(phantom)}"


def check_private_named_ordered_persisted_fields_are_registered() -> None:
    """Ordering a private persisted field stays a reviewed one-line diff."""

    _, live = _private_ordered_fields_by_tier(_PRIMARY_CATALOGS)
    assert live == PRIVATE_ORDERED_PERSISTED_FIELDS, (
        "private-named persisted ordered fields changed; only live: "
        f"{sorted(live - PRIVATE_ORDERED_PERSISTED_FIELDS)}; only registered: "
        f"{sorted(PRIVATE_ORDERED_PERSISTED_FIELDS - live)}"
    )


def check_private_ordered_field_reasons_are_nonempty() -> None:
    """Every tier-A ledger entry carries a real reason, not a placeholder."""

    for cls_name, fields in PRIVATE_ORDERED_DROP_FIELDS.items():
        for field, reason in fields.items():
            assert len(reason.strip()) >= 20, f"{cls_name}.{field} needs a real reason"


# ---------------------------------------------------------------------------
# Authority 3: generated artifacts vs their generators.
# ---------------------------------------------------------------------------


def _render_schema_bindings() -> str:
    """Return a fresh rendering of the storage-bindings module.

    Returns
    -------
    str
        Generated source text.
    """

    from tools.generate_record_schema import _collect, _render

    return _render(_collect())


def _render_op_record_manifest() -> str:
    """Return a fresh rendering of the op-record cell-source manifest.

    Returns
    -------
    str
        Generated source text.
    """

    from tools.generate_op_record_manifest import generate

    return generate()


def _render_perf_numbers_doc(baseline_name: str) -> str:
    """Return a fresh rendering of one perf-numbers doc from its gate JSON.

    IMPORTANT: this is a RENDERING check only. The measured numbers live in
    the checked-in gate JSON under ``benchmarks/perf_baselines/``; this never
    re-runs a benchmark, so the check is machine-invariant (a pure
    JSON-to-Markdown projection of already-recorded measurements).

    Parameters
    ----------
    baseline_name:
        File name of the checked-in gate JSON baseline.

    Returns
    -------
    str
        Generated Markdown text.
    """

    from benchmarks.generate_perf_numbers import render_numbers_markdown
    from benchmarks.perf_gate import load_gate_json

    return render_numbers_markdown(
        load_gate_json(REPO_ROOT / "benchmarks" / "perf_baselines" / baseline_name)
    )


#: The generation snippet published verbatim in the compatibility doc; the
#: renderer below must build its reports EXACTLY this way.
_COMPAT_DOC_SNIPPET = """import torch
from torch import nn
import torchlens as tl

models = {
    "linear_mlp": nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).eval(),
    "conv_pool": nn.Sequential(
        nn.Conv2d(1, 2, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    ).eval(),
}
inputs = {
    "linear_mlp": torch.ones(1, 4),
    "conv_pool": torch.ones(1, 1, 5, 5),
}

for name, model in models.items():
    print(name)
    print(tl.compat.report(model, inputs[name]).to_markdown())"""

_COMPAT_DOC_REGENERATE_COMMAND = (
    "python -c \"import sys; sys.path.insert(0, 'tests'); "
    'import test_schema_lockstep as m; m.write_method_x_model_compatibility_doc()"'
)


def _compat_reference_reports() -> dict[str, Any]:
    """Return the doc's representative ``tl.compat.report`` results.

    Returns
    -------
    dict[str, Any]
        Model name -> ``CompatReport``, built exactly as the published
        snippet builds them.
    """

    # Pin the belt-coverage row to its post-wrap status: the row honestly
    # reports ``not_tested`` before the lazy first-capture wrap derives the
    # belt, so its rendered status would otherwise depend on whether THIS
    # process already captured (pytest's conftest warm-up capture vs the
    # bare-python regenerate command) — exactly the process-state flap this
    # machine-invariant projection forbids. Wrapping is idempotent and every
    # full-suite pytest process is already wrapped by session setup.
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    models = {
        "linear_mlp": nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).eval(),
        "conv_pool": nn.Sequential(
            nn.Conv2d(1, 2, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        ).eval(),
    }
    inputs = {
        "linear_mlp": torch.ones(1, 4),
        "conv_pool": torch.ones(1, 1, 5, 5),
    }
    return {name: tl.compat.report(model, inputs[name]) for name, model in models.items()}


def _render_method_x_model_compat() -> str:
    """Return a fresh rendering of ``docs/method_x_model_compatibility.md``.

    The doc is generated from the MACHINE-INVARIANT projection of the two
    representative reports: the row LABELS (the check list, fixed by the
    torchlens code), the row COUNT, and each row's pass/non-pass STATUS on the
    toy CPU models (structural probes that never execute the model). Row
    ``details``/``severity`` strings embed environment facts (torch build
    capability flags, visible CUDA device counts) and are deliberately NOT
    rendered, so the regenerate-and-diff gate cannot flap across hosts.

    Returns
    -------
    str
        Generated Markdown text.
    """

    reports = _compat_reference_reports()
    label_sequences = {
        name: tuple(row.label for row in report.checks) for name, report in reports.items()
    }
    (first_labels, *other_labels) = label_sequences.values()
    if any(labels != first_labels for labels in other_labels):
        raise RuntimeError(
            "compat report rows differ across the representative models; "
            f"the doc renderer needs a redesign: {label_sequences}"
        )

    lines = [
        "# Method x Model Compatibility",
        "",
        "<!-- GENERATED FILE; do not hand-edit. tests/test_schema_lockstep.py",
        "regenerates and diffs this doc. Refresh from the repo root with:",
        _COMPAT_DOC_REGENERATE_COMMAND,
        "-->",
        "",
        "Generated from `tl.compat.report` on representative eager PyTorch models. These rows are a",
        "smoke reference for ordinary dense eager execution, not a complete certification matrix.",
        "",
        "Generation snippet:",
        "",
        "```python",
        _COMPAT_DOC_SNIPPET,
        "```",
        "",
        "## Representative Results",
        "",
        "| Model | Rows | Non-pass rows |",
        "| --- | ---: | --- |",
    ]
    any_non_pass = False
    for name, report in reports.items():
        non_pass = [row.label for row in report.checks if row.status != "pass"]
        any_non_pass = any_non_pass or bool(non_pass)
        cell = ", ".join(f"`{label}`" for label in non_pass) if non_pass else "none"
        lines.append(f"| `{name}` | {len(report.checks)} | {cell} |")
    lines.append("")
    if any_non_pass:
        lines.append(
            "Non-pass rows above name the checks that did not report `pass`; every other check"
        )
        lines.append("reported `pass`. The full check list:")
    else:
        lines.append("Both representative models report `pass` for every check:")
    lines.append("")
    lines.extend(f"- {label}" for label in first_labels)
    lines.extend(
        [
            "",
            "Interpretation: plain eager dense PyTorch models are the compatibility baseline. For"
            " wrappers,",
            "compiled execution, sharding/offload, quantization, or concurrent capture, run",
            "`tl.compat.report(model, x)` on the exact model/input pair and include the report when"
            " filing an",
            "issue.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_method_x_model_compatibility_doc() -> None:
    """Regenerate ``docs/method_x_model_compatibility.md`` in place."""

    path = REPO_ROOT / "docs" / "method_x_model_compatibility.md"
    path.write_text(_render_method_x_model_compat(), encoding="utf-8")
    print(f"wrote {path}")


@dataclass(frozen=True)
class GeneratedArtifact:
    """One checked-in generated text artifact and its in-process renderer.

    Parameters
    ----------
    path:
        Repo-relative path of the generated artifact (a package module or a
        generated doc).
    render:
        Callable returning the freshly generated text.
    regenerate_command:
        Command a developer runs to refresh the artifact.
    """

    path: str
    render: Callable[[], str]
    regenerate_command: str


#: Every registered generated text artifact: the generated modules under
#: ``torchlens/`` (kept exhaustive by
#: ``check_every_generated_module_is_registered``) plus the generated docs
#: (R53-6: docs were invisible to the old torchlens-only closure by
#: construction, which is how the collapse gallery and the compat matrix
#: drifted at birth).
GENERATED_ARTIFACTS: tuple[GeneratedArtifact, ...] = (
    GeneratedArtifact(
        "torchlens/_data_classes/_schema_bindings.py",
        _render_schema_bindings,
        "python tools/generate_record_schema.py",
    ),
    GeneratedArtifact(
        "torchlens/_ir/op_record_manifest.py",
        _render_op_record_manifest,
        "python -m tools.generate_op_record_manifest",
    ),
    GeneratedArtifact(
        "docs/_perf_numbers.md",
        lambda: _render_perf_numbers_doc("linux-cpu.json"),
        "python -m benchmarks.generate_perf_numbers "
        "benchmarks/perf_baselines/linux-cpu.json --out docs/_perf_numbers.md",
    ),
    GeneratedArtifact(
        "docs/_perf_numbers_provisional.md",
        lambda: _render_perf_numbers_doc("linux-cpu-provisional.json"),
        "python -m benchmarks.generate_perf_numbers "
        "benchmarks/perf_baselines/linux-cpu-provisional.json "
        "--out docs/_perf_numbers_provisional.md",
    ),
    GeneratedArtifact(
        "docs/method_x_model_compatibility.md",
        _render_method_x_model_compat,
        _COMPAT_DOC_REGENERATE_COMMAND,
    ),
)


@dataclass(frozen=True)
class GeneratedDirectoryArtifact:
    """One checked-in generated directory gated through its script's --check.

    Parameters
    ----------
    path:
        Repo-relative path of the generated directory.
    check_command:
        Argv (relative to the repo root, run with the current interpreter)
        that regenerates to a temp dir and byte-diffs; exits nonzero and
        names the differing files when stale.
    regenerate_command:
        Command a developer runs to refresh the directory.
    """

    path: str
    check_command: tuple[str, ...]
    regenerate_command: str


#: Generated directories too expensive for an in-process render (the collapse
#: gallery re-traces four models and renders 14 graphviz SVGs), gated by the
#: owning script's real --check mode in the heavy tier.
GENERATED_DIRECTORY_ARTIFACTS: tuple[GeneratedDirectoryArtifact, ...] = (
    GeneratedDirectoryArtifact(
        "docs/images/collapse",
        ("scripts/render_collapse_reference.py", "--check"),
        "python scripts/render_collapse_reference.py",
    ),
)

#: Header marker every generated module carries on its first line.
_GENERATED_MARKER = "GENERATED"


def _module_paths() -> Iterator[Path]:
    """Yield every Python module in the shipped package.

    Yields
    ------
    Path
        Absolute path of one package module.
    """

    yield from sorted((REPO_ROOT / "torchlens").rglob("*.py"))


def generated_module_paths(marker: str = _GENERATED_MARKER) -> set[str]:
    """Return repo-relative paths of modules declaring themselves generated.

    A generated module announces itself on its docstring's FIRST line; that
    keeps the scan from matching prose deeper in a hand-written file.

    Parameters
    ----------
    marker:
        Header token that marks a module as generated.

    Returns
    -------
    set[str]
        Repo-relative POSIX paths.
    """

    found: set[str] = set()
    for path in _module_paths():
        with path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        if marker in first_line:
            found.add(path.relative_to(REPO_ROOT).as_posix())
    return found


def artifact_registration_gaps(
    found: set[str],
    registered: set[str],
) -> tuple[set[str], set[str]]:
    """Return generated modules missing from, and phantom in, the registry.

    Parameters
    ----------
    found:
        Generated-module paths discovered in the tree.
    registered:
        Paths carried by :data:`GENERATED_ARTIFACTS`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unregistered, phantom)``.
    """

    return found - registered, registered - found


def check_every_generated_module_is_registered() -> None:
    """Every self-declared generated module has a regenerate-and-diff entry.

    The marker-scan closure covers ``torchlens/**/*.py``, so only the
    registry's ``torchlens/`` entries participate in the phantom check; the
    generated docs and directories carry explicit registry entries instead
    (there is no header-marker universe to derive them from).
    """

    unregistered, phantom = artifact_registration_gaps(
        generated_module_paths(),
        {
            artifact.path
            for artifact in GENERATED_ARTIFACTS
            if artifact.path.startswith("torchlens/")
        },
    )
    assert not unregistered, (
        "generated modules with no lockstep entry (register them in "
        f"GENERATED_ARTIFACTS): {sorted(unregistered)}"
    )
    assert not phantom, f"registered generated modules that no longer exist: {sorted(phantom)}"


def check_ruff_excludes_every_generated_artifact() -> None:
    """Ruff must not touch a generated module, and must not exclude a hand-written one.

    ``check_generated_artifact_is_current`` compares each artifact BYTE-FOR-BYTE
    against fresh generator output, so any ruff rewrite (format, isort, ``UP*``)
    makes it permanently red. Ruff's ``extend-exclude`` is therefore the matching
    half of this lockstep, and it is pinned here in BOTH directions: adding a
    generated module without its exclusion fails, and leaving an exclusion behind
    after a module stops being generated fails too.

    Regression: the grind r3 lint ratchet reformatted both artifacts and turned
    this gate red until the generators were re-run.
    """

    # Regex, not tomllib: the declared floor is python 3.10, where tomllib is absent.
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(
        r"^\s*extend-exclude\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL | re.MULTILINE
    )
    assert block is not None, "pyproject [tool.ruff] must declare extend-exclude"
    excluded = set(re.findall(r'"([^"]+)"', block.group(1)))

    generated = generated_module_paths()
    missing = sorted(generated - excluded)
    assert not missing, (
        f"generated modules ruff would rewrite (add to [tool.ruff] extend-exclude): {missing}"
    )

    # Only the generated half is pinned; the vendored-corpus directory entries are
    # excluded for unrelated provenance reasons and are legitimately extra.
    py_excludes = {path for path in excluded if path.endswith(".py")}
    assert py_excludes <= generated, (
        "extend-exclude names a .py file that is no longer a generated artifact: "
        f"{sorted(py_excludes - generated)}"
    )


def _check_generated_artifact_is_current(artifact: GeneratedArtifact) -> None:
    """The checked-in generated module matches a fresh in-process generation.

    Deliberately in-process rather than a subprocess: the same comparison at a
    fraction of the cost, which is what lets it live in the smoke tier where
    drift is caught the same minute it lands.
    """

    checked_in = (REPO_ROOT / artifact.path).read_text(encoding="utf-8")
    assert checked_in == artifact.render(), (
        f"{artifact.path} is stale -- run: {artifact.regenerate_command}"
    )


def check_generated_artifact_is_current() -> None:
    """Run the check body over every registered case."""

    _for_each_case(
        GENERATED_ARTIFACTS,
        lambda artifact: artifact.path,
        _check_generated_artifact_is_current,
    )


def _check_generated_directory_artifact_is_current(artifact: GeneratedDirectoryArtifact) -> None:
    """The checked-in generated directory matches a fresh regeneration.

    Runs the owning script's ``--check`` (regenerate to a temp dir, byte-diff,
    name the differing files). Rendering is byte-deterministic across hosts:
    every Graphviz render measures and draws text with the bundled face, so
    machines with different Pango releases and system fonts write the same
    bytes. A diff therefore means the committed gallery drifted from the
    current rendering code -- exactly the born-stale class this gate exists
    to end. Heavy tier, not smoke: the
    collapse gallery re-traces four models and renders 14 graphviz SVGs
    (~10s). Byte output ties to the installed graphviz, so a legitimate
    graphviz upgrade shows up here as a reviewed regen, never a silent drift.
    """

    result = subprocess.run(
        [sys.executable, *artifact.check_command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{artifact.path} is stale -- run: {artifact.regenerate_command}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def check_generated_directory_artifact_is_current() -> None:
    """Run the check body over every registered case."""

    _for_each_case(
        GENERATED_DIRECTORY_ARTIFACTS,
        lambda artifact: artifact.path,
        _check_generated_directory_artifact_is_current,
    )


# ---------------------------------------------------------------------------
# Authority 5: persistence version authorities (SF-40 drift plant).
# ---------------------------------------------------------------------------

#: Reviewed pins for the persistence version authorities. These are three
#: INDEPENDENT counters (a merged root manifest is a different discriminated
#: object from a rank core's manifest -- see ``torchlens/_merged/_enums.py``), so
#: this is deliberately NOT an equality assertion between them: it is a pin
#: apiece, with the co-change list a bump must honor.
#:
#: Bumping ``TLSPEC_VERSION``: update the pin, the manifest schema, and the
#: load-path tests. Bumping ``MIN_TLSPEC_VERSION`` (the rehydration floor):
#: also update the floor statement in ``CLAUDE.md`` and
#: ``tests/test_rehydration_floor.py``. Bumping ``MERGED_TLSPEC_VERSION``:
#: also update ``docs/reference/merged_trace_contract.md``, whose stated value
#: is checked against the code below.
VERSION_AUTHORITY_PINS: dict[str, int] = {
    # v10: the batch-2 coordinated schema write (R2-05 with R2-38) — one
    # bump for every new saved field of the window (admitted by the v10 field
    # registry in tests/test_schema_v10_contract_and_goldens.py), the floor
    # text corrected to 2.31.0 in the writer contract, and the reserved
    # generation step-score vocabulary credited to inseq (U22.3). Contract of
    # record: torchlens/schemas/writer_contract_v10.json (v9 frozen beside it).
    # v9: the completeness-work coordinated schema write (2026-08-27,
    # lane C07) — audit-row grammar admissions (ACT site source, PARAM
    # rows/recipes, EVENT envelope + hash-chain extension), the sidecar
    # annotations flip to plain persistence, reserved annotation families
    # (health_facts, capture_advisories), and the entry-dark v9 field slots
    # (injection provenance, source snapshots, structure evidence), each
    # with fail-closed load validation. Contract of record:
    # torchlens/schemas/writer_contract_v9.json (v8 pinned at
    # tests/release_goldens/writer_contract_v8.json). MERGED_TLSPEC_VERSION
    # versions the merged ROOT manifest independently and did not move.
    "TLSPEC_VERSION": 10,
    "MIN_TLSPEC_VERSION": 6,
    "MERGED_TLSPEC_VERSION": 7,
}


def _version_authority_values() -> dict[str, int]:
    """Return the live value of each persistence version authority.

    Returns
    -------
    dict[str, int]
        Authority name -> value.
    """

    from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION
    from torchlens._merged._enums import MERGED_TLSPEC_VERSION

    return {
        "TLSPEC_VERSION": TLSPEC_VERSION,
        "MIN_TLSPEC_VERSION": MIN_TLSPEC_VERSION,
        "MERGED_TLSPEC_VERSION": MERGED_TLSPEC_VERSION,
    }


def version_pin_drift(live: dict[str, int], pins: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Return authorities whose live value left its reviewed pin.

    Parameters
    ----------
    live:
        Live authority values.
    pins:
        Reviewed pinned values.

    Returns
    -------
    dict[str, tuple[int, int]]
        Authority -> ``(live, pinned)`` for every mismatch.
    """

    return {
        name: (value, pins[name])
        for name, value in live.items()
        if name in pins and value != pins[name]
    }


def check_version_authorities_match_their_reviewed_pins() -> None:
    """A persistence version bump is a reviewed diff, never a silent one."""

    live = _version_authority_values()
    assert set(live) == set(VERSION_AUTHORITY_PINS)
    drift = version_pin_drift(live, VERSION_AUTHORITY_PINS)
    assert not drift, (
        "persistence version authority moved without updating its lockstep pin "
        f"and co-change list: {drift}"
    )
    assert live["MIN_TLSPEC_VERSION"] <= live["TLSPEC_VERSION"]


def documented_merged_versions(text: str) -> set[int]:
    """Return every merged ``tlspec_version`` value stated in contract prose.

    Parameters
    ----------
    text:
        Contract document text.

    Returns
    -------
    set[int]
        Stated version numbers.
    """

    return {int(match) for match in re.findall(r"tlspec_version[:`\s]+(\d+)", text)}


def check_merged_contract_doc_states_the_shipped_version() -> None:
    """The merged contract doc's stated version tracks the code constant."""

    contract = REPO_ROOT / "docs" / "reference" / "merged_trace_contract.md"
    stated = documented_merged_versions(contract.read_text(encoding="utf-8"))
    assert stated, "merged contract must state its tlspec_version"
    live = _version_authority_values()["MERGED_TLSPEC_VERSION"]
    assert stated == {live}, (
        f"docs/reference/merged_trace_contract.md states tlspec_version {sorted(stated)} "
        f"but MERGED_TLSPEC_VERSION is {live}"
    )


# ---------------------------------------------------------------------------
# The mechanism must be able to go RED. Each checker gets a planted drift.
# ---------------------------------------------------------------------------


#: Negative controls for the checkers above: each plants drift and proves it is
#: reported. Without these, a lockstep gate could be silently vacuous -- a
#: registry that matches itself, a diff that compares a value with itself.


def check_catalog_registration_closure_detects_an_unregistered_catalog() -> None:
    """A new catalog with no registry entry is reported."""

    class _FakeConstants:
        NEW_THING_FIELD_ORDER = ["a", "b"]
        MODEL_LOG_FIELD_ORDER = ["c"]
        NOT_A_CATALOG = 3

    declared = declared_catalog_names(_FakeConstants)
    assert declared == {"NEW_THING_FIELD_ORDER", "MODEL_LOG_FIELD_ORDER"}
    unregistered, phantom = catalog_registration_gaps(declared, {"MODEL_LOG_FIELD_ORDER"})
    assert unregistered == {"NEW_THING_FIELD_ORDER"}
    assert not phantom


def check_catalog_registration_closure_detects_a_phantom_entry() -> None:
    """A registry entry for a deleted catalog is reported."""

    unregistered, phantom = catalog_registration_gaps({"A_FIELD_ORDER"}, {"GONE_FIELD_ORDER"})
    assert unregistered == {"A_FIELD_ORDER"}
    assert phantom == {"GONE_FIELD_ORDER"}


def check_policy_catalog_diff_detects_a_dropped_field() -> None:
    """A field present in the policy but missing from the catalog differs."""

    generated, declared = policy_catalog_diff(
        Trace.FIELD_POLICY, constants.MODEL_LOG_FIELD_ORDER[:-1]
    )
    assert generated != declared


def check_policy_catalog_diff_detects_a_reordering() -> None:
    """Order is part of the contract, so a swap must differ too."""

    reordered = list(constants.MODEL_LOG_FIELD_ORDER)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    generated, declared = policy_catalog_diff(Trace.FIELD_POLICY, reordered)
    assert generated != declared


def check_private_ordered_ledger_detects_an_unledgered_field() -> None:
    """A private ordered field missing from the ledger is reported."""

    pruned = {
        cls_name: {k: v for k, v in fields.items() if k != "_runnable"}
        for cls_name, fields in PRIVATE_ORDERED_DROP_FIELDS.items()
    }
    unledgered, _ = private_ordered_field_gaps(_PRIMARY_CATALOGS, pruned)
    assert unledgered == {"Trace._runnable"}


def check_private_ordered_ledger_detects_a_phantom_entry() -> None:
    """A ledger entry with no live ordered field is reported."""

    padded = {cls: dict(fields) for cls, fields in PRIVATE_ORDERED_DROP_FIELDS.items()}
    padded["Trace"]["_never_existed"] = "planted"
    _, phantom = private_ordered_field_gaps(_PRIMARY_CATALOGS, padded)
    assert phantom == {"Trace._never_existed"}


def check_private_ordered_tiers_partition_the_live_set() -> None:
    """The two tiers are disjoint and together cover every private field."""

    drop, persisted = _private_ordered_fields_by_tier(_PRIMARY_CATALOGS)
    assert drop and persisted
    assert not drop & persisted


def check_generated_artifact_closure_detects_an_unregistered_module() -> None:
    """A generated module absent from the registry is reported."""

    unregistered, phantom = artifact_registration_gaps(
        {"torchlens/x/_gen.py", "torchlens/_ir/op_record_manifest.py"},
        {"torchlens/_ir/op_record_manifest.py"},
    )
    assert unregistered == {"torchlens/x/_gen.py"}
    assert not phantom


def check_generated_artifact_scan_finds_the_known_generated_modules() -> None:
    """The header scan is not vacuous: it finds the real generated files."""

    found = generated_module_paths()
    assert "torchlens/_data_classes/_schema_bindings.py" in found
    assert "torchlens/_ir/op_record_manifest.py" in found


def check_generated_artifact_diff_detects_a_mutated_artifact() -> None:
    """A byte-level edit to a generated module is reported."""

    artifact = GENERATED_ARTIFACTS[0]
    checked_in = (REPO_ROOT / artifact.path).read_text(encoding="utf-8")
    assert checked_in + "# tampered\n" != artifact.render()


def check_generated_doc_diff_detects_a_mutated_doc() -> None:
    """A byte-level edit to a generated doc is reported (R53-6/R53-7)."""

    artifact = next(a for a in GENERATED_ARTIFACTS if a.path == "docs/_perf_numbers.md")
    checked_in = (REPO_ROOT / artifact.path).read_text(encoding="utf-8")
    assert checked_in + "hand-edited number\n" != artifact.render()


def check_generated_doc_registry_covers_the_known_docs() -> None:
    """The widened registry really carries the doc artifacts (R53-6)."""

    registered = {artifact.path for artifact in GENERATED_ARTIFACTS}
    assert {
        "docs/_perf_numbers.md",
        "docs/_perf_numbers_provisional.md",
        "docs/method_x_model_compatibility.md",
    } <= registered


def check_gallery_registry_points_at_a_real_check() -> None:
    """The directory-artifact registry names a live script and gallery.

    The gallery gate's red-capability was proven empirically: before the
    R53-4 regeneration, ``--check`` flagged all 14 committed SVGs as
    differing. This cheap control keeps the registration itself honest.
    """

    (artifact,) = GENERATED_DIRECTORY_ARTIFACTS
    script = REPO_ROOT / artifact.check_command[0]
    assert script.is_file()
    assert "--check" in artifact.check_command
    assert list((REPO_ROOT / artifact.path).glob("*.svg"))


def check_version_pin_checker_detects_a_bump() -> None:
    """A version bump without a pin update is reported."""

    drift = version_pin_drift({"TLSPEC_VERSION": 8}, {"TLSPEC_VERSION": 7})
    assert drift == {"TLSPEC_VERSION": (8, 7)}
    assert not version_pin_drift({"TLSPEC_VERSION": 7}, {"TLSPEC_VERSION": 7})


def check_merged_doc_parser_reads_stated_versions() -> None:
    """The doc parser finds prose and code-fence spellings, not noise."""

    assert documented_merged_versions("carries `tlspec_version: 7`,") == {7}
    assert documented_merged_versions("# tlspec_version: 9, descriptor") == {9}
    assert documented_merged_versions("nothing here") == set()


def check_field_policy_entries_declare_a_portable_policy() -> None:
    """Every declared field carries a real portable policy value.

    Cheap catch for a half-added field: present in the table, but with a
    policy value that is not a member of the closed vocabulary.
    """

    for catalog in _PRIMARY_CATALOGS:
        assert catalog.owner is not None
        for name, item in catalog.owner.FIELD_POLICY.items():
            assert isinstance(item.portable_policy, FieldPolicy), (
                f"{catalog.owner.__name__}.{name} has a non-FieldPolicy portable policy"
            )


CHECKS = (
    check_every_field_catalog_is_registered,
    check_catalog_is_generated_from_its_owning_field_policy,
    check_alias_catalogs_share_the_primary_object,
    check_private_named_ordered_drop_fields_are_ledgered,
    check_private_named_ordered_persisted_fields_are_registered,
    check_private_ordered_field_reasons_are_nonempty,
    check_every_generated_module_is_registered,
    check_ruff_excludes_every_generated_artifact,
    check_generated_artifact_is_current,
    check_generated_directory_artifact_is_current,
    check_version_authorities_match_their_reviewed_pins,
    check_merged_contract_doc_states_the_shipped_version,
    check_catalog_registration_closure_detects_an_unregistered_catalog,
    check_catalog_registration_closure_detects_a_phantom_entry,
    check_policy_catalog_diff_detects_a_dropped_field,
    check_policy_catalog_diff_detects_a_reordering,
    check_private_ordered_ledger_detects_an_unledgered_field,
    check_private_ordered_ledger_detects_a_phantom_entry,
    check_private_ordered_tiers_partition_the_live_set,
    check_generated_artifact_closure_detects_an_unregistered_module,
    check_generated_artifact_scan_finds_the_known_generated_modules,
    check_generated_artifact_diff_detects_a_mutated_artifact,
    check_generated_doc_diff_detects_a_mutated_doc,
    check_generated_doc_registry_covers_the_known_docs,
    check_gallery_registry_points_at_a_real_check,
    check_version_pin_checker_detects_a_bump,
    check_merged_doc_parser_reads_stated_versions,
    check_field_policy_entries_declare_a_portable_policy,
)


def main() -> int:
    """Run every schema_lockstep check and return the exit status."""

    return run_checks("schema_lockstep", CHECKS)


if __name__ == "__main__":
    sys.exit(main())
