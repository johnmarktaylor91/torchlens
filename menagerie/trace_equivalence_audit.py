"""Field-equivalence audit between the validation-config and metadata-config traces.

Bucket A of the post-sweep sharpening sprint sources the metadata summary row from
the trace the validation worker ALREADY built, instead of re-instantiating the model
and re-tracing it for metadata. That is only sound for a summary field if the field is
identical between the two capture configurations:

* VALIDATION config (``user_funcs._validate_forward_pass_torch`` ->
  ``_run_model_and_save_specified_outs``): ``layers_to_save="all"``,
  ``save_arg_values=True``, ``save_rng_states=True``, ``mark_layer_depths=False``,
  ``inference_only=False`` (autograd graph kept), device resolved (CPU on the
  CPU-global sweep).
* METADATA config (``structural_digest._trace_for_digest``): ``tl.trace`` defaults
  (``mark_layer_depths``/``compute_input_output_distances=True``), ``inference_only=True``,
  forced CPU.

This module is the correctness GATE: it runs BOTH configs on a STRATIFIED SAMPLE and
diffs every :func:`menagerie.trace_summary.summarize_trace` field. A field is EQUIVALENT
only if it is byte-identical across the whole sample; any field that ever differs is
DIVERGENT and must be recomputed for affected models rather than lifted from the
validation trace.

Run as a CLI for a corpus audit::

    python -m menagerie.trace_equivalence_audit --sample 60 --json out.json

or call :func:`audit_stable_id` / :func:`audit_sample` programmatically (the
regression test pins the KNOWN divergent set so a future capture change that silently
perturbs another field is caught).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from typing import Any

from menagerie.catalog import CATALOG_DB, CatalogRow, load_rows
from menagerie.recipe import build_input_for_row, instantiate_model
from menagerie.structural_digest import _trace_for_digest
from menagerie.trace_summary import TRACE_SUMMARY_COLUMNS, summarize_trace

# Fields that are provenance/identity, identical by construction regardless of trace
# config, and therefore excluded from the structural-equivalence comparison.
PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    {
        "stable_id",
        "trace_summary_version",
        "op_taxonomy_version",
        "recipe_revision_sha256",
        "torchlens_version",
    }
)

# The KNOWN-DIVERGENT set, root-caused to the ``mark_layer_depths`` capture-option
# difference (the validation trace is built with ``mark_layer_depths=False`` so
# per-op ``max_distance_from_input`` is ``None``; the metadata trace computes it).
# These are graph hop-distance derivations -- NOT a device-driven op-decomposition
# difference. ``structural_barcode_json`` is divergent only because it embeds
# ``graph_depth``/``max_width``. The regression test pins exactly this set: any field
# that joins or leaves it is a real capture change that must be re-audited.
KNOWN_DIVERGENT_FIELDS: frozenset[str] = frozenset(
    {
        "graph_depth",
        "graph_max_width",
        "structural_barcode_json",
    }
)


@dataclass
class RowAudit:
    """Per-model equivalence result.

    Attributes
    ----------
    stable_id:
        Audited model id.
    ok:
        Whether both traces built and were compared.
    error:
        Failure detail when ``ok`` is False.
    diff_fields:
        Mapping of field name -> ``(validation_value, metadata_value)`` for every
        field that differed between the two configs.
    """

    stable_id: str
    ok: bool
    error: str = ""
    diff_fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Corpus-level equivalence summary.

    Attributes
    ----------
    rows:
        Per-model audit results.
    divergent_fields:
        Union of every field that diverged on at least one model.
    compared:
        Number of models successfully compared.
    failed:
        Number of models that could not be built/traced for the audit.
    """

    rows: list[RowAudit]
    divergent_fields: set[str]
    compared: int
    failed: int

    @property
    def equivalent_fields(self) -> set[str]:
        """Return structural fields proven identical across every compared model."""

        comparable = set(TRACE_SUMMARY_COLUMNS) - PROVENANCE_COLUMNS
        return comparable - self.divergent_fields

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""

        return {
            "compared": self.compared,
            "failed": self.failed,
            "divergent_fields": sorted(self.divergent_fields),
            "equivalent_field_count": len(self.equivalent_fields),
            "rows": [
                {
                    "stable_id": r.stable_id,
                    "ok": r.ok,
                    "error": r.error,
                    "diff_fields": sorted(r.diff_fields),
                }
                for r in self.rows
            ],
        }


def _validation_config_summary(model: Any, example_input: Any, stable_id: str) -> dict[str, Any]:
    """Summarize a model from a VALIDATION-config trace via the worker observer path.

    Reproduces exactly how the menagerie validation worker observes its trace: it
    runs ``_validate_forward_pass_torch`` and reads the summary off the live trace
    inside the observer callback (pre-cleanup), so the audit compares the SAME object
    the unify wires into.

    Parameters
    ----------
    model:
        Constructed model (already ``eval()``).
    example_input:
        Example input for the model.
    stable_id:
        Model id for the summary row.

    Returns
    -------
    dict[str, Any]
        Summary row derived from the validation-config trace.
    """

    from torchlens.user_funcs import _validate_forward_pass_torch

    captured: dict[str, dict[str, Any]] = {}

    def observe(trace: Any) -> None:
        captured["row"] = summarize_trace(stable_id, trace, "")

    _validate_forward_pass_torch(
        model,
        example_input,
        validate_metadata=True,
        _trace_observer=observe,
    )
    if "row" not in captured:
        raise RuntimeError("validation observer did not fire")
    return captured["row"]


def _metadata_config_summary(model: Any, example_input: Any, stable_id: str) -> dict[str, Any]:
    """Summarize a model from a METADATA-config trace (the current re-trace path)."""

    trace = _trace_for_digest(model, example_input)
    return summarize_trace(stable_id, trace, "")


def audit_row(row: CatalogRow) -> RowAudit:
    """Audit one catalog row: build both trace configs and diff every summary field.

    Parameters
    ----------
    row:
        Catalog row to audit.

    Returns
    -------
    RowAudit
        Per-model result. ``ok`` is False (with ``error``) when either config cannot
        be built/traced -- those rows are excluded from the equivalence verdict rather
        than silently counted as agreeing.
    """

    try:
        example_input = build_input_for_row(row)
        model_v = instantiate_model(row)
        if hasattr(model_v, "eval"):
            model_v.eval()
        val_row = _validation_config_summary(model_v, example_input, row.stable_id)
        # Fresh instance for the metadata config: the validation forward pass mutates
        # nothing it does not restore, but a clean instance keeps the two paths
        # independent and matches the production split (separate processes).
        model_m = instantiate_model(row)
        if hasattr(model_m, "eval"):
            model_m.eval()
        meta_row = _metadata_config_summary(model_m, example_input, row.stable_id)
    except Exception as error:  # noqa: BLE001 -- audit must record, not crash
        return RowAudit(row.stable_id, ok=False, error=repr(error))

    diff_fields: dict[str, tuple[Any, Any]] = {}
    for column in TRACE_SUMMARY_COLUMNS:
        if column in PROVENANCE_COLUMNS:
            continue
        if val_row.get(column) != meta_row.get(column):
            diff_fields[column] = (val_row.get(column), meta_row.get(column))
    return RowAudit(row.stable_id, ok=True, diff_fields=diff_fields)


def audit_stable_id(stable_id: str, *, catalog_db: Any = CATALOG_DB) -> RowAudit:
    """Audit a single model by stable id."""

    rows = {r.stable_id: r for r in load_rows(db_path=catalog_db)}
    if stable_id not in rows:
        raise LookupError(f"stable_id not found: {stable_id}")
    return audit_row(rows[stable_id])


def stratified_sample(rows: Sequence[CatalogRow], sample_size: int) -> list[CatalogRow]:
    """Pick a deterministic stratified sample spanning zoos and domains.

    Round-robins across ``(zoo, domain)`` strata so the sample spans CPU-only and
    CUDA-eligible families, attention/conv/norm motifs, and container/dynamic graphs
    rather than over-weighting the largest zoo.

    Parameters
    ----------
    rows:
        Candidate catalog rows.
    sample_size:
        Target sample size.

    Returns
    -------
    list[CatalogRow]
        Up to ``sample_size`` rows, stable-id-sorted within each stratum for
        determinism.
    """

    strata: dict[tuple[str, str], list[CatalogRow]] = {}
    for row in rows:
        key = (row.zoo or "", row.domain or "")
        strata.setdefault(key, []).append(row)
    for bucket in strata.values():
        bucket.sort(key=lambda r: r.stable_id)
    ordered_keys = sorted(strata)
    sample: list[CatalogRow] = []
    index = 0
    while len(sample) < sample_size and any(strata[k] for k in ordered_keys):
        key = ordered_keys[index % len(ordered_keys)]
        if strata[key]:
            sample.append(strata[key].pop(0))
        index += 1
    return sample[:sample_size]


def audit_sample(
    *,
    sample_size: int = 60,
    catalog_db: Any = CATALOG_DB,
    stable_ids: Sequence[str] | None = None,
) -> AuditReport:
    """Run the equivalence audit over a stratified sample (or explicit ids).

    Parameters
    ----------
    sample_size:
        Number of models to sample when ``stable_ids`` is not given.
    catalog_db:
        Catalog SQLite path.
    stable_ids:
        Explicit ids to audit instead of sampling.

    Returns
    -------
    AuditReport
        Corpus-level equivalence summary.
    """

    rows = load_rows(db_path=catalog_db)
    if stable_ids:
        by_id = {r.stable_id: r for r in rows}
        selected = [by_id[s] for s in stable_ids if s in by_id]
    else:
        # Audit only buildable, non-quarantined rows where possible.
        candidates = [r for r in rows if not getattr(r, "quarantine", None)]
        selected = stratified_sample(candidates or rows, sample_size)

    results: list[RowAudit] = []
    divergent: set[str] = set()
    compared = 0
    failed = 0
    for row in selected:
        result = audit_row(row)
        results.append(result)
        if result.ok:
            compared += 1
            divergent.update(result.diff_fields)
        else:
            failed += 1
    return AuditReport(results, divergent, compared, failed)


def build_parser() -> argparse.ArgumentParser:
    """Build the audit CLI parser."""

    parser = argparse.ArgumentParser(
        description="Audit summary-field equivalence between validation and metadata traces"
    )
    parser.add_argument("--sample", type=int, default=60, help="stratified sample size")
    parser.add_argument("--stable-ids", nargs="*", help="explicit stable ids to audit")
    parser.add_argument("--json", type=str, help="write the full report JSON to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI; exit non-zero if a field outside the known set diverges."""

    args = build_parser().parse_args(argv)
    report = audit_sample(sample_size=args.sample, stable_ids=args.stable_ids)
    unexpected = report.divergent_fields - KNOWN_DIVERGENT_FIELDS
    print(f"compared={report.compared} failed={report.failed}")
    print(f"divergent_fields={sorted(report.divergent_fields)}")
    print(f"equivalent_field_count={len(report.equivalent_fields)}")
    if unexpected:
        print(f"UNEXPECTED DIVERGENT FIELDS (outside known set): {sorted(unexpected)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, default=str)
        print(f"wrote {args.json}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
