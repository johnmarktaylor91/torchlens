"""Resume machinery + the D16 signature for extraction v2 (lane F18).

The KNOWN-FIELDS signature builder, the manifest loader/creator, the D8
callable-identity resume rules, the field-by-field mismatch refusals, and
the disk-run preparation (trusted-prefix resolution, v1 migration entry,
completed-compatible no-op resume). The store pipeline itself lives in
:mod:`torchlens._extraction.engine`; this module never runs a forward.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .._data_substrate import (
    LEDGER_FILENAME,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SCHEMA_V2,
    STIMULUS_IDS_FILENAME,
    VALUE_REDUCTION_ALGORITHM_ID,
    VALUE_REDUCTION_ALGORITHM_VERSION,
    ArtifactWriter,
    compare_signatures,
    compute_model_identity,
    migrate_v1_artifact,
    read_trusted_rows,
    repair_ledger_tail,
    stimulus_ids_digest,
)
from .._errors import InvalidArgumentError, _actionable_message, _ActionableErrorMixin
from .._io import _json
from ..errors._base import ConfigurationError
from ..transforms._pipeline import OpaqueStep
from .callable_identity import classify_callable
from .selector_plan import SelectorPlan, selector_request_record
from .shards import shard_extension

if TYPE_CHECKING:
    from ..transforms import TransformPipeline
    from .engine import _RunPlan

__tl_layer__ = "L5"

#: Filename of the self-describing manifest inside an extraction directory.
MANIFEST_FILENAME = "manifest.json"

#: Native shard-format signature values (a manifest FIELD, so a later
#: format change is a value change, not a layout break).
NATIVE_FORMATS = {"pt": "pt_shards_v1", "safetensors": "safetensors_shards_v1"}

#: Maximum number of tensor elements sampled into the stimulus digest.
_DIGEST_SAMPLE_ELEMENTS = 4096

#: Closed D7 checksum-level vocabulary.
_CHECKSUM_LEVELS = ("fast", "crypto", "none")


class DatasetExtractionResumeError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when a resumable extraction artifact cannot be safely continued."""

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize an actionable extraction-resume refusal.

        Parameters
        ----------
        problem:
            Description of the artifact state and why it was rejected.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


# ---------------------------------------------------------------------------
# Kwarg validation
# ---------------------------------------------------------------------------


def validate_checksums_level(checksums: Any) -> str:
    """Validate the ``checksums=`` kwarg against the closed D7 vocabulary.

    Parameters
    ----------
    checksums:
        Requested level.

    Returns
    -------
    str
        The validated level.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_checksums_invalid`` outside the vocabulary.
    """

    if checksums in _CHECKSUM_LEVELS:
        return str(checksums)
    raise InvalidArgumentError(
        f"checksums= value {checksums!r} is not in the closed vocabulary "
        f"{_CHECKSUM_LEVELS} ('fast' = CRC-32 + on-device value reduction, "
        "'crypto' = blake2b file digests, 'none' = explicit recorded "
        "opt-out; verification refuses on 'none').",
        code="extraction_checksums_invalid",
        remedy="pass 'fast' (default), 'crypto', or 'none'",
        value=repr(checksums),
    )


# ---------------------------------------------------------------------------
# Collation + position policy
# ---------------------------------------------------------------------------


def _tensor_digest(stimuli: torch.Tensor) -> str:
    """Return a cheap sampled content digest for tensor stimuli.

    Parameters
    ----------
    stimuli:
        Stimulus tensor with a leading batch dimension.

    Returns
    -------
    str
        ``sha256:...`` digest string.
    """

    hasher = hashlib.sha256()
    hasher.update(repr(tuple(stimuli.shape)).encode())
    hasher.update(str(stimuli.dtype).encode())
    flat = stimuli.detach().reshape(-1)
    stride = max(1, flat.numel() // _DIGEST_SAMPLE_ELEMENTS)
    sample = flat[::stride].cpu().contiguous()
    hasher.update(sample.numpy().tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _stimuli_signature(stimuli: Any) -> dict[str, Any]:
    """Describe the stimulus set for the manifest signature.

    Parameters
    ----------
    stimuli:
        Tensor with a leading batch dimension or an iterable stimulus set.

    Returns
    -------
    dict[str, Any]
        Signature block; iterable identity is disclosed as unverifiable at
        signature level (the per-shard ``input_digest`` replay covers it).
    """

    if isinstance(stimuli, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(stimuli.shape),
            "dtype": str(stimuli.dtype),
            "digest": _tensor_digest(stimuli),
        }
    return {
        "kind": "iterable",
        "note": (
            "iterable stimulus identity is not verifiable at signature "
            "level; resume verifies the replayed prefix per shard through "
            "the ledgered input_digest"
        ),
    }


def _collate_signature(plan: _RunPlan) -> dict[str, Any]:
    """Build the signature's collate-policy disclosure (D9 + D16).

    Policy facts only (stable before the run starts); the resolved
    tokenizer's identity is an observation recorded at plan freeze.

    Parameters
    ----------
    plan:
        Resolved run configuration.

    Returns
    -------
    dict[str, Any]
        JSON-portable collate policy record.
    """

    if plan.collate is None:
        return {"kind": "default"}
    disclosure = getattr(plan.collate, "__tl_disclosure__", None)
    if isinstance(disclosure, Mapping):
        return dict(disclosure)
    identity = classify_callable(plan.collate, slot="collate")
    return {
        "kind": "user",
        "qualname": identity["qualname"],
        "classification": identity["classification"],
        "digest": identity["digest"],
    }


def _callable_identity_signature(plan: _RunPlan) -> dict[str, Any] | None:
    """Build the D8 callable-identity signature block.

    Classifies the raw callables the static chain cannot reconstruct: the
    user collate (when supplied) and every opaque transform step.

    Parameters
    ----------
    plan:
        Resolved run configuration.

    Returns
    -------
    dict[str, Any] | None
        ``{"slots": {...}, "pipeline_id": ...}`` or ``None`` when nothing
        needs classification and no pipeline id was asserted.
    """

    slots: dict[str, Any] = {}
    if plan.collate is not None and not getattr(plan.collate, "__tl_engine_collate__", False):
        slots["collate"] = classify_callable(plan.collate, slot="collate")
    chains: list[tuple[str, TransformPipeline | None]] = sorted(plan.pipelines.items())
    if plan.selector_pipeline is not None:
        chains.append(("*", plan.selector_pipeline))
    for key, pipeline in chains:
        if pipeline is None:
            continue
        for index, step in enumerate(pipeline.steps):
            if isinstance(step, OpaqueStep) and step.fn is not None:
                slots[f"transform:{key}:{index}"] = classify_callable(
                    step.fn, slot=f"transform[{key}][{index}]"
                )
    if not slots and plan.pipeline_id is None:
        return None
    return {"slots": slots, "pipeline_id": plan.pipeline_id}


def build_signature(plan: _RunPlan, model_identity_record: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 resume-compatibility signature (the D16 KNOWN-FIELDS).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    model_identity_record:
        The D6 identity record.

    Returns
    -------
    dict[str, Any]
        JSON-serializable signature compared field-by-field on resume.
    """

    shard_format = plan.shard_format or "safetensors"
    integrity: dict[str, Any] = {"checksums": plan.checksums}
    if plan.checksums == "fast":
        integrity["file_fact"] = "crc32_final_file_bytes"
        integrity["value_reduction"] = {
            "algorithm_id": VALUE_REDUCTION_ALGORITHM_ID,
            "algorithm_version": VALUE_REDUCTION_ALGORITHM_VERSION,
        }
    elif plan.checksums == "crypto":
        integrity["file_fact"] = "blake2b_final_file_bytes"
    selector_plan_record = None
    if plan.layers_kind == "selector":
        selector_plan_record = {"request": selector_request_record(plan.layers)}
    return {
        "schema_version": MANIFEST_SCHEMA_V2,
        "native_format": NATIVE_FORMATS[shard_format],
        "layer_plan": dict(plan.layer_plan),
        "layers_kind": plan.layers_kind,
        "batch_size": plan.batch_size,
        "transform_pipeline": plan.transform_record,
        "stimuli": _stimuli_signature(plan.stimuli),
        "stimulus_ids_digest": stimulus_ids_digest(plan.stimulus_ids),
        "model_identity": model_identity_record,
        "padding_side": "as_collated",
        "position_ids_source": "derived_or_refused",
        "model_mode": "eval_no_grad",
        "pool": plan.pool_record,
        "dtype_policy": plan.dtype_record,
        "ragged": plan.ragged,
        "integrity": integrity,
        "collate": _collate_signature(plan),
        "callable_identity": _callable_identity_signature(plan),
        "selector_plan": selector_plan_record,
    }


def _base_manifest(
    signature: dict[str, Any],
    stimulus_ids: list[str] | None,
    input_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a fresh in-progress v2 manifest document (the bounded header).

    Parameters
    ----------
    signature:
        Resume-compatibility signature block.
    stimulus_ids:
        Optional caller-supplied per-stimulus identifiers.
    input_preprocessing:
        The tvscope B4 provenance block; every NEW manifest carries one (an
        omitted block would be indistinguishable from an asserted-clean one),
        and it is allowed to say ``unknown``. ``None`` defaults to the
        honest undeclared block.

    Returns
    -------
    dict[str, Any]
        Bounded manifest header with no layer metadata or totals yet.
    """

    from .. import __version__
    from .._extraction_provenance import undeclared_input_block

    shard_format = "safetensors" if signature["native_format"] == "safetensors_shards_v1" else "pt"
    return {
        "schema": MANIFEST_SCHEMA_V2,
        "torchlens_version": __version__,
        "status": "in_progress",
        "signature": signature,
        "input_preprocessing": (
            input_preprocessing if input_preprocessing is not None else undeclared_input_block()
        ),
        "stimulus_provenance": {
            "order": (
                "row i of every shard, consumed in ledger order, is stimulus i "
                "in the caller's iteration order"
            ),
            "n_stimuli": None,
            "ids_recorded": stimulus_ids is not None,
            "ids_unique": (len(set(stimulus_ids)) == len(stimulus_ids))
            if stimulus_ids is not None
            else None,
            "ids_digest": signature.get("stimulus_ids_digest"),
        },
        "storage": {
            "shard_format": shard_format,
            "shard_pattern": f"batch_XXXXX{shard_extension(shard_format)}",
            "ledger": LEDGER_FILENAME,
        },
        "layers": None,
        "run": {},
        "totals": None,
        "ledger_digest": None,
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and structurally validate an extraction manifest.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json`` inside the extraction directory.

    Returns
    -------
    dict[str, Any]
        Parsed manifest document.

    Raises
    ------
    DatasetExtractionResumeError
        If the manifest is unparseable or not this module's schema.
    """

    try:
        # read_bounded, never json.loads(read_text()): the manifest is a
        # user-supplied artifact, so read_text() would materialize the whole
        # file before any ceiling could apply (an over-size manifest is an
        # allocation DoS).
        manifest = _json.read_bounded(manifest_path)
    except (OSError, ValueError) as exc:
        raise DatasetExtractionResumeError(
            f"Extraction manifest {str(manifest_path)!r} could not be parsed ({exc}).",
            code="extraction_manifest_invalid",
            remedy="delete the output directory and re-run the extraction from scratch",
            manifest_path=str(manifest_path),
        ) from exc
    known_schemas = (MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2)
    if not isinstance(manifest, dict) or manifest.get("schema") not in known_schemas:
        raise DatasetExtractionResumeError(
            f"Extraction manifest {str(manifest_path)!r} does not carry a known "
            f"schema {known_schemas} (found "
            f"{manifest.get('schema') if isinstance(manifest, dict) else type(manifest).__name__!r}).",
            code="extraction_manifest_invalid",
            remedy="delete the output directory and re-run the extraction from scratch",
            manifest_path=str(manifest_path),
        )
    return manifest


# ---------------------------------------------------------------------------
# Batch-zero freeze helpers
# ---------------------------------------------------------------------------


def validate_stimulus_ids(ids: list[str] | None) -> None:
    """Validate stimulus identifiers as non-empty strings (extract D2).

    Duplicates are LEGAL (recorded as ``ids_unique`` provenance); empty or
    non-string identifiers are not.

    Parameters
    ----------
    ids:
        Caller-supplied identifiers, or ``None``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_stimulus_ids_invalid`` naming the first offender.
    """

    if ids is None:
        return
    for position, item in enumerate(ids):
        if not isinstance(item, str) or not item:
            raise InvalidArgumentError(
                f"stimulus_ids[{position}] is {item!r}; identifiers are "
                "non-empty strings (duplicates are legal and recorded).",
                code="extraction_stimulus_ids_invalid",
                remedy="pass one non-empty string per stimulus",
                position=position,
                value=repr(item),
            )


def _apply_callable_identity_rules(
    recorded: Any, current: Any, *, continuation: bool, manifest_path: Path
) -> list[dict[str, Any]]:
    """Apply the D8 resume rules to the callable-identity blocks.

    A complete-class digest match resumes WITHOUT ceremony; a partial slot
    refuses unless ``pipeline_id=`` matches the recorded one (a resume-time
    -invented ID cannot retroactively attest old semantics); a
    complete-class digest MISMATCH (the deliberate-refactor case) needs the
    same recorded ``pipeline_id`` and is recorded as ASSERTED with both
    digests.

    Parameters
    ----------
    recorded:
        The artifact's ``callable_identity`` block (or ``None``).
    current:
        The resuming run's block (or ``None``).
    continuation:
        Whether new forwards will run (completed compatible artifacts skip
        the assertion ceremony — no new bytes are written).
    manifest_path:
        Manifest path for refusal text.

    Returns
    -------
    list[dict[str, Any]]
        Audit rows for every asserted override (empty when measured).

    Raises
    ------
    DatasetExtractionResumeError
        ``extraction_resume_signature_mismatch`` on slot-set drift;
        ``extraction_resume_pipeline_id_invalid`` on an invented ID;
        ``extraction_resume_callable_mismatch`` on an unattested
        complete-class digest mismatch;
        ``extraction_resume_opaque_transform`` on an unattested partial
        slot.
    """

    recorded_block = recorded if isinstance(recorded, Mapping) else {}
    current_block = current if isinstance(current, Mapping) else {}
    recorded_slots = dict(recorded_block.get("slots") or {})
    current_slots = dict(current_block.get("slots") or {})
    recorded_pid = recorded_block.get("pipeline_id")
    current_pid = current_block.get("pipeline_id")
    if set(recorded_slots) != set(current_slots):
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(manifest_path.parent)!r} records "
            f"callable-identity slots {sorted(recorded_slots)} but the "
            f"resuming run supplies {sorted(current_slots)}; the callable "
            "surface itself changed.",
            code="extraction_resume_signature_mismatch",
            remedy="re-run with the artifact's original collate/transform callables",
            mismatched_fields=["callable_identity"],
            recorded_slots=sorted(recorded_slots),
            current_slots=sorted(current_slots),
        )
    audits: list[dict[str, Any]] = []
    for slot in sorted(recorded_slots):
        rec = recorded_slots[slot] or {}
        cur = current_slots[slot] or {}
        both_complete = (
            rec.get("classification") == "complete" and cur.get("classification") == "complete"
        )
        if both_complete and rec.get("digest") == cur.get("digest"):
            continue  # measured identity: resumes without pipeline_id ceremony
        if not continuation:
            continue  # completed artifact: no new forwards, nothing to attest
        if current_pid is not None and recorded_pid is None:
            raise DatasetExtractionResumeError(
                f"pipeline_id={current_pid!r} was supplied at resume time but "
                "the artifact's prefix recorded NO pipeline_id; a newly "
                "invented ID cannot retroactively attest old callable "
                "semantics (extract D8, strict continuity).",
                code="extraction_resume_pipeline_id_invalid",
                remedy=(
                    "start a fresh artifact, or set pipeline_id= from the "
                    "FIRST run so later resumes can attest against it"
                ),
                slot=slot,
                supplied_pipeline_id=current_pid,
            )
        if current_pid is not None and current_pid == recorded_pid:
            audits.append(
                {
                    "kind": "callable_identity_asserted",
                    "slot": slot,
                    "pipeline_id": current_pid,
                    "recorded_digest": rec.get("digest"),
                    "current_digest": cur.get("digest"),
                    "recorded_classification": rec.get("classification"),
                    "current_classification": cur.get("classification"),
                }
            )
            continue
        if both_complete:
            raise DatasetExtractionResumeError(
                f"Callable slot {slot!r} classifies COMPLETE on both sides "
                "but the digests differ — the callable's observable "
                "behavior changed since the prefix was written.",
                code="extraction_resume_callable_mismatch",
                remedy=(
                    "resume with the original callable, extract into a "
                    "fresh directory, or attest the deliberate refactor "
                    "with the artifact's recorded pipeline_id="
                ),
                slot=slot,
                recorded_digest=rec.get("digest"),
                current_digest=cur.get("digest"),
            )
        opaque = sorted(
            set(rec.get("opaque_references") or []) | set(cur.get("opaque_references") or [])
        )
        raise DatasetExtractionResumeError(
            f"Callable slot {slot!r} classifies PARTIAL (opaque references: "
            f"{opaque[:5]}); an opaque reference is a disclosure, not a "
            "proof, so the continuation cannot be verified to produce the "
            "prefix's numbers.",
            code="extraction_resume_opaque_transform",
            remedy=(
                "register the callable under a versioned name "
                "(torchlens.transforms.register_transform), supply the "
                "artifact's recorded pipeline_id=, or extract into a fresh "
                "directory"
            ),
            slot=slot,
            opaque_references=opaque,
        )
    return audits


def _refuse_signature_mismatch(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    manifest_path: Path,
    *,
    continuation: bool,
) -> list[dict[str, Any]]:
    """Compare v2 signatures field by field and refuse typed on mismatch (D16).

    ``model_identity`` mismatches get their own code (the T-MODELSWAP
    hazard); ``callable_identity`` is governed by the D8 rules (which may
    permit an ASSERTED continuation and return its audit rows).

    Parameters
    ----------
    recorded:
        The artifact's recorded signature.
    current:
        The signature of the run asking to resume.
    manifest_path:
        Manifest path, for the refusal message.
    continuation:
        Whether new forwards will run.

    Returns
    -------
    list[dict[str, Any]]
        Callable-identity audit rows (recorded at terminal status).

    Raises
    ------
    DatasetExtractionResumeError
        ``extraction_resume_model_identity_mismatch`` /
        ``extraction_resume_signature_mismatch`` and the D8 refusals.
    """

    audits = _apply_callable_identity_rules(
        recorded.get("callable_identity"),
        current.get("callable_identity"),
        continuation=continuation,
        manifest_path=manifest_path,
    )
    mismatched = [
        field for field in compare_signatures(recorded, current) if field != "callable_identity"
    ]
    if not mismatched:
        return audits
    if "model_identity" in mismatched:
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(manifest_path.parent)!r} records a "
            f"different MODEL IDENTITY than the resuming run (all mismatched "
            f"fields: {mismatched}). Continuing would silently mix the "
            "artifact's activations with a different model's — the exact "
            "failure a random-init resume used to complete with.",
            code="extraction_resume_model_identity_mismatch",
            remedy=(
                "resume with the exact model state that produced the artifact "
                "(same checkpoint, same in-place edits), or extract into a "
                "fresh directory"
            ),
            mismatched_fields=mismatched,
            recorded_identity=recorded.get("model_identity"),
            requested_identity=current.get("model_identity"),
        )
    raise DatasetExtractionResumeError(
        f"Extraction artifact at {str(manifest_path.parent)!r} was produced by a "
        f"different run configuration (mismatched signature fields: {mismatched}).",
        code="extraction_resume_signature_mismatch",
        remedy=(
            "re-run with the artifact's original layers, batch_size, transform, "
            "collate, pool, dtype, ragged, stimuli, and stimulus_ids, or "
            "delete the output directory to start fresh"
        ),
        mismatched_fields=mismatched,
        recorded_signature=dict(recorded),
        requested_signature=dict(current),
    )


def _completed_prefix(manifest: dict[str, Any], container_path: Path) -> list[dict[str, Any]]:
    """Return the v1 ledgered shard prefix whose files are all present.

    Parameters
    ----------
    manifest:
        Parsed v1 manifest document.
    container_path:
        Extraction directory containing the shards.

    Returns
    -------
    list[dict[str, Any]]
        Contiguous v1 batch rows verified present on disk.
    """

    from .._data_substrate import shard_filename

    prefix: list[dict[str, Any]] = []
    for row in manifest.get("batches") or []:
        expected_name = shard_filename(int(row["index"]))
        if row.get("file") != expected_name or not (container_path / expected_name).exists():
            break
        prefix.append(row)
    return prefix


def _clean_orphan_tmp_files(container_path: Path) -> None:
    """Remove leftover atomic-write temp files from a crashed run.

    Parameters
    ----------
    container_path:
        Extraction directory to sweep.
    """

    for tmp_path in container_path.glob("*.tmp"):
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def _prepare_disk_run(
    plan: _RunPlan, container_path: Path, resume: bool
) -> tuple[_RunPlan, ArtifactWriter, list[dict[str, Any]], list[Path] | None, dict[str, Any]]:
    """Prepare the v2 artifact writer and resolve the resume state.

    Parameters
    ----------
    plan:
        Resolved run configuration (its ``shard_format`` may be ``None``;
        the effective format resolves here: the recorded format on resume,
        safetensors on fresh runs).
    container_path:
        Extraction directory.
    resume:
        Whether to continue from an existing ledger.

    Returns
    -------
    tuple
        ``(effective_plan, writer, completed_rows, complete_paths,
        prep_state)``: the plan with its resolved shard format, the bound
        writer, the trusted ledger prefix, the final shard paths when the
        artifact is already complete and compatible, and preparation facts
        (resume audit rows, adopted selector plan).

    Raises
    ------
    DatasetExtractionResumeError
        On unmanifested shard directories, signature/model-identity/
        callable-identity mismatches, or replay input mismatches.
    torchlens._data_substrate.ExtractionArtifactError
        On a broken ledger prefix or an unacknowledged in-progress v1
        artifact.
    """

    container_path.mkdir(parents=True, exist_ok=True)
    manifest_path = container_path / MANIFEST_FILENAME
    prep_state: dict[str, Any] = {"resume_audit": [], "selector_plan_record": None}

    if resume and manifest_path.exists():
        existing = load_manifest(manifest_path)
        if existing.get("schema") == MANIFEST_SCHEMA_V1:
            existing, _migrated_rows = migrate_v1_artifact(
                container_path, existing, acknowledge_in_progress=plan.acknowledge_v1_prefix
            )
        recorded_format = str((existing.get("storage") or {}).get("shard_format", "pt"))
        effective_format = plan.shard_format or recorded_format
        plan = dataclasses.replace(plan, shard_format=effective_format)
        identity = resolve_model_identity(plan.model, plan.model_identity)
        signature = build_signature(plan, identity)
        rows = read_trusted_rows(container_path)
        totals = existing.get("totals") or {}
        is_complete = existing.get("status") == "complete" and len(rows) == totals.get("n_shards")
        migrated_v1 = bool((existing.get("migration") or {}).get("migrated_from"))
        if migrated_v1:
            # A migrated prefix's semantic fields are UNRECORDED_V1 and the
            # compare skips them; the fields the migration DID carry still
            # compare field-by-field.
            prep_state["resume_audit"].extend(
                []
                if existing.get("status") == "complete"
                else [{"kind": "v1_in_progress_acknowledged", "semantics": "asserted_not_measured"}]
            )
        audits = _refuse_signature_mismatch(
            existing.get("signature") or {},
            signature,
            manifest_path,
            continuation=not is_complete,
        )
        prep_state["resume_audit"].extend(audits)
        if is_complete:
            # A completed compatible resume is a true no-op: the model is
            # neither moved to a device nor mode-flipped (extract memo D3).
            return (
                plan,
                ArtifactWriter(container_path, existing),
                rows,
                [container_path / str(row["file"]) for row in rows],
                prep_state,
            )
        if plan.input_opaque:
            raise DatasetExtractionResumeError(
                "Resuming this artifact would run forwards through an OPAQUE "
                "INPUT transform (tvscope B5); an opaque callable's identity "
                "is a disclosure, not a proof, so a continuation cannot be "
                "verified to produce the same numbers as the prefix "
                "(transforms memo decision 14).",
                code="extraction_resume_opaque_transform",
                remedy=(
                    "pass a resolution-backed input transform "
                    "(torchlens.preprocessing.resolve) or extract into a "
                    "fresh directory"
                ),
                input_opaque=True,
            )
        recorded_selector = (existing.get("run") or {}).get("selector_plan")
        if recorded_selector and recorded_selector.get("entries"):
            prep_state["selector_plan_record"] = SelectorPlan(
                tuple(
                    (str(entry["key"]), str(entry["label"]), entry.get("site_key"))
                    for entry in recorded_selector["entries"]
                ),
                dict(recorded_selector.get("request") or {}),
            )
        _clean_orphan_tmp_files(container_path)
        # A torn final ledger line is crash debris: cleared before appends
        # resume, or the next commit would concatenate into it.
        repair_ledger_tail(container_path)
        existing["status"] = "in_progress"
        # No manifest rewrite for a continuation: creation, batch-zero plan
        # freeze, and terminal status are the only three writes (D1).
        return plan, ArtifactWriter(container_path, existing), rows, None, prep_state

    if resume and (
        any(container_path.glob("batch_*.pt")) or any(container_path.glob("batch_*.safetensors"))
    ):
        raise DatasetExtractionResumeError(
            f"Output directory {str(container_path)!r} contains batch shards "
            "but no manifest; it predates resumable extraction or lost its "
            "ledger, so completed work cannot be verified.",
            code="extraction_resume_unmanifested_dir",
            remedy="delete the output directory (or point output_dir at a fresh one) and re-run",
            output_dir=str(container_path),
        )

    # Fresh run (resume=False, or resume=True into an empty directory): any
    # prior artifact machinery in the directory belongs to a different run
    # identity, so the append-only ledger and write-once sidecar reset with
    # the manifest.
    plan = dataclasses.replace(plan, shard_format=plan.shard_format or "safetensors")
    _clean_orphan_tmp_files(container_path)
    for stale in (LEDGER_FILENAME, STIMULUS_IDS_FILENAME):
        with contextlib.suppress(OSError):
            (container_path / stale).unlink()
    identity = resolve_model_identity(plan.model, plan.model_identity)
    manifest = _base_manifest(build_signature(plan, identity), plan.stimulus_ids, plan.input_block)
    writer = ArtifactWriter(container_path, manifest)
    writer.write_manifest()
    if plan.stimulus_ids is not None:
        writer.write_stimulus_ids_sidecar(plan.stimulus_ids)
    return plan, writer, [], None, prep_state


def resolve_model_identity(model: nn.Module, model_identity: Any) -> dict[str, Any]:
    """Resolve the ``model_identity=`` kwarg into the D6 identity record.

    Parameters
    ----------
    model:
        The model being harvested.
    model_identity:
        ``"measured"`` | ``"none"`` | an assertion Mapping.

    Returns
    -------
    dict[str, Any]
        The JSON-portable identity record for the run signature.
    """

    if isinstance(model_identity, Mapping):
        return compute_model_identity(model, level="asserted", assertion=dict(model_identity))
    return compute_model_identity(model, level=str(model_identity))
