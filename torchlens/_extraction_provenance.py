"""Input-preprocessing provenance seam for dataset extraction (tvscope B4/B5).

Split out of :mod:`torchlens.dataset_extraction` under the R43 size ratchet:
the manifest ``input_preprocessing`` block builders, the ``input_transform=``
/ ``input_provenance=`` coercion door, the per-batch input applier, and the
legacy-tolerant block reader. The engine module re-exports the public names
(``INPUT_PREPROCESSING_SCHEMA``, ``input_preprocessing_of``) so caller
spellings are unchanged.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

__tl_layer__ = "L5"

#: Schema id of the manifest input-preprocessing block (tvscope B4).
INPUT_PREPROCESSING_SCHEMA = "tl_input_preprocessing_v1"


def _preprocessing_versions() -> dict[str, str | None]:
    """Collect the upstream versions the provenance block discloses (B4).

    Returns
    -------
    dict[str, str | None]
        Installed versions of torchlens and the authority-shipping libraries
        (``None`` when a library is absent).
    """

    import importlib.metadata as _metadata

    from torchlens import __version__

    versions: dict[str, str | None] = {"torchlens": __version__}
    for library in ("torch", "torchvision", "timm", "transformers"):
        try:
            versions[library] = _metadata.version(library)
        except Exception:
            versions[library] = None
    return versions


def _input_preprocessing_block(
    authority: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    verdict: str,
    unknown_reasons: list[str],
    applied_by: str | None,
) -> dict[str, Any]:
    """Assemble the schema-versioned manifest block (tvscope B4/D10).

    The block is ALWAYS present in new manifests -- a manifest that omitted
    it would be indistinguishable from one asserting nothing was done -- and
    it is allowed to say UNKNOWN.

    Parameters
    ----------
    authority:
        JSON-portable authority record, or ``None`` when none was declared.
    audit:
        JSON-portable :class:`torchlens.preprocessing.PreprocessingAudit`
        report, or ``None``.
    verdict:
        ``"verified"`` / ``"mismatch"`` / ``"unknown"``.
    unknown_reasons:
        Why the verdict is unknown, when it is.
    applied_by:
        ``"extract_dataset"`` when this run applied the input transform,
        ``"caller"`` for provenance-only stamps, ``None`` when undeclared.

    Returns
    -------
    dict[str, Any]
        The ``input_preprocessing`` manifest block.
    """

    return {
        "schema": INPUT_PREPROCESSING_SCHEMA,
        "authority": authority,
        "audit": audit,
        "verdict": verdict,
        "unknown_reasons": list(unknown_reasons),
        "versions": _preprocessing_versions(),
        "applied_by": applied_by,
    }


def undeclared_input_block() -> dict[str, Any]:
    """The block a run with no declared input path carries (honest unknown)."""

    return _input_preprocessing_block(
        authority=None,
        audit=None,
        verdict="unknown",
        unknown_reasons=["input_preprocessing_undeclared"],
        applied_by=None,
    )


def coerce_input_preprocessing(
    input_transform: Any, input_provenance: Any
) -> tuple[Any, dict[str, Any], Any, bool]:
    """Coerce the B5 input path into (callable, block, identity, opaque).

    Parameters
    ----------
    input_transform:
        ``None`` | a :class:`torchlens.preprocessing.Resolution` (the golden
        path: callable and provenance travel together) | a bare callable
        (identification-only disclosure; makes the artifact non-resumable
        across interruptions, mirroring opaque output transforms).
    input_provenance:
        ``None`` | :class:`torchlens.preprocessing.PreprocessingAudit` |
        :class:`torchlens.preprocessing.Resolution` |
        ``ResolvedPreprocessing`` -- a provenance-only stamp for stimuli the
        caller preprocessed outside this run. When both arguments carry
        provenance, ``input_provenance`` wins (it is the richer claim).

    Returns
    -------
    tuple[Any, dict[str, Any], Any, bool]
        The transform callable (or ``None``), the manifest block, the
        JSON-portable signature identity (or ``None`` when no transform
        runs), and whether the transform is opaque.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        When a declaration-only resolution (no callable) is passed as
        ``input_transform``.
    """

    from torchlens._errors import InvalidArgumentError
    from torchlens.preprocessing import PreprocessingAudit, Resolution, audit as _audit

    transform_callable: Any = None
    identity: Any = None
    opaque = False
    authority_json: dict[str, Any] | None = None
    audit_json: dict[str, Any] | None = None
    verdict = "unknown"
    unknown_reasons: list[str] = []
    applied_by: str | None = None

    if input_transform is not None:
        applied_by = "extract_dataset"
        if isinstance(input_transform, Resolution):
            if input_transform.transform is None:
                raise InvalidArgumentError(
                    "input_transform received a declaration-only resolution "
                    f"(source {input_transform.record.source!r}) that carries "
                    "no callable, so there is nothing to apply.",
                    code="extraction_input_transform_missing_callable",
                    remedy=(
                        "resolve an authority that ships a transform "
                        "(weights preset, processor, timm config), or apply "
                        "your own transform and pass the resolution as "
                        "input_provenance= instead"
                    ),
                    source=input_transform.record.source,
                )
            transform_callable = input_transform.transform
            report = _audit(input_transform, input_transform)
            authority_json = report.to_json()["authority"]
            audit_json = report.to_json()
            verdict = report.verdict
            unknown_reasons = list(report.unknown_reasons)
            identity = {
                "kind": "resolved",
                "source": input_transform.record.source,
                "identifier": input_transform.record.identifier,
                "config_digest": "sha256:"
                + hashlib.sha256(
                    _canonical_json(input_transform.record.config).encode()
                ).hexdigest(),
            }
        else:
            transform_callable = input_transform
            opaque = True
            from torchlens.data_classes.trace import _scrubbed_transform_repr

            transform_repr = _scrubbed_transform_repr(input_transform) or "<transform>"
            identity = {"kind": "opaque", "repr": transform_repr}
            verdict = "unknown"
            unknown_reasons = ["opaque_input_transform"]

    if input_provenance is not None:
        applied_by = applied_by or "caller"
        if isinstance(input_provenance, PreprocessingAudit):
            audit_json = input_provenance.to_json()
            authority_json = audit_json["authority"]
            verdict = input_provenance.verdict
            unknown_reasons = list(input_provenance.unknown_reasons)
        elif isinstance(input_provenance, Resolution):
            record = input_provenance.record
            authority_json = {
                "source": record.source,
                "identifier": record.identifier,
                "verified": bool(record.verified),
                "status": input_provenance.status,
                "config": _json_safe(record.config),
                "description": record.description,
            }
            if verdict == "unknown" and "applied_not_audited" not in unknown_reasons:
                unknown_reasons.append("applied_not_audited")
        else:
            record = input_provenance
            authority_json = {
                "source": getattr(record, "source", "unknown"),
                "identifier": getattr(record, "identifier", "unknown"),
                "verified": bool(getattr(record, "verified", False)),
                "status": getattr(record, "status", "unknown"),
                "config": _json_safe(getattr(record, "config", {})),
                "description": getattr(record, "description", ""),
            }
            if verdict == "unknown" and "applied_not_audited" not in unknown_reasons:
                unknown_reasons.append("applied_not_audited")

    if input_transform is None and input_provenance is None:
        return None, undeclared_input_block(), None, False
    block = _input_preprocessing_block(
        authority=authority_json,
        audit=audit_json,
        verdict=verdict,
        unknown_reasons=unknown_reasons,
        applied_by=applied_by,
    )
    return transform_callable, block, identity, opaque


def _canonical_json(value: Any) -> str:
    """Canonical JSON for digesting (sorted keys, minimal separators)."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    """Best-effort JSON coercion (tuples -> lists, non-portables -> repr)."""

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def apply_input_transform(fn: Any, batch: Any) -> Any:
    """Apply the B5 input transform to one collated batch.

    Item-level transforms (torchvision presets, timm transforms) run per
    item and stack; batch-native processors (HF image processors, marked
    ``_tl_batch_input``) get the whole list in one call; tensor batches are
    applied unary. The stimulus-row contract is guarded: a transform that
    changes the row count would silently mislabel every downstream row.

    Parameters
    ----------
    fn:
        The coerced input-transform callable, or ``None``.
    batch:
        One collated batch (item list or tensor slice).

    Returns
    -------
    Any
        The transformed batch.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        When a tensor-valued transform changes the stimulus-row count.
    """

    if fn is None:
        return batch
    n_rows: int | None = None
    if isinstance(batch, torch.Tensor):
        n_rows = int(batch.shape[0])
        result: Any = fn(batch)
    elif isinstance(batch, list) and batch and not isinstance(batch[0], torch.Tensor):
        n_rows = len(batch)
        if getattr(fn, "_tl_batch_input", False):
            result = fn(batch)
        else:
            first = fn(batch[0])
            if isinstance(first, torch.Tensor):
                rest = [fn(item) for item in batch[1:]]
                result = torch.stack([first, *rest], dim=0)
            else:
                result = fn(batch)
    else:
        result = fn(batch)
    if n_rows is not None and isinstance(result, torch.Tensor) and int(result.shape[0]) != n_rows:
        from torchlens._errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"input_transform changed the stimulus-row count ({n_rows} in, "
            f"{int(result.shape[0])} out); every downstream row would be "
            "mislabeled.",
            code="extraction_input_transform_row_mismatch",
            remedy="return exactly one transformed row per stimulus, in order",
            rows_in=n_rows,
            rows_out=int(result.shape[0]),
        )
    return result


def input_preprocessing_of(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read a manifest's input-preprocessing block (tvscope B4/D10).

    Legacy artifacts (v1, and v2 written before the block existed) read as
    UNKNOWN with the legacy reason disclosed -- an absent block is never
    treated as an assertion that nothing was done, and resume never grafts
    new provenance onto old rows.

    Parameters
    ----------
    manifest:
        Parsed extraction ``manifest.json`` document.

    Returns
    -------
    dict[str, Any]
        The ``tl_input_preprocessing_v1`` block; synthesized legacy-unknown
        (with ``"legacy": True``) when the artifact predates it.
    """

    block = manifest.get("input_preprocessing")
    if isinstance(block, dict) and block.get("schema") == INPUT_PREPROCESSING_SCHEMA:
        return block
    return {
        "schema": INPUT_PREPROCESSING_SCHEMA,
        "authority": None,
        "audit": None,
        "verdict": "unknown",
        "unknown_reasons": ["legacy_artifact_predates_input_preprocessing_block"],
        "versions": None,
        "applied_by": None,
        "legacy": True,
    }


__all__ = [
    "INPUT_PREPROCESSING_SCHEMA",
    "apply_input_transform",
    "coerce_input_preprocessing",
    "input_preprocessing_of",
    "undeclared_input_block",
]
