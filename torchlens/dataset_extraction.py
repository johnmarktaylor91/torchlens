"""Batched dataset extraction with a self-describing, resumable disk artifact.

This module is the public face of the extraction v2 runtime (extract memo
items 4-19; the engine lives in :mod:`torchlens._extraction`). In disk mode
(``path=``) the artifact directory is the v2 layout (extract D1): a
BOUNDED ``manifest.json`` (written at creation, once when batch zero freezes
the plan, and at terminal status — never per shard), an APPEND-ONLY fsynced
``ledger.jsonl`` with one line per committed shard, a write-once ordered
``stimulus_ids.json`` sidecar, and immutable batch-major shards
(safetensors by default; ``.pt`` stays readable forever and writable via
``shard_format="pt"``). The commit protocol is validate -> temp shard ->
flush/fsync -> atomic rename -> append/fsync ledger row.

Every batch travels the ONE manifested store pipeline (extract D13)::

    collate -> position/mask validation -> model call -> selected output
            -> pool (captured dtype, raw axes, mask) -> postprocess
            -> row + logical-shape validation -> per-key save-dtype cast
            -> CPU-contiguous snapshot -> writer

The run signature carries MODEL IDENTITY (extract D6), the collate policy,
callable identity (extract D8), pool/dtype/ragged policies, and the
selector request — all compared FIELD BY FIELD on resume (extract D16).

Read artifacts back through :func:`open_extraction` (the documented lazy
path) or :func:`load_extraction` (the eager compatibility wrapper behind an
exact byte guard); export with :func:`export_extraction`.

Every spelling introduced here is DOCUMENTED-UNSTABLE pending the
naming/UI sprint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ._data_substrate import (
    shard_filename as _shard_filename,  # noqa: F401 - engine-internal alias
    stimulus_ids_digest,
)
from ._errors import InvalidArgumentError
from ._extraction import (
    BatchEnvelope,
    RaggedBatch,
    export_extraction,
    hf_collate,
    is_selector_request,
    open_extraction,
    register_pure_module,
    resolve_dtype_policy,
    resolve_pool_policy,
    validate_output_keys,
    validate_ragged_mode,
    validate_shard_format,
)
from ._extraction.engine import (
    _RunPlan,
    run_in_memory,
    run_to_disk,
)
from ._extraction.intervened import (
    InterventionPlan,
    coerce_intervention,
    refuse_intervened_selector_layers,
)
from ._extraction.resume import (
    MANIFEST_FILENAME,
    load_manifest as _load_manifest,
    validate_checksums_level,
    validate_stimulus_ids,
)
from ._extraction.token_store import (
    TokenRows,
    refuse_input_ids_collision,
    validate_store_input_ids,
)
from ._extraction_provenance import (
    coerce_input_preprocessing,
    input_preprocessing_of,
)
from .transforms import TransformPipeline
from .transforms._coerce import coerce_transform, coerce_transform_mapping
from .transforms._pipeline import pipeline_record as _pipeline_record


def _coerce_transform_slot(
    transform: Any, layer_plan: dict[str, str]
) -> tuple[dict[str, TransformPipeline | None], Any]:
    """Route the ``output_transform=`` slot through the one coercion door (memo B1).

    Parameters
    ----------
    transform:
        ``None`` | unary callable | registered name | spec | ordered
        sequence | per-site Mapping.
    layer_plan:
        Normalized ``output key -> layer lookup`` plan.

    Returns
    -------
    tuple[dict[str, TransformPipeline | None], Any]
        One coerced chain (or ``None``) per output key, and the
        JSON-portable signature record.
    """

    output_keys = list(layer_plan)
    if isinstance(transform, Mapping):
        pipelines = coerce_transform_mapping(transform, output_keys)
        record: Any = {
            "per_site": {key: _pipeline_record(chain) for key, chain in pipelines.items()}
        }
        return pipelines, record
    pipeline = coerce_transform(transform)
    return dict.fromkeys(output_keys, pipeline), _pipeline_record(pipeline)


def _refuse_disk_only_options(kwargs_in_memory: dict[str, Any]) -> None:
    """Refuse disk-only options in in-memory mode (false affordances).

    Parameters
    ----------
    kwargs_in_memory:
        ``{option_name: offending_value}`` of the non-default disk-only
        options the caller passed without ``path``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_disk_only_option`` naming every offender.
    """

    if not kwargs_in_memory:
        return
    raise InvalidArgumentError(
        f"Options {sorted(kwargs_in_memory)} affect only the disk artifact "
        "(shards, ledger, resume signature); without path= they could "
        "neither affect nor accompany the in-memory result — a false "
        "affordance.",
        code="extraction_disk_only_option",
        remedy="pass path= (disk mode), or drop these options",
        options={key: repr(value) for key, value in kwargs_in_memory.items()},
    )


def _resolve_token_and_intervention(
    intervene: Any, store_input_ids: Any, layers: Any
) -> tuple[InterventionPlan | None, bool]:
    """Resolve the ``intervene=`` and ``store_input_ids=`` slots.

    Parameters
    ----------
    intervene:
        ``None`` or an intervention spec.
    store_input_ids:
        The token-id storage flag.
    layers:
        The caller's ``layers=`` request.

    Returns
    -------
    tuple[InterventionPlan | None, bool]
        The resolved intervention (``None`` = clean) and the flag.
    """

    keep_input_ids = validate_store_input_ids(store_input_ids)
    if keep_input_ids and isinstance(layers, Mapping):
        refuse_input_ids_collision(layers.keys())
    intervention = coerce_intervention(intervene)
    if intervention is not None and is_selector_request(layers):
        refuse_intervened_selector_layers()
    return intervention, keep_input_ids


def _validate_run_options(  # noqa: PLR0913 - validates the public knob surface one-to-one
    *,
    batch_size: int,
    output_dir: str | Path | None,
    resume: bool,
    stimulus_ids: Iterable[str] | None,
    collate: Any,
    ragged: str,
    shard_format: str | None,
    checksums: str,
    pipeline_id: str | None,
) -> tuple[str, str]:
    """Validate the run's option surface before any work happens.

    Parameters
    ----------
    batch_size:
        Requested batch size.
    output_dir:
        Disk-mode destination, or ``None`` (in-memory mode).
    resume:
        Continue an interrupted disk-mode run from its last completed shard.
    stimulus_ids:
        Explicit stimulus identifiers accompanying the inputs (disk mode only).
    collate:
        Batch collation callable, validated against the in-memory gate.
    ragged:
        Ragged-output policy token from its closed vocabulary.
    shard_format:
        Shard payload format token, or ``None`` for the default.
    checksums:
        Checksum level token from its closed vocabulary.
    pipeline_id:
        Caller-supplied run signature component, or ``None``.

    Returns
    -------
    tuple[str, str]
        The validated ``(ragged_mode, checksums_level)``.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if resume and output_dir is None:
        raise InvalidArgumentError(
            "resume=True requires path: only disk-mode extraction leaves "
            "a shard ledger to resume from.",
            code="extraction_resume_requires_output_dir",
            remedy="pass path= (disk mode) or drop resume=True",
        )
    if stimulus_ids is not None and output_dir is None:
        raise InvalidArgumentError(
            "stimulus_ids= requires path= (disk mode): in-memory extraction returns "
            "bare tensors with no manifest, so validated identifiers could "
            "neither affect nor accompany the result (a false affordance).",
            code="extraction_stimulus_ids_in_memory_unsupported",
            remedy=(
                "pass path= to record stimulus identity in the manifest, or drop stimulus_ids="
            ),
        )
    ragged_mode = validate_ragged_mode(ragged)
    if ragged_mode == "as_captured":
        raise InvalidArgumentError(
            "ragged='as_captured' (byte-exact batch-padded reproduction) is "
            "reserved and not yet available; the enum admits it so adding "
            "the mode later is a value change, never a signature break.",
            code="extraction_as_captured_unavailable",
            remedy="use ragged='trim' (true per-stimulus extents) or pool=",
        )
    checksums_level = validate_checksums_level(checksums)
    if shard_format is not None:
        validate_shard_format(shard_format)
    if collate is not None and not callable(collate):
        raise InvalidArgumentError(
            f"collate= value {collate!r} is not callable.",
            code="extraction_collate_invalid",
            remedy="pass a callable (e.g. torchlens.dataset_extraction.hf_collate(...))",
            value=repr(collate),
        )
    if output_dir is None:
        candidates = {
            "shard_format": shard_format if shard_format is not None else None,
            "checksums": checksums_level if checksums_level != "fast" else None,
            "pipeline_id": pipeline_id,
            "ragged": ragged_mode if ragged_mode != "refuse" else None,
        }
        _refuse_disk_only_options(
            {name: value for name, value in candidates.items() if value is not None}
        )
    return ragged_mode, checksums_level


def extract_dataset(  # noqa: PLR0913 - the memo-specified public knob surface (extract D1-D16)
    model: nn.Module,
    stimuli: Any,
    layers: Any,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    path: str | Path | None = None,
    output_transform: Any = None,
    progress: bool = True,
    *,
    resume: bool = False,
    stimulus_ids: Iterable[str] | None = None,
    model_identity: Any = "measured",
    input_transform: Any = None,
    input_provenance: Any = None,
    collate: Any = None,
    pool: Any = None,
    dtype: Any = None,
    ragged: str = "refuse",
    shard_format: str | None = None,
    checksums: str = "fast",
    pipeline_id: str | None = None,
    intervene: Any = None,
    store_input_ids: bool = False,
) -> dict[str, torch.Tensor] | list[Path]:
    """Extract outputs from an iterable dataset in batches.

    Row ``i`` of every stored tensor corresponds to stimulus ``i`` in
    iteration order; shards are consumed in ledger order.

    Parameters
    ----------
    model:
        PyTorch model to run.
    stimuli:
        Tensor with a leading batch dimension, or an iterable of stimulus
        items (tensors, containers, or TEXT strings — text routes through
        the once-per-run tokenizer with padding and the full mapping
        forwarded).
    layers:
        List or mapping of layer lookups (:func:`torchlens.extract`
        semantics), a capture SELECTOR (``tl.func``/``tl.in_module``/
        composed predicates), or a selector-valued mapping. Selector
        requests resolve against batch zero in execution order, FREEZE
        (ordered site keys + pass-qualified labels), and every later batch
        must attest the exact plan (extract D12).
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for model and stimuli.
    path:
        Optional directory. When supplied, each batch is committed as an
        immutable shard under the v2 protocol and paths are returned.
    output_transform:
        Transform slot, through the ONE :func:`torchlens.transforms.
        coerce_transform` door (DOCUMENTED-UNSTABLE). Selector-valued
        ``layers=`` accepts ``None`` or one global chain (per-site
        mappings need knowable keys).
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    resume:
        Disk mode only (DOCUMENTED-UNSTABLE): continue an interrupted run
        from its trusted ledger prefix. Every semantic signature field is
        compared individually (extract D16); the skipped prefix is
        replayed at zero forward cost and verified against the ledgered
        per-shard ``input_digest`` where recorded.
    stimulus_ids:
        Optional per-stimulus identifiers (DOCUMENTED-UNSTABLE): non-empty
        strings, duplicates legal (recorded as ``ids_unique``); validated
        against cardinality BEFORE any forward for sized inputs, in
        lockstep for unsized iterables, and with exact equality before
        completion. Disk mode only.
    model_identity:
        Disk mode (DOCUMENTED-UNSTABLE): ``"measured"`` (default —
        cryptographic threaded-Merkle state digest), ``"none"`` (recorded
        opt-out), or an assertion Mapping for unmeasurable models.
    input_transform:
        The unambiguous INPUT-preprocessing path (tvscope B5,
        DOCUMENTED-UNSTABLE spelling), distinct from ``output_transform=`` which
        transforms OUTPUTS before storage. Pass a
        :class:`torchlens.preprocessing.Resolution` (the golden path: the
        authority's own callable and its provenance travel together, the
        manifest block is stamped verified-by-construction, and the
        transform identity joins the resume signature) or a bare callable
        (applied and disclosed as opaque; the artifact becomes
        non-resumable across interruptions, mirroring opaque output
        transforms). Item-level callables run per stimulus and stack;
        batch-native processors run once per batch.
    input_provenance:
        Provenance-only stamp (tvscope B4, DOCUMENTED-UNSTABLE spelling) for
        stimuli preprocessed OUTSIDE this run: a
        :class:`torchlens.preprocessing.PreprocessingComparison` (its verdict
        rides the manifest block), a ``Resolution``, or a
        ``ResolvedPreprocessing`` record (authority-only; verdict stays
        ``unknown`` with ``applied_not_audited`` disclosed). Disk mode only:
        the in-memory result carries no manifest, so a provenance stamp
        there is a false affordance and refuses typed.
    collate:
        Optional collate (DOCUMENTED-UNSTABLE): a callable receiving the
        batch's item list and returning a
        :class:`~torchlens.dataset_extraction.BatchEnvelope`, a Mapping
        (kwargs), or a Tensor (one positional arg) — bare tuples refuse as
        ambiguous. :func:`torchlens.dataset_extraction.hf_collate` builds
        the HF text helper (tokenizer-once, pad-token refusals, full
        mapping).
    pool:
        Optional pool preset (DOCUMENTED-UNSTABLE; extract D10): one of
        ``flatten`` / ``spatial_mean`` / ``spatial_max`` / ``token_mean``
        / ``token_max`` / ``token_sum`` / ``cls`` / ``last_token`` (token
        presets are MASK-AWARE), a ``{"preset": ..., "unmasked": true}``
        recorded override, or a per-key mapping. Runs in captured dtype on
        raw axes BEFORE transforms; the documented raggedness-killer.
    dtype:
        Optional save dtype (DOCUMENTED-UNSTABLE; extract D11): a
        ``torch.dtype``/name or per-key mapping; ``None`` preserves. Cast
        after pool/postprocess, before the CPU snapshot; bf16/fp8 store
        byte-exact in safetensors shards.
    ragged:
        ``"refuse"`` (default) | ``"trim"`` | ``"as_captured"`` (extract
        D4). ``refuse`` fails typed BEFORE committing the first
        width-drifted shard; ``trim`` stores true per-stimulus extents as
        the packed values/offsets/shapes carrier (mask-derived
        ``(start, extent)`` slicing — left padding is why extent alone
        mis-slices); ``as_captured`` is reserved (refuses typed until its
        point release).
    shard_format:
        ``None`` (default: safetensors on fresh runs, the recorded format
        on resume) | ``"safetensors"`` | ``"pt"``.
    checksums:
        The D7 integrity level: ``"fast"`` (final-file CRC-32 plus the
        on-device value reduction; default), ``"crypto"`` (blake2b file
        digests), ``"none"`` (recorded opt-out; verification refuses).
    pipeline_id:
        Caller-asserted pipeline identity (extract D8): permits resuming
        past a PARTIAL callable classification or a deliberate
        complete-class refactor, with strict continuity — the id must
        already be recorded in the artifact's prefix.
    intervene:
        Optional :class:`torchlens.intervention.spec.InterventionSpec`
        (``tl.when(selector, helper)``; DOCUMENTED-UNSTABLE): every
        batch is captured with the spec armed, so stored rows are the
        INTERVENED activations. Batch zero freezes the structural sites the
        spec fired at; a batch firing nowhere or elsewhere refuses typed.
        Disk mode records the counterfactual in the manifest's
        ``tl_counterfactual_harvest_v1`` block (address class, sites, fire
        receipt digest) and ledgers each shard's fire receipt; the
        intervention identity joins the resume signature. Module- and
        op-type-addressed specs run; finer address classes refuse typed.
        Helpers must declare ``batch_independent=True``. String or mapping
        ``layers=`` only.
    store_input_ids:
        Store each batch's token ids as the ordinary ``input_ids`` key
        (DOCUMENTED-UNSTABLE, token store), trimmed by the batch mask
        under ``ragged="trim"`` like the token-axis activations; read token
        rows back with ``open_extraction(...).token_rows()``.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outputs, or written shard paths.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        On closed-vocabulary violations, in-memory false affordances,
        left-padding that cannot be corrected, row-count drift, ragged
        refusals, pool/dtype geometry refusals, and id validation.
    DatasetExtractionResumeError
        When the artifact in ``path`` cannot be safely continued.
    torchlens.errors.TransformContractError
        When the transform slot cannot be coerced or violates its plan.

    Notes
    -----
    Every forward runs under ``torch.no_grad()`` with the model in ``eval``
    mode, and every submodule's exact ``training`` flag is restored
    afterward (exception paths included).
    """

    ragged_mode, checksums_level = _validate_run_options(
        batch_size=batch_size,
        output_dir=path,
        resume=resume,
        stimulus_ids=stimulus_ids,
        collate=collate,
        ragged=ragged,
        shard_format=shard_format,
        checksums=checksums,
        pipeline_id=pipeline_id,
    )
    if input_provenance is not None and path is None:
        raise InvalidArgumentError(
            "input_provenance= requires path= (disk mode): in-memory extraction "
            "returns bare tensors with no manifest, so a provenance stamp "
            "could neither affect nor accompany the result (a false "
            "affordance; tvscope B4).",
            code="extraction_input_provenance_in_memory_unsupported",
            remedy=(
                "pass path= to record input-preprocessing provenance in "
                "the manifest, or drop input_provenance="
            ),
        )
    input_callable, input_block, input_identity, input_opaque = coerce_input_preprocessing(
        input_transform, input_provenance
    )
    intervention, keep_input_ids = _resolve_token_and_intervention(
        intervene, store_input_ids, layers
    )
    pool_policy, pool_record = resolve_pool_policy(pool)
    dtype_policy, dtype_record = resolve_dtype_policy(dtype)
    ids = list(stimulus_ids) if stimulus_ids is not None else None
    validate_stimulus_ids(ids)
    if ids is not None and isinstance(stimuli, torch.Tensor) and len(ids) != stimuli.shape[0]:
        raise InvalidArgumentError(
            f"stimulus_ids has {len(ids)} entries but the stimulus tensor has "
            f"{stimuli.shape[0]} rows; a mis-lengthed id list would mislabel "
            "every row after the shorter of the two.",
            code="extraction_stimulus_ids_cardinality",
            remedy="pass exactly one identifier per stimulus row, in order",
            n_ids=len(ids),
            num_stimuli=int(stimuli.shape[0]),
        )

    selector_pipeline: TransformPipeline | None = None
    if is_selector_request(layers):
        if isinstance(output_transform, Mapping):
            raise InvalidArgumentError(
                "Selector-valued layers= cannot take a per-site transform "
                "Mapping: the plan's output keys (including namespaced "
                "children) are unknowable before batch zero freezes them.",
                code="extraction_selector_transform_mapping_unsupported",
                remedy=(
                    "pass one global transform chain, or name the sites "
                    "explicitly with a string-valued layers= mapping"
                ),
            )
        layers_kind = "selector"
        layer_plan: dict[str, str] = {}
        pipelines: dict[str, TransformPipeline | None] = {}
        selector_pipeline = coerce_transform(output_transform)
        transform_record = _pipeline_record(selector_pipeline)
        if isinstance(layers, Mapping):
            validate_output_keys(layers.keys())
    else:
        import torchlens as _tl

        if isinstance(layers, Mapping):
            # RAW keys, before normalization stringifies them: a non-string
            # key must refuse, never silently coerce into a manifest entry.
            validate_output_keys(layers.keys())
        layer_plan = _tl._normalize_extract_layers(layers)
        layers_kind = "mapping" if isinstance(layers, Mapping) else "sequence"
        pipelines, transform_record = _coerce_transform_slot(output_transform, layer_plan)

    if input_identity is not None:
        # The input-transform identity joins the resume signature as a
        # VALUE-level extension of the existing transform_pipeline field
        # (tvscope B5): runs without an input path keep the historical value
        # shape byte-for-byte, and any input-path difference (including
        # against a pre-block artifact) mismatches through the one D16 door.
        transform_record = {"output": transform_record, "input": input_identity}
    plan = _RunPlan(
        model=model,
        stimuli=stimuli,
        layers=layers,
        layer_plan=layer_plan,
        layers_kind=layers_kind,
        batch_size=batch_size,
        device=device,
        pipelines=pipelines,
        transform_record=transform_record,
        progress=progress,
        stimulus_ids=ids,
        model_identity=model_identity,
        collate=collate,
        pool_policy=pool_policy,
        pool_record=pool_record,
        dtype_policy=dtype_policy,
        dtype_record=dtype_record,
        ragged=ragged_mode,
        shard_format=shard_format,
        checksums=checksums_level,
        pipeline_id=pipeline_id,
        transform_raw=output_transform,
        selector_pipeline=selector_pipeline,
        input_transform=input_callable,
        input_block=input_block,
        input_opaque=input_opaque,
        intervention=intervention,
        store_input_ids=keep_input_ids,
    )
    if path is None:
        return run_in_memory(plan)
    return run_to_disk(plan, Path(path), resume)


@dataclasses.dataclass(frozen=True)
class LoadedExtraction:
    """A dataset-extraction artifact read back with its self-description.

    Attributes
    ----------
    manifest:
        Parsed ``manifest.json`` document (site identity, stimulus
        provenance, axis semantics, dtypes, devices, run signature,
        TorchLens version).
    activations:
        Materialized payloads keyed by output key: concatenated tensors
        for dense keys, one merged
        :class:`~torchlens.dataset_extraction.RaggedBatch` for trimmed
        keys (never silent padding, never object arrays).
    batch_paths:
        Shard files in ledger consumption order.
    """

    manifest: dict[str, Any]
    activations: dict[str, Any]
    batch_paths: list[Path]

    @property
    def input_preprocessing(self) -> dict[str, Any]:
        """The manifest's input-preprocessing block (legacy reads as unknown).

        Returns
        -------
        dict[str, Any]
            See :func:`input_preprocessing_of`.
        """

        return input_preprocessing_of(self.manifest)


def load_extraction(
    path: str | Path,
    layers: Iterable[str] | None = None,
    *,
    max_bytes: int | None = None,
) -> LoadedExtraction:
    """Eagerly load a disk-mode extraction artifact (the guarded wrapper).

    This is a thin materialization over :func:`open_extraction` — the
    documented lazy path — behind an EXACT byte guard computed from ledger
    facts before any allocation (extract D14).

    Parameters
    ----------
    path:
        Directory previously written by disk-mode :func:`extract_dataset`.
    layers:
        Optional subset of output keys to load; defaults to every key.
    max_bytes:
        Optional explicit byte-budget override for the guard (``None``
        applies the default: half of measurable available host memory).

    Returns
    -------
    LoadedExtraction
        Manifest, materialized activations, and shard paths.

    Raises
    ------
    DatasetExtractionResumeError
        If the manifest is missing/invalid or the artifact is incomplete.
    torchlens.errors.InvalidArgumentError
        ``extraction_eager_budget_exceeded`` when the exact requested
        bytes exceed the budget (the refusal names the reader and the
        override); ``extraction_reader_key_unknown`` for unknown keys.
    """

    container_path = Path(path)
    reader = open_extraction(container_path)
    selected = list(layers) if layers is not None else None
    activations = reader.materialize(selected, max_bytes=max_bytes)
    if selected is not None:
        activations = {key: activations[key] for key in selected if key in activations}
    batch_paths = [container_path / str(row["file"]) for row in reader._rows]
    return LoadedExtraction(
        manifest=reader.manifest, activations=activations, batch_paths=batch_paths
    )


def relabel_extraction(path: str | Path, new_ids: Iterable[str]) -> dict[str, Any]:
    """Relabel a COMPLETE artifact's stimulus ids (the audited verb, D2).

    Relabeling is deliberately OUT of resume — the sidecar is write-once
    there — and produces a NEW artifact identity: the manifest's id digest
    changes, the prior digest is recorded, and an audit line rides the
    manifest permanently. Per-shard ledger ``ids_range_digest`` facts keep
    describing the WRITE-TIME labeling (the ledger is append-only); the
    audit trail is the pointer between the two.

    Parameters
    ----------
    path:
        Artifact directory.
    new_ids:
        Replacement identifiers, one per stimulus row, in row order.

    Returns
    -------
    dict[str, Any]
        The appended audit record.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_relabel_invalid`` when the artifact is not complete
        or the id list mislengths;
        ``extraction_stimulus_ids_invalid`` for malformed identifiers.
    """

    container_path = Path(path)
    manifest = _load_manifest(container_path / MANIFEST_FILENAME)
    ids = [str(item) for item in new_ids]
    validate_stimulus_ids(ids)
    totals = manifest.get("totals") or {}
    if manifest.get("status") != "complete" or not isinstance(totals.get("n_stimuli"), int):
        raise InvalidArgumentError(
            f"relabel_extraction targets COMPLETE artifacts; this one has "
            f"status {manifest.get('status')!r}.",
            code="extraction_relabel_invalid",
            remedy="finish the run first: extract_dataset(..., resume=True)",
            status=manifest.get("status"),
        )
    if len(ids) != totals["n_stimuli"]:
        raise InvalidArgumentError(
            f"relabel_extraction received {len(ids)} identifiers for an "
            f"artifact with {totals['n_stimuli']} rows.",
            code="extraction_relabel_invalid",
            remedy="pass exactly one identifier per stimulus row, in order",
            n_ids=len(ids),
            num_stimuli=totals["n_stimuli"],
        )
    import json
    import os

    from . import __version__
    from ._data_substrate import STIMULUS_IDS_FILENAME, ArtifactWriter
    from ._io._durability import fsync_dir, fsync_file

    prior_digest = (manifest.get("signature") or {}).get("stimulus_ids_digest")
    new_digest = stimulus_ids_digest(ids)
    audit = {
        "kind": "relabel",
        "prior_ids_digest": prior_digest,
        "new_ids_digest": new_digest,
        "torchlens_version": __version__,
    }
    sidecar = container_path / STIMULUS_IDS_FILENAME
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(
        json.dumps({"schema": "tl_extract_stimulus_ids_v1", "ids": ids}, indent=1),
        encoding="utf-8",
    )
    fsync_file(tmp)
    os.replace(tmp, sidecar)
    fsync_dir(container_path)
    manifest.setdefault("relabel_audit", []).append(audit)
    signature = manifest.setdefault("signature", {})
    signature["stimulus_ids_digest"] = new_digest
    provenance = manifest.setdefault("stimulus_provenance", {})
    provenance["ids_recorded"] = True
    provenance["ids_unique"] = len(set(ids)) == len(ids)
    provenance["ids_digest"] = new_digest
    ArtifactWriter(container_path, manifest).write_manifest()
    return audit


__all__ = [
    "BatchEnvelope",
    "LoadedExtraction",
    "RaggedBatch",
    "TokenRows",
    "export_extraction",
    "hf_collate",
    "open_extraction",
    "register_pure_module",
    "relabel_extraction",
]
