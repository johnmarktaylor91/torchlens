"""Batched dataset extraction with a self-describing, resumable disk artifact.

This module owns :func:`torchlens.extract_dataset`'s implementation. In disk
mode (``output_dir=``) the artifact directory is the v2 layout (extract memo
D1): a BOUNDED ``manifest.json`` (written at creation, once when batch zero
freezes the plan, and at terminal status — never per shard), an APPEND-ONLY
fsynced ``ledger.jsonl`` with one line per committed shard, a write-once
ordered ``stimulus_ids.json`` sidecar, and immutable ``batch_XXXXX.pt``
shards. The commit protocol is validate -> temp shard -> flush/fsync ->
atomic rename -> append/fsync ledger row, so a crash at any point leaves an
ignorable temp file, an unledgered orphan, or a torn final ledger line —
never a trusted lie. Completed v1 artifacts migrate without a forward;
in-progress v1 artifacts refuse resume typed (they recorded neither model
identity nor mode/grad state).

The run signature carries MODEL IDENTITY (extract D6): by default a
cryptographic threaded-Merkle digest over the complete ordered model state,
compared field-by-field on resume — a pretrained prefix resumed with a
random-init model now REFUSES typed instead of completing (T-MODELSWAP).

The ``transform=`` slot goes through the ONE :func:`torchlens.transforms.
coerce_transform` door: frozen spec chains plan against batch zero and every
shard is validated against the frozen plan BEFORE publication (T-C6); raw
callables stay accepted with identification-only disclosure; ctx dispatch is
by explicit DECLARATION, never ``inspect.signature`` (transforms memo P2).

Every spelling introduced here (``resume=``, ``stimulus_ids=``,
``model_identity=``, :func:`load_extraction`, :class:`LoadedExtraction`, the
manifest schema) is DOCUMENTED-UNSTABLE pending the naming/UI sprint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ._data_substrate import (
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
    shard_filename as _shard_filename,
    stimulus_ids_digest,
    value_reduction,
)
from ._errors import InvalidArgumentError, _actionable_message, _ActionableErrorMixin
from ._io import _json
from .errors._base import ConfigurationError, TorchLensWarning
from .transforms import (
    TransformContext,
    TransformContractError,
    TransformPipeline,
    coerce_transform,
    coerce_transform_mapping,
    pipeline_record,
)

#: Legacy v1 manifest schema id (still readable forever; superseded on write).
MANIFEST_SCHEMA = MANIFEST_SCHEMA_V1

#: Filename of the self-describing manifest inside an extraction directory.
MANIFEST_FILENAME = "manifest.json"

#: Native shard format recorded in the v2 signature (a manifest FIELD, so a
#: later format change is a value change, not a layout break).
NATIVE_FORMAT = "pt_shards_v1"

#: Maximum number of tensor elements sampled into the stimulus digest.
_DIGEST_SAMPLE_ELEMENTS = 4096


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


def _move_nested_to_device(value: Any, device: torch.device | str | None) -> Any:
    """Move tensors in a nested value to a device.

    Parameters
    ----------
    value:
        Tensor or nested Python container.
    device:
        Target device, or ``None`` to leave values unchanged.

    Returns
    -------
    Any
        Value with tensors moved to ``device``.
    """

    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


@contextlib.contextmanager
def _inference_guard(model: nn.Module) -> Iterator[None]:
    """Run extraction forwards under ``no_grad`` + ``eval`` with exact restore.

    Extraction is a read, never a training step: without this guard a harvest
    over a train-mode model silently mutates live BatchNorm running statistics
    and samples dropout, so the stored activations match no deployable forward.
    ``torch.no_grad`` is used deliberately instead of ``inference_mode`` --
    inference-mode tensors poison later autograd use if a caller feeds returned
    activations into a loss.

    Every submodule's exact ``training`` flag is snapshotted before ``eval()``
    and restored in a ``finally`` block, so mixed train/eval trees and
    exception paths (including typed extraction refusals raised mid-run) leave
    the model in precisely the state the caller handed over.

    Parameters
    ----------
    model:
        Model whose forwards run inside the guard.
    """

    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, was_training in training_flags:
            module.training = was_training


def _positional_forward_parameters(model: nn.Module) -> list[inspect.Parameter] | None:
    """Return ``model.forward``'s bindable positional parameters, if inspectable.

    Parameters
    ----------
    model:
        Model whose forward signature is inspected (declaration-based; never
        arity sniffing).

    Returns
    -------
    list[inspect.Parameter] | None
        Positional parameters of ``forward`` (``self`` excluded), or ``None``
        when the signature is not inspectable or contains ``*args`` (which
        makes positional name binding undefined).
    """

    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return None
    return [
        param
        for param in parameters
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def _is_two_dim_attention_mask(value: Any) -> bool:
    """Return whether ``value`` is a ``[batch, seq]`` HF-convention mask tensor.

    Only the two-dimensional 0/1 convention is interpreted; additive 4D masks
    and other exotic layouts are skipped (their pad geometry cannot be read
    from a name alone).

    Parameters
    ----------
    value:
        Candidate attention-mask payload.

    Returns
    -------
    bool
        Whether the value can be checked for pad geometry.
    """

    return isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[-1] > 0


def _mask_not_right_aligned(mask: torch.Tensor) -> bool:
    """Return whether any mask row has a pad position before a token.

    Right-aligned padding means every row is tokens-then-pads; a token
    appearing after a pad (left or mixed padding, or an interior gap) makes
    default absolute-position indices wrong.

    Parameters
    ----------
    mask:
        Two-dimensional attention mask (nonzero = token present).

    Returns
    -------
    bool
        Whether pad geometry is not right-aligned.
    """

    present = mask != 0
    return bool((present[..., 1:] & ~present[..., :-1]).any().item())


def _derive_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """Derive per-row position indices from an attention mask.

    This is the exact recipe HuggingFace's own generation path uses: pad
    positions clamp to 0 and token positions count from each row's first
    token, so derived positions match single-sequence forwards exactly for
    absolute-position models and change nothing for rotary models (measured:
    BERT rel 25.6% wrong -> 4.7e-07; GPT-2 rel 41.6% wrong -> 3.3e-06;
    Qwen2.5/RoPE unchanged).

    Parameters
    ----------
    mask:
        Two-dimensional attention mask (nonzero = token present).

    Returns
    -------
    torch.Tensor
        ``int64`` position ids shaped like ``mask`` on the mask's device.
    """

    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def _raise_left_padding_refusal(model: nn.Module, batch_index: int, detail: str) -> None:
    """Raise the typed left-padding refusal (extract MEMO D5, rule 2/3).

    Parameters
    ----------
    model:
        Model whose forward cannot be given corrected positions.
    batch_index:
        Zero-based index of the offending batch.
    detail:
        Sentence naming why correction was impossible for this model.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        Always; left padding is corrected or refused, never silent.
    """

    from ._errors import InvalidArgumentError

    raise InvalidArgumentError(
        f"Batch {batch_index} carries an attention_mask whose pad geometry is "
        f"not right-aligned (a pad precedes a token in at least one row), and "
        f"{detail} Models with learned absolute position embeddings read WRONG "
        f"activations from such batches while every storage and integrity "
        f"check passes (measured rel 25.6% on BERT, 41.6% on GPT-2). Models "
        f"with purely relative position handling (T5-style bias) are "
        f"value-correct under left padding; this refusal is deliberately "
        f"fail-closed for them.",
        code="extraction_left_padding_unsupported",
        remedy=(
            'right-pad the batch (tokenizer padding_side="right"), extract with '
            "batch_size=1, or include correct position_ids in each batch"
        ),
        batch_index=batch_index,
        model_type=type(model).__name__,
    )


def _inject_position_ids_positionally(
    model: nn.Module,
    batch: tuple[Any, ...] | list[Any],
    parameters: list[inspect.Parameter],
    position_ids: torch.Tensor,
    batch_index: int,
) -> tuple[Any, ...] | list[Any]:
    """Rebuild a positional batch with derived ``position_ids`` in its slot.

    Intermediate parameters between the batch's last element and the
    ``position_ids`` slot are filled with their declared defaults; a gap
    parameter without a default makes injection impossible and raises the
    same typed refusal (fail closed, never a guessed value).

    Parameters
    ----------
    model:
        Model whose forward signature drives the rebuild.
    batch:
        Original positional batch container.
    parameters:
        ``model.forward``'s positional parameters.
    position_ids:
        Derived position ids to place.
    batch_index:
        Zero-based batch index for refusal messages.

    Returns
    -------
    tuple[Any, ...] | list[Any]
        Batch of the original container type with ``position_ids`` filled.
    """

    names = [param.name for param in parameters]
    target = names.index("position_ids")
    rebuilt = list(batch)
    for index in range(len(batch), target + 1):
        if index == target:
            rebuilt.append(position_ids)
            continue
        gap = parameters[index]
        if gap.default is inspect.Parameter.empty:
            _raise_left_padding_refusal(
                model,
                batch_index,
                f"derived position_ids cannot be injected positionally: the "
                f"forward parameter {gap.name!r} between the batch and the "
                f"position_ids slot has no default value.",
            )
        rebuilt.append(gap.default)
    return type(batch)(rebuilt) if isinstance(batch, tuple) else rebuilt


def _locate_attention_mask(
    model: nn.Module, batch: Any
) -> tuple[torch.Tensor | None, str | None, list[inspect.Parameter] | None, bool]:
    """Locate a checkable ``attention_mask`` in a collated batch.

    Detection is declaration-based, keyed on the name ``attention_mask``: a
    ``Mapping`` batch is checked by key, and a positional (tuple/list) batch
    is bound against ``model.forward``'s signature. Bare tensor batches carry
    no mask by construction.

    Parameters
    ----------
    model:
        Model about to consume the batch.
    batch:
        Collated, device-moved model input.

    Returns
    -------
    tuple[torch.Tensor | None, str | None, list[inspect.Parameter] | None, bool]
        ``(mask, carrier, parameters, position_ids_present)``: the located
        two-dimensional mask (or ``None``), its carrier kind (``"mapping"`` /
        ``"positional"``), the positional forward parameters when signature
        binding ran, and whether the batch already carries ``position_ids``.
    """

    if isinstance(batch, Mapping):
        candidate = batch.get("attention_mask")
        if _is_two_dim_attention_mask(candidate):
            return candidate, "mapping", None, batch.get("position_ids") is not None
        return None, None, None, False
    if isinstance(batch, (tuple, list)):
        parameters = _positional_forward_parameters(model)
        if parameters is not None and len(batch) <= len(parameters):
            names = [param.name for param in parameters[: len(batch)]]
            if "attention_mask" in names:
                candidate = batch[names.index("attention_mask")]
                if _is_two_dim_attention_mask(candidate):
                    return candidate, "positional", parameters, "position_ids" in names
    return None, None, None, False


def _correct_batch_positions(
    model: nn.Module, batch: Any, batch_index: int, run_state: dict[str, Any]
) -> Any:
    """Correct or refuse non-right-aligned pad geometry (extract MEMO D5).

    Detection is declaration-based, keyed on the name ``attention_mask``: a
    ``Mapping`` batch is checked by key, and a positional (tuple/list) batch
    is bound against ``model.forward``'s signature. Bare tensor batches carry
    no mask and are never touched -- an unmasked left-padded ``input_ids``
    tensor is undetectable by design (D5 keys on masks).

    When correction is possible (the forward declares a ``position_ids``
    parameter) the mask-derived positions are injected and disclosed with one
    :class:`~torchlens.errors.TorchLensWarning` per run; when it is not, the
    run refuses typed before the forward. A batch that already carries
    ``position_ids`` is trusted unchanged.

    Parameters
    ----------
    model:
        Model about to consume the batch.
    batch:
        Collated, device-moved model input.
    batch_index:
        Zero-based batch index for diagnostics.
    run_state:
        Mutable per-run dict used to deduplicate the disclosure warning.

    Returns
    -------
    Any
        The batch, possibly rebuilt with derived ``position_ids``.
    """

    mask, carrier, parameters, position_ids_present = _locate_attention_mask(model, batch)
    if mask is None or position_ids_present or not _mask_not_right_aligned(mask):
        return batch

    if carrier == "mapping":
        parameters = _positional_forward_parameters(model)
    accepts_position_ids = parameters is not None and any(
        param.name == "position_ids" for param in parameters
    )
    if not accepts_position_ids:
        detail = (
            "this model's forward does not declare a position_ids parameter, "
            "so absolute positions cannot be corrected."
            if parameters is not None
            else "this model's forward signature is not inspectable, so a "
            "position_ids correction cannot be proven to apply."
        )
        _raise_left_padding_refusal(model, batch_index, detail)

    position_ids = _derive_position_ids(mask)
    if not run_state.get("position_ids_disclosed", False):
        run_state["position_ids_disclosed"] = True
        warnings.warn(
            TorchLensWarning(
                f"extract_dataset derived position_ids from the attention mask "
                f"(pad geometry is not right-aligned) and passed them to "
                f"{type(model).__name__}.forward. Derived positions match "
                f"single-sequence forwards exactly for absolute-position models "
                f"and change nothing for rotary models. Remedy: right-pad the "
                f"batch or pass explicit position_ids to silence the derivation",
                code="extraction_position_ids_derived",
            ),
            stacklevel=4,
        )
    if carrier == "mapping":
        corrected = dict(batch)
        corrected["position_ids"] = position_ids
        return corrected
    # accepts_position_ids above proved parameters is not None; cast for mypy.
    positional = cast("list[inspect.Parameter]", parameters)
    return _inject_position_ids_positionally(model, batch, positional, position_ids, batch_index)


def _collate_batch(items: list[Any]) -> Any:
    """Collate a small list of stimuli into one model input.

    Parameters
    ----------
    items:
        Stimulus items accumulated for one batch.

    Returns
    -------
    Any
        Batched tensor or nested container.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    if isinstance(first, tuple):
        return tuple(_collate_batch([item[index] for item in items]) for index in range(len(first)))
    if isinstance(first, list):
        return [_collate_batch([item[index] for item in items]) for index in range(len(first))]
    if isinstance(first, dict):
        return {key: _collate_batch([item[key] for item in items]) for key in first}
    return items


def _iter_batches(stimuli: Any, batch_size: int) -> Iterable[Any]:
    """Yield batched model inputs from tensors or iterables.

    Parameters
    ----------
    stimuli:
        Tensor with batch dimension or iterable stimulus set.
    batch_size:
        Number of items per batch.

    Yields
    ------
    Any
        One batch suitable for ``model.forward``.
    """

    if isinstance(stimuli, torch.Tensor):
        for start in range(0, stimuli.shape[0], batch_size):
            yield stimuli[start : start + batch_size]
        return

    batch: list[Any] = []
    for item in stimuli:
        batch.append(item)
        if len(batch) == batch_size:
            yield _collate_batch(batch)
            batch = []
    if batch:
        yield _collate_batch(batch)


def _merge_batch_outputs(
    accumulator: dict[str, list[torch.Tensor]],
    batch_outputs: dict[str, torch.Tensor],
    pipelines: Mapping[str, TransformPipeline | None],
) -> None:
    """Append one batch of extracted outs to an accumulator.

    Parameters
    ----------
    accumulator:
        Mutable mapping from layer label to per-batch tensors.
    batch_outputs:
        Extraction output from one batch.
    pipelines:
        Per-output-key transform chains (declared ctx dispatch + T-C2 guard
        applied by :func:`_apply_site_pipeline`).
    """

    for layer_name, tensor in batch_outputs.items():
        stored = _apply_site_pipeline(pipelines.get(layer_name), layer_name, tensor)
        accumulator.setdefault(layer_name, []).append(stored.detach().cpu())


def _tensor_digest(stimuli: torch.Tensor) -> str:
    """Return a cheap sampled content digest for tensor stimuli.

    The digest hashes the shape, dtype, and up to ``_DIGEST_SAMPLE_ELEMENTS``
    strided elements. It catches accidental stimulus swaps on resume; it is not
    an adversarial integrity check.

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


def _coerce_transform_slot(
    transform: Any, layer_plan: dict[str, str]
) -> tuple[dict[str, TransformPipeline | None], Any]:
    """Route the ``transform=`` slot through the one coercion door (memo B1).

    A Mapping is resolved PER OUTPUT KEY before coercion (transforms memo
    decision 13: the engine holds ``(label, tensor)`` at both call sites, so
    heterogeneous per-site chains need no ctx machinery); everything else
    coerces to one chain applied to every key.

    Parameters
    ----------
    transform:
        ``None`` | unary callable | registered name | spec | ordered
        sequence | per-site Mapping.
    layer_plan:
        Normalized ``output key -> layer lookup`` plan (its keys are the
        run's output keys).

    Returns
    -------
    tuple[dict[str, TransformPipeline | None], Any]
        One coerced chain (or ``None``) per output key, and the
        JSON-portable signature record: ``None``, a single
        ``tl_transform_pipeline_v1`` record, or ``{"per_site": {...}}``.
    """

    output_keys = list(layer_plan)
    if isinstance(transform, Mapping):
        pipelines = coerce_transform_mapping(transform, output_keys)
        record: Any = {
            "per_site": {key: pipeline_record(chain) for key, chain in pipelines.items()}
        }
        return pipelines, record
    pipeline = coerce_transform(transform)
    return dict.fromkeys(output_keys, pipeline), pipeline_record(pipeline)


def _apply_site_pipeline(
    pipeline: TransformPipeline | None, key: str, tensor: torch.Tensor
) -> torch.Tensor:
    """Apply one output key's chain with the T-C2 stimulus-axis guard.

    Dispatch inside the chain is by DECLARATION (transforms memo P2): raw
    callables are invoked unary, declared :class:`ContextTransform` steps
    receive the context — never ``inspect.signature``. The guard runs for
    built-ins AND raw callables: every step must preserve row count and
    order, so a launderer (mask-blind flatten across the batch, per-batch
    standardization) refuses instead of writing wrong rows.

    Parameters
    ----------
    pipeline:
        The output key's coerced chain, or ``None``.
    key:
        Output key (rides the context for per-site refusal text).
    tensor:
        Captured batch tensor, stimulus axis leading.

    Returns
    -------
    torch.Tensor
        The transformed tensor.

    Raises
    ------
    TransformContractError
        ``transform_output_invalid`` when a step returns a non-tensor;
        ``transform_row_axis_violated`` when the stimulus axis changed
        (T-C2: the row count and order are inviolable).
    """

    if pipeline is None:
        return tensor
    result = pipeline.apply(tensor, TransformContext(site_label=key))
    if not isinstance(result, torch.Tensor):
        raise TransformContractError(
            f"Transform chain for output key {key!r} returned "
            f"{type(result).__name__}, not a tensor.",
            code="transform_output_invalid",
            remedy="return a torch.Tensor from every transform step",
            site=key,
            result_type=type(result).__name__,
        )
    if result.dim() == 0 or result.shape[0] != tensor.shape[0]:
        raise TransformContractError(
            f"Transform chain for output key {key!r} changed the stimulus "
            f"axis: {tensor.shape[0]} rows in, "
            f"{'a 0-dim scalar' if result.dim() == 0 else result.shape[0]} out. "
            "Every transform step must preserve row count and order (T-C2); "
            "a per-batch reduction over the stimulus axis makes each row "
            "depend on its batch-mates.",
            code="transform_row_axis_violated",
            remedy=(
                "reduce over non-batch axes only (e.g. tl.transforms.reduce "
                "with axis>=1), or drop the offending step"
            ),
            site=key,
            rows_in=int(tensor.shape[0]),
        )
    return result


def _stimuli_signature(stimuli: Any) -> dict[str, Any]:
    """Describe the stimulus set for the manifest signature.

    Parameters
    ----------
    stimuli:
        Tensor with a leading batch dimension or an iterable stimulus set.

    Returns
    -------
    dict[str, Any]
        Signature block. Tensor stimuli carry shape, dtype, and a sampled
        digest; iterable identity is disclosed as unverifiable.
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
            "iterable stimulus identity is not verifiable; resume assumes the "
            "same stimuli in the same iteration order"
        ),
    }


def _resolve_model_identity(model: nn.Module, model_identity: Any) -> dict[str, Any]:
    """Resolve the ``model_identity=`` kwarg into the D6 identity record.

    Parameters
    ----------
    model:
        The model being harvested.
    model_identity:
        ``"measured"`` (default: cryptographic state digest) | ``"none"``
        (explicit recorded opt-out) | a Mapping (an explicit caller
        assertion, e.g. ``{"checkpoint": ..., "revision": ...}``, for
        models whose state cannot be measured).

    Returns
    -------
    dict[str, Any]
        The JSON-portable identity record for the run signature.
    """

    if isinstance(model_identity, Mapping):
        return compute_model_identity(model, level="asserted", assertion=dict(model_identity))
    return compute_model_identity(model, level=str(model_identity))


def _build_signature(plan: _RunPlan, model_identity_record: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 resume-compatibility signature (the D16 KNOWN-FIELDS).

    Every field is SEMANTIC and compared individually on resume. Fields whose
    engine knobs have not shipped yet carry the engine's fixed policy string
    (honest constants that become variable when the knob lands): the engine
    always runs eval + no_grad (A11), always derives-or-refuses left-padded
    positions (D5), and refuses ragged outputs.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    model_identity_record:
        The D6 identity record from :func:`_resolve_model_identity`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable signature compared field-by-field on resume.
    """

    return {
        "schema_version": MANIFEST_SCHEMA_V2,
        "native_format": NATIVE_FORMAT,
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
        "pool": None,
        "dtype_policy": None,
        "ragged": "refuse",
        "integrity": {
            "checksums": "fast",
            "file_fact": "crc32_final_file_bytes",
            "value_reduction": {
                "algorithm_id": VALUE_REDUCTION_ALGORITHM_ID,
                "algorithm_version": VALUE_REDUCTION_ALGORITHM_VERSION,
            },
        },
    }


def _layer_metadata(
    layer_views: dict[str, Any],
    processed: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Build the per-site self-description block from the first computed batch.

    Parameters
    ----------
    layer_views:
        Mapping from output key to the resolved ``Layer`` view of the first
        computed batch's trace.
    processed:
        The same batch's stored (post-transform, CPU) tensors keyed identically.

    Returns
    -------
    dict[str, dict[str, Any]]
        Site identity, axis semantics, dtype, and device per output key.
    """

    from .errors._base import TorchLensError

    metadata: dict[str, dict[str, Any]] = {}
    for key, layer in layer_views.items():
        captured = layer.out
        site_key: str | None
        site_key_unavailable: str | None
        try:
            site_key = str(layer.site_key)
            site_key_unavailable = None
        except TorchLensError as exc:
            site_key = None
            code = exc.fields.get("code") if isinstance(exc.fields, dict) else None
            site_key_unavailable = str(code or type(exc).__name__)
        stored = processed[key]
        metadata[key] = {
            "layer_label": str(layer.layer_label),
            "site_key": site_key,
            "site_key_unavailable": site_key_unavailable,
            "captured_dtype": str(captured.dtype),
            "captured_device": str(captured.device),
            "per_stimulus_shape": list(captured.shape[1:]),
            "stored_dtype": str(stored.dtype),
            "stored_per_stimulus_shape": list(stored.shape[1:]),
            "batch_axis": 0,
        }
    return metadata


def _base_manifest(signature: dict[str, Any], stimulus_ids: list[str] | None) -> dict[str, Any]:
    """Create a fresh in-progress v2 manifest document (the bounded header).

    The manifest is written exactly three times (extract D1): at creation,
    once when batch zero freezes the plan, and at terminal status. Per-shard
    facts live in the append-only ledger, never here.

    Parameters
    ----------
    signature:
        Resume-compatibility signature block.
    stimulus_ids:
        Optional caller-supplied per-stimulus identifiers, in iteration order.

    Returns
    -------
    dict[str, Any]
        Bounded manifest header with no layer metadata or totals yet.
    """

    from . import __version__

    return {
        "schema": MANIFEST_SCHEMA_V2,
        "torchlens_version": __version__,
        "status": "in_progress",
        "signature": signature,
        "stimulus_provenance": {
            "order": (
                "row i of every shard, consumed in ledger order, is stimulus i "
                "in the caller's iteration order"
            ),
            "n_stimuli": None,
            "ids_recorded": stimulus_ids is not None,
            "ids_digest": signature.get("stimulus_ids_digest"),
        },
        "storage": {
            "shard_format": "pt",
            "shard_pattern": "batch_XXXXX.pt",
            "ledger": LEDGER_FILENAME,
        },
        "layers": None,
        "run": {},
        "totals": None,
        "ledger_digest": None,
    }


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
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
        # allocation DoS). It raises json.JSONDecodeError, a ValueError, so the
        # handler below catches over-size and over-nested payloads unchanged.
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


def _completed_prefix(manifest: dict[str, Any], container_path: Path) -> list[dict[str, Any]]:
    """Return the ledgered shard prefix whose files are all present on disk.

    The ledger is written contiguously from index 0; the first ledger row whose
    file is missing (user deletion, partial sync) truncates the trusted prefix,
    and everything after it is recomputed.

    Parameters
    ----------
    manifest:
        Parsed manifest document.
    container_path:
        Extraction directory containing the shards.

    Returns
    -------
    list[dict[str, Any]]
        Contiguous ledger rows (``index``, ``file``, ``n_stimuli``) verified
        present on disk.
    """

    prefix: list[dict[str, Any]] = []
    for row in manifest.get("batches") or []:
        expected_name = _shard_filename(int(row["index"]))
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


def _consume_skipped_stimuli(stimuli: Any, n_skip: int) -> Any:
    """Advance past already-extracted stimuli and return the remaining stream.

    Parameters
    ----------
    stimuli:
        Stimulus tensor or iterable.
    n_skip:
        Exact number of stimuli covered by the trusted shard prefix.

    Returns
    -------
    Any
        Remaining stimuli: a tensor slice, or the advanced iterator.

    Raises
    ------
    DatasetExtractionResumeError
        If an iterable stimulus stream ends before covering the ledgered
        prefix (the stimuli cannot be the ones the artifact was built from).
    """

    if isinstance(stimuli, torch.Tensor):
        return stimuli[n_skip:]
    iterator = iter(stimuli)
    consumed = 0
    while consumed < n_skip:
        try:
            next(iterator)
        except StopIteration:
            raise DatasetExtractionResumeError(
                f"Stimulus iterable ended after {consumed} items but the "
                f"artifact's completed shards cover {n_skip} stimuli.",
                code="extraction_resume_signature_mismatch",
                remedy=(
                    "re-run with the original stimuli, or delete the output "
                    "directory to start a fresh extraction"
                ),
                n_ledgered=n_skip,
                n_available=consumed,
            ) from None
        consumed += 1
    return iterator


def _refuse_signature_mismatch(
    recorded: Mapping[str, Any], current: Mapping[str, Any], manifest_path: Path
) -> None:
    """Compare v2 signatures field by field and refuse typed on mismatch (D16).

    ``model_identity`` mismatches get their own code: a pretrained prefix
    resumed with a random-init (or otherwise different) model is the
    T-MODELSWAP hazard, the exact fails-open defect this record exists to
    kill — the artifact would silently mix checkpoint activations with the
    resuming model's.

    Parameters
    ----------
    recorded:
        The artifact's recorded signature.
    current:
        The signature of the run asking to resume.
    manifest_path:
        Manifest path, for the refusal message.

    Raises
    ------
    DatasetExtractionResumeError
        ``extraction_resume_model_identity_mismatch`` when the model identity
        record differs (or either side cannot be compared);
        ``extraction_resume_signature_mismatch`` when any other semantic
        field differs. Both name the exact fields.
    """

    mismatched = compare_signatures(recorded, current)
    if not mismatched:
        return
    if "model_identity" in mismatched:
        recorded_identity = recorded.get("model_identity")
        current_identity = current.get("model_identity")
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
            recorded_identity=recorded_identity,
            requested_identity=current_identity,
        )
    raise DatasetExtractionResumeError(
        f"Extraction artifact at {str(manifest_path.parent)!r} was produced by a "
        f"different run configuration (mismatched signature fields: {mismatched}).",
        code="extraction_resume_signature_mismatch",
        remedy=(
            "re-run with the artifact's original layers, batch_size, transform, "
            "stimuli, and stimulus_ids, or delete the output directory to "
            "start fresh"
        ),
        mismatched_fields=mismatched,
        recorded_signature=dict(recorded),
        requested_signature=dict(current),
    )


@dataclasses.dataclass(frozen=True)
class _RunPlan:
    """Resolved extraction-run configuration shared by the run engines.

    Attributes
    ----------
    model:
        PyTorch model to run (already moved to ``device`` when one was given).
    stimuli:
        Stimulus tensor or iterable, as supplied by the caller.
    layers:
        The caller's original layer spec (mapping-versus-list semantics).
    layer_plan:
        Normalized ``output key -> layer lookup`` plan.
    layers_kind:
        ``"mapping"`` or ``"sequence"``.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for stimuli movement.
    pipelines:
        One coerced transform chain (or ``None``) per output key, from the
        single :func:`torchlens.transforms.coerce_transform` door.
    transform_record:
        JSON-portable signature record of the transform slot (``None``, one
        ``tl_transform_pipeline_v1`` record, or ``{"per_site": {...}}``).
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    stimulus_ids:
        Optional per-stimulus identifiers recorded as provenance.
    model_identity:
        The caller's ``model_identity=`` value (level string or assertion
        Mapping), resolved lazily in disk mode.
    """

    model: nn.Module
    stimuli: Any
    layers: Iterable[str] | Mapping[str, str]
    layer_plan: dict[str, str]
    layers_kind: str
    batch_size: int
    device: torch.device | str | None
    pipelines: dict[str, TransformPipeline | None]
    transform_record: Any
    progress: bool
    stimulus_ids: list[str] | None
    model_identity: Any

    @property
    def resume_verifiable(self) -> bool:
        """Whether every output key's chain reconstructs from its record.

        Returns
        -------
        bool
            ``False`` iff any chain holds an opaque (identification-only)
            step; the strict opaque-resume rule keys on this.
        """

        return all(
            pipeline is None or pipeline.resume_verifiable for pipeline in self.pipelines.values()
        )


def extract_dataset(
    model: nn.Module,
    stimuli: Any,
    layers: Iterable[str] | Mapping[str, str],
    batch_size: int = 32,
    device: torch.device | str | None = None,
    output_dir: str | Path | None = None,
    transform: Any = None,
    progress: bool = True,
    *,
    resume: bool = False,
    stimulus_ids: Iterable[str] | None = None,
    model_identity: Any = "measured",
) -> dict[str, torch.Tensor] | list[Path]:
    """Extract outs from an iterable dataset in batches.

    Row ``i`` of every returned tensor corresponds to stimulus ``i`` in
    iteration order. Batch files are consumed in ``batch_00000.pt``,
    ``batch_00001.pt``, ... order.

    In disk mode the output directory is a SELF-DESCRIBING artifact: shard
    writes are atomic, and ``manifest.json`` records the run signature, per-site
    identity (layer label and structural site key where derivable), stimulus
    ordering/provenance, axis semantics, dtypes, devices, and the TorchLens
    version, updated atomically after every shard. Read it back with
    :func:`torchlens.dataset_extraction.load_extraction`.

    Parameters
    ----------
    model:
        PyTorch model to run.
    stimuli:
        Tensor with a leading batch dimension or iterable of stimulus items.
    layers:
        List or mapping accepted by :func:`torchlens.extract`.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for model and stimuli.
    output_dir:
        Optional directory. When supplied, each batch output is written as
        ``batch_XXXXX.pt`` and paths are returned, alongside ``manifest.json``.
    transform:
        Transform slot, through the ONE :func:`torchlens.transforms.
        coerce_transform` door (DOCUMENTED-UNSTABLE): ``None`` | unary
        callable | registered name | frozen spec / chain | ordered sequence
        | per-site Mapping (``{output_key: chain, tl.transforms.DEFAULT:
        fallback}``). Frozen spec chains are planned against batch zero and
        every shard is validated against the plan before publication; raw
        callables are accepted with identification-only disclosure and make
        the artifact non-resumable across interruptions. Context dispatch is
        by explicit declaration (``tl.transforms.ContextTransform`` /
        ``with_context``), never signature inspection.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    resume:
        Disk mode only (DOCUMENTED-UNSTABLE): continue an interrupted run in
        ``output_dir`` from its trusted ledger prefix. Every semantic
        signature field (layers, batch size, transform chain, stimulus
        descriptor, stimulus-id digest, MODEL IDENTITY, ...) is compared
        individually; any mismatch refuses typed naming the exact fields.
        Iterable stimuli are assumed to replay in the original order, which
        resume cannot verify. A completed compatible artifact returns its
        shard paths without running the model or touching its device
        placement. Completed v1 artifacts migrate to v2 without a forward;
        in-progress v1 artifacts refuse typed.
    stimulus_ids:
        Optional per-stimulus identifiers (DOCUMENTED-UNSTABLE), recorded in
        the write-once ordered sidecar and digested into the run signature.
        Disk mode only: the in-memory result is a bare tensor mapping that
        could neither carry nor be affected by validated identifiers, so
        passing them there is a false affordance and refuses typed.
    model_identity:
        Disk mode (DOCUMENTED-UNSTABLE): ``"measured"`` (default — a
        cryptographic threaded-Merkle digest over the complete ordered model
        state: every parameter and persistent buffer, int/bool/0-dim
        included), ``"none"`` (explicit recorded opt-out; resume proceeds
        with no identity claim), or a Mapping (an explicit caller assertion,
        e.g. ``{"checkpoint": ..., "revision": ...}``, for models whose
        state cannot be measured — meta-device or disk-offloaded). When
        measurement is impossible and no assertion is given the record is
        ``"unavailable"`` and resume refuses typed.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outs, or written batch paths.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        If ``resume=True`` or ``stimulus_ids=`` is combined with in-memory
        mode, ``model_identity`` is outside its closed vocabulary, sized
        stimuli and ``stimulus_ids`` disagree on cardinality, or a batch's
        pad geometry is not right-aligned and correct ``position_ids``
        cannot be derived for this model.
    DatasetExtractionResumeError
        If the artifact in ``output_dir`` cannot be safely continued
        (signature/model-identity mismatch, opaque transform continuation,
        broken ledger prefix, in-progress v1 artifact, ...).
    torchlens.transforms.TransformContractError
        If the transform slot cannot be coerced, a chain violates the
        stimulus-axis contract, or a shard's observed output contradicts
        the frozen plan.

    Notes
    -----
    Every forward runs under ``torch.no_grad()`` with the model in ``eval``
    mode, and every submodule's exact ``training`` flag is restored afterward
    (exception paths included). This changed in the fails-open fix wave:
    previously a train-mode model silently mutated its BatchNorm running
    statistics during extraction.
    """

    from ._errors import InvalidArgumentError

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if resume and output_dir is None:
        raise InvalidArgumentError(
            "resume=True requires output_dir: only disk-mode extraction leaves "
            "a shard ledger to resume from.",
            code="extraction_resume_requires_output_dir",
            remedy="pass output_dir= (disk mode) or drop resume=True",
        )
    if stimulus_ids is not None and output_dir is None:
        raise InvalidArgumentError(
            "stimulus_ids= requires output_dir: in-memory extraction returns "
            "bare tensors with no manifest, so validated identifiers could "
            "neither affect nor accompany the result (a false affordance).",
            code="extraction_stimulus_ids_in_memory_unsupported",
            remedy=(
                "pass output_dir= to record stimulus identity in the manifest, "
                "or drop stimulus_ids="
            ),
        )

    import torchlens as _tl

    layer_plan = _tl._normalize_extract_layers(layers)
    pipelines, transform_record = _coerce_transform_slot(transform, layer_plan)
    ids = list(stimulus_ids) if stimulus_ids is not None else None
    if ids is not None and isinstance(stimuli, torch.Tensor) and len(ids) != stimuli.shape[0]:
        raise InvalidArgumentError(
            f"stimulus_ids has {len(ids)} entries but the stimulus tensor has "
            f"{stimuli.shape[0]} rows; a mis-lengthed id list would mislabel "
            "every row after the shorter of the two.",
            code="extraction_stimulus_ids_cardinality",
            remedy="pass exactly one identifier per stimulus row, in order",
            n_ids=len(ids),
            n_stimuli=int(stimuli.shape[0]),
        )
    plan = _RunPlan(
        model=model,
        stimuli=stimuli,
        layers=layers,
        layer_plan=layer_plan,
        layers_kind="mapping" if isinstance(layers, Mapping) else "sequence",
        batch_size=batch_size,
        device=device,
        pipelines=pipelines,
        transform_record=transform_record,
        progress=progress,
        stimulus_ids=ids,
        model_identity=model_identity,
    )
    if output_dir is None:
        return _extract_in_memory(plan)
    return _extract_to_disk(plan, Path(output_dir), resume)


def _batch_iterable(plan: _RunPlan, remaining: Any) -> Iterable[Any]:
    """Build the (optionally progress-wrapped) batch iterator for a run.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    remaining:
        Stimuli still to extract (full set, tensor slice, or advanced iterator).

    Returns
    -------
    Iterable[Any]
        Batches suitable for ``model.forward``.
    """

    batches = _iter_batches(remaining, plan.batch_size)
    total = None
    if isinstance(remaining, torch.Tensor):
        total = (remaining.shape[0] + plan.batch_size - 1) // plan.batch_size
    if plan.progress:
        from .utils.display import progress_bar

        batches = progress_bar(batches, total=total, desc="torchlens.extract", enabled=True)
    return batches


def _extract_in_memory(plan: _RunPlan) -> dict[str, torch.Tensor]:
    """Run the in-memory extraction engine.

    Parameters
    ----------
    plan:
        Resolved run configuration.

    Returns
    -------
    dict[str, torch.Tensor]
        Concatenated outs keyed as :func:`torchlens.extract` keys them.
    """

    import torchlens as _tl

    if plan.device is not None:
        plan.model.to(plan.device)
    accumulator: dict[str, list[torch.Tensor]] = {}
    run_state: dict[str, Any] = {}
    with _inference_guard(plan.model):
        for batch_index, batch in enumerate(_batch_iterable(plan, plan.stimuli)):
            batch = _move_nested_to_device(batch, plan.device)
            batch = _correct_batch_positions(plan.model, batch, batch_index, run_state)
            _trace, batch_outputs, _views = _tl._extract_layers_with_trace(
                plan.model, batch, plan.layers
            )
            _merge_batch_outputs(accumulator, batch_outputs, plan.pipelines)
    return {label: torch.cat(tensors, dim=0) for label, tensors in accumulator.items()}


def _prepare_disk_run(
    plan: _RunPlan, container_path: Path, resume: bool
) -> tuple[ArtifactWriter, list[dict[str, Any]], list[Path] | None]:
    """Prepare the v2 artifact writer and resolve the resume state.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    container_path:
        Extraction directory.
    resume:
        Whether to continue from an existing ledger.

    Returns
    -------
    tuple[ArtifactWriter, list[dict[str, Any]], list[Path] | None]
        The bound writer, the trusted completed-shard ledger rows, and —
        when the artifact is already complete with every shard present — the
        final shard paths (callers return them without running the model).

    Raises
    ------
    DatasetExtractionResumeError
        On unmanifested shard directories, signature or model-identity
        mismatches, or an opaque-transform continuation.
    ExtractionArtifactError
        On a broken ledger prefix or an in-progress v1 artifact.
    """

    container_path.mkdir(parents=True, exist_ok=True)
    manifest_path = container_path / MANIFEST_FILENAME

    if resume and manifest_path.exists():
        existing = _load_manifest(manifest_path)
        if existing.get("schema") == MANIFEST_SCHEMA_V1:
            # Completed v1 artifacts migrate without a forward; in-progress
            # v1 artifacts refuse typed inside (they recorded neither model
            # identity nor mode/grad state, so the prefix is unprovable).
            existing, _migrated_rows = migrate_v1_artifact(container_path, existing)
        identity = _resolve_model_identity(plan.model, plan.model_identity)
        signature = _build_signature(plan, identity)
        _refuse_signature_mismatch(existing.get("signature") or {}, signature, manifest_path)
        rows = read_trusted_rows(container_path)
        totals = existing.get("totals") or {}
        if existing.get("status") == "complete" and len(rows) == totals.get("n_shards"):
            # A completed compatible resume is a true no-op: the model is
            # neither moved to a device nor mode-flipped (extract memo D3).
            return (
                ArtifactWriter(container_path, existing),
                rows,
                [container_path / str(row["file"]) for row in rows],
            )
        if not plan.resume_verifiable:
            opaque = sorted(
                key
                for key, pipeline in plan.pipelines.items()
                if pipeline is not None and not pipeline.resume_verifiable
            )
            raise DatasetExtractionResumeError(
                f"Resuming this artifact would run forwards through an OPAQUE "
                f"transform step (output keys {opaque}); an opaque callable's "
                "identity is a disclosure, not a proof, so a continuation "
                "cannot be verified to produce the same numbers as the prefix "
                "(transforms memo decision 14).",
                code="extraction_resume_opaque_transform",
                remedy=(
                    "register the transform under a versioned name "
                    "(torchlens.transforms.register_transform) and pass the "
                    "registered spelling on both runs, or extract into a "
                    "fresh directory"
                ),
                opaque_keys=opaque,
            )
        _clean_orphan_tmp_files(container_path)
        # A torn final ledger line is crash debris: cleared before appends
        # resume, or the next commit would concatenate into it.
        repair_ledger_tail(container_path)
        existing["status"] = "in_progress"
        # No manifest rewrite for a continuation: creation, batch-zero plan
        # freeze, and terminal status are the only three writes (D1).
        return ArtifactWriter(container_path, existing), rows, None

    if resume and any(container_path.glob("batch_*.pt")):
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
    # the manifest (the v1 engine's manifest overwrite, made explicit).
    _clean_orphan_tmp_files(container_path)
    for stale in (LEDGER_FILENAME, STIMULUS_IDS_FILENAME):
        with contextlib.suppress(OSError):
            (container_path / stale).unlink()
    identity = _resolve_model_identity(plan.model, plan.model_identity)
    manifest = _base_manifest(_build_signature(plan, identity), plan.stimulus_ids)
    writer = ArtifactWriter(container_path, manifest)
    writer.write_manifest()
    if plan.stimulus_ids is not None:
        writer.write_stimulus_ids_sidecar(plan.stimulus_ids)
    return writer, [], None


def _freeze_transform_plans(
    plan: _RunPlan, batch_outputs: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Plan every output key's chain against batch zero (the T-C6 freeze).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    batch_outputs:
        Batch zero's captured (pre-transform) tensors, keyed by output key.

    Returns
    -------
    dict[str, Any]
        Per-key frozen plan: the planned step rows (``None`` for opaque or
        post-opaque steps, which are validated from observation per shard
        instead of trusted from batch one) and the final predicted output
        spec every later shard is validated against before publication.
    """

    from .transforms import TensorSpec

    plans: dict[str, Any] = {}
    for key, tensor in batch_outputs.items():
        pipeline = plan.pipelines.get(key)
        if pipeline is None or not pipeline.steps:
            plans[key] = None
            continue
        planned = pipeline.plan(TensorSpec.of(tensor), TransformContext(site_label=key))
        final = planned[-1] if planned else None
        plans[key] = {
            "steps": [
                None
                if step is None
                else {
                    "name": step.name,
                    "version": step.version,
                    "per_stimulus_shape": list(step.output.shape[1:]),
                    "dtype": step.output.dtype,
                    "stream_safe": step.stream_safe,
                    "may_alias": step.may_alias,
                }
                for step in planned
            ],
            "final_output": None
            if final is None
            else {
                "per_stimulus_shape": list(final.output.shape[1:]),
                "dtype": final.output.dtype,
            },
        }
    return plans


def _validate_against_plan(manifest: dict[str, Any], processed: dict[str, torch.Tensor]) -> None:
    """Refuse a shard whose observed output contradicts the frozen plan (T-C6).

    Parameters
    ----------
    manifest:
        The artifact manifest holding the batch-zero frozen plans.
    processed:
        The shard's stored (post-transform) tensors, keyed by output key.

    Raises
    ------
    TransformContractError
        ``transform_plan_violated`` naming the key, the plan, and the
        observation; the shard is refused BEFORE publication.
    """

    plans = (manifest.get("run") or {}).get("transform_plans") or {}
    for key, stored in processed.items():
        final = (plans.get(key) or {}).get("final_output") if plans.get(key) else None
        if not final:
            continue
        expected_shape = list(final.get("per_stimulus_shape") or [])
        expected_dtype = final.get("dtype")
        observed_shape = list(stored.shape[1:])
        shape_ok = len(observed_shape) == len(expected_shape) and all(
            want is None or want == got for want, got in zip(expected_shape, observed_shape)
        )
        if not shape_ok or str(stored.dtype) != expected_dtype:
            raise TransformContractError(
                f"Output key {key!r} produced {observed_shape} / "
                f"{stored.dtype}, contradicting the frozen transform plan "
                f"{expected_shape} / {expected_dtype}; the shard is refused "
                "BEFORE publication (T-C6: runtime output must match the "
                "declared plan, and ragged shape is validated per shard, "
                "never inferred forever from batch one).",
                code="transform_plan_violated",
                remedy=(
                    "keep per-stimulus shapes fixed across batches (pad or "
                    "pool before the chain), or re-extract into a fresh "
                    "directory if the chain itself changed"
                ),
                site=key,
                planned_shape=expected_shape,
                planned_dtype=expected_dtype,
                observed_shape=observed_shape,
                observed_dtype=str(stored.dtype),
            )


def _ids_range_digest(plan: _RunPlan, row_start: int, n_rows: int) -> str | None:
    """Digest one shard's stimulus-id slice for its ledger row (D1 ID facts).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    row_start:
        Global row index of the shard's first stimulus.
    n_rows:
        Number of stimulus rows in the shard.

    Returns
    -------
    str | None
        ``"sha256:..."`` over the ordered id slice, or ``None`` when the run
        carries no ids.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_stimulus_ids_cardinality`` when the id list runs out
        before the shard's rows (refused BEFORE the commit: a short id list
        would mislabel every later row).
    """

    if plan.stimulus_ids is None:
        return None
    ids_slice = plan.stimulus_ids[row_start : row_start + n_rows]
    if len(ids_slice) != n_rows:
        raise InvalidArgumentError(
            f"stimulus_ids supplies {len(plan.stimulus_ids)} identifiers but "
            f"the stimuli reach row {row_start + n_rows}; the shard is refused "
            "before its commit (a short id list would mislabel every later "
            "row).",
            code="extraction_stimulus_ids_cardinality",
            remedy="pass exactly one identifier per stimulus, in order",
            n_ids=len(plan.stimulus_ids),
            rows_needed=row_start + n_rows,
        )
    return stimulus_ids_digest(ids_slice)


def _extract_to_disk(plan: _RunPlan, container_path: Path, resume: bool) -> list[Path]:
    """Run the disk-mode extraction engine (the v2 commit protocol).

    Per shard: validate (frozen plan + stimulus axis + id cardinality) ->
    temp shard -> flush/fsync -> atomic rename -> append/fsync ledger row.
    The manifest is written at creation, once at the batch-zero plan freeze,
    and at terminal status — never per shard.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    container_path:
        Extraction directory.
    resume:
        Whether to continue from an existing ledger.

    Returns
    -------
    list[pathlib.Path]
        Every shard path in ledger order, including resumed prefixes.
    """

    import functools

    import torchlens as _tl

    writer, completed_rows, complete_paths = _prepare_disk_run(plan, container_path, resume)
    if complete_paths is not None:
        # A completed compatible resume is a true no-op: the model is neither
        # moved to a device nor mode-flipped (extract MEMO D3).
        return complete_paths
    if plan.device is not None:
        plan.model.to(plan.device)
    manifest = writer.manifest
    n_skip = sum(int(row["n_rows"]) for row in completed_rows)
    remaining = _consume_skipped_stimuli(plan.stimuli, n_skip) if n_skip else plan.stimuli
    start_index = len(completed_rows)
    container_paths = [container_path / str(row["file"]) for row in completed_rows]

    run_state: dict[str, Any] = {}
    row_start = n_skip
    n_committed = start_index
    with _inference_guard(plan.model):
        for offset, batch in enumerate(_batch_iterable(plan, remaining)):
            batch_index = start_index + offset
            batch = _move_nested_to_device(batch, plan.device)
            batch = _correct_batch_positions(plan.model, batch, batch_index, run_state)
            _trace, batch_outputs, layer_views = _tl._extract_layers_with_trace(
                plan.model, batch, plan.layers
            )
            n_rows = next(iter(batch_outputs.values())).shape[0] if batch_outputs else 0
            processed: dict[str, torch.Tensor] = {}
            shard_key_facts: dict[str, dict[str, Any]] = {}
            for key, tensor in batch_outputs.items():
                stored = _apply_site_pipeline(plan.pipelines.get(key), key, tensor)
                # D7: the order-sensitive value reduction runs ON the
                # tensor's device BEFORE the host copy, so a corruption
                # between here and the written file is catchable later.
                shard_key_facts[key] = {
                    "per_stimulus_shape": list(stored.shape[1:]),
                    "dtype": str(stored.dtype),
                    "value_reduction": value_reduction(key, stored),
                }
                processed[key] = stored.detach().cpu()
            if manifest.get("layers") is None:
                manifest["layers"] = _layer_metadata(layer_views, processed)
                manifest.setdefault("run", {})["transform_plans"] = _freeze_transform_plans(
                    plan, batch_outputs
                )
                writer.write_manifest()
            _validate_against_plan(manifest, processed)
            ids_range = _ids_range_digest(plan, row_start, n_rows)
            row = writer.commit_shard(
                index=batch_index,
                row_start=row_start,
                n_rows=n_rows,
                save_payload=functools.partial(torch.save, processed),
                row_facts={"keys": shard_key_facts, "ids_range_digest": ids_range},
            )
            container_paths.append(container_path / str(row["file"]))
            row_start += n_rows
            n_committed += 1

    manifest["stimulus_provenance"]["n_stimuli"] = row_start
    writer.finalize(n_shards=n_committed, n_stimuli=row_start)
    return container_paths


@dataclasses.dataclass(frozen=True)
class LoadedExtraction:
    """A dataset-extraction artifact read back with its self-description.

    Attributes
    ----------
    manifest:
        Parsed ``manifest.json`` document (site identity, stimulus provenance,
        axis semantics, dtypes, devices, run signature, TorchLens version).
    activations:
        Concatenated activations keyed by output key, stimulus axis leading.
    batch_paths:
        Shard files in consumption order.
    """

    manifest: dict[str, Any]
    activations: dict[str, torch.Tensor]
    batch_paths: list[Path]


def load_extraction(
    output_dir: str | Path,
    layers: Iterable[str] | None = None,
) -> LoadedExtraction:
    """Load a disk-mode extraction artifact with its self-description.

    Parameters
    ----------
    output_dir:
        Directory previously written by disk-mode :func:`extract_dataset`.
    layers:
        Optional subset of output keys to load; defaults to every key.

    Returns
    -------
    LoadedExtraction
        Manifest, concatenated activations, and shard paths.

    Raises
    ------
    DatasetExtractionResumeError
        If the manifest is missing/invalid, the artifact is incomplete, or a
        requested output key is not in the artifact.
    """

    container_path = Path(output_dir)
    manifest = _load_manifest(container_path / MANIFEST_FILENAME)
    if manifest.get("status") != "complete":
        raise DatasetExtractionResumeError(
            f"Extraction artifact at {str(container_path)!r} has status "
            f"{manifest.get('status')!r}, not 'complete'.",
            code="extraction_manifest_invalid",
            remedy="finish the run first: extract_dataset(..., resume=True)",
            status=manifest.get("status"),
        )
    if manifest.get("schema") == MANIFEST_SCHEMA_V2:
        # Readers consume LEDGER order, never filename order; a ledgered
        # shard that is missing or byte-size-mismatched refuses typed inside
        # read_trusted_rows (a broken member ends the trusted prefix).
        rows = read_trusted_rows(container_path)
        n_ledgered = (manifest.get("totals") or {}).get("n_shards")
        if len(rows) != n_ledgered:
            raise DatasetExtractionResumeError(
                f"Extraction artifact at {str(container_path)!r} ledgers "
                f"{len(rows)} trusted shard rows but its terminal totals "
                f"record {n_ledgered}.",
                code="extraction_manifest_invalid",
                remedy="re-run extract_dataset(..., resume=True) to finish the artifact",
                n_present=len(rows),
                n_ledgered=n_ledgered,
            )
    else:
        rows = [
            {"file": row["file"], "n_rows": row["n_stimuli"]}
            for row in _completed_prefix(manifest, container_path)
        ]
        if len(rows) != len(manifest.get("batches") or []):
            raise DatasetExtractionResumeError(
                f"Extraction artifact at {str(container_path)!r} is missing ledgered "
                f"shard files ({len(rows)} of {len(manifest.get('batches') or [])} present).",
                code="extraction_manifest_invalid",
                remedy="re-run extract_dataset(..., resume=True) to restore the missing shards",
                n_present=len(rows),
                n_ledgered=len(manifest.get("batches") or []),
            )
    available = set((manifest.get("layers") or {}).keys())
    selected = list(layers) if layers is not None else None
    if selected is not None:
        missing = sorted(set(selected) - available)
        if missing:
            raise DatasetExtractionResumeError(
                f"Output keys {missing} are not in the artifact at "
                f"{str(container_path)!r} (available: {sorted(available)}).",
                code="extraction_manifest_invalid",
                remedy="request only output keys recorded in the manifest's layers block",
                missing_keys=missing,
                available_keys=sorted(available),
            )
    per_key: dict[str, list[torch.Tensor]] = {}
    batch_paths = [container_path / str(row["file"]) for row in rows]
    for batch_path in batch_paths:
        # mmap=True is a measured 13.4x on selective reads and retroactive on
        # every existing artifact: the pickle no longer materializes each
        # shard's full byte payload up front, and torch.cat below copies the
        # selected tensors out of the mapping into owned memory.
        payload = torch.load(batch_path, weights_only=True, mmap=True)
        for key, tensor in payload.items():
            if selected is not None and key not in selected:
                continue
            per_key.setdefault(key, []).append(tensor)
    activations = {key: torch.cat(tensors, dim=0) for key, tensors in per_key.items()}
    return LoadedExtraction(manifest=manifest, activations=activations, batch_paths=batch_paths)


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_V2",
    "NATIVE_FORMAT",
    "DatasetExtractionResumeError",
    "LoadedExtraction",
    "extract_dataset",
    "load_extraction",
]
