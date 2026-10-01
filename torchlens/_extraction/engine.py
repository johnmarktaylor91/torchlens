"""The extraction v2 store pipeline and run engines (extract D13 + F18).

One manifested order per batch::

    collate -> position/mask validation -> model call -> selected output
            -> pool (captured dtype, raw axes, mask) -> postprocess(output, ctx)
            -> row + logical-shape validation -> per-key save-dtype cast
            -> CPU-contiguous snapshot -> writer

Every selected output must retain the INPUT batch's row count after
postprocessing or refuse typed (D13); the batch context carries identity
and geometry facts and never retains raw examples after commit (item 4);
resume compares every semantic fact field-by-field (D16) with the callable
identity rules (D8) and the skip-replay ``input_digest`` verification
(item 16) layered on top.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .._data_substrate import (
    ArtifactWriter,
    stimulus_ids_digest,
    value_reduction,
)
from .._errors import InvalidArgumentError
from .._extraction_provenance import apply_input_transform
from ..errors._base import TorchLensWarning
from ..transforms import (
    TransformContext,
    TransformContractError,
    TransformPipeline,
)
from .context import BatchContext, _run_post_store_hooks, validate_row_counts
from .dtype_policy import cast_for_store
from .envelope import (
    BatchEnvelope,
    coerce_envelope,
    default_collate,
    envelope_input_digest,
)
from .pool import apply_pool
from .ragged import (
    RaggedBatch,
    mask_row_geometry,
    raise_ragged_refusal,
    trim_batch,
)
from .selector_plan import (
    SelectorPlan,
    attest_selector_batch,
    freeze_selector_plan,
)
from .shards import shard_extension, write_shard

__tl_layer__ = "L5"

from .resume import (  # noqa: E402 - grouped after the layer marker deliberately
    DatasetExtractionResumeError,
    _prepare_disk_run,
)


@dataclasses.dataclass(frozen=True)
class _RunPlan:
    """Resolved extraction-run configuration shared by the run engines.

    Attributes
    ----------
    model:
        PyTorch model to run (moved to ``device`` when one was given).
    stimuli:
        Stimulus tensor or iterable, as supplied by the caller.
    layers:
        The caller's original layer spec (strings, mapping, or selectors).
    layer_plan:
        Normalized ``output key -> layer lookup`` plan (empty for selector
        requests, whose plan freezes on batch zero).
    layers_kind:
        ``"mapping"`` / ``"sequence"`` / ``"selector"``.
    batch_size:
        Number of stimuli per forward pass.
    device:
        Optional device for stimuli movement.
    pipelines:
        One coerced transform chain (or ``None``) per output key.
    transform_record:
        JSON-portable signature record of the transform slot.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    stimulus_ids:
        Optional per-stimulus identifiers recorded as provenance.
    model_identity:
        The caller's ``model_identity=`` value.
    collate:
        Optional collate callable (engine-flavored or plain user callable).
    pool_policy / pool_record:
        Resolved pool presets and their signature record (extract D10).
    dtype_policy / dtype_record:
        Resolved save-dtype casts and their signature record (extract D11).
    ragged:
        The validated ragged mode (extract D4).
    shard_format:
        Validated shard format (``None`` = default: recorded format on
        resume, safetensors on fresh runs).
    checksums:
        The D7 integrity level.
    pipeline_id:
        Caller-asserted pipeline identity for callable-identity resume
        overrides (extract D8).
    acknowledge_v1_prefix:
        The explicit in-progress-v1 resume acknowledgment (extract D16).
    transform_raw:
        The caller's original ``transform=`` value (classified for the
        callable-identity block when chains hold opaque steps).
    input_transform:
        The coerced B5 INPUT-transform callable (or ``None``), applied to
        every raw batch before collate (tvscope B5).
    input_block:
        The tvscope B4 manifest input-preprocessing block (or ``None`` for
        the honest undeclared default).
    input_opaque:
        Whether the input path runs an opaque (identification-only)
        callable; opaque input runs refuse resume continuation.
    """

    model: nn.Module
    stimuli: Any
    layers: Any
    layer_plan: dict[str, str]
    layers_kind: str
    batch_size: int
    device: torch.device | str | None
    pipelines: dict[str, TransformPipeline | None]
    transform_record: Any
    progress: bool
    stimulus_ids: list[str] | None
    model_identity: Any
    collate: Any
    pool_policy: dict[str, Any] | None
    pool_record: Any
    dtype_policy: dict[str, Any] | None
    dtype_record: Any
    ragged: str
    shard_format: str | None
    checksums: str
    pipeline_id: str | None
    acknowledge_v1_prefix: bool
    transform_raw: Any
    selector_pipeline: TransformPipeline | None = None
    input_transform: Any = None
    input_block: dict[str, Any] | None = None
    input_opaque: bool = False

    @property
    def resume_verifiable(self) -> bool:
        """Whether every transform in the run reconstructs from its record.

        Returns
        -------
        bool
            ``False`` iff any output key's chain holds an opaque step OR
            the input path runs an opaque callable (tvscope B5: an opaque
            input transform's identity is a disclosure, not a proof); the
            D8 callable identity rules govern output-chain resume there.
        """

        if self.input_opaque:
            return False
        chains = list(self.pipelines.values()) + [self.selector_pipeline]
        return all(pipeline is None or pipeline.resume_verifiable for pipeline in chains)


def _pipeline_for(plan: _RunPlan, key: str) -> TransformPipeline | None:
    """Look up the transform chain applying to one output key.

    Selector runs carry ONE global chain (per-site mappings refuse at
    entry: the namespaced child keys are unknowable before batch zero).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    key:
        Output key.

    Returns
    -------
    TransformPipeline | None
        The chain, or ``None``.
    """

    if key in plan.pipelines:
        return plan.pipelines[key]
    return plan.selector_pipeline


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


def _move_envelope_to_device(
    envelope: BatchEnvelope, device: torch.device | str | None
) -> BatchEnvelope:
    """Move an envelope's payloads to a device (mask included).

    Parameters
    ----------
    envelope:
        Collated envelope.
    device:
        Target device, or ``None``.

    Returns
    -------
    BatchEnvelope
        The moved envelope.
    """

    if device is None:
        return envelope
    return dataclasses.replace(
        envelope,
        args=tuple(_move_nested_to_device(item, device) for item in envelope.args),
        kwargs={key: _move_nested_to_device(item, device) for key, item in envelope.kwargs.items()},
        mask=envelope.mask.to(device) if envelope.mask is not None else None,
    )


def _iter_item_batches(stimuli: Any, batch_size: int) -> Iterator[Any]:
    """Yield raw per-batch material: tensor slices or item lists.

    Parameters
    ----------
    stimuli:
        Tensor with batch dimension or iterable stimulus set.
    batch_size:
        Number of items per batch.

    Yields
    ------
    Any
        A tensor slice, or a list of stimulus items.
    """

    if isinstance(stimuli, torch.Tensor):
        for start in range(0, stimuli.shape[0], batch_size):
            yield stimuli[start : start + batch_size]
        return
    batch: list[Any] = []
    for item in stimuli:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _collate_to_envelope(
    plan: _RunPlan, raw: Any, run_state: dict[str, Any], batch_index: int
) -> BatchEnvelope:
    """Collate one batch's raw material into the typed envelope (D9).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    raw:
        Tensor slice or list of stimulus items.
    run_state:
        Mutable per-run state (tokenizer cache, disclosures).
    batch_index:
        Zero-based batch index.

    Returns
    -------
    BatchEnvelope
        The typed, device-moved envelope.
    """

    if isinstance(raw, torch.Tensor):
        envelope = BatchEnvelope(
            args=(raw,),
            kwargs={},
            row_count=int(raw.shape[0]),
            mask=None,
            disclosure={"kind": "tensor_slice"},
        )
    elif plan.collate is not None:
        if getattr(plan.collate, "__tl_engine_collate__", False):
            envelope = plan.collate(list(raw), plan.model, run_state)
        else:
            envelope = coerce_envelope(
                plan.collate(list(raw)), n_items=len(raw), source="user", batch_index=batch_index
            )
            envelope.disclosure.setdefault("kind", "user")
    else:
        envelope = default_collate(list(raw), plan.model, run_state)
    return _move_envelope_to_device(envelope, plan.device)


def _positional_forward_parameters(model: nn.Module) -> list[Any] | None:
    """Return ``model.forward``'s bindable positional parameters, if inspectable.

    Parameters
    ----------
    model:
        Model whose forward signature is inspected (declaration-based;
        never arity sniffing).

    Returns
    -------
    list | None
        Positional parameters of ``forward`` (``self`` excluded), or
        ``None`` when the signature is not inspectable or contains
        ``*args``.
    """

    import inspect

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


def _locate_positional_mask(
    model: nn.Module, envelope: BatchEnvelope
) -> tuple[BatchEnvelope, list[Any] | None, bool]:
    """Locate a checkable ``attention_mask`` in a POSITIONAL envelope.

    Detection is declaration-based, keyed on the name ``attention_mask``:
    positional args are bound against ``model.forward``'s signature. Bare
    tensor batches carry no mask by construction.

    Parameters
    ----------
    model:
        Model about to consume the batch.
    envelope:
        Collated envelope (kwargs-shaped envelopes carry their mask
        already).

    Returns
    -------
    tuple[BatchEnvelope, list | None, bool]
        The (possibly mask-annotated) envelope, the positional forward
        parameters when binding ran, and whether ``position_ids`` already
        rides the batch.
    """

    if envelope.mask is not None or envelope.kwargs or len(envelope.args) < 2:
        return envelope, None, False
    parameters = _positional_forward_parameters(model)
    if parameters is None or len(envelope.args) > len(parameters):
        return envelope, parameters, False
    names = [param.name for param in parameters[: len(envelope.args)]]
    position_ids_present = "position_ids" in names
    if "attention_mask" in names:
        candidate = envelope.args[names.index("attention_mask")]
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 2 and candidate.shape[-1] > 0:
            return (
                dataclasses.replace(envelope, mask=candidate),
                parameters,
                position_ids_present,
            )
    return envelope, parameters, position_ids_present


def _inject_position_ids_positionally(
    model: nn.Module,
    envelope: BatchEnvelope,
    parameters: list[Any],
    position_ids: torch.Tensor,
    batch_index: int,
) -> BatchEnvelope:
    """Rebuild a positional envelope with derived ``position_ids`` in its slot.

    Intermediate parameters between the batch's last element and the
    ``position_ids`` slot are filled with their declared defaults; a gap
    parameter without a default makes injection impossible and raises the
    same typed refusal (fail closed, never a guessed value).

    Parameters
    ----------
    model:
        Model whose forward signature drives the rebuild.
    envelope:
        Original positional envelope.
    parameters:
        ``model.forward``'s positional parameters.
    position_ids:
        Derived position ids to place.
    batch_index:
        Zero-based batch index for refusal messages.

    Returns
    -------
    BatchEnvelope
        The envelope with ``position_ids`` filled positionally.
    """

    import inspect

    names = [param.name for param in parameters]
    target = names.index("position_ids")
    rebuilt = list(envelope.args)
    for index in range(len(envelope.args), target + 1):
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
    return dataclasses.replace(envelope, args=tuple(rebuilt))


def _mask_not_right_aligned(mask: torch.Tensor) -> bool:
    """Return whether any mask row has a pad position before a token.

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
    """Derive per-row position indices from an attention mask (D5).

    Parameters
    ----------
    mask:
        Two-dimensional attention mask (nonzero = token present).

    Returns
    -------
    torch.Tensor
        ``int64`` position ids shaped like ``mask``.
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


def _correct_envelope_positions(
    model: nn.Module, envelope: BatchEnvelope, batch_index: int, run_state: dict[str, Any]
) -> tuple[BatchEnvelope, str]:
    """Correct or refuse non-right-aligned pad geometry on one envelope (D5).

    Parameters
    ----------
    model:
        Model about to consume the batch.
    envelope:
        Collated, device-moved envelope.
    batch_index:
        Zero-based batch index.
    run_state:
        Mutable per-run dict (deduplicates the disclosure warning).

    Returns
    -------
    tuple[BatchEnvelope, str]
        The (possibly rebuilt) envelope and its ``position_ids_source``:
        ``"caller"`` (positions arrived in the batch), ``"derived"`` (the
        mask-derived correction was injected), or ``"model_default"``.
    """

    envelope, positional_params, positional_position_ids = _locate_positional_mask(model, envelope)
    mask = envelope.mask
    if mask is None:
        return envelope, "model_default"
    if envelope.kwargs.get("position_ids") is not None or positional_position_ids:
        return envelope, "caller"
    if not _mask_not_right_aligned(mask):
        return envelope, "model_default"
    parameters = (
        positional_params
        if positional_params is not None
        else _positional_forward_parameters(model)
    )
    accepts = parameters is not None and any(param.name == "position_ids" for param in parameters)
    if not accepts:
        detail = (
            "this model's forward does not declare a position_ids "
            "parameter, so absolute positions cannot be corrected."
            if parameters is not None
            else "this model's forward signature is not inspectable, so "
            "a position_ids correction cannot be proven to apply."
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
    if envelope.kwargs:
        corrected = dict(envelope.kwargs)
        corrected["position_ids"] = position_ids
        return dataclasses.replace(envelope, kwargs=corrected), "derived"
    narrowed = cast("list[Any]", parameters)  # accepts above proved it
    return (
        _inject_position_ids_positionally(model, envelope, narrowed, position_ids, batch_index),
        "derived",
    )


@contextlib.contextmanager
def _inference_guard(model: nn.Module) -> Iterator[None]:
    """Run extraction forwards under ``no_grad`` + ``eval`` with exact restore.

    ``torch.no_grad`` is used deliberately instead of ``inference_mode``
    (inference-mode tensors poison later autograd use); every submodule's
    exact ``training`` flag is snapshotted and restored in a ``finally``
    block, exception paths included (extract D3).

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


@contextlib.contextmanager
def _warning_dedup(run_state: dict[str, Any]) -> Iterator[None]:
    """Deduplicate capture warnings with counts for the manifest (D9).

    The first instance of each distinct warning is shown; repeats are
    counted into ``run_state["warning_counts"]`` and the counts ride the
    manifest at terminal status. Extraction is single-threaded by design,
    so patching ``warnings.showwarning`` for the run's duration is sound.

    Parameters
    ----------
    run_state:
        Mutable per-run state receiving the counts.
    """

    counts: dict[str, int] = run_state.setdefault("warning_counts", {})
    original = warnings.showwarning

    def _dedup(message, category, filename, lineno, file=None, line=None):  # type: ignore[no-untyped-def]  # noqa: PLR0913 - warnings.showwarning's fixed signature
        """Count one warning, forwarding only its first instance."""

        key = f"{category.__name__}: {message}"
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            original(message, category, filename, lineno, file, line)

    warnings.showwarning = _dedup
    try:
        yield
    finally:
        warnings.showwarning = original


# ---------------------------------------------------------------------------
# Capture + store pipeline
# ---------------------------------------------------------------------------


def _capture_outputs(
    plan: _RunPlan,
    envelope: BatchEnvelope,
    run_state: dict[str, Any],
    batch_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Run the model call and harvest the selected outputs (D13 model step).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    envelope:
        The corrected, device-moved batch envelope.
    run_state:
        Mutable per-run state (holds the frozen selector plan).
    batch_index:
        Zero-based batch index.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, Any] | None]
        Captured outputs by output key, and — on the legacy string path —
        the resolved ``Layer`` views for batch-zero metadata (``None`` on
        the selector path, whose plan entries carry site identity).
    """

    import torchlens as _tl

    if plan.layers_kind == "selector":
        frozen: SelectorPlan | None = run_state.get("selector_plan")
        if frozen is None:
            frozen, frozen_outputs = freeze_selector_plan(
                plan.model, envelope.args, dict(envelope.kwargs), plan.layers
            )
            run_state["selector_plan"] = frozen
            return frozen_outputs, None
        return (
            attest_selector_batch(
                plan.model,
                (envelope.args, dict(envelope.kwargs)),
                plan.layers,
                frozen,
                batch_index,
            ),
            None,
        )
    if envelope.kwargs:
        trace = _tl.trace(
            plan.model,
            envelope.args,
            dict(envelope.kwargs),
            capture=_tl.options.CaptureOptions(layers_to_save=list(plan.layer_plan.values())),
        )
        outputs: dict[str, torch.Tensor] = {}
        views: dict[str, Any] = {}
        if plan.layers_kind == "mapping":
            for label, pattern in plan.layer_plan.items():
                view = trace[pattern]
                outputs[label] = view.out
                views[label] = view
            return outputs, views
        for pattern in plan.layer_plan.values():
            matches = _tl._matching_saved_layer_labels(trace, pattern)
            if not matches:
                raise ValueError(f"Layer lookup {pattern!r} did not resolve to a saved layer.")
            for match in matches:
                view = trace[match]
                outputs[match] = view.out
                views[match] = view
        return outputs, views
    _trace, outputs, views = _tl._extract_layers_with_trace(
        plan.model, envelope.first_input(), plan.layers
    )
    return outputs, views


def _apply_site_pipeline(
    pipeline: TransformPipeline | None, key: str, tensor: torch.Tensor
) -> torch.Tensor:
    """Apply one output key's chain with the T-C2 stimulus-axis guard.

    Dispatch inside the chain is by DECLARATION (transforms memo P2): raw
    callables are invoked unary, declared context steps receive the context
    — never ``inspect.signature``.

    Parameters
    ----------
    pipeline:
        The output key's coerced chain, or ``None``.
    key:
        Output key (rides the context for per-site refusal text).
    tensor:
        Batch tensor, stimulus axis leading.

    Returns
    -------
    torch.Tensor
        The transformed tensor.

    Raises
    ------
    torchlens.transforms.TransformContractError
        ``transform_output_invalid`` when a step returns a non-tensor;
        ``transform_row_axis_violated`` when the stimulus axis changed.
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


def _apply_ragged_gate(
    plan: _RunPlan,
    key: str,
    pooled: torch.Tensor,
    run_state: dict[str, Any],
) -> bool:
    """Apply the D4 entry gate to one POST-POOL tensor.

    Pool is the documented raggedness-killer, so the gate runs on the
    pooled product: batch zero freezes each key's pooled per-stimulus
    shape and the trimmed-key set; later drift refuses typed
    (``ragged="refuse"``) or rides the trimmed carrier (``"trim"``).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    key:
        Output key.
    pooled:
        The post-pool, pre-transform tensor.
    run_state:
        Mutable per-run state (frozen shapes, trimmed-key set, and the
        current batch's index + mask geometry).

    Returns
    -------
    bool
        Whether this key is stored TRIMMED.
    """

    batch_index = int(run_state.get("current_batch_index", -1))
    geometry = run_state.get("current_geometry")
    frozen_shapes: dict[str, list[int]] = run_state.setdefault("pooled_shapes", {})
    trimmed: set[str] = run_state.setdefault("trimmed_keys", set())
    mask_width = run_state.get("current_mask_width")
    mask_shaped = (
        geometry is not None
        and pooled.ndim >= 2
        and mask_width is not None
        and pooled.shape[1] == mask_width
    )
    if key not in frozen_shapes:
        frozen_shapes[key] = list(pooled.shape[1:])
        if plan.ragged == "trim" and mask_shaped:
            trimmed.add(key)
        return key in trimmed
    expected = frozen_shapes[key]
    observed = list(pooled.shape[1:])
    if key in trimmed:
        return True  # trimmed keys legitimately vary on the token axis
    if observed == expected:
        return False
    if plan.ragged == "trim" and mask_shaped and observed[1:] == expected[1:]:
        # The key is mask-shaped but batch zero happened to match widths;
        # admit it to the trimmed set now (drift proves raggedness).
        trimmed.add(key)
        return True
    raise_ragged_refusal(key, batch_index, expected, observed)
    return False  # unreachable: the refusal always raises


def _store_batch(
    plan: _RunPlan,
    batch_outputs: dict[str, torch.Tensor],
    context: BatchContext,
    run_state: dict[str, Any],
) -> tuple[dict[str, torch.Tensor | RaggedBatch], dict[str, dict[str, Any]]]:
    """Run the store pipeline on one batch (D13 order after the model call).

    pool (captured dtype, raw axes, mask) -> postprocess -> row validation
    -> ragged trim -> save-dtype cast -> CPU-contiguous snapshot, with the
    D7 value reduction computed ON the tensor's device BEFORE the host
    copy.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    batch_outputs:
        Captured outputs by key.
    context:
        This batch's context.
    run_state:
        Mutable per-run state (trimmed-key set, dtype disclosures).

    Returns
    -------
    tuple[dict, dict]
        ``(processed, key_facts)``: snapshot payloads for the writer and
        the per-key ledger facts.
    """

    processed: dict[str, torch.Tensor | RaggedBatch] = {}
    key_facts: dict[str, dict[str, Any]] = {}
    geometry = run_state.get("current_geometry")
    for key, tensor in batch_outputs.items():
        pooled = apply_pool(key, tensor, plan.pool_policy, context.mask)
        key_trimmed = _apply_ragged_gate(plan, key, pooled, run_state)
        stored = _apply_site_pipeline(_pipeline_for(plan, key), key, pooled)
        stored, dtype_fact = cast_for_store(
            key, stored, plan.dtype_policy, plan.shard_format or "safetensors"
        )
        if dtype_fact is not None:
            run_state.setdefault("dtype_facts", {})[key] = dtype_fact
        if key_trimmed:
            if geometry is None:
                raise_ragged_refusal(key, context.batch_index, ["<mask>"], list(stored.shape[1:]))
            carrier = trim_batch(key, stored, cast("dict[str, Any]", geometry))
            values = carrier.values
            facts: dict[str, Any] = {
                "layout": "trimmed",
                "per_stimulus_shape": [None] + list(values.shape[1:]),
                "dtype": str(values.dtype),
                "n_values": int(values.shape[0]),
            }
            if plan.checksums != "none":
                facts["value_reduction"] = value_reduction(key, values)
            processed[key] = RaggedBatch(
                values=values.detach().cpu().contiguous(),
                offsets=carrier.offsets,
                row_shapes=carrier.row_shapes,
            )
            key_facts[key] = facts
            continue
        facts = {
            "per_stimulus_shape": list(stored.shape[1:]),
            "dtype": str(stored.dtype),
        }
        if plan.checksums != "none":
            # D7: the order-sensitive value reduction runs ON the tensor's
            # device BEFORE the host copy, so a corruption between here and
            # the written file is catchable later.
            facts["value_reduction"] = value_reduction(key, stored)
        processed[key] = stored.detach().cpu().contiguous()
        key_facts[key] = facts
    validate_row_counts(context, processed)
    return processed, key_facts


# ---------------------------------------------------------------------------
# Signature + manifest
# ---------------------------------------------------------------------------


def _freeze_transform_plans(
    plan: _RunPlan, batch_outputs: dict[str, torch.Tensor], trimmed_keys: set[str]
) -> dict[str, Any]:
    """Plan every output key's chain against batch zero (the T-C6 freeze).

    Trimmed keys (D4) freeze their token axis as ``None`` — ragged shape is
    validated per shard against the mask geometry, never inferred forever
    from batch one.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    batch_outputs:
        Batch zero's captured (pre-transform) tensors, keyed by output key.
    trimmed_keys:
        Keys stored as the trimmed carrier.

    Returns
    -------
    dict[str, Any]
        Per-key frozen plan rows.
    """

    from ..transforms import TensorSpec

    plans: dict[str, Any] = {}
    for key, tensor in batch_outputs.items():
        pipeline = _pipeline_for(plan, key)
        if pipeline is None or not pipeline.steps:
            plans[key] = None
            continue
        planned = pipeline.plan(TensorSpec.of(tensor), TransformContext(site_label=key))
        final = planned[-1] if planned else None
        final_shape = None if final is None else list(final.output.shape[1:])
        if final_shape and key in trimmed_keys:
            final_shape[0] = None
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
                "per_stimulus_shape": final_shape,
                "dtype": final.output.dtype,
            },
        }
    return plans


def _validate_against_plan(
    manifest: dict[str, Any], processed: Mapping[str, torch.Tensor | RaggedBatch]
) -> None:
    """Refuse a shard whose observed output contradicts the frozen plan (T-C6).

    Parameters
    ----------
    manifest:
        The artifact manifest holding the batch-zero frozen plans.
    processed:
        The shard's stored (post-transform, pre-cast-disclosure) payloads.

    Raises
    ------
    torchlens.transforms.TransformContractError
        ``transform_plan_violated`` naming the key, the plan, and the
        observation; the shard is refused BEFORE publication.
    """

    plans = (manifest.get("run") or {}).get("transform_plans") or {}
    for key, stored in processed.items():
        final = (plans.get(key) or {}).get("final_output") if plans.get(key) else None
        if not final or isinstance(stored, RaggedBatch):
            continue
        expected_shape = list(final.get("per_stimulus_shape") or [])
        expected_dtype = final.get("dtype")
        observed_shape = list(stored.shape[1:])
        shape_ok = len(observed_shape) == len(expected_shape) and all(
            want is None or want == got
            for want, got in zip(expected_shape, observed_shape, strict=True)
        )
        dtype_ok = expected_dtype is None or str(stored.dtype) == expected_dtype
        if not shape_ok or not dtype_ok:
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


def _layer_metadata(
    plan: _RunPlan,
    layer_views: dict[str, Any] | None,
    batch_outputs: dict[str, torch.Tensor],
    processed: Mapping[str, torch.Tensor | RaggedBatch],
    run_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the per-site self-description block from batch zero.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    layer_views:
        Resolved ``Layer`` views (legacy string path), or ``None`` (the
        selector path reads the frozen plan off ``run_state``).
    batch_outputs:
        Batch zero's captured tensors by output key.
    processed:
        Batch zero's stored payloads by output key.
    run_state:
        Per-run state (trimmed keys, dtype facts).

    Returns
    -------
    dict[str, dict[str, Any]]
        Site identity, axis semantics, dtype, layout, and device per key.
    """

    from ..errors._base import TorchLensError

    trimmed: set[str] = run_state.get("trimmed_keys", set())
    selector_plan: SelectorPlan | None = run_state.get("selector_plan")
    plan_sites = (
        {key: (label, site) for key, label, site in selector_plan.entries}
        if selector_plan is not None
        else {}
    )
    metadata: dict[str, dict[str, Any]] = {}
    for key, captured in batch_outputs.items():
        if layer_views is not None and key in layer_views:
            layer = layer_views[key]
            label = str(layer.layer_label)
            site_key: str | None
            site_key_unavailable: str | None
            try:
                site_key = str(layer.site_key)
                site_key_unavailable = None
            except TorchLensError as exc:
                site_key = None
                code = exc.fields.get("code") if isinstance(exc.fields, dict) else None
                site_key_unavailable = str(code or type(exc).__name__)
        else:
            label, site_key = plan_sites.get(key, (key, None))
            site_key_unavailable = None if site_key is not None else "site_key_unavailable"
        stored = processed[key]
        if isinstance(stored, RaggedBatch):
            stored_shape: list[Any] = [None] + list(stored.values.shape[1:])
            stored_dtype = str(stored.values.dtype)
            layout = "trimmed"
        else:
            stored_shape = list(stored.shape[1:])
            stored_dtype = str(stored.dtype)
            layout = "dense"
        entry: dict[str, Any] = {
            "layer_label": label,
            "site_key": site_key,
            "site_key_unavailable": site_key_unavailable,
            "captured_dtype": str(captured.dtype),
            "captured_device": str(captured.device),
            "per_stimulus_shape": list(captured.shape[1:]),
            "stored_dtype": stored_dtype,
            "stored_per_stimulus_shape": stored_shape,
            "layout": layout,
            "batch_axis": 0,
        }
        if key in trimmed:
            entry["stored_per_stimulus_shape"] = [None] + (stored_shape[1:] if stored_shape else [])
        dtype_fact = (run_state.get("dtype_facts") or {}).get(key)
        if dtype_fact is not None:
            entry["dtype_conversion"] = dtype_fact
        metadata[key] = entry
    return metadata


# ---------------------------------------------------------------------------
# Stimulus ids (item 7)
# ---------------------------------------------------------------------------


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
        ``"sha256:..."`` over the ordered id slice, or ``None``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_stimulus_ids_cardinality`` when the id list runs out
        before the shard's rows (refused BEFORE the commit).
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


def _check_ids_complete(plan: _RunPlan, n_stimuli: int) -> None:
    """Refuse completion when surplus identifiers remain (extract D2).

    Unsized iterables advance examples and IDs in lockstep; exact equality
    is checked BEFORE completion — a mismatch never yields
    ``status=complete``.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    n_stimuli:
        Total extracted stimulus rows.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_stimulus_ids_cardinality`` on surplus identifiers.
    """

    if plan.stimulus_ids is not None and len(plan.stimulus_ids) != n_stimuli:
        raise InvalidArgumentError(
            f"stimulus_ids supplies {len(plan.stimulus_ids)} identifiers but "
            f"the stimuli yielded {n_stimuli} rows; the artifact stays "
            "in_progress rather than completing with a mislabeled id list.",
            code="extraction_stimulus_ids_cardinality",
            remedy="pass exactly one identifier per stimulus, in order",
            n_ids=len(plan.stimulus_ids),
            n_stimuli=n_stimuli,
        )


# ---------------------------------------------------------------------------
# Resume: callable identity rules + skip-replay verification
# ---------------------------------------------------------------------------


def _consume_skipped_with_replay(
    plan: _RunPlan, completed_rows: list[dict[str, Any]], run_state: dict[str, Any]
) -> Any:
    """Advance past extracted stimuli, verifying the replayed prefix (item 16).

    Per-shard ``input_digest`` is recomputed during the EXISTING
    skipped-prefix replay — zero extra forward passes — turning "iterable
    stimuli are unverifiable" into "verified across the entire replayed
    prefix". Rows without a ledgered digest (older artifacts) advance
    without the comparison.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    completed_rows:
        The trusted ledger prefix.
    run_state:
        Mutable per-run state (tokenizer cache for the replay collation).

    Returns
    -------
    Any
        Remaining stimuli: a tensor slice, or the advanced iterator.

    Raises
    ------
    DatasetExtractionResumeError
        ``extraction_resume_signature_mismatch`` when the stimulus stream
        ends before covering the prefix;
        ``extraction_resume_input_mismatch`` when a replayed shard's input
        digest contradicts the ledgered fact.
    """

    n_skip = sum(int(row["n_rows"]) for row in completed_rows)
    if n_skip == 0:
        return plan.stimuli
    if isinstance(plan.stimuli, torch.Tensor):
        for row in completed_rows:
            _verify_replayed_row(plan, row, plan.stimuli, run_state)
        return plan.stimuli[n_skip:]
    iterator = iter(plan.stimuli)
    consumed = 0
    for row in completed_rows:
        items: list[Any] = []
        for _ in range(int(row["n_rows"])):
            try:
                items.append(next(iterator))
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
        _verify_replayed_row(plan, row, items, run_state)
    return iterator


def _verify_replayed_row(
    plan: _RunPlan, row: Mapping[str, Any], material: Any, run_state: dict[str, Any]
) -> None:
    """Compare one replayed shard's input digest against its ledger fact.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    row:
        The trusted ledger row.
    material:
        The full stimulus tensor (sliced here) or the shard's item list.
    run_state:
        Mutable per-run state.

    Raises
    ------
    DatasetExtractionResumeError
        ``extraction_resume_input_mismatch`` naming the shard.
    """

    recorded = row.get("input_digest")
    if not recorded:
        return
    if isinstance(material, torch.Tensor):
        start = int(row["row_start"])
        raw: Any = material[start : start + int(row["n_rows"])]
    else:
        raw = material
    # The ledgered digest is a fact about what the MODEL consumed, so the
    # replay applies the same B5 input transform before collating (opaque
    # input transforms never reach here: they refuse continuation typed).
    raw = apply_input_transform(plan.input_transform, raw)
    envelope = _collate_to_envelope(plan, raw, run_state, int(row.get("index", -1)))
    observed = envelope_input_digest(envelope)
    if observed != recorded:
        raise DatasetExtractionResumeError(
            f"Replayed shard {row.get('file')!r} collates to a DIFFERENT "
            "input than the one its activations were extracted from "
            f"(ledgered {recorded}, replayed {observed}); the stimuli, their "
            "order, or the collation changed since the prefix was written.",
            code="extraction_resume_input_mismatch",
            remedy=(
                "re-run with the original stimuli in the original order (and "
                "a deterministic collate), or extract into a fresh directory"
            ),
            shard=row.get("file"),
            ledgered_digest=recorded,
            replayed_digest=observed,
        )


# ---------------------------------------------------------------------------
# Disk preparation + engines
# ---------------------------------------------------------------------------


def _batch_iterable(plan: _RunPlan, remaining: Any) -> Iterable[Any]:
    """Build the (optionally progress-wrapped) raw batch iterator for a run.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    remaining:
        Stimuli still to extract.

    Returns
    -------
    Iterable[Any]
        Tensor slices or item lists.
    """

    batches: Iterable[Any] = _iter_item_batches(remaining, plan.batch_size)
    total = None
    if isinstance(remaining, torch.Tensor):
        total = (remaining.shape[0] + plan.batch_size - 1) // plan.batch_size
    if plan.progress:
        from ..utils.display import progress_bar

        batches = progress_bar(batches, total=total, desc="torchlens.extract", enabled=True)
    return batches


def _process_one_batch(
    plan: _RunPlan,
    raw: Any,
    batch_index: int,
    row_start: int,
    run_state: dict[str, Any],
) -> tuple[
    BatchEnvelope,
    BatchContext,
    dict[str, torch.Tensor],
    dict[str, Any] | None,
    dict[str, torch.Tensor | RaggedBatch],
    dict[str, dict[str, Any]],
    str,
]:
    """Run the full D13 pipeline on one raw batch.

    Parameters
    ----------
    plan:
        Resolved run configuration.
    raw:
        Tensor slice or item list.
    batch_index:
        Zero-based batch index.
    row_start:
        Global row index of this batch's first stimulus.
    run_state:
        Mutable per-run state.

    Returns
    -------
    tuple
        ``(envelope, context, batch_outputs, layer_views, processed,
        key_facts, input_digest)``.
    """

    raw = apply_input_transform(plan.input_transform, raw)
    envelope = _collate_to_envelope(plan, raw, run_state, batch_index)
    input_digest = envelope_input_digest(envelope)
    envelope, position_source = _correct_envelope_positions(
        plan.model, envelope, batch_index, run_state
    )
    geometry = mask_row_geometry(envelope.mask) if envelope.mask is not None else None
    run_state["current_geometry"] = geometry
    run_state["current_batch_index"] = batch_index
    run_state["current_mask_width"] = (
        int(envelope.mask.shape[1]) if envelope.mask is not None else None
    )
    batch_outputs, layer_views = _capture_outputs(plan, envelope, run_state, batch_index)
    ids_slice = (
        tuple(plan.stimulus_ids[row_start : row_start + envelope.row_count])
        if plan.stimulus_ids is not None
        else None
    )
    context = BatchContext(
        batch_index=batch_index,
        row_start=row_start,
        row_count=envelope.row_count,
        stimulus_ids=ids_slice,
        mask=envelope.mask,
        row_extents=(
            tuple(zip(geometry["starts"], geometry["extents"], strict=True))
            if geometry is not None
            else None
        ),
        padding_side=geometry["padding_side"] if geometry is not None else None,
        position_ids_source=position_source,
        device=str(plan.device) if plan.device is not None else None,
        collate_disclosure=envelope.disclosure,
    )
    processed, key_facts = _store_batch(plan, batch_outputs, context, run_state)
    return envelope, context, batch_outputs, layer_views, processed, key_facts, input_digest


def run_in_memory(plan: _RunPlan) -> dict[str, torch.Tensor]:
    """Run the in-memory extraction engine (pool/dtype/collate aware).

    Parameters
    ----------
    plan:
        Resolved run configuration.

    Returns
    -------
    dict[str, torch.Tensor]
        Concatenated outputs keyed as disk mode keys them.
    """

    if plan.device is not None:
        plan.model.to(plan.device)
    accumulator: dict[str, list[torch.Tensor]] = {}
    run_state: dict[str, Any] = {}
    row_start = 0
    with _inference_guard(plan.model), _warning_dedup(run_state):
        for batch_index, raw in enumerate(_batch_iterable(plan, plan.stimuli)):
            (_env, _ctx, _outs, _views, processed, _facts, _digest) = _process_one_batch(
                plan, raw, batch_index, row_start, run_state
            )
            for key, stored in processed.items():
                if isinstance(stored, RaggedBatch):
                    continue  # unreachable: memory mode refuses ragged='trim' at entry
                accumulator.setdefault(key, []).append(stored)
            row_start += _ctx.row_count
    return {label: torch.cat(tensors, dim=0) for label, tensors in accumulator.items()}


def _freeze_batch_zero(
    plan: _RunPlan,
    writer: ArtifactWriter,
    batch_zero: tuple[
        dict[str, Any] | None, dict[str, torch.Tensor], dict[str, Any], BatchEnvelope
    ],
    run_state: dict[str, Any],
) -> None:
    """Perform the ONE batch-zero manifest write (layers + plans; D1).

    Parameters
    ----------
    plan:
        Resolved run configuration.
    writer:
        The bound artifact writer.
    batch_zero:
        ``(layer_views, batch_outputs, processed, envelope)`` from the
        first computed batch.
    run_state:
        Mutable per-run state (frozen selector plan, trimmed keys).
    """

    layer_views, batch_outputs, processed, envelope = batch_zero
    manifest = writer.manifest
    selector_plan: SelectorPlan | None = run_state.get("selector_plan")
    manifest["layers"] = _layer_metadata(plan, layer_views, batch_outputs, processed, run_state)
    run_block = manifest.setdefault("run", {})
    run_block["transform_plans"] = _freeze_transform_plans(
        plan, batch_outputs, run_state.get("trimmed_keys", set())
    )
    if selector_plan is not None:
        run_block["selector_plan"] = selector_plan.to_record()
    run_block["collate_observed"] = dict(envelope.disclosure)
    writer.write_manifest()


def run_to_disk(plan: _RunPlan, container_path: Path, resume: bool) -> list[Path]:
    """Run the disk-mode extraction engine (the v2 commit protocol).

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

    plan, writer, completed_rows, complete_paths, prep_state = _prepare_disk_run(
        plan, container_path, resume
    )
    if complete_paths is not None:
        # A completed compatible resume is a true no-op: the model is neither
        # moved to a device nor mode-flipped (extract MEMO D3).
        return complete_paths
    if plan.device is not None:
        plan.model.to(plan.device)
    manifest = writer.manifest
    run_state: dict[str, Any] = {}
    if prep_state.get("selector_plan_record") is not None:
        run_state["selector_plan"] = prep_state["selector_plan_record"]
    remaining = _consume_skipped_with_replay(plan, completed_rows, run_state)
    n_skip = sum(int(row["n_rows"]) for row in completed_rows)
    start_index = len(completed_rows)
    container_paths = [container_path / str(row["file"]) for row in completed_rows]
    writer.shard_extension = shard_extension(plan.shard_format or "safetensors")
    writer.checksums = plan.checksums

    row_start = n_skip
    n_committed = start_index
    position_sources: set[str] = set()
    with _inference_guard(plan.model), _warning_dedup(run_state):
        for offset, raw in enumerate(_batch_iterable(plan, remaining)):
            batch_index = start_index + offset
            (
                envelope,
                context,
                batch_outputs,
                layer_views,
                processed,
                key_facts,
                input_digest,
            ) = _process_one_batch(plan, raw, batch_index, row_start, run_state)
            position_sources.add(context.position_ids_source)
            if manifest.get("layers") is None:
                _freeze_batch_zero(
                    plan, writer, (layer_views, batch_outputs, processed, envelope), run_state
                )
            _validate_against_plan(manifest, processed)
            ids_range = _ids_range_digest(plan, row_start, context.row_count)
            geometry = run_state.get("current_geometry")
            row_facts: dict[str, Any] = {
                "keys": key_facts,
                "ids_range_digest": ids_range,
                "input_digest": input_digest,
                "position_ids_source": context.position_ids_source,
            }
            if geometry is not None:
                row_facts["row_geometry"] = {
                    "starts": geometry["starts"],
                    "extents": geometry["extents"],
                    "padding_side": geometry["padding_side"],
                    "contiguous": geometry["contiguous"],
                }
            row = writer.commit_shard(
                index=batch_index,
                row_start=row_start,
                n_rows=context.row_count,
                save_payload=functools.partial(
                    write_shard, payload=processed, shard_format=plan.shard_format or "safetensors"
                ),
                row_facts=row_facts,
            )
            _run_post_store_hooks(context, row)
            container_paths.append(container_path / str(row["file"]))
            row_start += context.row_count
            n_committed += 1

    _check_ids_complete(plan, row_start)
    manifest["stimulus_provenance"]["n_stimuli"] = row_start
    run_block = manifest.setdefault("run", {})
    run_block["warnings"] = dict(run_state.get("warning_counts") or {})
    run_block["position_ids_sources"] = sorted(position_sources)
    if prep_state.get("resume_audit"):
        run_block.setdefault("resume_audit", []).extend(prep_state["resume_audit"])
    writer.finalize(n_shards=n_committed, n_stimuli=row_start)
    return container_paths
