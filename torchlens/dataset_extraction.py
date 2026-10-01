"""Batched dataset extraction with a self-describing, resumable disk artifact.

This module owns :func:`torchlens.extract_dataset`'s implementation. In disk
mode (``output_dir=``) the artifact directory is SELF-DESCRIBING: alongside the
``batch_XXXXX.pt`` shards it carries a ``manifest.json`` recording site
identity (layer label plus structural site key where derivable), stimulus
ordering and provenance, axis semantics, dtypes, devices, the transform
disclosure, and the TorchLens version — so the artifact can be handed to a
collaborator who never saw the producing script.

The manifest doubles as the RESUME ledger: shard writes are atomic (temp file
plus ``os.replace``) and the manifest is atomically rewritten after every
shard with that shard's exact row count, so a run killed mid-extraction can be
resumed with ``resume=True`` from the last completed shard. Resume-from-shard
was chosen over content-addressed caching because one-shot stimulus iterables
cannot be hashed without being consumed, and the shard layout is already the
public on-disk contract.

Every spelling introduced here (``resume=``, ``stimulus_ids=``,
:func:`load_extraction`, :class:`LoadedExtraction`, the manifest schema) is
DOCUMENTED-UNSTABLE pending the naming/UI sprint.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ._errors import _actionable_message, _ActionableErrorMixin
from ._io import _json
from .errors._base import ConfigurationError, TorchLensWarning

#: Manifest schema identifier written to and required from ``manifest.json``.
MANIFEST_SCHEMA = "tl_extract_manifest_v1"

#: Filename of the self-describing manifest inside an extraction directory.
MANIFEST_FILENAME = "manifest.json"

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
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> None:
    """Append one batch of extracted outs to an accumulator.

    Parameters
    ----------
    accumulator:
        Mutable mapping from layer label to per-batch tensors.
    batch_outputs:
        Extraction output from one batch.
    transform:
        Optional transform applied to each out before storage.
    """

    for layer_name, tensor in batch_outputs.items():
        stored = transform(tensor) if transform is not None else tensor
        accumulator.setdefault(layer_name, []).append(stored.detach().cpu())


def _shard_filename(index: int) -> str:
    """Return the canonical shard filename for a batch index.

    Parameters
    ----------
    index:
        Zero-based batch index.

    Returns
    -------
    str
        Filename of the form ``batch_00042.pt``.
    """

    return f"batch_{index:05d}.pt"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document atomically (temp file plus ``os.replace``).

    Parameters
    ----------
    path:
        Final destination path.
    payload:
        JSON-serializable document.
    """

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Save a torch payload atomically so a partial file never bears the final name.

    Parameters
    ----------
    payload:
        Object passed to ``torch.save``.
    path:
        Final destination path.
    """

    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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


def _transform_signature(
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> dict[str, str] | None:
    """Describe a transform callable for the manifest signature.

    Callable identity cannot be verified across processes; the qualified name
    is a DISCLOSURE (and a resume compatibility check), not a proof.

    Parameters
    ----------
    transform:
        Optional tensor transform supplied by the caller.

    Returns
    -------
    dict[str, str] | None
        ``{"module": ..., "qualname": ...}`` or ``None`` when no transform.
    """

    if transform is None:
        return None
    return {
        "module": getattr(transform, "__module__", "") or "",
        "qualname": getattr(
            transform, "__qualname__", getattr(type(transform), "__qualname__", "")
        ),
    }


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


def _build_signature(
    layer_plan: dict[str, str],
    layers_kind: str,
    batch_size: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
    stimuli: Any,
) -> dict[str, Any]:
    """Build the resume-compatibility signature block of the manifest.

    Parameters
    ----------
    layer_plan:
        Normalized ``output key -> layer lookup`` extraction plan.
    layers_kind:
        ``"mapping"`` or ``"sequence"``, preserving list-versus-dict semantics.
    batch_size:
        Number of stimuli per forward pass.
    transform:
        Optional tensor transform supplied by the caller.
    stimuli:
        Stimulus tensor or iterable.

    Returns
    -------
    dict[str, Any]
        JSON-serializable signature compared verbatim on resume.
    """

    return {
        "layer_plan": dict(layer_plan),
        "layers_kind": layers_kind,
        "batch_size": batch_size,
        "transform": _transform_signature(transform),
        "stimuli": _stimuli_signature(stimuli),
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
    """Create a fresh in-progress manifest document.

    Parameters
    ----------
    signature:
        Resume-compatibility signature block.
    stimulus_ids:
        Optional caller-supplied per-stimulus identifiers, in iteration order.

    Returns
    -------
    dict[str, Any]
        Manifest with an empty batch ledger and no layer metadata yet.
    """

    from . import __version__

    return {
        "schema": MANIFEST_SCHEMA,
        "torchlens_version": __version__,
        "status": "in_progress",
        "signature": signature,
        "stimulus_provenance": {
            "order": (
                "row i of every concatenated activation tensor corresponds to "
                "stimulus i in iteration order of the stimuli argument; within "
                "shard k, global stimulus index = (sum of prior shards' "
                "n_stimuli) + row"
            ),
            "n_stimuli": None,
            "stimulus_ids": list(stimulus_ids) if stimulus_ids is not None else None,
        },
        "storage": {
            "shard_filename_format": "batch_{index:05d}.pt",
            "shard_payload": ("dict[output key -> torch.Tensor] with the stimulus axis leading"),
            "tensor_placement": "cpu",
            "writes": (
                "atomic (temp file + os.replace); a shard file bearing its final name is complete"
            ),
        },
        "layers": None,
        "batches": [],
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
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise DatasetExtractionResumeError(
            f"Extraction manifest {str(manifest_path)!r} does not carry schema "
            f"{MANIFEST_SCHEMA!r} (found {manifest.get('schema') if isinstance(manifest, dict) else type(manifest).__name__!r}).",
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


def _check_resume_signature(
    existing: dict[str, Any], signature: dict[str, Any], manifest_path: Path
) -> None:
    """Refuse a resume whose run parameters differ from the artifact's.

    Parameters
    ----------
    existing:
        Manifest found in the output directory.
    signature:
        Signature block of the current call.
    manifest_path:
        Manifest path, for the refusal message.

    Raises
    ------
    DatasetExtractionResumeError
        If any signature field differs.
    """

    recorded = existing.get("signature")
    if recorded == signature:
        return
    mismatched = sorted(
        key
        for key in set(signature) | set(recorded or {})
        if (recorded or {}).get(key) != signature.get(key)
    )
    raise DatasetExtractionResumeError(
        f"Extraction artifact at {str(manifest_path.parent)!r} was produced by a "
        f"different run configuration (mismatched signature fields: {mismatched}).",
        code="extraction_resume_signature_mismatch",
        remedy=(
            "re-run with the artifact's original layers, batch_size, transform, "
            "and stimuli, or delete the output directory to start fresh"
        ),
        mismatched_fields=mismatched,
        recorded_signature=recorded,
        requested_signature=signature,
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
    transform:
        Optional tensor transform applied before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    stimulus_ids:
        Optional per-stimulus identifiers recorded as provenance.
    """

    model: nn.Module
    stimuli: Any
    layers: Iterable[str] | Mapping[str, str]
    layer_plan: dict[str, str]
    layers_kind: str
    batch_size: int
    device: torch.device | str | None
    transform: Callable[[torch.Tensor], torch.Tensor] | None
    progress: bool
    stimulus_ids: list[str] | None


def extract_dataset(
    model: nn.Module,
    stimuli: Any,
    layers: Iterable[str] | Mapping[str, str],
    batch_size: int = 32,
    device: torch.device | str | None = None,
    output_dir: str | Path | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress: bool = True,
    *,
    resume: bool = False,
    stimulus_ids: Iterable[str] | None = None,
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
        Optional tensor transform applied to each out before storage.
    progress:
        Whether to wrap batch iteration with ``tqdm``.
    resume:
        Disk mode only (DOCUMENTED-UNSTABLE): continue an interrupted run in
        ``output_dir`` from its last completed shard. The recorded run
        signature (layers, batch size, transform disclosure, stimulus
        descriptor) must match; iterable stimuli are assumed to replay in the
        original order, which resume cannot verify. A completed artifact
        returns its shard paths without running the model or touching its
        device placement.
    stimulus_ids:
        Optional per-stimulus identifiers (DOCUMENTED-UNSTABLE), recorded in
        the manifest as provenance in iteration order. Disk mode only: the
        in-memory result is a bare tensor mapping that could neither carry
        nor be affected by validated identifiers, so passing them there is a
        false affordance and refuses typed.

    Returns
    -------
    dict[str, torch.Tensor] | list[pathlib.Path]
        In-memory concatenated outs, or written batch paths.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        If ``resume=True`` or ``stimulus_ids=`` is combined with in-memory
        mode, or a batch's pad geometry is not right-aligned and correct
        ``position_ids`` cannot be derived for this model.
    DatasetExtractionResumeError
        If the artifact in ``output_dir`` cannot be safely continued.

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

    plan = _RunPlan(
        model=model,
        stimuli=stimuli,
        layers=layers,
        layer_plan=_tl._normalize_extract_layers(layers),
        layers_kind="mapping" if isinstance(layers, Mapping) else "sequence",
        batch_size=batch_size,
        device=device,
        transform=transform,
        progress=progress,
        stimulus_ids=list(stimulus_ids) if stimulus_ids is not None else None,
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
            _merge_batch_outputs(accumulator, batch_outputs, plan.transform)
    return {label: torch.cat(tensors, dim=0) for label, tensors in accumulator.items()}


def _prepare_disk_run(
    plan: _RunPlan, container_path: Path, resume: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path] | None]:
    """Prepare the disk-mode manifest and resolve the resume state.

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
    tuple[dict[str, Any], list[dict[str, Any]], list[Path] | None]
        The (written) manifest, the trusted completed-shard ledger rows, and —
        when the artifact is already complete with every shard present — the
        final shard paths (callers return them without running the model).

    Raises
    ------
    DatasetExtractionResumeError
        On unmanifested shard directories or signature mismatches.
    """

    container_path.mkdir(parents=True, exist_ok=True)
    manifest_path = container_path / MANIFEST_FILENAME
    signature = _build_signature(
        plan.layer_plan, plan.layers_kind, plan.batch_size, plan.transform, plan.stimuli
    )
    manifest: dict[str, Any] | None = None
    completed_rows: list[dict[str, Any]] = []
    if resume and manifest_path.exists():
        existing = _load_manifest(manifest_path)
        _check_resume_signature(existing, signature, manifest_path)
        ledgered_total = len(existing.get("batches") or [])
        completed_rows = _completed_prefix(existing, container_path)
        manifest = existing
        manifest["batches"] = list(completed_rows)
        if manifest.get("status") == "complete" and len(completed_rows) == ledgered_total:
            return (
                manifest,
                completed_rows,
                [container_path / str(row["file"]) for row in completed_rows],
            )
        manifest["status"] = "in_progress"
    elif resume and any(container_path.glob("batch_*.pt")):
        raise DatasetExtractionResumeError(
            f"Output directory {str(container_path)!r} contains batch shards "
            "but no manifest; it predates resumable extraction or lost its "
            "ledger, so completed work cannot be verified.",
            code="extraction_resume_unmanifested_dir",
            remedy="delete the output directory (or point output_dir at a fresh one) and re-run",
            output_dir=str(container_path),
        )
    if manifest is None:
        manifest = _base_manifest(signature, plan.stimulus_ids)
    elif plan.stimulus_ids is not None:
        manifest["stimulus_provenance"]["stimulus_ids"] = plan.stimulus_ids
    _clean_orphan_tmp_files(container_path)
    _atomic_write_json(manifest_path, manifest)
    return manifest, completed_rows, None


def _extract_to_disk(plan: _RunPlan, container_path: Path, resume: bool) -> list[Path]:
    """Run the disk-mode extraction engine (atomic shards + manifest ledger).

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
        Every shard path in consumption order, including resumed prefixes.
    """

    import torchlens as _tl

    manifest, completed_rows, complete_paths = _prepare_disk_run(plan, container_path, resume)
    if complete_paths is not None:
        # A completed compatible resume is a true no-op: the model is neither
        # moved to a device nor mode-flipped (extract MEMO D3).
        return complete_paths
    if plan.device is not None:
        plan.model.to(plan.device)
    n_skip = sum(int(row["n_stimuli"]) for row in completed_rows)
    remaining = _consume_skipped_stimuli(plan.stimuli, n_skip) if n_skip else plan.stimuli
    start_index = len(completed_rows)
    container_paths = [container_path / str(row["file"]) for row in completed_rows]

    run_state: dict[str, Any] = {}
    with _inference_guard(plan.model):
        for offset, batch in enumerate(_batch_iterable(plan, remaining)):
            batch_index = start_index + offset
            batch = _move_nested_to_device(batch, plan.device)
            batch = _correct_batch_positions(plan.model, batch, batch_index, run_state)
            _trace, batch_outputs, layer_views = _tl._extract_layers_with_trace(
                plan.model, batch, plan.layers
            )
            processed = {
                label: (plan.transform(tensor) if plan.transform is not None else tensor)
                .detach()
                .cpu()
                for label, tensor in batch_outputs.items()
            }
            if manifest.get("layers") is None:
                manifest["layers"] = _layer_metadata(layer_views, processed)
            batch_path = container_path / _shard_filename(batch_index)
            _atomic_torch_save(processed, batch_path)
            n_rows = next(iter(processed.values())).shape[0] if processed else 0
            manifest["batches"].append(
                {"index": batch_index, "file": batch_path.name, "n_stimuli": n_rows}
            )
            _atomic_write_json(container_path / MANIFEST_FILENAME, manifest)
            container_paths.append(batch_path)

    manifest["status"] = "complete"
    manifest["stimulus_provenance"]["n_stimuli"] = sum(
        int(row["n_stimuli"]) for row in manifest["batches"]
    )
    _atomic_write_json(container_path / MANIFEST_FILENAME, manifest)
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
    rows = _completed_prefix(manifest, container_path)
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
    "DatasetExtractionResumeError",
    "LoadedExtraction",
    "extract_dataset",
    "load_extraction",
]
