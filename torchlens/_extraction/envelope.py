"""The typed batch envelope and the collate door (extract memo D9 + item 5).

Every batch that reaches the model travels as ONE :class:`BatchEnvelope`:
positional args, kwargs, the INPUT-DERIVED row count, the auxiliary
attention mask, and a JSON-portable disclosure that rides the manifest and
the resume signature. Coercion rules are closed: a ``Mapping`` means kwargs,
a ``Tensor`` means one positional arg, a :class:`BatchEnvelope` passes
through, and an ambiguous bare tuple from a USER collate refuses typed
naming the explicit constructor (the engine's own default collation of
tuple-item stimuli keeps its historical positional meaning).

Tokenizer resolution happens ONCE PER RUN, in order: caller-supplied (via
:func:`hf_collate`), attached (``model.tokenizer`` / ``to_tokens``), hub
lookup. Text lists tokenize WITH padding and the FULL mapping is forwarded.
A missing pad token refuses by default; EOS-as-pad is an explicit, disclosed
opt-in recorded as ``pad_token_source``.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .dtype_policy import tensor_payload_bytes

__tl_layer__ = "L5"

__all__ = [
    "BatchEnvelope",
    "coerce_envelope",
    "default_collate",
    "envelope_input_digest",
    "hf_collate",
    "resolve_tokenizer_once",
]


@dataclasses.dataclass(frozen=True)
class BatchEnvelope:
    """One collated batch, typed (extract D9).

    Attributes
    ----------
    args:
        Positional arguments for ``model.forward``.
    kwargs:
        Keyword arguments for ``model.forward``.
    row_count:
        INPUT-derived number of stimulus rows in this batch — the row-truth
        every selected output is validated against (extract D13).
    mask:
        The batch's two-dimensional attention mask when the collator
        supplied one (nonzero = token present); ``None`` otherwise.
    disclosure:
        JSON-portable collate disclosure recorded in the manifest and the
        resume signature (collate kind, tokenizer identity, pad-token
        source, ...).
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    row_count: int
    mask: torch.Tensor | None
    disclosure: dict[str, Any]

    def first_input(self) -> Any:
        """Return the batch's primary input container for legacy paths.

        Returns
        -------
        Any
            The single positional arg when there is exactly one and no
            kwargs, else the kwargs mapping when purely keyword-shaped,
            else the args tuple.
        """

        if not self.kwargs and len(self.args) == 1:
            return self.args[0]
        if not self.args and self.kwargs:
            return dict(self.kwargs)
        return self.args


def _rows_of(value: Any) -> int | None:
    """Read a leading-axis row count off one collated value.

    Parameters
    ----------
    value:
        Candidate batch payload.

    Returns
    -------
    int | None
        ``value.shape[0]`` for tensors with at least one axis, else ``None``.
    """

    if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return int(value.shape[0])
    return None


def _mapping_rows_and_mask(mapping: Mapping[str, Any]) -> tuple[int | None, torch.Tensor | None]:
    """Derive the row count and attention mask from a kwargs mapping.

    Parameters
    ----------
    mapping:
        Kwargs-shaped batch.

    Returns
    -------
    tuple[int | None, torch.Tensor | None]
        Row count (from ``input_ids`` first, else the first tensor) and the
        two-dimensional ``attention_mask`` when present.
    """

    mask = mapping.get("attention_mask")
    if not (isinstance(mask, torch.Tensor) and mask.ndim == 2):
        mask = None
    for key in ("input_ids", "pixel_values"):
        rows = _rows_of(mapping.get(key))
        if rows is not None:
            return rows, mask
    for value in mapping.values():
        rows = _rows_of(value)
        if rows is not None:
            return rows, mask
    return None, mask


def coerce_envelope(
    raw: Any,
    *,
    n_items: int | None,
    source: str,
    batch_index: int,
) -> BatchEnvelope:
    """Coerce a collate result into the ONE typed envelope (extract D9).

    Parameters
    ----------
    raw:
        The collate output: :class:`BatchEnvelope` (passes through),
        ``Mapping`` (kwargs), ``Tensor`` (one positional arg), or — from
        the engine's own default collation only — a tuple/list positional
        container.
    n_items:
        Number of stimulus items collated into this batch, when known; the
        authoritative row count for item-collated batches.
    source:
        ``"default"`` for the engine's own collation, ``"user"`` for a
        ``collate=`` callable's return value.
    batch_index:
        Zero-based batch index for refusal messages.

    Returns
    -------
    BatchEnvelope
        The typed envelope.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_collate_ambiguous`` when a USER collate returns a bare
        tuple/list (positional args or one tuple-valued input? — refused,
        naming the explicit constructor);
        ``extraction_collate_invalid`` when a row count cannot be derived
        from the collate result at all.
    """

    if isinstance(raw, BatchEnvelope):
        if raw.row_count <= 0:
            raise InvalidArgumentError(
                f"collate= returned a BatchEnvelope with row_count="
                f"{raw.row_count} for batch {batch_index}; the row count is "
                "the row-truth every stored shard is validated against, so "
                "it must be a positive integer.",
                code="extraction_collate_invalid",
                remedy="construct the BatchEnvelope with the batch's true row count",
                batch_index=batch_index,
            )
        return raw
    if isinstance(raw, Mapping):
        rows, mask = _mapping_rows_and_mask(raw)
        rows = rows if rows is not None else n_items
        if rows is None:
            raise InvalidArgumentError(
                f"The collated mapping for batch {batch_index} holds no "
                "tensor with a leading batch axis, so its row count cannot "
                "be derived.",
                code="extraction_collate_invalid",
                remedy=(
                    "return a BatchEnvelope with an explicit row_count, or "
                    "include a batched tensor in the mapping"
                ),
                batch_index=batch_index,
            )
        return BatchEnvelope(
            args=(),
            kwargs=dict(raw),
            row_count=rows,
            mask=mask,
            disclosure={"kind": source},
        )
    if isinstance(raw, torch.Tensor):
        rows = _rows_of(raw)
        rows = rows if rows is not None else n_items
        if rows is None:
            raise InvalidArgumentError(
                f"The collated tensor for batch {batch_index} is 0-dimensional; "
                "a batch tensor needs a leading stimulus axis.",
                code="extraction_collate_invalid",
                remedy="return a tensor with a leading batch axis",
                batch_index=batch_index,
            )
        return BatchEnvelope(
            args=(raw,), kwargs={}, row_count=rows, mask=None, disclosure={"kind": source}
        )
    if isinstance(raw, (tuple, list)):
        return _envelope_from_sequence(raw, n_items, source, batch_index)
    raise InvalidArgumentError(
        f"The collate result for batch {batch_index} has unsupported type {type(raw).__name__}.",
        code="extraction_collate_invalid",
        remedy=(
            "return a BatchEnvelope, a Mapping (kwargs), or a Tensor from "
            "collate=; default collation supports tensor/tuple/list/dict items"
        ),
        batch_index=batch_index,
        result_type=type(raw).__name__,
    )


def _envelope_from_sequence(
    raw: tuple[Any, ...] | list[Any], n_items: int | None, source: str, batch_index: int
) -> BatchEnvelope:
    """Coerce a positional-sequence collate result (engine default only).

    Positional masks are located by forward-signature name later, never
    guessed here.

    Parameters
    ----------
    raw:
        The sequence (positional args).
    n_items:
        Item count when known.
    source:
        Collate source label (``"user"`` sequences refuse as ambiguous).
    batch_index:
        Zero-based batch index.

    Returns
    -------
    BatchEnvelope
        The typed envelope.
    """

    if source == "user":
        raise InvalidArgumentError(
            f"collate= returned a bare {type(raw).__name__} for batch "
            f"{batch_index}; a bare sequence is ambiguous (positional "
            "args, or one sequence-valued input?).",
            code="extraction_collate_ambiguous",
            remedy=(
                "return a torchlens.dataset_extraction.BatchEnvelope "
                "(args=..., kwargs=..., row_count=...), a Mapping "
                "(kwargs), or a single Tensor"
            ),
            batch_index=batch_index,
        )
    rows = n_items
    for value in raw:
        if rows is None:
            rows = _rows_of(value)
    if rows is None:
        raise InvalidArgumentError(
            f"The default-collated positional batch {batch_index} holds "
            "no tensor with a leading batch axis, so its row count "
            "cannot be derived.",
            code="extraction_collate_invalid",
            remedy="pass stimuli whose items collate to batched tensors",
            batch_index=batch_index,
        )
    return BatchEnvelope(
        args=tuple(raw), kwargs={}, row_count=rows, mask=None, disclosure={"kind": source}
    )


def resolve_tokenizer_once(model: nn.Module, run_state: dict[str, Any]) -> tuple[Any, str]:
    """Resolve the run's tokenizer ONCE, in the D9 precedence order.

    Order: caller-supplied (recorded by :func:`hf_collate` into
    ``run_state``), attached (``model.to_tokens`` / callable
    ``model.tokenizer``), hub lookup from ``model.config.name_or_path``.
    The resolution is cached in ``run_state`` so every batch of the run
    tokenizes through the same object.

    Parameters
    ----------
    model:
        Model about to consume the batches.
    run_state:
        Mutable per-run dict caching the resolution.

    Returns
    -------
    tuple[Any, str]
        ``(tokenizer, source)`` with source one of ``"caller"`` /
        ``"attached"`` / ``"hub"``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_tokenizer_unresolvable`` when no tokenizer can be
        resolved by any of the three doors.
    """

    cached = run_state.get("tokenizer_resolution")
    if cached is not None:
        return cached
    tokenizer = run_state.get("caller_tokenizer")
    if tokenizer is not None:
        run_state["tokenizer_resolution"] = (tokenizer, "caller")
        return tokenizer, "caller"
    attached = getattr(model, "tokenizer", None)
    if callable(attached):
        run_state["tokenizer_resolution"] = (attached, "attached")
        return attached, "attached"
    name_or_path = getattr(getattr(model, "config", None), "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        except Exception as exc:
            raise InvalidArgumentError(
                f"Text stimuli need a tokenizer, and the hub lookup for "
                f"{name_or_path!r} failed ({exc}).",
                code="extraction_tokenizer_unresolvable",
                remedy=(
                    "pass collate=torchlens.dataset_extraction.hf_collate("
                    "tokenizer=...), or attach one: model.tokenizer = "
                    "AutoTokenizer.from_pretrained(...)"
                ),
                name_or_path=name_or_path,
            ) from exc
        run_state["tokenizer_resolution"] = (tokenizer, "hub")
        return tokenizer, "hub"
    raise InvalidArgumentError(
        "Text stimuli need a tokenizer and none could be resolved: no "
        "caller-supplied tokenizer, no attached model.tokenizer, and the "
        "model config names no checkpoint for a hub lookup.",
        code="extraction_tokenizer_unresolvable",
        remedy=(
            "pass collate=torchlens.dataset_extraction.hf_collate("
            "tokenizer=...), or attach one: model.tokenizer = "
            "AutoTokenizer.from_pretrained(...)"
        ),
        model_type=type(model).__name__,
    )


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Build a JSON-portable identity disclosure for one tokenizer.

    Parameters
    ----------
    tokenizer:
        Resolved tokenizer.

    Returns
    -------
    dict[str, Any]
        Type name plus best-effort checkpoint/vocab facts.
    """

    identity: dict[str, Any] = {"type": type(tokenizer).__qualname__}
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        identity["name_or_path"] = name_or_path
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int):
        identity["vocab_size"] = vocab_size
    padding_side = getattr(tokenizer, "padding_side", None)
    if isinstance(padding_side, str):
        identity["padding_side"] = padding_side
    return identity


def _require_pad_token(tokenizer: Any, pad_token: str | None, disclosure: dict[str, Any]) -> None:
    """Enforce the D9 pad-token rule before a padded batch tokenization.

    Parameters
    ----------
    tokenizer:
        Resolved tokenizer (mutated only under the explicit EOS opt-in).
    pad_token:
        ``None`` (require a configured pad token) or ``"eos"`` (the
        explicit, disclosed opt-in).
    disclosure:
        Mutable disclosure receiving ``pad_token_source``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_pad_token_missing`` when the tokenizer has no pad
        token and the EOS opt-in was not given, or the opt-in names a
        tokenizer without an EOS token.
    """

    if getattr(tokenizer, "pad_token", None) is not None and pad_token is None:
        disclosure["pad_token_source"] = "tokenizer"
        return
    if pad_token == "eos":
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise InvalidArgumentError(
                f"pad_token='eos' was requested but {type(tokenizer).__qualname__} "
                "has no EOS token to borrow.",
                code="extraction_pad_token_missing",
                remedy="configure tokenizer.pad_token explicitly",
            )
        tokenizer.pad_token = eos
        disclosure["pad_token_source"] = "eos_opt_in"
        return
    if pad_token is not None:
        raise InvalidArgumentError(
            f"hf_collate(pad_token={pad_token!r}) is outside the closed "
            "vocabulary; the only opt-in is 'eos'.",
            code="extraction_pad_token_missing",
            remedy="pass pad_token='eos' or configure tokenizer.pad_token",
            pad_token=pad_token,
        )
    if getattr(tokenizer, "pad_token", None) is None:
        raise InvalidArgumentError(
            f"Batched text tokenization needs a pad token and "
            f"{type(tokenizer).__qualname__} has none configured (GPT-2 "
            "class tokenizers ship without one).",
            code="extraction_pad_token_missing",
            remedy=(
                "opt in explicitly with collate=hf_collate(tokenizer, "
                "pad_token='eos'), or set tokenizer.pad_token yourself"
            ),
            tokenizer_type=type(tokenizer).__qualname__,
        )
    disclosure["pad_token_source"] = "tokenizer"


def _tokenize_batch(
    tokenizer: Any,
    items: list[str],
    source: str,
    pad_token: str | None,
    tokenizer_kwargs: Mapping[str, Any] | None,
) -> BatchEnvelope:
    """Tokenize one text batch into a full-mapping envelope (extract D9).

    Parameters
    ----------
    tokenizer:
        Resolved tokenizer.
    items:
        The batch's text items.
    source:
        Tokenizer resolution source for the disclosure.
    pad_token:
        Pad-token policy (``None`` or ``"eos"``).
    tokenizer_kwargs:
        Extra tokenizer call kwargs (truncation, max_length, ...).

    Returns
    -------
    BatchEnvelope
        Kwargs-shaped envelope carrying the FULL tokenizer mapping.
    """

    disclosure: dict[str, Any] = {
        "kind": "hf_tokenizer",
        "tokenizer_source": source,
        "tokenizer": _tokenizer_identity(tokenizer),
    }
    if len(items) > 1 or getattr(tokenizer, "pad_token", None) is None:
        _require_pad_token(tokenizer, pad_token, disclosure)
    else:
        disclosure["pad_token_source"] = "unused_single_row"
    call_kwargs: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
    if tokenizer_kwargs:
        call_kwargs.update(tokenizer_kwargs)
    encoded = tokenizer(list(items), **call_kwargs)
    mapping = dict(encoded)
    mask = mapping.get("attention_mask")
    if not (isinstance(mask, torch.Tensor) and mask.ndim == 2):
        mask = None
    disclosure["tokenizer_padding_side"] = getattr(tokenizer, "padding_side", None)
    return BatchEnvelope(
        args=(),
        kwargs=mapping,
        row_count=len(items),
        mask=mask,
        disclosure=disclosure,
    )


def hf_collate(
    tokenizer: Any = None,
    *,
    pad_token: str | None = None,
    **tokenizer_kwargs: Any,
) -> Callable[[list[Any], nn.Module, dict[str, Any]], BatchEnvelope]:
    """Build the HF text collate helper (extract D9; DOCUMENTED-UNSTABLE).

    Parameters
    ----------
    tokenizer:
        Caller-supplied tokenizer (first in the resolution order); ``None``
        defers to the attached-then-hub doors.
    pad_token:
        ``None`` (a missing pad token refuses typed) or ``"eos"`` — the
        explicit, disclosed opt-in recorded as ``pad_token_source``.
    **tokenizer_kwargs:
        Extra tokenizer call kwargs (``truncation=``, ``max_length=``, ...).

    Returns
    -------
    Callable
        An engine collate callable: ``(items, model, run_state) ->
        BatchEnvelope``.
    """

    def _collate(items: list[Any], model: nn.Module, run_state: dict[str, Any]) -> BatchEnvelope:
        """Tokenize one batch of text items through the run's one tokenizer."""

        if tokenizer is not None:
            run_state.setdefault("caller_tokenizer", tokenizer)
        resolved, source = resolve_tokenizer_once(model, run_state)
        texts = [str(item) for item in items]
        return _tokenize_batch(resolved, texts, source, pad_token, tokenizer_kwargs)

    _collate.__tl_engine_collate__ = True  # type: ignore[attr-defined]
    _collate.__tl_disclosure__ = {  # type: ignore[attr-defined]
        "kind": "hf_tokenizer",
        "pad_token": pad_token,
        "tokenizer_supplied": tokenizer is not None,
        "tokenizer_kwargs": sorted(tokenizer_kwargs),
    }
    return _collate


def _legacy_collate_items(items: list[Any]) -> Any:
    """Collate a list of non-text stimulus items (the historical rules).

    Parameters
    ----------
    items:
        Stimulus items accumulated for one batch.

    Returns
    -------
    Any
        Batched tensor or nested container of batched tensors.
    """

    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    if isinstance(first, tuple):
        return tuple(
            _legacy_collate_items([item[index] for item in items]) for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _legacy_collate_items([item[index] for item in items]) for index in range(len(first))
        ]
    if isinstance(first, dict):
        return {key: _legacy_collate_items([item[key] for item in items]) for key in first}
    return items


def default_collate(items: list[Any], model: nn.Module, run_state: dict[str, Any]) -> BatchEnvelope:
    """The engine's default collation (extract D9 defaults).

    Tensor items stack; tuple/list/dict items collate recursively with the
    historical positional/kwargs meaning; STRING items route through the
    once-per-run tokenizer with padding and the full mapping forwarded (the
    repair for "auto-tokenize breaks on string batches > 1").

    Parameters
    ----------
    items:
        Stimulus items accumulated for one batch.
    model:
        Model about to consume the batch.
    run_state:
        Mutable per-run state (tokenizer cache, disclosures).

    Returns
    -------
    BatchEnvelope
        The typed envelope.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    if isinstance(items[0], str):
        tokenizer, source = resolve_tokenizer_once(model, run_state)
        return _tokenize_batch(tokenizer, [str(item) for item in items], source, None, None)
    collated = _legacy_collate_items(items)
    return coerce_envelope(collated, n_items=len(items), source="default", batch_index=-1)


def _digest_value(hasher: Any, value: Any) -> None:
    """Fold one envelope value into an input digest.

    Parameters
    ----------
    hasher:
        Active hash object.
    value:
        Tensor or nested container.
    """

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        hasher.update(f"tensor|{tuple(detached.shape)}|{detached.dtype}".encode())
        hasher.update(tensor_payload_bytes(detached))
        return
    if isinstance(value, Mapping):
        hasher.update(f"map|{len(value)}".encode())
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            hasher.update(str(key).encode())
            _digest_value(hasher, item)
        return
    if isinstance(value, (tuple, list)):
        hasher.update(f"seq|{len(value)}".encode())
        for item in value:
            _digest_value(hasher, item)
        return
    hasher.update(f"scalar|{value!r}".encode())


def envelope_input_digest(envelope: BatchEnvelope) -> str:
    """Digest one batch's replayable input for its ledger row (item 16).

    Recorded per shard at write time and recomputed during resume's
    EXISTING skipped-prefix replay — zero extra forward passes — turning
    "iterable stimuli are unverifiable" into "verified across the entire
    replayed prefix".

    Parameters
    ----------
    envelope:
        The collated batch envelope.

    Returns
    -------
    str
        ``"sha256:..."`` over the envelope's args/kwargs structure and
        tensor bytes.
    """

    hasher = hashlib.sha256()
    _digest_value(hasher, envelope.args)
    _digest_value(hasher, dict(envelope.kwargs))
    return f"sha256:{hasher.hexdigest()}"
