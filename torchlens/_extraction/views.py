"""Thin views over the lazy reader (extract D14's views layer).

These functions consume ONLY the reader's public protocol — metadata,
``iter_batches``, ``rows`` — and are therefore structurally unable to reach
storage internals. They ship in the same release as the reader so the
documented access patterns (the SAE/CLT shard-shuffle pattern, feature
matrices, torch datasets) never route users through eager materialization.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .ragged import RaggedBatch

__tl_layer__ = "L5"

__all__ = ["as_torch_dataset", "feature_matrix", "shuffled_batches"]


def shuffled_batches(
    reader: Any,
    batch_size: int,
    *,
    seed: int = 0,
    keys: list[str] | None = None,
    buffer_shards: int = 1,
    tokens: bool = False,
) -> Any:
    """Yield shuffled training batches without thrashing shard handles.

    The SAE/CLT access pattern: SHARD order is shuffled, then the shuffle
    buffer holds ``buffer_shards`` consecutive shards of that order at once
    and shuffles rows ACROSS them (the cross-shard mixing buffer, R2-08) —
    each shard is still touched exactly once. ``buffer_shards=1`` (the
    default) shuffles within one shard at a time.

    Parameters
    ----------
    reader:
        An open extraction reader.
    batch_size:
        Rows per yielded batch.
    seed:
        Shuffle seed (deterministic across runs).
    keys:
        Optional output-key subset.
    buffer_shards:
        Shards mixed per shuffle buffer (DOCUMENTED-UNSTABLE; R2-08). Peak
        memory is that many shards of the selected keys.
    tokens:
        Yield TOKEN rows instead of stimulus rows (DOCUMENTED-UNSTABLE;
        R2-08): every selected key is read through
        ``reader.token_batch`` and must share one token axis.

    Yields
    ------
    dict[str, torch.Tensor]
        Batches of at most ``batch_size`` rows (each buffer's tail batch
        may run short; rows never cross buffer boundaries). Stimulus-row
        mode yields dense keys only; trimmed keys need ``tokens=True``.
    """

    if batch_size <= 0:
        raise InvalidArgumentError(
            f"shuffled_batches(batch_size={batch_size}) needs a positive batch size.",
            code="extraction_reader_batch_size_invalid",
            remedy="pass batch_size >= 1",
            batch_size=batch_size,
        )
    _validate_buffer_shards(buffer_shards)
    rng = random.Random(seed)
    order = list(range(int(reader.n_shards)))
    rng.shuffle(order)
    for start in range(0, len(order), buffer_shards):
        pooled = _load_buffer(reader, order[start : start + buffer_shards], keys, tokens)
        if not pooled:
            continue
        n_rows = min(int(tensor.shape[0]) for tensor in pooled.values())
        permutation = list(range(n_rows))
        rng.shuffle(permutation)
        for offset in range(0, n_rows, batch_size):
            rows = permutation[offset : offset + batch_size]
            yield {key: tensor[rows] for key, tensor in pooled.items()}


def _validate_buffer_shards(buffer_shards: Any) -> None:
    """Refuse a mixing-buffer size that is not a positive integer.

    Parameters
    ----------
    buffer_shards:
        The requested buffer size in shards.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_reader_buffer_shards_invalid``.
    """

    if isinstance(buffer_shards, int) and not isinstance(buffer_shards, bool) and buffer_shards > 0:
        return
    raise InvalidArgumentError(
        f"shuffled_batches(buffer_shards={buffer_shards!r}) needs a positive integer.",
        code="extraction_reader_buffer_shards_invalid",
        remedy="pass buffer_shards >= 1 (1 shuffles within one shard at a time)",
        buffer_shards=repr(buffer_shards),
    )


def _load_buffer(
    reader: Any, shards: list[int], keys: list[str] | None, tokens: bool
) -> dict[str, torch.Tensor]:
    """Load one shuffle buffer: the listed shards' rows, concatenated.

    Parameters
    ----------
    reader:
        An open extraction reader.
    shards:
        Ledger positions in buffer order.
    keys:
        Optional output-key subset.
    tokens:
        Whether to read token rows.

    Returns
    -------
    dict[str, torch.Tensor]
        ``key -> rows`` over the buffer (empty when no key qualifies).
    """

    parts: list[dict[str, torch.Tensor]] = []
    for position in shards:
        if tokens:
            parts.append(dict(reader.token_batch(position, keys=keys).values))
            continue
        payload = reader.batch(position, keys=keys)
        dense = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        if dense:
            parts.append(dense)
    if len(parts) == 1:
        return parts[0]
    common = [key for key in (parts[0] if parts else {}) if all(key in part for part in parts)]
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in common}


def feature_matrix(reader: Any, key: str, *, max_bytes: int | None = None) -> torch.Tensor:
    """Materialize one dense key as a 2D ``[n_stimuli, features]`` matrix.

    Parameters
    ----------
    reader:
        An open extraction reader.
    key:
        Dense output key.
    max_bytes:
        Optional explicit byte-budget override, forwarded to the reader's
        guarded materialization.

    Returns
    -------
    torch.Tensor
        Row-per-stimulus feature matrix (features flattened).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_reader_ragged_matrix_unsupported`` for trimmed keys —
        a ragged key has no rectangular matrix; ``to_padded`` is the
        explicit densifier.
    """

    materialized = reader.materialize([key], max_bytes=max_bytes)[key]
    if isinstance(materialized, RaggedBatch):
        raise InvalidArgumentError(
            f"Output key {key!r} is stored trimmed (true per-stimulus "
            "extents); a ragged key has no rectangular feature matrix.",
            code="extraction_reader_ragged_matrix_unsupported",
            remedy=(
                "densify explicitly with reader.to_padded(key) and flatten "
                "the result, or pool at extraction time"
            ),
            key=key,
        )
    return materialized.reshape(materialized.shape[0], -1)


def as_torch_dataset(reader: Any, keys: list[str] | None = None) -> Any:
    """Wrap the reader as a ``torch.utils.data.Dataset`` of per-row dicts.

    Parameters
    ----------
    reader:
        An open extraction reader.
    keys:
        Optional output-key subset.

    Returns
    -------
    torch.utils.data.Dataset
        Map-style dataset; ``dataset[i]`` returns the reader's ``row(i)``.
    """

    from torch.utils.data import Dataset

    selected = keys

    class _ExtractionDataset(Dataset):
        """Map-style dataset over one extraction artifact."""

        def __len__(self) -> int:
            """Return the artifact's trusted row count."""

            return int(reader.n_stimuli)

        def __getitem__(self, index: int) -> dict[str, Any]:
            """Return one row across the selected keys.

            Parameters
            ----------
            index:
                Global row index.
            """

            return reader.row(int(index), keys=selected)

    return _ExtractionDataset()
