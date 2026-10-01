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
) -> Any:
    """Yield shuffled training batches without thrashing shard handles.

    The SAE/CLT access pattern: SHARD order is shuffled, then rows shuffle
    WITHIN each shard — an approximate global shuffle that touches each
    shard exactly once instead of thrashing thousands of handles.

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

    Yields
    ------
    dict[str, torch.Tensor]
        Batches of at most ``batch_size`` rows (per-shard tail batches may
        run short; rows never cross shard boundaries).
    """

    if batch_size <= 0:
        raise InvalidArgumentError(
            f"shuffled_batches(batch_size={batch_size}) needs a positive batch size.",
            code="extraction_reader_batch_size_invalid",
            remedy="pass batch_size >= 1",
            batch_size=batch_size,
        )
    rng = random.Random(seed)
    order = list(range(int(reader.n_shards)))
    rng.shuffle(order)
    for position in order:
        payload = reader.batch(position, keys=keys)
        dense = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        if not dense:
            continue
        n_rows = min(int(tensor.shape[0]) for tensor in dense.values())
        permutation = list(range(n_rows))
        rng.shuffle(permutation)
        for start in range(0, n_rows, batch_size):
            rows = permutation[start : start + batch_size]
            yield {key: tensor[rows] for key, tensor in dense.items()}


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
