"""The token SAE store (spellings DOCUMENTED-UNSTABLE).

SAE and transcoder training reads activations one TOKEN per row, not one
stimulus per row. Three pieces serve that pattern over the existing
artifact, with no new shard format:

- ``extract_dataset(..., store_input_ids=True)`` stores each batch's token
  ids as the ordinary ``input_ids`` key, trimmed by the same mask as the
  token-axis activations, so stored id ``j`` of stimulus ``i`` names the
  token whose activation sits at the same position;
- :meth:`ExtractionReader.token_batch` / :meth:`ExtractionReader.token_rows`
  flatten stored keys to token rows (true extents only: a trimmed key's
  packed values ARE its token rows; a dense key qualifies only when its
  shard ledgered a mask with no padded position at the key's width) with
  the stimulus row and position of every token;
- ``shuffled_batches(..., buffer_shards=k, tokens=True)`` mixes rows across
  ``k`` shards at once (the cross-shard mixing buffer), so a corpus written
  in semantic order does not yield locally correlated training batches.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NoReturn

import torch

from .._errors import InvalidArgumentError
from .ragged import RaggedBatch

__tl_layer__ = "L5"

__all__ = [
    "INPUT_IDS_KEY",
    "TokenRows",
    "concat_token_rows",
    "envelope_input_ids",
    "refuse_input_ids_collision",
    "shard_token_rows",
    "validate_store_input_ids",
]

#: The reserved output key under which ``store_input_ids=True`` stores ids.
INPUT_IDS_KEY = "input_ids"


@dataclasses.dataclass(frozen=True)
class TokenRows:
    """Stored keys flattened to one row per token (DOCUMENTED-UNSTABLE).

    Attributes
    ----------
    values:
        ``key -> [n_tokens, *feature_shape]`` per selected key, every key
        aligned token for token.
    stimulus_row:
        ``int64 [n_tokens]``: the global stimulus row of each token.
    position:
        ``int64 [n_tokens]``: each token's index within its stimulus's
        true (unpadded) extent.
    """

    values: dict[str, torch.Tensor]
    stimulus_row: torch.Tensor
    position: torch.Tensor

    @property
    def n_tokens(self) -> int:
        """Return the number of token rows.

        Returns
        -------
        int
            ``len(stimulus_row)``.
        """

        return int(self.stimulus_row.shape[0])


def envelope_input_ids(envelope: Any, batch_index: int) -> torch.Tensor:
    """Return the batch envelope's token ids for storage.

    Parameters
    ----------
    envelope:
        The batch's :class:`~torchlens.dataset_extraction.BatchEnvelope`.
    batch_index:
        Zero-based batch index (refusal text).

    Returns
    -------
    torch.Tensor
        ``[rows, width]`` integer ids: the ``input_ids`` kwarg, else a sole
        integer-dtype 2-D positional argument.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_store_input_ids_absent`` when the batch carries no
        token ids to store.
    """

    candidate = envelope.kwargs.get(INPUT_IDS_KEY)
    if candidate is None and len(envelope.args) == 1:
        candidate = envelope.args[0]
    if (
        isinstance(candidate, torch.Tensor)
        and candidate.ndim == 2
        and not candidate.is_floating_point()
        and not candidate.is_complex()
        and candidate.dtype != torch.bool
    ):
        return candidate
    raise InvalidArgumentError(
        f"store_input_ids=True, but batch {batch_index} carries no token ids: "
        "neither an input_ids keyword nor a sole 2-D integer positional "
        "argument reaches the model.",
        code="extraction_store_input_ids_absent",
        remedy="collate text batches with input_ids (hf_collate does), or drop store_input_ids=",
        batch_index=batch_index,
    )


def validate_store_input_ids(value: Any) -> bool:
    """Validate the ``store_input_ids=`` flag.

    Parameters
    ----------
    value:
        The caller's value.

    Returns
    -------
    bool
        The flag.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_store_input_ids_invalid`` for a non-bool.
    """

    if isinstance(value, bool):
        return value
    raise InvalidArgumentError(
        f"store_input_ids= takes True or False; got {type(value).__name__}.",
        code="extraction_store_input_ids_invalid",
        remedy="pass store_input_ids=True to store token ids, or omit it",
        value_type=type(value).__name__,
    )


def refuse_input_ids_collision(keys: Any) -> None:
    """Refuse an output key that would shadow the stored ``input_ids``.

    Parameters
    ----------
    keys:
        The batch's activation output keys.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_store_input_ids_key_collision``.
    """

    if INPUT_IDS_KEY not in keys:
        return
    raise InvalidArgumentError(
        f"store_input_ids=True stores token ids under the output key "
        f"{INPUT_IDS_KEY!r}, which an activation key already uses.",
        code="extraction_store_input_ids_key_collision",
        remedy="rename that layer key in a {key: layer} mapping",
        key=INPUT_IDS_KEY,
    )


def _refuse_token_key(key: str, reason: str, **fields: Any) -> NoReturn:
    """Refuse one key that cannot be read as token rows.

    Parameters
    ----------
    key:
        Output key.
    reason:
        Why the key has no honest token-row reading.
    **fields:
        Structured refusal context.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_token_rows_unsupported_key``.
    """

    raise InvalidArgumentError(
        f"Output key {key!r} cannot be read as token rows: {reason}.",
        code="extraction_token_rows_unsupported_key",
        remedy=(
            "extract token-axis sites with ragged='trim' (true extents), or "
            "request only token-axis keys"
        ),
        key=key,
        **fields,
    )


def _dense_extents(key: str, tensor: torch.Tensor, geometry: Mapping[str, Any] | None) -> list[int]:
    """Return each row's token count for one DENSE key, refusing pads.

    A dense key is read as token rows only when the shard ledgered its mask
    geometry and every row is unpadded at the key's width: then axis 1 is
    provably the token axis and every stored position is a token.

    Parameters
    ----------
    key:
        Output key.
    tensor:
        ``[rows, width, *feature]`` stored payload.
    geometry:
        The shard's ledgered mask geometry, or ``None``.

    Returns
    -------
    list[int]
        ``width`` for every row.
    """

    if geometry is None:
        _refuse_token_key(
            key,
            "its shard recorded no attention mask, so which stored axis holds "
            "tokens (and which positions are padding) is unknown",
        )
    if tensor.ndim < 2:
        _refuse_token_key(key, "it has no token axis (one value per stimulus)")
    width = int(tensor.shape[1])
    starts = [int(start) for start in geometry.get("starts") or []]
    extents = [int(extent) for extent in geometry.get("extents") or []]
    if any(start != 0 for start in starts) or any(extent != width for extent in extents):
        _refuse_token_key(
            key,
            f"it is stored dense with axis 1 of width {width}, which is not "
            f"this shard's unpadded token axis (true extents {extents})",
            width=width,
        )
    return [width] * int(tensor.shape[0])


def _key_rows(
    key: str, value: Any, geometry: Mapping[str, Any] | None
) -> tuple[torch.Tensor, list[int]]:
    """Flatten one key's shard payload to token rows plus per-row counts.

    Parameters
    ----------
    key:
        Output key.
    value:
        Dense tensor or trimmed :class:`RaggedBatch`.
    geometry:
        The shard's ledgered mask geometry, or ``None``.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        ``[n_tokens, *feature]`` values and each stimulus row's token count.
    """

    if isinstance(value, RaggedBatch):
        offsets = [int(offset) for offset in value.offsets.tolist()]
        return value.values, [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    extents = _dense_extents(key, value, geometry)
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]), extents


def shard_token_rows(
    payload: Mapping[str, Any], geometry: Mapping[str, Any] | None, row_start: int
) -> TokenRows:
    """Flatten one shard's payload to aligned token rows.

    Parameters
    ----------
    payload:
        ``key -> tensor | RaggedBatch`` for the selected keys.
    geometry:
        The shard's ledgered mask geometry (``row_geometry``), or ``None``.
    row_start:
        Global row index of the shard's first stimulus.

    Returns
    -------
    TokenRows
        Every selected key's token rows with their stimulus rows and
        positions.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_token_rows_unsupported_key`` for a key with no honest
        token reading; ``extraction_token_rows_misaligned`` when two keys
        disagree on a stimulus's token count.
    """

    values: dict[str, torch.Tensor] = {}
    reference: tuple[str, list[int]] | None = None
    for key, value in payload.items():
        rows, extents = _key_rows(key, value, geometry)
        if reference is not None and extents != reference[1]:
            raise InvalidArgumentError(
                f"Output keys {reference[0]!r} and {key!r} disagree on the "
                "token count of at least one stimulus, so their token rows "
                "cannot be aligned.",
                code="extraction_token_rows_misaligned",
                remedy="read token rows for keys that share one token axis",
                keys=[reference[0], key],
            )
        reference = reference or (key, extents)
        values[key] = rows
    extents = reference[1] if reference is not None else []
    stimulus_row = torch.repeat_interleave(
        torch.arange(row_start, row_start + len(extents), dtype=torch.int64),
        torch.tensor(extents, dtype=torch.int64),
    )
    position = torch.cat(
        [torch.arange(extent, dtype=torch.int64) for extent in extents]
        or [torch.zeros(0, dtype=torch.int64)]
    )
    return TokenRows(values=values, stimulus_row=stimulus_row, position=position)


def concat_token_rows(parts: list[TokenRows]) -> TokenRows:
    """Concatenate per-shard token rows in order.

    Parameters
    ----------
    parts:
        Per-shard :class:`TokenRows` over the same keys.

    Returns
    -------
    TokenRows
        One carrier over every part.
    """

    if not parts:
        empty = torch.zeros(0, dtype=torch.int64)
        return TokenRows(values={}, stimulus_row=empty, position=empty)
    keys = list(parts[0].values)
    return TokenRows(
        values={key: torch.cat([part.values[key] for part in parts], dim=0) for key in keys},
        stimulus_row=torch.cat([part.stimulus_row for part in parts]),
        position=torch.cat([part.position for part in parts]),
    )
