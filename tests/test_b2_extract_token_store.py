"""The token SAE store (R2-08, lane EXTRACT).

A small token model's activations are extracted with ``ragged="trim"`` and
``store_input_ids=True``, then read back one row per TOKEN: every token row
equals the stored per-stimulus row at the same position, the stored ids are
the stimulus's true ids (padding never stored), and the cross-shard mixing
buffer yields exactly the stored token multiset while mixing shards.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._errors import InvalidArgumentError
from torchlens._extraction.ragged import RaggedBatch
from torchlens._extraction.token_store import shard_token_rows
from torchlens.dataset_extraction import (
    TokenRows,
    extract_dataset,
    open_extraction,
    shuffled_batches,
)


class _TokenModel(nn.Module):
    """Toy token model taking the HF-shaped kwargs envelope."""

    def __init__(self) -> None:
        """Build the embedding + projection stack deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(100, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        """Embed, offset by positions, and project."""

        hidden = self.emb(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        hidden = hidden + position_ids.unsqueeze(-1).float() * 0.01
        return self.proj(torch.relu(hidden)) * 1.0


#: Each text is its own token ids; ids are unique per stimulus, so a token
#: row's id names the stimulus (and therefore the shard) it came from.
_TEXTS = [
    "1 2 3",
    "4",
    "5 6 7 8 9",
    "10 11",
    "12 13 14",
    "15 16",
    "17",
    "18 19 20 21",
]


def _collate(items, *, left: bool = False):
    """Collate id-texts into an HF-shaped padded mapping batch."""

    rows = [[int(word) for word in text.split()] for text in items]
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        span = slice(width - len(row), width) if left else slice(0, len(row))
        ids[i, span] = torch.tensor(row)
        mask[i, span] = 1
    return {"input_ids": ids, "attention_mask": mask}


def _extract(path: Path, *, collate=_collate, **kwargs) -> None:
    """Run the trimmed token extraction with stored ids."""

    extract_dataset(
        _TokenModel().eval(),
        list(_TEXTS),
        {"h": "proj"},
        batch_size=2,
        output_dir=path,
        collate=collate,
        ragged="trim",
        store_input_ids=True,
        progress=False,
        **kwargs,
    )


def test_token_rows_read_back_equal_per_token(tmp_path: Path) -> None:
    """Every token row equals the stored row at its position; ids are true."""

    _extract(tmp_path)
    reader = open_extraction(tmp_path)
    tokens = reader.token_rows()
    assert isinstance(tokens, TokenRows)
    true_ids = [int(word) for text in _TEXTS for word in text.split()]
    assert tokens.n_tokens == len(true_ids)
    assert tokens.values["input_ids"].tolist() == true_ids
    model = _TokenModel().eval()
    for index, text in enumerate(_TEXTS):
        rows = tokens.stimulus_row == index
        stored = reader.row(index)
        assert torch.equal(tokens.values["h"][rows], stored["h"])
        assert torch.equal(tokens.values["input_ids"][rows], stored["input_ids"])
        assert tokens.position[rows].tolist() == list(range(len(text.split())))
        with torch.no_grad():
            solo = model(_collate([text])["input_ids"])[0]
        assert torch.allclose(tokens.values["h"][rows], solo, atol=1e-5)
    manifest = reader.manifest
    assert manifest["layers"]["input_ids"]["source"] == "stimulus_input_ids"
    assert manifest["layers"]["input_ids"]["layout"] == "trimmed"


def test_left_padding_never_stores_pad_ids(tmp_path: Path) -> None:
    """Left-padded batches trim ids by (start, extent): no pad id is stored."""

    with pytest.warns(UserWarning, match="derived position_ids"):
        _extract(tmp_path, collate=lambda items: _collate(items, left=True))
    tokens = open_extraction(tmp_path).token_rows(["input_ids"])
    assert tokens.values["input_ids"].tolist() == [
        int(word) for text in _TEXTS for word in text.split()
    ]


def _row_counter(batches) -> Counter:
    """Count (id, activation) token rows across yielded batches."""

    counter: Counter = Counter()
    for batch in batches:
        for token_id, row in zip(batch["input_ids"].tolist(), batch["h"].tolist(), strict=True):
            counter[(token_id, tuple(row))] += 1
    return counter


def _shard_of(token_id: int) -> int:
    """Map a token id to the shard (batch of two stimuli) that stored it."""

    for index, text in enumerate(_TEXTS):
        if token_id in [int(word) for word in text.split()]:
            return index // 2
    raise AssertionError(token_id)


def test_mixing_buffer_yields_the_token_multiset_across_shards(tmp_path: Path) -> None:
    """buffer_shards=k mixes shards; every token row is yielded exactly once."""

    _extract(tmp_path)
    reader = open_extraction(tmp_path)
    tokens = reader.token_rows()
    expected = _row_counter([tokens.values])
    within = list(shuffled_batches(reader, 3, seed=3, tokens=True))
    mixed = list(shuffled_batches(reader, 3, seed=3, tokens=True, buffer_shards=4))
    assert _row_counter(within) == expected
    assert _row_counter(mixed) == expected
    assert all(len({_shard_of(i) for i in batch["input_ids"].tolist()}) == 1 for batch in within)
    assert any(len({_shard_of(i) for i in batch["input_ids"].tolist()}) > 1 for batch in mixed)


def _reference_shuffled(reader, batch_size: int, seed: int):
    """The pre-R2-08 within-shard shuffle, verbatim, as the oracle."""

    rng = random.Random(seed)
    order = list(range(int(reader.n_shards)))
    rng.shuffle(order)
    for position in order:
        payload = reader.batch(position)
        dense = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        if not dense:
            continue
        n_rows = min(int(tensor.shape[0]) for tensor in dense.values())
        permutation = list(range(n_rows))
        rng.shuffle(permutation)
        for start in range(0, n_rows, batch_size):
            rows = permutation[start : start + batch_size]
            yield {key: tensor[rows] for key, tensor in dense.items()}


class _Linear(nn.Module):
    """One affine site on dense float stimuli (no token ids)."""

    def __init__(self) -> None:
        """Build the site deterministically."""

        super().__init__()
        torch.manual_seed(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the site; the mul keeps fc distinct from the output."""

        return self.fc(x) * 1.0


def test_default_shuffle_is_unchanged(tmp_path: Path) -> None:
    """buffer_shards=1, tokens=False reproduces the shipped shuffle exactly."""

    extract_dataset(
        _Linear().eval(),
        torch.randn(11, 4, generator=torch.Generator().manual_seed(5)),
        {"y": "fc"},
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
    )
    reader = open_extraction(tmp_path)
    got = list(shuffled_batches(reader, 2, seed=9))
    want = list(_reference_shuffled(reader, 2, seed=9))
    assert len(got) == len(want)
    for left, right in zip(got, want, strict=True):
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_dense_padded_and_pooled_keys_refuse_token_rows(tmp_path: Path) -> None:
    """A dense key stored at padded width, or one with no token axis, refuses."""

    texts = ["1 2 3", "4 5", "6 7 8", "9"]
    extract_dataset(
        _TokenModel().eval(),
        texts,
        {"h": "proj", "pooled": "proj"},
        batch_size=2,
        output_dir=tmp_path,
        collate=lambda items: {
            key: torch.nn.functional.pad(value, (0, 3 - value.shape[1]))
            for key, value in _collate(items).items()
        },
        pool={"pooled": "token_mean"},
        progress=False,
    )
    reader = open_extraction(tmp_path)
    for key in ("h", "pooled"):
        with pytest.raises(InvalidArgumentError) as excinfo:
            reader.token_rows([key])
        assert excinfo.value.fields["code"] == "extraction_token_rows_unsupported_key"


def test_misaligned_keys_refuse() -> None:
    """Two keys disagreeing on a stimulus's token count cannot align."""

    left = RaggedBatch(torch.zeros(3, 2), torch.tensor([0, 1, 3]), ((1, 2), (2, 2)))
    right = RaggedBatch(torch.zeros(3, 2), torch.tensor([0, 2, 3]), ((2, 2), (1, 2)))
    with pytest.raises(InvalidArgumentError) as excinfo:
        shard_token_rows({"a": left, "b": right}, None, 0)
    assert excinfo.value.fields["code"] == "extraction_token_rows_misaligned"


def test_store_input_ids_refusals(tmp_path: Path) -> None:
    """Bad flag values, key collisions, and id-less batches refuse typed."""

    model = _TokenModel().eval()
    cases = [
        ({"layers": {"h": "proj"}, "store_input_ids": 1}, "extraction_store_input_ids_invalid"),
        (
            {"layers": {"input_ids": "proj"}, "store_input_ids": True},
            "extraction_store_input_ids_key_collision",
        ),
    ]
    for kwargs, code in cases:
        with pytest.raises(InvalidArgumentError) as excinfo:
            extract_dataset(
                model, list(_TEXTS), collate=_collate, batch_size=2, progress=False, **kwargs
            )
        assert excinfo.value.fields["code"] == code
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _Linear().eval(),
            torch.randn(4, 4),
            {"y": "fc"},
            batch_size=2,
            progress=False,
            store_input_ids=True,
        )
    assert excinfo.value.fields["code"] == "extraction_store_input_ids_absent"
    reader_dir = tmp_path / "artifact"
    _extract(reader_dir)
    with pytest.raises(InvalidArgumentError) as excinfo:
        list(shuffled_batches(open_extraction(reader_dir), 2, buffer_shards=0))
    assert excinfo.value.fields["code"] == "extraction_reader_buffer_shards_invalid"
