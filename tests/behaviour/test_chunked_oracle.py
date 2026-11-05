"""The chunked-capture oracle compares against forwards on the views the product chunks on.

``chunk_size=`` capture runs each chunk's forward on an uncloned row view of the
caller's batch (``value[start:end]``). The oracle's reference forwards must see
the same thing: views at the same row offsets into one fresh copy of the batch,
never cloned pieces. A clone starts at a fresh allocation, and on some CPUs
(AMD AVX2 with MKL) a matmul over a row-offset view rounds 1 ulp apart from the
same rows at an aligned start, so a cloned reference reds a faithful product.
Exact equality stays; these tests also show the oracle still fails a product
that feeds a chunk from the wrong rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.behaviour._fill_io import _check_chunked_capture, _row_wise_pieces
from tests.behaviour.cases import Case
from tests.behaviour.harness import Cell, OracleFailure, case_facts
from torchlens._capture.prepare import door
from torchlens._capture.prepare.chunked import _replace_paths

_ROWS = 5
_WIDTH = 3
_SIZE = (_ROWS + 1) // 2


class _InputRecorder(nn.Module):
    """A row-wise model that records how each forward's input sits in memory."""

    def __init__(self) -> None:
        """Start with no recorded forwards."""

        super().__init__()
        self.seen: list[tuple[int, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record ``(storage offset, storage address)`` and double the input."""

        self.seen.append((x.storage_offset(), x.untyped_storage().data_ptr()))
        return x * 2


def _build_linear() -> tuple[nn.Module, Any]:
    """A small row-wise case: one linear layer over a five-row batch."""

    torch.manual_seed(0)
    return nn.Linear(_WIDTH, 4).eval(), torch.randn(_ROWS, _WIDTH)


@pytest.fixture
def linear_cell(tmp_path: Path) -> Iterator[Cell]:
    """A real behaviour-net cell for the linear case, with no features."""

    case = Case("chunked-oracle-linear", _build_linear, frozenset(), 0.1, "toy")
    model, inputs = case.build()
    yield Cell(case, (), tmp_path, model, inputs, case_facts(case, model, inputs))


def test_reference_forwards_run_on_uncloned_row_views_at_the_chunk_offsets() -> None:
    """Both reference forwards read one fresh copy of the batch, at offsets 0 and one chunk."""

    recorder = _InputRecorder()
    stimuli = torch.randn(_ROWS, _WIDTH)
    pieces = _row_wise_pieces(SimpleNamespace(model=recorder), stimuli, _SIZE)
    assert pieces is not None
    assert [offset for offset, _ in recorder.seen] == [0, _SIZE * _WIDTH]
    assert len({address for _, address in recorder.seen}) == 1
    assert recorder.seen[0][1] != stimuli.untyped_storage().data_ptr()
    assert torch.equal(torch.cat(pieces, dim=0), stimuli * 2)


def test_a_faithful_chunked_capture_passes_the_oracle(linear_cell: Cell) -> None:
    """The unmodified product passes the chunked-capture check."""

    _check_chunked_capture(linear_cell, linear_cell.facts.plain)


def _chunks_from_shifted_rows(input_args: Any, plan: Any) -> list[Any]:
    """A planted product bug: every chunk after the first starts one row early."""

    chunks = []
    for start in range(0, plan.total_size, plan.chunk_size):
        end = min(start + plan.chunk_size, plan.total_size)
        shift = 1 if start else 0
        chunks.append(_replace_paths(input_args, plan.paths, start - shift, end - shift))
    return chunks


def test_a_chunk_fed_from_the_wrong_offset_fails_the_oracle(
    linear_cell: Cell, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A product that slices a chunk one row early reds the chunked capture output."""

    monkeypatch.setattr(door, "iter_chunked_inputs", _chunks_from_shifted_rows)
    with pytest.raises(OracleFailure, match="chunked capture output"):
        _check_chunked_capture(linear_cell, linear_cell.facts.plain)
