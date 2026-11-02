"""Tensor payloads of the 3.0 saved format: chunked safetensors files and lazy references.

Every tensor is one opaque key (``t<k>``) in one chunk
``blobs/<codec>-<nnnn>.safetensors``; a chunk closes when it would pass
``PAYLOAD_CHUNK_BYTES`` so streaming saves append by opening the next chunk,
and relabeling a row edits its record cell, never a blob. No tensor is written
inline. Design:
``docs/maintainers/saved_format_3_design.md``, section 6. Filled by the writer
stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NoReturn

from .contract import refuse_reader_not_built, refuse_writer_not_built

_LANE_MODULE = "torchlens/_io/format3/payloads.py"

#: Bytes after which the chunk writer closes the current chunk and opens the next.
PAYLOAD_CHUNK_BYTES: Final = 256 << 20


def open_chunk_writer(root: str | Path, *, codec: str) -> NoReturn:
    """Return the chunk writer for one codec family under ``root/blobs``."""
    refuse_writer_not_built("open_chunk_writer", _LANE_MODULE)


def read_tensor(root: str | Path, *, blob: str, key: str) -> NoReturn:
    """Read one tensor by chunk file and key; the eager half of a lazy reference."""
    refuse_reader_not_built("read_tensor", _LANE_MODULE)
