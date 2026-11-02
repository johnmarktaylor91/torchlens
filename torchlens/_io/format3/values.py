"""The typed-value codec of the 3.0 saved format: Python values to tagged JSON and back.

Both directions live in one module so the closed tag vocabulary
(``contract.TYPED_VALUE_TAGS`` plus ``component``) stays symmetric. Decoding
never evaluates, unpickles or imports by name: dtypes, classes and enums
resolve through tables built at import time. A value outside the vocabulary
is written as an ``unencodable`` disclosure. Design:
``docs/maintainers/saved_format_3_design.md``, sections 4 and 5. Filled by the
writer stream.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .contract import refuse_reader_not_built, refuse_writer_not_built

_LANE_MODULE = "torchlens/_io/format3/values.py"


def encode_value(value: Any, *, at: str) -> NoReturn:
    """Encode one Python value as JSON-ready data, tagging what JSON cannot carry.

    Parameters
    ----------
    value:
        The cell value.
    at:
        The ``record[row].column`` path, stamped into any disclosure.
    """
    refuse_writer_not_built("encode_value", _LANE_MODULE)


def decode_value(data: Any, *, schema: Any, at: str) -> NoReturn:
    """Decode one JSON cell against its schema node; refuses anything off-vocabulary."""
    refuse_reader_not_built("decode_value", _LANE_MODULE)
