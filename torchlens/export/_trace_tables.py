"""Transitional bridge: the trace tables live in :mod:`torchlens.stats._trace_tables`.

The record door ``_data_classes._doors.export()`` still imports this path until the
record lanes re-point it at ``_doors.stats()``; this module is deleted in that change.
"""

from __future__ import annotations

from ..stats._trace_tables import (
    decode_output as decode_output,
    output_table as output_table,
    to_pandas as to_pandas,
)
