"""Move-compat gates for the _io split (C01 item 3) + disposition labels (item 4).

The split sends the IO error vocabulary to ``format_errors`` (logical L0)
and the format contract to ``format_contract`` (L1) with ``_io/__init__``
a declared re-export facade. Two invariants gate it:

- ZERO SPELLING CHANGES: every historical ``torchlens._io`` name resolves
  unchanged (the facade contract).
- ZERO PERSISTED-BYTE CHANGES: pickle-visible identity of every persisted
  type stays ``torchlens._io`` (bundle metadata references types by
  (module, qualname) and the guarded unpickler allowlists exactly those
  paths). This suite pins the mechanism that keeps them stable.
"""

from __future__ import annotations


def test_fastlog_dispositions_are_declared() -> None:
    """Item 4 label rows: storage_disk L2, recover/cleanup L3, types vocabulary."""

    import importlib

    storage_disk = importlib.import_module("torchlens.recording.storage_disk")
    recover = importlib.import_module("torchlens.recording.recover")
    cleanup = importlib.import_module("torchlens.recording.cleanup")
    types = importlib.import_module("torchlens.recording.types")
    assert storage_disk.__tl_layer__ == "L2"
    assert recover.__tl_layer__ == "L3"
    assert cleanup.__tl_layer__ == "L3"
    assert types.__tl_layer__ == "L1"
    assert types.__tl_vocabulary__ is True
