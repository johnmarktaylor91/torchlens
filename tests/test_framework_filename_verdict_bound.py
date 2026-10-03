"""grind-r3 T-CACHES: the framework-filename verdict cache must be bounded.

``_FRAMEWORK_FILENAME_VERDICTS`` is a process-global keyed by ``co_filename``.
Generated-code, notebook, plugin, and long-running service workloads mint
fresh filenames without limit (grind-r2 b4-sol R39: 1,000 distinct compiled
filenames grew the cache to exactly 1,000 entries), so the cache FIFO-evicts
at a cap instead of growing for the process lifetime.
"""

from __future__ import annotations

import pytest

from torchlens.backends.torch import completeness_witness as witness_module


def _call_dispatch_callsite_from(filename: str) -> None:
    """Invoke ``_dispatch_callsite`` with ``filename`` as the walked frame."""

    code = compile("probe()", filename, "exec")
    exec(code, {"probe": lambda: witness_module._dispatch_callsite()})  # noqa: S102 - test-owned source


def test_framework_filename_verdicts_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct generated filenames beyond the cap evict FIFO, never grow."""

    cap = 64
    monkeypatch.setattr(witness_module, "_FRAMEWORK_FILENAME_VERDICTS_MAX_ENTRIES", cap)
    snapshot = dict(witness_module._FRAMEWORK_FILENAME_VERDICTS)
    witness_module._FRAMEWORK_FILENAME_VERDICTS.clear()
    try:
        for index in range(cap + 200):
            _call_dispatch_callsite_from(f"<tl-verdict-bound-{index}>")
        assert len(witness_module._FRAMEWORK_FILENAME_VERDICTS) <= cap
        # The newest verdicts survive; the oldest were evicted.
        assert f"<tl-verdict-bound-{cap + 199}>" in witness_module._FRAMEWORK_FILENAME_VERDICTS
        assert "<tl-verdict-bound-0>" not in witness_module._FRAMEWORK_FILENAME_VERDICTS
    finally:
        witness_module._FRAMEWORK_FILENAME_VERDICTS.clear()
        witness_module._FRAMEWORK_FILENAME_VERDICTS.update(snapshot)


def test_framework_filename_verdicts_still_memoize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated filenames re-use their verdict rather than re-resolving."""

    snapshot = dict(witness_module._FRAMEWORK_FILENAME_VERDICTS)
    witness_module._FRAMEWORK_FILENAME_VERDICTS.clear()
    try:
        _call_dispatch_callsite_from("<tl-verdict-memo>")
        _call_dispatch_callsite_from("<tl-verdict-memo>")
        assert (
            sum(
                1
                for key in witness_module._FRAMEWORK_FILENAME_VERDICTS
                if key == "<tl-verdict-memo>"
            )
            == 1
        )
    finally:
        witness_module._FRAMEWORK_FILENAME_VERDICTS.clear()
        witness_module._FRAMEWORK_FILENAME_VERDICTS.update(snapshot)
