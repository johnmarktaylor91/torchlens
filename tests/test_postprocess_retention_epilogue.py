"""B1-18: the H2 hot-AST release is a TOTAL postprocess epilogue.

``ast_branches`` keeps a process-wide, 256-file-bounded ``_file_cache`` split
into a cold tier (spans, projected calls, retained source) and a hot tier
(``_HeavyAst``: parsed module AST, parent map, scope nodes) for whole source
files -- including torch-library files. The H2 retention seal drops the hot
tier at the postprocess epilogue so no capture leaves multi-megabyte ASTs
pinned between captures.

That drop used to be a success-path STATEMENT at the end of ``postprocess()``,
so any failure inside steps 0-20 or the core freeze skipped it and the hot tier
survived. Distinct failing files accumulate up to the cache bound, which
resurrects exactly the retention class the seal exists to close. It now runs in
a ``finally``.

Releasing is safe on every exit by design: the cold tier survives, so a later
query for an unprojected scope re-parses from RETAINED source (never disk).
"""

from __future__ import annotations

import pytest

import torchlens._postprocess as pp
from torchlens._postprocess import ast_branches

# Markers are per-test, not module-level: every case here is smoke-fast EXCEPT the
# `-O` gate proof, which spawns a fresh interpreter that pays a full cold torchlens
# import (measured 37s). Marking that one `heavy` keeps the smoke partition honest;
# combining `smoke` with `heavy` would fail tests/test_marker_lint.py.


@pytest.fixture(autouse=True)
def _clean_file_cache():
    """Isolate each test from process-wide AST-cache carryover."""

    ast_branches._file_cache.clear()
    yield
    ast_branches._file_cache.clear()


def test_the_release_is_reached_through_a_finally_not_a_tail_statement() -> None:
    """Source lockstep on the totality itself.

    The runtime tests above would also pass if someone re-added the release as
    a duplicated statement on each exit path -- which is how the zero-layer
    early return came to be missed in the first place. Pin the structure.
    """

    import inspect

    source = inspect.getsource(pp.postprocess)
    assert "finally:" in source
    finally_body = source.split("finally:", 1)[1]
    assert "release_parsed_asts()" in finally_body
    # The body function must NOT carry its own tail release.
    body_source = inspect.getsource(pp._postprocess_body)
    assert "release_parsed_asts()" not in body_source
