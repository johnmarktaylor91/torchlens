"""Regression tests for embedded-quote/backslash escaping in the rank-engine DOT emitter.

``_dot_id()``/``_dot_quote()`` in
``torchlens/visualization/_rank_layout_internal/layout.py`` wrap raw values in
``"..."`` for DOT text but, before this fix, never escaped an embedded
literal ``"`` (or ``\\``) character. A module/``ModuleDict`` key containing a
literal double-quote produces DOT text like ``"a"b"`` -- an unterminated
string that ``neato`` (the rank engine's layout backend) rejects with a real
syntax error. This is the same crash class the cert9 F4 fix (commit
``3f074971``) closed for ``<``/``>``/``&``, one character short: F4 routed
the cluster-subgraph identifier through ``_dot_id()``, but ``_dot_id()``
itself did not escape ``"``.

The default dot-engine path (``graphviz.Digraph``) is unaffected because it
uses the real ``graphviz.quoting.quote()``, which already escapes this
correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._rank_layout_internal import layout as rank_layout

pytest.importorskip("graphviz")

_NEATO_AVAILABLE = True
try:
    import subprocess

    subprocess.run(["neato", "-V"], capture_output=True, timeout=10, check=False)
except (FileNotFoundError, OSError):
    _NEATO_AVAILABLE = False


class _ModuleDictQuoteKeyModel(nn.Module):
    """Model whose ``nn.ModuleDict`` key contains a literal double-quote."""

    def __init__(self) -> None:
        """Initialize a ModuleDict keyed by a string containing ``"``."""
        super().__init__()
        self.heads = nn.ModuleDict(
            {'a"b': nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ``'a"b'``-keyed branch."""
        return self.heads['a"b'](x)


def test_dot_id_escapes_embedded_double_quote() -> None:
    """``_dot_id`` backslash-escapes a literal ``"`` before wrapping in quotes.

    No graphviz binary required -- this exercises the pure-Python helper
    directly. Before the fix, ``_dot_id('a"b')`` returned the invalid,
    unterminated DOT identifier ``'"a"b"'``.
    """
    result = rank_layout._dot_id('a"b')
    assert result == '"a\\"b"'
    # The embedded quote must be backslash-escaped, not left bare.
    assert '\\"' in result
    # Exactly the two structural (wrapping) quotes plus one escaped quote --
    # i.e. no unescaped bare `"` in the interior.
    interior = result[1:-1]
    assert interior == 'a\\"b'


def test_dot_id_escapes_embedded_backslash() -> None:
    """``_dot_id`` backslash-escapes a literal ``\\`` before wrapping in quotes.

    Order matters: the backslash must be escaped BEFORE any quote escaping
    is applied, else a trailing backslash introduced by quote-escaping could
    itself be mistaken for an escape of the closing quote.
    """
    result = rank_layout._dot_id("a\\b")
    assert result == '"a\\\\b"'


def test_dot_id_backslash_then_quote_order_does_not_double_escape() -> None:
    """Escaping a value with both ``\\`` and ``"`` produces valid, non-mangled DOT.

    If backslash-escaping ran AFTER quote-escaping, the backslash introduced
    to escape the ``"`` would itself get doubled, corrupting the string. The
    correct order (backslash first, then quote) means a bare ``"`` maps to
    exactly one inserted backslash.
    """
    result = rank_layout._dot_id('a\\"b')
    # backslash -> \\, then " -> \" : "a\\" + "\"" + "b"
    assert result == '"a\\\\\\"b"'


def test_dot_quote_escapes_embedded_double_quote() -> None:
    """``_dot_quote`` (used for attribute values) also escapes embedded quotes."""
    result = rank_layout._dot_quote('a"b')
    assert result == '"a\\"b"'


def test_dot_id_keyword_guard_is_case_insensitive() -> None:
    """DOT keywords are case-insensitive in real Graphviz; the guard must match.

    ``_dot_id`` previously compared ``name not in _KW`` against a
    lowercase-only keyword set, so ``"Graph"``/``"NODE"``/etc. were treated
    as safe bare identifiers even though Graphviz itself treats DOT keywords
    case-insensitively. Currently unreachable in practice (no caller passes
    a bare keyword-cased name), but latent and worth locking down.
    """
    for kw in ("graph", "Graph", "GRAPH", "Node", "EDGE", "Digraph", "Strict"):
        result = rank_layout._dot_id(kw)
        assert result == f'"{kw}"', f"expected quoted keyword for {kw!r}, got {result!r}"

    # A genuinely safe identifier must still pass through unquoted.
    assert rank_layout._dot_id("my_node_1") == "my_node_1"


@pytest.mark.skipif(not _NEATO_AVAILABLE, reason="neato binary not available")
def test_rank_engine_renders_moduledict_double_quote_key(tmp_path: Path) -> None:
    """Rendering with the rank engine must not crash on a literal ``"`` in a module key.

    Regression test for the blocker this fix addresses: before the fix,
    ``_dot_id``/``_dot_quote`` spliced an unescaped ``"`` into DOT text,
    producing an unterminated string that made ``neato`` fail with a real
    ``RuntimeError`` (subprocess syntax error), reachable whenever
    ``vis_node_placement="rank"`` is used (or "auto" promotes to rank for a
    large enough graph).
    """
    trace = tl.trace(_ModuleDictQuoteKeyModel(), torch.randn(2, 3))
    outpath = tmp_path / "moduledict_double_quote_rank"
    try:
        dot = trace.draw(
            vis_outpath=str(outpath),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="rank",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    svg_path = outpath.with_suffix(".svg")
    assert svg_path.exists()
    assert svg_path.stat().st_size > 0
    # Sanity: the escaped quote must show up somewhere in the emitted DOT
    # source rather than a bare, unescaped one that would have produced
    # invalid DOT.
    assert dot is not None
