"""Ratified V2 relocation gates (C01 item 6; architecture memo 3.3).

Executed moves (each with its published closure and a shim keeping every
historical spelling + pickle-visible identity):

- ``_trace_state`` -> ``_vocab/trace_state`` (closure: 1 module, enum only)
- ``visualization/node_spec`` VOCABULARY -> ``_vocab/node_spec`` (closure:
  1 module; the render-behavior helpers stay at L7 under Rule V3)
- ``SiteKeyMinter`` (``postprocess/_site_key``) -> ``data_classes/_site_key``
  (closure: 1 module, typing only; V3: it MINTS keys, so it moves to L1
  with the coordinates it mints)
- the ``_io`` schema half (item 3, gated in test_arch_spine_io_split.py)

PUBLISHED-CLOSURE HANDOFF (move not executed in C01): the intervention
triple (``intervention/types`` + ``errors`` + ``selectors``) drags
``SelectionError`` and ``_SelectionOperand`` out of ``torchlens._selection.selection``
-- "six modules move" vs "drag the selection substrate too" is exactly the
decision Rule V2 requires publishing, and lane C03 is concurrently
rewriting those same modules (immutable InterventionSpec). The move rides a
C03-coordinated amendment; the closure facts are pinned here so the
handoff cannot silently rot.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestTraceStateRelocation:
    def test_home_closure_is_vocabulary_only(self) -> None:
        tree = ast.parse((REPO / "torchlens/_vocab/trace_state.py").read_text())
        torchlens_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.level or "torchlens" in (node.module or ""))
        ]
        assert torchlens_imports == [], "trace_state vocabulary must import nothing from torchlens"


class TestNodeSpecRelocation:
    def test_vocabulary_home_smuggles_no_behavior(self) -> None:
        """Rule V3: the vocabulary home holds no deferred behavior imports."""

        tree = ast.parse((REPO / "torchlens/_vocab/node_spec.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    assert not isinstance(inner, (ast.Import, ast.ImportFrom)), (
                        f"deferred import inside {node.name}: behavior smuggling (V3)"
                    )
