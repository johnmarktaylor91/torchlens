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
``SelectionError`` and ``_SelectionOperand`` out of ``torchlens.selection``
-- "six modules move" vs "drag the selection substrate too" is exactly the
decision Rule V2 requires publishing, and lane C03 is concurrently
rewriting those same modules (immutable InterventionSpec). The move rides a
C03-coordinated amendment; the closure facts are pinned here so the
handoff cannot silently rot.
"""

from __future__ import annotations

import ast
import importlib
import pickle
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parent.parent


class TestTraceStateRelocation:
    def test_shim_and_home_serve_one_object(self) -> None:
        shim = importlib.import_module("torchlens._trace_state")
        home = importlib.import_module("torchlens._vocab.trace_state")
        assert shim.TraceState is home.TraceState
        assert shim.__tl_layer__ == "FACADE"
        assert home.__tl_layer__ == "L0"
        assert home.__tl_vocabulary__ is True

    def test_pickle_identity_is_the_historical_path(self) -> None:
        home = importlib.import_module("torchlens._vocab.trace_state")
        assert home.TraceState.__module__ == "torchlens._trace_state"
        state = pickle.loads(pickle.dumps(home.TraceState.PRISTINE))
        assert state is home.TraceState.PRISTINE

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
    def test_shim_and_home_serve_one_object(self) -> None:
        shim = importlib.import_module("torchlens.visualization.node_spec")
        home = importlib.import_module("torchlens._vocab.node_spec")
        assert shim.NodeSpec is home.NodeSpec
        assert shim.NodeSpecFn is home.NodeSpecFn
        assert shim.BackwardNodeSpecFn is home.BackwardNodeSpecFn
        assert shim.CollapsedNodeSpecFn is home.CollapsedNodeSpecFn
        assert shim.INTERVENTION_SITE_COLOR == home.INTERVENTION_SITE_COLOR

    def test_pickle_identity_is_the_historical_path(self) -> None:
        home = importlib.import_module("torchlens._vocab.node_spec")
        assert home.NodeSpec.__module__ == "torchlens.visualization.node_spec"
        spec = home.NodeSpec(lines=["a"])
        assert pickle.loads(pickle.dumps(spec)) == spec

    def test_vocabulary_home_smuggles_no_behavior(self) -> None:
        """Rule V3: the vocabulary home holds no deferred behavior imports."""

        tree = ast.parse((REPO / "torchlens/_vocab/node_spec.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    assert not isinstance(inner, (ast.Import, ast.ImportFrom)), (
                        f"deferred import inside {node.name}: behavior smuggling (V3)"
                    )

    def test_lower_layer_consumers_import_the_vocabulary_home(self) -> None:
        for consumer in (
            "torchlens/options.py",
            "torchlens/viz/feature_maps.py",
            "torchlens/receptive_field/_viz.py",
        ):
            source = (REPO / consumer).read_text()
            assert "_vocab.node_spec import" in source, (
                f"{consumer} must import NodeSpec vocabulary from the L0 home, "
                "not upward from visualization/"
            )
            assert "visualization.node_spec import" not in source


class TestSiteKeyRelocation:
    def test_shim_and_home_serve_one_object(self) -> None:
        shim = importlib.import_module("torchlens.postprocess._site_key")
        home = importlib.import_module("torchlens.data_classes._site_key")
        assert shim.SiteKeyMinter is home.SiteKeyMinter
        assert shim.parse_site_key is home.parse_site_key
        assert shim.SITE_KEY_PREFIX == home.SITE_KEY_PREFIX
        assert home.__tl_layer__ == "L1"

    def test_engine_and_product_consumers_import_the_l1_home(self) -> None:
        for consumer in (
            "torchlens/backends/_finalize.py",
            "torchlens/backends/jax/_site_dialect.py",
            "torchlens/data_classes/_trace_inventory.py",
            "torchlens/validation/_invariants_sites.py",
            "torchlens/_io/forgery_validation.py",
            "torchlens/postprocess/loop_detection.py",
        ):
            source = (REPO / consumer).read_text()
            assert "postprocess._site_key" not in source, (
                f"{consumer} still imports the old postprocess home"
            )


class TestInterventionTripleHandoff:
    """The published-closure facts for the deferred intervention-triple move."""

    def test_closure_still_drags_the_selection_substrate(self) -> None:
        """The blocker is real: types/errors eagerly import torchlens.selection.

        If this test starts failing, the closure shrank (C03 or a later lane
        broke the selection dependency) and the triple move is unblocked --
        execute it as the C01/F35 amendment instead of deleting this pin.
        """

        types_source = (REPO / "torchlens/intervention/types.py").read_text()
        errors_source = (REPO / "torchlens/intervention/errors.py").read_text()
        assert "from ..selection import" in types_source
        assert "from ..selection import" in errors_source

    def test_selectors_depend_on_types(self) -> None:
        selectors_source = (REPO / "torchlens/intervention/selectors.py").read_text()
        assert "from .types import" in selectors_source
