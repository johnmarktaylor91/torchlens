"""Phase 9 tests for the single intervention Bundle type."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from example_models import TinyReluAdd as _ReluModel
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import (
    BaselineUndeterminedError,
    BundleComparisonError,
)
from torchlens.types import Relationship

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword


class _TanhModel(nn.Module):
    """Different-class model used for relationship blocking tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return torch.tanh(x) + 1


def _log(model: nn.Module | None = None, x: torch.Tensor | None = None) -> tl.Trace:
    """Capture an intervention-ready log.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tl.Trace
        Captured model log.
    """

    model = model or _ReluModel()
    x = x if x is not None else torch.randn(2, 3)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def test_bundle_construction_shapes_and_member_indexing() -> None:
    """Bundle accepts every Phase 9 construction shape."""

    log_a = _log()
    log_b = _log()
    log_a.name = "auto_a"
    log_b.name = "auto_b"

    from_dict = tl.Bundle({"a": log_a, "b": log_b})
    assert len(from_dict) == 2
    assert from_dict["a"] is log_a
    assert list(from_dict.names) == ["a", "b"]
    assert repr(from_dict) == (
        "Bundle(n_members=2, names=['a', 'b'], baseline=None, structurally_consistent=True)"
    )

    from_names = tl.Bundle([log_a, log_b], names=["a", "b"])
    assert from_names["b"] is log_b

    from_tuples = tl.Bundle([("a", log_a), ("b", log_b)])
    assert from_tuples["a"] is log_a

    from_auto = tl.Bundle([log_a, log_b])
    assert list(from_auto.names) == ["auto_a", "auto_b"]

    with pytest.raises(ValueError, match="unique"):
        tl.Bundle([("dup", log_a), ("dup", log_b)])


def test_bundle_getitem_returns_modellog_and_nodeview_accessor() -> None:
    """String indexing returns Trace; node access returns SuperOp member accessors."""

    log_a = _log()
    log_b = _log()
    bundle = tl.Bundle({"a": log_a, "b": log_b})

    assert isinstance(bundle["a"], type(log_a))
    assert bundle._supergraph is None

    node = bundle.at(tl.func("relu"))
    assert set(node.members) == {"a", "b"}
    assert set(node.names) == {"a", "b"}
    assert node.absent_names == set()
    assert node.label
    assert bundle._supergraph is not None


def test_map_passes_the_member_name_only_where_fn_takes_it() -> None:
    """A defaulted second parameter keeps its default; ``name`` arrives by keyword or slot."""

    bundle = tl.Bundle({"a": _log(), "b": _log()})
    sizes = {key: len(bundle[key]._layer_list) for key in ("a", "b")}

    # The README's one-argument function with a keyword default after it.
    assert bundle.map(lambda t, k=3: len(t._layer_list) * k) == {
        key: size * 3 for key, size in sizes.items()
    }
    # The documented two-argument form, any spelling of the second parameter.
    assert bundle.map(lambda log, name: name) == {"a": "a", "b": "b"}
    assert bundle.map(lambda log, member_name: member_name) == {"a": "a", "b": "b"}
    # A parameter called ``name`` receives it by keyword, wherever it sits.
    assert bundle.map(lambda log, name=None: name) == {"a": "a", "b": "b"}
    assert bundle.map(lambda log, k=3, name=None: (k, name)) == {"a": (3, "a"), "b": (3, "b")}

    def keyword_only(log: Any, *, name: str) -> str:
        return name

    assert bundle.map(keyword_only) == {"a": "a", "b": "b"}
    # ``*args`` keeps the historical positional (member, name) call.
    assert bundle.map(lambda *args: args[1]) == {"a": "a", "b": "b"}
    # Builtins (no inspectable signature) get the member only.
    assert bundle.map(type) == {"a": type(bundle["a"]), "b": type(bundle["b"])}


def test_capacity_eviction_preserves_baseline() -> None:
    """Capacity eviction never removes the configured baseline."""

    clean = _log()
    a = _log()
    b = _log()
    bundle = tl.Bundle({"clean": clean, "a": a, "b": b}, baseline="clean")

    bundle.capacity = 2

    assert "clean" in bundle
    assert len(bundle) == 2


def test_relationship_matrix_gates_at_operations() -> None:
    """Insufficient relationships fail lazily at operation time."""

    good = _log(_ReluModel())
    different = _log(_TanhModel())
    for member in (good, different):
        member._provenance_model_object_id = None
        member._source_class_qualname = None
        member._provenance_param_hash_quick = None
        member._provenance_param_hash_full = None
        member._provenance_fingerprint = None
        member._provenance_input_signature_hash = None
    bundle = tl.Bundle({"good": good, "different": different})

    assert bundle.comparability("good", "different") is Relationship.UNKNOWN
    with pytest.raises(BundleComparisonError, match="'at'"):
        bundle.at(tl.func("relu"))

    assert bundle.map(lambda member: member.name) == {
        "good": good.name,
        "different": different.name,
    }


def test_relationship_derivation_shared_graph_same_input() -> None:
    """Matching graph and input hashes derive shared_graph_same_input."""

    x = torch.randn(2, 3)
    left = _log(_ReluModel(), x)
    right = _log(_ReluModel(), x)
    bundle = tl.Bundle({"left": left, "right": right})

    assert bundle.comparability("left", "right") in {
        Relationship.SAME_OBJECT,
        Relationship.SAME_MODEL_OBJECT_AT_CAPTURE,
        Relationship.SHARED_GRAPH_SAME_INPUT,
    }
    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)


def test_baseline_auto_detect_and_ambiguity() -> None:
    """A single pristine member auto-detects; ambiguous pristine members raise on need."""

    clean = _log()
    dirty = _log()
    dirty._attach_hooks(tl.func("relu"), tl.zero_ablate())

    bundle = tl.Bundle({"clean": clean, "dirty": dirty})
    assert bundle.baseline_name == "clean"

    ambiguous = tl.Bundle({"a": _log(), "b": _log()})
    with pytest.raises(BaselineUndeterminedError):
        ambiguous.rank_changes()


def test_cluster_removed_and_tracebundle_not_top_level() -> None:
    """Bundle has no cluster placeholder and TraceBundle is not publicly exported."""

    bundle = tl.Bundle({"a": _log(), "b": _log()})

    assert not hasattr(bundle, "cluster")
    with pytest.raises(AttributeError):
        getattr(tl, "TraceBundle")
