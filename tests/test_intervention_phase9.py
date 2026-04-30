"""Phase 9 tests for the single intervention Bundle type."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._run_state import RunState
from torchlens.intervention.errors import (
    BaselineUndeterminedError,
    BundleRelationshipError,
)
from torchlens.intervention.types import Relationship


class _ReluModel(nn.Module):
    """Small model with one relu site."""

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

        return torch.relu(x) + 1


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


def _log(model: nn.Module | None = None, x: torch.Tensor | None = None) -> tl.ModelLog:
    """Capture an intervention-ready log.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    model = model or _ReluModel()
    x = x if x is not None else torch.randn(2, 3)
    return tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)


def test_bundle_construction_shapes_and_member_indexing() -> None:
    """Bundle accepts every Phase 9 construction shape."""

    log_a = _log()
    log_b = _log()
    log_a.name = "auto_a"
    log_b.name = "auto_b"

    from_dict = tl.bundle({"a": log_a, "b": log_b})
    assert len(from_dict) == 2
    assert from_dict["a"] is log_a
    assert list(from_dict.names) == ["a", "b"]

    from_names = tl.bundle([log_a, log_b], names=["a", "b"])
    assert from_names["b"] is log_b

    from_tuples = tl.bundle([("a", log_a), ("b", log_b)])
    assert from_tuples["a"] is log_a

    from_auto = tl.bundle([log_a, log_b])
    assert list(from_auto.names) == ["auto_a", "auto_b"]

    with pytest.raises(ValueError, match="unique"):
        tl.bundle([("dup", log_a), ("dup", log_b)])


def test_bundle_getitem_returns_modellog_and_nodeview_dicts() -> None:
    """String indexing returns ModelLog; node access returns dict-keyed NodeView fields."""

    log_a = _log()
    log_b = _log()
    bundle = tl.bundle({"a": log_a, "b": log_b})

    assert isinstance(bundle["a"], type(log_a))
    assert bundle._supergraph is None

    node = bundle.node(tl.func("relu"))
    assert set(node.activations) == {"a", "b"}
    assert set(node.labels) == {"a", "b"}
    assert set(node.members) == {"a", "b"}
    assert bundle._supergraph is not None


def test_metric_and_joint_metric_contracts() -> None:
    """metric is per-member and joint_metric receives the bundle."""

    bundle = tl.bundle({"a": _log(), "b": _log()})

    assert bundle.metric(lambda member: len(member.layer_list)) == {
        "a": len(bundle["a"].layer_list),
        "b": len(bundle["b"].layer_list),
    }
    assert bundle.joint_metric(lambda b: b.names) == ["a", "b"]


def test_set_capacity_preserves_baseline() -> None:
    """Capacity eviction never removes the configured baseline."""

    clean = _log()
    a = _log()
    b = _log()
    bundle = tl.bundle({"clean": clean, "a": a, "b": b}, baseline="clean")

    bundle.set_capacity(2)

    assert "clean" in bundle
    assert len(bundle) == 2


def test_relationship_matrix_gates_node_operations() -> None:
    """Insufficient relationships fail lazily at operation time."""

    good = _log(_ReluModel())
    different = _log(_TanhModel())
    for member in (good, different):
        member.source_model_id = None
        member.source_model_class = None
        member.weight_fingerprint_at_capture = None
        member.weight_fingerprint_full = None
        member.graph_shape_hash = None
        member.input_shape_hash = None
    bundle = tl.bundle({"good": good, "different": different})

    assert bundle.relationship("good", "different") is Relationship.UNKNOWN
    with pytest.raises(BundleRelationshipError, match="node"):
        bundle.node(tl.func("relu"))

    assert bundle.metric(lambda member: member.name) == {
        "good": good.name,
        "different": different.name,
    }


def test_relationship_derivation_shared_graph_same_input() -> None:
    """Matching graph and input hashes derive shared_graph_same_input."""

    x = torch.randn(2, 3)
    left = _log(_ReluModel(), x)
    right = _log(_ReluModel(), x)
    bundle = tl.bundle({"left": left, "right": right})

    assert bundle.relationship("left", "right") in {
        Relationship.SAME_OBJECT,
        Relationship.SAME_MODEL_OBJECT_AT_CAPTURE,
        Relationship.SHARED_GRAPH_SAME_INPUT,
    }
    matrix = bundle.compare_at(tl.func("relu"))
    assert matrix.shape == (2, 2)


def test_baseline_auto_detect_and_ambiguity() -> None:
    """A single pristine member auto-detects; ambiguous pristine members raise on need."""

    clean = _log()
    dirty = _log()
    dirty._spec_revision = 1
    dirty.run_state = RunState.SPEC_STALE

    bundle = tl.bundle({"clean": clean, "dirty": dirty})
    assert bundle.baseline_name == "clean"

    ambiguous = tl.bundle({"a": _log(), "b": _log()})
    with pytest.raises(BaselineUndeterminedError):
        ambiguous.most_changed()


def test_cluster_deferred_and_tracebundle_not_top_level() -> None:
    """cluster is explicitly deferred and TraceBundle is not publicly exported."""

    bundle = tl.bundle({"a": _log(), "b": _log()})

    with pytest.raises(NotImplementedError, match="v1"):
        bundle.cluster()
    with pytest.raises(AttributeError):
        getattr(tl, "TraceBundle")


def test_bundle_help_lists_readiness() -> None:
    """help includes per-member run-state and baseline status."""

    clean = _log()
    dirty = _log()
    dirty._spec_revision = 1
    dirty.run_state = RunState.SPEC_STALE
    bundle = tl.bundle({"clean": clean, "dirty": dirty}, baseline="clean")

    text = bundle.help()

    assert "Bundle (2 members):" in text
    assert "clean: baseline" in text
    assert "dirty:" in text
    assert "SPEC_STALE" in text
