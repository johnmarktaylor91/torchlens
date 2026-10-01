"""A-GATE lattice legs: the two-predicate Bundle comparison gate (foldB D6).

The measured defect: the old ``_RELATIONSHIP_RANK`` lattice collapsed the
model axis and the input axis into one rank, so the gate refused in NONE of
four probe legs via TWO independent paths — (1) an identity relationship was
accepted as proof of input equality before the input was ever consulted, and
(2) the reachable input check hashed shape/dtype/device only (value-blind).

Every refusal test here obeys test law 3 (foldB D18): it asserts the refusal
fired for the RIGHT REASON — the stable ``fields["code"]`` plus the failed
predicate — never merely "the read raised". One measured probe leg passed a
naive blocked-test only because ``cosine_distance`` happened to refuse an
element-count mismatch downstream of the broken gate.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BundleRelationshipError
from torchlens.intervention.types import Relationship

pytestmark = pytest.mark.smoke

_OPTS = tl.options.CaptureOptions(intervention_ready=True)


class _TwoLayer(nn.Module):
    """Tiny two-linear model used for gate probe legs."""

    def __init__(self) -> None:
        """Initialize the two linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear -> relu -> linear.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.fc2(torch.relu(self.fc1(x)))


class _TanhModel(nn.Module):
    """Different-class model for model-axis floor legs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one tanh.

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


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture one intervention-ready trace.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Forward input.

    Returns
    -------
    tl.Trace
        Captured trace.
    """

    return tl.trace(model, x, capture=_OPTS)


def test_leg1_identity_shortcircuit_same_shape_different_values_refuses() -> None:
    """LEG1: one model object, same input shape, DIFFERENT values refuses.

    The old gate accepted the pair because the derivation returned
    ``same_object`` before consulting the input. The relationship STAYS
    identity-ranked (the reported vocabulary is unchanged) while the input
    predicate refuses on value evidence.
    """

    model = _TwoLayer()
    torch.manual_seed(0)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)
    assert not torch.equal(x1, x2)

    left = _capture(model, x1)
    right = _capture(model, x2)
    bundle = tl.bundle({"a": left, "b": right})

    # Path-1 shape: the relationship itself is identity-ranked.
    assert bundle.relationship("a", "b") is Relationship.SAME_OBJECT

    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert excinfo.value.fields["operation"] == "compare_at"
    assert {excinfo.value.fields["left_member"], excinfo.value.fields["right_member"]} == {
        "a",
        "b",
    }


def test_leg2_different_shape_refuses_at_the_gate_not_downstream() -> None:
    """LEG2: the refusal is the GATE's, not a downstream metric accident.

    The measured probe leg 'passed' a naive blocked-test only because
    ``cosine_distance`` threw an element-count mismatch AFTER the gate had
    already waved the pair through. The gate must now refuse first, with its
    own stable code (test law 3).
    """

    model = _TwoLayer()
    torch.manual_seed(1)
    left = _capture(model, torch.randn(2, 4))
    right = _capture(model, torch.randn(3, 4))
    bundle = tl.bundle({"a": left, "b": right})

    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    # The right-reason assertion: nothing about element counts or metrics.
    assert "cosine" not in str(excinfo.value)
    assert "element" not in str(excinfo.value)


def test_leg3_value_blind_hash_path_distinct_objects_refuses() -> None:
    """LEG3: two DISTINCT model objects, different input VALUES, same shape.

    This pair reaches the input check (no identity short-circuit), where the
    old shape/dtype/device hash reported "same input" for genuinely different
    values. The derived relationship legitimately stays
    ``shared_graph_same_input`` (a SHAPE-level report); the gate must still
    refuse on value evidence.
    """

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(2)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)

    left = _capture(model_a, x1)
    right = _capture(model_b, x2)
    bundle = tl.bundle({"a": left, "b": right})

    # The value-blind rank IS derived (shape-level vocabulary unchanged) ...
    assert bundle.relationship("a", "b") is Relationship.SHARED_GRAPH_SAME_INPUT
    # ... and is still not accepted as proof of input equality.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"


def test_leg4_distinct_objects_identical_input_succeeds() -> None:
    """LEG4: distinct model objects with the IDENTICAL input stay comparable."""

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(3)
    x = torch.randn(2, 4)

    bundle = tl.bundle({"a": _capture(model_a, x), "b": _capture(model_b, x)})

    matrix = bundle.compare_at(tl.func("relu"))
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


def test_same_value_different_tensor_objects_succeeds() -> None:
    """Value-level identity keys on VALUES, not tensor object identity."""

    model = _TwoLayer()
    torch.manual_seed(4)
    x = torch.randn(2, 4)

    bundle = tl.bundle({"a": _capture(model, x), "b": _capture(model, x.clone())})

    matrix = bundle.compare_at(tl.func("relu"))
    assert matrix.shape == (2, 2)


def test_node_keeps_model_floor_and_carries_no_input_requirement() -> None:
    """``node`` reads across different inputs; the model floor still gates it."""

    model = _TwoLayer()
    tanh_model = _TanhModel()
    torch.manual_seed(5)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)

    # Different input VALUES: node is a view read, not a comparison — allowed.
    cross_input = tl.bundle({"a": _capture(model, x1), "b": _capture(model, x2)})
    view = cross_input.node(tl.func("relu"))
    assert set(view.members) == {"a", "b"}

    # Different model class entirely: the model-axis floor refuses.
    cross_model = tl.bundle({"a": _capture(model, x1), "b": _capture(tanh_model, x1)})
    with pytest.raises(BundleRelationshipError) as excinfo:
        cross_model.node(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_model_axis_unmet"
    assert excinfo.value.fields["operation"] == "node"
    assert excinfo.value.fields["required_floor"] == "same_param_shapes"


def test_unproven_input_identity_refuses_fail_closed() -> None:
    """Members without retained input payloads refuse UNPROVEN, never pass.

    A selective-save capture leaves the input ops unsaved, so value-level
    input identity is unprovable — even though the two captures genuinely
    used the same input. Unproven is not proven-same (fail-closed), and the
    refusal names the members lacking evidence.
    """

    model = _TwoLayer()
    torch.manual_seed(6)
    x = torch.randn(2, 4)

    left = tl.trace(model, x, save=tl.func("relu"))
    right = tl.trace(model, x, save=tl.func("relu"))
    bundle = tl.bundle({"a": left, "b": right})

    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_identity_unproven"
    assert excinfo.value.fields["unproven_members"] == ["a", "b"]


def test_operand_scoping_pair_reads_survive_a_foreign_member() -> None:
    """Item 3: one foreign member no longer disables every gated read.

    ``diff_pair(a, b)`` gates exactly its named pair, and ``most_changed``
    gates only baseline pairs — while the whole-bundle reads (``compare_at``)
    honestly keep gating every pair they actually consume.
    """

    model = _TwoLayer()
    torch.manual_seed(7)
    x = torch.randn(2, 4)
    z = torch.randn(2, 4)

    bundle = tl.bundle(
        {
            "a": _capture(model, x),
            "b": _capture(model, x),
            "foreign": _capture(model, z),
        },
        baseline="a",
    )

    # The named pair a/b is clean: scoped read works.
    rows = bundle.diff_pair("a", "b")
    assert isinstance(rows, list)

    # compare_at consumes ALL pairs, so the foreign member still refuses it.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert "foreign" in {
        excinfo.value.fields["left_member"],
        excinfo.value.fields["right_member"],
    }

    # most_changed compares baseline vs others: 'foreign' IS an operand there.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.most_changed()
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"


def test_ordering_topology_refusal_direct_and_transitive() -> None:
    """Item 4: members joined by an ordering relation are not comparison operands.

    Inputs are IDENTICAL across members so the topology refusal is provably
    the reason (test law 3): remove the ordering rows and the same reads pass.
    """

    model = _TwoLayer()
    torch.manual_seed(8)
    x = torch.randn(2, 4)

    turn1 = _capture(model, x)
    turn2 = _capture(model, x)
    turn3 = _capture(model, x)
    bundle = tl.bundle({"turn1": turn1, "turn2": turn2, "turn3": turn3})
    bundle.relate(
        {"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}},
        {"kind": "successor_of", "from": "turn3", "to": "turn2", "params": {}},
    )

    # Direct edge.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.diff_pair("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]

    # Transitive directed path turn3 -> turn2 -> turn1.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.diff_pair("turn1", "turn3")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of", "successor_of"]

    # Control: an identical bundle WITHOUT ordering rows passes the same read.
    control = tl.bundle({"turn1": _capture(model, x), "turn2": _capture(model, x)})
    assert isinstance(control.diff_pair("turn1", "turn2"), list)


def test_ordering_siblings_and_alternatives_stay_comparable() -> None:
    """Only a DIRECTED ordering path refuses: siblings and peers compare."""

    model = _TwoLayer()
    torch.manual_seed(9)
    x = torch.randn(2, 4)

    bundle = tl.bundle(
        {
            "turn1": _capture(model, x),
            "turn2a": _capture(model, x),
            "turn2b": _capture(model, x),
        }
    )
    bundle.relate(
        {"kind": "successor_of", "from": "turn2a", "to": "turn1", "params": {}},
        {"kind": "successor_of", "from": "turn2b", "to": "turn1", "params": {}},
        {"kind": "alternative_of", "from": "turn2b", "to": "turn2a", "params": {}},
    )

    # Siblings through a common predecessor are not on one directed path,
    # and alternative_of is deliberately not an ordering kind.
    rows = bundle.diff_pair("turn2a", "turn2b")
    assert isinstance(rows, list)

    # The ordered pairs still refuse.
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.diff_pair("turn1", "turn2a")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"


def test_topology_refusal_precedes_input_evidence() -> None:
    """Ordered operands refuse on topology even when inputs also differ."""

    model = _TwoLayer()
    torch.manual_seed(10)
    turn1 = _capture(model, torch.randn(2, 4))
    turn2 = _capture(model, torch.randn(3, 4))
    bundle = tl.bundle({"turn1": turn1, "turn2": turn2})
    bundle.relate({"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}})

    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.diff_pair("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"


def test_fork_comparison_workflow_stays_open() -> None:
    """The canonical clean-vs-ablated fork bundle keeps working."""

    model = _TwoLayer()
    torch.manual_seed(11)
    x = torch.randn(2, 4)

    clean = _capture(model, x)
    ablated = clean.fork("ablated")
    ablated.attach_hooks(tl.func("relu"), tl.zero_ablate())
    ablated.push()

    bundle = tl.bundle({"clean": clean, "ablated": ablated}, baseline="clean")
    matrix = bundle.compare_at(tl.func("relu"))
    assert matrix.shape == (2, 2)
    rows = bundle.most_changed()
    assert rows
