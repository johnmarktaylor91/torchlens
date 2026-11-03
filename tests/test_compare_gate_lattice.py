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
from torchlens.intervention.errors import BundleComparisonError

_OPTS = tl.options.CaptureOptions(replayable=True)


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


def test_leg4_distinct_objects_identical_input_succeeds() -> None:
    """LEG4: distinct model objects with the IDENTICAL input stay comparable."""

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(3)
    x = torch.randn(2, 4)

    bundle = tl.Bundle({"a": _capture(model_a, x), "b": _capture(model_b, x)})

    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


def test_at_keeps_model_floor_and_carries_no_input_requirement() -> None:
    """``at(site)`` reads across different inputs; the model floor still gates it."""

    model = _TwoLayer()
    tanh_model = _TanhModel()
    torch.manual_seed(5)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)

    # Different input VALUES: at(site) is a view read, not a comparison — allowed.
    cross_input = tl.Bundle({"a": _capture(model, x1), "b": _capture(model, x2)})
    view = cross_input.at(tl.func("relu"))
    assert set(view.members) == {"a", "b"}

    # Different model class entirely: the model-axis floor refuses.
    cross_model = tl.Bundle({"a": _capture(model, x1), "b": _capture(tanh_model, x1)})
    with pytest.raises(BundleComparisonError) as excinfo:
        cross_model.at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_model_axis_unmet"
    assert excinfo.value.fields["operation"] == "at"
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
    bundle = tl.Bundle({"a": left, "b": right})

    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.at(tl.func("relu")).compare()
    assert excinfo.value.fields["code"] == "bundle_gate_input_identity_unproven"
    assert excinfo.value.fields["unproven_members"] == ["a", "b"]


def test_operand_scoping_pair_reads_survive_a_foreign_member() -> None:
    """Item 3: one foreign member no longer disables every gated read.

    ``compare(a, b)`` gates exactly its named pair, and ``most_changed``
    gates only baseline pairs — while the whole-bundle reads (``at(site).compare()``)
    honestly keep gating every pair they actually consume.
    """

    model = _TwoLayer()
    torch.manual_seed(7)
    x = torch.randn(2, 4)
    z = torch.randn(2, 4)

    bundle = tl.Bundle(
        {
            "a": _capture(model, x),
            "b": _capture(model, x),
            "foreign": _capture(model, z),
        },
        baseline="a",
    )

    # The named pair a/b is clean: scoped read works.
    rows = bundle.compare("a", "b")
    assert isinstance(rows, list)

    # at(site).compare() consumes ALL pairs, so the foreign member still refuses it.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.at(tl.func("relu")).compare()
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert "foreign" in {
        excinfo.value.fields["left_member"],
        excinfo.value.fields["right_member"],
    }

    # most_changed compares baseline vs others: 'foreign' IS an operand there.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.rank_changes()
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"


def test_ordering_siblings_and_alternatives_stay_comparable() -> None:
    """Only a DIRECTED ordering path refuses: siblings and peers compare."""

    model = _TwoLayer()
    torch.manual_seed(9)
    x = torch.randn(2, 4)

    bundle = tl.Bundle(
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
    rows = bundle.compare("turn2a", "turn2b")
    assert isinstance(rows, list)

    # The ordered pairs still refuse.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("turn1", "turn2a")
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

    bundle = tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")
    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)
    rows = bundle.rank_changes()
    assert rows


def test_at_label_compare_runs_the_comparison_gate() -> None:
    """``at(label).compare()`` carries the gate ``compare_at`` ran; the label view itself reads."""

    model = _TwoLayer()
    torch.manual_seed(11)
    cross_input = tl.Bundle(
        {"a": _capture(model, torch.randn(2, 4)), "b": _capture(model, torch.randn(2, 4))}
    )
    label = next(
        str(layer.layer_label)
        for layer in cross_input["a"]._layer_list
        if "relu" in str(layer.layer_label)
    )
    view = cross_input.at(label)
    assert set(view.members) == {"a", "b"}
    with pytest.raises(BundleComparisonError) as excinfo:
        view.compare()
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert excinfo.value.fields["operation"] == "at.compare"
