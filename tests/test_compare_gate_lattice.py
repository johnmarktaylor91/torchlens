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

Runs on DIFFERENT inputs are comparison operands by design (clean vs.
patched, two prompts, two checkpoints) when their sites line up: equal
graph-shape hashes. The input digest still decides line-up for one model
object, whose identity relationship cannot see a divergent control-flow path
(``_Branchy``), and a site whose shapes differ refuses typed
(``metric_shape_mismatch``) rather than failing inside a metric.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.errors import BundleComparisonError
from torchlens.types import Relationship

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


class _Branchy(nn.Module):
    """Data-dependent control flow: the graph depends on the input values."""

    def __init__(self) -> None:
        """Initialize the shared stem and the two branch layers."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.up = nn.Linear(4, 4)
        self.down = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, then one of two branches chosen by the stem's sum.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        h = self.stem(x)
        # Data-dependent branch: the sign of the stem output picks the graph.
        h = torch.relu(self.up(h)) if h.sum() > 0 else torch.sigmoid(self.down(h)) * 2.0
        return h + 1


def _branch_inputs(model: _Branchy) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one input per branch of ``model`` (positive, then negative stem sum).

    Parameters
    ----------
    model:
        Branching model.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Inputs that take the ``up`` and the ``down`` branch.
    """

    generator = torch.Generator().manual_seed(12)
    candidates = [torch.randn(1, 4, generator=generator) for _ in range(64)]
    with torch.no_grad():
        up = next(x for x in candidates if model.stem(x).sum() > 0)
        down = next(x for x in candidates if model.stem(x).sum() <= 0)
    return up, down


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


def test_leg1_identity_cross_input_lined_up_graphs_compare() -> None:
    """LEG1: one model object, same input shape, DIFFERENT values compares.

    The relationship stays identity-ranked (the reported vocabulary is
    unchanged); the pair is admitted because both graphs have the same
    graph-shape hash, so every site pairs one to one, not because of the
    identity rank.
    """

    model = _TwoLayer()
    torch.manual_seed(0)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)
    assert not torch.equal(x1, x2)

    left = _capture(model, x1)
    right = _capture(model, x2)
    bundle = tl.Bundle({"a": left, "b": right}, baseline="a")

    assert bundle.comparability("a", "b") is Relationship.SAME_OBJECT
    assert left._provenance_fingerprint == right._provenance_fingerprint

    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()
    assert float(matrix[0, 1]) > 0.0
    rows = dict(bundle.compare("a", "b"))
    assert rows and all(value >= 0.0 for value in rows.values())
    assert any(value > 0.0 for value in rows.values())
    assert bundle.rank_changes(k=3)


class _PerToken(nn.Module):
    """A per-token loop: the graph grows with the input length, like a prompt."""

    def __init__(self) -> None:
        """Initialize the shared token projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the projection of every row, one op chain per row.

        Parameters
        ----------
        x:
            Input tensor of shape ``(tokens, 4)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(4,)``.
        """

        total = torch.zeros(4)
        for row in x:
            total = total + self.proj(row)
        return total


def test_leg1c_different_length_inputs_name_the_shapes() -> None:
    """LEG1c: inputs of different lengths run different graphs; the refusal says why.

    A per-token loop records one op chain per row, so a 3-row and a 2-row
    input do not line up. The refusal names both input shapes instead of
    blaming control flow alone.
    """

    model = _PerToken()
    generator = torch.Generator().manual_seed(14)
    left = _capture(model, torch.randn(3, 4, generator=generator))
    right = _capture(model, torch.randn(2, 4, generator=generator))
    bundle = tl.Bundle({"a": left, "b": right}, baseline="a")

    assert left._provenance_fingerprint != right._provenance_fingerprint
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("a", "b")
    message = str(excinfo.value)
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert "differ in shape" in message
    assert "[3, 4]" in message.replace("(", "[").replace(")", "]")
    assert "[2, 4]" in message.replace("(", "[").replace(")", "]")


def test_leg1b_identity_divergent_branches_refuse_on_line_up() -> None:
    """LEG1b: one model object, different inputs, DIFFERENT control-flow paths.

    The identity relationship passes the model-axis floor even though the
    graphs differ; the line-up predicate (input digests differ AND the
    graph-shape hashes differ) refuses with its own code, naming the pair.
    """

    model = _Branchy()
    up, down = _branch_inputs(model)
    left = _capture(model, up)
    right = _capture(model, down)
    bundle = tl.Bundle({"a": left, "b": right}, baseline="a")

    assert bundle.comparability("a", "b") is Relationship.SAME_OBJECT
    assert left._provenance_fingerprint != right._provenance_fingerprint

    for read in (
        lambda: bundle.compare("a", "b"),
        lambda: bundle.rank_changes(k=3),
        lambda: bundle.at(tl.module("stem")).compare(),
    ):
        with pytest.raises(BundleComparisonError) as excinfo:
            read()
        assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
        assert {excinfo.value.fields["left_member"], excinfo.value.fields["right_member"]} == {
            "a",
            "b",
        }
        assert "do not line up" in str(excinfo.value)
        assert "differ in shape" not in str(excinfo.value)

    # Same branch on different inputs: the graphs line up again and compare.
    generator = torch.Generator().manual_seed(13)
    same_branch = next(
        x
        for x in (torch.randn(1, 4, generator=generator) for _ in range(64))
        if float(model.stem(x).sum()) > 0 and not torch.equal(x, up)
    )
    lined_up = tl.Bundle({"a": left, "c": _capture(model, same_branch)})
    assert lined_up.compare("a", "c")


def test_leg2_different_shape_refuses_typed_naming_the_site() -> None:
    """LEG2: inputs of different shapes refuse typed at the first mismatched site.

    The measured probe leg 'passed' a naive blocked-test only because
    ``cosine_distance`` threw an element-count mismatch by accident. The graphs
    line up (shapes are not part of the graph-shape hash), so the refusal is
    the per-site shape guard's: ``metric_shape_mismatch`` naming the site and
    the shapes, on every comparison read (test law 3).
    """

    model = _TwoLayer()
    torch.manual_seed(1)
    left = _capture(model, torch.randn(2, 4))
    right = _capture(model, torch.randn(3, 4))
    bundle = tl.Bundle({"a": left, "b": right}, baseline="a")

    with pytest.raises(InvalidArgumentError) as excinfo:
        bundle.at(tl.func("relu")).compare()
    assert excinfo.value.fields["code"] == "metric_shape_mismatch"
    assert "relu" in str(excinfo.value.fields["site"])
    assert excinfo.value.fields["member_shapes"] == {"a": (2, 4), "b": (3, 4)}
    assert not isinstance(excinfo.value, BundleComparisonError)

    for read in (lambda: bundle.compare("a", "b"), lambda: bundle.rank_changes(k=3)):
        with pytest.raises(InvalidArgumentError) as excinfo:
            read()
        assert excinfo.value.fields["code"] == "metric_shape_mismatch"
        assert excinfo.value.fields["site"]


def test_leg3_value_blind_hash_path_distinct_objects_compare() -> None:
    """LEG3: two DISTINCT model objects, different input VALUES, same shape.

    The derived relationship stays ``shared_graph_same_input`` (a SHAPE-level
    report, never consulted by the gate). The pair compares because the
    graph-shape hashes are equal; the value digests still tell the inputs
    apart (the comparison reports real, nonzero distances).
    """

    from torchlens.bundle._compare_gate import _input_value_digest

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(2)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)

    left = _capture(model_a, x1)
    right = _capture(model_b, x2)
    bundle = tl.Bundle({"a": left, "b": right})

    assert bundle.comparability("a", "b") is Relationship.SHARED_GRAPH_SAME_INPUT
    assert _input_value_digest(left) != _input_value_digest(right)
    matrix = bundle.at(tl.func("relu")).compare()
    assert float(matrix[0, 1]) > 0.0


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


def test_same_value_different_tensor_objects_succeeds() -> None:
    """Value-level identity keys on VALUES, not tensor object identity."""

    model = _TwoLayer()
    torch.manual_seed(4)
    x = torch.randn(2, 4)

    bundle = tl.Bundle({"a": _capture(model, x), "b": _capture(model, x.clone())})

    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)


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


def test_unproven_input_identity_needs_lined_up_graphs() -> None:
    """Members without retained input payloads compare only when graphs line up.

    A selective-save capture leaves the input ops unsaved, so value-level
    input identity is unprovable. Unproven is not proven-same: such a pair is
    admitted only on equal graph-shape hashes, and refuses UNPROVEN, naming
    the members lacking evidence, when the graphs differ.
    """

    model = _TwoLayer()
    torch.manual_seed(6)
    x = torch.randn(2, 4)

    left = tl.trace(model, x, save=tl.func("relu"))
    right = tl.trace(model, x, save=tl.func("relu"))
    lined_up = tl.Bundle({"a": left, "b": right})
    assert lined_up.at(tl.func("relu")).compare().shape == (2, 2)

    branchy = _Branchy()
    up, down = _branch_inputs(branchy)
    divergent = tl.Bundle(
        {
            "a": tl.trace(branchy, up, save=tl.module("stem")),
            "b": tl.trace(branchy, down, save=tl.module("stem")),
        }
    )
    with pytest.raises(BundleComparisonError) as excinfo:
        divergent.at(tl.module("stem")).compare()
    assert excinfo.value.fields["code"] == "bundle_gate_input_identity_unproven"
    assert excinfo.value.fields["unproven_members"] == ["a", "b"]


def test_operand_scoping_pair_reads_survive_a_foreign_member() -> None:
    """Item 3: one foreign member no longer disables every gated read.

    ``compare(a, b)`` gates exactly its named pair, and ``most_changed``
    gates only baseline pairs, while the whole-bundle reads (``at(site)``
    and its ``compare()``) honestly keep gating every pair they consume.
    The foreign member is a different model class (a different input alone
    no longer makes a member foreign).
    """

    model = _TwoLayer()
    torch.manual_seed(7)
    x = torch.randn(2, 4)

    bundle = tl.Bundle(
        {
            "a": _capture(model, x),
            "b": _capture(model, torch.randn(2, 4)),
            "foreign": _capture(_TanhModel(), x),
        },
        baseline="a",
    )

    # The named pair a/b lines up (different inputs, same graph): scoped read works.
    rows = bundle.compare("a", "b")
    assert rows and isinstance(rows, list)

    # at(site) consumes ALL pairs, so the foreign member still refuses it.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.at(tl.func("relu")).compare()
    assert excinfo.value.fields["code"] == "bundle_gate_model_axis_unmet"
    assert "foreign" in {
        excinfo.value.fields["left_member"],
        excinfo.value.fields["right_member"],
    }

    # most_changed compares baseline vs others: 'foreign' IS an operand there.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.rank_changes()
    assert excinfo.value.fields["code"] == "bundle_gate_model_axis_unmet"


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
    bundle = tl.Bundle({"turn1": turn1, "turn2": turn2, "turn3": turn3})
    bundle.relate(
        {"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}},
        {"kind": "successor_of", "from": "turn3", "to": "turn2", "params": {}},
    )

    # Direct edge.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]

    # Transitive directed path turn3 -> turn2 -> turn1.
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("turn1", "turn3")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of", "successor_of"]

    # Control: an identical bundle WITHOUT ordering rows passes the same read.
    control = tl.Bundle({"turn1": _capture(model, x), "turn2": _capture(model, x)})
    assert isinstance(control.compare("turn1", "turn2"), list)


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


def test_topology_refusal_precedes_input_evidence() -> None:
    """Ordered operands refuse on topology even when inputs also differ."""

    model = _TwoLayer()
    torch.manual_seed(10)
    turn1 = _capture(model, torch.randn(2, 4))
    turn2 = _capture(model, torch.randn(3, 4))
    bundle = tl.Bundle({"turn1": turn1, "turn2": turn2})
    bundle.relate({"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}})

    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"


def test_fork_comparison_workflow_stays_open() -> None:
    """The canonical clean-vs-ablated fork bundle keeps working."""

    model = _TwoLayer()
    torch.manual_seed(11)
    x = torch.randn(2, 4)

    clean = _capture(model, x)
    ablated = clean.fork("ablated")
    ablated._attach_hooks(tl.func("relu"), tl.zero_ablate())
    ablated.push()

    bundle = tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")
    matrix = bundle.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)
    rows = bundle.rank_changes()
    assert rows


def test_at_label_compare_runs_the_comparison_gate() -> None:
    """``at(label).compare()`` carries the gate ``compare_at`` ran; the label view itself reads."""

    branchy = _Branchy()
    up, down = _branch_inputs(branchy)
    divergent = tl.Bundle({"a": _capture(branchy, up), "b": _capture(branchy, down)})
    label = next(
        str(layer.layer_label)
        for layer in divergent["a"]._layer_list
        if str(layer.layer_label).startswith("linear")
    )
    view = divergent.at(label)
    assert set(view.members) == {"a", "b"}
    with pytest.raises(BundleComparisonError) as excinfo:
        view.compare()
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"
    assert excinfo.value.fields["operation"] == "at.compare"

    model = _TwoLayer()
    torch.manual_seed(11)
    cross_input = tl.Bundle(
        {"a": _capture(model, torch.randn(2, 4)), "b": _capture(model, torch.randn(2, 4))}
    )
    relu_label = next(
        str(layer.layer_label)
        for layer in cross_input["a"]._layer_list
        if "relu" in str(layer.layer_label)
    )
    assert cross_input.at(relu_label).compare().shape == (2, 2)
