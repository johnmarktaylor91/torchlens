"""A-GATE save/load legs: every gate claim repeated after a round trip.

Test law 2 (foldB D18): in process, every pair short-circuits toward
identity evidence, so gate bugs are invisible at notebook scale — every
cross-member claim must therefore also be proven AFTER a save/load round
trip. This file repeats the cross-input line-up law (lined-up graphs compare,
divergent control flow refuses), the legitimate-pair success, the unproven
refusal on payload-free artifacts whose graphs differ, and
the topology refusal through the Bundle artifact door, and carries the
rank-upgrade-across-save/load audit (A-GATE item 5) as executable law.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._compare_gate import _MODEL_AXIS_RANK
from torchlens.intervention.errors import BundleComparisonError
from torchlens.types import Relationship

_OPTS = tl.options.CaptureOptions(replayable=True)


class _TwoLayer(nn.Module):
    """Tiny two-linear model used for round-trip gate legs."""

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
        if h.sum() > 0:
            h = torch.relu(self.up(h))
        else:
            h = torch.sigmoid(self.down(h)) * 2.0
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


def _round_trip(trace: tl.Trace, path: Path) -> tl.Trace:
    """Save one trace at the default level and load it back.

    Parameters
    ----------
    trace:
        Trace to persist.
    path:
        Destination ``.tlspec`` path.

    Returns
    -------
    tl.Trace
        Loaded trace.
    """

    trace.save(str(path))
    return tl.load(str(path))


def test_leg1_cross_input_line_up_survives_save_load(tmp_path: Path) -> None:
    """Different input values: lined-up graphs compare, divergent ones refuse, after loading.

    Default saves retain the input payloads and the graph-shape hash, so the
    line-up predicate is decided identically on loaded artifacts: the
    same-graph pair reports real distances and the divergent-branch pair
    refuses on line-up.
    """

    model = _TwoLayer()
    torch.manual_seed(0)
    left = _capture(model, torch.randn(2, 4))
    right = _capture(model, torch.randn(2, 4))

    loaded = tl.Bundle(
        {
            "a": _round_trip(left, tmp_path / "a.tlspec"),
            "b": _round_trip(right, tmp_path / "b.tlspec"),
        }
    )
    matrix = loaded.at(tl.func("relu")).compare()
    assert float(matrix[0, 1]) > 0.0

    branchy = _Branchy()
    up, down = _branch_inputs(branchy)
    divergent = tl.Bundle(
        {
            "a": _round_trip(_capture(branchy, up), tmp_path / "up.tlspec"),
            "b": _round_trip(_capture(branchy, down), tmp_path / "down.tlspec"),
        }
    )
    with pytest.raises(BundleComparisonError) as excinfo:
        divergent.compare("a", "b")
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"


def test_leg4_success_survives_save_load(tmp_path: Path) -> None:
    """Distinct models with the identical input stay comparable after loading."""

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(1)
    x = torch.randn(2, 4)

    loaded = tl.Bundle(
        {
            "a": _round_trip(_capture(model_a, x), tmp_path / "a.tlspec"),
            "b": _round_trip(_capture(model_b, x), tmp_path / "b.tlspec"),
        }
    )
    matrix = loaded.at(tl.func("relu")).compare()
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


def test_unproven_after_payload_free_round_trip(tmp_path: Path) -> None:
    """A payload-free (runnable-level) artifact refuses UNPROVEN when graphs differ.

    The sparse runnable core carries no activation payloads, so value-level
    input identity is unprovable after the round trip. A pair whose
    graph-shape hashes are equal still lines up (the gate admits it); a pair
    whose graphs differ refuses with the gate's own code, names the members,
    and never falls back to the shape-level hash.
    """

    model = _TwoLayer()
    torch.manual_seed(2)
    x = torch.randn(2, 4)
    _capture(model, x).save(str(tmp_path / "a.tlspec"), level="runnable")
    _capture(model, x).save(str(tmp_path / "b.tlspec"), level="runnable")
    lined_up = tl.Bundle(
        {
            "a": tl.load(str(tmp_path / "a.tlspec")),
            "b": tl.load(str(tmp_path / "b.tlspec")),
        }
    )
    lined_up._require_comparable("at.compare")

    branchy = _Branchy()
    up, down = _branch_inputs(branchy)
    _capture(branchy, up).save(str(tmp_path / "up.tlspec"), level="runnable")
    _capture(branchy, down).save(str(tmp_path / "down.tlspec"), level="runnable")
    divergent = tl.Bundle(
        {
            "a": tl.load(str(tmp_path / "up.tlspec")),
            "b": tl.load(str(tmp_path / "down.tlspec")),
        }
    )
    with pytest.raises(BundleComparisonError) as excinfo:
        divergent._require_comparable("at.compare")
    assert excinfo.value.fields["code"] == "bundle_gate_input_identity_unproven"
    assert excinfo.value.fields["unproven_members"] == ["a", "b"]


def test_rank_upgrade_audit_across_save_load(tmp_path: Path) -> None:
    """Item 5: save/load never manufactures a relationship-rank upgrade.

    Loads remap ``model_object_id`` (every loaded artifact in this suite
    reports the same small id), so trusting persisted object ids for
    identity ranks would grade two loaded captures of DIFFERENT models as
    ``same_object``. The audit pins: (a) loaded pairs never derive an
    identity rank, and (b) the loaded model-axis rank never exceeds the
    live one, for every probe-leg pair shape.
    """

    model = _TwoLayer()
    other = _TwoLayer()
    torch.manual_seed(3)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)

    captures = {
        "m1x1": _capture(model, x1),
        "m1x2": _capture(model, x2),
        "m2x1": _capture(other, x1),
        "m2x2": _capture(other, x2),
    }
    live = tl.Bundle(dict(captures))
    loaded = tl.Bundle(
        {name: _round_trip(trace, tmp_path / f"{name}.tlspec") for name, trace in captures.items()}
    )

    # The collision premise the audit exists for: persisted object ids are
    # NOT unique across loaded artifacts.
    assert loaded["m1x1"].provenance.model_object_id == loaded["m2x1"].provenance.model_object_id

    identity_ranks = {Relationship.SAME_OBJECT, Relationship.SAME_MODEL_OBJECT_AT_CAPTURE}
    names = list(captures)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            live_rel = live.comparability(left_name, right_name)
            loaded_rel = loaded.comparability(left_name, right_name)
            assert _MODEL_AXIS_RANK[loaded_rel] <= _MODEL_AXIS_RANK[live_rel], (
                f"{left_name}/{right_name}: loaded {loaded_rel.value!r} outranks "
                f"live {live_rel.value!r}"
            )
            assert loaded_rel not in identity_ranks, (
                f"{left_name}/{right_name}: loaded pair manufactured identity "
                f"rank {loaded_rel.value!r}"
            )

    # The definitive shape: same live model object was same_object in
    # session; after loading, the pair settles to the graph evidence that
    # actually round-trips.
    assert live.comparability("m1x1", "m1x2") is Relationship.SAME_OBJECT
    assert loaded.comparability("m1x1", "m1x2") is Relationship.SHARED_GRAPH_SAME_INPUT


def test_topology_refusal_survives_bundle_round_trip(tmp_path: Path) -> None:
    """The ordering-topology refusal fires identically on a loaded Bundle.

    Inputs are identical across members so the topology code is provably the
    refusal reason on both sides of the artifact door (test law 3).
    """

    model = _TwoLayer()
    torch.manual_seed(4)
    x = torch.randn(2, 4)
    bundle = tl.Bundle({"turn1": _capture(model, x), "turn2": _capture(model, x)})
    bundle.relate({"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}})

    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.compare("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"

    path = tmp_path / "chain.tlspec"
    bundle.save(str(path))
    loaded = tl.load(str(path))

    with pytest.raises(BundleComparisonError) as excinfo:
        loaded.compare("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]


def test_scoped_pair_read_works_after_bundle_round_trip(tmp_path: Path) -> None:
    """Item 3 after the artifact door: the clean pair still reads on a loaded
    Bundle that also carries an ordered member."""

    model = _TwoLayer()
    torch.manual_seed(5)
    x = torch.randn(2, 4)
    bundle = tl.Bundle(
        {
            "a": _capture(model, x),
            "b": _capture(model, x),
            "next": _capture(model, x),
        }
    )
    bundle.relate({"kind": "successor_of", "from": "next", "to": "b", "params": {}})

    path = tmp_path / "scoped.tlspec"
    bundle.save(str(path))
    loaded = tl.load(str(path))

    rows = loaded.compare("a", "b")
    assert isinstance(rows, list)
    with pytest.raises(BundleComparisonError) as excinfo:
        loaded.compare("b", "next")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
