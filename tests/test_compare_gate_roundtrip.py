"""A-GATE save/load legs: every gate claim repeated after a round trip.

Test law 2 (foldB D18): in process, every pair short-circuits toward
identity evidence, so gate bugs are invisible at notebook scale — every
cross-member claim must therefore also be proven AFTER a save/load round
trip. This file repeats both defect-path refusals, the legitimate-pair
success, the fail-closed unproven refusal on payload-free artifacts, and
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
from torchlens.intervention.errors import BundleRelationshipError
from torchlens.intervention.types import Relationship

pytestmark = pytest.mark.smoke

_OPTS = tl.options.CaptureOptions(intervention_ready=True)


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

    tl.save(trace, str(path))
    return tl.load(str(path))


def test_leg1_refusal_survives_save_load(tmp_path: Path) -> None:
    """Same-shape/different-values still refuses on VALUES after a round trip.

    Default saves retain the input payloads, so the value-level predicate
    stays provable — and provably different — on loaded artifacts.
    """

    model = _TwoLayer()
    torch.manual_seed(0)
    left = _capture(model, torch.randn(2, 4))
    right = _capture(model, torch.randn(2, 4))

    loaded = tl.bundle(
        {
            "a": _round_trip(left, tmp_path / "a.tlspec"),
            "b": _round_trip(right, tmp_path / "b.tlspec"),
        }
    )
    with pytest.raises(BundleRelationshipError) as excinfo:
        loaded.compare_at(tl.func("relu"))
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"


def test_leg4_success_survives_save_load(tmp_path: Path) -> None:
    """Distinct models with the identical input stay comparable after loading."""

    model_a = _TwoLayer()
    model_b = _TwoLayer()
    torch.manual_seed(1)
    x = torch.randn(2, 4)

    loaded = tl.bundle(
        {
            "a": _round_trip(_capture(model_a, x), tmp_path / "a.tlspec"),
            "b": _round_trip(_capture(model_b, x), tmp_path / "b.tlspec"),
        }
    )
    matrix = loaded.compare_at(tl.func("relu"))
    assert matrix.shape == (2, 2)
    assert torch.isfinite(matrix).all()


def test_unproven_after_payload_free_round_trip(tmp_path: Path) -> None:
    """A payload-free (runnable-level) artifact refuses UNPROVEN, fail-closed.

    The sparse runnable core carries no activation payloads, so value-level
    input identity is unprovable after the round trip even for genuinely
    identical inputs. The refusal is the gate's own, names the members, and
    never falls back to the shape-level hash. The persisted carrier that
    will prove this case is the A-GATE/digest slice (after C07-X).
    """

    model = _TwoLayer()
    torch.manual_seed(2)
    x = torch.randn(2, 4)
    left = _capture(model, x)
    right = _capture(model, x)

    tl.save(left, str(tmp_path / "a.tlspec"), level="runnable")
    tl.save(right, str(tmp_path / "b.tlspec"), level="runnable")
    loaded = tl.bundle(
        {
            "a": tl.load(str(tmp_path / "a.tlspec")),
            "b": tl.load(str(tmp_path / "b.tlspec")),
        }
    )
    with pytest.raises(BundleRelationshipError) as excinfo:
        loaded.compare_at(tl.func("relu"))
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
    live = tl.bundle(dict(captures))
    loaded = tl.bundle(
        {name: _round_trip(trace, tmp_path / f"{name}.tlspec") for name, trace in captures.items()}
    )

    # The collision premise the audit exists for: persisted object ids are
    # NOT unique across loaded artifacts.
    assert loaded["m1x1"].model_object_id == loaded["m2x1"].model_object_id

    identity_ranks = {Relationship.SAME_OBJECT, Relationship.SAME_MODEL_OBJECT_AT_CAPTURE}
    names = list(captures)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            live_rel = live.relationship(left_name, right_name)
            loaded_rel = loaded.relationship(left_name, right_name)
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
    assert live.relationship("m1x1", "m1x2") is Relationship.SAME_OBJECT
    assert loaded.relationship("m1x1", "m1x2") is Relationship.SHARED_GRAPH_SAME_INPUT


def test_topology_refusal_survives_bundle_round_trip(tmp_path: Path) -> None:
    """The ordering-topology refusal fires identically on a loaded Bundle.

    Inputs are identical across members so the topology code is provably the
    refusal reason on both sides of the artifact door (test law 3).
    """

    model = _TwoLayer()
    torch.manual_seed(4)
    x = torch.randn(2, 4)
    bundle = tl.bundle({"turn1": _capture(model, x), "turn2": _capture(model, x)})
    bundle.relate({"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}})

    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.diff_pair("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"

    path = tmp_path / "chain.tlspec"
    bundle.save(str(path))
    loaded = tl.load(str(path))

    with pytest.raises(BundleRelationshipError) as excinfo:
        loaded.diff_pair("turn1", "turn2")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]


def test_scoped_pair_read_works_after_bundle_round_trip(tmp_path: Path) -> None:
    """Item 3 after the artifact door: the clean pair still reads on a loaded
    Bundle that also carries an ordered member."""

    model = _TwoLayer()
    torch.manual_seed(5)
    x = torch.randn(2, 4)
    bundle = tl.bundle(
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

    rows = loaded.diff_pair("a", "b")
    assert isinstance(rows, list)
    with pytest.raises(BundleRelationshipError) as excinfo:
        loaded.diff_pair("b", "next")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
