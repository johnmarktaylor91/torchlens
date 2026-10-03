"""F02 ``tl.compose``: the explicit value algebra (edits memo D20-D22; row A5).

Left-to-right order against hand oracles, syntactic flattening as audit-visible
identity, non-commutativity, the construction refusal roster, one FireRecord
with ordered leaf identities, AND-combined flags, pickle round trip through the
builtin registry preserving leaf order, and byte comparison against explicit
sequential hooks.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import compose, reference, sample_from


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _toy_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _Toy(), torch.randn(5, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
    )


def test_left_to_right_order_and_noncommutativity() -> None:
    """compose(a, b)(x) == b(a(x)); scale/clamp do not commute."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    fork_lr = log.fork()
    fork_lr.do(tl.label("relu_1_2"), compose(tl.scale(2.0), tl.clamp(max=0.5)))
    fork_rl = log.fork()
    fork_rl.do(tl.label("relu_1_2"), compose(tl.clamp(max=0.5), tl.scale(2.0)))
    assert torch.equal(fork_lr["relu_1_2"].out, (baseline * 2).clamp(max=0.5))
    assert torch.equal(fork_rl["relu_1_2"].out, baseline.clamp(max=0.5) * 2)
    assert not torch.equal(fork_lr["relu_1_2"].out, fork_rl["relu_1_2"].out)


def test_flattening_is_syntactic_and_audit_visible() -> None:
    """compose(a, compose(b, c)) and compose(a, b, c) are ONE identity."""

    nested = compose(tl.scale(2.0), compose(tl.clamp(max=0.5), tl.scale(0.5)))
    flat = compose(tl.scale(2.0), tl.clamp(max=0.5), tl.scale(0.5))
    assert nested == flat
    assert [leaf.helper_name for leaf in nested.args] == ["scale", "clamp", "scale"]
    assert dict(nested.metadata)["compose_leaves"] == "scale|clamp|scale"


def test_construction_refusals() -> None:
    """D20/D21: empty, non-helper, unseeded-stochastic, shape-change, backward."""

    with pytest.raises(Exception) as excinfo:
        compose()
    assert excinfo.value.fields["code"] == "compose_empty"
    with pytest.raises(Exception) as excinfo:
        compose(lambda out, *, hook: out)
    assert excinfo.value.fields["code"] == "compose_leaf_invalid"
    with pytest.raises(Exception) as excinfo:
        compose(tl.scale(2.0), tl.noise(0.1))
    assert excinfo.value.fields["code"] == "compose_unseeded_stochastic"
    with pytest.raises(Exception) as excinfo:
        compose(tl.scale(2.0, force_shape_change=True))
    assert excinfo.value.fields["code"] == "compose_shape_change_unsupported"
    with pytest.raises(Exception) as excinfo:
        compose(tl.scale(2.0), tl.grad_zero())
    assert excinfo.value.fields["code"] == "compose_kind_mismatch"


def test_one_fire_one_record_with_ordered_leaves() -> None:
    """ONE FireRecord per firing; the ordered leaf identities ride the spec."""

    log = _toy_trace()
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), compose(tl.scale(3.0), tl.clamp(max=1.0)))
    fires = list(fork.ops["relu_1_2"].interventions or ())
    assert len(fires) == 1, "one composed fire, never one record per leaf"
    assert fires[0].helper is not None
    assert fires[0].helper.helper_name == "compose"
    assert dict(fires[0].helper.metadata)["compose_leaves"] == "scale|clamp"


def test_flags_and_combine_and_seeded_stochastic_leaf_allowed() -> None:
    """Flags AND-combine; a SEEDED stochastic leaf composes (granularity mixing legal)."""

    seeded = compose(tl.scale(2.0), tl.noise(0.1, seed=7))
    assert [leaf.helper_name for leaf in seeded.args] == ["scale", "noise"]
    assert seeded.batch_independent is True
    baseline_flags = compose(tl.scale(2.0), tl.mean_ablate())
    assert baseline_flags.batch_independent is False, "mean_ablate self-mean couples the batch"


def test_matches_explicit_sequential_hooks_bytewise() -> None:
    """A5: the composed value equals explicit sequential edits, byte for byte."""

    log = _toy_trace()
    fork_composed = log.fork()
    fork_composed.do(tl.label("relu_1_2"), compose(tl.scale(2.0), tl.clamp(max=0.5)))
    fork_sequential = log.fork()
    fork_sequential.do(tl.label("relu_1_2"), tl.scale(2.0))
    fork_sequential.do(tl.label("relu_1_2"), tl.clamp(max=0.5))
    assert torch.equal(fork_composed["relu_1_2"].out, fork_sequential["relu_1_2"].out)


def test_pickle_round_trip_preserves_leaf_order() -> None:
    """The builtin registry rebuilds the chain with leaf order intact."""

    spec = compose(tl.scale(2.0), tl.clamp(max=0.5), tl.scale(0.25))
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    assert [leaf.helper_name for leaf in restored.args] == ["scale", "clamp", "scale"]
    assert restored.factory is not None, "the registry re-derives the runtime factory"


def test_leaf_failure_is_atomic_and_named() -> None:
    """A failing leaf aborts the whole fire naming compose[i]."""

    def _boom() -> object:
        raise RuntimeError("leaf exploded")

    exploding = tl.scale(2.0)
    log = _toy_trace()
    fork = log.fork()
    chain = compose(exploding, tl.clamp(max=0.5))
    # Sabotage leaf 0's hook after construction to prove fire-time atomicity.
    object.__setattr__(chain.args[0], "factory", lambda: lambda out, *, hook: _boom())
    baseline = fork["relu_1_2"].out.clone()
    with pytest.raises(Exception) as excinfo:
        fork.do(tl.label("relu_1_2"), chain)
    assert excinfo.value.fields["code"] == "compose_leaf_failed"
    assert excinfo.value.fields["leaf_index"] == 0
    assert torch.equal(fork["relu_1_2"].out, baseline), "no half-applied chain"


def test_compose_threads_leaf_path_into_derived_seeds() -> None:
    """D6: two stochastic leaves inside one compose cannot collide."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline + 1, baseline + 2, baseline + 3]), origin="d")
    plan_a = sample_from(donors, seed=5)
    plan_b = sample_from(donors, seed=5)
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), compose(tl.patch_from(plan_a), tl.patch_from(plan_b)))
    from torchlens.intervention import sampling_records

    records = sampling_records(fork)
    assert len(records) == 2
    assert records[0]["derived_seed"] != records[1]["derived_seed"], (
        "distinct donor groups + leaf paths separate the draws"
    )
