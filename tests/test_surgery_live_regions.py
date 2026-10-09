"""Lane F01 regions checkpoint: derived admissibility on the existing verbs.

Surgery memo 3.4 / foldA D12: a region is a user-selected contiguous chunk of
the executed graph treated as ONE unit. Covered here: derived complete exits
(user list leak refusal), convexity (``between(R, R) <= R`` with an offending
path printed), pass-instance partition with ONE replacement evaluation per
instance and atomic multi-exit commit, replay-only delegated effect closure
(buffer-write evidence, eval-mode BatchNorm allowed per the D18 evidence
class), container-consumed exit extension, ``splice_module`` lowering to the
region path, the engine-set ``execution_effect``
(``exits_substituted_interior_not_replayed``) on the renderer-neutral fact
row, save/load round-trip, and validation passing unchanged on a
region-intervened fork (the tripwire stays armed). Real-model gate rows:
GPT-2 (distilgpt2) and ResNet-50 block regions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import RegionError
from torchlens.intervention.regions import RegionTarget, region

HF_HUB_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"

_CAPTURE = tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3: the linear-chain region substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


class _TwoHead(nn.Module):
    """Shared stem feeding two heads: the multi-exit region substrate."""

    def __init__(self) -> None:
        """Build stem + two heads."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.h1 = nn.Linear(4, 2)
        self.h2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concat the two sigmoid heads (region exits feed cat's list arg)."""

        s = torch.relu(self.stem(x))
        return torch.cat([torch.sigmoid(self.h1(s)), torch.sigmoid(self.h2(s))], dim=1)


class _SharedBlock(nn.Module):
    """Calls the SAME submodule twice: the recurrent/shared region substrate."""

    def __init__(self) -> None:
        """Build one shared linear."""

        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two passes of the shared block with a relu between."""

        return self.block(torch.relu(self.block(x)))


class _BN(nn.Module):
    """Linear -> BatchNorm -> Linear: the effect-closure substrate."""

    def __init__(self) -> None:
        """Build the pieces."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear -> bn -> linear."""

        return self.head(self.bn(self.fc(x)))


@pytest.fixture()
def chain_fork():
    """A fork of an intervention-ready chain capture plus the model/input."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    log = tl.trace(model, x, capture=_CAPTURE)
    return model, x, log.fork()


def _region_fact_rows(trace) -> list[dict]:
    """The renderer-neutral region fact rows on the free-form stream."""

    return [row for row in trace.state_history if row.get("op") == "region_do"]


# ---------------------------------------------------------------------------
# derivation: boundary, instances, admissibility refusals
# ---------------------------------------------------------------------------


def test_region_derivation_boundary_and_instances(chain_fork) -> None:
    """A chain sub-region derives its typed boundary and one instance."""

    _model, _x, fork = chain_fork
    slice_ = fork.between(fork["linear_1_1"], fork["tanh_1_4"])
    target = slice_.as_region()
    assert isinstance(target, RegionTarget)
    assert target.members == ("linear_1_1:1", "relu_1_2:1", "linear_2_3:1", "tanh_1_4:1")
    (exit_edge,) = target.boundary.exits
    assert (exit_edge.parent, exit_edge.child) == ("tanh_1_4:1", "linear_3_5:1")
    assert exit_edge.addresses, "exit edges resolve pass-exact occurrence addresses"
    (entry_edge,) = target.boundary.entries
    assert entry_edge.child == "linear_1_1:1"
    assert len(target.instances) == 1
    assert target.instances[0].exit_ops == ("tanh_1_4:1",)
    assert target.region_digest.startswith("region-")
    assert "interior not replayed" in repr(target)


def test_region_not_convex_refuses_with_path(chain_fork) -> None:
    """Members whose connecting path exits and re-enters refuse convexity."""

    _model, _x, fork = chain_fork
    gappy = fork.subgraph(fork["linear_1_1"].__selection__() | fork["tanh_1_4"].__selection__())
    with pytest.raises(RegionError) as excinfo:
        gappy.as_region()
    assert excinfo.value.fields["code"] == "region_not_convex"
    assert "->" in str(excinfo.value)  # the offending path is printed


def test_region_exit_list_checked_never_trusted(chain_fork) -> None:
    """User exit lists are validated: unknown entries and leaks refuse."""

    _model, _x, fork = chain_fork
    slice_ = fork.between(fork["linear_1_1"], fork["tanh_1_4"])
    # unknown: an internal edge is not an exit
    with pytest.raises(RegionError) as excinfo:
        slice_.as_region(exits=[("relu_1_2:1", "linear_2_3:1")])
    assert excinfo.value.fields["code"] == "region_exits_unknown"
    # complete: the derived set passes
    complete = slice_.as_region(exits=[("tanh_1_4:1", "linear_3_5:1")])
    assert len(complete.boundary.exits) == 1


def test_region_exit_leak_named(chain_fork) -> None:
    """A declared exit list missing a derived exit refuses, naming the leak."""

    _model, x, _fork = chain_fork
    torch.manual_seed(0)
    model = _TwoHead().eval()
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    slice_ = fork.subgraph(
        fork["relu_1_2"].__selection__()
        | fork["linear_2_3"].__selection__()
        | fork["linear_3_5"].__selection__()
    )
    with pytest.raises(RegionError) as excinfo:
        slice_.as_region(exits=[("linear_2_3:1", "sigmoid_1_4:1")])
    assert excinfo.value.fields["code"] == "region_exits_incomplete"
    assert "linear_3_5:1" in str(excinfo.value)  # the leaked edge is named


def test_region_construction_refusals(chain_fork) -> None:
    """Non-slice targets and empty slices refuse typed."""

    _model, _x, fork = chain_fork
    with pytest.raises(RegionError) as excinfo:
        region("not a slice")
    assert excinfo.value.fields["code"] == "region_target_invalid"
    empty = fork.between(fork["tanh_1_4"], fork["linear_1_1"])  # no path
    assert empty.empty
    with pytest.raises(RegionError) as excinfo:
        empty.as_region()
    assert excinfo.value.fields["code"] == "region_empty"


# ---------------------------------------------------------------------------
# the replay lowering: substitution semantics + audit facts
# ---------------------------------------------------------------------------


def test_region_do_substitutes_exits_interior_not_replayed(chain_fork) -> None:
    """The region edit lands at the exit; downstream recomputes; facts ride."""

    model, x, fork = chain_fork
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    fork.do(target, tl.zero_ablate())
    assert torch.allclose(fork["linear_3_5"].out, model.fc3(torch.zeros(3, 4)), atol=1e-6)
    (fact,) = _region_fact_rows(fork)
    assert fact["execution_effect"] == "exits_substituted_interior_not_replayed"
    assert fact["instances"] == [
        {"instance": 1, "members": 4, "exit_ops": ["tanh_1_4:1"], "evaluations": 1}
    ]
    assert fact["effect_closure"]["certified_channels"] == [
        "buffer_writes",
        "collectives",
        "inplace_ops",
    ]
    assert "not certified" in fact["effect_closure"]["note"]  # never "effect-free"
    assert fact["source"] == "current_transaction"
    # the do() transaction envelope landed through the one chokepoint
    assert any(row.get("kind") == "ACT" for row in fork.intervention_audit)
    # interior values are capture truth, not replayed: relu output unchanged
    assert torch.allclose(fork["relu_1_2"].out, torch.relu(model.fc1(x)), atol=1e-6)


def test_region_do_validation_tripwire_unchanged(chain_fork) -> None:
    """Validation gives the corroborated boundary verdict, never a skip."""

    from torchlens.validation.core import _check_edge_intervention_boundary

    _model, _x, fork = chain_fork
    fork.do(fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region(), tl.zero_ablate())
    verdict = _check_edge_intervention_boundary(fork, fork["linear_3_5"].ops[0])
    assert verdict is not None
    assert verdict.decision == "edge_intervention_boundary"
    assert not verdict.failed


def test_region_do_round_trips(chain_fork, tmp_path) -> None:
    """A region-intervened fork saves and loads with its fact row intact."""

    _model, _x, fork = chain_fork
    fork.do(fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region(), tl.zero_ablate())
    path = str(tmp_path / "region.tlspec")
    tl.save(fork, path)
    loaded = tl.load(path)
    (fact,) = _region_fact_rows(loaded)
    assert fact["execution_effect"] == "exits_substituted_interior_not_replayed"


def test_region_splice_module_lowering(chain_fork) -> None:
    """splice_module(input='in') maps region ENTRY values to exit values."""

    model, _x, fork = chain_fork

    class _Sub(nn.Module):
        """Replacement block: constant twos of the entry's shape."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Return twos like the entry."""

            return torch.ones_like(value) * 2.0

    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    fork.do(target, tl.splice_module(_Sub(), input="in"))
    assert torch.allclose(fork["linear_3_5"].out, model.fc3(torch.full((3, 4), 2.0)), atol=1e-6)
    (fact,) = _region_fact_rows(fork)
    assert fact["edit"] == "splice_module"


def test_region_multi_exit_atomic_once_per_instance() -> None:
    """Multi-exit instances take ONE evaluation; helpers refuse typed."""

    torch.manual_seed(0)
    model = _TwoHead().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    members = (
        fork["relu_1_2"].__selection__()
        | fork["linear_2_3"].__selection__()
        | fork["linear_3_5"].__selection__()
    )
    target = fork.subgraph(members).as_region()
    assert len(target.instances) == 1
    assert set(target.instances[0].exit_ops) == {"linear_2_3:1", "linear_3_5:1"}
    with pytest.raises(RegionError) as excinfo:
        fork.do(target, tl.zero_ablate())
    assert excinfo.value.fields["code"] == "region_edit_multi_exit_unsupported"

    calls = {"n": 0}

    def _region_edit(outs: tuple, *, hook) -> tuple:
        """Zero both exits from ONE coordinated evaluation."""

        calls["n"] += 1
        return tuple(torch.zeros_like(out) for out in outs)

    fork2 = tl.trace(model, x, capture=_CAPTURE).fork()
    target2 = fork2.subgraph(
        fork2["relu_1_2"].__selection__()
        | fork2["linear_2_3"].__selection__()
        | fork2["linear_3_5"].__selection__()
    ).as_region()
    fork2.do(target2, _region_edit)
    assert calls["n"] == 1  # once per pass instance, never per exit
    expected = torch.cat([torch.sigmoid(torch.zeros(3, 2)), torch.sigmoid(torch.zeros(3, 2))], 1)
    assert torch.allclose(fork2["cat_1_7"].out, expected, atol=1e-6)


def test_region_stochastic_once_per_pass() -> None:
    """A stochastic region edit draws once per instance, reused at each exit."""

    torch.manual_seed(0)
    model = _TwoHead().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    draws: list[torch.Tensor] = []

    def _stochastic(outs: tuple, *, hook) -> tuple:
        """One draw per evaluation, applied to every exit."""

        draw = torch.randn(())
        draws.append(draw)
        return tuple(out * 0.0 + draw for out in outs)

    target = fork.subgraph(
        fork["relu_1_2"].__selection__()
        | fork["linear_2_3"].__selection__()
        | fork["linear_3_5"].__selection__()
    ).as_region()
    fork.do(target, _stochastic)
    assert len(draws) == 1
    # both heads saw the SAME draw
    left = fork["sigmoid_1_4"].out
    right = fork["sigmoid_2_6"].out
    assert torch.allclose(left, right, atol=1e-6)


def test_region_container_consumed_exit_extends() -> None:
    """Exit values consumed inside cat's LIST argument splice at nested paths."""

    torch.manual_seed(0)
    model = _TwoHead().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    # region up to the sigmoids: exits feed torch.cat's list argument
    members = (
        fork["relu_1_2"].__selection__()
        | fork["linear_2_3"].__selection__()
        | fork["linear_3_5"].__selection__()
        | fork["sigmoid_1_4"].__selection__()
        | fork["sigmoid_2_6"].__selection__()
    )
    target = fork.subgraph(members).as_region()
    assert all(edge.child == "cat_1_7:1" for edge in target.boundary.exits)
    assert any(edge.nested for edge in target.boundary.exits), "cat consumes inside a list"

    def _halves(outs: tuple, *, hook) -> tuple:
        """Replace both exits with constant halves."""

        return tuple(torch.full_like(out, 0.5) for out in outs)

    fork.do(target, _halves)
    assert torch.allclose(fork["cat_1_7"].out, torch.full((3, 4), 0.5), atol=1e-6)


def test_region_shared_block_pass_instances() -> None:
    """A region over both passes of a shared block partitions per traversal."""

    torch.manual_seed(0)
    model = _SharedBlock().eval()
    x = torch.randn(2, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    # pass-1 only: pass-qualified endpoints, one instance
    single = fork.between(fork["linear_1_1:1"], fork["linear_1_1:1"]).as_region()
    assert single.members == ("linear_1_1:1",)
    assert len(single.instances) == 1
    fork.do(single, tl.zero_ablate())
    # pass 1 zeroed -> relu(0) = 0 -> pass 2 = bias-only forward
    expected = model.block(torch.relu(torch.zeros(2, 4)))
    assert torch.allclose(fork["linear_1_1:2"].out, expected, atol=1e-6)
    (fact,) = _region_fact_rows(fork)
    assert fact["instances"][0]["exit_ops"] == ["linear_1_1:1"]


def test_region_disconnected_members_evaluate_per_instance() -> None:
    """Disconnected member groups form separate instances, one draw each."""

    torch.manual_seed(0)
    model = _TwoHead().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    # the two head linears are disconnected inside the region
    target = fork.subgraph(
        fork["linear_2_3"].__selection__() | fork["linear_3_5"].__selection__()
    ).as_region()
    assert len(target.instances) == 2
    calls = {"n": 0}

    def _edit(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Count evaluations; zero the exit."""

        calls["n"] += 1
        return torch.zeros_like(out)

    fork.do(target, _edit)
    assert calls["n"] == 2  # one evaluation PER instance


# ---------------------------------------------------------------------------
# effect closure (replay lane only, by delegation)
# ---------------------------------------------------------------------------


def test_region_effect_closure_refuses_train_mode_bn() -> None:
    """A value-changing buffer write inside the region refuses by evidence."""

    import warnings

    torch.manual_seed(0)
    model = _BN().train()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fork = tl.trace(model, x, capture=_CAPTURE).fork()
    bn_label = next(op.layer_label for op in fork.layer_list if op.layer_type == "batchnorm")
    target = fork.between(fork["linear_1_1"], fork[bn_label]).as_region()
    with pytest.raises(RegionError) as excinfo:
        fork.do(target, tl.zero_ablate())
    assert excinfo.value.fields["code"] == "region_effect_closure_violated"
    assert "buffer_writes" in str(excinfo.value)  # the channel is named


def test_region_effect_closure_allows_eval_mode_bn() -> None:
    """Value-unchanged buffer sinks (eval BN) are disclosed non-exits (D18)."""

    torch.manual_seed(0)
    model = _BN().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    bn_label = next(op.layer_label for op in fork.layer_list if op.layer_type == "batchnorm")
    target = fork.between(fork["linear_1_1"], fork[bn_label]).as_region()
    assert [(e.parent, e.child) for e in target.boundary.exits] == [
        (f"{bn_label}:1", "linear_2_3:1")
    ]
    fork.do(target, tl.zero_ablate())
    assert torch.allclose(fork["linear_2_3"].out, model.head(torch.zeros(3, 4)), atol=1e-6)


def test_region_engine_and_trace_refusals(chain_fork) -> None:
    """Off-replay engines and foreign traces refuse typed."""

    _model, _x, fork = chain_fork
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    with pytest.raises(RegionError) as excinfo:
        fork.do(
            target, tl.zero_ablate(), intervention=tl.options.InterventionOptions(engine="set_only")
        )
    assert excinfo.value.fields["code"] == "region_engine_unsupported"
    other = fork.fork()
    with pytest.raises(RegionError) as excinfo:
        other.do(target, tl.zero_ablate())
    assert excinfo.value.fields["code"] == "region_trace_mismatch"
    with pytest.raises(RegionError) as excinfo:
        fork.do(target, None)
    assert excinfo.value.fields["code"] == "region_edit_invalid"


# ---------------------------------------------------------------------------
# GATE rows: GPT-2 and ResNet-50 block regions
# ---------------------------------------------------------------------------


def _distilgpt2_block_fork(*, use_cache: bool) -> tuple[object, object]:
    """Capture the cached distilgpt2 snapshot and select its ``transformer.h.2`` block region.

    Parameters
    ----------
    use_cache:
        Value for ``model.config.use_cache`` before capture.

    Returns
    -------
    tuple[object, object]
        The intervention-ready fork and the block's region target.
    """

    transformers = pytest.importorskip("transformers")
    if not (HF_HUB_CACHE / "models--distilgpt2").exists():
        pytest.skip("distilgpt2 snapshot not cached; fetch once online.")
    model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    model.config.use_cache = use_cache
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    fork = tl.trace(model, ids, capture=_CAPTURE).fork()
    slice_ = fork.subgraph(tl.in_module("transformer.h.2"))
    assert len(slice_) > 10  # a real block, not a single op
    return fork, slice_.as_region()


@pytest.mark.heavy
@pytest.mark.real_model
def test_region_gpt2_block_module_aligned() -> None:
    """GATE row: a module-aligned distilgpt2 transformer-block region (KV cache off).

    With the cache off, every exit of the block feeds a later op, so the block region
    applies. The cache-on form, whose exits feed the returned ``past_key_values``, is
    the refusal row below.
    """

    fork, target = _distilgpt2_block_fork(use_cache=False)
    assert len(target.instances) >= 1
    total_exit_edges = len(target.boundary.exits)
    assert total_exit_edges >= 1
    fork.do(target, tl.scale(0.0)) if all(
        len(i.exit_ops) == 1 for i in target.instances
    ) else fork.do(
        target,
        lambda outs, *, hook: tuple(torch.zeros_like(out) for out in outs),
    )
    (fact,) = _region_fact_rows(fork)
    assert fact["execution_effect"] == "exits_substituted_interior_not_replayed"


@pytest.mark.heavy
@pytest.mark.real_model
def test_region_gpt2_block_with_kv_cache_refuses_output_alias_exit() -> None:
    """GATE row, cache on: the block's exits into returned cache tensors refuse by name.

    With ``use_cache`` the block's present key/value are returned in
    ``past_key_values``: exits into model-output aliases, which run no function and so
    have no argument occurrence to splice. The edit must refuse
    ``region_exit_address_underivable`` naming the alias, and leave the fork untouched
    (no region fact row, every output value unchanged) rather than drop the edit.
    """

    import re

    fork, target = _distilgpt2_block_fork(use_cache=True)
    output_labels = {op.layer_label for op in fork.output_ops}
    # Exit edges name pass-qualified children (``output_6:1``); output ops carry bare labels.
    alias_exits = {
        edge.child for edge in target.boundary.exits if edge.child.split(":")[0] in output_labels
    }
    assert alias_exits, "with the cache on the block must exit into a returned value"
    before = [op.out.clone() for op in fork.output_ops]
    with pytest.raises(RegionError) as excinfo:
        fork.do(target, tl.scale(0.0))
    assert excinfo.value.fields["code"] == "region_exit_address_underivable"
    message = str(excinfo.value)
    assert "model-output alias" in message
    named = re.search(r"alias node\(s\) \[([^\]]*)\]", message)
    assert named is not None, message
    named_labels = {label.strip().strip("'\"") for label in named.group(1).split(",")}
    assert named_labels == alias_exits
    # Nothing was silently applied: no region transaction, outputs bit-identical.
    assert _region_fact_rows(fork) == []
    after = [op.out for op in fork.output_ops]
    assert len(after) == len(before)
    assert all(torch.equal(a, b) for a, b in zip(after, before, strict=True))


@pytest.mark.heavy
@pytest.mark.real_model
def test_region_resnet50_block_module_aligned() -> None:
    """GATE row: a module-aligned ResNet-50 bottleneck region (random init)."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet50(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    slice_ = fork.subgraph(tl.in_module("layer2.0"))
    target = slice_.as_region()
    assert len(target.members) > 5
    edit = (
        tl.scale(0.0)
        if all(len(i.exit_ops) == 1 for i in target.instances)
        else (lambda outs, *, hook: tuple(torch.zeros_like(out) for out in outs))
    )
    fork.do(target, edit)
    (fact,) = _region_fact_rows(fork)
    assert fact["execution_effect"] == "exits_substituted_interior_not_replayed"
    assert fact["effect_closure"]["certified_channels"]


def test_region_exit_address_underivable_refuses(chain_fork) -> None:
    """An exit edge with no derivable occurrence address refuses at do()."""

    import dataclasses

    from torchlens.intervention.regions import RegionBoundary

    _model, _x, fork = chain_fork
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    stripped = tuple(
        dataclasses.replace(edge, addresses=(), nested=False) for edge in target.boundary.exits
    )
    doctored = RegionTarget(
        target.source_slice,
        RegionBoundary(entries=target.boundary.entries, exits=stripped),
        target.instances,
    )
    with pytest.raises(RegionError) as excinfo:
        fork.do(doctored, tl.zero_ablate())
    assert excinfo.value.fields["code"] == "region_exit_address_underivable"
    assert "tanh_1_4:1" in str(excinfo.value)  # the crossing is named


def test_region_exit_container_unsupported_refuses() -> None:
    """The nested splice refuses container kinds it cannot rebuild."""

    from torchlens.intervention.regions import _replace_at_path

    with pytest.raises(RegionError) as excinfo:
        _replace_at_path(frozenset({1}), (0,), torch.zeros(1))
    assert excinfo.value.fields["code"] == "region_exit_container_unsupported"


def test_region_splice_arity_mismatch_refuses() -> None:
    """splice_module returning too few values for a multi-exit region refuses."""

    torch.manual_seed(0)
    model = _TwoHead().eval()
    x = torch.randn(3, 4)
    fork = tl.trace(model, x, capture=_CAPTURE).fork()
    target = fork.subgraph(
        fork["relu_1_2"].__selection__()
        | fork["linear_2_3"].__selection__()
        | fork["linear_3_5"].__selection__()
    ).as_region()
    assert len(target.instances[0].exit_ops) == 2

    class _OneOut(nn.Module):
        """Replacement returning ONE tensor for a TWO-exit region."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Return one tensor."""

            return torch.zeros_like(value)

    with pytest.raises(RegionError) as excinfo:
        fork.do(target, tl.splice_module(_OneOut(), input="in"))
    assert excinfo.value.fields["code"] == "region_splice_arity_mismatch"
