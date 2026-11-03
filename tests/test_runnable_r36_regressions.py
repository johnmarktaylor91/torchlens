"""Round-36 regression harness: every r36 witness/exec finding pinned at the class level.

Each test pins the AGREED post-r37 contract (round-37 cross-lab plan):

* hon2_1/hon2_2/hon2_3/corr2-1 -- INV-1 single-exit escape/event attribution: no
  value-collision, no ``.item()`` re-extraction, no autograd-purity argument can
  discharge an escape; every dispatch outcome has exactly one disposition.
* hon1_1/corr2-4/corr2-2 -- INV-2 absolute-interval alias engine: cross-storage
  overlap is caught, overlapping bound state refuses at save
  (``state_alias_topology_unsupported``), enumeration is device-independent.
* hon1_2 -- host-RNG conservative ceiling over the frozen channel vocabulary.
* corr2-5 -- live-provider verdicts route through the shared poison spine.
* hon1_3/corr2-8/corr2-3/corr2-7/corr2-6/hon1_4 -- execution fidelity: lazy CUDA
  staging + readiness, scoped default-device restore, requires_grad-preserving
  clones, phase-0 subclass admission, inference-mode capture.
* secB_1/corr1_1 -- INV-3 output capability table + zero-tensor-slot save refusal
  (``output_container_contract_missing``).
* INV-4 -- closed-vocabulary context parsing (``context_field_invalid``).

Fail-closed pins land red on b1a18687 and green after the staged fixes; the
recovery pins (pure-param escapes, original-input ``.data`` escapes) go green at
stage 4 (positive origin resolution) and guard against over-trigger regressions.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import warnings
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)
from torchlens.backends.torch import rescue
from torchlens.errors import (
    PathDivergenceError,
    PoisonedRunError,
    RunnablePreflightError,
    RunnableTLSPECError,
)
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
    seed=0,
)


def _save_load(
    model: nn.Module,
    cap_inputs: Any,
    tmp: Path,
    *,
    weights: bool = True,
    acts: bool = False,
    name: str = "r36.tlspec",
) -> tl.Trace:
    """Capture, save a runnable ``.tlspec``, and reload it."""

    trace = tl.trace(model, cap_inputs, capture=_CAPTURE)
    path = tmp / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=weights, include_activations=acts)
    return tl.load(path)


def _verdict(loaded: tl.Trace, inputs: Any, **kwargs: Any) -> tuple[str, str]:
    """Run with ``on_divergence="return"`` and return (path_faithfulness, attestation) values."""

    result = loaded.run(inputs=inputs, on_divergence="return", **kwargs)
    return (
        result.report.path_faithfulness.value,
        result.report.numeric_attestation.value,
    )


# ======================================================================================
# hon2_1 -- multi-element unlabelled escape source of aten.equal / aten.allclose
# ======================================================================================


class _EqualDataBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.register_buffer("ref", torch.full((4,), 0.5))


class _TorchEqualData(_EqualDataBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * 2.0 if torch.equal(x.data, self.ref) else h * 3.0


class _MethodEqualData(_EqualDataBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * 2.0 if x.data.equal(self.ref) else h * 3.0


class _AllcloseData(_EqualDataBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * 2.0 if torch.allclose(x.data, self.ref) else h * 3.0


class _TorchEqualDataSwapped(_EqualDataBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * 2.0 if torch.equal(self.ref, x.data) else h * 3.0


class _EqualLeafControl(_EqualDataBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * 2.0 if torch.equal(x, self.ref) else h * 3.0


_HON2_1_PROBES = (
    ("torch_equal_data", _TorchEqualData),
    ("method_equal_data", _MethodEqualData),
    ("allclose_data", _AllcloseData),
    ("torch_equal_data_swapped", _TorchEqualDataSwapped),
)


class TestHon21MultiElementEscape:
    """A multi-element ``.data`` operand of equal/allclose can never verify a changed input."""

    # Kept: shares a parametrize list that is not trimmable per id with kept params.
    @pytest.mark.parametrize(("label", "model_cls"), _HON2_1_PROBES[:3])
    def test_original_input_data_escape_recovers_verified(
        self, label: str, model_cls: type[nn.Module], tmp_path: Path
    ) -> None:
        """Stage-4 recovery: a ``.data``-of-input escape attributes to the input slot,
        so the ORIGINAL input re-digests identically and stays VERIFIED."""

        torch.manual_seed(0)
        x0 = torch.full((4,), 0.5)
        loaded = _save_load(model_cls().eval(), x0, tmp_path)
        verdict, _ = _verdict(loaded, x0.clone())
        assert verdict == "verified", label


# ======================================================================================
# hon2_2 -- value-collision blessing of an unattributable .data scalar escape
# ======================================================================================


class _SinkCollision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        dead = h.sum() * 0.0 + 0.5  # constant internal sink; 0.5 for ANY input
        v = x.data.item()  # unattributable escape; 0.5 at capture
        _ = dead
        return h * 2.0 if v == 0.5 else h * 3.0


class _StateCollision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 4, bias=False)
        self.register_buffer("thresh", torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x) + self.thresh * 0.0
        v = x.data.item()
        return h * 2.0 if v == 0.5 else h * 3.0


class _NoCollision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        v = x.data.item()
        return h * 2.0 if v == 0.5 else h * 3.0


class TestHon22ValueCollision:
    """Scalar value equality may add witnesses but can never discharge attribution."""

    # Kept: shares a parametrize list that is not trimmable per id with kept params.
    @pytest.mark.parametrize(
        ("label", "model_cls"),
        (
            ("sink_collision", _SinkCollision),
            ("state_collision", _StateCollision),
            ("no_collision", _NoCollision),
        ),
    )
    def test_changed_input_never_verified(
        self, label: str, model_cls: type[nn.Module], tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        loaded = _save_load(model_cls().eval(), torch.full((1,), 0.5), tmp_path)
        verdict, attest = _verdict(loaded, torch.ones(1))
        assert verdict != "verified", label
        assert attest == "not_applicable", label


# ======================================================================================
# hon2_3 -- detached-operand contamination of the param-derived escape resolution
# ======================================================================================


class _ParamEscapeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.w = nn.Parameter(torch.tensor([0.0, 1.0, 2.0, 3.0]))


def _param_escape_model(compute: Any) -> nn.Module:
    class _Model(_ParamEscapeBase):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.lin(x)
            v = compute(self, x)
            return h * 2.0 if v > 1.0 else h * 3.0

    torch.manual_seed(0)
    return _Model().eval()


_CONTAMINATED = ()


# The probes deliberately float() requires_grad reductions; torch's "converting to
# scalar" UserWarning is expected noise from the probe shape, not a torchlens signal.
@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")
class TestHon23ParamDerivedContamination:
    """No autograd-graph argument may prove operand totality (exp1 miss set pinned)."""

    # Kept: shares a parametrize list that is not trimmable per id with kept params.
    @pytest.mark.parametrize(
        ("label", "compute"),
        (
            ("w_sum", lambda m, x: float(m.w.sum())),
            ("w_times_2", lambda m, x: float((m.w * 2).sum())),
            ("w_plus_scalar", lambda m, x: float((m.w + 1.5).sum())),
        ),
        ids=("w_sum", "w_times_2", "w_plus_scalar"),
    )
    def test_pure_param_escape_recovers_verified(
        self, label: str, compute: Any, tmp_path: Path
    ) -> None:
        """Stage-4 recovery pin (exp1 over-trigger): pure param-derived host reads
        stay VERIFIED on unchanged state -- via positive origins, never autograd."""

        model = _param_escape_model(compute)
        loaded = _save_load(model, torch.full((4,), 0.5), tmp_path)
        verdict, _ = _verdict(loaded, torch.full((4,), 0.9))
        assert verdict == "verified", label


# ======================================================================================
# corr2-1 -- tensor-returning raw aten events must be dispositioned (INV-1 ledger)
# ======================================================================================


class _RawMutation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch._C.TensorBase.add_(x, 1)
        return torch.relu(x)


class _RawNonMutating(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch._C._VariableFunctions.mul(x, 2)
        return torch.relu(y)


# Raw-dispatch products feeding traced ops legitimately warn about missing provenance;
# the pinned contract here is the ledger fact + INCOMPLETE ceiling, not the warning.
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments with no graph")
class TestCorr21EventLedgerExhaustive:
    """Every dispatch outcome -- including ``returned_tensor`` -- has one disposition."""

    # Kept: sole branch coverage of _io/runnable.py, backends/torch/_completeness_dispatch.py; a sole mutation kill.
    def test_raw_nonmutating_aten_is_incomplete_never_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-closed adjacent: the non-mutating twin bakes an unlabelled constant.

        Fail-closed either way: the unattributed product feeding a traced call may be
        refused at SAVE (``tensor_constant_unsupported``), and when an artifact does
        save, the ``unmodeled_tensor_return`` ledger fact must ceiling it INCOMPLETE
        and never VERIFIED.
        """
        monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)  # keep primary capture

        from torchlens.backends.torch._completeness_dispatch import runnable_ledger_facts

        x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        trace = tl.trace(_RawNonMutating(), x.clone(), capture=_CAPTURE)
        facts = runnable_ledger_facts(trace)
        assert any(fact.get("kind") == "unmodeled_tensor_return" for fact in facts)
        path = tmp_path / "raw_nonmut.tlspec"
        try:
            trace.save(path, level="runnable", include_weights=True)
        except RunnablePreflightError as exc:
            assert "tensor_constant_unsupported" in str(exc.fields.get("diagnostics", ""))
            return
        loaded = tl.load(path)
        descriptor = loaded._sparse_run_descriptor
        assert descriptor is not None
        assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
        verdict, _ = _verdict(loaded, x.clone())
        assert verdict != "verified"

    # Kept: a sole mutation kill.
    def test_batchnorm_running_stats_stay_discharged(self, tmp_path: Path) -> None:
        """r15-C3 control: tracked BN buffer updates are owner-accounted, not facts."""

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).eval()
        x = torch.randn(3, 4)
        loaded = _save_load(model, x, tmp_path)
        descriptor = loaded._sparse_run_descriptor
        assert descriptor is not None
        assert descriptor.witness_completeness is WitnessCompleteness.COMPLETE
        verdict, _ = _verdict(loaded, x)
        assert verdict == "verified"


# ======================================================================================
# hon1_1 -- cross-storage overlapping-memory inputs (INV-2 absolute intervals)
# ======================================================================================


class _DirectMutate(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a.add_(1.0)
        return a + b


class TestHon11CrossStorageAlias:
    """Distinct torch storages over one host buffer are not disjoint by pointer."""

    def _loaded(self, tmp_path: Path) -> tl.Trace:
        return _save_load(_DirectMutate(), (torch.zeros(6), torch.ones(6)), tmp_path, acts=True)


# ======================================================================================
# corr2-4 -- overlapping bound state refuses at save (state_alias_topology_unsupported)
# ======================================================================================


class _OverlappingBuffers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.tensor([2.0, 3.0, 4.0])
        self.register_buffer("left", base[:2], persistent=False)
        self.register_buffer("right", base[1:], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.left.add_(10)
        return x + self.right


class _DisjointViewBuffers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.tensor([2.0, 3.0, 4.0, 5.0])
        self.register_buffer("left", base[:2], persistent=False)
        self.register_buffer("right", base[2:], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.left + self.right


class _TiedWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = self.emb.weight  # same live Parameter object, two names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.emb(x)))


class TestCorr24StateAliasTopology:
    """Distinct-object overlap refuses at save; live identity becomes one alias group."""

    # Kept: sole branch coverage of _io/runnable.py, backends/torch/_storage_overlap.py, backends/torch/buffer_writes.py; a sole mutation kill; the catch of its regression fault.
    def test_overlapping_nonpersistent_buffers_refuse_at_save(self, tmp_path: Path) -> None:
        x = torch.tensor([100.0, 200.0])
        trace = tl.trace(_OverlappingBuffers(), x, capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace.save(tmp_path / "overlap.tlspec", level="runnable", include_weights=True)
        assert "state_alias_topology_unsupported" in str(excinfo.value.fields.get("diagnostics"))

    # Kept: sole branch coverage of _io/bundle.py, _runnable/state.py, validation/__init__.py; a sole mutation kill.
    def test_disjoint_views_of_one_storage_still_save_and_verify(self, tmp_path: Path) -> None:
        x = torch.tensor([100.0, 200.0])
        loaded = _save_load(_DisjointViewBuffers(), x, tmp_path)
        verdict, _ = _verdict(loaded, x)
        assert verdict == "verified"


# ======================================================================================
# hon1_2 -- host-RNG channel vocabulary (conservative ceiling)
# ======================================================================================


_OUTSIDE_HELD_GEN = np.random.default_rng(7)
_PRIVATE_RANDOM = random.Random(3)


class _MyRandom(random.Random):
    pass


_SUBCLASS_RANDOM = _MyRandom(5)


def _branch_model(draw: Any) -> nn.Module:
    class _Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2 if draw() else x + 1

    return _Model()


class _ModelAttrGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng(1234)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 if self.rng.random() > 0.5 else x + 1


_RNG_CHANNELS = ()


class TestHon12HostRngChannels:
    """A touch on any named channel permanently ceilings; absence never over-triggers."""

    # Kept: sole branch coverage of _utils/rng/monitor.py.
    def test_no_draw_generator_holder_does_not_ceiling(self, tmp_path: Path) -> None:
        class _Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rng = np.random.default_rng(11)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        loaded = _save_load(_Holder(), torch.ones(4), tmp_path)
        descriptor = loaded._sparse_run_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is False

    # Kept: sole branch coverage of _runnable/providers.py, _runnable/transaction.py, _utils/rng/streams.py.
    def test_global_engine_stays_replayable(self, tmp_path: Path) -> None:
        """Global ``random`` module draws keep the seeded-reproduction semantics."""

        class _GlobalBranch(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 if random.random() > 0.5 else x + 1

        random.seed(0)
        loaded = _save_load(_GlobalBranch(), torch.ones(4), tmp_path)
        descriptor = loaded._sparse_run_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is True
        assert descriptor.rng_profile.capture_seed == 0
        verdict, _ = _verdict(loaded, torch.ones(4), seed=0)
        assert verdict == "verified"
        verdict_off, _ = _verdict(loaded, torch.ones(4))
        assert verdict_off != "verified"


# ======================================================================================
# corr2-5 -- live-provider poison routes through the shared spine
# ======================================================================================


@dataclass
class _LossyOutput:
    value: torch.Tensor
    metadata: int

    def __post_init__(self) -> None:
        self.derived = self.metadata * 2


class _LossyLiveModel(nn.Module):
    def forward(self, x: torch.Tensor) -> _LossyOutput:
        return _LossyOutput(x + 1, 7)


class _FaithfulLiveModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class TestCorr25LivePoisonSpine:
    """Live-provider verdicts derive poison from the same faithfulness lattice."""

    # Kept: sole branch coverage of _ir/container.py, _runnable/call_outputs.py, _runnable/output_contracts.py.
    def test_lossy_live_output_is_poisoned(self) -> None:
        x = torch.tensor([1.0, 2.0])
        # The live provider holds only a weakref to the source model; keep a
        # strong ref for the run (an inline temporary survives only until the
        # next GC pass, making the test GC-timing-dependent).
        model = _LossyLiveModel()
        trace = tl.trace(model, x, capture=_CAPTURE)
        result = trace.run(inputs=x, on_divergence="return")
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
        assert result.report.is_poisoned is True
        with pytest.raises(PoisonedRunError):
            result.trace.to_pandas()


# ======================================================================================
# corr2-2 -- alias enumeration is caller-device independent
# ======================================================================================


class _AddPair(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class TestCorr22DeviceIndependentEnumeration:
    """The alias proof is pure host-side integer geometry."""

    def _loaded(self, tmp_path: Path) -> tl.Trace:
        capture_base = torch.randn(4, 4)
        return _save_load(
            _AddPair(),
            (capture_base[:2, :2].clone(), capture_base[:2, 2:].clone()),
            tmp_path,
        )


# ======================================================================================
# corr2-3 -- scoped default-device restore (no DeviceContext leak)
# ======================================================================================

_CORR2_3_PRODUCE = """
import sys, torch, torchlens as tl
from torch import nn
from torchlens.options import CaptureOptions


class AddOne(nn.Module):
    def forward(self, x):
        return x + 1


x = torch.tensor([1.0, 2.0], device="cpu")
torch.set_default_device("meta")
trace = tl.trace(AddOne(), x, capture=CaptureOptions(
    replayable=True, capture_container_structure=True, cache=False))
trace.save(sys.argv[1], level="runnable")
"""

_CORR2_3_CONSUME = """
import sys, torch, torchlens as tl
from torchlens._utils._torch_compat import get_current_function_mode_stack

before = get_current_function_mode_stack()
assert before == [], before
result = tl.load(sys.argv[1]).run(inputs=torch.tensor([1.0, 2.0], device="cpu"))
after = get_current_function_mode_stack()
assert result.report.path_faithfulness.value == "verified", result.report.path_faithfulness
assert after == [], after
assert str(torch.get_default_device()) == "cpu"
print("CORR2_3_OK")
"""


# ======================================================================================
# corr2-6 -- tensor-subclass admission is typed before any property read
# ======================================================================================


class _PropertyTrapTensor(torch.Tensor):
    @classmethod
    def __torch_function__(
        cls,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del cls, func, types, args, kwargs
        raise RuntimeError("subclass property trap")


class TestCorr26SubclassAdmission:
    """Unsupported subclasses fail typed at phase 0 under both policies."""

    # Kept: sole branch coverage of _runnable/input_metadata.py.
    @pytest.mark.parametrize("policy", ("raise",))
    def test_property_trap_subclass_fails_typed(self, policy: str, tmp_path: Path) -> None:
        x = torch.randn(2, 3)
        loaded = _save_load(_AddPair(), (x, x.clone()), tmp_path, acts=True)
        exotic = torch.Tensor._make_subclass(_PropertyTrapTensor, x, False)
        with pytest.raises(PathDivergenceError) as excinfo:
            loaded.run(inputs=(exotic, x.clone()), on_divergence=policy)
        # The TYPED hard-precondition refusal, never the subclass's raw trap error.
        assert "subclass" in str(excinfo.value).lower()
        assert "property trap" not in str(excinfo.value)


# ======================================================================================
# corr2-7 -- requires_grad-preserving runtime clones
# ======================================================================================


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


# ======================================================================================
# hon1_3 / corr2-8 -- CUDA staging + readiness
# ======================================================================================


_CPU_ONLY_READINESS = """
import sys, torch, torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens._runnable import ReadinessStatus

assert not torch.cuda.is_available()
loaded = tl.load(sys.argv[1])
assert loaded.readiness is not None
assert loaded.readiness.status is not ReadinessStatus.READY, loaded.readiness.status
try:
    loaded.run(inputs=torch.randn(2, 3))
except (RunCapabilityUnavailableError, Exception) as exc:
    code = getattr(exc, "code", "")
    assert "run_capability_unavailable" in str(code) or type(exc).__name__ == "ReattachError", (
        type(exc).__name__, code)
else:
    raise AssertionError("CUDA artifact ran on a CPU-only host")
print("CPU_ONLY_OK")
"""


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCudaStagingAndReadiness:
    """hon1_3/corr2-8: transport stays lazy; run stages state to slot devices."""

    def _loaded(
        self, tmp_path: Path, *, acts: bool = True
    ) -> tuple[tl.Trace, torch.Tensor, nn.Module]:
        torch.manual_seed(0)
        model = nn.Linear(3, 2).cuda().eval()
        x = torch.randn(4, 3, device="cuda")
        loaded = _save_load(model, x, tmp_path, acts=acts)
        return loaded, x, model

    # Kept: id not measured per id (rare tier).
    def test_embedded_state_runs_verified_and_attested(self, tmp_path: Path) -> None:
        loaded, x, _ = self._loaded(tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
        assert result.output.device.type == "cuda"

    # Kept: id not measured per id (rare tier).
    def test_cpu_user_state_dict_is_staged_to_slot_device(self, tmp_path: Path) -> None:
        loaded, x, model = self._loaded(tmp_path)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        loaded.load_state_dict(cpu_state)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    # Kept: id not measured per id (rare tier).
    def test_nondeterministic_context_marker_leg(self, tmp_path: Path) -> None:
        """3-ADJ-9: the R1 attestation-ineligibility marker is exercised E2E.

        The suite conftest enables deterministic algorithms globally (under which
        CUDA conv_transpose IS deterministic and attesting is honest); the marker
        leg needs the nondeterministic ambient context the finding described.
        """

        deterministic_before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            torch.manual_seed(0)
            model = nn.ConvTranspose2d(2, 2, 3).cuda().eval()
            x = torch.randn(1, 2, 5, 5, device="cuda")
            loaded = _save_load(model, x, tmp_path, acts=True)
            descriptor = loaded._sparse_run_descriptor
            assert descriptor is not None
            assert descriptor.ambient_context.attestation_ineligible_context is True
            result = loaded.run(inputs=x, on_divergence="return")
        finally:
            torch.use_deterministic_algorithms(deterministic_before)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE

    # Kept: id not measured per id (rare tier).
    @pytest.mark.heavy
    def test_cpu_only_host_reports_readiness_unavailable(self, tmp_path: Path) -> None:
        loaded, _, _ = self._loaded(tmp_path)
        del loaded
        artifact = next(tmp_path.glob("*.tlspec"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run(
            [sys.executable, "-c", _CPU_ONLY_READINESS, str(artifact)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert proc.returncode == 0, proc.stderr
        assert "CPU_ONLY_OK" in proc.stdout


# ======================================================================================
# hon1_4 -- ambient inference-mode capture
# ======================================================================================


class _InferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class TestHon14InferenceModeCapture:
    """Ambient inference-mode capture must not crash and must round-trip."""

    # Kept: sole branch coverage of _runnable/state_context.py, _utils/_torch_compat.py, backends/torch/_completeness_patches.py and more.
    def test_runnable_e2e_under_inference_mode(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = _InferenceModel().eval()
        x = torch.randn(2, 3)
        with torch.inference_mode():
            trace = tl.trace(model, x, capture=_CAPTURE)
        path = tmp_path / "inference.tlspec"
        trace.save(path, level="runnable", include_weights=True)
        loaded = tl.load(path)
        descriptor = loaded._sparse_run_descriptor
        assert descriptor is not None
        assert all(call.execution_context.inference_mode for call in descriptor.calls)
        before_inference = torch.is_inference_mode_enabled()
        result = loaded.run(inputs=x)
        assert torch.is_inference_mode_enabled() == before_inference
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# corr1_1 / secB_1 -- INV-3 output capability table + zero-tensor refusal
# ======================================================================================


class _AllLiteralOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[dict[str, int], list[str], tuple[()]]:
        _ = x + 1
        return ({"answer": 42}, ["literal"], ())


class _LiteralRootOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> int:
        _ = x + 1
        return 42


class _EmptyContainerOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, int]:
        _ = x + 1
        return {}


_PlainNT = namedtuple("_PlainNT", ["a", "b"])


class _StatefulNT(_PlainNT):
    def __new__(cls, a: torch.Tensor, b: torch.Tensor) -> _StatefulNT:
        self = super().__new__(cls, a, b)
        self.total = a + b
        return self


class _SlottedNT(_PlainNT):
    __slots__ = ()


class _TypedNT(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor


def _nt_model(factory: Any) -> nn.Module:
    class _Model(nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            return factory(x + 1, x * 2)

    return _Model()


class TestOutputCapabilityTable:
    """secB_1 + corr1_1: one capability table; unrepresentable outputs refuse at save."""

    # Kept: sole branch coverage of _utils/introspection.py, backends/torch/_ops_containers.py; its other params stay because the list is not trimmable per id.
    @pytest.mark.parametrize(
        ("label", "model_cls"),
        (
            ("all_literal_tree", _AllLiteralOutput),
            ("literal_root", _LiteralRootOutput),
            ("empty_container", _EmptyContainerOutput),
        ),
    )
    def test_zero_tensor_slot_output_refuses_at_save(
        self, label: str, model_cls: type[nn.Module], tmp_path: Path
    ) -> None:
        trace = tl.trace(model_cls(), torch.tensor([1.0]), capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo:
            trace.save(tmp_path / "literal.tlspec", level="runnable")
        assert "output_container_contract_missing" in str(excinfo.value.fields.get("diagnostics"))

    # Kept: a sole mutation kill.
    def test_stateful_namedtuple_subclass_refuses_at_save(self, tmp_path: Path) -> None:
        trace = tl.trace(_nt_model(_StatefulNT), torch.tensor([1.0, 2.0]), capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo:
            trace.save(tmp_path / "stateful_nt.tlspec", level="runnable")
        assert "output_container_contract_missing" in str(excinfo.value.fields.get("diagnostics"))
        assert "namedtuple_instance_state" in str(excinfo.value.fields.get("diagnostics"))

    # Kept: shares a parametrize list that is not trimmable per id with kept params.
    @pytest.mark.parametrize(
        ("label", "factory"),
        (
            ("plain_namedtuple", _PlainNT),
            ("typing_namedtuple", _TypedNT),
            ("slotted_subclass", _SlottedNT),
        ),
    )
    def test_stateless_namedtuples_stay_verified(
        self, label: str, factory: Any, tmp_path: Path
    ) -> None:
        x = torch.tensor([1.0, 2.0])
        loaded = _save_load(_nt_model(factory), x, tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert type(result.output) is factory

    # Kept: a sole mutation kill.
    def test_torch_return_types_stay_verified(self, tmp_path: Path) -> None:
        class _MaxModel(nn.Module):
            def forward(self, x: torch.Tensor) -> Any:
                return torch.max(x, dim=0)

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        loaded = _save_load(_MaxModel(), x, tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    # Kept: a sole mutation kill.
    def test_tensor_output_models_still_save(self, tmp_path: Path) -> None:
        """Over-refusal pin: ordinary tensor and mixed-container outputs keep saving."""

        class _MixedOutput(nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, Any]:
                return {"y": x + 1, "meta": 7}

        x = torch.tensor([1.0])
        loaded = _save_load(_MixedOutput(), x, tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.output["meta"] == 7


# ======================================================================================
# INV-4 -- closed-vocabulary context parsing (context_field_invalid)
# ======================================================================================


class TestContextFieldValidation:
    """Persisted context bytes validate at parse time, before any torch setter."""

    def _tampered(self, tmp_path: Path, mutate: Any) -> Path:
        torch.manual_seed(0)
        model = nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        trace = tl.trace(model, x, capture=_CAPTURE)
        bundle = tmp_path / "ctx.tlspec"
        # Value-free descriptor: a weight payload would make the load hard-fail on
        # the missing parsed descriptor before readiness can carry the context
        # diagnostic (still a pre-callable refusal, but the pinned surface here is
        # the typed context_field_invalid readiness).
        trace.save(bundle, level="runnable")
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest))
        return bundle

    # Kept: sole branch coverage of _io/runnable_load.py; a sole mutation kill.
    @pytest.mark.parametrize(
        ("label", "field", "value"),
        (
            ("bad_device", "default_device", "hax; import os"),
            ("bad_matmul_precision", "float32_matmul_precision", "turbo"),
            ("bad_dtype", "default_dtype", "torch.evil"),
            ("bad_bool", "deterministic_algorithms", "yes"),
        ),
    )
    def test_invalid_ambient_field_refuses_typed(
        self, label: str, field: str, value: Any, tmp_path: Path
    ) -> None:
        def _mutate(manifest: dict[str, Any]) -> None:
            manifest["run"]["ambient_context"][field] = value

        bundle = self._tampered(tmp_path, _mutate)
        try:
            loaded = tl.load(bundle)
        except RunnableTLSPECError as exc:
            assert "context_field_invalid" in str(exc), label
            return
        readiness = loaded.readiness
        assert readiness is not None
        codes = " ".join(str(d.code.value) for d in readiness.diagnostics)
        if readiness.status.value == "ready":
            with pytest.raises(RunnableTLSPECError) as excinfo:
                loaded.run(inputs=torch.randn(2, 4))
            assert "context_field_invalid" in str(excinfo.value), label
        else:
            assert "context_field_invalid" in codes, (label, codes)


# ======================================================================================
# Class-closure meta-gates (review r37): structural guards a round-38 hunter cannot bypass
# ======================================================================================


_COUPLED_WITNESS_EXEC_FILES = (
    "torchlens/_runnable/execution.py",
    "torchlens/_runnable/attestation.py",
    "torchlens/_runnable/call_arguments.py",
    "torchlens/_runnable/call_outputs.py",
    "torchlens/_runnable/input_aliases.py",
    "torchlens/_runnable/input_metadata.py",
    "torchlens/_runnable/input_sites.py",
    "torchlens/_runnable/output_contracts.py",
    "torchlens/_runnable/path_faithfulness.py",
    "torchlens/_runnable/providers.py",
    "torchlens/_runnable/retention.py",
    "torchlens/_runnable/state_context.py",
    "torchlens/_runnable/transaction.py",
    "torchlens/_runnable/verification.py",
    "torchlens/_runnable/witness_contracts.py",
    "torchlens/_runnable/state.py",
    "torchlens/_runnable/state_live.py",
    "torchlens/_io/runnable.py",
    "torchlens/backends/torch/completeness_witness.py",
    "torchlens/backends/torch/backend.py",
    "torchlens/backends/torch/_outputs.py",
    "torchlens/backends/torch/_failed_session.py",
    "torchlens/backends/torch/ops.py",
)


class TestClassClosureMetaGates:
    """Source-scan gates: banned mechanisms cannot silently reappear."""

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]


class TestOutputCapabilityMetaGates:
    """R11 meta-guards: one table, all kinds covered, forged flags impotent."""

    # Kept: sole branch coverage of backends/torch/_ops_containers.py; a sole mutation kill.
    def test_registered_container_without_declaration_refuses_stateful_instance(
        self, tmp_path: Path
    ) -> None:
        """3-ADJ-3: undeclared registrations refuse instances with extra state."""

        from torchlens._ir.container import _CONTAINER_REGISTRY, register_container

        class _Holder:
            def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
                self.a = a
                self.b = b

        register_container(
            _Holder,
            lambda holder: ([holder.a, holder.b], None),
            lambda aux, children: _Holder(children[0], children[1]),
        )
        try:

            class _Model(nn.Module):
                def forward(self, x: torch.Tensor) -> _Holder:
                    return _Holder(x + 1, x * 2)

            trace = tl.trace(_Model(), torch.tensor([1.0, 2.0]), capture=_CAPTURE)
            with pytest.raises(RunnablePreflightError) as excinfo:
                trace.save(tmp_path / "registered.tlspec", level="runnable")
            assert "registered_container_instance_state" in str(
                excinfo.value.fields.get("diagnostics")
            )
        finally:
            _CONTAINER_REGISTRY.pop(_Holder, None)


class TestAliasEngineUnit:
    """INV-2 unit pins on the ONE shared absolute-byte relation engine."""

    @staticmethod
    def _relation(left: torch.Tensor, right: torch.Tensor) -> str:
        from torchlens._utils.alias_footprint import touched_bytes_relation

        return touched_bytes_relation(left, right)

    # Kept: sole branch coverage of _utils/alias_footprint.py.
    def test_negative_stride_via_numpy_sound(self) -> None:
        arr = np.arange(8.0, dtype=np.float32)
        reversed_view = torch.from_numpy(arr[::-1].copy())  # torch rejects neg strides
        assert self._relation(reversed_view, reversed_view) == "overlap"  # identity

    # Kept: sole branch coverage of _utils/alias_footprint.py.
    def test_mixed_element_sizes_byte_expanded(self) -> None:
        buf = bytearray(16)
        left = torch.frombuffer(buf, dtype=torch.float32, count=2, offset=0)
        right = torch.frombuffer(buf, dtype=torch.float64, count=1, offset=4)
        assert self._relation(left, right) == "overlap"
        right_disjoint = torch.frombuffer(buf, dtype=torch.float64, count=1, offset=8)
        assert self._relation(left, right_disjoint) == "disjoint"

    # Kept: sole branch coverage of _utils/alias_footprint.py.
    def test_empty_views_disjoint(self) -> None:
        base = torch.arange(4.0)
        assert self._relation(base[:0], base) == "disjoint"

    # Kept: sole branch coverage of _utils/alias_footprint.py.
    def test_meta_tensor_is_unknown(self) -> None:
        meta = torch.empty(4, device="meta")
        assert self._relation(meta, meta.clone()) == "unknown"


class TestMappingInstanceState:
    """3-ADJ-2: trusted-base mapping instances with extra attrs refuse at save."""

    # Kept: sole branch coverage of backends/torch/_ops_containers.py; a sole mutation kill.
    def test_ordereddict_with_extra_attr_refuses(self, tmp_path: Path) -> None:
        from collections import OrderedDict

        class _Model(nn.Module):
            def forward(self, x: torch.Tensor) -> Any:
                out = OrderedDict(y=x + 1)
                out.stash = x * 2  # instance state the key/value rebuild drops
                return out

        trace = tl.trace(_Model(), torch.tensor([1.0, 2.0]), capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo:
            trace.save(tmp_path / "od.tlspec", level="runnable")
        assert "mapping_instance_state" in str(excinfo.value.fields.get("diagnostics"))

    # Kept: sole branch coverage of _ir/container.py.
    def test_plain_ordereddict_stays_verified(self, tmp_path: Path) -> None:
        from collections import OrderedDict

        class _Model(nn.Module):
            def forward(self, x: torch.Tensor) -> Any:
                return OrderedDict(y=x + 1)

        x = torch.tensor([1.0, 2.0])
        loaded = _save_load(_Model(), x, tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert type(result.output) is OrderedDict
