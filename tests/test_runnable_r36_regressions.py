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
  (``missing_output_container_contract``).
* INV-4 -- closed-vocabulary context parsing (``context_field_invalid``).

Fail-closed pins land red on b1a18687 and green after the staged fixes; the
recovery pins (pure-param escapes, original-input ``.data`` escapes) go green at
stage 4 (positive origin resolution) and guard against over-trigger regressions.
"""

from __future__ import annotations

import json
import os
import random
import secrets
import subprocess
import sys
import time
import uuid
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
from torchlens.errors import (
    PathDivergenceError,
    PoisonedRunError,
    RunnablePreflightError,
    RunnableTLSPECError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
    random_seed=0,
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
    """Run with ``return_diverged`` and return (path_faithfulness, attestation) values."""

    result = loaded.run(inputs=inputs, on_divergence="return_diverged", **kwargs)
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

    @pytest.mark.parametrize(("label", "model_cls"), _HON2_1_PROBES)
    def test_changed_input_never_verified(
        self, label: str, model_cls: type[nn.Module], tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        loaded = _save_load(model_cls().eval(), torch.full((4,), 0.5), tmp_path)
        verdict, attest = _verdict(loaded, torch.ones(4))
        assert verdict != "verified", label
        assert attest == "not_applicable", label

    def test_labelled_leaf_control_stays_honest(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        loaded = _save_load(_EqualLeafControl().eval(), torch.full((4,), 0.5), tmp_path)
        verdict, _ = _verdict(loaded, torch.ones(4))
        assert verdict != "verified"

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

    def test_original_input_scalar_data_escape_recovers_verified(self, tmp_path: Path) -> None:
        """Stage-4 recovery: the ``.data`` scalar escape resolves to the input slot by
        storage identity, so the original input remains VERIFIED."""

        torch.manual_seed(0)
        x0 = torch.full((1,), 0.5)
        loaded = _save_load(_NoCollision().eval(), x0, tmp_path)
        verdict, _ = _verdict(loaded, x0.clone())
        assert verdict == "verified"


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


_CONTAMINATED = (
    ("mul_data", lambda m, x: float((m.w * x.data).sum()), 0.5, 0.05),
    (
        "masked_fill",
        lambda m, x: float(m.w.masked_fill(x.data > 0.25, 0.0).sum()),
        0.0,
        0.5,
    ),
    ("getitem_long_index", lambda m, x: float(m.w[x.data.long().clamp(0, 3)].sum()), 0.0, 3.0),
    (
        "index_select",
        lambda m, x: float(torch.index_select(m.w, 0, x.data.long().clamp(0, 3)).sum()),
        0.0,
        3.0,
    ),
    (
        "masked_select",
        lambda m, x: float(torch.masked_select(m.w, x.data > 0.25).sum()),
        0.5,
        0.0,
    ),
    (
        "where_condition",
        lambda m, x: float(torch.where(x.data > 0.25, m.w, torch.zeros(4)).sum()),
        0.5,
        0.0,
    ),
    ("detach_operand", lambda m, x: float((m.w * x.detach()).sum()), 0.5, 0.05),
)


class TestHon23ParamDerivedContamination:
    """No autograd-graph argument may prove operand totality (exp1 miss set pinned)."""

    @pytest.mark.parametrize(
        ("label", "compute", "cap_fill", "changed_fill"),
        _CONTAMINATED,
        ids=[row[0] for row in _CONTAMINATED],
    )
    def test_contaminated_escape_never_verifies_changed_input(
        self, label: str, compute: Any, cap_fill: float, changed_fill: float, tmp_path: Path
    ) -> None:
        model = _param_escape_model(compute)
        loaded = _save_load(model, torch.full((4,), cap_fill), tmp_path)
        verdict, attest = _verdict(loaded, torch.full((4,), changed_fill))
        assert verdict != "verified", label
        assert attest == "not_applicable", label

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
        torch.ops.aten.add_.Tensor(x, 1)
        return torch.relu(x)


class _RawNonMutating(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.aten.mul.Tensor(x, 2)
        return torch.relu(y)


class TestCorr21EventLedgerExhaustive:
    """Every dispatch outcome -- including ``returned_tensor`` -- has one disposition."""

    def test_raw_mutating_aten_is_incomplete_never_verified(self, tmp_path: Path) -> None:
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        loaded = _save_load(_RawMutation(), x.clone(), tmp_path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
        verdict, attest = _verdict(loaded, x.clone())
        assert verdict != "verified"
        assert attest == "not_applicable"

    def test_raw_nonmutating_aten_is_incomplete_never_verified(self, tmp_path: Path) -> None:
        """Pre-closed adjacent: the non-mutating twin bakes an unlabelled constant."""

        from torchlens.backends.torch.completeness_witness import runnable_ledger_facts

        x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        trace = tl.trace(_RawNonMutating(), x.clone(), capture=_CAPTURE)
        facts = runnable_ledger_facts(trace)
        assert any(fact.get("kind") == "unmodeled_tensor_return" for fact in facts)
        path = tmp_path / "raw_nonmut.tlspec"
        trace.save(path, level="runnable", include_weights=True)
        loaded = tl.load(path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
        verdict, _ = _verdict(loaded, x.clone())
        assert verdict != "verified"

    def test_plain_model_has_no_ledger_facts_and_verifies(self, tmp_path: Path) -> None:
        """Over-trigger pin: an ordinary capture stays COMPLETE and VERIFIED."""

        from torchlens.backends.torch.completeness_witness import runnable_ledger_facts

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
        x = torch.randn(2, 4)
        trace = tl.trace(model, x, capture=_CAPTURE)
        assert runnable_ledger_facts(trace) == ()
        path = tmp_path / "plain.tlspec"
        trace.save(path, level="runnable", include_weights=True)
        loaded = tl.load(path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.witness_completeness is WitnessCompleteness.COMPLETE
        verdict, _ = _verdict(loaded, x)
        assert verdict == "verified"

    def test_batchnorm_running_stats_stay_discharged(self, tmp_path: Path) -> None:
        """r15-C3 control: tracked BN buffer updates are owner-accounted, not facts."""

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).eval()
        x = torch.randn(3, 4)
        loaded = _save_load(model, x, tmp_path)
        descriptor = loaded.runnable_descriptor
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

    def test_from_numpy_overlap_fails_closed(self, tmp_path: Path) -> None:
        loaded = self._loaded(tmp_path)
        arr = np.arange(8.0, dtype=np.float32)
        with pytest.raises(PathDivergenceError):
            loaded.run(inputs=(torch.from_numpy(arr[:6]), torch.from_numpy(arr[2:8])))

    def test_frombuffer_overlap_fails_closed(self, tmp_path: Path) -> None:
        loaded = self._loaded(tmp_path)
        buf = bytearray(8 * 4)
        a = torch.frombuffer(buf, dtype=torch.float32, count=6, offset=0)
        b = torch.frombuffer(buf, dtype=torch.float32, count=6, offset=8)
        with pytest.raises(PathDivergenceError):
            loaded.run(inputs=(a, b))

    def test_same_storage_overlap_control(self, tmp_path: Path) -> None:
        loaded = self._loaded(tmp_path)
        base = torch.arange(8.0)
        with pytest.raises(PathDivergenceError):
            loaded.run(inputs=(base[:6], base[2:8]))

    def test_distinct_allocations_stay_verified(self, tmp_path: Path) -> None:
        """Over-trigger pin: genuinely independent inputs keep replaying fine."""

        loaded = self._loaded(tmp_path)
        verdict, _ = _verdict(loaded, (torch.zeros(6), torch.ones(6)))
        assert verdict == "verified"


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

    def test_overlapping_nonpersistent_buffers_refuse_at_save(self, tmp_path: Path) -> None:
        x = torch.tensor([100.0, 200.0])
        trace = tl.trace(_OverlappingBuffers(), x, capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace.save(tmp_path / "overlap.tlspec", level="runnable", include_weights=True)
        assert "state_alias_topology_unsupported" in str(excinfo.value)

    def test_disjoint_views_of_one_storage_still_save_and_verify(self, tmp_path: Path) -> None:
        x = torch.tensor([100.0, 200.0])
        loaded = _save_load(_DisjointViewBuffers(), x, tmp_path)
        verdict, _ = _verdict(loaded, x)
        assert verdict == "verified"

    def test_tied_weights_save_stage_once_and_verify(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 4)
        loaded = _save_load(_TiedWeights().eval(), x, tmp_path)
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


_RNG_CHANNELS = (
    ("os_urandom", lambda: os.urandom(1)[0] > 127),
    ("secrets_randbelow", lambda: bool(secrets.randbelow(2))),
    ("secrets_token_bytes", lambda: secrets.token_bytes(1)[0] > 127),
    ("uuid4", lambda: uuid.uuid4().int % 2 == 0),
    ("time_ns", lambda: (time.time_ns() // 1000) % 2 == 0),
    ("time_time", lambda: (int(time.time() * 1e6) % 2) == 0),
    ("perf_counter", lambda: (int(time.perf_counter() * 1e6) % 2) == 0),
    ("outside_held_generator", lambda: _OUTSIDE_HELD_GEN.normal() > 0.0),
    ("default_rng_in_forward", lambda: np.random.default_rng().random() > 0.5),
    ("private_random_instance", lambda: _PRIVATE_RANDOM.random() > 0.5),
    ("random_subclass_instance", lambda: _SUBCLASS_RANDOM.random() > 0.5),
    ("system_random", lambda: random.SystemRandom().random() > 0.5),
)


class TestHon12HostRngChannels:
    """A touch on any named channel permanently ceilings; absence never over-triggers."""

    @pytest.mark.parametrize(
        ("label", "draw"), _RNG_CHANNELS, ids=[row[0] for row in _RNG_CHANNELS]
    )
    def test_channel_touch_ceilings_permanently(
        self, label: str, draw: Any, tmp_path: Path
    ) -> None:
        loaded = _save_load(_branch_model(draw), torch.ones(4), tmp_path, acts=True)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is True, label
        verdict, attest = _verdict(loaded, torch.ones(4), seed=0)
        assert verdict != "verified", label
        assert attest == "not_applicable", label

    def test_model_attribute_generator_ceilings(self, tmp_path: Path) -> None:
        loaded = _save_load(_ModelAttrGenerator(), torch.ones(4), tmp_path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is True
        verdict, _ = _verdict(loaded, torch.ones(4), seed=0)
        assert verdict != "verified"

    def test_plain_model_does_not_ceiling(self, tmp_path: Path) -> None:
        """CRITICAL over-trigger pin: TL-internal timing reads must not mark the trace."""

        torch.manual_seed(0)
        loaded = _save_load(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval(), torch.randn(2, 4), tmp_path
        )
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is False
        verdict, _ = _verdict(loaded, torch.randn(2, 4))
        assert verdict == "verified"

    def test_no_draw_generator_holder_does_not_ceiling(self, tmp_path: Path) -> None:
        class _Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rng = np.random.default_rng(11)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        loaded = _save_load(_Holder(), torch.ones(4), tmp_path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is False

    def test_global_engine_stays_replayable(self, tmp_path: Path) -> None:
        """Global ``random`` module draws keep the seeded-reproduction semantics."""

        class _GlobalBranch(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 if random.random() > 0.5 else x + 1

        random.seed(0)
        loaded = _save_load(_GlobalBranch(), torch.ones(4), tmp_path)
        descriptor = loaded.runnable_descriptor
        assert descriptor is not None
        assert descriptor.rng_profile.host_rng_consumed is True
        assert descriptor.rng_profile.capture_seed == 0
        verdict, _ = _verdict(loaded, torch.ones(4), seed=0)
        assert verdict == "verified"
        verdict_off, _ = _verdict(loaded, torch.ones(4))
        assert verdict_off != "verified"

    def test_forward_channel_calls_still_work_and_restore(self, tmp_path: Path) -> None:
        """The scoped patches pass values through and restore on exit."""

        urandom_before = os.urandom
        loaded = _save_load(_branch_model(lambda: os.urandom(4)[0] > 127), torch.ones(4), tmp_path)
        assert loaded is not None
        assert os.urandom is urandom_before
        assert len(os.urandom(4)) == 4
        assert sys.getprofile() is None


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

    def test_lossy_live_output_is_poisoned(self) -> None:
        x = torch.tensor([1.0, 2.0])
        trace = tl.trace(_LossyLiveModel(), x, capture=_CAPTURE)
        result = trace.run(inputs=x, on_divergence="return_diverged")
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
        assert result.report.poisoned is True
        with pytest.raises(PoisonedRunError):
            result.trace.to_pandas()

    def test_faithful_live_refresh_stays_unpoisoned(self) -> None:
        x = torch.tensor([1.0, 2.0])
        trace = tl.trace(_FaithfulLiveModel(), x, capture=_CAPTURE)
        result = trace.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.poisoned is False
        assert len(result.trace.to_pandas()) > 0


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

    def test_disjoint_tiles_verified_under_all_device_contexts(self, tmp_path: Path) -> None:
        loaded = self._loaded(tmp_path)
        runtime_base = torch.randn(4, 4)
        tiles = (runtime_base[:2, :2], runtime_base[:2, 2:])
        verdict_plain, _ = _verdict(loaded, tiles)
        assert verdict_plain == "verified"
        with torch.device("meta"):
            verdict_meta, _ = _verdict(loaded, tiles)
        assert verdict_meta == "verified"
        with torch.device("meta"), torch.device("meta"):
            verdict_nested, _ = _verdict(loaded, tiles)
        assert verdict_nested == "verified"


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
    intervention_ready=True, capture_container_structure=True, cache=False))
trace.save(sys.argv[1], level="runnable")
"""

_CORR2_3_CONSUME = """
import sys, torch, torchlens as tl
from torchlens.utils._torch_compat import get_current_function_mode_stack

before = get_current_function_mode_stack()
assert before == [], before
result = tl.load(sys.argv[1]).run(inputs=torch.tensor([1.0, 2.0], device="cpu"))
after = get_current_function_mode_stack()
assert result.report.path_faithfulness.value == "verified", result.report.path_faithfulness
assert after == [], after
assert str(torch.get_default_device()) == "cpu"
print("CORR2_3_OK")
"""


@pytest.mark.heavy
class TestCorr23ScopedDefaultDevice:
    """Applying a recorded default device never leaks a process-level DeviceContext."""

    def test_consumer_mode_stack_is_untouched(self, tmp_path: Path) -> None:
        artifact = tmp_path / "meta_default.tlspec"
        for snippet, args in (
            (_CORR2_3_PRODUCE, [str(artifact)]),
            (_CORR2_3_CONSUME, [str(artifact)]),
        ):
            proc = subprocess.run(
                [sys.executable, "-c", snippet, *args],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            assert proc.returncode == 0, proc.stderr


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

    @pytest.mark.parametrize("policy", ("raise", "return_diverged"))
    def test_property_trap_subclass_fails_typed(self, policy: str, tmp_path: Path) -> None:
        x = torch.randn(2, 3)
        loaded = _save_load(_AddPair(), (x, x.clone()), tmp_path, acts=True)
        exotic = torch.Tensor._make_subclass(_PropertyTrapTensor, x, False)
        with pytest.raises(PathDivergenceError) as excinfo:
            loaded.run(inputs=(exotic, x.clone()), on_divergence=policy)
        assert "input_layout" in str(excinfo.value) or "input_tree_mismatch" in str(
            getattr(excinfo.value, "code", "")
        )


# ======================================================================================
# corr2-7 -- requires_grad-preserving runtime clones
# ======================================================================================


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class TestCorr27CloneGradFidelity:
    """Defensive clones mirror the runtime leaf's autograd metadata."""

    def test_requires_grad_input_verifies_and_attests(self, tmp_path: Path) -> None:
        x = torch.randn(3, requires_grad=True)
        loaded = _save_load(_Double(), x, tmp_path, acts=True)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
        assert result.output.requires_grad is True
        assert result.output.grad_fn is not None

    def test_changed_flag_input_is_physical_change(self, tmp_path: Path) -> None:
        """An intentionally changed requires_grad stays a physical input change."""

        x = torch.randn(3, requires_grad=True)
        loaded = _save_load(_Double(), x, tmp_path, acts=True)
        result = loaded.run(inputs=x.detach(), on_divergence="return_diverged")
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
        assert result.output.requires_grad is False


# ======================================================================================
# hon1_3 / corr2-8 -- CUDA staging + readiness
# ======================================================================================


_CPU_ONLY_READINESS = """
import sys, torch, torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.runnable import ReadinessStatus

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

    def test_embedded_state_runs_verified_and_attested(self, tmp_path: Path) -> None:
        loaded, x, _ = self._loaded(tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
        assert result.output.device.type == "cuda"

    def test_cpu_user_state_dict_is_staged_to_slot_device(self, tmp_path: Path) -> None:
        loaded, x, model = self._loaded(tmp_path)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        loaded.load_state_dict(cpu_state)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    def test_nondeterministic_context_marker_leg(self, tmp_path: Path) -> None:
        """3-ADJ-9: the R1 attestation-ineligibility marker is exercised E2E."""

        torch.manual_seed(0)
        model = nn.ConvTranspose2d(2, 2, 3).cuda().eval()
        x = torch.randn(1, 2, 5, 5, device="cuda")
        loaded = _save_load(model, x, tmp_path, acts=True)
        result = loaded.run(inputs=x, on_divergence="return_diverged")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE

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

    def test_plain_capture_under_inference_mode(self) -> None:
        torch.manual_seed(0)
        model = _InferenceModel().eval()
        with torch.inference_mode():
            trace = tl.trace(model, torch.randn(2, 3))
        assert trace is not None

    def test_runnable_e2e_under_inference_mode(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = _InferenceModel().eval()
        x = torch.randn(2, 3)
        with torch.inference_mode():
            trace = tl.trace(model, x, capture=_CAPTURE)
        path = tmp_path / "inference.tlspec"
        trace.save(path, level="runnable", include_weights=True)
        loaded = tl.load(path)
        descriptor = loaded.runnable_descriptor
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
    def __new__(cls, a: torch.Tensor, b: torch.Tensor) -> "_StatefulNT":
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
        assert "missing_output_container_contract" in str(excinfo.value)

    def test_stateful_namedtuple_subclass_refuses_at_save(self, tmp_path: Path) -> None:
        trace = tl.trace(_nt_model(_StatefulNT), torch.tensor([1.0, 2.0]), capture=_CAPTURE)
        with pytest.raises(RunnablePreflightError) as excinfo:
            trace.save(tmp_path / "stateful_nt.tlspec", level="runnable")
        assert "missing_output_container_contract" in str(excinfo.value)

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

    def test_torch_return_types_stay_verified(self, tmp_path: Path) -> None:
        class _MaxModel(nn.Module):
            def forward(self, x: torch.Tensor) -> Any:
                return torch.max(x, dim=0)

        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        loaded = _save_load(_MaxModel(), x, tmp_path)
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

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
        trace.save(bundle, level="runnable", include_weights=True)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest))
        return bundle

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

    def test_valid_context_round_trips(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        loaded = _save_load(model, x, tmp_path)
        verdict, _ = _verdict(loaded, x)
        assert verdict == "verified"
