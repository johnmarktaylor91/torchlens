"""Wave-0 distributed capture-fidelity census: rows, ZI gate, reds, report.

The L8 census plan (sec 1) wave-0 merge: the topology-row registry with the
full-conjunction ``row_green`` gate, the dual-channel bare ground truth, the
zero-interference gate (ZI-1..ZI-5 + arm-parity), Group A control rows at
their wave-0 obligation ("ZI baseline + criterion-1 green" -- NEVER called
row green), Group B/C rows asserted red-by-typed-refusal (wave 0 grants NO
relaxation to ANY refusal), the Group N self-honesty reds constructible
against shipped C0 records (N1, N3b, N3c, N3d), and the census report
generator with NOT-RUN/NOT-COVERED disclosures.

CUDA/NCCL rows are NOT-RUN-DISCLOSED: this box has no NVIDIA GPU.
Single-host gloo sims are CPU-only, sequential, RAM-preflighted.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl  # noqa: E402
from tests.support.census_harness import (  # noqa: E402
    CENSUS_ROWS,
    COMPLETION_FLOOR_MISSING,
    FULL_CRITERIA,
    INTERPOSED_COMPLETION_OPS,
    NOT_COVERED_LINE,
    REFUSAL_PRODUCT_NAME,
    WAVE0_PRODUCT_NAME,
    ZI_GATE_FAILED,
    CensusReportError,
    CensusResult,
    dispatcher_interposition_counters,
    generate_census_report,
    model_state_digest,
    preflight_spawn_world,
    run_census_row,
    run_refusal_row,
)
from torchlens.distributed import (  # noqa: E402
    _lifecycle as lifecycle,
    _recognizer as recognizer_mod,
    has_vetted_snapshot,
)

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.has_vetted_snapshot());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Applied to the classes/tests that arm (directly or through a shared
# helper) before exercising their row; TestRowGreenGate, TestDualChannelBareLeg,
# and the handful of other tests named below never arm and must keep running
# everywhere.
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)


def _dense_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _dense_input() -> torch.Tensor:
    return torch.randn(2, 4)


@contextmanager
def _single_rank_group() -> Iterator[Any]:
    """Single-rank CPU gloo world with guaranteed disarm + teardown.

    Armed state is dormant while no process group is initialized (plane-P and
    the funcol session only observe captures that could issue collectives), so
    every armed leg that must exercise the real plane-P path runs inside one.
    """

    import socket

    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    lifecycle.disarm()
    saved_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(probe.getsockname()[1])
    try:
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        except (OSError, RuntimeError, ValueError) as error:  # pragma: no cover - env dependent
            pytest.skip(f"gloo init failed: {error}")
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture()
def single_rank_world() -> Iterator[Any]:
    """Single-rank CPU gloo world with guaranteed disarm + teardown."""

    with _single_rank_group() as dist:
        yield dist


@pytest.fixture()
def single_rank_mesh(single_rank_world):
    """1-rank CPU device mesh on the single-rank world."""

    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


# ===========================================================================
# The row_green gate + report generator (pure logic; plan rules 1.1(1), 1.4).
# ===========================================================================


class TestRowGreenGate:
    def _result(self, **overrides) -> CensusResult:
        base: dict[str, object] = {
            "outputs_bit_identical": True,
            "ground_truth_ops": ["aten.mm.default"],
            "criteria_run": FULL_CRITERIA,
            "failures": [],
            "row_id": "A1",
            "world": "1proc",
            "floors": {"content": "met"},
            "zi_gate_passed": True,
        }
        base.update(overrides)
        return CensusResult(**base)  # type: ignore[arg-type]

    def test_full_conjunction_is_green(self):
        assert self._result().row_green

    def test_partial_criteria_never_green(self):
        result = self._result(criteria_run=(1,))
        assert not result.row_green
        assert result.product_name == WAVE0_PRODUCT_NAME

    def test_failures_kill_green(self):
        assert not self._result(failures=["output 0 not bit-identical (x)"]).row_green

    def test_floor_miss_kills_green_even_with_clean_failures_list(self):
        # Belt: floors are wired INTO row_green directly, not only via failures.
        assert not self._result(floors={"content": "missing"}).row_green

    def test_zi_gate_none_or_false_never_green(self):
        assert not self._result(zi_gate_passed=None).row_green
        assert not self._result(zi_gate_passed=False).row_green

    def test_registry_covers_the_plan_matrix(self):
        groups = {spec.group for spec in CENSUS_ROWS.values()}
        assert groups == {"A", "B", "C", "D", "N", "CUDA"}
        assert {f"B{i}" for i in range(1, 9)} <= set(CENSUS_ROWS)
        assert {"C1r", "C2r", "C3r", "C4r", "C5r"} <= set(CENSUS_ROWS)
        assert {"D1", "D2", "D3"} <= set(CENSUS_ROWS)
        assert {"N1", "N2", "N3a", "N3b", "N3c", "N3d", "N4", "N5"} <= set(CENSUS_ROWS)

    def test_wave0_runnable_rows_grant_no_relaxation(self):
        # Wave 0 runs controls, refusal reds, and plane-P-free self-honesty
        # reds ONLY; every Group B/D relaxation row stays gated.
        runnable = {r for r, s in CENSUS_ROWS.items() if s.runnable_wave0}
        assert runnable == {
            "A1",
            "A2",
            "A3",
            "C1r",
            "C2r",
            "C3r",
            "C4r",
            "C5r",
            "N1",
            "N3b",
            "N3c",
            "N3d",
        }


class TestCensusReportGenerator:
    def test_report_refuses_unknown_row(self, tmp_path):
        rogue = CensusResult(outputs_bit_identical=True, row_id="Z9")
        with pytest.raises(CensusReportError, match="unknown census row"):
            generate_census_report(
                [rogue],
                torch_version="t",
                repo_sha="s",
            )

    def test_wave0_report_never_says_row_green(self, tmp_path):
        result = CensusResult(
            outputs_bit_identical=True,
            ground_truth_ops=["aten.mm.default"],
            criteria_run=(1,),
            row_id="A1",
            world="1proc",
            floors={"content": "met"},
            zi_gate_passed=True,
        )
        md_text, payload = generate_census_report(
            [result],
            torch_version=torch.__version__,
            repo_sha="testsha",
            md_path=tmp_path / "census.md",
            json_path=tmp_path / "census.json",
        )
        assert "row green" not in md_text
        assert WAVE0_PRODUCT_NAME in md_text
        assert payload["rows"][0]["row_green"] is False
        assert NOT_COVERED_LINE in md_text
        # Every registry row that did not run is disclosed, CUDA annex included.
        not_run_ids = {row["row"] for row in payload["not_run"]}
        assert "CUDA-multigpu-realism" in not_run_ids
        assert "B8" in not_run_ids and "N2" in not_run_ids
        assert (tmp_path / "census.md").exists() and (tmp_path / "census.json").exists()
        json.loads((tmp_path / "census.json").read_text())

    def test_full_conjunction_row_reports_green(self):
        result = CensusResult(
            outputs_bit_identical=True,
            ground_truth_ops=["aten.mm.default"],
            criteria_run=FULL_CRITERIA,
            row_id="A1",
            world="1proc",
            floors={"content": "met"},
            zi_gate_passed=True,
        )
        md_text, payload = generate_census_report(
            [result],
            torch_version="t",
            repo_sha="s",
        )
        assert payload["rows"][0]["product"] == "row green"

    @requires_vetted_snapshot
    def test_criteria_2_through_4_refuse_without_plane_p(self):
        """A green can never be vacuous: unarmed captures carry no plane-P
        journal, so requesting criteria 2-4 on one raises instead of
        silently passing (the wave-0 honesty property, wave-1 form)."""

        lifecycle.disarm()
        with pytest.raises(NotImplementedError, match="C2"):
            run_census_row("A1", _dense_model, _dense_input, criteria=(1, 2, 3, 4))


# ===========================================================================
# Dual-channel bare leg: dispatcher interposition (plan 1.1 + 1.3).
# ===========================================================================


@pytest.mark.heavy
class TestDualChannelBareLeg:
    def test_interposition_fires_and_tears_down_clean(self, single_rank_world):
        dist = single_rank_world
        import torch.distributed._functional_collectives as funcol

        with dispatcher_interposition_counters() as counters:
            out = funcol.all_reduce(torch.ones(3), "sum", dist.group.WORLD)
            _ = out + 1  # materializing read triggers the ACT wait
            assert counters["_c10d_functional::wait_tensor"] >= 1
        # Context exit ran the post-teardown probe internally; a leaked
        # registration would have raised InterpositionTeardownError.

    def test_mode_invisibility_of_wait_is_reconfirmed(self, single_rank_world):
        # The P2 hazard the dual channel exists for: the mode does NOT see
        # the ACT-triggered wait, the dispatcher counter does.
        dist = single_rank_world
        import torch.distributed._functional_collectives as funcol

        from tests.support.census_harness import ReferenceDispatchLogger

        with dispatcher_interposition_counters() as counters:
            logger = ReferenceDispatchLogger()
            with logger:
                out = funcol.all_reduce(torch.ones(3), "sum", dist.group.WORLD)
                _ = out + 1
        assert counters["_c10d_functional::wait_tensor"] >= 1
        assert not any("wait_tensor" in op for op in logger.ops)


# ===========================================================================
# Zero-interference gate (plan 1.3: ZI-1..ZI-5 + arm parity).
# ===========================================================================


def _trace_shape_summary(log) -> list[tuple[str, object]]:
    return [(op.layer_label, tuple(op.shape) if op.shape else None) for op in log.ops]


@pytest.mark.heavy
class TestZeroInterferenceGate:
    @requires_vetted_snapshot
    def test_zi1_armed_on_vs_off_dense_capture_identical(self):
        """ZI-1: arming changes NOTHING about a dense non-distributed capture.

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path.
        """

        from torchlens._capture_fingerprint import _fingerprint_model_content

        lifecycle.disarm()
        torch.manual_seed(7)
        model = _dense_model()
        x = torch.randn(2, 4)
        content_before = _fingerprint_model_content(model)

        log_off = tl.trace(model, x)
        with _single_rank_group():
            lifecycle.arm()
            log_on = tl.trace(model, x)
            assert log_on._distributed_plane_p is not None, "armed leg must run plane-P"

        assert _trace_shape_summary(log_off) == _trace_shape_summary(log_on)
        assert torch.equal(log_off.output_ops[0].out, log_on.output_ops[0].out)
        assert _fingerprint_model_content(model) == content_before
        assert model_state_digest(model) == model_state_digest(model)

    @requires_vetted_snapshot
    def test_zi4_disarm_restores_every_wrapper_twice(self):
        """ZI-4: disarm restores every wrapped attr; a second cycle is clean."""

        lifecycle.disarm()
        for _ in range(2):
            lifecycle.arm()
            state = lifecycle.armed_state()
            assert state is not None
            originals = dict(state.originals)
            assert originals, "arming installed no wraps -- the check would be vacuous"
            lifecycle.disarm()
            assert lifecycle.armed_state() is None
            for (module, name), original in originals.items():
                assert getattr(module, name) is original, f"{name} not restored"

    def test_zi5_degraded_arming_never_raises_at_capture_entry(
        self, single_rank_world, monkeypatch
    ):
        """ZI-5: broken arming precondition -> capture COMPLETES, typed degradation."""

        vetted = dict(recognizer_mod.VETTED_NAMESPACE_SNAPSHOTS[0][1])
        vetted["c10d"] = vetted["c10d"] - {"allreduce_"}
        monkeypatch.setattr(recognizer_mod, "VETTED_NAMESPACE_SNAPSHOTS", (("tampered", vetted),))
        lifecycle.disarm()
        with pytest.warns(UserWarning, match="could not arm"):
            log = tl.trace(_dense_model(), _dense_input())
        assert log.output_ops  # the capture completed
        degradation = lifecycle.auto_arm_degradation()
        assert degradation is not None and "uncaptured_collective_op" in degradation

    @requires_vetted_snapshot
    def test_auto_arm_vs_explicit_arm_parity(self, single_rank_world):
        """One workload, lazy auto-arm vs explicit arm(): boundary records
        equivalent modulo the install-epoch diagnostic."""

        dist = single_rank_world

        class CollectiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.fc(x)
                dist.all_reduce(hidden)
                return hidden

        def boundary_facts(log):
            entries = log.annotations["distributed"]["boundaries"]
            return [
                (e["kind"], e["correlation"]["channel"], e["correlation"]["seq"]) for e in entries
            ]

        torch.manual_seed(3)
        model = CollectiveModel()
        x = torch.randn(2, 4)

        lifecycle.disarm()  # lazy leg: capture entry arms
        lazy_facts = boundary_facts(tl.trace(model, x))

        lifecycle.disarm()  # explicit leg: fresh armed state, process-start arm
        lifecycle.arm()
        explicit_facts = boundary_facts(tl.trace(model, x))

        assert lazy_facts == explicit_facts
        assert len(lazy_facts) == 1 and lazy_facts[0][0] == "all_reduce"


@pytest.mark.slow
@requires_vetted_snapshot
class TestZeroInterferenceSuiteAndPerf:
    #: ZI-2's NAMED representative subset -- enumerated here, not "whatever was
    #: convenient"; silent truncation of the subset is itself a red.
    ZI2_SUBSET = ("capture_core", "save_load_roundtrip", "validation", "viz_smoke")

    def _zi2_run_subset(self, tmp_path, tag: str) -> dict[str, object]:
        from torchlens.validation import validate_forward_pass

        summaries: dict[str, object] = {}
        torch.manual_seed(11)
        model = _dense_model()
        x = torch.randn(2, 4)

        log = tl.trace(model, x)
        summaries["capture_core"] = _trace_shape_summary(log)

        save_path = tmp_path / f"zi2-{tag}.tlspec"
        tl.save(log, save_path)
        loaded = tl.load(save_path)
        summaries["save_load_roundtrip"] = _trace_shape_summary(loaded)

        summaries["validation"] = bool(validate_forward_pass(model, x))

        # Structure only: the raw summary text embeds volatile timing/memory
        # figures whose rendered width varies run to run.
        summaries["viz_smoke"] = (
            len(log.summary().splitlines()),
            tuple(log.module_collapse_order),
        )
        assert tuple(summaries) == self.ZI2_SUBSET, "ZI-2 subset silently truncated"
        return summaries

    def test_zi2_named_subset_identical_armed_off_then_on(self, tmp_path):
        """ZI-2: the named subset is identical armed off, then on.

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path.
        """

        lifecycle.disarm()
        off = self._zi2_run_subset(tmp_path, "off")
        with _single_rank_group():
            lifecycle.arm()
            on = self._zi2_run_subset(tmp_path, "on")
        assert off == on

    def test_zi3_armed_overhead_inside_p2_gate(self):
        """ZI-3: armed-ON overhead on the A1 workload stays inside P2's
        blocking 10% gate (P2 owns the ceiling; asserted, not invented).

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path."""

        from tests.test_perf_capture_ab import GATE_CEILING_FRACTION

        torch.manual_seed(5)
        model = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10)
        )
        x = torch.randn(8, 64)

        def median_capture_seconds(reps: int) -> float:
            times = []
            for _ in range(reps):
                start = time.perf_counter()
                tl.trace(model, x)
                times.append(time.perf_counter() - start)
            return sorted(times)[len(times) // 2]

        def measure(reps: int) -> tuple[float, float]:
            lifecycle.disarm()
            tl.trace(model, x)  # warmup
            off = median_capture_seconds(reps)
            with _single_rank_group():
                lifecycle.arm()
                on = median_capture_seconds(reps)
            return off, on

        off, on = measure(reps=7)
        if on > off * (1 + GATE_CEILING_FRACTION):
            # One measurement refinement at higher rep count before failing:
            # medians on a loaded box are noisy; the ceiling itself never moves.
            off, on = measure(reps=15)
        assert on <= off * (1 + GATE_CEILING_FRACTION), (
            f"armed-ON capture overhead {on / off - 1:.1%} exceeds P2's "
            f"{GATE_CEILING_FRACTION:.0%} gate (off={off * 1e3:.1f}ms, on={on * 1e3:.1f}ms)"
        )


# ===========================================================================
# Group A controls at their wave-0 obligation (K1 + ZI; phasing note 1.2).
# ===========================================================================


@pytest.mark.heavy
@requires_vetted_snapshot
class TestGroupAControlsWave0:
    def test_a1_dense_armed_on_zi_baseline_criterion1_green(self, single_rank_world):
        """A1 wave 0: the armed dense anchor is criterion-1 green, never row green.

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path.
        """

        lifecycle.disarm()
        lifecycle.arm()
        try:
            result = run_census_row(
                "A1",
                _dense_model,
                _dense_input,
                content_floor=lambda ops: any("addmm" in op or "mm" in op for op in ops),
                zi_gate_passed=True,
            )
        finally:
            lifecycle.disarm()
        assert result.green and result.floors["content"] == "met"
        assert result.product_name == WAVE0_PRODUCT_NAME
        # The honesty pin: the wave-0 product is NEVER row green.
        assert not result.row_green

    def test_a1_zi_gate_failure_is_typed_and_kills_green(self):
        lifecycle.disarm()
        result = run_census_row(
            "A1",
            _dense_model,
            _dense_input,
            zi_gate_passed=False,
        )
        assert ZI_GATE_FAILED in result.failures
        assert not result.row_green and result.product_name == "red"


def _a2_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch.distributed as dist

    from tests.support.census_harness import run_census_row
    from torchlens.distributed import arm

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    arm()

    class DenseCollectives(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            hidden = self.fc(x)
            dist.all_reduce(hidden)
            work = dist.all_reduce(hidden, async_op=True)
            work.wait()
            gathered = [torch.empty_like(hidden) for _ in range(world_size)]
            dist.all_gather(gathered, hidden)
            dist.broadcast(hidden, src=0)
            scattered = torch.empty_like(hidden)
            scatter_list = [hidden + i for i in range(world_size)] if rank == 0 else None
            dist.scatter(scattered, scatter_list, src=0)
            dist.barrier()
            return hidden + gathered[-1] + scattered

    result = run_census_row(
        "A2",
        DenseCollectives,
        lambda: torch.ones(2, 4),
        content_floor=lambda ops: any("c10d" in op for op in ops),
        zi_gate_passed=True,
    )
    # The coalescing-manager leg joins the wave-1 full-criteria re-run; the
    # deferral is disclosed on the row, never silently dropped.
    result.not_run.append("coalescing-manager leg deferred to the wave-1 full-criteria re-run")
    # Full-criteria A2 stays OWED (never silently green): its async c10d
    # traffic completes via Work.wait, which is a C++ method invisible to BOTH
    # ground-truth channels (unlike funcol's dispatcher-level wait_tensor), so
    # the completion-event floor of plan rule 1.1(3) is not yet evaluable here.
    # Widening completion observation to c10d Work objects is named plane-W
    # follow-on work; until it lands, an A2 row-green claim would overstate.
    result.not_run.append(
        "full-criteria re-run blocked on c10d Work-level completion observation "
        "(plane-W interposes funcol wait_tensor only)"
    )
    payload = {
        "rank": rank,
        "product": result.product_name,
        "row_green": result.row_green,
        "failures": result.failures,
        "floors": result.floors,
        "ground_truth_op_count": len(result.ground_truth_ops),
        "not_run": result.not_run,
    }
    with open(os.path.join(out_dir, f"a2_rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)
    dist.destroy_process_group()


def _a3_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch.distributed as dist

    from tests.support.census_harness import run_census_row
    from torchlens.distributed import arm

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    arm()

    if rank == 0:

        class Peer(nn.Module):
            def forward(self, x):
                doubled = x * 2
                dist.send(doubled, dst=1, tag=3)
                buffer = torch.empty(3)
                dist.recv(buffer, src=1, tag=4)
                return doubled + buffer

    else:

        class Peer(nn.Module):
            def forward(self, x):
                buffer = torch.empty(3)
                dist.recv(buffer, src=0, tag=3)
                work = dist.isend(buffer + 1, dst=0, tag=4)
                work.wait()
                return buffer + x

    result = run_census_row(
        "A3",
        Peer,
        lambda: torch.ones(3),
        content_floor=lambda ops: any("c10d" in op for op in ops),
        zi_gate_passed=True,
    )
    payload = {
        "rank": rank,
        "product": result.product_name,
        "row_green": result.row_green,
        "failures": result.failures,
        "floors": result.floors,
    }
    with open(os.path.join(out_dir, f"a3_rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)
    dist.destroy_process_group()


@pytest.mark.slow
@requires_vetted_snapshot
class TestGroupASpawnSims:
    """W2 spawn gloo sims -- CPU-only, sequential, RAM-preflighted.

    Both spawned workers (_a2_worker, _a3_worker) call arm() unconditionally;
    the F1 ruling gate (see requires_vetted_snapshot above) was applied to
    every OTHER arming class/test in this file but missed this one.
    """

    def _spawn(self, worker, tmp_path, world_size: int = 2) -> list[dict]:
        import torch.multiprocessing as mp

        ok, available_gb = preflight_spawn_world(world_size)
        if not ok:
            pytest.skip(f"RAM preflight failed: {available_gb:.1f} GB available")
        lifecycle.disarm()
        if torch.distributed.is_initialized():
            pytest.skip("a process group is already initialized in this process")
        mp.spawn(
            worker,
            args=(world_size, str(tmp_path / "store"), str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
        prefix = worker.__name__.strip("_").split("_")[0]
        payloads = []
        for rank in range(world_size):
            with open(tmp_path / f"{prefix}_rank{rank}.json") as handle:
                payloads.append(json.load(handle))
        return payloads

    def test_a2_dense_collectives_w2_zi_baseline_criterion1(self, tmp_path):
        for payload in self._spawn(_a2_worker, tmp_path):
            assert payload["failures"] == [], payload
            assert payload["product"] == WAVE0_PRODUCT_NAME
            assert payload["row_green"] is False  # never row green in wave 0
            assert payload["floors"]["content"] == "met"
            assert payload["ground_truth_op_count"] > 0
            assert any("coalescing" in item for item in payload["not_run"])

    def test_a3_p2p_w2_zi_baseline_criterion1(self, tmp_path):
        for payload in self._spawn(_a3_worker, tmp_path):
            assert payload["failures"] == [], payload
            assert payload["product"] == WAVE0_PRODUCT_NAME
            assert payload["row_green"] is False


# ===========================================================================
# Group B: relaxation candidates -- wave 0 grants NOTHING; every constructible
# row is asserted red-by-typed-refusal (STAYS-REFUSED until census + D-L8-CAP).
# ===========================================================================


@pytest.mark.heavy
@requires_vetted_snapshot
class TestGroupBRefusedInWave0:
    def _refusal(self, row_id, build, expected=("dtensor",)):
        lifecycle.arm()
        result = run_refusal_row(row_id, build, expected)
        assert result.green, result.failures
        assert result.product_name == REFUSAL_PRODUCT_NAME
        assert not result.row_green

    def test_b1_plain_dtensor_refuses(self, single_rank_mesh):
        from torch.distributed.tensor import Shard, distribute_tensor

        def build():
            model = nn.Linear(4, 4)
            model.weight = nn.Parameter(
                distribute_tensor(torch.randn(4, 4), single_rank_mesh, [Shard(0)])
            )
            return tl.trace(model, torch.randn(2, 4))

        self._refusal("B1", build)

    def test_b2_colwise_refuses(self, single_rank_mesh):
        from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

        def build():
            model = nn.Sequential(nn.Linear(4, 8))
            parallelize_module(model, single_rank_mesh, {"0": ColwiseParallel()})
            return tl.trace(model, torch.randn(2, 4))

        self._refusal("B2", build, expected=("dtensor", "tensor_parallel"))

    def test_b3_rowwise_refuses(self, single_rank_mesh):
        from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

        def build():
            model = nn.Sequential(nn.Linear(8, 4))
            parallelize_module(model, single_rank_mesh, {"0": RowwiseParallel()})
            return tl.trace(model, torch.randn(2, 8))

        self._refusal("B3", build, expected=("dtensor", "tensor_parallel"))

    def test_b4_sequence_parallel_refuses(self, single_rank_mesh):
        from torch.distributed.tensor.parallel import SequenceParallel, parallelize_module

        def build():
            model = nn.Sequential(nn.LayerNorm(8))
            parallelize_module(model, single_rank_mesh, {"0": SequenceParallel()})
            return tl.trace(model, torch.randn(2, 3, 8))

        self._refusal("B4", build)

    def test_b5_tp_mlp_pair_refuses(self, single_rank_mesh):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        def build():
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
            parallelize_module(
                model,
                single_rank_mesh,
                {"0": ColwiseParallel(), "2": RowwiseParallel()},
            )
            return tl.trace(model, torch.randn(2, 4))

        self._refusal("B5", build, expected=("dtensor", "tensor_parallel"))

    def test_b6_fsdp2_fully_shard_refuses(self, single_rank_mesh):
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard

        def build():
            model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
            fully_shard(model, mesh=single_rank_mesh)
            return tl.trace(model, torch.randn(2, 4))

        self._refusal("B6", build, expected=("dtensor", "tensor_parallel"))

    def test_b7_shard_dim_alltoall_rides_b1(self, single_rank_mesh):
        from torch.distributed.tensor import Shard, distribute_tensor

        def build():
            class Redistributor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(
                        distribute_tensor(torch.randn(4, 4), single_rank_mesh, [Shard(0)])
                    )

                def forward(self, x):
                    moved = self.weight.redistribute(single_rank_mesh, [Shard(1)])
                    return x @ moved.to_local()

            return tl.trace(Redistributor(), torch.randn(2, 4))

        self._refusal("B7", build)


# ===========================================================================
# Group C: refusal parity reds (asserted, never skipped; plan 1.1(4)).
# ===========================================================================


@pytest.mark.heavy
class TestGroupCRefusalParity:
    @requires_vetted_snapshot
    def test_c1r_sharded_tensor_refuses_with_variant_tag(self, single_rank_world):
        from torch.distributed._shard import sharded_tensor
        from torch.distributed._shard.sharding_spec import ChunkShardingSpec

        from torchlens._distributed import DistributedCaptureUnsupportedError

        lifecycle.arm()
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cpu"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch's own deprecation
            # empty(), not zeros(): under wrapped torch, ShardedTensor's
            # identity-keyed __torch_function__ table no longer recognizes the
            # TL-wrapped nn.init.constant_ that zeros() routes through.
            legacy = sharded_tensor.empty(spec, 4, 4)
        model = nn.Linear(4, 4)
        model.legacy_state = legacy

        result = run_refusal_row("C1r", lambda: tl.trace(model, torch.randn(2, 4)), ("dtensor",))
        assert result.green, result.failures
        # The monolithic finding carries the ShardedTensor variant tag today
        # (post-split this row asserts the per-arm kind instead).
        with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
            tl.trace(model, torch.randn(2, 4))
        finding = next(f for f in excinfo.value.fields["findings"] if f.kind == "dtensor")
        assert "ShardedTensor" in finding.detail

    @requires_vetted_snapshot
    def test_c2r_prepare_module_input_refuses(self, single_rank_mesh):
        from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

        lifecycle.arm()

        def build():
            model = nn.Sequential(nn.Linear(4, 4))
            parallelize_module(
                model,
                single_rank_mesh,
                {"0": PrepareModuleInput(desired_input_layouts=None)},
            )
            return tl.trace(model, torch.randn(2, 4))

        result = run_refusal_row("C2r", build, ("tensor_parallel",))
        assert result.green, result.failures

    @requires_vetted_snapshot
    def test_c3r_synthetic_unattributable_tp_hook_fails_closed(self, single_rank_world):
        import torch.distributed.tensor.parallel  # noqa: F401  (namespace gate)

        lifecycle.arm()

        def synthetic_hook(module, args):
            return args

        synthetic_hook.__module__ = "torch.distributed.tensor.parallel._census_synthetic"
        model = nn.Linear(4, 4)
        model.register_forward_pre_hook(synthetic_hook)

        result = run_refusal_row(
            "C3r", lambda: tl.trace(model, torch.randn(2, 4)), ("tensor_parallel",)
        )
        assert result.green, result.failures

    @requires_vetted_snapshot
    def test_c4r_pp_composed_with_tp_refuses(self, single_rank_world):
        import torch.distributed.tensor.parallel  # noqa: F401  (namespace gate)
        from torch.distributed.pipelining import PipelineStage

        lifecycle.arm()

        def tp_ish_hook(module, args):
            return args

        tp_ish_hook.__module__ = "torch.distributed.tensor.parallel._census_synthetic"

        class Composed(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
                self.stage = PipelineStage(self.fc, 0, 1, torch.device("cpu"))
                self.fc.register_forward_pre_hook(tp_ish_hook)

            def forward(self, x):
                return self.fc(x)

        result = run_refusal_row(
            "C4r",
            lambda: tl.trace(Composed(), torch.randn(2, 4)),
            ("pipeline_parallel", "tensor_parallel"),
        )
        assert result.green, result.failures

    def test_c5r_shard_local_marker_substrate_and_merge_refusal(self):
        """C5r post-substrate: the distributed_scope marker SHIPPED (dark --
        no capture can set it until the D-L8-CAP relaxation; every sharded
        topology still refuses at entry per the Group B/C rows above), and
        the marker-keyed merge-scope refusal is asserted on BOTH derive_merge
        entry paths in tests/test_distributed_shard_local_marker.py."""

        from torchlens.constants import MODEL_LOG_FIELD_ORDER

        assert "distributed_scope" in MODEL_LOG_FIELD_ORDER
        log = tl.trace(_dense_model(), _dense_input())
        # A plain capture NEVER carries the marker (over-labeling guard).
        assert log.distributed_scope is None


# ===========================================================================
# Wave-1 criteria 2-4 (C2 recording lane): A-row full-criteria re-run + the
# red-first pinned constructions N2 / N3a / N4 (plan 1.5 wave 1).
# ===========================================================================


@pytest.mark.heavy
@requires_vetted_snapshot
class TestWave1FullCriteria:
    def test_a1_full_criteria_is_the_first_row_green(self, single_rank_world):
        """A1 re-run to FULL criteria: the zero-interference anchor must be
        row green (all four criteria, floors, ZI conjunct).

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path."""

        lifecycle.disarm()
        lifecycle.arm()
        try:
            result = run_census_row(
                "A1",
                _dense_model,
                _dense_input,
                criteria=(1, 2, 3, 4),
                content_floor=lambda ops: any("addmm" in op or "mm" in op for op in ops),
                zi_gate_passed=True,
            )
        finally:
            lifecycle.disarm()
        assert result.failures == []
        assert result.criteria_run == FULL_CRITERIA
        assert result.row_green and result.product_name == "row green"

    def _mixed_capture(self, dist):
        import torch.distributed._functional_collectives as funcol

        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.fc(x)
                dist.all_reduce(hidden)
                reduced = funcol.all_reduce(hidden, "sum", dist.group.WORLD)
                return reduced + 1

        from tests.support.census_harness import _seq_counter_snapshot

        lifecycle.arm()
        seq_before = _seq_counter_snapshot()
        log = tl.trace(Mixed(), torch.ones(2, 4))
        seq_after = _seq_counter_snapshot()
        return log, seq_before, seq_after

    def test_k3_green_on_mixed_c10d_funcol_capture(self, single_rank_world):
        """K3 passes on a real boundary-crossing capture: every collective
        dispatch discharged, completions classified, seq deltas == journal."""

        from tests.support.census_harness import (
            run_census_criterion_3,
            run_census_criterion_4,
        )

        log, seq_before, seq_after = self._mixed_capture(single_rank_world)
        assert run_census_criterion_3(log, seq_before, seq_after) == []
        assert run_census_criterion_4(log) == []

    def test_n2_dropped_op_class_turns_k2_red(self, single_rank_world):
        """N2: the zero-collective-Colwise regression formalized -- dropping a
        captured compute-op class must fail K2."""

        lifecycle.arm()
        result = run_census_row(
            "N2",
            _dense_model,
            _dense_input,
            criteria=(1, 2),
            drop_op_class="aten.addmm",
            zi_gate_passed=True,
        )
        assert any(failure.startswith("K2:") for failure in result.failures)
        assert not result.row_green and result.product_name == "red"

    def test_n3a_double_tick_turns_k3_red(self, single_rank_world):
        """N3a: a synthetic extra issue tick (the double-tick construction)
        must violate the per-(group_uid, channel) seq invariant."""

        from tests.support.census_harness import run_census_criterion_3

        log, seq_before, seq_after = self._mixed_capture(single_rank_world)
        failures = run_census_criterion_3(log, seq_before, seq_after, extra_ticks=1)
        assert any("seq invariant" in failure for failure in failures)

    def test_n3a_boundary_without_tick_turns_k3_red(self, single_rank_world):
        """The seq invariant is two-sided: a journaled boundary with no
        corresponding issue tick fails too."""

        from tests.support.census_harness import run_census_criterion_3

        log, seq_before, _seq_after = self._mixed_capture(single_rank_world)
        failures = run_census_criterion_3(log, seq_before, seq_before)
        assert any("seq invariant" in failure for failure in failures)

    def test_n4_injected_orphan_turns_k4_red(self, single_rank_world):
        """N4: an orphan interior record (no plane-S owner, no module context,
        not discharged, not TorchLens-internal) must fail K4."""

        from tests.support.census_harness import run_census_criterion_4

        log, _before, _after = self._mixed_capture(single_rank_world)
        orphan = ("aten.mm.default", None, False, False, False)
        failures = run_census_criterion_4(log, injected_orphans=(orphan,))
        assert any("orphan interior record" in failure for failure in failures)


# ===========================================================================
# Group N: census self-honesty reds constructible against shipped C0 records.
# ===========================================================================


@pytest.mark.heavy
@requires_vetted_snapshot
class TestGroupNSelfHonesty:
    def test_n1_perturbing_factory_goes_red(self):
        """N1: K1's red -- a factory that perturbs between legs must fail."""

        lifecycle.disarm()
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            model = nn.Linear(4, 4)
            if calls["n"] > 1:
                with torch.no_grad():
                    model.weight.add_(1.0)
            return model

        result = run_census_row("N1", factory, lambda: torch.randn(2, 4))
        assert not result.green and not result.row_green
        assert any("not bit-identical" in failure for failure in result.failures)

    def test_n3b_fake_op_in_allowlisted_namespace_ceilings_arming(self, single_rank_world):
        """N3b: a REAL runtime op injected into _c10d_functional must refuse
        arming typed (layer-1 set inequality), and re-arming after removal
        must succeed."""

        from torchlens.distributed._recognizer import UncapturedCollectiveOpError

        lifecycle.disarm()
        library = torch.library.Library("_c10d_functional", "FRAGMENT")
        try:
            library.define("census_fake_op(Tensor x) -> Tensor")
            with pytest.raises(UncapturedCollectiveOpError) as excinfo:
                lifecycle.arm()
            assert excinfo.value.fields["kind"] == "uncaptured_collective_op"
            assert excinfo.value.fields["layer"] == 1
        finally:
            library._destroy()
        record = lifecycle.arm()  # the ceiling lifts once the fake op is gone
        assert record.recognizer_snapshot

    def test_n3c_destination_read_before_observed_completion(self, single_rank_world):
        """N3c: async destination read -> typed unobserved fallback + the
        read_of_inflight_destination disclosure (shipped C0 records)."""

        dist = single_rank_world
        lifecycle.arm()

        class AsyncRead(nn.Module):
            def forward(self, x):
                doubled = x * 2
                work = dist.all_reduce(doubled, async_op=True)
                work.wait()
                return doubled + 1

        log = tl.trace(AsyncRead(), torch.randn(2, 4))
        info = next(op for op in log.ops if op.type == "allreduce").annotations["collective"]
        assert info["events"]["completion_binding"] == "unobserved"
        assert "read_of_inflight_destination" in info["disclosures"]

    def test_n3d_suppressed_completion_channel_is_unverifiable(self, single_rank_world):
        """N3d: the completion denominator is load-bearing -- suppressing the
        dispatcher counters while async traffic runs must miss the floor."""

        dist = single_rank_world
        import torch.distributed._functional_collectives as funcol

        lifecycle.arm()

        class FuncolAsync(nn.Module):
            def forward(self, x):
                doubled = x * 2
                reduced = funcol.all_reduce(doubled, "sum", dist.group.WORLD)
                return reduced + 1

        def completion_floor(events):
            return events.get("_c10d_functional::wait_tensor", 0) >= 1

        # FLIPPED FOR FIDELITY (C2 recording, plane S/W): the funcol call used
        # to be INVISIBLE to the C1 python-wrap capture -- the captured leg
        # completed with the provenance warning as its escape signal. The
        # funcol boundary wraps + the capture-scoped wait interposition now
        # record it first-class, so the escape signal is asserted GONE and the
        # boundary evidence asserted PRESENT (the row changed because capture
        # fidelity changed, never by editing the expectation alone).
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            control = run_census_row(
                "N3d",
                FuncolAsync,
                lambda: torch.ones(3),
                interposition=True,
                completion_floor=completion_floor,
                zi_gate_passed=True,
            )
        assert control.floors["completion"] == "met"
        assert control.completion_events["_c10d_functional::wait_tensor"] >= 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            log = tl.trace(FuncolAsync(), torch.ones(3))
        boundaries = log.annotations["distributed"]["boundaries"]
        funcol_entries = [
            entry
            for entry in boundaries
            if entry.get("schema") == "functional_collective_boundary_v0"
        ]
        assert len(funcol_entries) == 1 and funcol_entries[0]["kind"] == "all_reduce"
        # The ACT wait fired inside the captured forward (reduced + 1), so the
        # mode-independent completion authority must have observed it.
        assert funcol_entries[0]["events"]["completion_binding"] == "observed_wait"

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            suppressed = run_census_row(
                "N3d",
                FuncolAsync,
                lambda: torch.ones(3),
                suppress_interposition=True,
                completion_floor=completion_floor,
                zi_gate_passed=True,
            )
        assert suppressed.floors["completion"] == "missing"
        assert COMPLETION_FLOOR_MISSING in suppressed.failures
        assert not suppressed.row_green and suppressed.product_name == "red"


# ===========================================================================
# The wave-0 census report emission (plan 1.4).
# ===========================================================================


@pytest.mark.heavy
@requires_vetted_snapshot
class TestWave0ReportEmission:
    def test_wave0_report_end_to_end(self, tmp_path):
        """The wave-0 report emits end to end; its A1 row is armed in a live group.

        The armed leg runs inside a single-rank gloo group on purpose: armed state is
        dormant by design while no process group is initialized (plane-P observes only
        captures that could issue collectives), so a groupless armed leg would measure
        the unarmed path.
        """

        with _single_rank_group():
            lifecycle.arm()
            a1 = run_census_row(
                "A1",
                _dense_model,
                _dense_input,
                content_floor=lambda ops: any("mm" in op for op in ops),
                zi_gate_passed=True,
            )
        n1_calls = {"n": 0}

        def perturbing_factory():
            n1_calls["n"] += 1
            model = nn.Linear(4, 4)
            if n1_calls["n"] > 1:
                with torch.no_grad():
                    model.weight.add_(1.0)
            return model

        n1 = run_census_row("N1", perturbing_factory, lambda: torch.randn(2, 4))
        md_text, payload = generate_census_report(
            [a1, n1],
            torch_version=torch.__version__,
            repo_sha="wave0-test",
            md_path=tmp_path / "census-report.md",
            json_path=tmp_path / "census-report.json",
        )
        assert "row green" not in md_text
        assert WAVE0_PRODUCT_NAME in md_text and NOT_COVERED_LINE in md_text
        by_row = {entry["row"]: entry for entry in payload["rows"]}
        assert by_row["A1"]["product"] == WAVE0_PRODUCT_NAME
        assert by_row["N1"]["product"] == "red"
        not_run_ids = {row["row"] for row in payload["not_run"]}
        for cuda_row in ("CUDA-p2p-tags", "CUDA-multigpu-realism"):
            assert cuda_row in not_run_ids
        assert payload["interposed_completion_ops"] == list(INTERPOSED_COMPLETION_OPS)
