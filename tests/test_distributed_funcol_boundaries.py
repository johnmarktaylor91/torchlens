"""Funcol boundary capture + plane-W completion authority: gloo sims.

Merge-ranks C2 recording slice (design-merge-ranks-c v5 5.2 planes S/W; L8
census plan). Pins the closure of the wave-0 pinned gap -- funcol collectives
used to be INVISIBLE to the C1 python-wrap capture -- plus the fail-closed
edges: ACT never materialized, unwaited completions disclosed typed, digest
witnesses only at observed completions, merge/runnable/fastlog refusal parity,
interposition lifetime, and disarm restoration.
"""

from __future__ import annotations

import gc
import warnings
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl  # noqa: E402
from torchlens._distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens.backends.torch.funcol import (  # noqa: E402
    FUNCOL_BOUNDARY_SCHEMA,
    active_funcol_session,
)
from torchlens.distributed import is_vetted_build  # noqa: E402
from torchlens.errors._base import CompatibilityError  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.is_vetted_build());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Applied to the classes/tests that call a real arm(); test_unarmed_capture_
# is_status_quo_for_funcol and test_dead_session_kernel_executes_real_wait_
# and_skips_bookkeeping deliberately exercise the UNARMED/no-session path and
# must keep running everywhere.
requires_vetted_snapshot = pytest.mark.skipif(
    not is_vetted_build(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.is_vetted_build() is False here)",
)


@pytest.fixture()
def gloo_world(tmp_path):
    """Single-process gloo world, armed state guaranteed clean around a test."""

    import torch.distributed as dist

    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


def _funcol():
    import torch.distributed._functional_collectives as funcol_module

    return funcol_module


class WaitedFuncol(nn.Module):
    """Forward whose funcol destination is materialized inside the capture."""

    def forward(self, x):
        funcol_module = _funcol()
        doubled = x * 2
        reduced = funcol_module.all_reduce(doubled, "sum", torch.distributed.group.WORLD)
        return reduced + 1


class NeverWaitedFuncol(nn.Module):
    """Forward that issues a funcol collective and never touches its result."""

    def forward(self, x):
        funcol_module = _funcol()
        doubled = x * 2
        _inflight = funcol_module.all_reduce(doubled, "sum", torch.distributed.group.WORLD)
        return doubled + 1


def _funcol_boundaries(log):
    return [
        entry
        for entry in log.annotations["distributed"]["boundaries"]
        if entry.get("schema") == FUNCOL_BOUNDARY_SCHEMA
    ]


@pytest.mark.heavy
@requires_vetted_snapshot
class TestFuncolBoundaryNode:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_dtensor.py.
    def test_funcol_becomes_boundary_node_no_provenance_escape(self, gloo_world):
        """The pinned gap, closed: boundary node + parenting, zero escape signals."""

        lifecycle.arm()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            log = tl.trace(WaitedFuncol(), torch.ones(3))
        labels = [op.layer_label for op in log.ops]
        assert any(label.startswith("funcolallreduce") for label in labels)
        add_op = next(op for op in log.ops if op.layer_label.startswith("add"))
        assert any("funcolallreduce" in parent for parent in add_op.parents)
        assert torch.equal(log.output_ops[0].out, torch.full((3,), 3.0))

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_boundary_payload_shape_and_observed_completion(self, gloo_world):
        """v0 payload: correlation key, funcol event mapping, plane-W binding."""

        lifecycle.arm()
        log = tl.trace(WaitedFuncol(), torch.ones(3))
        entries = _funcol_boundaries(log)
        assert len(entries) == 1
        payload = entries[0]
        assert payload["kind"] == "all_reduce"
        assert payload["func"].endswith("_functional_collectives.all_reduce")
        assert payload["correlation"]["channel"] == "coll"
        assert isinstance(payload["correlation"]["seq"], int)
        assert payload["events"]["event_model"] == "funcol_issue_is_launch"
        assert payload["events"]["completion_interposition"] == "installed"
        # The ACT wait fired inside the forward: the dispatcher-level
        # completion authority must have observed it, mode stack or not.
        assert payload["events"]["completion_binding"] == "observed_wait"
        assert payload["events"]["destination_completions"] == [True]
        assert "read_of_inflight_destination" in payload["disclosures"]
        assert payload["reduce_op"] == "sum"
        roles = payload["roles"]
        assert {entry["role"] for entry in roles} == {"contribution", "destination"}

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_never_waited_settles_unobserved_with_typed_disclosure(self, gloo_world):
        """v5 1.4b: never-waited funcol = unobserved + unwitnessed disclosure."""

        lifecycle.arm()
        log = tl.trace(NeverWaitedFuncol(), torch.ones(3))
        payload = _funcol_boundaries(log)[0]
        assert payload["events"]["completion_binding"] == "unobserved"
        assert "async_unwaited_output_unwitnessed" in payload["disclosures"]
        funcol_op = next(op for op in log.ops if "funcolallreduce" in op.layer_label)
        assert (
            "async_unwaited_output_unwitnessed"
            in funcol_op.annotations["collective"]["disclosures"]
        )

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/funcol.py, campaign mutant kills.
    def test_chained_funcol_parents_and_blocked_act_witness(self, gloo_world):
        """ACT contribution: parenting via the label chokepoint, digest withheld."""

        dist = gloo_world

        class Chained(nn.Module):
            def forward(self, x):
                funcol_module = _funcol()
                first = funcol_module.all_reduce(x * 2, "sum", dist.group.WORLD)
                second = funcol_module.all_reduce(first, "max", dist.group.WORLD)
                return second + 1

        lifecycle.arm()
        log = tl.trace(
            Chained(),
            torch.ones(3),
            capture=tl.options.CaptureOptions(distributed_witness="digest"),
        )
        entries = _funcol_boundaries(log)
        assert len(entries) == 2
        funcol_ops = [op for op in log.ops if "funcolallreduce" in op.layer_label]
        assert funcol_ops[0].layer_label in funcol_ops[1].parents
        # The second boundary's contribution was an unwaited ACT: digesting it
        # would have forced completion, so the witness is withheld typed.
        assert "witness_would_force_completion" in entries[1]["disclosures"]
        assert entries[1]["witness"]["contribution_digests"] is None

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_digest_witness_destination_digest_only_after_observed_wait(self, gloo_world):
        """Destination digests exist iff the completion was observed (v5 payload policy)."""

        lifecycle.arm()
        options = tl.options.CaptureOptions(distributed_witness="digest")
        waited = tl.trace(WaitedFuncol(), torch.ones(3), capture=options)
        waited_payload = _funcol_boundaries(waited)[0]
        assert waited_payload["witness"]["contribution_digests"]
        assert waited_payload["witness"]["destination_digests"] == [
            waited_payload["witness"]["destination_digests"][0]
        ]
        assert waited_payload["witness"]["destination_digests"][0] is not None
        assert waited_payload["witness"]["not_present_reason"] is None

        unwaited = tl.trace(NeverWaitedFuncol(), torch.ones(3), capture=options)
        unwaited_payload = _funcol_boundaries(unwaited)[0]
        assert unwaited_payload["witness"]["destination_digests"] == [None]
        assert unwaited_payload["witness"]["not_present_reason"] == "async_completion_unobserved"


@pytest.mark.heavy
class TestPlaneWLifetime:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/funcol.py, campaign mutant kills.
    @requires_vetted_snapshot
    def test_session_cleared_and_interposition_torn_down(self, gloo_world):
        """The capture-scoped session never survives the capture window."""

        lifecycle.arm()
        assert active_funcol_session() is None
        log = tl.trace(WaitedFuncol(), torch.ones(3))
        assert active_funcol_session() is None
        assert _funcol_boundaries(log)
        # Post-capture funcol traffic runs on the pristine dispatcher path.
        funcol_module = _funcol()
        out = funcol_module.all_reduce(torch.ones(2), "sum", gloo_world.group.WORLD)
        assert float((out + 1).sum()) == 4.0

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    def test_unarmed_capture_is_status_quo_for_funcol(self, gloo_world, monkeypatch):
        """Arming relaxes nothing and unarmed captures stay pre-C2: no journal."""

        from torchlens._distributed import _lifecycle as lifecycle_module

        lifecycle.disarm()
        # Force the lazy auto-arm off so this capture is genuinely unarmed.
        monkeypatch.setattr(lifecycle_module, "maybe_auto_arm", lambda: None)
        with pytest.warns(UserWarning, match="no graph/source provenance"):
            log = tl.trace(WaitedFuncol(), torch.ones(3))
        assert "distributed" not in log.annotations

    # Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
    @requires_vetted_snapshot
    def test_disarm_restores_funcol_functions(self, gloo_world):
        """Every wrapped funcol attr is restored to its pristine function."""

        funcol_module = _funcol()
        pristine = funcol_module.all_reduce
        lifecycle.arm()
        assert getattr(funcol_module.all_reduce, "__tl_distributed_wrap__", False)
        assert funcol_module.all_reduce.__wrapped__ is pristine
        lifecycle.disarm()
        assert funcol_module.all_reduce is pristine

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/funcol.py, campaign mutant kills.
    @requires_vetted_snapshot
    def test_group_resolver_capability_gap_refuses_typed(self, gloo_world, monkeypatch):
        """Fail-closed: no resolvable group -> typed refusal, never a silent omission."""

        from torchlens._distributed import DistributedCaptureUnsupportedError
        from torchlens.backends.torch import funcol as funcol_backend

        lifecycle.arm()
        monkeypatch.setattr(
            funcol_backend, "_resolve_funcol_process_group", lambda group, tag: None
        )
        with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
            tl.trace(WaitedFuncol(), torch.ones(3))
        kinds = {finding.kind for finding in excinfo.value.fields["findings"]}
        assert kinds == {"uncaptured_collective_op"}


@pytest.mark.heavy
@requires_vetted_snapshot
class TestFuncolRefusalParity:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/funcol.py, campaign mutant kills.
    def test_fastlog_funcol_refuses_instead_of_dropping_journal(self, gloo_world):
        """tl.record cannot represent the boundary journal: typed refusal."""

        lifecycle.arm()
        with pytest.raises(CompatibilityError) as excinfo:
            tl.record(WaitedFuncol(), torch.ones(3), save=tl.func("mul"))
        assert excinfo.value.fields["kind"] == "collective_boundary_fastlog_unsupported"


@pytest.mark.heavy
@requires_vetted_snapshot
class TestFuncolPersistence:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of _io/_load_validators.py, _merged/_evidence.py, campaign mutant kills.
    def test_save_load_round_trips_the_boundary_disclosure(self, gloo_world, tmp_path):
        """The v0 payload survives save/load; refusal parity holds on the load."""

        from torchlens._merged._errors import MergeInputError
        from torchlens._merged._evidence import extract_rank_evidence

        lifecycle.arm()
        log = tl.trace(WaitedFuncol(), torch.ones(3))
        path = tmp_path / "funcol-core.tlspec"
        log.save(str(path))
        loaded = tl.load(str(path))
        entries = _funcol_boundaries(loaded)
        assert len(entries) == 1
        assert entries[0]["events"]["completion_binding"] == "observed_wait"
        funcol_op = next(op for op in loaded.ops if "funcolallreduce" in op.layer_label)
        assert funcol_op.annotations["collective"]["schema"] == FUNCOL_BOUNDARY_SCHEMA
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(loaded, str(path))
        assert excinfo.value.fields["reason"] == "functional_collective_boundary_unsupported"


@pytest.mark.heavy
class TestFuncolSessionLeak:
    """The armed-capture Trace leak, closed (fix/funcol-session-leak).

    The dispatcher can retain the ``wait_tensor`` kernel past
    ``Library._destroy()``; the kernel used to close over the session
    strongly, so ``session.trace`` pinned every armed process's LATEST Trace
    -- retained activations included -- for the life of the process, with
    zero gc-visible referrers. The kernel now holds the session by weakref;
    these tests pin collectability, the intact live path, and the honest
    dead-ref (leaked-kernel) behavior.
    """

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/funcol.py.
    def test_dead_session_kernel_executes_real_wait_and_skips_bookkeeping(self):
        """Pinned dead-ref behavior: redispatch the REAL wait, record nothing.

        A leaked kernel outliving its capture must never swallow or refuse a
        user's collective completion; the owning capture already settled
        fail-closed (``completion_binding="unobserved"``), so skipping the
        dead session's bookkeeping is the honest degradation.
        """

        from torchlens._utils._torch_compat import get_funcol_wait_redispatch
        from torchlens.backends.torch.funcol import (
            _FuncolCaptureSession,
            _make_observed_wait_tensor,
        )

        redispatch = get_funcol_wait_redispatch()
        if redispatch is None:
            pytest.skip("funcol wait interposition capability absent on this build")
        wait_op, exclude_cpu_guard = redispatch
        session = _FuncolCaptureSession(trace=object())
        session_ref = weakref.ref(session)
        del session
        gc.collect()
        assert session_ref() is None
        kernel = _make_observed_wait_tensor(session_ref, wait_op, exclude_cpu_guard)
        tensor = torch.ones(3)
        result = kernel(tensor)
        assert torch.equal(result, tensor)
