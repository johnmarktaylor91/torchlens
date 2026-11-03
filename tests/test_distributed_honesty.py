"""Honest distributed detection: compat rows plus a typed capture refusal.

Before this guard, a tensor-parallel model traced "successfully" while reporting
``0 total modules`` and ``0 params total; 0 B`` -- the ``linear`` op replaced by
bare ``viewas`` nodes -- and ``tl.report.doctor`` reported all-pass for the same
model. These tests pin the honest replacement: four report rows and a typed
refusal at capture entry.

Two layers of coverage, both exercising the real detection code path:

* Synthesized-signal unit tests build tensor subclasses and mesh/stage stand-ins
  and feed them to the production
  :func:`torchlens._distributed.detect_distributed_state`. They cover the
  structural fallback branches that fire on torch builds where an exact class
  probe is unavailable, and they need no process group.
* Real single-rank tests initialize a one-process ``gloo`` group, build genuine
  ``DTensor`` parameters via ``parallelize_module``, and assert the exact
  ``isinstance`` branch plus the end-to-end refusal. Single-process and CPU-only:
  no multi-GPU, no ``torchrun``.

There are no test-only branches in the production code; the synthesized objects
are ordinary Python objects that the real classifier inspects.
"""

from __future__ import annotations

import os
import socket
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._distributed import (
    REFUSING_KINDS,
    DistributedCaptureUnsupportedError,
    DistributedFinding,
    detect_distributed_state,
)
from torchlens._utils import _torch_compat
from torchlens.backends.torch._ops_interventions import (
    _pop_tensor_live_fire_results,
    _set_tensor_live_fire_results,
)
from torchlens.distributed import is_vetted_build

DISTRIBUTED_ROW_KEYS = ("dtensor", "device_mesh", "tensor_parallel", "pipeline_parallel")


@pytest.fixture(autouse=True)
def _restore_pipelining_probe_state() -> Iterator[None]:
    """Restore the pipelining capability-probe cache poisoned by module stubs.

    Several tests here stub ``torch.distributed.pipelining`` with an EMPTY
    module to synthesize pipeline-parallel state. The lazy probe in
    ``get_pipelining_module_types`` then sees the stub in ``sys.modules``,
    finds no stage types, and caches ``HAS_PIPELINING=False`` process-globally.
    ``monkeypatch`` restores ``sys.modules`` but not that cache, which polluted
    every later capability snapshot in the session: the generated
    ``docs/method_x_model_compatibility.md`` gate rendered a phantom
    ``missing=HAS_PIPELINING`` non-pass row, and the clean-model no-warning
    guard in ``test_robustness_pr2`` caught a stray ``TorchCapabilityWarning``
    (the conftest warn-once reset re-arms the warning per test).

    Yields
    ------
    None
        Runs the test, then restores the pre-test probe state.
    """

    saved = {
        name: getattr(_torch_compat, name)
        for name in ("_PIPELINING_PROBED", "_PIPELINING_TYPES", "HAS_PIPELINING")
    }
    yield
    for name, value in saved.items():
        setattr(_torch_compat, name, value)


class TinyModel(nn.Module):
    """Minimal model with one linear layer, used as the capture subject."""

    def __init__(self) -> None:
        """Initialize the single linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Activated linear output.
        """

        return torch.relu(self.fc(x))


class _FakeDTensor(torch.Tensor):
    """Tensor subclass standing in for ``DTensor`` on the structural path."""


# The structural fallback keys on the *type identity* a real DTensor has: a
# tensor subclass named DTensor defined under a torch.distributed namespace.
# Renaming the stand-in exercises that branch without a process group.
_FakeDTensor.__name__ = "DTensor"
_FakeDTensor.__qualname__ = "DTensor"
_FakeDTensor.__module__ = "torch.distributed._tensor_stub_for_tests"


class _FakeShardedTensor(torch.Tensor):
    """Tensor subclass standing in for ``ShardedTensor``."""


_FakeShardedTensor.__name__ = "ShardedTensor"
_FakeShardedTensor.__qualname__ = "ShardedTensor"
_FakeShardedTensor.__module__ = "torch.distributed._shard_stub_for_tests"


class _FakePipelineStage:
    """Stand-in for a pipeline-parallel stage object held as a module attribute."""


_FakePipelineStage.__module__ = "torch.distributed.pipelining.stage_stub_for_tests"


def _fake_dtensor(shape: tuple[int, ...] = (2, 2)) -> torch.Tensor:
    """Return a ``_FakeDTensor`` view over a dense tensor.

    Parameters
    ----------
    shape:
        Shape of the underlying dense tensor.

    Returns
    -------
    torch.Tensor
        Tensor whose type matches the DTensor structural signature.
    """

    return torch.zeros(shape).as_subclass(_FakeDTensor)


def _find(findings: tuple[DistributedFinding, ...], kind: str) -> DistributedFinding:
    """Return the single finding of ``kind``.

    Parameters
    ----------
    findings:
        Findings returned by detection.
    kind:
        Finding kind to select.

    Returns
    -------
    DistributedFinding
        Matching finding.
    """

    matches = [finding for finding in findings if finding.kind == kind]
    assert len(matches) == 1, f"expected exactly one {kind} finding, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Clean single-process models stay clean
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
def test_failed_initialized_probe_refuses_instead_of_skipping_mesh_scan(monkeypatch) -> None:
    """A failed is_initialized() probe is a scan gap, never "not initialized".

    The probe's catch-all used to swallow EVERY exception and return False,
    which silently disabled the device-mesh attribute scan FAIL-OPEN: a
    process whose distributed runtime was broken mid-probe reported zero
    findings instead of disclosing that absence of sharded state could not
    be established.
    """

    def broken_probe() -> bool:
        raise RuntimeError("capability probe failed")

    monkeypatch.setattr(torch.distributed, "is_initialized", broken_probe)
    findings = detect_distributed_state(TinyModel(), torch.randn(2, 4))
    scan = _find(findings, "scan_incomplete")
    assert scan.refuses_capture is True
    assert "torch.distributed.is_initialized()" in scan.sites

    import torchlens._distributed as distributed_mod

    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        distributed_mod.check_distributed_capture(TinyModel(), torch.randn(2, 4))
    assert "scan_incomplete" in [finding.kind for finding in excinfo.value.fields["findings"]]


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
def test_unreadable_module_scan_refuses_instead_of_root_only_fallback(monkeypatch) -> None:
    """A failed child-module traversal cannot be treated as a complete root scan."""

    class UnreadableModules(TinyModel):
        """Model that rejects recursive module enumeration."""

        def named_modules(self, *args: Any, **kwargs: Any) -> Any:
            """Refuse module enumeration."""

            _ = args, kwargs
            raise RuntimeError("modules unavailable")

    import torchlens._distributed as distributed_mod

    monkeypatch.setattr(distributed_mod, "_distributed_namespace_imported", lambda: True)
    monkeypatch.setattr(distributed_mod, "_distributed_initialized", lambda: True)
    findings = detect_distributed_state(UnreadableModules(), torch.randn(2, 4))
    assert _find(findings, "scan_incomplete").refuses_capture is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_fire_results_survive_tensor_attribute_rejection() -> None:
    """Intervention evidence falls back out of band when tensor attrs reject writes."""

    class RejectingTensor(torch.Tensor):
        """Tensor subclass that rejects TorchLens fire-result attributes."""

        def __setattr__(self, name: str, value: Any) -> None:
            """Reject only the transient intervention-evidence attribute."""

            if name == "_tl_live_fire_results":
                raise RuntimeError("dynamic attributes disabled")
            super().__setattr__(name, value)

    tensor = torch.ones(2).as_subclass(RejectingTensor)
    marker = object()
    _set_tensor_live_fire_results(tensor, (marker,))  # type: ignore[arg-type]
    assert _pop_tensor_live_fire_results(tensor) == (marker,)


# ---------------------------------------------------------------------------
# Synthesized-signal detection (structural fallback branches)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
def test_synthesized_sharded_tensor_is_detected_as_dtensor_kind() -> None:
    """ShardedTensor state shares the refusing ``dtensor`` kind."""

    model = TinyModel()
    model.register_buffer("shard", _FakeShardedTensor(), persistent=True)
    model._buffers["shard"] = torch.zeros(2, 2).as_subclass(_FakeShardedTensor)

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "dtensor")
    assert "shard" in finding.sites
    assert "ShardedTensor" in finding.detail
    assert finding.refuses_capture is True


def test_refusing_kinds_is_the_single_source_of_truth() -> None:
    """Every active distributed execution mode that omits work refuses."""

    assert (
        frozenset({"dtensor", "tensor_parallel", "pipeline_parallel", "scan_incomplete"})
        == REFUSING_KINDS
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
def test_site_list_is_bounded_with_explicit_remainder() -> None:
    """Long site lists are truncated but never silently: the remainder is counted."""

    finding = DistributedFinding(
        kind="dtensor",
        detail="detail",
        suggestion="suggestion",
        sites=tuple(f"p{index}" for index in range(9)),
    )
    described = finding.describe_sites()
    assert described.startswith("p0, p1, p2, p3, p4")
    assert "(+4 more)" in described


# ---------------------------------------------------------------------------
# Report wiring
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _utils/_torch_compat.py, report/_environment.py.
def test_non_refusing_row_is_warning_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported-but-not-refused condition must not claim capture refuses."""

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stages = [_FakePipelineStage()]  # type: ignore[assignment]
    refusing_row = tl.report.doctor(model, torch.randn(2, 4)).check("pipeline_parallel")
    assert refusing_row.severity == "error"

    clean_row = tl.report.doctor(TinyModel(), torch.randn(2, 4)).check("device_mesh")
    assert clean_row.severity == "ok"
    assert "DistributedCaptureUnsupportedError" not in clean_row.details

    dtensor_clear = tl.report.doctor(TinyModel(), torch.randn(2, 4)).check("dtensor")
    assert "bounded entry scan" in dtensor_clear.details
    assert "slots-only" in dtensor_clear.details

    tp_clear = tl.report.doctor(TinyModel(), torch.randn(2, 4)).check("tensor_parallel")
    assert "user-wrapped or opaque hook" in tp_clear.details


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
def test_report_renders_with_distributed_rows() -> None:
    """Both renderers include the new rows without raising."""

    model = TinyModel()
    model.fc.weight = nn.Parameter(_fake_dtensor((4, 4)), requires_grad=False)
    report = tl.report.doctor(model, torch.randn(2, 4))
    assert "DTensor / sharded tensors" in report.show()
    assert "Tensor parallel (TP)" in report.to_markdown()


# Kept by shrink-r4 (per-id subset rule): sole coverage of errors/__init__.py.
def test_typed_error_is_reachable_from_public_errors_namespace() -> None:
    """The refusal type is part of the public ``torchlens.errors`` surface."""

    import torchlens.errors as errors

    assert errors.DistributedCaptureUnsupportedError is DistributedCaptureUnsupportedError
    assert "DistributedCaptureUnsupportedError" in errors.__all__
    assert "DistributedCaptureUnsupportedError" in dir(errors)
    assert issubclass(DistributedCaptureUnsupportedError, errors.CompatibilityError)


# ---------------------------------------------------------------------------
# Real single-rank DTensor (exact isinstance branch, one process, CPU only)
# ---------------------------------------------------------------------------


@pytest.fixture
def single_rank_cpu_mesh() -> Iterator[object]:
    """Yield a real one-rank CPU device mesh, or skip when gloo is unavailable.

    Yields
    ------
    object
        Initialized ``DeviceMesh`` over a single CPU rank.

    Notes
    -----
    One process, one rank, ``gloo``, CPU: this exercises the genuine DTensor
    code path without ``torchrun`` or any GPU. The process group is torn down
    unconditionally so a failure here cannot leak global distributed state into
    later tests.
    """

    torch_distributed = pytest.importorskip("torch.distributed")
    if not torch_distributed.is_available():
        pytest.skip("torch.distributed is not available in this build")
    if torch_distributed.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:  # pragma: no cover - torch without device_mesh
        pytest.skip("torch.distributed.device_mesh is unavailable")

    previous_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # OS-assigned ephemeral port: a hardcoded port (29577 historically)
    # collides across parallel lanes/worktrees on one box, and the collision
    # lands in the except-skip below -- distributed coverage silently degrades
    # to SKIP instead of failing loudly (T14-3).
    with socket.socket() as _probe:
        _probe.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(_probe.getsockname()[1])
    try:
        torch_distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception as exc:  # pragma: no cover - sandboxes without loopback
        pytest.skip(f"single-rank gloo process group unavailable: {exc}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        if torch_distributed.is_initialized():
            torch_distributed.destroy_process_group()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Kept by shrink-r4 (per-id subset rule): sole coverage of _distributed.py.
@pytest.mark.heavy
def test_real_dtensor_in_custom_input_container_is_detected(
    single_rank_cpu_mesh: object,
) -> None:
    """A real DTensor inside user container state reaches the typed boundary."""

    tensor_api = pytest.importorskip("torch.distributed.tensor")

    class Box:
        """User input container holding a real DTensor."""

        def __init__(self, value: torch.Tensor) -> None:
            """Store the wrapped tensor.

            Parameters
            ----------
            value:
                DTensor input payload.
            """

            self.value = value

    value = tensor_api.distribute_tensor(
        torch.randn(2, 4),
        single_rank_cpu_mesh,
        placements=[tensor_api.Replicate()],
    )
    finding = _find(detect_distributed_state(nn.Identity(), Box(value)), "dtensor")
    assert finding.exact is True
    assert finding.sites == ("input.value",)
    with pytest.raises(DistributedCaptureUnsupportedError):
        tl.trace(nn.Identity(), Box(value))


@pytest.mark.heavy
def test_real_tensor_parallel_capture_refuses_instead_of_lying(
    single_rank_cpu_mesh: object,
) -> None:
    """The regression this guard exists for: capture must refuse, not mis-report.

    Historically this exact model traced "successfully" while reporting zero
    modules and zero parameters, with the linear op absent from the graph.
    """

    parallel = pytest.importorskip("torch.distributed.tensor.parallel")
    model = TinyModel()
    parallel.parallelize_module(model, single_rank_cpu_mesh, {"fc": parallel.ColwiseParallel()})
    x = torch.randn(2, 4)

    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        tl.trace(model, x)
    message = str(excinfo.value)
    assert "fc.weight" in message
    assert "LIMITATIONS.md" in message
    assert [item.kind for item in excinfo.value.fields["findings"]] == [
        "dtensor",
        "tensor_parallel",
    ]

    row = tl.report.doctor(model, x).check("dtensor")
    assert row.is_detected is True
    assert row.severity == "error"
    assert "Detected structurally" not in row.details, "real DTensor is an exact match"


# Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_recognizer.py.
@pytest.mark.heavy
def test_real_dense_model_still_captures_with_distributed_initialized(
    single_rank_cpu_mesh: object,
) -> None:
    """An initialized process group must not disturb ordinary dense capture.

    ``tl.trace`` still succeeds either way. On a census-vetted torch build
    ``maybe_auto_arm`` arms silently and no disclosure fires; on an unvetted
    build (F1, Lead ruling 2026-10-01) it degrades to unarmed capture and
    WARNS on every capture entry -- the correct, honest product behavior
    (see ``test_distributed_boundary_gloo.py::TestUnvettedTorchRefusesArming``),
    not a gap to route around. The suite's
    ``error::UserWarning:torchlens`` filter promotes that warning to a hard
    failure unless it is explicitly expected here.
    """

    model = TinyModel()
    x = torch.randn(2, 4)
    assert detect_distributed_state(model, x) == ()
    if is_vetted_build():
        trace = tl.trace(model, x)
    else:
        with pytest.warns(UserWarning, match="uncaptured_collective_op"):
            trace = tl.trace(model, x)
    assert trace.num_params == sum(p.numel() for p in model.parameters())
    assert any("linear" in label for label in trace.layer_labels)
