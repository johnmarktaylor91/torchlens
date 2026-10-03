"""Honest distributed detection: compat rows plus a typed capture refusal.

Before this guard, a tensor-parallel model traced "successfully" while reporting
``0 total modules`` and ``0 params total; 0 B`` -- the ``linear`` op replaced by
bare ``viewas`` nodes -- and ``tl.compat.report`` reported all-pass for the same
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
from torchlens.backends.torch._ops_interventions import (
    _pop_tensor_live_fire_results,
    _set_tensor_live_fire_results,
)
from torchlens.distributed import has_vetted_snapshot
from torchlens.utils import _torch_compat

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


def _finding_kinds(findings: tuple[DistributedFinding, ...]) -> set[str]:
    """Return the set of finding kinds.

    Parameters
    ----------
    findings:
        Findings returned by detection.

    Returns
    -------
    set[str]
        Kinds present.
    """

    return {finding.kind for finding in findings}


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


def test_dense_model_reports_no_distributed_findings() -> None:
    """An ordinary dense model must produce no findings at all."""

    model = TinyModel()
    assert detect_distributed_state(model, torch.randn(2, 4)) == ()


def test_dense_model_rows_all_pass_and_capture_succeeds() -> None:
    """All four rows read ``pass``/``ok`` and capture is unaffected."""

    model = TinyModel()
    x = torch.randn(2, 4)
    report = tl.compat.report(model, x)
    for key in DISTRIBUTED_ROW_KEYS:
        row = report.row(key)
        assert row.status == "pass", f"{key} status"
        assert row.severity == "ok", f"{key} severity"
        assert row.detected is False, f"{key} detected"
        assert row.suggestion == "", f"{key} suggestion"

    trace = tl.trace(model, x)
    assert trace.num_params > 0
    assert any("linear" in label for label in trace.layer_labels)


def test_all_four_rows_are_always_present() -> None:
    """The rows exist whether or not anything was detected."""

    report = tl.compat.report(TinyModel(), torch.randn(2, 4))
    keys = {row.key for row in report.rows}
    for key in DISTRIBUTED_ROW_KEYS:
        assert key in keys


def test_unreadable_state_scan_refuses_instead_of_passing(monkeypatch) -> None:
    """Accessor failures produce a typed scan-incomplete refusal."""

    class UnreadableModel(TinyModel):
        """Model whose distributed state accessors require an unavailable context."""

        def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
            """Refuse both supported named-parameter call spellings."""

            _ = args, kwargs
            raise RuntimeError("state requires summon context")

    import torchlens._distributed as distributed_mod

    monkeypatch.setattr(distributed_mod, "_distributed_namespace_imported", lambda: True)
    monkeypatch.setattr(distributed_mod, "_sharded_tensor_namespace_imported", lambda: True)
    findings = detect_distributed_state(UnreadableModel(), torch.randn(2, 4))
    scan = _find(findings, "scan_incomplete")
    assert scan.refuses_capture is True
    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        distributed_mod.check_distributed_capture(UnreadableModel(), torch.randn(2, 4))
    assert [finding.kind for finding in excinfo.value.fields["findings"]] == ["scan_incomplete"]


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


def test_synthesized_dtensor_parameter_is_detected_and_refuses() -> None:
    """A DTensor-shaped parameter is detected, named, and refuses capture."""

    model = TinyModel()
    model.fc.weight = nn.Parameter(_fake_dtensor((4, 4)), requires_grad=False)

    findings = detect_distributed_state(model, torch.randn(2, 4))
    assert "dtensor" in _finding_kinds(findings)
    finding = _find(findings, "dtensor")
    assert finding.refuses_capture is True
    assert "fc.weight" in finding.sites
    # Structural match, not an exact isinstance against torch's own class.
    assert finding.exact is False
    assert finding.suggestion

    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        tl.trace(model, torch.randn(2, 4))
    assert "fc.weight" in str(excinfo.value)
    assert [item.kind for item in excinfo.value.fields["findings"]] == ["dtensor"]


def test_synthesized_dtensor_input_is_detected() -> None:
    """A DTensor-shaped *input* is detected and named by access path."""

    model = TinyModel()
    findings = detect_distributed_state(model, _fake_dtensor((2, 4)))
    finding = _find(findings, "dtensor")
    assert finding.sites == ("input",)
    assert finding.refuses_capture is True


def test_synthesized_dtensor_in_nested_input_container_is_detected() -> None:
    """Nested list/dict inputs are walked and reported with their access path."""

    payload = [torch.randn(2, 4), {"weights": _fake_dtensor((2, 4))}]
    findings = detect_distributed_state(TinyModel(), payload)
    finding = _find(findings, "dtensor")
    assert finding.sites == ("input[1]['weights']",)


def test_synthesized_dtensor_in_custom_input_container_is_detected() -> None:
    """User-defined input containers cannot hide refusing distributed tensors."""

    class Box:
        """Simple user container exposing a tensor through instance state."""

        def __init__(self, value: torch.Tensor) -> None:
            """Store the wrapped tensor.

            Parameters
            ----------
            value:
                Tensor payload.
            """

            self.value = value

    finding = _find(
        detect_distributed_state(TinyModel(), Box(_fake_dtensor((2, 4)))),
        "dtensor",
    )
    assert finding.sites == ("input.value",)


def test_synthesized_dtensor_plain_module_attribute_is_detected() -> None:
    """An unregistered tensor attribute must be covered by the typed refusal."""

    model = TinyModel()
    model.unregistered_shard = _fake_dtensor((2, 4))  # type: ignore[assignment]

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "dtensor")
    assert finding.sites == ("<root>.unregistered_shard",)
    with pytest.raises(DistributedCaptureUnsupportedError):
        tl.trace(model, torch.randn(2, 4))


def test_synthesized_dtensor_keyword_input_is_detected() -> None:
    """Keyword-argument inputs are walked too."""

    findings = detect_distributed_state(TinyModel(), None, {"mask": _fake_dtensor((2, 4))})
    finding = _find(findings, "dtensor")
    assert finding.sites == ("input['mask']",)


def test_synthesized_sharded_tensor_is_detected_as_dtensor_kind() -> None:
    """ShardedTensor state shares the refusing ``dtensor`` kind."""

    model = TinyModel()
    model.register_buffer("shard", _FakeShardedTensor(), persistent=True)
    model._buffers["shard"] = torch.zeros(2, 2).as_subclass(_FakeShardedTensor)

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "dtensor")
    assert "shard" in finding.sites
    assert "ShardedTensor" in finding.detail
    assert finding.refuses_capture is True


def test_synthesized_pipeline_stage_attribute_is_detected_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline-stage object on a module attribute refuses capture."""

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stage = _FakePipelineStage()  # type: ignore[assignment]

    findings = detect_distributed_state(model, torch.randn(2, 4))
    finding = _find(findings, "pipeline_parallel")
    assert finding.refuses_capture is True
    assert finding.sites == ("<root>.stage",)
    assert finding.exact is False

    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        tl.trace(model, torch.randn(2, 4))
    assert "pipeline_parallel" in str(excinfo.value)


def test_pipeline_stage_inside_container_attribute_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage held in a plain list attribute is still found."""

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stages = [_FakePipelineStage()]  # type: ignore[assignment]

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "pipeline_parallel")
    assert finding.sites == ("<root>.stages[0]",)


def test_pipeline_stage_after_the_eighth_container_item_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection cannot silently stop before a later pipeline stage."""

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stages = [object() for _ in range(8)] + [_FakePipelineStage()]  # type: ignore[assignment]

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "pipeline_parallel")
    assert finding.sites == ("<root>.stages[8]",)


def test_pipeline_stage_in_a_nested_custom_container_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested user containers cannot hide pipeline-stage state."""

    class StageBox:
        """User container holding a nested pipeline stage."""

        def __init__(self) -> None:
            """Build the nested state."""

            self.payload = {"nested": [_FakePipelineStage()]}

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stage_box = StageBox()  # type: ignore[assignment]

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "pipeline_parallel")
    assert finding.sites == ("<root>.stage_box.payload['nested'][0]",)


def test_module_internal_attributes_are_not_scanned() -> None:
    """Torch's own bookkeeping attributes never produce findings.

    They cannot hold user objects, and skipping them is what keeps capture-entry
    detection off the cost budget for large models.
    """

    model = TinyModel()
    assert detect_distributed_state(model, torch.randn(2, 4)) == ()
    # The skip set is probed from a live bare module, so it must be non-trivial.
    from torchlens._distributed import _NN_MODULE_INTERNAL_ATTRS

    assert "_parameters" in _NN_MODULE_INTERNAL_ATTRS
    assert "_modules" in _NN_MODULE_INTERNAL_ATTRS
    assert "training" in _NN_MODULE_INTERNAL_ATTRS


def test_properties_are_never_evaluated_during_detection() -> None:
    """Detection reads instance ``__dict__`` only, never triggering descriptors."""

    calls: list[str] = []

    class ExplodingProperty(nn.Module):
        """Module whose property must not be evaluated by a compat probe."""

        def __init__(self) -> None:
            """Initialize the wrapped linear layer."""

            super().__init__()
            self.fc = nn.Linear(4, 4)

        @property
        def landmine(self) -> str:
            """Record evaluation and fail the test if reached.

            Returns
            -------
            str
                Never returned in a passing run.
            """

            calls.append("landmine")
            raise AssertionError("detection must not evaluate properties")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one forward pass.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Linear output.
            """

            return self.fc(x)

    assert detect_distributed_state(ExplodingProperty(), torch.randn(2, 4)) == ()
    assert calls == []


def test_class_merely_named_like_a_stage_in_user_module_is_not_detected() -> None:
    """Namespace anchoring, not name matching: a user class is not a false positive."""

    class PipelineStage:
        """User class that shares a name with torch's pipeline stage."""

    model = TinyModel()
    model.stage = PipelineStage()  # type: ignore[assignment]
    assert detect_distributed_state(model, torch.randn(2, 4)) == ()


def test_dense_tensor_subclass_is_not_a_false_positive() -> None:
    """An ordinary user tensor subclass must not read as distributed state."""

    class MyTensor(torch.Tensor):
        """Plain user tensor subclass."""

    model = TinyModel()
    model.fc.weight = nn.Parameter(torch.zeros(4, 4).as_subclass(MyTensor), requires_grad=False)
    assert detect_distributed_state(model, torch.randn(2, 4)) == ()


def test_refusing_kinds_is_the_single_source_of_truth() -> None:
    """Every active distributed execution mode that omits work refuses."""

    assert (
        frozenset({"dtensor", "tensor_parallel", "pipeline_parallel", "scan_incomplete"})
        == REFUSING_KINDS
    )


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


def test_report_row_marks_refusal_and_matches_detection() -> None:
    """A refusing row says capture refuses, and says so with severity ``error``."""

    model = TinyModel()
    model.fc.weight = nn.Parameter(_fake_dtensor((4, 4)), requires_grad=False)

    row = tl.compat.report(model, torch.randn(2, 4)).row("dtensor")
    assert row.detected is True
    assert row.status == "scope"
    assert row.severity == "error"
    assert "DistributedCaptureUnsupportedError" in row.details
    assert "fc.weight" in row.details
    assert "Detected structurally" in row.details
    assert row.suggestion


def test_non_refusing_row_is_warning_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported-but-not-refused condition must not claim capture refuses."""

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.pipelining",
        types.ModuleType("torch.distributed.pipelining"),
    )
    model = TinyModel()
    model.stages = [_FakePipelineStage()]  # type: ignore[assignment]
    refusing_row = tl.compat.report(model, torch.randn(2, 4)).row("pipeline_parallel")
    assert refusing_row.severity == "error"

    clean_row = tl.compat.report(TinyModel(), torch.randn(2, 4)).row("device_mesh")
    assert clean_row.severity == "ok"
    assert "DistributedCaptureUnsupportedError" not in clean_row.details

    dtensor_clear = tl.compat.report(TinyModel(), torch.randn(2, 4)).row("dtensor")
    assert "bounded entry scan" in dtensor_clear.details
    assert "slots-only" in dtensor_clear.details

    tp_clear = tl.compat.report(TinyModel(), torch.randn(2, 4)).row("tensor_parallel")
    assert "user-wrapped or opaque hook" in tp_clear.details


def test_report_renders_with_distributed_rows() -> None:
    """Both renderers include the new rows without raising."""

    model = TinyModel()
    model.fc.weight = nn.Parameter(_fake_dtensor((4, 4)), requires_grad=False)
    report = tl.compat.report(model, torch.randn(2, 4))
    assert "DTensor / sharded tensors" in report.show()
    assert "Tensor parallel (TP)" in report.to_markdown()


def test_typed_error_is_reachable_from_public_errors_namespace() -> None:
    """The refusal type is part of the public ``torchlens.errors`` surface."""

    import torchlens.errors as errors

    assert errors.DistributedCaptureUnsupportedError is DistributedCaptureUnsupportedError
    assert "DistributedCaptureUnsupportedError" in errors.__all__
    assert "DistributedCaptureUnsupportedError" in dir(errors)
    assert issubclass(DistributedCaptureUnsupportedError, errors.CompatibilityError)


def test_capability_snapshot_exposes_distributed_flags() -> None:
    """Every graceful-degradation point is visible as a named ``HAS_*`` flag."""

    from torchlens.utils._torch_compat import get_torch_capability_snapshot

    snapshot = get_torch_capability_snapshot()
    for flag in ("HAS_DTENSOR", "HAS_DEVICE_MESH", "HAS_PIPELINING"):
        assert flag in snapshot
        assert isinstance(snapshot[flag], bool)


def test_doctor_reports_distributed_capability_flags() -> None:
    """``tl.utils.doctor()`` surfaces the new flags like every other capability.

    The capability row is a grouped absences-first summary (r-b4 R26-4): a
    flag appears in the row text by name only when it is absent; a present
    flag is served through the full snapshot accessor the row points at.
    """

    snapshot = tl.utils.capability_snapshot()
    text = str(tl.utils.doctor())
    for flag in ("HAS_DTENSOR", "HAS_DEVICE_MESH", "HAS_PIPELINING"):
        assert flag in snapshot
        if not snapshot[flag]:
            assert flag in text


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


@pytest.mark.heavy
def test_real_tensor_parallel_model_is_detected_exactly(single_rank_cpu_mesh: object) -> None:
    """Real ``parallelize_module`` state hits the exact ``isinstance`` branch."""

    parallel = pytest.importorskip("torch.distributed.tensor.parallel")
    model = TinyModel()
    parallel.parallelize_module(model, single_rank_cpu_mesh, {"fc": parallel.ColwiseParallel()})
    assert type(model.fc.weight).__name__ == "DTensor", "precondition: TP produced DTensors"

    findings = detect_distributed_state(model, torch.randn(2, 4))
    kinds = _finding_kinds(findings)
    assert {"dtensor", "device_mesh", "tensor_parallel"} <= kinds

    dtensor_finding = _find(findings, "dtensor")
    assert dtensor_finding.exact is True, "real DTensor must match the probed class exactly"
    assert "fc.weight" in dtensor_finding.sites
    assert "fc.bias" in dtensor_finding.sites

    mesh_finding = _find(findings, "device_mesh")
    assert mesh_finding.refuses_capture is False
    assert "DeviceMesh(" in mesh_finding.detail

    tp_finding = _find(findings, "tensor_parallel")
    assert tp_finding.refuses_capture is True
    assert "non-replicated placement" in tp_finding.detail


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
def test_real_dtensor_in_plain_module_attribute_is_detected(
    single_rank_cpu_mesh: object,
) -> None:
    """A real unregistered DTensor attribute reaches the typed boundary."""

    tensor_api = pytest.importorskip("torch.distributed.tensor")
    model = TinyModel()
    model.unregistered_shard = tensor_api.distribute_tensor(  # type: ignore[assignment]
        torch.randn(2, 4),
        single_rank_cpu_mesh,
        placements=[tensor_api.Replicate()],
    )

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "dtensor")
    assert finding.exact is True
    assert finding.sites == ("<root>.unregistered_shard",)
    with pytest.raises(DistributedCaptureUnsupportedError):
        tl.trace(model, torch.randn(2, 4))


@pytest.mark.heavy
def test_real_dense_tensor_parallel_hook_is_detected_and_refuses(
    single_rank_cpu_mesh: object,
) -> None:
    """``PrepareModuleInput`` TP hooks omit collectives even when parameters stay dense."""

    parallel = pytest.importorskip("torch.distributed.tensor.parallel")
    tensor_api = pytest.importorskip("torch.distributed.tensor")
    model = nn.ReLU()
    parallel.parallelize_module(
        model,
        single_rank_cpu_mesh,
        parallel.PrepareModuleInput(
            input_layouts=tensor_api.Shard(0),
            desired_input_layouts=tensor_api.Replicate(),
            use_local_output=True,
        ),
    )

    finding = _find(detect_distributed_state(model, torch.randn(2, 4)), "tensor_parallel")
    assert finding.refuses_capture is True
    assert "forward hook" in finding.detail
    with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
        tl.trace(model, torch.randn(2, 4))
    assert [item.kind for item in excinfo.value.fields["findings"]] == ["tensor_parallel"]


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

    row = tl.compat.report(model, x).row("dtensor")
    assert row.detected is True
    assert row.severity == "error"
    assert "Detected structurally" not in row.details, "real DTensor is an exact match"


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
    if has_vetted_snapshot():
        trace = tl.trace(model, x)
    else:
        with pytest.warns(UserWarning, match="uncaptured_collective_op"):
            trace = tl.trace(model, x)
    assert trace.num_params == sum(p.numel() for p in model.parameters())
    assert any("linear" in label for label in trace.layer_labels)
