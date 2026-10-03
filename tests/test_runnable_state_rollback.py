"""Run-preparation staging must publish NOTHING on failure (grind b3-l1, F-R07)."""

from __future__ import annotations

import pytest
import torch

from torchlens import _runnable_state
from torchlens._runnable_state import PreparedRunnableState, _apply_state_metadata_facts
from torchlens.errors import StateBindingError
from torchlens.runnable import StateSource


class _Binding:
    """State-binding stand-in carrying the recorded metadata facts."""

    def __init__(self, name: str, requires_grad: bool) -> None:
        """Record the declared name and capture-time trainable bit."""

        self.state_dict_name = name
        self.captured_grad_fn = False
        self.captured_requires_grad = requires_grad


class _Slot:
    """Tensor-slot stand-in with an owning state binding."""

    def __init__(self, slot_id: str, binding: _Binding) -> None:
        """Attach the binding to the slot identity."""

        self.slot_id = slot_id
        self.state_binding = binding


class _Descriptor:
    """Descriptor stand-in exposing only the state-slot surface."""

    def __init__(self, slots: list[_Slot]) -> None:
        """Hold the slot list the staging loop walks."""

        self.tensor_slots = slots


@pytest.mark.smoke
def test_state_metadata_staging_rolls_back_flipped_bits_on_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-loop metadata refusal restores every already-flipped staged bit.

    Staged slot values can be the trace-persisted clones themselves, so a
    slot-N failure that left slots 1..N-1 flipped would leak mutated
    ``requires_grad`` bits into embedded/staged user state across future runs,
    contradicting the publish-nothing-on-failure staging contract.
    """

    float_value = torch.zeros(2)
    int_value = torch.zeros(2, dtype=torch.int64)
    assert not float_value.requires_grad
    slots = [
        _Slot("s1", _Binding("layer.weight", requires_grad=True)),
        _Slot("s2", _Binding("layer.counter", requires_grad=True)),
    ]
    descriptor = _Descriptor(slots)
    binding_facts = {
        "layer.weight": {"grad_fn": False, "requires_grad": True},
        "layer.counter": {"grad_fn": False, "requires_grad": True},
    }
    monkeypatch.setattr(
        _runnable_state, "recorded_state_metadata_facts", lambda _desc: binding_facts
    )
    prepared = PreparedRunnableState(
        slot_values={"s1": float_value, "s2": int_value},
        state_source=StateSource.USER_STATE_DICT,
        initializer_policy_version=None,
        seed=None,
        random_filled_slot_ids=(),
    )

    with pytest.raises(StateBindingError):
        _apply_state_metadata_facts(descriptor, prepared)  # type: ignore[arg-type]

    # The int slot refused (integers cannot require grad); the float slot's
    # already-applied flip must have been rolled back, not published.
    assert not float_value.requires_grad
    assert not int_value.requires_grad


# =============================================================================
# L4 5.2/5.3/5.4: live run() declared-state snapshot-restore + carry_state
# =============================================================================


class _EmaParamModel(torch.nn.Module):
    """Mutates declared state in-forward: a no_grad param EMA (the 5.1 class).

    Deliberately params-only: a buffer counter would be a value-changing buffer
    SINK and the D18 projector refuses those on the default live path -- the
    restore bracket exists for exactly the declared-state writes the projector
    admits (no_grad param updates).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        with torch.no_grad():
            self.lin.weight.mul_(0.999)
        return out


class _GrowingBufferModel(torch.nn.Module):
    """Resizes its own buffer every forward, so a post-run restore must fail."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.register_buffer("buf", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.buf.resize_(self.buf.numel() + 1)
        return self.lin(x)


class _OverlappingStateModel(torch.nn.Module):
    """Registers two distinct buffer objects over overlapping storage bytes."""

    def __init__(self) -> None:
        super().__init__()
        base = torch.randn(6)
        self.register_buffer("full", base)
        self.register_buffer("head", base[:3])
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) + self.full.sum()


def _declared_state_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Deep-clone the model's full declared state for bit-identity assertions."""

    return {
        name: value.detach().clone()
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
    }


def _assert_declared_state_equal(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> None:
    """Assert the model's declared state is bit-identical to the snapshot."""

    current = dict(list(model.named_parameters()) + list(model.named_buffers()))
    assert current.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(current[name], value), name


def test_default_run_restores_declared_state_bit_identical() -> None:
    """L4 5.2: repeated default run() calls leave the live model bit-identical."""

    import torchlens as tl

    model = _EmaParamModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    before = _declared_state_clone(model)
    first = log.run(inputs=torch.randn(2, 4))
    _assert_declared_state_equal(model, before)
    assert first.report.state_carried is False
    second = log.run(inputs=torch.randn(2, 4))
    _assert_declared_state_equal(model, before)
    assert second.report.state_carried is False


def test_carry_state_persists_mutations_and_discloses() -> None:
    """L4 5.3: carry_state=True skips the restore and the report discloses it."""

    import torchlens as tl

    model = _EmaParamModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    weight_before = model.lin.weight.detach().clone()
    result = log.run(inputs=torch.randn(2, 4), carry_state=True)
    assert result.report.state_carried is True
    assert not torch.equal(model.lin.weight, weight_before)


def test_restore_runs_on_failed_forward() -> None:
    """L4 5.2 exception safety: restore runs in finally on the failure path too."""

    import torchlens as tl

    model = _EmaParamModel()
    log = tl.trace(model, torch.randn(2, 4))
    before = _declared_state_clone(model)
    with pytest.raises((RuntimeError, ValueError)):
        log.run(inputs=torch.randn(2, 9))
    _assert_declared_state_equal(model, before)


def test_carry_state_fast_refuses_typed() -> None:
    """Conflict-matrix row: carry_state=True x fast=True refuses typed."""

    import torchlens as tl

    model = _EmaParamModel()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(ValueError) as exc_info:
        log.run(inputs=torch.randn(2, 4), fast=True, carry_state=True)
    assert exc_info.value.fields["code"] == "run_fast_carry_state_unsupported"


def test_carry_state_loaded_sparse_refuses_typed(tmp_path) -> None:
    """Conflict-matrix row: carry_state=True on the loaded provider refuses typed."""

    import torchlens as tl
    from torchlens.options import CaptureOptions

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=CaptureOptions(intervention_ready=True, capture_container_structure=True),
    )
    path = tmp_path / "carry.tlspec"
    tl.save(log, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    with pytest.raises(ValueError) as exc_info:
        loaded.run(inputs=x, carry_state=True)
    assert exc_info.value.fields["code"] == "run_carry_state_requires_live_model"


def test_carry_state_legacy_surface_refuses_typed() -> None:
    """carry_state= is unified-only: the legacy run surface refuses typed."""

    import torchlens as tl
    from torchlens._errors import KeywordConflictError

    model = _EmaParamModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    with pytest.raises(KeywordConflictError) as exc_info:
        log.run(model, x, carry_state=True)
    assert exc_info.value.fields["code"] == "run_legacy_options_conflict"


def test_snapshot_preflight_refuses_overlapping_state_typed() -> None:
    """L4 5.4 fail-before-execute: unprovable/overlapping alias topology refuses."""

    import torchlens as tl

    model = _OverlappingStateModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    before = _declared_state_clone(model)
    with pytest.raises(StateBindingError) as exc_info:
        log.run(inputs=torch.randn(2, 4))
    assert exc_info.value.fields["code"] == "run_state_snapshot_unsupported"
    # No forward ran, so no state moved.
    _assert_declared_state_equal(model, before)
    # carry_state=True is the documented escape: the run proceeds without the bracket.
    result = log.run(inputs=torch.randn(2, 4), carry_state=True)
    assert result.report.state_carried is True


def test_restore_failure_marks_and_raises() -> None:
    """L4 5.4: a post-execution restore failure raises typed with structured fields."""

    import torchlens as tl

    model = _GrowingBufferModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    with pytest.raises(StateBindingError) as exc_info:
        log.run(inputs=torch.randn(2, 4))
    error = exc_info.value
    assert error.fields["code"] == "run_state_restore_failed"
    assert error.fields["state_dict_name"] == "buf"
    assert isinstance(error.fields["groups_restored"], int)
    assert error.__cause__ is not None
    latch = log._runnable.state_compromised
    assert latch is not None and latch["state_dict_name"] == "buf"


@pytest.mark.smoke
def test_restore_failure_source_refuses_next_live_run() -> None:
    """L4 5.4: the state-compromised latch refuses the live and fast doors typed."""

    import torchlens as tl

    model = _GrowingBufferModel()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    with pytest.raises(StateBindingError):
        log.run(inputs=torch.randn(2, 4))
    for kwargs in ({}, {"fast": True}):
        with pytest.raises(StateBindingError) as exc_info:
            log.run(inputs=torch.randn(2, 4), **kwargs)
        assert exc_info.value.fields["code"] == "run_state_restore_failed"
    # The legacy live surface reads the same live model and refuses identically.
    with pytest.raises(StateBindingError):
        log.run(model, x)


def test_restore_failure_loaded_sparse_still_legal(tmp_path) -> None:
    """L4 5.4 ALLOW cell: loaded-sparse runs never read the live model.

    The latch is session-scoped by construction (RunnableTraceState is always
    dropped by portable state handling), so a runnable artifact saved from a
    latched source loads and runs against staged clones.
    """

    import torchlens as tl
    from torchlens.options import CaptureOptions

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=CaptureOptions(intervention_ready=True, capture_container_structure=True),
    )
    log._runnable.state_compromised = {"state_dict_name": "synthetic", "groups_restored": 0}
    with pytest.raises(StateBindingError):
        log.run(inputs=x)
    path = tmp_path / "latched.tlspec"
    tl.save(log, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    assert loaded._runnable.state_compromised is None
    result = loaded.run(inputs=x)
    assert result.report is not None
