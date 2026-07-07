"""Phase 3b capture-unification alias, compatibility, and orphan tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_selector_helpers import _make_layers_to_save_predicate
from torchlens.fastlog.exceptions import PredicateError
from torchlens.fastlog import RecordContext
from torchlens.intervention.errors import SelectorCompositionError
from torchlens.validation.invariants import _check_backend_neutral_graph_topology


class ViewThenMutate(nn.Module):
    """Model with a view alias that mutates an unsaved parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a view-then-in-place mutation before a saved child."""

        parent = x + 1
        view = parent.view_as(parent)
        view.add_(2)
        return parent * 3


class OutKwargMutate(nn.Module):
    """Model using a non-first mutated input through an ``out=`` kwarg."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into an unsaved parent via ``out=`` before a saved child."""

        parent = x + 1
        torch.add(x, 2, out=parent)
        return parent * 3


class MultiOutputAlias(nn.Module):
    """Model where a multi-output view aliases an unsaved parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a split view before saving a later child."""

        parent = x + 1
        first, _second = parent.split(2, dim=1)
        first.add_(2)
        return parent * 3


class TinyLinear(nn.Module):
    """Small differentiable model for compatibility matrix checks."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple linear forward pass."""

        return torch.relu(self.linear(x))


class SavedOrphan(nn.Module):
    """Model producing a saved unused factory tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create an orphan tensor and return an unrelated output."""

        _unused = torch.randn(x.shape)
        return x + 1


def _save_only_mul(ctx: RecordContext) -> bool:
    """Select only the downstream multiplication op."""

    return ctx.func_name in {"__mul__", "mul"}


@pytest.mark.parametrize(
    "model_factory",
    (ViewThenMutate, OutKwargMutate, MultiOutputAlias),
)
def test_alias_contract_snapshots_unsaved_mutated_parent(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Selective save forces parent snapshots for aliased unsaved parents."""

    model = model_factory()
    x = torch.ones(2, 4)
    full = tl.trace(model, x.clone(), layers_to_save="all", save_arg_values=True)
    selective = tl.trace(model, x.clone(), save=_save_only_mul, save_arg_values=True)
    saved_children = [
        op
        for op in selective.layer_list
        if op.has_saved_activation and op.func_name in {"__mul__", "mul"}
    ]
    assert saved_children
    unsaved_snapshot_parents = [
        op
        for op in selective.layer_list
        if not op.has_saved_activation and bool(op.out_versions_by_child)
    ]
    assert unsaved_snapshot_parents
    assert any(op.out_versions_by_child for op in full.layer_list)

    expected = [model_factory()(x.clone()).detach().clone()]
    try:
        assert selective.validate_forward_pass(expected, validate_metadata=False)
    except ValueError as exc:
        assert "Cannot validate saved layer" in str(exc) or "was not saved" in str(exc)


def test_layers_to_save_supports_disk_streaming(tmp_path: Path) -> None:
    """Selective layers_to_save streams through predicate-backed capture."""

    bundle_path = tmp_path / "bundle"
    log = tl.trace(
        TinyLinear(),
        torch.randn(2, 4),
        layers_to_save=["linear"],
        save_outs_to=bundle_path,
    )
    saved = [op for op in log.layer_list if op.has_saved_activation and op.layer_type == "linear"]
    assert saved
    assert bundle_path.exists()


def test_layers_to_save_absorbed_path_honors_substrings() -> None:
    """Selective ``layers_to_save`` keeps genuine substrings on the absorbed path."""

    log = tl.trace(
        TinyLinear(),
        torch.randn(2, 4),
        layers_to_save=["lin"],
    )
    saved = [op for op in log.layer_list if op.has_saved_activation and op.layer_type == "linear"]
    assert saved


def test_layers_to_save_absorbed_path_honors_pass_qualified_substrings() -> None:
    """Pass-qualified substrings save only the matching model-call pass."""

    with tl.fastlog.Recorder(TinyLinear(), save=lambda ctx: ctx.kind == "op") as recorder:
        recorder.log(torch.randn(2, 4))
        recorder.log(torch.randn(2, 4))
    contexts = [
        record.ctx for record in recorder.recording.records if record.ctx.layer_type == "linear"
    ]
    assert contexts
    predicate = _make_layers_to_save_predicate(["lin:2"])
    assert [ctx.pass_index for ctx in contexts if predicate(ctx)] == [2]


def test_layers_to_save_supports_backward_ready_and_gradients_two_pass() -> None:
    """Selective two-pass supports backward-ready and gradient selection."""

    log = tl.trace(
        TinyLinear(),
        torch.randn(2, 4),
        layers_to_save=["linear"],
        save_grads=["linear"],
        backward_ready=True,
    )
    saved = [op for op in log.layer_list if op.has_saved_activation and op.layer_type == "linear"]
    assert saved
    assert saved[0].out.requires_grad


def test_layers_to_save_supports_intervention_ready_and_hooks() -> None:
    """Selective layers_to_save composes with intervention-ready and live hooks."""

    x = torch.randn(2, 4)
    intervention_ready = tl.trace(
        TinyLinear(),
        x,
        layers_to_save=["linear"],
        intervention_ready=True,
    )
    hooked = tl.trace(
        TinyLinear(),
        x,
        layers_to_save=["linear"],
        hooks=tl.tap(tl.func("relu")),
    )
    assert any(
        op.has_saved_activation and op.layer_type == "linear"
        for op in intervention_ready.layer_list
    )
    assert any(op.has_saved_activation and op.layer_type == "linear" for op in hooked.layer_list)


def test_layers_to_save_supports_intervene_halt_and_save_predicate() -> None:
    """Selective layers_to_save composes with live predicates."""

    x = torch.randn(2, 4)
    intervened = tl.trace(
        TinyLinear(),
        x,
        layers_to_save=["linear"],
        intervene=tl.when(tl.func("relu"), tl.add(0.0)),
    )
    union_saved = tl.trace(
        TinyLinear(),
        x,
        layers_to_save=["linear"],
        save=lambda ctx: ctx.kind == "op" and ctx.layer_type == "relu",
    )
    halted = tl.trace(
        TinyLinear(),
        x,
        layers_to_save=["linear"],
        halt=lambda ctx: ctx.kind == "op" and ctx.layer_type == "relu",
    )

    assert any(
        op.has_saved_activation and op.layer_type == "linear" for op in intervened.layer_list
    )
    saved_types = {op.layer_type for op in union_saved.layer_list if op.has_saved_activation}
    assert {"linear", "relu"}.issubset(saved_types)
    assert getattr(halted, "halted", False) is True


def test_orphan_records_expose_saved_payload_when_pruned_or_retained() -> None:
    """Saved orphan payloads are exposed regardless of graph pruning mode."""

    x = torch.ones(2, 2)
    pruned = tl.trace(SavedOrphan(), x, save=tl.func("randn"), random_seed=1)
    assert pruned.orphan_records
    assert pruned.orphan_records[0]["raw_label"].startswith("randn")
    assert isinstance(pruned.orphan_records[0]["payload_ref"], torch.Tensor)
    assert not any(label.startswith("randn") for label in pruned.op_labels)

    retained = tl.trace(
        SavedOrphan(),
        x,
        save=tl.func("randn"),
        random_seed=1,
        keep_orphans=True,
    )
    assert retained.orphan_records
    assert retained.orphans


def test_followed_by_trace_supported_record_rejected_and_bad_shapes_error() -> None:
    """followed_by is either supported explicitly or rejected loudly."""

    x = torch.ones(2, 4)
    selector = tl.func("linear") & tl.followed_by(tl.func("relu"))
    traced = tl.trace(
        TinyLinear(),
        x,
        save=selector,
        lookback=4,
        lookback_payload_policy="detached_raw",
    )
    assert any(op.layer_type == "linear" and op.has_saved_activation for op in traced.layer_list)

    with pytest.raises(PredicateError, match="record\\(save=.*followed_by"):
        tl.record(TinyLinear(), x, save=selector)
    with pytest.raises(SelectorCompositionError):
        _ = ~tl.followed_by(tl.func("relu"))
    with pytest.raises(SelectorCompositionError):
        _ = tl.func("linear") | tl.followed_by(tl.func("relu"))


def test_followed_by_intervene_and_halt_fail_at_capture_start() -> None:
    """followed_by selectors on intervene/halt raise preflight capability errors."""

    x = torch.ones(2, 4)
    selector = tl.func("linear") & tl.followed_by(tl.func("relu"))

    with pytest.raises(PredicateError, match="trace\\(intervene=.*does not support"):
        tl.trace(
            TinyLinear(),
            x,
            intervene=tl.when(selector, tl.add(0.0)),
        )
    with pytest.raises(PredicateError, match="trace\\(halt=.*does not support"):
        tl.trace(TinyLinear(), x, halt=selector)
    with pytest.raises(PredicateError, match="unsupported followed_by predicate shape"):
        tl.trace(TinyLinear(), x, halt=tl.followed_by(tl.func("relu")))


def test_tf_trace_exercises_backend_neutral_topology_invariant() -> None:
    """A real TensorFlow trace should satisfy backend-neutral topology checks."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "tensorflow":
        pytest.skip(f"active keras backend is {keras.backend.backend()!r}")

    trace = tl.trace(tf.identity, tf.constant([1.0, 2.0]), backend="tf")

    assert trace.backend == "tf"
    _check_backend_neutral_graph_topology(trace)
