"""Regression coverage for payload selection outside CaptureOptions."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._selection.selector_base import CompositeSelector
from torchlens.errors import FacadeTeachingError
from torchlens.options import CaptureOptions, merge_capture_options


@pytest.mark.parametrize("removed", ["layers_to_save", "save_preview"])
def test_removed_capture_fields_reject_constructor_keywords(removed: str) -> None:
    assert removed not in inspect.signature(CaptureOptions).parameters
    assert removed not in CaptureOptions().as_dict()
    with pytest.raises(FacadeTeachingError) as excinfo:  # the removed field names save=
        CaptureOptions(**{removed: False})
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert "save=" in str(excinfo.value)


def test_private_selection_survives_option_merges() -> None:
    selection = ["relu"]
    original = merge_capture_options(capture=CaptureOptions(seed=7), _layers_to_save=selection)
    merged = merge_capture_options(capture=original, save_args=True)
    assert merged._layers_to_save == selection
    assert merged._layers_to_save_explicit
    assert merged.seed == 7
    assert merged.save_args is True
    assert original.save_args is False
    assert "_layers_to_save" not in merged.as_dict()


def test_public_save_selects_final_labels_and_none() -> None:
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    x = torch.ones(1, 3)
    full = tl.trace(model, x)
    label = next(op.label for op in full.ops if op.func_name == "relu")
    selected = tl.trace(model, x, save=[label])
    assert selected[label].has_saved_activation
    assert not next(op for op in selected.ops if op.func_name == "linear").has_saved_activation
    assert not any(op.has_saved_activation for op in tl.trace(model, x, save=None).ops)


def test_save_predicate_composes_the_module_selection() -> None:
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        save=tl.where(lambda context: "1" in context.module_stack),
    )
    assert any(op.has_saved_activation for op in trace.ops if op.func_name == "relu")
    assert not any(op.has_saved_activation for op in trace.ops if op.func_name == "linear")
    predicate = tl.where(lambda context: "1" in context.module_stack)
    # A bare label string composes with a selector (the selector algebra lifts it);
    # a list or a callable is not a selector operand and keeps Python's TypeError.
    assert isinstance("all" & predicate, CompositeSelector)
    for selection in (["relu"], lambda context: True):
        with pytest.raises(TypeError):
            selection & predicate


def test_explicit_all_preserves_structure_only_refusal() -> None:
    with pytest.raises(ValueError, match="structure_only=True cannot honor"):
        tl.trace(nn.ReLU(), torch.ones(1), save="all", capture=CaptureOptions(structure_only=True))


def test_debug_seed_copy_preserves_private_selection() -> None:
    from torchlens.debug._rerun import isolated_capture

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    capture = merge_capture_options(capture=CaptureOptions(), _layers_to_save=["relu"])
    trace = isolated_capture(model, torch.ones(1, 3), seed=7, capture=capture)
    assert any(op.has_saved_activation for op in trace if op.func_name == "relu")
    assert not any(op.has_saved_activation for op in trace if op.func_name == "linear")


@pytest.mark.parametrize(
    ("kwargs", "expected_relu", "expected_linear"),
    [({}, True, True), ({"save": ["relu"]}, True, False), ({"save": None}, False, False)],
)
def test_autoroute_reentry_preserves_resolved_save(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, expected_relu: bool, expected_linear: bool
) -> None:
    def detector(model, payload, **forwarded):
        if isinstance(payload, str):
            return tl.trace(model, torch.ones(1, 3), **forwarded)
        return None

    monkeypatch.setattr("torchlens._autoroute.input.iter_by_priority", lambda: [detector])
    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), "route", **kwargs)
    assert (
        any(op.has_saved_activation for op in trace.ops if op.func_name == "relu") is expected_relu
    )
    assert (
        any(op.has_saved_activation for op in trace.ops if op.func_name == "linear")
        is expected_linear
    )


def test_lit_probe_retains_its_save_all_default() -> None:
    from torchlens.bridge.lit._runtime import traced_forward

    x = torch.tensor([-1.0, 2.0])
    trace = traced_forward(nn.ReLU(), {"input": x}, save=None)
    assert torch.equal(trace.reconstruct_output(), torch.relu(x))
