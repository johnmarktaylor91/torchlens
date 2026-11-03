"""Phase 4 capture target tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._data_classes import _op_save as op_save_module
from torchlens.backends.torch._ops_activations import _save_activation_fields
from torchlens.options import CaptureOptions


class _ConfiguredClassifier(torch.nn.Module):
    """Tiny classifier carrying HF-style output metadata on ``config``."""

    def __init__(self) -> None:
        """Initialize the classifier and its portable config metadata."""

        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(id2label={"0": "negative", 1: "positive"}, num_labels=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear classifier.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.linear(x)


def _minimal_activation_fields(label: str) -> dict[str, Any]:
    """Return the field subset required by ``_save_activation_fields``."""

    return {
        "_label_raw": f"{label}_raw",
        "_layer_label_raw": label,
        "annotations": {},
        "backward_ready": False,
        "detach_saved_activations": False,
        "func_name": "manual_test",
        "output_device": "same",
    }


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups_capture.py.
def test_capture_memory_fields_and_forward_source_line() -> None:
    """Populate Phase 4 Op memory fields and forward line number."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    log = tl.trace(model, torch.ones(1, 2))
    assert isinstance(log.source.forward_line, int)
    assert all(hasattr(layer, "bytes_delta_at_call") for layer in log._layer_list)
    assert all(hasattr(layer, "bytes_peak_at_call") for layer in log._layer_list)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _trace_selector_helpers.py, campaign mutant kills.
@pytest.mark.parametrize(
    ("kwargs", "capture_kwargs", "saved_type"),
    [
        ({"save": tl.func("relu")}, {}, "relu"),
    ],
)
def test_selective_capture_cache_preserves_unsaved_payload_contract(
    tmp_path: Path,
    kwargs: dict[str, Any],
    capture_kwargs: dict[str, Any],
    saved_type: str,
) -> None:
    """Selective capture cache writes should not read unsaved public payloads."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    x = torch.ones(1, 2)

    first = tl.trace(
        model,
        x,
        **kwargs,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path, **capture_kwargs),
    )
    second = tl.trace(
        model,
        x,
        **kwargs,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path, **capture_kwargs),
    )

    assert first.provenance.cache_hit is False
    assert second.provenance.cache_hit is True
    assert first.provenance.cache_key == second.provenance.cache_key
    saved = first.find_sites(tl.func(saved_type)).first()
    assert isinstance(saved.out, torch.Tensor)
    with pytest.raises(ValueError, match="was not saved"):
        _ = first["input_1"].out


# Kept by shrink-r4 (per-id subset rule): sole coverage of _trace_selector_helpers.py.
def test_absorbed_layers_to_save_cache_key_is_stable(tmp_path: Path) -> None:
    """Identical absorbed ``layers_to_save`` requests should hash equally."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    x = torch.ones(1, 2)

    first = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path),
        save=["relu"],
    )
    second = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path),
        save=["relu"],
    )

    assert first.provenance.cache_key == second.provenance.cache_key
    assert second.provenance.cache_hit is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of _autoroute/_builtin_output.py.
def test_config_output_metadata_pickles_into_capture_cache(tmp_path: Path) -> None:
    """HF-style config metadata is captured before cache serialization."""

    model = _ConfiguredClassifier()
    x = torch.ones(1, 2)
    capture = CaptureOptions(cache=True, cache_dir=tmp_path)

    first = tl.trace(model, x, capture=capture)
    second = tl.trace(model, x, capture=capture)

    assert first.provenance.cache_hit is False
    assert first.output_id2label == {0: "negative", 1: "positive"}
    assert first.output_num_classes == 2
    assert second.provenance.cache_hit is True
    assert second.output_id2label == {0: "negative", 1: "positive"}
    assert second.output_num_classes == 2


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/op.py, backends/torch/wrappers.py.
def test_trace_reference_save_mode_raises_if_saved_out_mutates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference-mode saved outputs fail loudly after mutation."""

    monkeypatch.setattr(op_save_module, "_WARNED_REFERENCE_SAVE_MODE", False)
    with pytest.warns(UserWarning, match="save_mode='reference'"):
        log = tl.trace(
            torch.nn.ReLU(),
            torch.ones(1, 2),
            capture=tl.options.CaptureOptions(save_mode="reference"),
        )
    op = next(layer for layer in log._layer_list if layer.func_name == "relu")
    saved_out = op._slot("out")

    saved_out.add_(1)

    with pytest.raises(tl.errors.MutatedReferenceError, match="mutated after capture"):
        _ = op.out


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_dedup_source_id_is_a_dense_trace_local_ordinal() -> None:
    """``dedup_source_id`` must never be a raw ``id()`` memory address (B3R4-R21-1).

    The raw ``id()`` bookkeeping key leaked into the persisted annotation
    dict, so artifacts of the SAME captured program carried per-process
    memory addresses (byte-compare defeated, meaningless public value).
    The annotation now carries the trace-local dense dedup-source ordinal.
    """

    trace = tl.trace(torch.nn.Identity(), torch.randn(1, 2))
    source = torch.randn(2, 2)
    other = torch.randn(2, 2)
    fields = [_minimal_activation_fields(f"manual_{i}") for i in range(4)]

    _save_activation_fields(trace, fields[0], source, (), {}, None)
    _save_activation_fields(trace, fields[1], source, (), {}, None)
    _save_activation_fields(trace, fields[2], other, (), {}, None)
    _save_activation_fields(trace, fields[3], other, (), {}, None)

    first_hit = fields[1]["annotations"]["dedup_source_id"]
    second_hit = fields[3]["annotations"]["dedup_source_id"]
    assert first_hit == 1
    assert second_hit == 2
    assert first_hit != id(source)
    assert second_hit != id(other)
