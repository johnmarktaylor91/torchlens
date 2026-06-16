"""Tests for forward ``chunk_size`` capture and rerun chunking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import (
    BackendCapabilities,
    BackendSpec,
    BackendUnsupportedError,
    register_backend_spec,
    unregister_backend_spec,
)
from torchlens.intervention.errors import BatchChunkInputAmbiguityError, ChunkedForwardConfigError


class DeterministicToy(nn.Module):
    """Small deterministic model with stable topology across chunks."""

    def __init__(self) -> None:
        """Initialize fixed linear layers."""

        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple pointwise forward pass."""

        return self.out(torch.relu(self.proj(x)))


class TwoInputToy(nn.Module):
    """Model that consumes two batched positional tensors."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Combine two batched tensors."""

        return torch.relu(left + right)


class MaskToy(nn.Module):
    """Model with one batched tensor input and one shared matrix input."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Use the full shared mask for every chunk."""

        return x @ mask


class KwargToy(nn.Module):
    """Model accepting a keyword bias."""

    def forward(self, x: torch.Tensor, *, bias: torch.Tensor) -> torch.Tensor:
        """Add a keyword bias."""

        return x + bias


def _toy_inputs() -> torch.Tensor:
    """Return deterministic batched toy inputs."""

    return torch.arange(30, dtype=torch.float32).reshape(10, 3) / 10.0


def _manual_chunk_trace(model: nn.Module, x: torch.Tensor, chunk_size: int) -> tl.Trace:
    """Build the manual trace-plus-append equivalent for one tensor input."""

    chunks = list(torch.split(x, chunk_size, dim=0))
    trace = tl.trace(model, chunks[0], layers_to_save="all")
    for chunk in chunks[1:]:
        trace.rerun(model, chunk, append=True, transform=False)
    return trace


def _tensor_or_none(value: Any) -> torch.Tensor | None:
    """Return ``value`` if it is a tensor."""

    return value if isinstance(value, torch.Tensor) else None


def _normalized_append_history(trace: tl.Trace) -> list[dict[str, Any]]:
    """Return append history with volatile timing fields removed."""

    volatile = {"timestamp", "started_at", "duration_s"}
    return [
        {key: value for key, value in row.items() if key not in volatile}
        for row in trace.append_history
    ]


def _assert_equivalent_chunked_trace(actual: tl.Trace, expected: tl.Trace) -> None:
    """Compare chunked traces while ignoring intentionally last-chunk metadata."""

    assert actual.layer_labels == expected.layer_labels
    assert [layer._layer_label_raw for layer in actual.layer_list] == [
        layer._layer_label_raw for layer in expected.layer_list
    ]
    assert actual.graph_shape_hash == expected.graph_shape_hash
    assert actual.output_layers == expected.output_layers
    for label in actual.layer_labels:
        actual_layer = actual[label]
        expected_layer = expected[label]
        for field_name in ("out", "transformed_out"):
            actual_tensor = _tensor_or_none(getattr(actual_layer, field_name, None))
            expected_tensor = _tensor_or_none(getattr(expected_layer, field_name, None))
            if actual_tensor is None or expected_tensor is None:
                assert actual_tensor is expected_tensor
            else:
                torch.testing.assert_close(actual_tensor, expected_tensor)


def test_trace_chunk_size_matches_manual_append_loop() -> None:
    """One-line chunked capture should match manual append accumulation."""

    torch.manual_seed(0)
    model = DeterministicToy().eval()
    x = _toy_inputs()

    actual = tl.trace(model, x, chunk_size=4, layers_to_save="all")
    expected = _manual_chunk_trace(model, x, 4)

    _assert_equivalent_chunked_trace(actual, expected)
    assert len(actual.append_history) == 3
    assert [row["chunk_size"] for row in actual.append_history] == [4, 4, 2]


def test_chunk_size_default_matches_plain_trace() -> None:
    """Leaving chunk_size unset preserves plain capture behavior."""

    model = DeterministicToy().eval()
    x = _toy_inputs()

    plain = tl.trace(model, x, layers_to_save="all")
    default = tl.trace(model, x, layers_to_save="all", chunk_size=None)

    _assert_equivalent_chunked_trace(default, plain)
    assert default.chunked_forward is False
    assert default.append_history == plain.append_history


def test_chunk_size_shape_and_remainder() -> None:
    """A 10-item batch with chunk_size 4 should produce 4, 4, and 2 chunks."""

    model = DeterministicToy().eval()
    trace = tl.trace(model, _toy_inputs(), chunk_size=4, layers_to_save="all")

    output_label = trace.output_layers[0]
    assert trace[output_label].out.shape[0] == 10
    assert len(trace.append_history) == 3
    assert [row["chunk_size"] for row in trace.append_history] == [4, 4, 2]
    assert trace.chunked_forward is True


def test_chunk_size_with_save_predicate_appends_saved_payloads_only() -> None:
    """Chunked predicate-save capture should append saved matching payloads."""

    model = DeterministicToy().eval()
    trace = tl.trace(model, _toy_inputs(), chunk_size=4, save=tl.func("relu"))

    relu = trace.find_sites(tl.func("relu")).first()
    assert relu.out.shape[0] == 10
    with pytest.raises(ValueError, match="was not saved"):
        _ = trace["input_1"].out


def test_explicit_path_keeps_shared_matrix_unsplit() -> None:
    """Explicit chunk paths split only selected leaves."""

    model = MaskToy().eval()
    x = torch.arange(100, dtype=torch.float32).reshape(10, 10) / 10.0
    mask = torch.eye(10)

    chunked = tl.trace(model, (x, mask), chunk_size=4, chunk_paths=["0"], layers_to_save="all")
    expected = tl.trace(model, (x[:4], mask), layers_to_save="all")
    expected.rerun(model, (x[4:8], mask), append=True, transform=False)
    expected.rerun(model, (x[8:], mask), append=True, transform=False)

    _assert_equivalent_chunked_trace(chunked, expected)
    torch.testing.assert_close(chunked[chunked.output_layers[0]].out, x)


def test_auto_mode_rejects_multiple_batched_tensor_leaves() -> None:
    """Auto mode should never silently choose among multiple batched tensors."""

    model = TwoInputToy().eval()
    x = torch.ones(6, 3)
    y = torch.ones(6, 3)

    with pytest.raises(BatchChunkInputAmbiguityError, match="0.*1"):
        tl.trace(model, (x, y), chunk_size=2)


def test_explicit_paths_split_multiple_inputs_and_reject_mismatch() -> None:
    """Explicit paths can name multiple batched leaves with matching sizes."""

    model = TwoInputToy().eval()
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    y = torch.ones(6, 3)

    trace = tl.trace(model, (x, y), chunk_size=4, chunk_paths=["0", "1"], layers_to_save="all")
    torch.testing.assert_close(trace[trace.output_layers[0]].out, torch.relu(x + y))

    with pytest.raises(ChunkedForwardConfigError, match="identical leading batch"):
        tl.trace(model, (x, y[:5]), chunk_size=4, chunk_paths=["0", "1"])


def test_chunk_size_edges() -> None:
    """Edge chunk sizes should either fall through or raise clearly."""

    model = DeterministicToy().eval()
    x = _toy_inputs()

    large = tl.trace(model, x, chunk_size=20, layers_to_save="all")
    assert large.chunked_forward is False
    assert large.append_history == []

    with pytest.raises(ChunkedForwardConfigError, match="positive integer"):
        tl.trace(model, x, chunk_size=0)


def test_chunk_size_guarded_combinations(tmp_path: Path) -> None:
    """Unsupported chunked-forward combinations should raise typed errors."""

    model = DeterministicToy().eval()
    x = _toy_inputs()

    with pytest.raises(ChunkedForwardConfigError, match="backward_ready"):
        tl.trace(model, x, chunk_size=4, backward_ready=True)
    with pytest.raises(ChunkedForwardConfigError, match="save_grads"):
        tl.trace(model, x, chunk_size=4, save_grads=True)
    with pytest.raises(ChunkedForwardConfigError, match="hooks"):
        tl.trace(model, x, chunk_size=4, hooks={"relu_1_1": lambda op: None})
    with pytest.raises(ChunkedForwardConfigError, match="intervene"):
        tl.trace(model, x, chunk_size=4, intervene=lambda ctx: None)
    with pytest.raises(ChunkedForwardConfigError, match="streaming"):
        tl.trace(model, x, chunk_size=4, storage=tl.to_disk(tmp_path / "chunked.tlspec"))
    with pytest.raises(ChunkedForwardConfigError, match="keyword"):
        tl.trace(KwargToy().eval(), x, input_kwargs={"bias": torch.ones_like(x)}, chunk_size=4)
    with pytest.raises(BackendUnsupportedError, match="chunk_size"):
        tl.trace(model, x, chunk_size=4, backend="jax")


def test_omitted_chunk_size_does_not_block_non_torch_backend() -> None:
    """A non-torch backend should accept omitted chunking options."""

    class FakeChunkModel:
        """Marker model for fake chunk capability tests."""

    def can_handle(
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None,
    ) -> bool:
        """Return whether the fake backend accepts this model."""

        del input_args, input_kwargs
        return isinstance(model, FakeChunkModel)

    def capture_trace(*args: Any, **kwargs: Any) -> tl.Trace:
        """Return a small trace relabeled as the fake backend."""

        assert "chunk_size" not in kwargs
        assert "chunk_paths" not in kwargs
        del args, kwargs
        trace = tl.trace(DeterministicToy().eval(), _toy_inputs()[:1], backend="torch")
        trace.backend = "chunk_fake"
        return trace

    def validate_entry(*args: Any, **kwargs: Any) -> bool:
        """Return a fake validation-entry result."""

        del args, kwargs
        return True

    def validate_trace(*args: Any, **kwargs: Any) -> bool:
        """Return a fake trace-validation result."""

        del args, kwargs
        return True

    register_backend_spec(
        BackendSpec(
            name="chunk_fake",
            can_handle=can_handle,
            capture_trace=capture_trace,
            validate_entry=validate_entry,
            validate_trace=validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=False,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=False,
                streaming=False,
            ),
        )
    )
    try:
        trace = tl.trace(FakeChunkModel(), object(), backend="chunk_fake")
    finally:
        unregister_backend_spec("chunk_fake")

    assert trace.backend == "chunk_fake"


def test_log_backward_rejects_chunked_forward() -> None:
    """Chunked traces should not accept deferred backward capture."""

    trace = tl.trace(DeterministicToy().eval(), _toy_inputs(), chunk_size=4)

    with pytest.raises(Exception, match="chunk_size"):
        trace.log_backward(torch.tensor(1.0, requires_grad=True))


def test_rerun_chunk_size_matches_manual_append_loop() -> None:
    """Trace.rerun(chunk_size=N) should match per-chunk append reruns."""

    model = DeterministicToy().eval()
    x = _toy_inputs()
    actual = tl.trace(model, x[:4], layers_to_save="all")
    expected = tl.trace(model, x[:4], layers_to_save="all")

    actual.rerun(model, x, chunk_size=4, transform=False)
    expected.rerun(model, x[:4], transform=False)
    expected.rerun(model, x[4:8], append=True, transform=False)
    expected.rerun(model, x[8:], append=True, transform=False)

    _assert_equivalent_chunked_trace(actual, expected)
    assert [row["chunk_size"] for row in actual.append_history] == [4, 4, 2]


def test_chunked_forward_marker_survives_tlspec(tmp_path: Path) -> None:
    """The sticky chunked-forward marker should round-trip through tlspec."""

    chunked = tl.trace(DeterministicToy().eval(), _toy_inputs(), chunk_size=4)
    plain = tl.trace(DeterministicToy().eval(), _toy_inputs())
    chunked_path = tmp_path / "chunked.tlspec"
    plain_path = tmp_path / "plain.tlspec"

    tl.save(chunked, chunked_path)
    tl.save(plain, plain_path)

    assert tl.load(chunked_path).chunked_forward is True
    assert tl.load(plain_path).chunked_forward is False
