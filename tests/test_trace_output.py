"""The live forward return behind ``Trace.output`` is never serialized."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _ContainerOutput(nn.Module):
    """Return a fresh tuple or dict whose identity a root hook can observe."""

    def __init__(self, kind: str = "dict") -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.kind = kind

    def forward(self, x: torch.Tensor) -> Any:
        """Return tensors alongside a literal leaf."""

        hidden = torch.relu(self.fc(x))
        pair = (hidden, "tag")
        return {"logits": hidden, "extra": pair} if self.kind == "dict" else pair


@pytest.mark.parametrize("kind", ["dict", "tuple"])
@pytest.mark.parametrize("save", ["all", "none"])
def test_output_is_the_exact_forward_return(kind: str, save: str) -> None:
    """Output identity is independent of per-op activation retention."""

    seen: list[Any] = []
    model = _ContainerOutput(kind)
    handle = model.register_forward_hook(lambda _module, _args, output: seen.append(output))
    try:
        trace = tl.trace(model, torch.ones(1, 3), save=save)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert trace.output is seen[0]
    assert trace.transformed_output is None


def test_output_refuses_after_stop_after() -> None:
    """A halted trace has a frontier value but no model return."""

    trace = tl.trace(_ContainerOutput(), torch.ones(1, 3), stop_after=tl.func("linear"))
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = trace.output
    assert exc_info.value.fields["code"] == "output_capture_stopped"


def test_output_is_independent_of_the_output_transform() -> None:
    """The exact return survives a transform that keeps only one tensor."""

    trace = tl.trace(
        _ContainerOutput(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(output_transform=lambda output: output["logits"]),
    )
    assert isinstance(trace.output, dict)
    assert trace.transformed_output is trace.output["logits"]


@pytest.mark.parametrize("level", ["audit", "portable"])
def test_output_is_not_stored_with_saved_traces(tmp_path: Path, level: str) -> None:
    """Both bundle save levels omit the live return, preserving the source slot."""

    trace = tl.trace(_ContainerOutput(), torch.ones(1, 3))
    output = trace.output
    path = tmp_path / "trace.tlspec"
    trace.save(path, level=level)
    loaded = tl.load(path)
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = loaded.output
    assert exc_info.value.fields["code"] == "output_not_stored"
    assert trace.output is output


def test_output_is_not_pickled() -> None:
    """Unpickling loses the slot even when the output carries an opaque callable."""

    trace = tl.trace(_ContainerOutput(), torch.ones(1, 3))
    trace.output["unpicklable"] = lambda: None
    restored = pickle.loads(pickle.dumps(trace))
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = restored.output
    assert exc_info.value.fields["code"] == "output_not_stored"


def test_fork_does_not_claim_a_forward_return() -> None:
    """A fork carries derived graph state without an independent forward return."""

    trace = tl.trace(_ContainerOutput(), torch.ones(1, 3))
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = trace.fork().output
    assert exc_info.value.fields["code"] == "output_not_stored"


@pytest.mark.parametrize("fast", [False, True])
def test_live_run_keeps_its_own_forward_return(fast: bool) -> None:
    """Ordinary refresh and native fast runs expose their actual return object."""

    model = _ContainerOutput()
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    source_output = trace.output
    seen: list[Any] = []
    handle = model.register_forward_hook(lambda _module, _args, output: seen.append(output))
    try:
        result = trace.run(inputs=torch.full((1, 3), 2.0), fast=fast)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert result.trace.output is seen[0]
    if not fast:
        assert trace.output is source_output


@pytest.mark.parametrize("fast", [False, True])
def test_truncated_live_run_has_no_forward_return(fast: bool) -> None:
    """A live run cut at a saved site cannot expose a prior run's return."""

    model = _ContainerOutput()
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    site = next(op.label for op in trace.ops if op.type == "linear")
    result = trace.run(inputs=torch.full((1, 3), 2.0), fast=fast, until=site)
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = result.trace.output
    assert exc_info.value.fields["code"] == "output_capture_stopped"


def test_recording_conversion_has_no_live_trace_return() -> None:
    """Cooking a Recording does not invent a Trace forward."""

    recording = tl.record(_ContainerOutput(), torch.ones(1, 3), save=lambda _ctx: True)
    trace = recording.to_trace()
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = trace.output
    assert exc_info.value.fields["code"] == "output_not_stored"


def test_chunked_trace_does_not_expose_one_chunks_return() -> None:
    """Merged chunks have no single exact forward return for the whole batch."""

    trace = tl.trace(
        _ContainerOutput(),
        torch.ones(4, 3),
        capture=tl.options.CaptureOptions(chunk_size=2),
    )
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = trace.output
    assert exc_info.value.fields["code"] == "output_not_stored"
