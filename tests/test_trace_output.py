"""The live forward return behind ``Trace.output`` is never serialized."""

from __future__ import annotations

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
    site = next(layer.layer_label for layer in trace.layers if layer.layer_type == "linear")
    result = trace.run(inputs=torch.full((1, 3), 2.0), fast=fast, until=site)
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = result.trace.output
    assert exc_info.value.fields["code"] == "output_capture_stopped"


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
