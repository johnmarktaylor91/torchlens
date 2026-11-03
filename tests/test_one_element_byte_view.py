"""One-element tensors with a non-unit stride read as their value at every digest site.

A one-element tensor keeps its source stride through ``contiguous()`` and
``reshape(-1)`` (``x[0:1, 1]`` of a 2x4 tensor has stride 4), and
``view(torch.uint8)`` refuses it. The replay commit digest raised on such a
site (behaviour matrix cell ``CIF_(Continuous_Integrate-and-Fire)-intervene-fork_do``);
every other byte-view digest shared the idiom, so they all read through
``flat_byte_view`` now.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_fingerprint import _hash_tensor_content
from torchlens._runnable.state import runnable_tensor_byte_digest
from torchlens._transport import digest_byte_view
from torchlens._utils.tensor_leaves import flat_byte_view, tensor_payload_bytes
from torchlens.intervention._replay_values import tensor_content_digest


def _one_element_strided() -> torch.Tensor:
    """Return a contiguous one-element tensor whose only stride is 4."""

    tensor = torch.arange(8.0).view(2, 4)[0:1, 1]
    assert tensor.is_contiguous() and tensor.stride() == (4,)
    return tensor


def test_flat_byte_view_reads_a_one_element_strided_tensor() -> None:
    tensor = _one_element_strided()
    twin = torch.tensor([1.0])
    assert flat_byte_view(tensor).tolist() == flat_byte_view(twin).tolist()
    assert flat_byte_view(tensor).data_ptr() == tensor.data_ptr()
    assert flat_byte_view(torch.tensor(1.0)).tolist() == flat_byte_view(twin).tolist()
    assert flat_byte_view(torch.empty(0)).numel() == 0


def test_digest_sites_read_a_one_element_strided_tensor_as_its_value() -> None:
    tensor = _one_element_strided()
    twin = torch.tensor([1.0])
    assert bytes(digest_byte_view(tensor)) == bytes(digest_byte_view(twin))
    assert tensor_content_digest(tensor) == tensor_content_digest(twin)
    assert tensor_payload_bytes(tensor) == tensor_payload_bytes(twin)
    assert runnable_tensor_byte_digest(tensor) == runnable_tensor_byte_digest(twin)
    assert _hash_tensor_content(tensor) == _hash_tensor_content(twin)
    assert tl.fingerprint.content(tensor) == tl.fingerprint.content(twin)


class _OneElementTail(nn.Module):
    """Ends in a one-element slice whose stride is the row width."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.proj(x))
        return hidden[0:1, 1]


@pytest.fixture
def tail_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = _OneElementTail()
    x = torch.rand(2, 4) + 1.0
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def test_fork_push_through_a_one_element_strided_site(tail_trace: tl.Trace) -> None:
    relu = next(op.label for op in tail_trace.ops if op.func_name == "relu")
    fork = tail_trace.fork()
    fork.do(tl.label(relu), tl.scale(0.5))
    output = next(op.label for op in tail_trace.ops if op.label.startswith("output"))
    assert torch.equal(fork[output].out, tail_trace[output].out * 0.5)
