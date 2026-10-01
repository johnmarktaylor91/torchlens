"""Default-path guarantees (memo sec 4.4 / 8.1 item 10 / D19 provenance).

Two pins protect everyone who never touches meta: ordinary captures are
byte-identical with the weightsfree machinery quiescent (W1 and W1-CTX touch
the recording path), and the ambient-device-context row is pinned per D19's
disposition.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.smoke


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = y + torch.ones(1, 4)
        return torch.relu(2.0 * y)


def test_ordinary_capture_pays_no_weightsfree_machinery() -> None:
    """The factory slot is empty, the transparency belt is dormant, and no
    admission registers on an ordinary capture."""

    from torchlens.backends.torch._weightsfree_ctx import active_factory_device
    from torchlens.capture._weightsfree_admission import (
        admission_record_for,
        weightsfree_meta_active,
    )

    trace = tl.trace(Toy(), torch.randn(2, 4))
    assert active_factory_device() is None
    assert admission_record_for(trace) is None
    assert not weightsfree_meta_active(trace)
    assert trace.structure_evidence is None


def test_ordinary_digest_stable_across_repeat_captures() -> None:
    """Same model, same input: the public digest is deterministic and the
    weightsfree wave leaves the ordinary path byte-identical run to run."""

    torch.manual_seed(0)
    model = Toy()
    model.eval()
    x = torch.randn(2, 4)
    first = tl.hash.trace(tl.trace(model, x))
    second = tl.hash.trace(tl.trace(model, x))
    assert first == second


def test_ambient_device_context_row_d19() -> None:
    """D19 pin: a plain real capture inside `with torch.device("cpu")` must
    match the no-context digest once absorption lands; until then this row is
    the documented KNOWN-RED — never a silent skip.

    The dunder respelling (defect L3's default-path form): the ambient
    DeviceContext's catch-all torch-function re-entry respells `2.0 * y`
    (`__rmul__`) into functional spellings, silently changing the public
    digest for the same model and input.
    """

    from torchlens.utils._torch_compat import HAS_TORCH_FUNCTION_STACK_SURGERY

    torch.manual_seed(0)
    model = Toy()
    model.eval()
    x = torch.randn(2, 4)
    bare = tl.hash.trace(tl.trace(model, x))
    with torch.device("cpu"):
        inside = tl.hash.trace(tl.trace(model, x))
    if inside == bare:
        return  # absorption (or an inert context) already holds the pin
    if not HAS_TORCH_FUNCTION_STACK_SURGERY:
        pytest.skip("mode-stack surgery unavailable; absorption cannot land here")
    pytest.xfail(
        "KNOWN-RED (documented, weightsfree memo D19/item 19): the ambient "
        "DeviceContext respells dunder ops and changes the public digest; "
        "W1-ABSORB is the funded default-path fix"
    )


def test_admitted_capture_leaves_wrap_state_clean() -> None:
    """After an admitted meta capture, an ordinary capture behaves exactly as
    before (wrapper install state, factory slot, autocast object)."""

    original_autocast = torch.autocast
    with torch.device("meta"):
        meta_model = Toy()
    meta_model.eval()
    tl.trace(
        meta_model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    assert torch.autocast is original_autocast
    trace = tl.trace(Toy(), torch.randn(2, 4))
    assert trace.outcome.status.value == "complete"
    assert trace["ones_1_2"].out is not None  # factory op landed on CPU with real values
