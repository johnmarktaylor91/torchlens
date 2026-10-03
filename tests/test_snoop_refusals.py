"""Echo typed refusals: every door teaches at the point of failure.

Each refusal here is CONTRACTED (S-17): stable ``fields["code"]``, non-empty
``fields["remedy"]``, a contract-doc row -- and the negative composition
cells from snoop memo section 6 (cache, chunked, hooks-mount,
narrator-as-save-predicate, halt-only fast path).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.fastlog.options import RecordingOptions
from torchlens.options import EchoOptions
from torchlens.snoop import EchoConfigError, normalize_echo


class OneOp(nn.Module):
    """Single-op module for refusal tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relu."""

        return torch.relu(x)


def _code(excinfo: pytest.ExceptionInfo) -> str:
    """Return the typed refusal code of a raised TorchLens error."""

    return excinfo.value.fields.get("code", "")


def test_echo_string_selection_refuses_with_post_hoc_remedy() -> None:
    """Label/substring echo= spellings refuse: final labels are not live."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(OneOp(), torch.randn(2, 4), echo="relu_1_1")
    assert _code(excinfo) == "echo_argument_invalid"
    assert "narrate" in str(excinfo.value)


def test_echo_unsupported_type_refuses() -> None:
    """Non-callable echo= objects refuse typed."""

    with pytest.raises(EchoConfigError) as excinfo:
        normalize_echo(123)
    assert _code(excinfo) == "echo_argument_invalid"


def test_echo_unknown_stats_rung_refuses() -> None:
    """The four measured cost classes are a closed vocabulary."""

    with pytest.raises(EchoConfigError) as excinfo:
        normalize_echo(EchoOptions(select=True, stats="approximate"))
    assert _code(excinfo) == "echo_argument_invalid"


def test_finalized_label_selector_refuses_before_execution() -> None:
    """Selectors needing finalized graph facts refuse BEFORE the forward."""

    ran: list[bool] = []

    class Spy(nn.Module):
        """Records whether the forward executed."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Mark execution and pass through."""

            ran.append(True)
            return torch.relu(x)

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(Spy(), torch.randn(2, 4), echo=tl.output(0))
    assert _code(excinfo) == "echo_selector_not_live"
    assert ran == []


def test_echo_options_as_save_predicate_refuses_with_receipt() -> None:
    """The narrator-as-save-predicate mount refuses citing double evaluation."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(OneOp(), torch.randn(2, 4), save=EchoOptions(select=True))
    assert _code(excinfo) == "echo_as_save_predicate"
    assert "twice" in str(excinfo.value)


def test_echo_options_via_hooks_refuses_with_receipt() -> None:
    """The hooks-mount refuses citing the measured +101% tax."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(
            OneOp(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(hooks=[EchoOptions(select=True)]),
        )
    assert _code(excinfo) == "echo_via_hooks_unsupported"


def test_cache_true_with_echo_refuses_before_execution() -> None:
    """T-D: a cache hit runs no forward, so nothing live exists to narrate."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(
            OneOp(),
            torch.randn(2, 4),
            echo=True,
            capture=tl.options.CaptureOptions(cache=True),
        )
    assert _code(excinfo) == "echo_cache_unsupported"


def test_chunked_forward_with_echo_refuses() -> None:
    """Chunk fan-out runs several forwards into one Trace: refused typed."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(OneOp(), torch.randn(4, 4), echo=True, chunk_size=2)
    assert _code(excinfo) == "echo_chunked_unsupported"


def test_structure_only_with_stats_rung_refuses() -> None:
    """A stats rung reads values; structure_only captures none."""

    with pytest.raises(EchoConfigError) as excinfo:
        tl.trace(
            OneOp(),
            torch.randn(2, 4),
            echo=EchoOptions(select=True, stats="sampled"),
            capture=tl.options.CaptureOptions(structure_only=True),
        )
    assert _code(excinfo) == "echo_stats_requires_values"


def test_record_with_only_echo_is_a_legal_capture() -> None:
    """The echo term in the non-empty-capture validation (snoop row 1)."""

    recording = tl.record(
        OneOp(), torch.randn(2, 4), echo=EchoOptions(select=True, sink=lambda _: None)
    )
    assert recording.records == []


def test_echo_on_non_torch_backend_refuses_typed() -> None:
    """echo= on a preview backend refuses typed, never a silent no-op."""

    from types import SimpleNamespace

    from torchlens.snoop._entry import refuse_echo_non_torch

    with pytest.raises(EchoConfigError) as excinfo:
        refuse_echo_non_torch(True, SimpleNamespace(name="tf"))
    assert _code(excinfo) == "echo_backend_unsupported"
    refuse_echo_non_torch(None, SimpleNamespace(name="tf"))
    refuse_echo_non_torch(True, SimpleNamespace(name="torch"))


def test_record_without_echo_or_save_still_refuses() -> None:
    """The pre-existing empty-capture refusal is untouched."""

    from torchlens.fastlog.exceptions import RecordingConfigError

    with pytest.raises(RecordingConfigError):
        tl.record(OneOp(), torch.randn(2, 4))


def test_halt_only_fast_path_declines_when_echo_is_armed() -> None:
    """The echo term in ``_is_halt_only_capture`` (snoop row 1)."""

    from torchlens.capture.predicates import _is_halt_only_capture

    halt_only = RecordingOptions(halt=lambda ctx: False)
    assert _is_halt_only_capture(halt_only)
    with_echo = RecordingOptions(halt=lambda ctx: False, echo=EchoOptions(select=True))
    assert not _is_halt_only_capture(with_echo)
