"""Echo failure package: two tails, exact ordering, original exception wins.

Pins snoop memo test rows 5 and 7 (crash tails on both tiers, exact tail
order, exception identity preserved), the double-fault contract (a broken
sink never outranks the model's own error), interrupt handling, the
raise_on_nan composition (last narrated line is the minting op), and the
halted-vs-failed footer distinction.
"""

from __future__ import annotations

import io
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import EchoOptions

pytestmark = pytest.mark.smoke


class ShapeCrash(nn.Module):
    """Real shape crash: (2,8) activations reach LayerNorm(768)."""

    def __init__(self) -> None:
        """Build fc -> relu -> mis-shaped LayerNorm."""

        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.ln = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Crash inside the LayerNorm."""

        return self.ln(torch.relu(self.fc(x)))


def _crash_tail(sink: io.StringIO) -> list[str]:
    """Run the shape crash on the trace tier and return the sink lines."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            tl.trace(
                ShapeCrash(),
                torch.randn(2, 4),
                echo=EchoOptions(select=True, sink=sink),
            )
    return [line for line in sink.getvalue().split("\n") if line]


def test_crash_tail_order_and_attempted_marker() -> None:
    """Memo test 5: qualified attempted line last, frontier fields correct."""

    lines = _crash_tail(io.StringIO())
    failed_index = next(i for i, line in enumerate(lines) if line.startswith("!! forward failed"))
    assert "RuntimeError" in lines[failed_index]
    assert any(line.startswith("!! last completed event=") and "relu" in line for line in lines)
    assert any(line.startswith("!! last narrated event=") for line in lines)
    attempted = [line for line in lines if line.startswith("!! attempted call=layer_norm")]
    assert len(attempted) == 1
    assert "[2,8]" in attempted[0]
    assert "(not proven culprit)" in attempted[0]
    assert lines[-1].startswith("-- end echo tail")
    # The tail REPRINTS lines that already appeared live.
    live_relu = [i for i, line in enumerate(lines) if "relu_1" in line and "#" in line]
    assert len(live_relu) >= 2


def test_on_error_only_narrates_nothing_until_failure() -> None:
    """Memo test 9 shape: happy passes emit zero bytes; the crash speaks."""

    sink = io.StringIO()
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    tl.trace(model, torch.randn(2, 4), echo=EchoOptions(select=True, sink=sink, on_error_only=True))
    assert sink.getvalue() == ""
    tail_sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            tl.trace(
                ShapeCrash(),
                torch.randn(2, 4),
                echo=EchoOptions(select=True, sink=tail_sink, on_error_only=True),
            )
    assert "!! forward failed" in tail_sink.getvalue()


def test_record_tier_disposition_trio_flushes_the_tail() -> None:
    """Memo test 7 shape: raise / attach_partial / return_partial all flush."""

    x = torch.randn(2, 4)
    for mode in ("raise", "attach_partial", "return_partial"):
        sink = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = tl.record(
                    ShapeCrash(),
                    x,
                    echo=EchoOptions(select=True, sink=sink),
                    on_forward_error=mode,
                )
        except RuntimeError as exc:
            assert mode in ("raise", "attach_partial")
            if mode == "attach_partial":
                assert getattr(exc, "partial_recording", None) is not None
        else:
            assert mode == "return_partial"
            assert result.failed
        assert "!! forward failed" in sink.getvalue()
        assert "attempted call=layer_norm" in sink.getvalue()


def test_partial_narrate_works_with_echo_off() -> None:
    """Tail 1, the headline: post-hoc narration needs no foresight."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError) as excinfo:
            tl.trace(ShapeCrash(), torch.randn(2, 4))
    rendered = excinfo.value.partial_log.narrate(20)
    assert "!! forward failed: RuntimeError" in rendered
    assert "relu" in rendered
    assert "the raising call is not in the record" in rendered


def test_trace_and_recording_narrate_render_the_same_grammar() -> None:
    """One renderer, three carriers: tail idiom on finished products."""

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    x = torch.randn(2, 4)
    trace_block = tl.trace(model, x).narrate(3)
    assert len([line for line in trace_block.split("\n") if line]) == 3
    recording = tl.record(model, x, save=tl.func("relu"))
    recording_block = recording.narrate()
    assert "relu" in recording_block
    filtered = tl.trace(model, x).narrate(select="relu")
    assert "relu" in filtered and "linear" not in filtered


def test_broken_sink_disables_narrator_and_capture_continues() -> None:
    """Observer failures are instrumentation failures: warn once, continue."""

    calls: list[str] = []

    def broken(line: str) -> None:
        """Explode on the second line."""

        calls.append(line)
        if len(calls) >= 2:
            raise OSError("sink died")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
            torch.randn(2, 4),
            echo=EchoOptions(select=True, sink=broken),
        )
    assert log["relu_1_2"].shape == (2, 8)
    codes = [getattr(item.message, "fields", {}).get("code") for item in caught]
    assert codes.count("echo_sink_disabled") == 1


def test_double_fault_original_exception_wins() -> None:
    """A sink that dies during the crash tail never outranks the model error."""

    def broken(line: str) -> None:
        """Explode on the failure tail."""

        if line.startswith("!!"):
            raise OSError("sink died mid-tail")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError) as excinfo:
            tl.trace(
                ShapeCrash(),
                torch.randn(2, 4),
                echo=EchoOptions(select=True, sink=broken),
            )
    assert "normalized_shape" in str(excinfo.value)


def test_keyboard_interrupt_stays_an_interrupt() -> None:
    """KeyboardInterrupt gets a best-effort flush and keeps its type."""

    class Interrupted(nn.Module):
        """Raises KeyboardInterrupt mid-forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op then interrupt."""

            _ = torch.relu(x)
            raise KeyboardInterrupt

    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyboardInterrupt):
            tl.trace(
                Interrupted(),
                torch.randn(2, 4),
                echo=EchoOptions(select=True, sink=sink),
            )


def test_raise_on_nan_minting_op_is_last_narrated_and_census_exact() -> None:
    """Memo crash D shape: tripwire reuse -- the crash line has the tensor."""

    class Minter(nn.Module):
        """Mints an inf via division by zero."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """relu then divide by zero."""

            y = torch.relu(x)
            return y / torch.zeros_like(y)

    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as excinfo:
            tl.trace(
                Minter(),
                torch.randn(2, 4),
                echo=EchoOptions(select=True, sink=sink),
                capture=tl.options.CaptureOptions(raise_on_nan=True),
            )
    assert "div" in str(excinfo.value) or "nonfinite" in str(excinfo.value).lower()
    lines = [line for line in sink.getvalue().split("\n") if line]
    tripwire = [line for line in lines if "raise_on_nan tripped at" in line]
    # Once live, once in the deliberately REPRINTED tail (locating the
    # frontier in a long scrollback IS the feature).
    assert len(tripwire) == 2
    assert "div" in tripwire[0]
    assert "(exact)" in tripwire[0]
    narrated = [line for line in lines if line.startswith("!! last narrated event=")]
    assert narrated and "div" in narrated[0]


def test_halted_capture_footer_says_halted_not_failed() -> None:
    """Halted (halt=/stop_after) is not failed; the footer distinguishes."""

    sink = io.StringIO()
    log = tl.trace(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
        torch.randn(2, 4),
        halt=tl.func("relu"),
        echo=EchoOptions(select=True, sink=sink),
    )
    assert log is not None
    transcript = sink.getvalue()
    assert "halted, not failed" in transcript
    assert "!! forward failed" not in transcript
