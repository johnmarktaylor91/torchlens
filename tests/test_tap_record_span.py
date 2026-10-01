"""Phase 5a tests for taps, record spans, and report values."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.report import log_value


class MetricModel(torch.nn.Module):
    """Model that records a report value during capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ReLU and record a scalar value.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        y = torch.relu(x)
        log_value("out_sum", float(y.sum()))
        return y


def test_tap_records_without_modifying_output() -> None:
    """Tap hooks should record values and leave model output unchanged."""

    model = torch.nn.ReLU()
    x = torch.tensor([[-1.0, 2.0]])
    tap = tl.tap(tl.func("relu"))

    hooked = tl.trace(
        model, x, capture=tl.options.CaptureOptions(intervention_ready=True, hooks=tap)
    )
    plain = tl.trace(model, x)

    assert tap.values()
    assert torch.equal(tap.values()[0], torch.relu(x))
    assert torch.equal(
        hooked[hooked.output_layers[0]].out,
        plain[plain.output_layers[0]].out,
    )


def test_record_span_and_log_value_metadata() -> None:
    """Record spans and logged values should land on the captured Trace."""

    with tl.span("phase_name"):
        log = tl.trace(MetricModel(), torch.tensor([[-1.0, 2.0]]))

    assert log.observer_spans
    assert log.observer_spans[0]["name"] == "phase_name"
    assert log.observer_spans[0]["end"] is not None
    assert log.annotations["logged_values"]["out_sum"] == 2.0
    assert not hasattr(log, "report_values")


def test_record_spans_are_context_local_across_threads() -> None:
    """One thread's active span must never annotate another thread's reads (R54).

    ``_state._active_record_spans`` was a plain process-global list: a
    two-thread barrier probe produced ``right=('left', 'right')`` -- one
    thread observed the other's active span, so interleaved captures/taps
    received foreign annotations. The registry is now a context-local
    ContextVar; each thread sees exactly its own spans.
    """

    import threading

    from torchlens import observers

    results: dict[str, tuple[str, ...]] = {}
    barrier = threading.Barrier(2, timeout=10)

    def worker(name: str) -> None:
        with observers.span(name):
            barrier.wait()
            results[name] = tuple(str(record["name"]) for record in observers.active_span_records())
            barrier.wait()

    threads = [threading.Thread(target=worker, args=(side,)) for side in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert results["left"] == ("left",)
    assert results["right"] == ("right",)
    assert observers.active_span_records() == []


def test_pause_logging_exit_never_blinds_a_newly_published_capture() -> None:
    """A stale pause restore must not disable another thread's live capture (R54/b2:A2).

    Interleaving: analysis thread B reads owner=None an instant before
    capture thread A's locked publication enables logging; B's __exit__ then
    restored its stale pre-pause value and silently blinded the remainder of
    A's forward (ops dropped, no error). The restore now re-checks ownership.
    """

    import threading

    from torchlens import _state

    saved_owner = _state._active_owner_thread_id
    saved_enabled = _state._logging_enabled
    try:
        # B enters pause_logging with no owner published yet.
        _state._active_owner_thread_id = None
        _state._logging_enabled = False
        pause = _state.pause_logging()
        pause.__enter__()
        # A's atomic publication lands between B's enter and exit.
        _state._active_owner_thread_id = threading.get_ident() + 1
        _state._logging_enabled = True
        pause.__exit__(None, None, None)
        assert _state._logging_enabled is True, (
            "a non-owner pause restore blinded the newly published capture"
        )
    finally:
        _state._active_owner_thread_id = saved_owner
        _state._logging_enabled = saved_enabled
