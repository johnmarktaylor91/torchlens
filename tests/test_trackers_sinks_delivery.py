"""F26: sink protocol, JSONL artifact, TB duck-writer path, transfer oracle."""

from __future__ import annotations

import pytest

import torchlens.trackers as trk
from torchlens.trackers._errors import SinkDeliveryError, SinkProtocolError, TrackersError


class _DuckWriter:
    """A dependency-free object with the SummaryWriter surface."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.raw_histograms: list[dict] = []
        self.texts: list[tuple[str, str, int]] = []
        self.flushed = 0
        self.closed = 0

    def add_scalar(self, tag, value, global_step):  # noqa: ANN001
        self.scalars.append((tag, value, global_step))

    def add_histogram_raw(self, tag, **kwargs):  # noqa: ANN001
        self.raw_histograms.append({"tag": tag, **kwargs})

    def add_text(self, tag, text, global_step):  # noqa: ANN001
        self.texts.append((tag, text, global_step))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


class TestProtocol:
    """Capability law: refuse by name before partial emission."""

    def test_non_sink_refuses(self) -> None:
        with pytest.raises(SinkProtocolError) as info:
            trk.require_capabilities(object(), ("scalar",))
        assert info.value.fields["code"] == "tracker_sink_invalid"

    def test_unknown_capability_refuses(self) -> None:
        class _Weird:
            def capabilities(self):
                return frozenset({"scalar", "telepathy"})

        with pytest.raises(SinkProtocolError) as info:
            trk.require_capabilities(_Weird(), ("scalar",))
        assert info.value.fields["unknown"] == ("telepathy",)


class TestJSONLSink:
    """Build item 11: persisted rows, header first, footer marks complete."""

    def test_write_failure_latches(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "run.jsonl"
        sink = trk.JSONLSink(path)
        sink._handle.close()  # simulate the filesystem dying mid-run
        with pytest.raises(SinkDeliveryError) as info:
            sink.emit_scalar(trk.ScalarPoint("a/b/c", 0, 1.0))
        assert info.value.fields["code"] == "tracker_sink_delivery_failed"
        with pytest.raises(SinkDeliveryError):
            sink.emit_scalar(trk.ScalarPoint("a/b/c", 1, 1.0))
        # The partial file has no closing footer: honestly marked partial.
        assert '"closed"' not in path.read_text()


class TestTensorBoardSinkDuckPath:
    """The reference sink over an EXISTING writer object (dep-free)."""

    def test_emits_through_writer_at_caller_step(self) -> None:
        writer = _DuckWriter()
        sink = trk.TensorBoardSink(writer)
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 104, 2.0))
        sink.emit_histogram(
            trk.HistogramPoint(
                "gradients/hist/w",
                104,
                (1, 2, 3),
                (0.0, 1.0, 2.0, 4.0),
                {"num": 6.0, "min": 0.5, "max": 3.5, "sum": 7.0, "sum_squares": 9.0},
            )
        )
        sink.close()
        assert writer.scalars == [("gradients/norm/w", 2.0, 104)]
        raw = writer.raw_histograms[0]
        assert raw["bucket_counts"] == [1, 2, 3]
        assert raw["bucket_limits"] == [1.0, 2.0, 4.0]
        assert raw["num"] == 6
        assert writer.flushed >= 1
        # The sink never closes a writer it did not open.
        assert writer.closed == 0

    def test_relay_detected_refuses_histograms(self, monkeypatch) -> None:  # noqa: ANN001
        """G6: non-uniform log2 edges over a detected relay refuse typed."""

        import sys
        import types

        fake_wandb = types.ModuleType("wandb")
        fake_wandb.patched = {"tensorboard": [("torch.utils.tensorboard", "SummaryWriter")]}
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        writer = _DuckWriter()
        sink = trk.TensorBoardSink(writer)
        assert sink.relays == {"wandb": "wandb.patched['tensorboard'] non-empty"}
        assert sink.relay_state()[0] == "relay_configured"
        # Scalars flow (faithful on every measured route)...
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 0, 1.0))
        # ...histograms refuse: the relay would deliver wrong distributions.
        with pytest.raises(TrackersError) as info:
            sink.emit_histogram(trk.HistogramPoint("gradients/hist/w", 0, (1,), (0.0, 1.0), {}))
        assert info.value.fields["code"] == "tracker_relay_histogram_unsupported"


class TestWandbSinkContract:
    """Native sink invariants that need no wandb install."""

    def test_requires_live_run(self) -> None:
        with pytest.raises(SinkProtocolError) as info:
            trk.WandbSink("/tmp/not-a-run")
        assert info.value.fields["code"] == "tracker_sink_invalid"

    def test_bucket_cap_refuses_before_rebinning(self) -> None:
        class _Run:
            def log(self, *a, **k):  # noqa: ANN001,ANN002,ANN003
                raise AssertionError("must refuse before logging")

        sink = trk.WandbSink(_Run())
        point = trk.HistogramPoint(
            "gradients/hist/w",
            0,
            tuple([1] * 600),
            tuple(float(i) for i in range(601)),
            {},
        )
        with pytest.raises(TrackersError) as info:
            sink.emit_histogram(point)
        assert info.value.fields["code"] == "tracker_histogram_bucket_cap"

    def test_scalar_logs_at_caller_step(self) -> None:
        logged: list[tuple[dict, int]] = []

        class _Run:
            def log(self, payload, step=None):  # noqa: ANN001
                logged.append((payload, step))

        sink = trk.WandbSink(_Run())
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 102, 3.0))
        assert logged == [({"gradients/norm/w": 3.0}, 102)]
