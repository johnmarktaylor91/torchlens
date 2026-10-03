"""F26: sink protocol, JSONL artifact, TB duck-writer path, transfer oracle."""

from __future__ import annotations

import json

import pytest
import torch

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

    def test_missing_capability_named(self) -> None:
        sink = trk.MemorySink(capabilities=frozenset({"scalar"}))
        with pytest.raises(SinkProtocolError) as info:
            trk.require_capabilities(sink, ("scalar", "raw_histogram"))
        assert info.value.fields["missing"] == ("raw_histogram",)


class TestJSONLSink:
    """Build item 11: persisted rows, header first, footer marks complete."""

    def test_round_trip(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "run.jsonl"
        sink = trk.JSONLSink(path)
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 100, 1.5))
        sink.emit_histogram(
            trk.HistogramPoint("gradients/hist/w", 100, (1, 2), (0.0, 1.0, 2.0), {"num": 3.0})
        )
        sink.emit_text(trk.TextPoint("torchlens/meta/manifest", 0, "{}"))
        sink.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["format"] == trk.JSONL_FORMAT
        assert rows[1] == {
            "kind": "scalar",
            "tag": "gradients/norm/w",
            "step": 100,
            "value": 1.5,
        }
        assert rows[2]["counts"] == [1, 2]
        assert rows[2]["edges"] == [0.0, 1.0, 2.0]
        assert rows[-1] == {"kind": "closed"}

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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
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

    def test_no_relay_reports_clean(self) -> None:
        sink = trk.TensorBoardSink(_DuckWriter())
        assert sink.relays == {}
        assert trk.detect_relays() == {}


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

    def test_safe_descriptor_fits_the_cap(self) -> None:
        """The named remedy really renders under 512 buckets."""

        bins = 2 * trk.WANDB_SAFE_DESCRIPTOR.bins_per_side + 1
        assert bins <= trk.WANDB_BUCKET_CAP

    def test_scalar_logs_at_caller_step(self) -> None:
        logged: list[tuple[dict, int]] = []

        class _Run:
            def log(self, payload, step=None):  # noqa: ANN001
                logged.append((payload, step))

        sink = trk.WandbSink(_Run())
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 102, 3.0))
        assert logged == [({"gradients/norm/w": 3.0}, 102)]


class TestBoundedTransferOracle:
    """C-WATCH's CPU-shaped half: emitted bytes independent of tensor numel.

    The CUDA transfer row runs in the D02 cluster campaign; what is
    provable here with zero GPUs is the sink-boundary law from memo 3.12 --
    O(bins) numbers cross per site per sampled step, INDEPENDENT of numel.
    """

    @staticmethod
    def _emitted_bytes(width: int, tmp_path) -> tuple[int, int]:  # noqa: ANN001
        torch.manual_seed(0)
        model = torch.nn.Linear(width, width)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        path = tmp_path / f"run_{width}.jsonl"
        session = trk.watch(
            model,
            to=trk.JSONLSink(path),
            signals=("gradients",),
            optimizer=opt,
            every=1,
            hist_every=1,
        )
        with session, session.step(0):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(2, width)).sum().backward()
            opt.step()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        payload_rows = [r for r in rows if r.get("kind") in ("scalar", "histogram")]
        per_step_bytes = sum(len(json.dumps(r)) for r in payload_rows if r.get("step") == 0)
        return (len(payload_rows), per_step_bytes)

    @pytest.mark.smoke
    def test_numel_independence(self, tmp_path) -> None:  # noqa: ANN001
        """4096x more elements -> the SAME emission row count and geometry."""

        rows_small, bytes_small = self._emitted_bytes(8, tmp_path)
        rows_large, bytes_large = self._emitted_bytes(512, tmp_path)
        assert rows_small == rows_large
        # Byte payloads differ only in digit widths, never in structure:
        # well under 2x while numel grew 4096x.
        assert bytes_large < 2 * bytes_small
