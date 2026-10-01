"""TensorBoardSink: the reference, CI-tested transport (memo 3.1, 3.10).

Writes through torch's own ``SummaryWriter`` (the exact path the relays
patch). Histograms go through ``add_histogram_raw`` with exact counts on
explicit edges -- the receiving API that makes an honest low-cost watcher
possible at all (credits, memo section 12).

Relay law (G6, measured): the C06 signed-log2 grid is NON-uniform, and both
relays reconstruct or resample edges in ways that are exact only for uniform
bins (wandb) or never (ClearML). So when a relay is DETECTED, histogram
emission refuses typed -- scalars flow (both relays carry scalars, wandb
rewrites their step axis and that is disclosed, not repaired). Detection:
wandb patches ``tensorboard`` visibly (``wandb.patched["tensorboard"]``);
ClearML rebinds ``FileWriter.add_event`` (module check). The ClearML
detector evidences CONFIGURATION, not capture -- the delivery vocabulary
survives the weakest detector (memo 3.10).
"""

from __future__ import annotations

import sys
from typing import Any

from ._errors import SinkProtocolError, TrackersError
from ._records import HistogramPoint, ScalarPoint, TextPoint

__tl_layer__ = "L8"


def _summary_writer_class() -> Any:
    """Import torch's SummaryWriter; refuse typed naming the extra."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SinkProtocolError(
            "TensorBoardSink needs the tensorboard package (torch's "
            f"SummaryWriter imports it): {exc}.",
            code="tracker_sink_unavailable",
            sink="TensorBoardSink",
            remedy='pip install "torchlens[tensorboard]" (or tensorboard>=2.21).',
        ) from exc
    return SummaryWriter


def detect_relays() -> dict[str, str]:
    """Best-effort relay detection: ``{vendor: evidence}`` for active relays.

    wandb: ``wandb.patched["tensorboard"]`` non-empty after
    ``wandb.init(sync_tensorboard=True)`` (verified, one line). ClearML:
    torch's ``FileWriter.add_event`` rebound into
    ``clearml.binding.frameworks.tensorflow_bind`` (configuration evidence
    only). Detection never imports a vendor package that is not already in
    the process.
    """

    relays: dict[str, str] = {}
    wandb_module = sys.modules.get("wandb")
    if wandb_module is not None:
        patched = getattr(wandb_module, "patched", None)
        if isinstance(patched, dict) and patched.get("tensorboard"):
            relays["wandb"] = "wandb.patched['tensorboard'] non-empty"
    if "clearml" in sys.modules:
        try:
            from torch.utils.tensorboard.writer import FileWriter

            module = getattr(FileWriter.add_event, "__module__", "") or ""
            if module.startswith("clearml."):
                relays["clearml"] = f"FileWriter.add_event rebound into {module}"
        except ImportError:
            # clearml present but tensorboard absent: no FileWriter to have
            # been patched, so no relay evidence exists either way.
            pass
    return relays


class TensorBoardSink:
    """The reference sink over an existing writer or a fresh log dir."""

    def __init__(self, writer_or_logdir: Any) -> None:
        """Wrap an existing ``SummaryWriter`` or open one on a log dir.

        Relay detection runs at construction (after any
        ``wandb.init(sync_tensorboard=True)`` the user issued first, per the
        documented ordering) and is re-checked per histogram emission.
        """

        if hasattr(writer_or_logdir, "add_scalar"):
            self._writer = writer_or_logdir
            self._owns_writer = False
        else:
            self._writer = _summary_writer_class()(log_dir=str(writer_or_logdir))
            self._owns_writer = True
        self._closed = False
        self.relays = detect_relays()

    def capabilities(self) -> frozenset[str]:
        """Scalars, raw histograms, text, graph, and run metadata."""

        return frozenset({"scalar", "raw_histogram", "text_manifest", "graph", "run_metadata"})

    def relay_state(self) -> tuple[str, str | None]:
        """Delivery-state token + detail for the close report (3.10)."""

        if not self.relays:
            return ("emitted", None)
        detail = "; ".join(f"{vendor}: {evidence}" for vendor, evidence in self.relays.items())
        return ("relay_configured", detail)

    def preflight_histograms(self, descriptor: Any) -> None:
        """Refuse histogram requests over a DETECTED relay at attach.

        Same refusal as ``emit_histogram`` (G6), moved to the moment the user
        is looking: raised inside emission it was swallowed by the engine's
        runtime latch and silently stopped every later row (AUD-CODE 2.14).
        """

        del descriptor
        self.relays = detect_relays() or self.relays
        if self.relays:
            raise TrackersError(
                f"A TensorBoard relay is active ({', '.join(sorted(self.relays))}) "
                "and histogram series ride NON-uniform log2 edges the relay would "
                "silently deliver wrong; refusing at attach, before any emission.",
                code="tracker_relay_histogram_unsupported",
                sink="TensorBoardSink",
                relays=tuple(sorted(self.relays)),
                remedy=(
                    "Send histograms through a native sink (WandbSink / "
                    "JSONLSink), or drop hist_every= on this relay run; "
                    "scalar series remain faithful on every measured route."
                ),
            )

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Write one scalar through ``add_scalar`` at the caller's step."""

        self._writer.add_scalar(point.tag, point.value, global_step=point.step)

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Write exact counts/edges through ``add_histogram_raw``.

        Refuses when a relay is active: the signed-log2 edges are
        non-uniform, and a relay would silently reconstruct them wrong
        (wandb: linear outer-edge extrapolation; ClearML: resampled 3-D
        surface losing ~25% of the mass -- both measured).
        """

        self.relays = detect_relays() or self.relays
        if self.relays:
            raise TrackersError(
                f"A TensorBoard relay is active ({', '.join(sorted(self.relays))}) "
                f"and histogram series ride NON-uniform log2 edges; the relay "
                "would silently deliver wrong distributions (measured: wandb "
                "reconstructs outer edges by linear extrapolation, ClearML "
                "resamples to ~48 columns conserving 75% of the mass).",
                code="tracker_relay_histogram_unsupported",
                sink="TensorBoardSink",
                relays=tuple(sorted(self.relays)),
                remedy=(
                    "Send histograms through a native sink (WandbSink / "
                    "JSONLSink), or drop histogram signals on this relay run; "
                    "scalar series remain faithful on every measured route."
                ),
            )
        summary = point.summary
        self._writer.add_histogram_raw(
            point.tag,
            min=summary.get("min", 0.0),
            max=summary.get("max", 0.0),
            num=int(summary.get("num", sum(point.counts))),
            sum=summary.get("sum", 0.0),
            sum_squares=summary.get("sum_squares", 0.0),
            bucket_limits=list(point.edges[1:]),
            bucket_counts=list(point.counts),
            global_step=point.step,
        )

    def emit_text(self, point: TextPoint) -> None:
        """Write one text payload at the caller's step."""

        self._writer.add_text(point.tag, point.text, global_step=point.step)

    def flush(self) -> None:
        """Flush the writer."""

        self._writer.flush()

    def close(self) -> None:
        """Close the writer iff this sink opened it; idempotent."""

        if self._closed:
            return
        self._closed = True
        self._writer.flush()
        if self._owns_writer:
            self._writer.close()


__all__ = ["TensorBoardSink", "detect_relays"]
