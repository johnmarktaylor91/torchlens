"""WandbSink: the thin native serializer over the same records (memo 3.1).

Justified by measurement, not preference: the wandb TB relay rewrites the
caller's step axis (100/102/104 arrive as 0/1/2), destroys every histogram
summary field, and silently re-bins >512 buckets -- so the native sink
exists to preserve exactly what the relay loses. It is a serializer
(``run.log`` + ``wandb.Histogram(np_histogram=(counts, edges))``), never an
engine; ~100 lines over the shared emission view, as designed.

The 512-bucket cap is wandb's own (documented constructor limit); the
default C06 grid at bpo=4 over [2^-48, 2^16] renders 513 bins per signed
sketch, so the sink REDUCES bins-per-octave pressure honestly: it refuses
typed rather than letting wandb silently re-bin. Callers watching into wandb
pick a narrower descriptor (the refusal names the arithmetic).
"""

from __future__ import annotations

from typing import Any

from ..observability import HistogramDescriptor
from ._errors import SinkProtocolError, TrackersError
from ._records import HistogramPoint, ScalarPoint, TextPoint

__tl_layer__ = "L8"

#: wandb.Histogram's documented maximum bucket count.
WANDB_BUCKET_CAP = 512

#: A descriptor whose signed render (2 * bins_per_side + 1 center band) fits
#: the wandb cap: (16 - (-46)) * 4 = 248 bins/side -> 497 buckets. The C06
#: DEFAULT grid renders 513 and would refuse; watch(..., descriptor=
#: WANDB_SAFE_DESCRIPTOR) is the one-line remedy the refusal names.
WANDB_SAFE_DESCRIPTOR = HistogramDescriptor(lo_exp=-46)


def _wandb_module() -> Any:
    """Import wandb lazily; refuse typed naming the extra."""

    try:
        import wandb
    except ImportError as exc:
        raise SinkProtocolError(
            f"WandbSink needs the wandb package: {exc}.",
            code="tracker_sink_unavailable",
            sink="WandbSink",
            remedy='pip install "torchlens[wandb]".',
        ) from exc
    return wandb


class WandbSink:
    """Native wandb receiver over an existing ``wandb.Run``."""

    def __init__(self, run: Any) -> None:
        """Wrap an existing run object (never calls ``wandb.init`` itself).

        Activation is a visible call in user source: constructing the run is
        the user's act, and this sink never creates one (the enablement law,
        memo 3.16).
        """

        if not callable(getattr(run, "log", None)):
            raise SinkProtocolError(
                f"WandbSink needs a live wandb Run (got {type(run).__name__}); "
                "it never calls wandb.init itself -- constructing the run is "
                "the user's visible enablement act.",
                code="tracker_sink_invalid",
                sink="WandbSink",
                remedy="Pass the object returned by wandb.init(...).",
            )
        self._run = run
        self._closed = False

    def capabilities(self) -> frozenset[str]:
        """Scalars, raw histograms, and text; graphs are demoted (memo s9)."""

        return frozenset({"scalar", "raw_histogram", "text_manifest", "run_metadata"})

    def preflight_histograms(self, descriptor: HistogramDescriptor) -> None:
        """Refuse an over-cap grid at ATTACH, before any emission.

        The engine calls this when histograms are requested. Refusing here
        (typed, while the user is at the console) replaces the old shape of
        the same refusal: raised inside ``emit_histogram`` on the first
        sampled step, where the engine's runtime latch swallowed it and
        stopped ALL delivery to this sink after step 0 (AUD-CODE 2.14).
        """

        buckets = 2 * len(descriptor.bucket_edges()) - 1
        if buckets > WANDB_BUCKET_CAP:
            raise TrackersError(
                f"wandb.Histogram accepts at most {WANDB_BUCKET_CAP} buckets; "
                f"the requested descriptor renders {buckets} per signed "
                "sketch. Refusing at attach beats a latched sink after step 0.",
                code="tracker_histogram_bucket_cap",
                sink="WandbSink",
                buckets=buckets,
                cap=WANDB_BUCKET_CAP,
                remedy=(
                    "Watch with descriptor=torchlens.trackers.WANDB_SAFE_"
                    "DESCRIPTOR (497 buckets), or route histograms to "
                    "TensorBoardSink/JSONLSink."
                ),
            )

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Log one scalar at the caller's step (never wandb's own counter)."""

        self._run.log({point.tag: point.value}, step=point.step)

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Log exact counts/edges as ``wandb.Histogram(np_histogram=...)``."""

        if len(point.counts) > WANDB_BUCKET_CAP:
            raise TrackersError(
                f"wandb.Histogram accepts at most {WANDB_BUCKET_CAP} buckets; "
                f"this sketch renders {len(point.counts)}. wandb would "
                "silently re-bin above the cap (measured), so the sink "
                "refuses instead.",
                code="tracker_histogram_bucket_cap",
                sink="WandbSink",
                buckets=len(point.counts),
                cap=WANDB_BUCKET_CAP,
                remedy=(
                    "Watch with descriptor=torchlens.trackers.WANDB_SAFE_"
                    "DESCRIPTOR (497 buckets), or route histograms to "
                    "TensorBoardSink/JSONLSink."
                ),
            )
        wandb = _wandb_module()
        histogram = wandb.Histogram(
            np_histogram=(list(point.counts), list(point.edges)),
        )
        self._run.log({point.tag: histogram}, step=point.step)

    def emit_text(self, point: TextPoint) -> None:
        """Log one text payload (manifests land in the run's media table)."""

        wandb = _wandb_module()
        self._run.log({point.tag: wandb.Html(f"<pre>{point.text}</pre>")}, step=point.step)

    def flush(self) -> None:
        """wandb buffers internally; nothing to force here."""

    def close(self) -> None:
        """Never finishes the user's run (they own its lifecycle); idempotent."""

        self._closed = True


__all__ = ["WANDB_BUCKET_CAP", "WandbSink"]
