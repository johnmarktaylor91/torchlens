"""HF Trainer + Lightning callbacks over the same engine (memo 3.16, item 10).

Both are EXPLICIT wrappers the user constructs in their own source -- that
construction is the enablement act; no environment variable ever activates
instrumentation (the TORCHLENS_AUTO refusal is the governing precedent, and
FORK #2's env-CONFIG question is a maintainer's -- neither branch is implemented
until it rules, so these callbacks read no env var except the off-only kill
switch, which the engine itself honors).

Both derive cadence from the framework's own logging cadence (HF's
``max(100, logging_steps)`` pattern, credited) or run length when visible
(``clamp(max_steps // 100, 1, 500)`` targets ~100 points), take the
framework's global step as the step source, and never invent a counter.

The shipped ``torchlens.callbacks.lightning.LayerProfilerCallback`` is NOT a
foundation for this engine (it re-traces under eval()/no_grad() -- the wrong
mode for dropout/BN -- cannot see training gradients, and pays an extra
forward); it stays semantically separate. Migration note:
``docs/reference/trackers.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._errors import WatchConfigError
from ._watch import DEFAULT_EVERY, WatchSession, watch

__tl_layer__ = "L8"


def derived_cadence(
    *,
    logging_steps: int | None = None,
    max_steps: int | None = None,
) -> int:
    """Derive a watch cadence from the framework's own signals (memo 3.8).

    A framework that logs every N steps gets tracker samples at the same
    rhythm (never sparser than the panel the user already reads); a visible
    run length targets ~100 points. Neither signal -> the package default.
    """

    if logging_steps is not None and logging_steps > 0:
        return max(1, int(logging_steps))
    if max_steps is not None and max_steps > 0:
        return max(1, min(500, int(max_steps) // 100 or 1))
    return DEFAULT_EVERY


class HFTrainerWatchCallback:
    """HuggingFace ``TrainerCallback`` driving one watch session.

    Construct it yourself and pass it to ``Trainer(callbacks=[...])``; the
    Trainer's own ``state.global_step`` is the step source and its
    ``logging_steps`` derives the cadence. Requires the transformers
    package at construction (typed refusal otherwise).
    """

    def __init__(
        self,
        *,
        to: Any,
        signals: Iterable[str] = ("gradients",),
        select: Iterable[str] | None = None,
        every: int | None = None,
        **watch_kwargs: Any,
    ) -> None:
        """Store the watch configuration; attachment happens at train begin."""

        try:
            from transformers import TrainerCallback
        except ImportError as exc:
            raise WatchConfigError(
                f"HFTrainerWatchCallback needs the transformers package: {exc}.",
                code="tracker_sink_unavailable",
                sink="HFTrainerWatchCallback",
                remedy="pip install transformers.",
            ) from exc
        # Single-inheritance-at-runtime: subclassing lazily keeps the
        # transformers import out of module import time (extras gating law).
        self._base = TrainerCallback
        self.to = to
        self.signals = tuple(signals)
        self.select = tuple(select) if select is not None else None
        self.every = every
        self.watch_kwargs = watch_kwargs
        self.session: WatchSession | None = None

    # transformers duck-types callbacks through named methods; explicit
    # subclassing is unnecessary for dispatch, so the lazy-import pattern
    # above stays honest.

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Attach the watch when the Trainer hands over model + optimizer."""

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None:
            return
        every = self.every
        if every is None:
            every = derived_cadence(
                logging_steps=getattr(args, "logging_steps", None),
                max_steps=getattr(args, "max_steps", None) or None,
            )
        self.session = watch(
            model,
            to=self.to,
            signals=self.signals,
            select=self.select,
            optimizer=optimizer,
            # The Trainer increments ``state.global_step`` AFTER
            # ``optimizer.step()`` returns and logs at the incremented value,
            # so the value visible inside the optimizer boundary is one behind
            # HF's own log axis; ``+ 1`` lands every row on the step HF's
            # panels show for the same update (AUD-CODE 2.14).
            step=lambda: int(state.global_step) + 1,
            every=every,
            **self.watch_kwargs,
        )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Close the session; the close report rides the session object."""

        del args, state, control, kwargs
        if self.session is not None:
            self.session.close()
            self.session = None


class LightningWatchCallback:
    """Lightning ``Callback`` driving one watch session.

    ``trainer.global_step`` is the step source; ``log_every_n_steps``
    derives the cadence. Requires lightning at construction.
    """

    def __init__(
        self,
        *,
        to: Any,
        signals: Iterable[str] = ("gradients",),
        select: Iterable[str] | None = None,
        every: int | None = None,
        **watch_kwargs: Any,
    ) -> None:
        """Store the watch configuration; attachment happens at fit start."""

        try:
            import lightning  # noqa: F401
        except ImportError:
            try:
                import pytorch_lightning  # noqa: F401
            except ImportError as exc:
                raise WatchConfigError(
                    f"LightningWatchCallback needs lightning: {exc}.",
                    code="tracker_sink_unavailable",
                    sink="LightningWatchCallback",
                    remedy="pip install lightning.",
                ) from exc
        self.to = to
        self.signals = tuple(signals)
        self.select = tuple(select) if select is not None else None
        self.every = every
        self.watch_kwargs = watch_kwargs
        self.session: WatchSession | None = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Attach the watch against the LightningModule and its optimizer."""

        optimizers = getattr(trainer, "optimizers", None) or []
        optimizer = optimizers[0] if optimizers else None
        every = self.every
        if every is None:
            every = derived_cadence(
                logging_steps=getattr(trainer, "log_every_n_steps", None),
                max_steps=getattr(trainer, "max_steps", None) or None,
            )
        self.session = watch(
            pl_module,
            to=self.to,
            signals=self.signals,
            select=self.select,
            optimizer=optimizer,
            step=lambda: int(trainer.global_step),
            every=every,
            **self.watch_kwargs,
        )

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """Close the session on fit end."""

        del trainer, pl_module
        if self.session is not None:
            self.session.close()
            self.session = None


__all__ = [
    "HFTrainerWatchCallback",
    "LightningWatchCallback",
    "derived_cadence",
]
