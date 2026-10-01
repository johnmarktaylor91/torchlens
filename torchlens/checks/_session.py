"""The loop-session chassis + standalone check registry (checks memo D1/4.2).

ONE internal loop session owns the S-A hook family (per-parameter
``register_post_accumulate_grad_hook`` counting hooks), the S-B/S-C
optimizer step hooks, local backward / attempted / accepted step ids, the
scale/skip/clip ledgers, exact snapshots, the batched scan kernel, and one
bounded immutable event stream (the C06 ``EventStream``). The registry is
standalone and CAPTURE-FREE: it constructs and runs with TorchLens capture
entirely off, and it runs on ``torch.compile``'d models (the step-check
family uses only public optimizer/tensor hooks -- the deliberate Recorder
guards that refuse checks-only construction and compiled models are exactly
why the mount fork dissolved 3-0 for standalone, memo section 8).

Contracts, all tested (memo 4.2): every hook handle removed on every exit
path; all check arithmetic under ``torch.no_grad()`` + ``pause_logging()``;
zero user-RNG consumption; a check that cannot run emits ``unavailable``
with a reason, never silence; tied tensors dedup by identity with every
alias preserved; no raise ever fires mid-mutation.

S-A mechanism note (memo DR-1): the shipped tier is N counting hooks; the
armed magnitude pass snapshots each parameter's pre-clip gradient norm as a
0-dim device tensor AT FIRE TIME and performs ONE host sync per backward at
the close boundary. This keeps the single-sync contract and stays correct
when the expected parameter set never closes (partial backwards flush at
the S-B boundary with disclosed coverage); re-batching through a foreach
reduction is a C-CHECK-gated optimization, not a semantic change.
``mode="all"`` (``register_multi_grad_hook``) remains the priced fallback
behind ``HAS_MULTI_GRAD_HOOK``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .. import _state
from ..errors._base import TorchLensWarning
from ..observability._chassis import EventStream, ObserverEvent
from ._adapters import OptimizerFacts, optimizer_facts
from ._constants import WATCHDOG_FACTOR
from ._errors import CheckConfigError, CheckLifecycleError, CheckViolationError
from ._ledgers import ClipLedger, ScaleLedger, Watchdog
from ._param_checks import (
    FRONTIER_FOLLOW_UP,
    ChangeCheck,
    FrozenCheck,
    MagnitudeCheck,
    NonfiniteGradientCheck,
    NonfiniteParamCheck,
    UpdateRatioCheck,
    decay_only_movement_fact,
    no_gradient_finding,
    validate_action,
    validate_window,
)
from ._records import CheckFinding, CheckReport, severity_sorted
from ._scan import scan_named_tensors, tensor_digest

__tl_layer__ = "L5"

#: Capability flag for the ``mode="all"`` fallback mechanism (memo D9).
#: The fallback on older torch must be LOUD: absence is visible here and in
#: the attach-time capability disclosures.
HAS_MULTI_GRAD_HOOK = hasattr(torch.autograd.graph, "register_multi_grad_hook")

#: Hard bound on stored findings; crossing it increments a disclosed
#: truncation counter rather than growing without bound.
MAX_STORED_FINDINGS = 10_000

#: How many names a per-step aggregate fact lists before eliding.
_NAME_SAMPLE = 20


def _to_float64_scalar(value: torch.Tensor) -> torch.Tensor:
    """Cast one 0-dim stat to float64 (exact for counts and extrema)."""

    return value.to(torch.float64)


def _synced_scalars(stats: list[tuple[str, torch.Tensor]]) -> dict[str, float]:
    """Sync named 0-dim device tensors with ONE host sync per device."""

    by_device: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in stats:
        by_device.setdefault(str(tensor.device), []).append((name, tensor))
    out: dict[str, float] = {}
    for group in by_device.values():
        stacked = torch.stack([_to_float64_scalar(tensor) for _, tensor in group])
        values = stacked.cpu().tolist()  # the ONE sync for this device group
        for (name, _), value in zip(group, values, strict=True):
            out[name] = value
    return out


@dataclass(frozen=True)
class _RatioObservation:
    """One parameter's update-ratio facts at an accepted step (memo D13)."""

    ratio: float
    lr: float | None
    lr_normalized: float | None
    delta_norm: float
    zero_baseline: bool
    out_of_band: bool


class _StepBoundary:
    """The explicit per-attempt boundary (memo D18, boundary mode).

    One ``with`` line per optimizer attempt: the canonical exact mode and
    the ONLY source of caller ``global_step`` and exact accumulation
    grouping. Hook-only evidence stays ``inferred``.
    """

    def __init__(
        self,
        session: ChecksSession,
        global_step: int | None,
        micro_batches: int | None,
    ) -> None:
        self._session = session
        self.global_step = global_step
        self.micro_batches = micro_batches
        self._accepted_at_entry = 0
        self._backwards_at_entry = 0

    def __enter__(self) -> _StepBoundary:
        if self._session._boundary is not None:
            raise CheckLifecycleError(
                "step() boundaries do not nest: one boundary is one optimizer attempt.",
                code="check_boundary_nested",
                remedy="Close the open step() context before opening another.",
            )
        self._session._boundary = self
        self._accepted_at_entry = self._session.accepted_step_id
        self._backwards_at_entry = self._session.backward_id
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        session = self._session
        session._boundary = None
        accepted = session.accepted_step_id > self._accepted_at_entry
        backwards = session.backward_id - self._backwards_at_entry
        session._explicit_attempts += 1
        if accepted:
            session._explicit_accepted += 1
        elif backwards > 0:
            session._explicit_skipped += 1
        if self.micro_batches is not None and backwards not in (0, self.micro_batches):
            session._disclose(
                "boundary_micro_batches",
                f"a step() boundary declared micro_batches={self.micro_batches} "
                f"but observed {backwards} backwards; declared grouping wins "
                "for attempt attribution and the mismatch is disclosed",
            )


class ChecksSession:
    """Standalone, capture-free training-check registry on one chassis.

    Parameters
    ----------
    model:
        ``nn.Module`` or name->parameter Mapping supplying the watched
        inventory. Tied tensors dedup by identity, aliases preserved.
    optimizer:
        One optimizer or a sequence. Step-family checks refuse at attach
        without one; overlapping param ownership across several refuses
        typed (explicit ownership required, memo 4.2).
    scaler:
        The loop's ``GradScaler`` handle, or ``None`` as the DECLARATION
        that this loop runs no scaler (first-class disclosure, memo D10).
    clip_norm:
        Declared ``clip_grad_norm_`` max_norm (clip tier b, memo D12).
    accumulation_steps:
        Declared static accumulation factor K (watchdog threshold + exact
        boundary grouping).
    event_stream:
        Optional external :class:`EventStream`; by default the session owns
        one (the trackers seam subscribes to it, memo 4.7).

    Construction is SILENT and does no device work (memo D20): no hooks, no
    syncs, no stdout. ``estimated_snapshot_bytes`` is pure metadata;
    ``profile()`` runs the measured scan on demand.
    """

    def __init__(  # noqa: PLR0913 -- the memo-4.2 chassis constructor: model + optimizer + the four declared loop facts (scaler/clip_norm/accumulation/event_stream); bundling them would hide the disclosure contract
        self,
        model: Any,
        optimizer: Any | None = None,
        *,
        scaler: Any | None = None,
        clip_norm: float | None = None,
        accumulation_steps: int | None = None,
        event_stream: EventStream | None = None,
    ) -> None:
        self._entries = self._named_params(model)
        self._model = model if isinstance(model, torch.nn.Module) else None
        self._optimizers: list[torch.optim.Optimizer] = (
            []
            if optimizer is None
            else list(optimizer)
            if isinstance(optimizer, (list, tuple))
            else [optimizer]
        )
        self._scaler = scaler
        self._clip_norm = float(clip_norm) if clip_norm is not None else None
        self._accumulation_steps = accumulation_steps
        self.events = event_stream if event_stream is not None else EventStream()

        # Registrations (explicit; a bare attach() is the free counting tier).
        self._change: ChangeCheck | None = None
        self._frozen: FrozenCheck | None = None
        self._ratio: UpdateRatioCheck | None = None
        self._nonfinite_grad: NonfiniteGradientCheck | None = None
        self._nonfinite_param: NonfiniteParamCheck | None = None
        self._magnitude: MagnitudeCheck | None = None

        # Chassis state (the ONE owner of the step axis).
        self.backward_id = 0
        self.accepted_step_id = 0
        self._fired_this_backward: set[str] = set()
        self._fire_counts: dict[str, int] = {}
        self._pending_norms: list[tuple[str, torch.Tensor]] = []
        self._pending_scale: float | None = None
        self._last_pre_clip_total: float | None = None
        self._last_ratios: list[float] = []
        self._prestep_none: tuple[str, ...] = ()
        self._handles: list[Any] = []
        self._attached = False
        self._paused = False
        self._boundary: _StepBoundary | None = None
        self._explicit_attempts = 0
        self._explicit_accepted = 0
        self._explicit_skipped = 0
        self._snapshots: dict[str, torch.Tensor] = {}
        self._prestep_stats: dict[str, float] = {}
        # Cached at attach: the S-A expected set (requires_grad flips after
        # attach are out of contract; re-attach to re-inventory).
        self._watched_cache: list[tuple[str, torch.Tensor]] = []
        self._lookup_cache: dict[str, torch.Tensor] = {}
        self._scale_ledger = ScaleLedger(scaler)
        self._watchdog = Watchdog(accumulation_steps, WATCHDOG_FACTOR)
        self._clip_ledger = ClipLedger(self._clip_norm)
        self._findings: list[CheckFinding] = []
        self._truncated_findings = 0
        self._unavailable: dict[str, str] = {}
        self._disclosures: dict[str, Any] = {}
        self._checks_run: list[str] = []
        self._final_report: CheckReport | None = None
        self._magnitude_censorship_noted = False

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    @staticmethod
    def _named_params(model: Any) -> list[tuple[str, tuple[str, ...], torch.Tensor]]:
        """Normalize the watched inventory, deduping tied tensors."""

        if isinstance(model, torch.nn.Module):
            # remove_duplicate=False so tied tensors surface EVERY alias
            # (memo 4.2: dedup by identity, no alias lost).
            raw: list[tuple[str, torch.Tensor]] = list(
                model.named_parameters(remove_duplicate=False)
            )
        elif isinstance(model, Mapping):
            raw = [(str(name), tensor) for name, tensor in model.items()]
        else:
            raise CheckConfigError(
                f"model must be an nn.Module or a name->parameter Mapping, got "
                f"{type(model).__name__}.",
                code="check_target_invalid",
                remedy="Pass the module, or dict(model.named_parameters()).",
            )
        seen: dict[int, int] = {}
        deduped: list[tuple[str, list[str], torch.Tensor]] = []
        for name, tensor in raw:
            key = id(tensor)
            if key in seen:
                deduped[seen[key]][1].append(name)
                continue
            seen[key] = len(deduped)
            deduped.append((name, [], tensor))
        return [(name, tuple(aliases), tensor) for name, aliases, tensor in deduped]

    def _watched(self) -> list[tuple[str, torch.Tensor]]:
        """Return the requires_grad inventory (the S-A expected set)."""

        if self._watched_cache:
            return self._watched_cache
        return [(name, tensor) for name, aliases, tensor in self._entries if tensor.requires_grad]

    def _name_lookup(self) -> dict[str, torch.Tensor]:
        """Return the canonical-and-alias name -> tensor lookup (cached)."""

        if not self._lookup_cache:
            lookup: dict[str, torch.Tensor] = {}
            for canonical, aliases, tensor in self._entries:
                lookup[canonical] = tensor
                for alias in aliases:
                    lookup[alias] = tensor
            self._lookup_cache = lookup
        return self._lookup_cache

    def _known_names(self) -> set[str]:
        """Return every canonical name and alias in the inventory."""

        names: set[str] = set()
        for name, aliases, _ in self._entries:
            names.add(name)
            names.update(aliases)
        return names

    def _validate_within(self, kwarg: str, names: Any | None) -> tuple[str, ...] | None:
        """Validate exact qualified names against the inventory."""

        if names is None:
            return None
        requested = [str(name) for name in names]
        unknown = sorted(set(requested) - self._known_names())
        if unknown:
            raise CheckConfigError(
                f"{kwarg} names not present in the session inventory: {unknown}. "
                "A silently ignored name is the silent-no-op defect class this "
                "kit exists to kill.",
                code="check_within_unknown_name",
                unknown_names=unknown,
                remedy="Use exact qualified names from named_parameters().",
            )
        return tuple(requested)

    # ------------------------------------------------------------------
    # Registration (silent; validation only)
    # ------------------------------------------------------------------

    def register_change_check(
        self,
        *,
        within: Any | None = None,
        window: tuple[int, int] | None = None,
        action: str = "warn",
    ) -> None:
        """Register the params-change fact vocabulary (memo D2/D3).

        The primary detector is the GRADIENT FACT (``grad is None`` /
        ``grad_norm == 0``) on a sustained M-of-N window; exact cross-step
        change facts and the decay-band movement statistic are recorded as
        corroborating facts. No finding says "learning" or "dead".
        """

        self._change = ChangeCheck(
            within=self._validate_within("within", within),
            window=validate_window(window),
            action=validate_action(action),
        )

    def register_frozen_check(
        self,
        names: Any,
        *,
        evidence: str = "clone",
        action: str = "raise",
    ) -> None:
        """Register the declared-frozen invariant (memo D5).

        ``evidence="clone"`` (default) keeps exact detached dtype-preserving
        snapshots and can SHOW the offending delta; ``evidence="digest"`` is
        the labeled opt-in (content digests; a differing digest PROVES
        change, equality is probabilistic and may never emit an exact pass).
        """

        if evidence not in ("clone", "digest"):
            raise CheckConfigError(
                f"frozen evidence={evidence!r} must be 'clone' or 'digest'.",
                code="check_vocab_invalid",
                field="evidence",
                remedy="Pass evidence='clone' (exact) or evidence='digest' (labeled opt-in).",
            )
        validated = self._validate_within("names", names)
        if not validated:
            raise CheckConfigError(
                "frozen check needs at least one declared-frozen name.",
                code="check_within_unknown_name",
                remedy="Pass the qualified names of the tensors declared frozen.",
            )
        self._frozen = FrozenCheck(
            names=validated,
            evidence=evidence,
            action=validate_action(action),
        )

    def register_update_ratio_check(
        self,
        *,
        within: Any | None = None,
        bounds: tuple[float | None, float | None] | None = None,
        action: str = "collect",
    ) -> None:
        """Register the update-to-weight ratio (memo D13).

        Reports the raw Karpathy ratio AND the lr-normalized companion. NO
        default band ships (a healthy model measured a 38x spread across
        parameters); ``bounds`` are explicit per registration. A zero
        baseline yields ``inf`` plus ``zero_baseline=True`` (0.0 when the
        delta is also zero), never an epsilon-clamped fake.
        """

        self._ratio = UpdateRatioCheck(
            within=self._validate_within("within", within),
            bounds=bounds,
            action=validate_action(action),
        )

    def register_nonfinite_gradient_check(self, *, action: str = "raise") -> None:
        """Register the live parameter-gradient nonfinite check (memo D6).

        Site-conditional by construction: at S-A nonfinite gradients are
        COLLECTED into the skip ledger (under a GradScaler they are the
        mechanism working); at S-B the check RAISES before the weights are
        written -- structurally unreachable under a scaler (the skip happens
        first, measured 0/3), and the only pre-write tripwire under fp32 and
        bf16, where torch's own ``error_if_nonfinite`` defaults OFF.
        """

        self._nonfinite_grad = NonfiniteGradientCheck(action=validate_action(action))

    def register_nonfinite_param_check(
        self,
        *,
        every: int = 1,
        include_buffers: bool = True,
        action: str = "raise",
    ) -> None:
        """Register the scheduled parameter/buffer nonfinite scan (memo D7).

        Runs the shared scan kernel every ``every`` accepted steps at the
        safe post-mutation boundary (S-C); corrupt weights make every later
        step wasted compute, so the default action raises.
        """

        if int(every) < 1:
            raise CheckConfigError(
                f"every={every!r} must be >= 1 accepted steps.",
                code="check_window_invalid",
                remedy="Pass every=1 (each accepted step) or a larger cadence.",
            )
        self._nonfinite_param = NonfiniteParamCheck(
            every=int(every),
            include_buffers=include_buffers,
            action=validate_action(action),
        )

    def register_magnitude_check(
        self,
        *,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e4,
        action: str = "collect",
    ) -> None:
        """Arm the S-A pre-clip magnitude pass (memo D8/D9).

        Magnitude evidence lives ONLY at the pre-clip S-A site: the
        optimizer step site is post-clip, where clipping pins the total to
        ``max_norm`` (measured 6/6 on a real fine-tune) and an absolute
        magnitude check is structurally incapable of firing. Registration
        is the explicit arming -- the pass is OFF by default until the
        canonical gate set (idle-box + CUDA + multi-rank DDP + at-scale
        compile) passes.
        """

        if not 0 < vanishing_threshold < exploding_threshold:
            raise CheckConfigError(
                f"need 0 < vanishing_threshold ({vanishing_threshold!r}) < "
                f"exploding_threshold ({exploding_threshold!r}).",
                code="check_window_invalid",
                remedy="Pass positive thresholds with vanishing < exploding.",
            )
        self._magnitude = MagnitudeCheck(
            vanishing_threshold=float(vanishing_threshold),
            exploding_threshold=float(exploding_threshold),
            action=validate_action(action),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def estimated_snapshot_bytes(self) -> int:
        """Clone-tier snapshot budget as pure numel x dtype-size metadata."""

        lookup = self._name_lookup()
        return sum(
            lookup[name].numel() * lookup[name].element_size() for name in self._snapshot_names()
        )

    def _snapshot_names(self) -> tuple[str, ...]:
        """Canonical names needing pre-step clones (change/frozen/ratio)."""

        canonical = {name: name for name, _, _ in self._entries}
        for name, aliases, _ in self._entries:
            for alias in aliases:
                canonical[alias] = name
        names: set[str] = set()
        watched = {name for name, _ in self._watched()}
        if self._change is not None:
            names.update(self._change.within or watched)
        if self._ratio is not None:
            names.update(self._ratio.within or watched)
        if self._frozen is not None and self._frozen.evidence == "clone":
            names.update(self._frozen.names)
        return tuple(sorted({canonical[name] for name in names}))

    def attach(self) -> ChecksSession:
        """Install the S-A / S-B / S-C hooks (the first non-silent moment).

        Raises
        ------
        CheckLifecycleError
            On double attach (``check_already_attached``).
        CheckConfigError
            When a step-family check is registered without an optimizer
            (``check_optimizer_required``) or the same parameter is owned
            by several optimizers (``check_optimizer_overlap``).
        """

        if self._attached:
            raise CheckLifecycleError(
                "session is already attached.",
                code="check_already_attached",
                remedy="Detach first, or use one session per training loop.",
            )
        step_family = any(
            check is not None
            for check in (
                self._change,
                self._frozen,
                self._ratio,
                self._nonfinite_grad,
                self._nonfinite_param,
            )
        )
        if step_family and not self._optimizers:
            raise CheckConfigError(
                "step-family checks (change / frozen / ratio / nonfinite) need "
                "the optimizer boundary; without optimizer= their accepted-step "
                "truth does not exist.",
                code="check_optimizer_required",
                remedy="Pass optimizer= at construction, or register only the S-A tier.",
            )
        self._validate_ownership()
        self._collect_disclosures()
        try:
            watched: list[tuple[str, torch.Tensor]] = []
            for name, param in self._watched():
                try:
                    handle = param.register_post_accumulate_grad_hook(
                        self._make_grad_fire_hook(name)
                    )
                except RuntimeError as exc:
                    # A non-leaf or otherwise unhookable tensor never fires;
                    # disclosed, never silent (memo 4.2).
                    self._unavailable[f"counting_hook::{name}"] = str(exc)
                    continue
                self._handles.append(handle)
                watched.append((name, param))
            self._watched_cache = watched
            for optimizer in self._optimizers:
                self._handles.append(optimizer.register_step_pre_hook(self._step_pre_hook))
                self._handles.append(optimizer.register_step_post_hook(self._step_post_hook))
            if self._frozen is not None:
                self._baseline_frozen()
        except Exception:
            self._remove_handles()
            raise
        self._attached = True
        return self

    def detach(self) -> None:
        """Remove every hook handle; idempotent on every exit path."""

        self._remove_handles()
        self._attached = False

    def _remove_handles(self) -> None:
        """Remove all installed handles, tolerating partial installs."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ChecksSession:
        if not self._attached:
            self.attach()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        reason = "exception_exit" if exc_type is not None else "scope_exit"
        self.detach()
        self._final_report = self.report(finalized_reason=reason)

    def paused(self) -> _PausedScope:
        """Context manager suspending all checks (torcheck's one good ergonomic)."""

        return _PausedScope(self)

    def disable(self) -> None:
        """Suspend all checks until :meth:`enable`."""

        self._paused = True

    def enable(self) -> None:
        """Re-enable checks after :meth:`disable`."""

        self._paused = False

    def step(
        self, *, global_step: int | None = None, micro_batches: int | None = None
    ) -> _StepBoundary:
        """Open the explicit per-attempt boundary (memo D18 boundary mode)."""

        return _StepBoundary(self, global_step, micro_batches)

    # ------------------------------------------------------------------
    # Attach-time capability + disclosures (memo D4: disclosures, not checks)
    # ------------------------------------------------------------------

    def _validate_ownership(self) -> None:
        """Refuse typed when one parameter is owned by several optimizers."""

        if len(self._optimizers) < 2:
            return
        owners: dict[int, str] = {}
        names_by_id = {id(tensor): name for name, _aliases, tensor in self._entries}
        overlap: set[str] = set()
        for index, optimizer in enumerate(self._optimizers):
            for group in optimizer.param_groups:
                for param in group.get("params", ()):
                    key = id(param)
                    if key in owners and owners[key] != f"optimizer[{index}]":
                        overlap.add(names_by_id.get(key, f"id={key}"))
                    owners[key] = f"optimizer[{index}]"
        if overlap:
            raise CheckConfigError(
                f"parameters owned by more than one optimizer: "
                f"{sorted(overlap)[:8]}. Multiple optimizers require explicit "
                "ownership on overlap (memo 4.2).",
                code="check_optimizer_overlap",
                remedy="Give each parameter exactly one optimizer, or run one session per optimizer.",
            )

    def _collect_disclosures(self) -> None:
        """Record membership / train-eval / precondition disclosures (D4).

        Disclosures carry NO severity and NO action: they are the coverage
        plumbing the parameter checks depend on -- ``requires_grad=False``
        parameters have ``grad is None`` forever, so without membership the
        no-gradient detector would name every frozen parameter.
        """

        optimizer_param_ids: set[int] = set()
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                optimizer_param_ids.update(id(param) for param in group.get("params", ()))
        frozen_names = [
            name for name, _aliases, tensor in self._entries if not tensor.requires_grad
        ]
        unowned = [
            name
            for name, _aliases, tensor in self._entries
            if tensor.requires_grad and self._optimizers and id(tensor) not in optimizer_param_ids
        ]
        self._disclosures["membership"] = {
            "n_params": len(self._entries),
            "n_requires_grad": len(self._watched()),
            "requires_grad_false": frozen_names[:_NAME_SAMPLE],
            "n_requires_grad_false": len(frozen_names),
            "in_no_optimizer": unowned[:_NAME_SAMPLE],
            "n_in_no_optimizer": len(unowned),
        }
        if self._model is not None:
            eval_modules = [
                name for name, module in self._model.named_modules() if not module.training
            ]
            self._disclosures["train_eval"] = {
                "model_training": self._model.training,
                "n_eval_mode_modules": len(eval_modules),
                "eval_mode_modules": eval_modules[:_NAME_SAMPLE],
            }
        self._disclosures["optimizers"] = [
            {
                "kind": facts.kind,
                "known_adapter": facts.known,
                "decoupled_decay": facts.decoupled_decay,
                "band_reason": facts.band_reason,
            }
            for facts in (optimizer_facts(optimizer) for optimizer in self._optimizers)
        ]
        self._disclosures["scaler_declared"] = self._scaler is not None
        self._disclosures["clip_norm_declared"] = self._clip_norm
        self._disclosures["accumulation_steps_declared"] = self._accumulation_steps
        self._disclosures["has_multi_grad_hook"] = HAS_MULTI_GRAD_HOOK
        self._disclosures["estimated_snapshot_bytes"] = self.estimated_snapshot_bytes

    def _baseline_frozen(self) -> None:
        """Take the frozen baselines (clone or digest) at attach time."""

        if self._frozen is None:
            return
        lookup = self._name_lookup()
        with torch.no_grad(), _state.pause_logging():
            for name in self._frozen.names:
                tensor = lookup[name]
                if self._frozen.evidence == "digest":
                    self._frozen.baselines[name] = tensor_digest(tensor)
                else:
                    self._frozen.baselines[name] = tensor.detach().clone()

    # ------------------------------------------------------------------
    # S-A: the counting tier + armed magnitude pass
    # ------------------------------------------------------------------

    def _make_grad_fire_hook(self, name: str) -> Any:
        """Build one parameter's post-accumulate counting hook."""

        def _fire(param: torch.Tensor) -> None:
            """Route one post-accumulate fire to the session unless paused."""

            if self._paused:
                return
            self._on_fire(name, param)

        return _fire

    def _on_fire(self, name: str, param: torch.Tensor) -> None:
        """Handle one S-A fire: counting, ledger read, watchdog, magnitude."""

        if name in self._fired_this_backward:
            self._close_backward()
        if not self._fired_this_backward:
            self._scale_ledger.read()
            scaler = self._scaler
            self._pending_scale = (
                float(scaler.get_scale())
                if self._scale_ledger.enabled and scaler is not None
                else None
            )
            if self._watchdog.on_backward():
                self._warn_watchdog()
        self._fired_this_backward.add(name)
        self._fire_counts[name] = self._fire_counts.get(name, 0) + 1
        if self._magnitude is not None and param.grad is not None:
            with torch.no_grad(), _state.pause_logging():
                self._pending_norms.append((name, torch.linalg.vector_norm(param.grad.detach())))
        if len(self._fired_this_backward) == len(self._watched()):
            self._close_backward()

    def _close_backward(self) -> None:
        """Close one local backward: sync the armed magnitude pass, reset."""

        if not self._fired_this_backward:
            return
        self.backward_id += 1
        coverage = f"{len(self._fired_this_backward)}/{len(self._watched())} params fired"
        if self._magnitude is not None and self._pending_norms:
            self._flush_magnitude(coverage)
        self._fired_this_backward.clear()
        self._pending_norms = []
        self._pending_scale = None

    def _flush_magnitude(self, coverage: str) -> None:
        """Run the armed magnitude pass: ONE sync, pre-clip verdicts (D8)."""

        if self._magnitude is None:
            return
        synced = _synced_scalars(self._pending_norms)
        scale = self._pending_scale
        if self._scaler is not None:
            stage, provenance = "pre_clip_scaled", "gradscaler"
        else:
            # scaler=None at construction is the user's declaration that this
            # loop runs no GradScaler; disclosed, never guessed (memo D10).
            stage, provenance = "pre_clip_unscaled", "unscaled"
        for name, norm in synced.items():
            unscaled = norm / scale if scale else norm
            verdict: str | None = None
            if not math.isfinite(unscaled):
                verdict = "nonfinite"
            elif unscaled > self._magnitude.exploding_threshold:
                verdict = "exploding"
            elif 0.0 < unscaled < self._magnitude.vanishing_threshold:
                verdict = "vanishing"
            if verdict is None:
                continue
            if verdict == "nonfinite" and self._scale_ledger.enabled:
                # The mechanism working (memo D6): collected into the ledger,
                # never raised, never a finding flood.
                continue
            self._record(
                CheckFinding(
                    check="grad_magnitude",
                    code="grad_magnitude_flagged",
                    severity="warning",
                    action=self._magnitude.action,
                    message=(
                        f"{name} pre-clip gradient norm {unscaled:.6g} flagged "
                        f"{verdict} at backward {self.backward_id} "
                        f"(thresholds: vanishing {self._magnitude.vanishing_threshold:g}, "
                        f"exploding {self._magnitude.exploding_threshold:g})."
                    ),
                    names=(name,),
                    backward_id=self.backward_id,
                    step_provenance=self._provenance(),
                    global_step=self._global_step(),
                    stage=stage,
                    scale_provenance=provenance,
                    evidence="magnitude_pass",
                    coverage=coverage,
                    values={"grad_norm": unscaled, "grad_norm_as_captured": synced[name]},
                    remedy="Trace the module with backward_ready=True and read the gradient frontier",
                    follow_up=FRONTIER_FOLLOW_UP,
                )
            )
        self._last_pre_clip_total = math.sqrt(
            sum((value / scale if scale else value) ** 2 for value in synced.values())
        )

    def _warn_watchdog(self) -> None:
        """Emit the death-spiral watchdog warning (memo D11)."""

        snapshot = self._watchdog.snapshot()
        finding = CheckFinding(
            check="scale_collapse_watchdog",
            code="check_no_accepted_steps",
            severity="warning",
            action="warn",
            message=(
                f"{snapshot.backwards_since_accepted} backwards since the last "
                f"accepted optimizer step (threshold {snapshot.threshold}). "
                "Under a GradScaler this is the scale-collapse signature: "
                "every attempt is being skipped, and every retrospective "
                "method stays silent forever."
            ),
            backward_id=self.backward_id,
            step_provenance=self._provenance(),
            global_step=self._global_step(),
            evidence="counting_hook",
            values={"backwards_since_accepted": float(snapshot.backwards_since_accepted)},
            remedy=(
                "Inspect the skip ledger (report().ledgers['scale']); check the "
                "loss for nonfinites and the scaler's init_scale"
            ),
        )
        self._record(finding)
        warnings.warn(
            TorchLensWarning(
                finding.message + " Remedy: " + finding.remedy,
                code="check_no_accepted_steps",
            ),
            stacklevel=3,
        )
        self._publish_watchdog()

    # ------------------------------------------------------------------
    # S-B: pre-write evidence + the nonfinite raise
    # ------------------------------------------------------------------

    def _step_pre_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """The S-B site: post-clip, pre-write, accepted steps only."""

        del optimizer, args, kwargs
        if self._paused:
            return
        self._close_backward()
        watched = self._watched()
        none_names = [name for name, param in watched if param.grad is None]
        self._prestep_stats = {}
        self._prestep_none = tuple(none_names)
        needs_device_work = any(
            check is not None
            for check in (self._change, self._frozen, self._ratio, self._nonfinite_grad)
        )
        if not needs_device_work:
            return
        stats: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad(), _state.pause_logging():
            for name, param in watched:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                # S-B grads are already unscaled -- NEVER divide by the scale
                # here (standing law, memo 4.3: dividing again under-reports
                # by exactly the loss scale; 17.2% of a real model mislabeled).
                stats.append((f"grad_norm::{name}", torch.linalg.vector_norm(grad)))
                stats.append((f"nonfinite::{name}", (~torch.isfinite(grad)).sum()))
            lookup = self._name_lookup()
            for name in self._snapshot_names():
                current = lookup[name].detach()
                stats.append((f"param_norm::{name}", torch.linalg.vector_norm(current)))
                self._snapshots[name] = current.clone()
        self._prestep_stats = _synced_scalars(stats)
        if self._nonfinite_grad is not None:
            self._raise_on_nonfinite_grads()

    def _raise_on_nonfinite_grads(self) -> None:
        """The D6 raise: pre-write, weights bitwise clean, batch re-runnable."""

        if self._nonfinite_grad is None:
            return
        culprits = sorted(
            key.split("::", 1)[1]
            for key, value in self._prestep_stats.items()
            if key.startswith("nonfinite::") and value > 0
        )
        if not culprits:
            return
        zeroed = sorted(
            key.split("::", 1)[1]
            for key, value in self._prestep_stats.items()
            if key.startswith("grad_norm::") and value == 0.0
        )
        attribution = "exact" if len(culprits) == 1 else "smeared"
        clip_evidence = self._clip_norm is not None
        finding = CheckFinding(
            check="param_grad_nonfinite",
            code="param_grad_nonfinite",
            severity="critical",
            action=self._nonfinite_grad.action,
            message=(
                f"nonfinite gradient(s) reached the optimizer step pre-write "
                f"boundary on {len(culprits)} parameter(s): {culprits[:_NAME_SAMPLE]}. "
                + (
                    f"{len(zeroed)} healthy parameter gradients are zeroed "
                    "(clip_grad_norm_ collateral: a nonfinite total norm zeroes "
                    "every gradient and NaN-poisons the culprit). "
                    if clip_evidence and zeroed
                    else ""
                )
                + (
                    "Attribution is smeared: NaN propagates through backward, so "
                    "no single culprit can be named without pre-clip evidence."
                    if attribution == "smeared"
                    else "Attribution is exact: one parameter carries the nonfinite."
                )
            ),
            names=tuple(culprits),
            backward_id=self.backward_id,
            accepted_step_id=self.accepted_step_id,
            global_step=self._global_step(),
            step_provenance=self._provenance(),
            stage="post_clip_applied",
            scale_provenance="unscaled" if self._scaler is None else "gradscaler",
            evidence="step_hook",
            attribution=attribution,
            collateral_zeroed=len(zeroed) if clip_evidence else None,
            values={"n_nonfinite_params": float(len(culprits)), "n_zeroed": float(len(zeroed))},
            remedy=(
                "The raise fired BEFORE the write: weights are bitwise clean and "
                "the batch is re-runnable. Upstream, "
                "clip_grad_norm_(error_if_nonfinite=True) is torch's own "
                "tripwire (off by default); under fp16 use a GradScaler, which "
                "makes this event a harmless skipped step"
            ),
            follow_up="tl.debug.bisect_nan / tl.debug.find_nan on a captured trace",
        )
        self._record(finding)
        if finding.action == "raise":
            raise CheckViolationError(
                finding.message + " Remedy: " + finding.remedy,
                code="param_grad_nonfinite",
                finding=finding.to_dict(),
                names=list(finding.names),
                accepted_step_id=self.accepted_step_id,
                backward_id=self.backward_id,
                report=self.report().to_dict(),
                remedy=finding.remedy,
            )
        if finding.action == "warn":
            warnings.warn(
                TorchLensWarning(
                    finding.message + " Remedy: " + finding.remedy,
                    code="param_grad_nonfinite",
                ),
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # S-C: post-write facts, ratios, frozen, scheduled scan
    # ------------------------------------------------------------------

    def _step_post_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """The S-C site: after the write -- deltas, ratios, scheduled scans."""

        del args, kwargs
        if self._paused:
            return
        self.accepted_step_id += 1
        self._scale_ledger.accepted()
        self._watchdog.on_accepted()
        # The free disconnection report (site matrix S-B row): p.grad is None
        # at an accepted step is host-side evidence, catching the
        # reentrant-checkpoint detached-input footgun (a silently untrained
        # segment) at zero device cost.
        self._disclosures["grad_none_last_accepted_step"] = list(self._prestep_none[:_NAME_SAMPLE])
        self._commit_no_grad_windows()
        delta_stats = self._delta_stats()
        self._commit_change_facts(delta_stats)
        self._commit_update_ratios(delta_stats, optimizer)
        self._commit_frozen(delta_stats)
        self._commit_clip_ledger()
        if (
            self._nonfinite_param is not None
            and self.accepted_step_id % self._nonfinite_param.every == 0
        ):
            self._scheduled_scan()
        self._publish_step_events()
        self._snapshots = {}
        self._prestep_stats = {}

    def _delta_stats(self) -> dict[str, float]:
        """Batched post-step delta facts for every snapshotted parameter."""

        if not self._snapshots:
            return {}
        lookup = self._name_lookup()
        stats: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad(), _state.pause_logging():
            for name, snapshot in self._snapshots.items():
                current = lookup[name].detach()
                delta = current - snapshot
                stats.append((f"changed::{name}", (current != snapshot).any()))
                stats.append((f"delta_norm::{name}", torch.linalg.vector_norm(delta)))
                stats.append((f"delta_absmax::{name}", delta.abs().max()))
        return _synced_scalars(stats)

    def _commit_no_grad_windows(self) -> None:
        """Update the gradient-fact windows on this accepted step (D2)."""

        if self._change is None:
            return
        watched_names = self._change.within or tuple(name for name, _ in self._watched())
        none_set = set(self._prestep_none)
        for name in watched_names:
            grad_norm = self._prestep_stats.get(f"grad_norm::{name}")
            if name in none_set:
                kind = "grad is None"
                bad = True
            elif grad_norm is not None:
                kind = "grad_norm == 0"
                bad = grad_norm == 0.0
            else:
                continue
            window = self._change.window_for(name)
            if window.observe(bad) and bad:
                finding = no_gradient_finding(
                    name,
                    kind=kind,
                    window=self._change.window,
                    accepted_step_id=self.accepted_step_id,
                    global_step=self._global_step(),
                    step_provenance=self._provenance(),
                    action=self._change.action,
                )
                self._record(finding)
                if self._change.action == "warn":
                    warnings.warn(
                        TorchLensWarning(
                            finding.message + " Remedy: " + finding.remedy,
                            code="param_received_no_gradient",
                        ),
                        stacklevel=4,
                    )

    def _commit_change_facts(self, delta_stats: dict[str, float]) -> None:
        """Record the exact did-it-change aggregate fact (corroborating)."""

        if self._change is None or not delta_stats:
            return
        names = self._change.within or tuple(name for name, _ in self._watched())
        changed = [name for name in names if delta_stats.get(f"changed::{name}", 0.0) > 0]
        unchanged = [
            name for name in names if f"changed::{name}" in delta_stats and name not in set(changed)
        ]
        self._record(
            CheckFinding(
                check="params_changed_fact",
                code="params_changed_fact",
                severity="info",
                action="collect",
                message=(
                    f"accepted step {self.accepted_step_id}: {len(changed)} "
                    f"parameter(s) changed, {len(unchanged)} unchanged"
                    + (f" (unchanged: {unchanged[:_NAME_SAMPLE]})" if unchanged else "")
                    + ". Exact cross-step comparison; under decoupled weight "
                    "decay a dead parameter still changes every step, so this "
                    "fact CORROBORATES and never detects."
                ),
                names=tuple(unchanged[:_NAME_SAMPLE]),
                accepted_step_id=self.accepted_step_id,
                global_step=self._global_step(),
                step_provenance=self._provenance(),
                evidence="clone",
                values={"n_changed": float(len(changed)), "n_unchanged": float(len(unchanged))},
                remedy="Read param_received_no_gradient for the primary detector",
            )
        )
        self._maybe_decay_band_facts(delta_stats)

    def _maybe_decay_band_facts(self, delta_stats: dict[str, float]) -> None:
        """Record decay-band movement facts where an adapter licenses them."""

        if self._change is None or not self._optimizers:
            return
        facts: OptimizerFacts = optimizer_facts(self._optimizers[0])
        if not facts.known:
            self._unavailable.setdefault("param_movement_decay_band", facts.band_reason or "")
            return
        if not facts.decoupled_decay:
            return
        group = facts.param_group_facts[0] if facts.param_group_facts else {}
        lr, weight_decay = group.get("lr"), group.get("weight_decay")
        if not lr or not weight_decay:
            return
        betas = group.get("betas") or (None, None)
        decay_rate = float(lr) * float(weight_decay)
        names = self._change.within or tuple(name for name, _ in self._watched())
        for name in names:
            delta_norm = delta_stats.get(f"delta_norm::{name}")
            pre_norm = self._prestep_stats.get(f"param_norm::{name}")
            if not delta_norm or not pre_norm:
                continue
            ratio = delta_norm / pre_norm
            if 0.5 * decay_rate <= ratio <= 2.0 * decay_rate:
                self._record(
                    decay_only_movement_fact(
                        name,
                        ratio=ratio,
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        beta1=betas[0],
                        beta2=betas[1],
                        accepted_step_id=self.accepted_step_id,
                    )
                )

    def _commit_update_ratios(self, delta_stats: dict[str, float], optimizer: Any) -> None:
        """Record raw + lr-normalized update ratios (memo D13)."""

        if self._ratio is None or not delta_stats:
            return
        facts = optimizer_facts(optimizer)
        group = facts.param_group_facts[0] if facts.param_group_facts else {}
        lr = group.get("lr")
        names = self._ratio.within or tuple(name for name, _ in self._watched())
        ratios: list[float] = []
        for name in names:
            observation = self._observe_ratio(name, delta_stats, lr, self._ratio)
            if observation is None:
                continue
            if math.isfinite(observation.ratio):
                ratios.append(observation.ratio)
            if observation.zero_baseline or observation.out_of_band:
                self._emit_ratio_finding(name, observation, self._ratio)
        self._last_ratios = ratios

    def _observe_ratio(
        self,
        name: str,
        delta_stats: dict[str, float],
        lr: float | None,
        check: UpdateRatioCheck,
    ) -> _RatioObservation | None:
        """Compute one parameter's ratio facts, or None without evidence."""

        delta_norm = delta_stats.get(f"delta_norm::{name}")
        pre_norm = self._prestep_stats.get(f"param_norm::{name}")
        if delta_norm is None or pre_norm is None:
            return None
        zero_baseline = pre_norm == 0.0
        if zero_baseline:
            ratio = 0.0 if delta_norm == 0.0 else float("inf")
        else:
            ratio = delta_norm / pre_norm
        lr_normalized = (ratio / lr) if lr else None
        if lr is not None and lr == 0.0:
            self._unavailable.setdefault(
                "update_ratio_lr_normalized",
                "lr == 0: the lr-normalized companion is undefined at zero learning rate",
            )
        out_of_band = False
        if check.bounds is not None:
            low, high = check.bounds
            out_of_band = (low is not None and ratio < low) or (high is not None and ratio > high)
        return _RatioObservation(
            ratio=ratio,
            lr=lr,
            lr_normalized=lr_normalized,
            delta_norm=delta_norm,
            zero_baseline=zero_baseline,
            out_of_band=out_of_band,
        )

    def _emit_ratio_finding(
        self, name: str, obs: _RatioObservation, check: UpdateRatioCheck
    ) -> None:
        """Record (and possibly warn) one ratio finding (memo D13)."""

        # Plain assignments keep BOTH codes visible to the contract
        # lockstep scanner (a conditional expression hides the second
        # literal from the code= census).
        code = "update_ratio_zero_baseline"
        if obs.out_of_band:
            code = "update_ratio_out_of_band"
        finding = CheckFinding(
            check="update_ratio",
            code=code,
            severity="warning" if obs.out_of_band else "info",
            action=check.action,
            message=(
                f"{name} update ratio {obs.ratio!r} at accepted step "
                f"{self.accepted_step_id}"
                + (f" outside declared bounds {check.bounds}" if obs.out_of_band else "")
                + (
                    " (zero parameter-norm baseline: ratio is inf by "
                    "declaration, never epsilon-clamped)"
                    if obs.zero_baseline and obs.delta_norm
                    else ""
                )
                + "."
            ),
            names=(name,),
            accepted_step_id=self.accepted_step_id,
            global_step=self._global_step(),
            step_provenance=self._provenance(),
            evidence="clone",
            zero_baseline=obs.zero_baseline,
            values={
                "update_ratio": obs.ratio if math.isfinite(obs.ratio) else None,
                "update_ratio_lr_normalized": (
                    obs.lr_normalized
                    if obs.lr_normalized is not None and math.isfinite(obs.lr_normalized)
                    else None
                ),
                "lr": float(obs.lr) if obs.lr is not None else None,
            },
            remedy="Bounds are per-registration; no universal band exists (38x healthy spread measured)",
        )
        self._record(finding)
        if check.action == "warn" and obs.out_of_band:
            window = check.window_for(name, (1, 1))
            if window.observe(True):
                warnings.warn(
                    TorchLensWarning(
                        finding.message + " Remedy: " + finding.remedy,
                        code="update_ratio_out_of_band",
                    ),
                    stacklevel=4,
                )

    def _commit_frozen(self, delta_stats: dict[str, float]) -> None:
        """Enforce the declared-frozen invariant after the write (memo D5)."""

        if self._frozen is None:
            return
        lookup = self._name_lookup()
        aliases_by_name = {
            name: (canonical, aliases)
            for canonical, aliases, _ in self._entries
            for name in (canonical, *aliases)
        }
        for name in self._frozen.names:
            if self._frozen.evidence == "digest":
                moved = tensor_digest(lookup[name]) != self._frozen.baselines[name]
                evidence_note = (
                    "content-digest evidence: a differing digest PROVES change; "
                    "digest equality is probabilistic and never an exact pass"
                )
                delta_absmax = None
            else:
                moved = delta_stats.get(f"changed::{name}", 0.0) > 0
                delta_absmax = delta_stats.get(f"delta_absmax::{name}")
                evidence_note = "exact clone comparison"
            if not moved:
                continue
            canonical, aliases = aliases_by_name[name]
            all_names = tuple(dict.fromkeys((name, canonical, *aliases)))
            finding = CheckFinding(
                check="frozen_param_changed",
                code="frozen_param_changed",
                severity="critical",
                action=self._frozen.action,
                message=(
                    f"declared-frozen parameter {name} changed at accepted step "
                    f"{self.accepted_step_id} ({evidence_note}"
                    + (f"; max abs delta {delta_absmax:.6g}" if delta_absmax else "")
                    + f"). Every alias of the moved tensor: {list(all_names)}."
                ),
                names=all_names,
                accepted_step_id=self.accepted_step_id,
                global_step=self._global_step(),
                step_provenance=self._provenance(),
                evidence=self._frozen.evidence,
                values={"delta_absmax": delta_absmax},
                remedy=(
                    "Remove the parameter from the optimizer (or set "
                    "requires_grad=False) if it is meant to be frozen"
                ),
            )
            self._record(finding)
            if finding.action == "raise":
                raise CheckViolationError(
                    finding.message + " Remedy: " + finding.remedy,
                    code="frozen_param_changed",
                    finding=finding.to_dict(),
                    names=list(all_names),
                    accepted_step_id=self.accepted_step_id,
                    report=self.report().to_dict(),
                    remedy=finding.remedy,
                )
            if finding.action == "warn":
                warnings.warn(
                    TorchLensWarning(
                        finding.message + " Remedy: " + finding.remedy,
                        code="frozen_param_changed",
                    ),
                    stacklevel=4,
                )

    def _commit_clip_ledger(self) -> None:
        """Feed the clip ledger with this accepted step's totals (D12)."""

        grad_norms = [
            value for key, value in self._prestep_stats.items() if key.startswith("grad_norm::")
        ]
        if not grad_norms:
            return
        post_total = math.sqrt(sum(value**2 for value in grad_norms))
        pre_total = self._last_pre_clip_total
        self._clip_ledger.observe_step(post_total, pre_total)
        if (
            self._magnitude is None
            and self._clip_norm is not None
            and not self._magnitude_censorship_noted
        ):
            self._magnitude_censorship_noted = True
            self._unavailable["grad_magnitude"] = (
                "pre-clip magnitude evidence is not armed and clipping is "
                "declared: the post-clip S-B norm is pinned to max_norm on "
                "every clipped step (measured 6/6), so magnitude verdicts "
                "here would be structurally incapable of firing -- "
                "register_magnitude_check() arms the pre-clip S-A pass"
            )
            self.events.publish(
                ObserverEvent(
                    key="checks/grad_magnitude",
                    kind="unavailable",
                    reason=self._unavailable["grad_magnitude"],
                    accepted_step_id=self.accepted_step_id,
                    axis_provenance=self._axis_provenance(),
                )
            )

    def _scheduled_scan(self) -> None:
        """The S-C scheduled parameter/buffer scan (memo D7): safe raise."""

        if self._nonfinite_param is None:
            return
        entries: list[tuple[str, str, Any]] = [
            (name, "parameter", tensor) for name, _aliases, tensor in self._entries
        ]
        if self._nonfinite_param.include_buffers and self._model is not None:
            entries.extend((name, "buffer", buffer) for name, buffer in self._model.named_buffers())
        rows = scan_named_tensors(entries)
        self._checks_run.append(f"scheduled_scan@{self.accepted_step_id}")
        corrupt = [row for row in rows if row.audited and row.n_nonfinite]
        if not corrupt:
            return
        names = tuple(row.name for row in corrupt)
        finding = CheckFinding(
            check="param_nonfinite",
            code="param_value_nonfinite",
            severity="critical",
            action=self._nonfinite_param.action,
            message=(
                f"nonfinite values in parameter/buffer space after accepted "
                f"step {self.accepted_step_id}: {list(names[:_NAME_SAMPLE])}. A "
                "checkpoint taken now is worthless; every later step is wasted "
                "compute."
            ),
            names=names,
            accepted_step_id=self.accepted_step_id,
            global_step=self._global_step(),
            step_provenance=self._provenance(),
            evidence="scan_kernel",
            values={"n_corrupt_tensors": float(len(corrupt))},
            remedy="Reload the last healthy checkpoint; the raise fired at the safe post-mutation boundary, never mid-step()",
        )
        self._record(finding)
        if finding.action == "raise":
            raise CheckViolationError(
                finding.message + " Remedy: " + finding.remedy,
                code="param_value_nonfinite",
                finding=finding.to_dict(),
                names=list(names),
                accepted_step_id=self.accepted_step_id,
                report=self.report().to_dict(),
                remedy=finding.remedy,
            )
        if finding.action == "warn":
            warnings.warn(
                TorchLensWarning(
                    finding.message + " Remedy: " + finding.remedy,
                    code="param_value_nonfinite",
                ),
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # Events (the trackers seam, memo 4.7)
    # ------------------------------------------------------------------

    def _axis_provenance(self) -> str:
        """Map the boundary state onto the chassis axis vocabulary."""

        return "explicit" if self._boundary is not None else "hook_only"

    def _provenance(self) -> str:
        """Step provenance for findings (memo D10 vocabulary)."""

        return "explicit" if self._boundary is not None else "inferred"

    def _global_step(self) -> int | None:
        """Caller global_step: the explicit boundary's unique property."""

        return self._boundary.global_step if self._boundary is not None else None

    def _publish_watchdog(self) -> None:
        """Publish the watchdog verdict series (memo 4.7 new series)."""

        snapshot = self._watchdog.snapshot()
        self.events.publish(
            ObserverEvent(
                key="checks/watchdog",
                kind="verdict",
                verdict="tripped" if snapshot.tripped else "armed",
                backward_id=self.backward_id,
                global_step=self._global_step(),
                axis_provenance=self._axis_provenance(),
            )
        )

    def _publish_step_events(self) -> None:
        """Publish the per-accepted-step scalar series (memo 4.7)."""

        provenance = self._axis_provenance()
        common: dict[str, Any] = {
            "accepted_step_id": self.accepted_step_id,
            "backward_id": self.backward_id,
            "global_step": self._global_step(),
            "axis_provenance": provenance,
        }
        ledger = self._scale_ledger.snapshot()
        if ledger.scaler_present:
            self.events.publish(
                ObserverEvent(
                    key="checks/scale",
                    kind="scalar",
                    value=ledger.current_scale,
                    **common,
                )
                if ledger.current_scale is not None
                else ObserverEvent(
                    key="checks/scale",
                    kind="unavailable",
                    reason="no backward observed yet: the scale is read once per backward",
                    **common,
                )
            )
            self.events.publish(
                ObserverEvent(
                    key="checks/skipped_attempts",
                    kind="scalar",
                    value=float(ledger.skipped_attempts),
                    **common,
                )
            )
        clip = self._clip_ledger.snapshot()
        if clip.applied_clip_factors:
            self.events.publish(
                ObserverEvent(
                    key="checks/applied_clip_factor",
                    kind="scalar",
                    value=clip.applied_clip_factors[-1],
                    gradient=True,
                    stage="post_clip_applied",
                    scale_provenance="unscaled",
                    **common,
                )
            )
        ratios = self._last_ratios
        if ratios:
            self.events.publish(
                ObserverEvent(
                    key="checks/update_ratio_median",
                    kind="scalar",
                    value=sorted(ratios)[len(ratios) // 2],
                    **common,
                )
            )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _record(self, finding: CheckFinding) -> None:
        """Store one finding under the disclosed hard bound."""

        if len(self._findings) >= MAX_STORED_FINDINGS:
            self._truncated_findings += 1
            return
        self._findings.append(finding)

    def _disclose(self, key: str, value: Any) -> None:
        """Record one disclosure fact."""

        self._disclosures[key] = value

    def report(self, *, finalized_reason: str = "on_demand") -> CheckReport:
        """Assemble the deterministic check report (memo 4.5).

        Callable at any time; the session also finalizes one automatically
        on scope exit including exceptional exit (``final_report``).
        """

        checks_run = list(dict.fromkeys(self._checks_run))
        for registration, label in (
            (self._change, "param_received_no_gradient"),
            (self._frozen, "frozen_param_changed"),
            (self._ratio, "update_ratio"),
            (self._nonfinite_grad, "param_grad_nonfinite"),
            (self._nonfinite_param, "param_value_nonfinite"),
            (self._magnitude, "grad_magnitude"),
        ):
            if registration is not None:
                checks_run.append(label)
        counters = {
            "backwards": self.backward_id,
            "accepted_steps": self.accepted_step_id,
            "explicit_attempts": self._explicit_attempts,
            "explicit_accepted": self._explicit_accepted,
            "explicit_skipped": self._explicit_skipped,
            "truncated_findings": self._truncated_findings,
        }
        coverage = {
            "watched_params": len(self._watched()),
            "inventory_params": len(self._entries),
            "fire_counts_nonzero": sum(1 for count in self._fire_counts.values() if count),
            "scope": (
                "rank_local_shard"
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else "process_local"
            ),
        }
        return CheckReport(
            findings=severity_sorted(self._findings),
            checks_run=tuple(checks_run),
            unavailable=tuple(sorted(self._unavailable.items())),
            disclosures=dict(self._disclosures),
            ledgers={
                "scale": self._scale_ledger.snapshot().to_dict(),
                "clip": self._clip_ledger.snapshot().to_dict(),
                "watchdog": self._watchdog.snapshot().to_dict(),
            },
            coverage=coverage,
            counters=counters,
            finalized_reason=finalized_reason,
        )

    @property
    def final_report(self) -> CheckReport | None:
        """The report finalized at scope exit, when the session was entered."""

        return self._final_report

    def profile(self) -> dict[str, Any]:
        """Run the measured scan on demand (memo D20/D21).

        Cost numbers are DATA at named shapes on this machine state --
        min-of-3 wall time with the load average recorded in the artifact;
        never a percentage, never an adjective.
        """

        import os
        import time

        entries = [(name, "parameter", tensor) for name, _aliases, tensor in self._entries]
        timings: list[float] = []
        for _ in range(3):
            start = time.perf_counter()
            scan_named_tensors(entries)
            timings.append(time.perf_counter() - start)
        return {
            "scan_ms_min_of_3": min(timings) * 1e3,
            "n_tensors": len(entries),
            "n_elements": sum(
                tensor.numel()
                for _name, _kind, tensor in entries
                if isinstance(tensor, torch.Tensor)
            ),
            "estimated_snapshot_bytes": self.estimated_snapshot_bytes,
            "loadavg": os.getloadavg(),
        }


class _PausedScope:
    """Context manager for ``session.paused()`` (memo 4.5)."""

    def __init__(self, session: ChecksSession) -> None:
        self._session = session
        self._was_paused = False

    def __enter__(self) -> None:
        self._was_paused = self._session._paused
        self._session._paused = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._session._paused = self._was_paused


__all__ = ["HAS_MULTI_GRAD_HOOK", "MAX_STORED_FINDINGS", "ChecksSession"]
