"""The episode declaration dataclass for ``tl.trace(..., episode=...)``.

Split from :mod:`torchlens.options` along the episode option-family seam
(R43 file-size ratchet); the public spelling stays
``tl.options.EpisodeSpec`` (re-exported there). Semantics are pinned by the
ratified S2/S6/S7 contracts; the doc of record is
``docs/reference/episode_capture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["EpisodeSpec"]


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode declaration for ``tl.trace(..., episode=EpisodeSpec(...))``.

    Declaring an episode makes ONE wrapped session capture the episode
    product (``capture_kind=episode``): the episode root's ``forward`` steps
    ``stepped_module`` N times, and the per-step status ledger lands ON the
    product at ``trace.annotations["episode"]``.

    Every spelling here is DOCUMENTED-UNSTABLE pending the rolling naming
    session (no deprecation shim owed); semantics are pinned by the ratified
    S2/S6/S7 contracts.

    DIAGNOSTIC-TIER COST WARNING: the wrapped episode tier is the
    verification oracle / deep-dive product for TENS of steps, not hundreds.
    Measured on gpt2-124M (CPU): N=20 costs 79 s / 146 MB artifact / 1.9 GB
    peak RSS; N=100 costs 657 s (323x native) / 947 MB / 5.4 GB. Cost is
    SUPERLINEAR in step count. The guarded-fast tier remains the default
    engine for episode-scale work.

    Parameters
    ----------
    stepped_module:
        The stepped model: the ``nn.Module`` whose successive top-level calls
        define step boundaries (call 1 is the prefill, ledger row 0). Must be
        a submodule of the traced episode root; refused typed
        (``episode_declaration_invalid``) otherwise. Step tallying is FLAT:
        each top-level call of this module is one step regardless of any
        loop nesting inside the episode root's ``forward``. On a
        bound-method root (F41: ``tl.trace(model.generate, ids,
        episode=...)``) it DEFAULTS to the bound method's owner
        (``method.__self__``); on a module root it is required and ``None``
        refuses typed.
    n_steps:
        Optional declared step count, recorded on the ledger header and
        validated against the observed call count on COMPLETE captures.
        Declarations beyond the wrapped-tier cost ceiling
        (``torchlens.capture._episode_ledger.EPISODE_DECLARED_STEP_CEILING``,
        100 steps) refuse typed (``episode_step_ceiling_exceeded``) at
        declaration time unless ``acknowledge_step_cost=True`` — the cost is
        superlinear and N=100 already measured 657 s / 5.4 GB peak RSS.
    step_axis:
        Axis of the step-output source along which per-step evidence lies.
        Positions are TAIL-ALIGNED: the LAST ``n_steps`` positions along
        this axis are the per-step emissions (one appended position per
        step), so a root returning what real ``generate()`` returns —
        prompt+completion — keeps its prompt prefix out of the evidence
        column, and an emitted-only root is the equal-size special case.
    step_output_kind:
        Declared per-step evidence kind (foldA D8): ``"tokens"`` (default)
        reads integer token ids from the source; ``"digest"`` reads
        per-step content digests (any dtype — the float/hidden-state
        shape); ``"none"`` derives no evidence at all (a status-only
        ledger; the shape for roots with no per-step output structure,
        e.g. diffusion images), and admits value-free save policies.
    step_output_from:
        Declared step-output SOURCE: which root-output slot the per-step
        evidence derives from, as a dot-separated container path over the
        root output structure (dict keys and namedtuple/``ModelOutput``
        field names by name, tuple positions by index — ``"sequences"``,
        ``"0"``, ``"0.logits"``). ``None`` (default) requires the root to
        return a single output tensor; dict/``ModelOutput``/tuple roots
        declare the slot. Contradicts ``step_output_kind="none"`` (refused
        typed).
    forced_tokens:
        Optional teacher-forced feed declaration: the token sequence the
        driver feeds instead of model emissions. Declaring it stamps
        ``token_feed="forced"`` and ``fidelity_basis="forced"`` on the
        ledger header — an explicitly NON-VERIFYING disclosed mode; token
        fidelity obligations (E-A3) never verify a forced episode.
    state:
        Declared episode-carried state items beyond the built-in scope
        (token prefix, KV cache, RNG streams). EVERY declared item is
        preflighted at DECLARATION time, unconditionally: an item without
        snapshot/restore support refuses typed
        (``episode_state_unsnapshotable``) before execution (E-A4).
    rng:
        Seeding discipline. Only ``"managed"`` ships: the capture's
        effective ``random_seed`` is drawn-or-passed as today and recorded
        as the ledger header's ``entry_seed``.
    acknowledge_step_cost:
        Explicit override for the declared-step cost ceiling (placeholder
        spelling, [UI-SPRINT]). ``True`` accepts a declaration beyond the
        ceiling with the diagnostic-tier cost acknowledged; the default
        refuses typed so hundreds-of-steps wrapped captures are a decision,
        never an accident.
    escalated_from:
        Producer digest of the cheap-tier product this capture escalates
        (present iff escalation; travels with ``reason`` — E-A2). Build the
        whole escalation declaration with
        ``torchlens.capture._episode_ledger.escalation_spec(producer, ...)``.
    reason:
        Escalation reason, closed vocabulary
        ``{"step_failed", "divergence", "requested"}``.
    feed:
        Join-arm strictness (lane F40c). ``"open"`` (default) captures
        across cross-step feed breaks and grades every join in the measured
        ``step_join`` envelope; episode-dependent claims refuse across a
        broken join, everything else stays usable. ``"closed"`` is the
        opt-in STRICT arm: the feed is declared closed (each step's input
        directly continues the prior step's output), an UNDECLARED crossing
        halts capture at the next step entry -- the first observable point
        (one-step detection latency) -- with typed
        ``episode_feed_closed_violation`` and the partial product on
        ``exc.partial_log``, and a DECLARED crossing stops before entry
        (``episode_declared_crossing_stop``).
    on_feed_break:
        The FORK-4 arm switch (both arms built; the ruling selects the
        default's shape). ``"disclose"`` (default, the panel 2-1): return
        the one Trace with the break marked and claims refused across it.
        ``"refuse"``: the settlement write raises typed
        ``episode_feed_break_exogenous`` carrying the settled product on
        ``exc.partial_log`` (recoverable partial evidence).
    crossings:
        DECLARED exogenous crossings: step indices (1-based joins into
        steps 1+) whose entries are declared to come from outside the
        capture (the tool-call shape). A declared crossing grades
        ``declared`` -- disclosed, never silently continuous; series claims
        still refuse across it (``episode_join_declared_crossing``), the
        run stays chain-shaped by declaration.
    expected_tokens:
        The cheap-tier product's per-step token column (one tuple per step),
        carried so the escalated capture can discharge the E-A3 fidelity
        obligation at write time: prefix-equal columns record
        ``fidelity_basis="tokens"``; a mismatch records ``"diverged"`` — the
        escalated product is still a valid capture of what it ran, it just
        is not an escalation of the original episode, and says so. Never a
        settlement input.
    """

    stepped_module: Any = None
    n_steps: int | None = None
    step_axis: int = -1
    step_output_kind: Literal["tokens", "digest", "none"] = "tokens"
    step_output_from: str | None = None
    forced_tokens: tuple[int, ...] | None = None
    state: tuple[Any, ...] = ()
    rng: Literal["managed"] = "managed"
    escalated_from: str | None = None
    reason: str | None = None
    expected_tokens: tuple[tuple[int, ...], ...] | None = None
    acknowledge_step_cost: bool = False
    feed: Literal["open", "closed"] = "open"
    on_feed_break: Literal["disclose", "refuse"] = "disclose"
    crossings: tuple[int, ...] = ()
