"""Live site-key minting (C03; surgery memo Build 0c).

Site keys (``site_key_v1``) are minted at postprocess step 7 today, which
makes structural site addressing REPLAY-ONLY: mid-forward, no op carries a
key. This module moves the SAME streaming minter (:class:`SiteKeyMinter`,
per-cohort ordinal counters fed in execution order) to capture time, so
``tl.site(...)`` WHERE terms can resolve during a live capture -- the address
law's promise that a structural address is valid in EVERY lane, including a
first capture.

Design constraints honored:

- ZERO capture-core edits: the minter is armed from the ``tl.trace`` entry
  (``user_funcs``) through a ContextVar ONLY when the configured predicates
  contain a ``site`` selector; unarmed evaluation refuses typed. Captures
  are single-threaded by design, so one armed minter per capture is exact.
- Mint-once-per-event: every predicate slot sees every candidate op in
  execution order, and several rules may evaluate one op, so the minter
  memoizes by the context's monotone ``event_index`` -- ordinals count each
  executed candidate exactly once.
- THE HONEST BOUNDARY (surgery risk 1, verified by the parity pin in
  ``tests/test_interv_substrate_site_selector.py``): live ordinals count
  EXECUTED occurrences; postprocess ordinals count RETAINED ops (orphans
  consume no ordinals, SF-63). On a model whose pruned dead ops share a
  cohort with later live ops the two numberings diverge -- the parity test
  pins agreement on orphan-free models, and the divergence is detectable
  post hoc by comparing ``op.site_key`` against the live-minted key.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..postprocess._site_key import SiteKeyMinter

_ACTIVE_LIVE_MINTER: ContextVar[LiveSiteKeyMinter | None] = ContextVar(
    "tl_active_live_site_key_minter", default=None
)


class LiveSiteKeyMinter:
    """Streaming ``site_key_v1`` minter over capture-time record contexts."""

    __slots__ = ("_minter", "_by_event")

    def __init__(self) -> None:
        self._minter = SiteKeyMinter()
        self._by_event: dict[int, str] = {}

    def key_for_context(self, ctx: Any) -> str:
        """Mint (once per event) the live site key for one op context.

        Parameters
        ----------
        ctx:
            Capture-time ``RecordContext`` for an op event.

        Returns
        -------
        str
            The ``site_key_v1`` string for this executed occurrence.
        """

        event_index = int(getattr(ctx, "event_index", -1))
        if event_index in self._by_event:
            return self._by_event[event_index]
        frames = tuple(getattr(ctx, "module_stack", ()) or ())
        entries = tuple((frame.address, frame.pass_index) for frame in frames)
        key = self._minter.mint(
            entries,
            str(getattr(ctx, "layer_type", None) or getattr(ctx, "type", "")),
            getattr(ctx, "output_index", None),
        )
        if event_index >= 0:
            self._by_event[event_index] = key
        return key


def active_live_minter() -> LiveSiteKeyMinter | None:
    """Return the armed live minter for the current capture, if any."""

    return _ACTIVE_LIVE_MINTER.get()


@contextmanager
def armed_live_minter() -> Iterator[LiveSiteKeyMinter]:
    """Arm one fresh live minter for the duration of one capture."""

    minter = LiveSiteKeyMinter()
    token = _ACTIVE_LIVE_MINTER.set(minter)
    try:
        yield minter
    finally:
        _ACTIVE_LIVE_MINTER.reset(token)


def predicate_needs_live_site_keys(*predicates: Any) -> bool:
    """Whether any configured predicate tree contains a ``site`` selector.

    The public InterventionSpec exposes its rules' WHERE terms; bare
    selectors and legacy predicate wrappers are walked through the shared
    ``selector_contains_kind`` machinery.
    """

    from ..ir.selector_eval import selector_contains_kind

    for predicate in predicates:
        if predicate is None:
            continue
        rules = getattr(predicate, "rules", None)
        candidates = (
            [rule.where for rule in rules]
            if rules is not None and not callable(rules)
            else [predicate]
        )
        for candidate in candidates:
            if selector_contains_kind(candidate, "site", unwrap=True):
                return True
    return False


def mint_keys_in_execution_order(trace: Any) -> dict[str, str]:
    """Re-mint keys over the trace's retained ops in execution order (pure).

    The parity harness: feeding the SAME streaming minter the retained ops
    in raw execution order must reproduce ``op.site_key`` byte-identically
    (property P4 sibling). Used by the C03 parity pin; also the reference
    implementation the F01 bind runtime can consume.

    Returns
    -------
    dict[str, str]
        Final op label -> re-minted key.
    """

    minter = SiteKeyMinter()
    keys: dict[str, str] = {}
    for label in trace.op_labels:
        op = trace.ops[label]
        keys[label] = minter.mint(
            getattr(op, "modules", None) or (),
            op.layer_type,
            getattr(op, "multi_output_index", None),
        )
    return keys


__all__ = [
    "LiveSiteKeyMinter",
    "active_live_minter",
    "armed_live_minter",
    "mint_keys_in_execution_order",
    "predicate_needs_live_site_keys",
]
