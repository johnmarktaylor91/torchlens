"""Replay run-context store: pass-qualified site keys and the capture-truth digest ledger.

Session-time state the replay/push engine keeps on ``Trace.last_run`` (RUNTIME
storage, never persisted): the pass-qualified key every replay structure is
addressed by, the run-context accessor that seeds it, and the ledger of
capture-time out digests recorded the first time replay overwrites a site.
Split out of ``replay.py`` so validation can read the ledger without importing
the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from .edge_substitution import tensor_content_digest

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


def _replay_site_key(site: Op) -> str:
    """Return the pass-qualified replay key for one op record.

    ``Op.label`` is the pass-qualified ``layer_label:pass`` spelling on every
    finished-trace op (single-pass ops carry ``:1``), and every such spelling
    is a ``layer_dict_all_keys`` lookup key, so replay state keyed by it can
    never collide across passes of a recurrence-grouped layer. Bare
    ``layer_label`` keys map to the LAST pass only — keying replay state by
    them is exactly the pass-blind corruption this key exists to prevent.
    """

    label = getattr(site, "label", None)
    if isinstance(label, str) and label:
        return label
    return site.layer_label


def _ensure_replay_run_ctx(log: Trace) -> dict[str, Any]:
    """Return a mutable replay run context on ``log``.

    Parameters
    ----------
    log:
        Model log being replayed.

    Returns
    -------
    dict[str, Any]
        Run context dictionary.
    """

    if not isinstance(getattr(log, "last_run", None), dict):
        log.last_run = {}
    run_ctx = cast(dict[str, Any], log.last_run)
    # Seed law D5 (F02): thread the capture's recorded seed so seed='auto'
    # stochastic edits canonicalize their base seed on the replay door too.
    trace_seed = getattr(log, "random_seed", None)
    if trace_seed is not None:
        run_ctx.setdefault("trace_random_seed", trace_seed)
    return run_ctx


def _record_capture_digest(
    site: Op, known_digests: Mapping[str, str], new_digests: dict[str, str]
) -> None:
    """Record a site's capture-truth out digest the FIRST time replay overwrites it.

    The site's current out IS the capture-time value its children's
    ``saved_args`` snapshotted, so it is digested before being replaced;
    validation compares those snapshots against the digest once the value
    itself is gone (see :func:`replay_capture_digest`). Later overwrites of an
    already-recorded site never re-digest (the pushed value is not truth).
    """

    site_key = _replay_site_key(site)
    if site_key in known_digests or site_key in new_digests:
        return
    current = site.out
    if isinstance(current, torch.Tensor):
        new_digests[site_key] = tensor_content_digest(current)


REPLAY_CAPTURE_DIGESTS_KEY = "replay_capture_digests"
"""Run-context key holding ``{replay key: sha256}`` of every site's out as it
stood the FIRST time this trace's replay engine overwrote it.

A pushed trace no longer carries the capture-time value of any recomputed
site, but its children's ``saved_args`` snapshots still do; the digest is the
one piece of capture truth that lets validation keep comparing those
snapshots against the producer (:func:`replay_capture_digest`) instead of
against a pushed value that legitimately differs. Session-time only: it rides
``Trace.last_run`` (the replay run context, RUNTIME storage) and is never
persisted.
"""


def replay_capture_digest(log: Trace, site: Op) -> str | None:
    """Return the capture-truth digest of a replay-recomputed site's out.

    ``None`` means the replay engine has never overwritten this site on this
    trace (its current ``out`` IS capture truth, compare values directly) or
    that no digest was recordable (the pre-edit out was not a tensor).
    """

    run_ctx = getattr(log, "last_run", None)
    if not isinstance(run_ctx, dict):
        return None
    digests = run_ctx.get(REPLAY_CAPTURE_DIGESTS_KEY)
    if not isinstance(digests, dict):
        return None
    digest = digests.get(_replay_site_key(site))
    return digest if isinstance(digest, str) else None
