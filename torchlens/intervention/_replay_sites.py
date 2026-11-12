"""Site-level replay decisions: source records, stochastic draws, buffer threading.

Split from :mod:`torchlens.intervention._replay`. Each helper classifies one site or
answers one pure question about it (is it a source record, does a random draw keep
its captured value, what value threads through a buffer version record); the replay
loop in :mod:`~torchlens.intervention._replay` acts on the answers and raises the
typed refusals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from .._utils._stochastic_funcs import _STOCHASTIC_FUNC_NAMES
from ..backends._backend_names import TORCH_BACKEND_NAME
from ._backend_adapter_lookup import backend_adapter
from ._replay_cone import _disclosure_label
from ._replay_context import _replay_site_key
from .errors import ReplayPreconditionError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _is_source_record(site: Op) -> bool:
    """Whether a site is a source record with nothing to re-execute.

    A buffer version record and a model-input leaf carry no callable: their
    replay value is threaded or captured, never recomputed, and a hook aimed
    at one fires on that value.

    Parameters
    ----------
    site:
        Site to classify.

    Returns
    -------
    bool
        True for buffer records and input leaves.
    """

    if record_member(site, "is_buffer", False):
        return True
    return bool(getattr(site, "is_input", False)) and site.func is None and not site.parents


def _is_passthrough_replacement(site: Op) -> bool:
    """Whether a site is a module-edit replacement op: no callable, one parent.

    A ``tl.module(...)`` edit is recorded as an ``interventionreplacement``
    op between the module's last op and its consumers (at capture, or by a
    post-capture ``do``). It executed nothing: its replay value is its
    parent's current value, on which the hooks aimed at it fire.

    Parameters
    ----------
    site:
        Site to classify.

    Returns
    -------
    bool
        True for a single-parent functionless replacement op.
    """

    return (
        site.func is None
        and getattr(site, "func_name", None) == "intervention_replacement"
        and len(tuple(getattr(site, "parents", ()) or ())) == 1
    )


def _thread_passthrough_site(site: Op, state: Any, commit: Any) -> None:
    """Stage a replacement op's replay value: its parent's value, hooks applied.

    Parameters
    ----------
    site:
        Replacement op in the replay cone.
    state:
        The replay pass (overlay, hook targets, pending updates).
    commit:
        ``replay._commit_member``, which fires the site's hooks and stages it.
    """

    parent_label = tuple(site.parents)[0]
    keys = state.label_keys.get(parent_label) or (parent_label,)
    value = state.overlay.get(keys[0])
    if value is None:
        parent = state.log._layer_dict_all_keys.get(keys[0])
        value = getattr(parent, "out", None)
    if isinstance(value, torch.Tensor):
        commit(site, value, state)


def _no_func_message(site: Op, *, loaded: bool) -> str:
    """Return the refusal text for a site with no callable to replay.

    Parameters
    ----------
    site:
        Site that cannot be replayed.
    loaded:
        Whether the trace came from a saved file.

    Returns
    -------
    str
        Refusal message, naming the loaded-trace condition when that is the cause.
    """

    if not loaded:
        return f"{site.layer_label!r} has no func for replay"
    return (
        f"{site.layer_label!r} has no func for replay: this trace was loaded from a saved "
        "file, and a save keeps no capture-time callables or argument templates, so edits "
        "cannot be replayed on it. Edit a trace captured in this process "
        "(tl.trace(model, inputs)), or rerun the model with the edit staged: "
        "trace.run(model, inputs)"
    )


def _unmet_log_precondition(log: Trace) -> str | None:
    """Return the first log-level replay precondition ``log`` fails, else None.

    Parameters
    ----------
    log:
        Model log to check.

    Returns
    -------
    str | None
        The refusal message for the first unmet precondition (a completed
        capture, then per-op argument templates), or None.
    """

    if not getattr(log, "_tracing_finished", False):
        return "replay requires a completed Trace"
    if not (
        getattr(log, "is_replayable", False)
        or getattr(getattr(log, "options", None), "save_arg_templates", False)
    ):
        backend = str(log._provenance_backend or TORCH_BACKEND_NAME)
        if backend != TORCH_BACKEND_NAME:
            if backend_adapter(log) is not None:
                # Replayed op by op through the backend's replay adapter
                # (intervention/_backend_replay.py), which refuses per op.
                return None
            return (
                f"the {backend} preview backend cannot edit or patch a finished trace: replay "
                "needs per-op argument templates, which only PyTorch captures record; stage "
                f"the edit in a new run: tl.trace(model, inputs, backend={backend!r}, "
                "intervene=...)"
            )
        return (
            "replay requires per-op argument templates, and this trace has none (a "
            "tl.record recording and a structure_only capture keep none; every other "
            "tl.trace capture records them); "
            "capture with tl.trace(model, inputs), or rerun the model with the edit staged: "
            "trace.do(site, edit, model=model, inputs=inputs)"
        )
    return None


def _require_hook_origins(origins: list[Op]) -> list[Op]:
    """Return a push's resolved origin sites, refusing an empty set.

    Parameters
    ----------
    origins:
        Hook target sites resolved for one push, in execution order.

    Returns
    -------
    list[Op]
        ``origins`` unchanged.

    Raises
    ------
    ReplayPreconditionError
        When no hook resolved to a site, so the push has nothing to replay from.
    """

    if not origins:
        raise ReplayPreconditionError("push requires at least one hook target")
    return origins


def _keeps_captured_draw(
    representative: Op,
    replay_group: Sequence[Op],
    *,
    pending_updates: Mapping[str, torch.Tensor],
    label_keys: Mapping[str, tuple[str, ...]],
    log: Trace | None = None,
) -> bool:
    """Whether a stochastic call keeps its captured output instead of re-executing.

    Re-executing a random-source op (``randn_like``, ``dropout``, ...) with no
    recorded RNG state draws fresh numbers: the replay would replace the
    captured value with an unrelated one before any hook fires, so an edit
    aimed at the site would edit a value the trace never held. When no
    argument of the call changed in this replay there is nothing to
    recompute and the captured draw stands. With RNG states recorded
    (``save_rng_states``) the call re-executes and reproduces its draw; a
    call whose arguments did change re-executes as well (its new draw is the
    disclosed stochastic-propagation behaviour).

    "Changed" means the parent's replay VALUE differs from its captured
    value. The cone over-approximates (a hooked member's call-group
    siblings and their consumers are recomputed too), so a parent is often
    recomputed to exactly the value the trace holds; that recomputation is
    not a changed argument, and the draw keeps. Without ``log`` the check
    degrades to presence in ``pending_updates``.

    Parameters
    ----------
    representative:
        Representative site of the function-call group.
    replay_group:
        Sites the call produced.
    pending_updates:
        Values this replay has recomputed so far, by replay site key.
    label_keys:
        Label spelling to replay site keys.
    log:
        The trace being replayed, for the parents' captured values.

    Returns
    -------
    bool
        True when the captured output is the correct replay value.
    """

    if not _is_unseeded_single_draw(representative, replay_group):
        return False
    if record_member(representative, "edge_substitutions", None):
        return False
    if _edited_at_capture(representative):
        # The captured out already carries the capture-time edit, and the
        # replay composes that edit again at the site: keeping the value
        # would apply it twice. The raw draw is only recoverable by
        # re-execution (seeded when RNG states were saved).
        return False
    return not _any_parent_updated(representative, pending_updates, label_keys, log=log)


def _edited_at_capture(site: Op) -> bool:
    """Whether ``site`` carries a capture-time fire record or replaced-op mark.

    Parameters
    ----------
    site:
        Op record to inspect.

    Returns
    -------
    bool
        True when a capture-time hook already shaped the recorded out.
    """

    if getattr(site, "is_edited", False):
        return True
    return bool(getattr(site, "edits", None))


def _is_unseeded_single_draw(representative: Op, replay_group: Sequence[Op]) -> bool:
    """Whether a group is one tensor-valued random draw with no recorded RNG state.

    Parameters
    ----------
    representative:
        Representative site of the function-call group.
    replay_group:
        Sites the call produced.

    Returns
    -------
    bool
        True for a single-output stochastic op whose ``func_rng_states`` is empty.
    """

    func_name = str(getattr(representative, "func_name", "") or "").rstrip("_")
    if func_name not in _STOCHASTIC_FUNC_NAMES:
        return False
    if record_member(representative, "func_rng_states", None):
        return False
    if len(replay_group) != 1 or tuple(representative.container_path or ()):
        return False
    return isinstance(representative.out, torch.Tensor)


def _any_parent_updated(
    site: Op,
    pending_updates: Mapping[str, torch.Tensor],
    label_keys: Mapping[str, tuple[str, ...]],
    *,
    log: Trace | None = None,
) -> bool:
    """Whether any parent of ``site`` carries a changed value in this replay.

    Parameters
    ----------
    site:
        Site whose parents are checked.
    pending_updates:
        Values this replay has recomputed so far, by replay site key.
    label_keys:
        Label spelling to replay site keys.
    log:
        The trace being replayed. When given, a pending parent whose replay
        value equals its captured value does not count as updated; a parent
        whose captured record cannot be found or compared does.

    Returns
    -------
    bool
        True when a parent's replay key carries a pending update whose value
        differs from the captured one (or any pending update, without ``log``).
    """

    for parent in site.parents or ():
        for key in label_keys.get(parent, (parent,)):
            if key not in pending_updates:
                continue
            if log is None or _pending_value_differs(log, key, pending_updates[key]):
                return True
    return False


def _pending_value_differs(log: Trace, key: str, pending: torch.Tensor) -> bool:
    """Whether a staged replay value differs from the captured value at ``key``.

    Parameters
    ----------
    log:
        The trace being replayed.
    key:
        Replay site key (pass-qualified label) of the recomputed site.
    pending:
        The value this replay staged for that site.

    Returns
    -------
    bool
        False only when the captured record is found, holds a tensor of the
        same shape and dtype, and compares equal element-wise.
    """

    captured = log._layer_dict_all_keys.get(key)
    captured_out = getattr(captured, "out", None)
    if not isinstance(captured_out, torch.Tensor) or not isinstance(pending, torch.Tensor):
        return True
    if captured_out.shape != pending.shape or captured_out.dtype != pending.dtype:
        return True
    try:
        return not torch.equal(captured_out.detach(), pending.detach().to(captured_out.device))
    except RuntimeError:
        return True


def _frontier_leaf(
    frontier: dict[str, torch.Tensor],
    label: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh replay-frontier leaf for a saved tensor.

    Parameters
    ----------
    frontier:
        Mutable frontier cache keyed by source label.
    label:
        Source label for the boundary tensor.
    tensor:
        Saved tensor entering the replay cone.

    Returns
    -------
    torch.Tensor
        Detached cloned leaf requiring grad.
    """

    if label not in frontier:
        frontier[label] = tensor.detach().clone().requires_grad_()
    return frontier[label]


def _buffer_thread_candidate(
    site: Op,
    trace: Trace,
    overlay: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor | None, str | None]:
    """Return the value to thread through one buffer record and the gap reason, if any.

    A written-buffer version record's single parent is the op that performed
    the write, and for ``inplace``/``reassign`` write kinds that op's output
    IS the post-write buffer state. Threading is self-certifying: the
    capture-time buffer value must equal the capture-time producer out, or
    the record keeps its captured value and the returned reason discloses
    the gap (the caller warns or raises on it).

    Parameters
    ----------
    site:
        Buffer record inside the replay cone.
    trace:
        Model log being replayed.
    overlay:
        Current replay outs keyed by pass-qualified replay key.

    Returns
    -------
    tuple[torch.Tensor | None, str | None]
        ``(recomputed producer out, None)`` when the value threads;
        ``(None, reason)`` for a disclosed gap; ``(None, None)`` when the
        record simply keeps its captured value (no parent, or a producer not
        recomputed in this replay).
    """

    parents = tuple(getattr(site, "parents", ()) or ())
    if not parents:
        return None, None
    if len(parents) != 1:
        return None, f"record has {len(parents)} parents, not one writing op"
    producer_key = parents[0]
    producer = trace._layer_dict_all_keys.get(producer_key)
    recomputed = overlay.get(_replay_site_key(producer)) if producer is not None else None
    if producer is None or recomputed is None:
        return None, None
    write_kind = getattr(site, "buffer_write_kind", None)
    if write_kind not in {"inplace", "reassign"}:
        if getattr(site, "buffer_value_changed", None) is not False:
            return None, (
                f"write kind {write_kind!r} does not prove the writing op's output equals "
                "the post-write buffer state"
            )
        return None, None
    if not _buffer_write_corroborated(site.out, producer.out):
        return None, (
            f"capture-time corroboration failed: the buffer value does not equal the "
            f"writing op {producer_key!r}'s captured output"
        )
    return recomputed, None


def _buffer_write_corroborated(captured_site: object, captured_producer: object) -> bool:
    """Whether the captured buffer value equals the writing op's captured output.

    Both must be tensors of one shape with equal values; anything else leaves
    the threading unproven.
    """

    return (
        isinstance(captured_site, torch.Tensor)
        and isinstance(captured_producer, torch.Tensor)
        and captured_site.shape == captured_producer.shape
        and bool(torch.equal(captured_site, captured_producer))
    )


def _parent_value(
    parent: Op,
    pass_log: Op,
    overlay: dict[str, torch.Tensor],
    differentiable_frontier: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Return the value ``pass_log`` reads from ``parent`` during replay.

    In order: the replay overlay, the version of the parent this child saw at
    capture (``out_versions_by_child``), then the parent's saved out; a
    differentiable replay reads the latter two through frontier leaves.
    """

    parent_key = _replay_site_key(parent)
    if parent_key in overlay:
        return overlay[parent_key]
    versions = getattr(parent, "out_versions_by_child", {}) or {}
    child_spelling = next(
        (
            spelling
            for spelling in (_replay_site_key(pass_log), pass_log.layer_label)
            if spelling in versions
        ),
        None,
    )
    if child_spelling is not None:
        version = versions[child_spelling]
        if isinstance(version, torch.Tensor):
            if differentiable_frontier is not None:
                return _frontier_leaf(
                    differentiable_frontier,
                    f"{_disclosure_label(parent)}->{_disclosure_label(pass_log)}",
                    version,
                )
            return version
    if isinstance(parent.out, torch.Tensor):
        if differentiable_frontier is not None:
            return _frontier_leaf(differentiable_frontier, _disclosure_label(parent), parent.out)
        return parent.out
    raise ReplayPreconditionError(
        f"parent {parent.layer_label!r} for {pass_log.layer_label!r} has no out"
    )
