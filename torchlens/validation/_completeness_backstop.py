"""Completeness backstop census: dispatcher-witness counts against captured ops.

``completeness_backstop_counts`` reduces the aten-dispatch completeness witness
signals of one forward (decompositions, captured ops, unaccounted-dispatch
diagnostics) to the two censuses ``CHECK_COMPLETENESS`` compares, including the
tripwire rule that counts a module-forward-owned dispatch like an unowned one
(``_is_unowned_gap_diagnostic``). Split from ``validation/core.py`` along the
census seam (R43 file-size ratchet); the logic is unchanged.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

_MODULE_FORWARD_OWNER_PREFIX = "module_forward:"


def _is_unowned_gap_diagnostic(entry: dict[str, Any]) -> bool:
    """Return whether a witness diagnostic is an aten dispatch no torch wrapper owned.

    ``unowned_dispatch`` has no owner token at all. An ``owner_not_captured`` entry whose
    owner is a module-forward token (``module_forward:exhaustive`` / ``:predicate``) is the
    same event one nesting level down: the module wrapper only brackets
    ``orig_forward``, so the innermost token being the module's means no torch-function
    wrapper intervened between module code and the dispatcher -- a stale pre-wrap torch
    reference (for example a module-global ``from torch import f`` alias) inside a wrapped
    submodule. Whether a drop counts must not depend on whether the call sits in a nested
    submodule or the top-level forward. Benign pure-read ``owner_not_captured`` entries
    (``torch.equal`` / ``torch.allclose`` deciding a branch) are owned by their own
    ``torch_func:*`` wrapper and stay excluded.
    """

    reason = entry.get("reason")
    if reason == "unowned_dispatch":
        return True
    owner_wrapper = entry.get("owner_wrapper")
    return (
        reason == "owner_not_captured"
        and isinstance(owner_wrapper, str)
        and owner_wrapper.startswith(_MODULE_FORWARD_OWNER_PREFIX)
    )


def completeness_backstop_counts(trace: "Trace") -> tuple[int, int]:
    """Return ``(dispatch_census, captured_census)`` for the completeness backstop.

    Two independent censuses of "distinct dispatching operations" that must agree
    1:1 EXCEPT in the narrow, provably-scoped regions where TorchLens
    intentionally breaks that correspondence. The censuses are built from three
    per-forward signals collected by the aten-dispatch completeness witness:

    * ``completeness_decompositions`` -- one entry per WRAPPED call that owned at
      least one aten dispatch, tagged ``capture_accounted`` (the op emitted a real
      aten dispatch) and ``in_replacement_hook`` (the dispatch fired inside a
      genuine raw ``register_forward_hook`` output replacement).
    * ``layer_list`` -- the captured computational ops (``func_call_id``).
    * ``completeness_diagnostics`` -- one entry per UNACCOUNTED aten dispatch,
      including ``unowned_dispatch`` (a real aten op with NO capturing owner --
      a directly-dispatched ``torch.ops.aten.*`` call or an unwrapped route) and
      ``owner_not_captured`` (a WRAPPED op with a live owner that emitted no
      captured op). Each diagnostic also carries ``mutates`` -- whether the aten
      operator writes to any argument, read from the operator's own
      ``FunctionSchema`` (not a name heuristic).

    **Captured census** = captured ops that OWN an accounted aten dispatch
    (``captured_fcids & accounted_owner_fcids``). A captured op that legitimately
    emits NO owned aten dispatch is NOT counted, because it has no dispatch
    counterpart by design and is therefore not a census mismatch. This covers
    both a ``torch.func`` transform boundary (its interior is opaque) AND a
    no-op/view/meta high-level call such as same-shape ``torch.broadcast_tensors``
    (which returns its inputs and dispatches nothing). This never masks a real
    drop: a dropped op is one TorchLens FAILED to capture, which surfaces as an
    unowned aten dispatch below -- a captured op being present cannot hide it.

    **Dispatch census** = every accounted owner, MINUS orphaned owners excused as
    genuine replacement construction, PLUS every unowned aten dispatch that fired
    OUTSIDE a replacement hook:

    * An UNOWNED aten dispatch outside a replacement hook is a real op with no
      captured owner -- a silent capture drop. Each one is added to the dispatch
      census so it fails ``CHECK_COMPLETENESS``. This is exactly the tripwire the
      backstop exists to arm, and NOTHING in the carve-outs relaxes it.
    * An ``owner_not_captured`` aten dispatch whose owner is a MODULE-FORWARD token
      (no torch-function wrapper between module code and the dispatcher) is the
      unowned case one nesting level down and is counted exactly like it
      (``_is_unowned_gap_diagnostic``).
    * Any other ``owner_not_captured`` aten dispatch is a WRAPPED op whose owner emitted
      no captured op. On correct models this is benign PURE-READ control flow
      (``torch.equal`` / ``torch.allclose`` deciding a branch), so it is NOT
      counted -- masking that would false-fail correct models. But an
      ``owner_not_captured`` dispatch that MUTATES an argument (``mutates=True``)
      is a value-affecting drop the graph missed: a real completeness failure,
      NOT benign control flow. Each such mutating drop (outside a replacement
      hook) IS added to the dispatch census so the backstop fails. This is the
      sound, narrow strengthening added for the round-3 hidden-mutation hunt.

    OBSERVATIONAL BOUNDARY (documented, not papered over): the witness is a
    ``TorchDispatchMode`` and can only census aten dispatches it actually
    observes. A tensor subclass whose ``__torch_dispatch__`` performs a mutation
    under ``torch._C._DisableTorchDispatch()`` hides that nested aten op from the
    dispatcher entirely (PyTorch suppresses mode re-entry while a subclass handles
    an op), so the mutation is INVISIBLE to the witness and cannot be counted.
    This is the cooperative-model boundary: the witness assumes standard dispatch
    and cannot see ops a subclass deliberately executes with dispatch disabled.
    The strengthening above closes every OBSERVABLE uncaptured mutation; a
    mutation deliberately hidden under disabled dispatch remains out of reach and
    is documented in ``docs/reference`` rather than silently claimed as caught.
    * A genuine output-replacement ``register_forward_hook`` builds its
      replacement with untraceable dispatch -- raw-aten calls (unowned) and
      python-wrapped calls (accounted owners) -- all of which fire inside the
      torchlens ``wrapped_hook`` frame and are represented in the trace by a
      single functionless ``intervention_replacement`` placeholder. That activity
      is excused PER-EVENT via the witness ``in_replacement_hook`` flag: an
      orphaned owner tagged ``in_replacement_hook`` is removed from the dispatch
      census, and an unowned dispatch tagged ``in_replacement_hook`` is not
      counted as a drop. The exemption is scoped to the EXACT ops attributable to
      the replacement, so an UNRELATED real drop alongside a genuine replacement
      (its dispatch fires OUTSIDE the hook) still leaves a residual and fails.

    Parameters
    ----------
    trace:
        Disposable validation-capture trace carrying the dispatcher census.

    Returns
    -------
    tuple[int, int]
        ``(dispatch_census_count, captured_census_count)``.
    """

    layer_list = getattr(trace, "layer_list", ())
    decompositions = getattr(trace, "completeness_decompositions", ())
    diagnostics = getattr(trace, "completeness_diagnostics", ())

    accounted_owner_fcids: set[int] = {
        entry.get("owner_func_call_id")
        for entry in decompositions
        if entry.get("capture_accounted") is True
        and isinstance(entry.get("owner_func_call_id"), int)
    }
    # Orphaned accounted owners whose aten dispatch fired inside a genuine raw
    # replacement hook: their op was orphaned out of the final trace ON PURPOSE
    # (its only consumer is the untraceable replacement tensor). Only these are
    # excused; an orphaned owner outside a replacement hook is a real silent drop.
    replacement_hook_owner_fcids: set[int] = {
        entry.get("owner_func_call_id")
        for entry in decompositions
        if entry.get("in_replacement_hook") is True
        and isinstance(entry.get("owner_func_call_id"), int)
    }
    captured_fcids: set[int] = {
        op.func_call_id for op in layer_list if isinstance(getattr(op, "func_call_id", None), int)
    }

    # Captured census: captured ops that own an accounted aten dispatch. A captured
    # op with no owned dispatch (transform boundary OR benign no-op such as
    # same-shape broadcast_tensors) has no dispatch counterpart by design and is
    # excluded -- it is not an "extra captured op" mismatch.
    captured_census = captured_fcids & accounted_owner_fcids

    # Dispatch census: accounted owners, minus orphaned owners excused as genuine
    # replacement construction, plus every unowned aten dispatch that fired outside
    # a replacement hook (each such dispatch is a real silent capture drop).
    orphaned_owner_fcids = accounted_owner_fcids - captured_fcids
    excused_orphan_fcids = orphaned_owner_fcids & replacement_hook_owner_fcids
    unowned_gap_dispatch_count = sum(
        1
        for entry in diagnostics
        if _is_unowned_gap_diagnostic(entry) and entry.get("in_replacement_hook") is not True
    )
    # An owner_not_captured dispatch that MUTATES an argument is a value-affecting
    # capture drop, distinct from benign pure-read equal/allclose control flow
    # (which never mutates). It is added to the census so the backstop fails --
    # the sound strengthening for the round-3 hidden-mutation hunt. Benign
    # owner_not_captured entries have mutates != True and are still excluded.
    owner_not_captured_mutation_count = sum(
        1
        for entry in diagnostics
        if entry.get("reason") == "owner_not_captured"
        and entry.get("mutates") is True
        and entry.get("in_replacement_hook") is not True
        # A module-forward-owned entry was already counted as an unowned gap above.
        and not _is_unowned_gap_diagnostic(entry)
    )
    dispatch_census_count = (
        len(accounted_owner_fcids)
        - len(excused_orphan_fcids)
        + unowned_gap_dispatch_count
        + owner_not_captured_mutation_count
    )

    # Dispatchable ops that were CAPTURED and then INTENTIONALLY orphan-pruned
    # (dead computation that never reaches an output -- see
    # ``postprocess.graph_traversal._remove_orphan_nodes``). These emitted a real
    # aten dispatch (so the dispatch census counts them among ``accounted_owner_fcids``)
    # but were legitimately removed from the final graph, so they are NOT captured
    # ops. Counting them apples-to-apples with the census: keep ONLY orphan-pruned
    # ids that are accounted owners (the same "dispatchable" class the census counts),
    # and exclude replacement-hook orphans already subtracted from the dispatch
    # census above so they are not double-counted. An accounted owner missing from
    # the final trace that was NOT recorded as orphan-pruned stays unaccounted and
    # still trips the backstop -- a genuine silent drop is not masked.
    orphan_pruned_fcids: set[int] = {
        fcid
        for fcid in getattr(trace, "_orphan_pruned_func_call_ids", ()) or ()
        if isinstance(fcid, int)
    }
    pruned_dispatchable = (orphan_pruned_fcids & accounted_owner_fcids) - excused_orphan_fcids
    try:
        trace._validation_pruned_dispatchable_op_count = len(pruned_dispatchable)
    except (AttributeError, TypeError):
        pass

    # Buffer-WRITE accessor dispatches: the ``aten.detach``/``aten.alias`` a registered
    # buffer's ``.data`` property getter emits during the ``self.b.data.copy_(x)`` write idiom.
    # ``.data`` is a C-level tensor property (not a wrapped torch function), so the detach has
    # NO python-wrapper owner and is counted above as an ``unowned_dispatch`` in
    # ``unowned_gap_dispatch_count``. The write itself (``copy_``) IS captured; only this
    # accessor view is legitimately uncaptured -- a THIRD not-in-captured-count category
    # alongside orphan-pruned dead computation. The witness flags each such event
    # (``state_view_accessor``: unowned + non-mutating + pure-view + registered buffer only),
    # so counting them here is apples-to-apples with the census: it is a strict SUBSET of the
    # exact ``unowned_dispatch`` entries already added to the dispatch census, so crediting it
    # in the backstop exactly offsets its own contribution. A genuine untraced dispatch (a
    # value-producing or mutating op, or an unowned dispatch on a non-buffer tensor) is never
    # flagged and stays unaccounted -- the tripwire remains armed.
    buffer_write_dispatch_count = sum(
        1
        for entry in diagnostics
        if entry.get("reason") == "unowned_dispatch"
        and entry.get("state_view_accessor") is True
        and entry.get("in_replacement_hook") is not True
    )
    try:
        trace._validation_buffer_write_dispatch_op_count = buffer_write_dispatch_count
    except (AttributeError, TypeError):
        pass

    return dispatch_census_count, len(captured_census)
