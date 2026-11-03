"""Hook parsing and hook-spec ownership for TorchLens interventions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import torch

from .._errors import InvalidArgumentError
from .._selection.selectors import (
    BackwardPassSelector,
    BaseSelector,
    CompositeSelector,
    FacetSelector,
    GradKindSelector,
    NotSelector,
    label,
)
from ..errors import TorchLensWarning
from ._facet_writes import (
    _check_facet_write_conflicts,
    _facet_fire_scope,
    _facet_home_label,
    _iter_facet_records,
    _scoped_facet_view,
)
from ._hooks_support import (
    _display_label,
    _existing_facet_write_claims,
    _needs_tensor_backward_adapter,
    _selector_call_index,
    _snapshot_layer_log,
    _targets_accumulate_grad,
    _validate_hook_signature,
    gradient_inner,
    make_live_site_proxy,
)
from .errors import (
    HelperMountError,
    HookSignatureError,
    HookSiteCoverageError,
    LiveModeLabelError,
    ReplayPreconditionError,
    SiteResolutionError,
)
from .types import (
    Edit,
    FrozenTargetSpec,
    HookSpec,
    InterventionSpec,
    TargetSpec,
    TargetValueSpec,
)

HookTiming: TypeAlias = Literal["pre", "post"]
HookDirection: TypeAlias = Literal["forward", "backward"]
#: Which tensor a site addresses (SPEC 1.4): the activation of a forward op,
#: the gradient of that op's output (``tl.grad(site)``), or an autograd node's
#: ``grad_input`` tuple (``tl.grad_fn(...)`` and the other node selectors).
SiteUniverse: TypeAlias = Literal["activation", "gradient", "node"]
_UNIVERSE_SIGNAL: dict[str, HookDirection] = {
    "activation": "forward",
    "gradient": "backward",
    "node": "backward",
}
#: Selector kinds that address autograd nodes (the node universe). Kept equal
#: to the node bucket of ``selectors.classify_selector_universe``.
_NODE_SELECTOR_KINDS = frozenset(
    {"grad_fn", "grad_fn_label", "grad_kind", "backward_pass", "intervening", "without_op"}
)
HookCallable: TypeAlias = Callable[..., Any]
HookInput: TypeAlias = Callable[..., Any] | Edit
_DEFAULT_HEAD_FACET_NAMES = ("q", "k", "v")


@dataclass(frozen=True, slots=True)
class HookContext:
    """Frozen metadata snapshot passed to hook callables.

    Parameters
    ----------
    name:
        Display name of the hook.
    timing:
        Hook timing, either ``"pre"`` or ``"post"``.
    signal:
        ``"forward"`` for an activation fire or ``"backward"`` for a gradient
        fire; one callable mounted on ``tl.grad(s) | s`` reads it to tell which.
    layer_log:
        Mapping proxy over selected layer metadata, never a live Op.
    state:
        Per resolved ``(site, hook instance)`` scratch dictionary; it persists
        across fires of the same hook at the same site.
    run_state:
        Mutable dictionary shared across all hooks in a run.
    step_index:
        Trace index for bundle operations.
    trace_name:
        Trace name for bundle operations.
    args:
        Frozen positional tensor arguments at the hook site.
    kwargs:
        Mapping proxy of tensor keyword arguments at the hook site.
    """

    name: str
    timing: HookTiming
    signal: HookDirection
    op: Mapping[str, Any]
    state: dict[str, Any]
    run_state: dict[str, Any]
    step_index: int
    trace_name: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NormalizedHookEntry:
    """Uniform representation of one requested hook attachment."""

    site_target: Any
    normalized_callable: HookCallable
    helper_spec: Edit | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def signal(self) -> HookDirection:
        """Return the pass this entry fires in, derived from its site branch.

        Returns
        -------
        HookDirection
            ``"forward"`` for an activation branch, ``"backward"`` for a
            gradient or node branch.
        """

        value = self.metadata.get("signal", "forward")
        return "backward" if value == "backward" else "forward"

    @property
    def universe(self) -> SiteUniverse:
        """Return the site universe this entry mounts in.

        Returns
        -------
        SiteUniverse
            ``"activation"``, ``"gradient"`` or ``"node"``.
        """

        value = self.metadata.get("universe")
        if value in {"activation", "gradient", "node"}:
            return cast(SiteUniverse, value)
        return "node" if self.signal == "backward" else "activation"


def make_hook_context(
    *,
    name: str,
    timing: HookTiming = "post",
    signal: HookDirection = "forward",
    op: Any | None = None,
    state: dict[str, Any] | None = None,
    run_context: dict[str, Any] | None = None,
    step_index: int = 0,
    trace_name: str = "",
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> HookContext:
    """Build a frozen hook context with mapping-proxy snapshots.

    Parameters
    ----------
    name:
        Display name of the hook.
    timing:
        Hook timing.
    signal:
        Hook signal, ``"forward"`` or ``"backward"``.
    op:
        Optional Op-like object or mapping to snapshot.
    state:
        Per-(site, hook) scratch dictionary. A new dictionary is created when
        omitted.
    run_context:
        Shared run dictionary. A new dictionary is created when omitted.
    step_index:
        Trace index.
    trace_name:
        Trace name.
    args:
        Positional tensor args at this site.
    kwargs:
        Keyword tensor args at this site.

    Returns
    -------
    HookContext
        Frozen hook context object.
    """

    return HookContext(
        name=name,
        timing=timing,
        signal=signal,
        op=MappingProxyType(_snapshot_layer_log(op)),
        state={} if state is None else state,
        run_state={} if run_context is None else run_context,
        step_index=step_index,
        trace_name=trace_name,
        args=tuple(args),
        kwargs=MappingProxyType(dict(kwargs or {})),
    )


def normalize_hook(fn: HookInput) -> HookCallable:
    """Normalize one callable or helper spec into a tensor hook callable.

    The protocol is ``fn(tensor, *, ctx)`` whichever tensor the site names
    (an activation or an output gradient); the site, not the hook, decides
    the pass.

    Parameters
    ----------
    fn:
        User callable or helper spec.

    Returns
    -------
    HookCallable
        Signature-validated hook callable.
    """

    return _normalize_hook_for_signal(fn, signal="forward")


def _normalize_hook_for_signal(fn: HookInput, *, signal: HookDirection) -> HookCallable:
    """Normalize one callable or helper spec for the pass it mounts in.

    Parameters
    ----------
    fn:
        User callable or helper spec.
    signal:
        Pass of the site branch; a backward branch also admits the internal
        ``grad_input`` tuple callback protocol of node-universe observers.

    Returns
    -------
    HookCallable
        Signature-validated hook callable.
    """

    helper_spec: Edit | None = fn if isinstance(fn, Edit) else None
    hook_callable = helper_spec() if helper_spec is not None else fn
    if not callable(hook_callable):
        raise HookSignatureError(f"hook object {hook_callable!r} is not callable")
    _validate_hook_signature(hook_callable, signal=signal)
    return cast(HookCallable, hook_callable)


def normalize_hook_plan(
    hooks_or_site: Any,
    hook: HookInput | None = None,
    *,
    default_site_target: Any | None = None,
    allow_replay_only_site_targets: bool = False,
) -> list[NormalizedHookEntry]:
    """Normalize all supported attach-hook shapes into hook-plan entries.

    The pass each entry fires in comes from its site: a bare site is an
    activation (forward), ``tl.grad(site)`` is that op's output gradient
    (backward) and an autograd-node selector is a node's ``grad_input``
    (backward). A union such as ``tl.grad(s) | s`` yields one entry per
    branch, each mounted and recorded separately.

    Parameters
    ----------
    hooks_or_site:
        Callable/helper, mapping, list-of-tuples, or site target.
    hook:
        Optional hook for the ``(site, hook)`` shape.
    default_site_target:
        Site target for bare callable/helper input. Without this, bare hooks
        fail closed because they do not imply a model-log site.
    allow_replay_only_site_targets:
        Whether replay-capable structural kinds (``site``) may pass.

    Returns
    -------
    list[NormalizedHookEntry]
        Uniform hook plan in left-to-right composition order.

    Raises
    ------
    InvalidArgumentError
        ``edit_signal_mismatch`` when an edit that reads recorded forward
        state is addressed to a gradient site. Raised for the whole plan
        before any entry is returned, so no caller mutates anything first.
    """

    pairs = _dispatch_hook_pairs(hooks_or_site, hook, default_site_target=default_site_target)
    entries: list[NormalizedHookEntry] = []
    for order, (raw_site_target, hook_like) in enumerate(pairs):
        site_target = lower_record_site_target(raw_site_target)
        helper_spec = hook_like if isinstance(hook_like, Edit) else None
        # The escape hatch is requested by the helper's own allow_shape_change
        # kwarg (e.g. tl.replace_with(t, allow_shape_change=True)); it rides
        # entry metadata so a helper-requested shape change reaches
        # validate_hook_output, mirroring the append-safety check in rerun.py.
        entry_force_shape_change = helper_spec is not None and bool(
            dict(helper_spec.kwargs).get("allow_shape_change", False)
        )
        for universe, branch_target in split_site_branches(site_target):
            for live_target in (
                gradient_inners(branch_target) if universe == "gradient" else (branch_target,)
            ):
                _validate_live_site_target(
                    live_target, allow_replay_only=allow_replay_only_site_targets
                )
            _validate_helper_mount(branch_target, helper_spec, universe=universe)
            entries.append(
                NormalizedHookEntry(
                    site_target=branch_target,
                    normalized_callable=_normalize_branch_hook(hook_like, universe=universe),
                    helper_spec=helper_spec,
                    metadata=_plan_entry_metadata(
                        order, allow_shape_change=entry_force_shape_change, universe=universe
                    ),
                )
            )
    return entries


def _plan_entry_metadata(
    order: int, *, allow_shape_change: bool, universe: str
) -> MappingProxyType[str, Any]:
    """Return the frozen metadata of one attach-hook plan entry.

    Parameters
    ----------
    order:
        Zero-based attach order of the entry's ``(site, hook)`` pair.
    allow_shape_change:
        Whether the entry's helper requested ``allow_shape_change``.
    universe:
        Site universe of the entry's branch.

    Returns
    -------
    MappingProxyType[str, Any]
        Read-only entry metadata.
    """

    return MappingProxyType(
        {
            "attach_order": order,
            "composition": "left_to_right",
            "allow_shape_change": allow_shape_change,
            "signal": _UNIVERSE_SIGNAL[universe],
            "universe": universe,
            "timing": "post",
        }
    )


def gradient_inners(site_target: Any) -> tuple[Any, ...]:
    """Return the forward site of every ``tl.grad(...)`` leaf of a gradient branch.

    Parameters
    ----------
    site_target:
        Gradient-branch target: ``tl.grad(s)``, a union of them, or one joined
        to ``tl.in_backward_pass(n)``.

    Returns
    -------
    tuple[Any, ...]
        Inner forward selectors in tree order.
    """

    inner = gradient_inner(site_target)
    if inner is not None:
        return (inner,)
    _operator, children = _target_operands(site_target)
    return tuple(found for child in children for found in gradient_inners(child))


def _target_operands(site_target: Any) -> tuple[str | None, tuple[Any, ...]]:
    """Return the connective and operands of a live or stored composite target.

    A live ``CompositeSelector`` / ``NotSelector`` and the ``TargetSpec`` /
    ``FrozenTargetSpec`` a recipe stores for it (kind ``"and"``, ``"or"`` or
    ``"not"``, operands in ``selector_value``) read the same way, so a sticky
    rule re-normalised from its stored spec keeps the branch it was attached
    in.

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    tuple[str | None, tuple[Any, ...]]
        ``("and" | "or" | "not", operands)``, or ``(None, ())`` for a leaf.
    """

    if isinstance(site_target, CompositeSelector):
        return site_target.operator, tuple(site_target.selectors)
    if isinstance(site_target, NotSelector):
        return "not", (site_target.selector,)
    if not isinstance(site_target, (TargetSpec, FrozenTargetSpec)):
        return None, ()
    kind = site_target.selector_kind
    value = site_target.selector_value
    if kind in ("and", "or") and isinstance(value, (tuple, list)):
        return kind, tuple(value)
    if kind == "not" and value is not None:
        return "not", (value,)
    return None, ()


def site_universe(site_target: Any) -> SiteUniverse:
    """Return the universe of one union operand (SPEC 1.4).

    A ``&`` subtree that mixes universes refuses at construction, so the
    first gradient or node leaf decides; ``tl.in_backward_pass(n)`` joined to
    a gradient address is a qualifier of that gradient branch.

    Parameters
    ----------
    site_target:
        Selector-like target with no top-level union.

    Returns
    -------
    SiteUniverse
        ``"gradient"``, ``"node"`` or ``"activation"``.
    """

    if gradient_inner(site_target) is not None:
        return "gradient"
    operator, children = _target_operands(site_target)
    if operator is not None:
        universes = {site_universe(child) for child in children}
        if "gradient" in universes:
            return "gradient"
        return "node" if "node" in universes else "activation"
    if not isinstance(site_target, (str, TargetSpec, FrozenTargetSpec, BaseSelector)):
        return "activation"
    kind = getattr(site_target, "selector_kind", None)
    return "node" if kind in _NODE_SELECTOR_KINDS else "activation"


def split_site_branches(site_target: Any) -> tuple[tuple[SiteUniverse, Any], ...]:
    """Split a site into one target per universe, in first-appearance order.

    ``tl.grad(a) | tl.grad(b)`` stays one gradient branch; ``tl.grad(s) | s``
    becomes a gradient branch and an activation branch, so both halves
    resolve, mount and record separately (SPEC 1.5, 1.6).

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    tuple[tuple[SiteUniverse, Any], ...]
        ``(universe, target)`` per branch.
    """

    operator, operands = _target_operands(site_target)
    if operator != "or":
        return ((site_universe(site_target), site_target),)
    groups: dict[SiteUniverse, list[Any]] = {}
    for child in operands:
        groups.setdefault(site_universe(child), []).append(child)
    if len(groups) == 1:
        return ((next(iter(groups)), site_target),)
    stored = not isinstance(site_target, CompositeSelector)

    def _rejoin(children: list[Any]) -> Any:
        """Rejoin one universe's operands in the input's own form."""

        if len(children) == 1:
            return children[0]
        if stored:
            return type(site_target)(selector_kind="or", selector_value=tuple(children))
        return CompositeSelector("or", tuple(children))

    return tuple((universe, _rejoin(children)) for universe, children in groups.items())


def lower_record_site_target(site_target: Any) -> Any:
    """Lower an ``Op``/``Layer`` RECORD used as a hook site to an exact selector.

    A record is the most precise site spelling a user can hold (``do(
    log["relu_1_2:2"], edit)``), yet the post-hoc resolver only knew
    selector-likes and refused it ("Unsupported site query"), while the
    spec lowering would have dropped an ``Op`` to its BARE ``layer_label``
    -- the last pass of a multi-pass layer -- had it been accepted
    (AUD-CODE 4.10). Lower here, at plan normalization, so every door
    downstream (validation, target spec, replay hook targets) sees the
    PASS-QUALIFIED ``Op.label`` of each addressed pass: one ``Op`` record
    is exactly its own pass; a ``Layer`` record is the explicit ALL-passes
    spelling (a composite over every pass label), mirroring
    ``Layer.__selection__()``. Selector-likes and everything else pass
    through unchanged.
    """

    labels = _record_site_pass_labels(site_target)
    if labels is None:
        return site_target
    if len(labels) == 1:
        return label(labels[0])
    return CompositeSelector("or", tuple(label(name) for name in labels))


def _record_site_pass_labels(site_target: Any) -> list[str] | None:
    """Return the distinct pass-qualified ``Op.label``s an ``Op``/``Layer`` record addresses.

    ``None`` means ``site_target`` is not a lowerable record (a selector-like,
    a string, an object without ``layer_label``/iterable ``ops``, an empty
    record, or one whose ops lack a string label) and must pass through
    :func:`lower_record_site_target` unchanged.
    """

    if isinstance(site_target, (BaseSelector, TargetSpec, FrozenTargetSpec, str)) or not hasattr(
        site_target, "layer_label"
    ):
        return None
    from .._data_classes.op import Op

    # ``Layer.ops`` is an accessor (iterable, not a builtin sequence); an ``Op`` is itself.
    ops = (site_target,) if isinstance(site_target, Op) else getattr(site_target, "ops", None)
    if ops is None or isinstance(ops, (str, bytes)):
        return None
    try:
        op_records = list(ops)
    except TypeError:
        return None
    labels: list[str] = []
    for op in op_records:
        op_label = getattr(op, "label", None)
        if not isinstance(op_label, str) or not op_label:
            return None
        if op_label not in labels:
            labels.append(op_label)
    return labels or None


def _validate_live_site_target(site_target: Any, *, allow_replay_only: bool = False) -> None:
    """Reject selector targets that cannot be live hook application sites.

    Refusal happens upfront, before any per-site evaluation, so a capability
    error can never hide behind a short-circuiting non-matching sibling.
    Facet targets pass through: the mutator path expands them to home-op
    label targets before matching.

    Parameters
    ----------
    site_target:
        Selector-like target passed to live hook attachment.
    allow_replay_only:
        Whether replay-capable structural kinds (``site``, the C03
        structural selector) may pass: the trace-side attach door resolves
        them post hoc on the finished graph, and the RERUN preflight
        (:func:`normalize_hooks_from_spec`) re-validates without this
        allowance, so a sticky site hook still refuses typed BEFORE any
        rerun forward.
    """

    selector = _normalize_live_selector(site_target)
    _reject_live_incapable_selector(selector, allow_replay_only=allow_replay_only)


def _reject_live_incapable_selector(
    selector: BaseSelector, *, allow_replay_only: bool = False
) -> None:
    """Raise for live-incapable selector kinds nested in a live hook target.

    Parameters
    ----------
    selector:
        Normalized selector to inspect.
    allow_replay_only:
        Whether the replay-capable ``site`` kind may pass (attach-door
        carve-out; the rerun preflight never sets it).
    """
    from .._ir.selector_eval import _UPFRONT_UNSUPPORTED_KINDS, _capability_error

    kind = str(selector.selector_kind)
    if kind in _UPFRONT_UNSUPPORTED_KINDS["live"] and not (allow_replay_only and kind == "site"):
        raise _capability_error(kind, "live")
    if isinstance(selector, CompositeSelector):
        for child in selector.selectors:
            _reject_live_incapable_selector(
                _normalize_live_selector(child), allow_replay_only=allow_replay_only
            )
    if isinstance(selector, NotSelector):
        _reject_live_incapable_selector(
            _normalize_live_selector(selector.selector), allow_replay_only=allow_replay_only
        )


def _normalize_branch_hook(hook_like: HookInput, *, universe: SiteUniverse) -> HookCallable:
    """Normalize a hook for the site branch it mounts in.

    An activation or gradient branch takes the tensor protocol
    ``fn(tensor, *, ctx)``; an observer's ``record_backward`` takes over on a
    backward branch. A node branch adapts a tensor hook to the autograd
    node's ``grad_input`` tuple callback.

    Parameters
    ----------
    hook_like:
        User callable, observer, or helper spec.
    universe:
        Universe of the site branch.

    Returns
    -------
    HookCallable
        Branch-specific hook callable.
    """

    signal = _UNIVERSE_SIGNAL[universe]
    if signal == "backward" and hasattr(hook_like, "record_backward"):
        return _normalize_hook_for_signal(getattr(hook_like, "record_backward"), signal=signal)
    helper_spec: Edit | None = hook_like if isinstance(hook_like, Edit) else None
    hook_callable = _normalize_hook_for_signal(hook_like, signal=signal)
    if universe == "node" and _needs_tensor_backward_adapter(hook_callable, helper_spec):
        return _adapt_tensor_backward_hook(hook_callable, helper_spec)
    return hook_callable


def _backward_context_op(grad_fn_handle: Any) -> Any:
    """Return the forward record a live backward hook context describes.

    ``GradFn.op`` names the paired layer, which on a multi-pass layer is the
    aggregate ``Layer`` whose per-pass fields refuse; the context snapshots
    per-pass fields, so it describes the one pass this GradFn differentiates.

    Parameters
    ----------
    grad_fn_handle:
        GradFn receiving the backward callback.

    Returns
    -------
    Any
        The paired pass-level op, the GradFn's op, or ``grad_fn_handle`` itself
        when it carries no ``op``.
    """
    from .._ir._selector_universe import _grad_fn_forward_op

    if not hasattr(grad_fn_handle, "op"):
        return grad_fn_handle
    return _grad_fn_forward_op(grad_fn_handle)


def _adapt_tensor_backward_hook(
    hook_callable: HookCallable,
    helper_spec: Edit | None,
) -> HookCallable:
    """Wrap a tensor-gradient hook for PyTorch ``GradFn`` tuple callbacks.

    Parameters
    ----------
    hook_callable:
        Callable accepting one gradient tensor and ``ctx=HookContext``.
    helper_spec:
        Helper spec used for display metadata.

    Returns
    -------
    HookCallable
        Callable accepting ``grad_input`` tuple callback arguments.
    """

    helper_name = helper_spec.name if helper_spec is not None else "user_backward_hook"

    def _tuple_hook(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> tuple[torch.Tensor | None, ...] | None:
        """Apply a tensor hook independently to each non-None grad_input slot."""

        updated: list[torch.Tensor | None] = []
        changed = False
        for index, grad in enumerate(grad_input):
            if grad is None:
                updated.append(None)
                continue
            context = make_hook_context(
                name=helper_name,
                signal="backward",
                op=_backward_context_op(grad_fn_handle),
                run_context=run_ctx,
                args=(grad,),
                kwargs={
                    "grad_output": grad_output,
                    "grad_fn_handle": grad_fn_handle,
                    "call_index": call_index,
                    "tuple_index": index,
                },
            )
            result = hook_callable(grad, ctx=context)
            if result is None:
                updated.append(grad)
                continue
            if not isinstance(result, torch.Tensor):
                raise HookSignatureError(
                    f"backward tensor hook {helper_name!r} returned "
                    f"{type(result).__name__}; expected torch.Tensor or None"
                )
            updated.append(result)
            changed = changed or result is not grad
        return tuple(updated) if changed else None

    return cast(HookCallable, _tuple_hook)


def normalize_hooks_from_spec(
    spec: InterventionSpec | None,
    *,
    live_forward: bool = True,
) -> list[NormalizedHookEntry]:
    """Normalize an intervention spec into live-capture hook-plan entries.

    Parameters
    ----------
    spec:
        Mutable intervention spec attached to a ``Trace``. ``None`` or an
        empty spec produces no hook entries.
    live_forward:
        Whether the plan will fire against LIVE mid-forward proxies (the
        rerun/backward engines). Live-forward plans preflight every entry
        against the live capability set here, before any forward runs, so a
        sticky replay-only target (a ``site`` selector staged through the
        attach door) refuses typed at the door instead of mid-forward. The
        replay engine matches against finished ops and skips the live
        preflight.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook entries suitable for ``_state._active_hook_plan`` during rerun.
    """

    if spec is None:
        return []

    replay_ok = not live_forward
    entries: list[NormalizedHookEntry] = []
    entries.extend(_normalize_value_specs(spec.target_value_specs))
    entries.extend(
        _normalize_sticky_hook_specs(spec.hook_specs, allow_replay_only_site_targets=replay_ok)
    )

    if spec.targets:
        hook_like = _hook_like_from_spec(spec)
        if hook_like is not None:
            for target in spec.targets:
                entries.extend(
                    normalize_hook_plan(target, hook_like, allow_replay_only_site_targets=replay_ok)
                )
    if live_forward:
        # RERUN/backward preflight: sticky entries staged through the attach
        # door's replay-only carve-out (site selectors) must refuse HERE,
        # before any forward runs -- a mid-forward capability error is a
        # late, ugly refusal.
        for entry in entries:
            live_targets = (
                gradient_inners(entry.site_target)
                if entry.universe == "gradient"
                else (entry.site_target,)
            )
            for live_target in live_targets:
                _validate_live_site_target(live_target)
    return entries


def is_facet_target(site_target: Any) -> bool:
    """Return whether a site target is a semantic facet selector.

    Parameters
    ----------
    site_target:
        Selector-like site target.

    Returns
    -------
    bool
        Whether the target should be expanded before storage.
    """

    return _facet_selector_from_target(site_target) is not None


def _fused_attention_write_remedy(facet_name: str, *, reconstructed: bool = True) -> str:
    """Explain why an attention-interior facet cannot be written, and what can.

    Parameters
    ----------
    facet_name:
        The requested facet (``pattern``, ``scores``, ``z`` or ``result``).
    reconstructed:
        Whether the facet is rebuilt from a fused kernel's saved arguments
        (fused attention). ``False`` means a facet computed from captured
        values (``result``, ``z_h @ W_O_h``), read-only on eager attention too.

    Returns
    -------
    str
        The cause and the remedies, for the refusal message.
    """

    if not reconstructed:
        return (
            f"The {facet_name!r} facet is computed from other captured values (for "
            "'result', each head's 'z' times its slice of the output projection), so no "
            "captured op holds it and it is read-only under every attention "
            "implementation. Write the facet it is computed from instead: replacing a "
            "head's 'z' replaces exactly its 'result'."
        )
    return (
        f"Under fused attention (torch scaled_dot_product_attention, the transformers "
        f"default attn_implementation='sdpa') the {facet_name!r} facet is reconstructed "
        "read-only from the kernel's saved arguments: the kernel computes it internally, "
        "so there is no captured tensor op to write. To write it, load the model with "
        "attn_implementation='eager' (on a transformers model, "
        "model.set_attn_implementation('eager')) and capture again. On the fused model "
        "the 'q'/'k'/'v' facets (the projection outputs feeding the kernel) stay "
        "writable, and tl.mechinterp.patch_head lowers a single head or pattern patch "
        "exactly to the real output projection."
    )


def _facet_write_mask(selector: FacetSelector, spec: Any) -> torch.Tensor:
    """Return a detached copy of ``spec``'s write mask.

    A read-only attention facet (fused attention, or a reconstructed facet)
    raises ``RuntimeError`` from ``write_mask``; that becomes a
    ``SiteResolutionError`` naming the eager-attention remedy.
    """

    try:
        return spec.write_mask().detach().clone()
    except RuntimeError as exc:
        if selector.name in {"pattern", "scores", "z", "result"}:
            raise SiteResolutionError(
                f"facet selector {selector!r} resolved only read-only "
                "attention facets. "
                + _fused_attention_write_remedy(
                    selector.name, reconstructed=spec.capability_flags.reconstructed
                )
            ) from exc
        raise


def expand_facet_hook_entries(
    log: Any, entries: Sequence[NormalizedHookEntry]
) -> list[NormalizedHookEntry]:
    """Expand facet hook entries into ordinary home-op hook entries.

    Parameters
    ----------
    log:
        Trace that owns the facet registry snapshot and captured home ops.
    entries:
        Normalized hook entries, possibly including ``tl.facet`` or
        ``tl.head`` targets.

    Returns
    -------
    list[NormalizedHookEntry]
        Entries whose facet targets have been replaced by home-op label
        targets and slice-aware wrapper hooks.
    """

    expanded: list[NormalizedHookEntry] = []
    claimed = _existing_facet_write_claims(log)
    for entry in entries:
        selector = _facet_selector_from_target(entry.site_target)
        if selector is None:
            expanded.append(entry)
            continue
        specs = _resolve_facet_specs(log, selector)
        if not specs:
            if selector.name in {"pattern", "scores", "z", "result"}:
                raise SiteResolutionError(
                    f"facet selector {selector!r} matched no write-capable real tensor facets. "
                    + _fused_attention_write_remedy(selector.name)
                )
            raise SiteResolutionError(
                f"facet selector {selector!r} matched 0 write-capable facets."
            )
        written: list[tuple[str, str]] = []
        for facet_name, spec in specs:
            mask = _facet_write_mask(selector, spec)
            home_label = _facet_home_label(spec)
            _check_facet_write_conflicts(claimed, home_label=home_label, mask=mask)
            claimed.append((home_label, mask, facet_name))
            written.append((home_label, facet_name))
            metadata = {
                **entry.metadata,
                "facet_write": True,
                "facet_name": facet_name,
                "facet_home_label": home_label,
                "facet_write_mask": mask,
            }
            expanded.append(
                NormalizedHookEntry(
                    site_target=TargetSpec("label", home_label),
                    normalized_callable=_facet_scatter_hook(
                        entry.normalized_callable,
                        spec,
                        facet_name=facet_name,
                        selector=FacetSelector(
                            facet_name,
                            head_index=selector.head_index,
                            module_address=selector.module_address,
                        ),
                    ),
                    helper_spec=entry.helper_spec,
                    metadata=MappingProxyType(metadata),
                )
            )
        _warn_facet_fanout(log, selector, written)
    return expanded


def _warn_facet_fanout(log: Any, selector: FacetSelector, written: list[tuple[str, str]]) -> None:
    """Say how many sites a facet selector writes when it writes more than one.

    Coded ``facet_selector_fanout``; ``fields["site_count"]`` carries the number.

    ``tl.head(5)`` keeps its broadcast meaning: it writes head 5 of every
    matching facet in every module. The count is the number of installed
    facet writes (one per home op and facet slice, after de-duplication), so it
    equals the slices the run edits.

    Parameters
    ----------
    log:
        Trace whose modules own the facet views.
    selector:
        The facet selector as the caller spelled it.
    written:
        ``(home op label, facet name)`` for every installed write.
    """

    if len(written) <= 1:
        return
    import warnings

    homes = list(dict.fromkeys(_display_label(log, home) for home, _ in written))
    facet_names = ", ".join(dict.fromkeys(name for _, name in written))
    shown = ", ".join(homes[:4]) + (f" and {len(homes) - 4} more" if len(homes) > 4 else "")
    scope = _facet_owner_address(log, selector, homes[0], written[0][1])
    narrower = (
        f"{selector!r}.in_module({scope!r})"
        if scope is not None and selector.module_address is None
        else "a narrower facet selector"
    )
    warnings.warn(
        TorchLensWarning(
            f"selector {selector!r} writes {len(written)} sites ({facet_names} at "
            f"{len(homes)} home ops: {shown}). Remedy: to write fewer, scope it, "
            f"e.g. {narrower}",
            code="facet_selector_fanout",
            site_count=len(written),
        ),
        stacklevel=5,
    )


def _facet_owner_address(log: Any, selector: FacetSelector, home_label: str, facet: str) -> Any:
    """Return the innermost module address owning ``facet`` at ``home_label``, if any."""

    try:
        home = log._layer_logs[home_label] if home_label in log._layer_logs else log[home_label]
        modules = getattr(log, "modules", None)
        for call_label in reversed(list(getattr(home, "modules", ()) or ())):
            address = str(call_label).rsplit(":", 1)[0]
            view = getattr(modules[address], "facets", None) if modules is not None else None
            if view is not None and view.has(facet):
                return address
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


def _normalize_value_specs(value_specs: Sequence[TargetValueSpec]) -> list[NormalizedHookEntry]:
    """Normalize set-replacement specs into hook-plan entries.

    Parameters
    ----------
    value_specs:
        Mutable set-replacement specs from an intervention recipe.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook-plan entries that replace matching outs.
    """

    entries: list[NormalizedHookEntry] = []
    for order, value_spec in enumerate(value_specs):

        def _set_value_hook(
            out: torch.Tensor,
            *,
            ctx: HookContext,
            replacement: Any = value_spec.value,
        ) -> torch.Tensor:
            """Return a static or one-shot callable replacement.

            Parameters
            ----------
            out:
                Activation at the matched site.
            ctx:
                Hook context supplied by TorchLens.
            replacement:
                Replacement tensor or callable captured by default argument.

            Returns
            -------
            torch.Tensor
                Replacement out.
            """

            del ctx
            if callable(replacement):
                return cast(torch.Tensor, replacement(out))
            return cast(torch.Tensor, replacement)

        metadata = {
            "attach_order": order,
            "composition": "left_to_right",
            "allow_shape_change": False,
            "signal": "forward",
            "universe": "activation",
            "timing": "post",
            **value_spec.metadata,
        }
        entries.append(
            NormalizedHookEntry(
                site_target=value_spec.site_target,
                normalized_callable=_set_value_hook,
                helper_spec=None,
                metadata=MappingProxyType(metadata),
            )
        )
    return entries


def _normalize_sticky_hook_specs(
    hook_specs: Sequence[HookSpec],
    *,
    allow_replay_only_site_targets: bool = False,
) -> list[NormalizedHookEntry]:
    """Normalize sticky hook specs into hook-plan entries.

    Parameters
    ----------
    hook_specs:
        Mutable sticky hook specs from an intervention recipe.
    allow_replay_only_site_targets:
        Whether replay-capable structural targets (``site``) may pass; set
        by the replay-engine caller, never the live-forward engines.

    Returns
    -------
    list[NormalizedHookEntry]
        Hook-plan entries in stored composition order.
    """

    entries: list[NormalizedHookEntry] = []
    for hook_spec in hook_specs:
        if hook_spec.metadata.get("facet_write"):
            # Facet-slice hooks fire through the scatter wrapper built at attach time
            # (_facet_scatter_hook), which closes over the resolved FacetSpec and CANNOT be
            # rebuilt from the raw helper: the generic path below would silently apply the
            # helper to the WHOLE home tensor (tl.head(0, "q") + tl.zero_ablate() zeroing
            # every head). Fail closed when the wrapper is absent (e.g. a spec loaded from disk).
            stored_hook = hook_spec.hook
            if not getattr(stored_hook, "_tl_facet_scatter", False):
                facet_name = hook_spec.metadata.get("facet_name", "<unknown>")
                home_label = hook_spec.metadata.get("facet_home_label", "<unknown>")
                raise ReplayPreconditionError(
                    "Sticky facet-slice hook lost its scatter wrapper: the stored hook "
                    f"for facet {facet_name!r} on home {home_label!r} is not the live "
                    "facet scatter wrapper, so firing it would write the whole home "
                    "tensor instead of the selected facet slice. Facet hooks cannot be "
                    "rebuilt from a serialized helper; re-attach the facet intervention "
                    "on a live trace (e.g. trace.attach_hooks(tl.head(...), helper))."
                )
            entries.append(
                NormalizedHookEntry(
                    site_target=hook_spec.site_target,
                    normalized_callable=cast(HookCallable, stored_hook),
                    helper_spec=hook_spec.helper,
                    metadata=MappingProxyType(dict(hook_spec.metadata)),
                )
            )
            continue
        hook_like = hook_spec.helper if hook_spec.helper is not None else hook_spec.hook
        _refuse_legacy_backward_hook_spec(hook_spec)
        normalized_entries = normalize_hook_plan(
            hook_spec.site_target,
            hook_like,
            allow_replay_only_site_targets=allow_replay_only_site_targets,
        )
        for entry in normalized_entries:
            # The branch decides the pass: the freshly derived signal and
            # universe win over anything the stored metadata carries.
            metadata = {
                **entry.metadata,
                **hook_spec.metadata,
                "signal": entry.metadata["signal"],
                "universe": entry.metadata["universe"],
            }
            entries.append(
                NormalizedHookEntry(
                    site_target=entry.site_target,
                    normalized_callable=entry.normalized_callable,
                    helper_spec=entry.helper_spec,
                    metadata=MappingProxyType(metadata),
                )
            )
    return entries


def _refuse_legacy_backward_hook_spec(hook_spec: HookSpec) -> None:
    """Refuse a stored hook staged backward on an activation site by the old keyword.

    Before gradient sites, ``direction="backward"`` on a bare label meant the
    gradient at that op. That spelling is gone; re-normalizing the stored
    hook from its site would silently turn a gradient edit into an
    activation edit, so it refuses typed instead.

    Parameters
    ----------
    hook_spec:
        Stored sticky hook spec.

    Raises
    ------
    InvalidArgumentError
        ``edit_signal_mismatch`` for a legacy backward hook on an activation site.
    """

    if hook_spec.metadata.get("direction") not in {"backward", "both"}:
        return
    if site_universe(hook_spec.site_target) != "activation":
        return
    site = hook_spec.site_target
    raise InvalidArgumentError(
        f"A stored hook at {site!r} was staged with the removed direction="
        f"{hook_spec.metadata.get('direction')!r} keyword; the site now decides the pass, "
        "and this site is an activation. Nothing was attached.",
        code="edit_signal_mismatch",
        remedy="re-attach the edit on the gradient address: trace.attach_hooks(tl.grad(site), edit)",
        argument="hooks_or_site",
        site=repr(site),
    )


def _hook_like_from_spec(spec: InterventionSpec) -> HookInput | None:
    """Return the callable/helper represented by an intervention spec.

    Parameters
    ----------
    spec:
        Intervention spec to inspect.

    Returns
    -------
    HookInput | None
        Normalizable hook input, or ``None`` when the spec is empty.
    """

    if spec.hook is not None:
        return cast(HookInput, spec.hook)
    if spec.helper is not None:
        return spec.helper
    if spec.value is not None:
        value = spec.value

        def _value_hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
            """Return a fixed replacement value from an intervention spec.

            Parameters
            ----------
            out:
                Activation being replaced.
            ctx:
                Hook context supplied by TorchLens.

            Returns
            -------
            torch.Tensor
                Replacement tensor.
            """

            del out, ctx
            if callable(value):
                return cast(torch.Tensor, value())
            return cast(torch.Tensor, value)

        return _value_hook
    return None


def live_selector_matches_site(selector_like: Any, site: Any) -> bool:
    """Return whether a selector can match one capture-time site.

    Parameters
    ----------
    selector_like:
        Selector or target-spec-like object from a normalized hook entry.
    site:
        Capture-time pass proxy with raw labels and module context populated.

    Returns
    -------
    bool
        Whether the selector matches the site during live capture.

    Raises
    ------
    LiveModeLabelError
        If a finalized-looking ``tl.label(...)`` selector cannot exist yet.
    SiteResolutionError
        If the selector kind is unsupported or a predicate fails.
    """
    from .._ir.selector_eval import (
        evaluate,
        live_label_error_message,
        looks_like_finalized_label,
    )

    selector = _normalize_live_selector(selector_like)
    matched = evaluate(selector, site, lifecycle="live")
    if not matched and selector.selector_kind == "label":
        label_value = str(selector.selector_value)
        if looks_like_finalized_label(label_value):
            raise LiveModeLabelError(
                live_label_error_message(label_value),
                code="label_selector_not_live",
            )
    return matched


def _normalize_live_selector(selector_like: Any) -> BaseSelector:
    """Normalize a live-capture selector input.

    Parameters
    ----------
    selector_like:
        Selector-like object from the hook plan.

    Returns
    -------
    BaseSelector
        Normalized selector.
    """
    from .._ir.selector_eval import normalize_selector_like

    return normalize_selector_like(selector_like, lifecycle="live")


def live_backward_selector_matches(
    selector_like: Any,
    grad_fn_handle: Any,
    call_index: int,
    *,
    grad_input: tuple[torch.Tensor | None, ...] | None = None,
    grad_output: tuple[torch.Tensor | None, ...] | None = None,
) -> bool:
    """Return whether a selector can match one grad_fn_handle callback site.

    Parameters
    ----------
    selector_like:
        Selector or target spec from a normalized hook entry.
    grad_fn_handle:
        GradFn receiving a backward hook callback.
    call_index:
        One-based callback index.
    grad_input:
        Current grad-input tuple, if available.
    grad_output:
        Current grad-output tuple, if available.

    Returns
    -------
    bool
        Whether the selector matches this backward site.
    """

    from .._selection.resolver import _resolve_unchecked

    expected_call_index = _selector_call_index(selector_like)
    if expected_call_index is not None and expected_call_index != call_index:
        return False
    selector = _normalize_live_selector(selector_like)
    if not _live_backward_context_matches(
        selector,
        grad_fn_handle=grad_fn_handle,
        grad_input=grad_input,
        grad_output=grad_output,
    ):
        return False
    return bool(_resolve_unchecked((grad_fn_handle,), selector, strict=False))


def _live_backward_context_matches(
    selector: BaseSelector,
    *,
    grad_fn_handle: Any,
    grad_input: tuple[torch.Tensor | None, ...] | None,
    grad_output: tuple[torch.Tensor | None, ...] | None,
) -> bool:
    """Return whether live-only backward selector facets match this callback."""

    if isinstance(selector, CompositeSelector):
        child_matches = (
            _live_backward_context_matches(
                _normalize_live_selector(child),
                grad_fn_handle=grad_fn_handle,
                grad_input=grad_input,
                grad_output=grad_output,
            )
            for child in selector.selectors
        )
        return all(child_matches) if selector.operator == "and" else any(child_matches)
    if isinstance(selector, NotSelector):
        return not _live_backward_context_matches(
            _normalize_live_selector(selector.selector),
            grad_fn_handle=grad_fn_handle,
            grad_input=grad_input,
            grad_output=grad_output,
        )
    if isinstance(selector, GradKindSelector):
        values = grad_input if selector.grad_kind == "grad_input" else grad_output
        return any(isinstance(value, torch.Tensor) for value in values or ())
    if isinstance(selector, BackwardPassSelector):
        trace = getattr(grad_fn_handle, "trace", None)
        return getattr(trace, "_active_backward_pass_number", None) == selector.pass_index
    return True


def _validate_helper_mount(
    site_target: Any,
    helper_spec: Edit | None,
    *,
    universe: SiteUniverse,
) -> None:
    """Refuse an edit addressed to a site whose tensor it cannot edit.

    Runs at attach time for every branch of the plan, before any mutation
    (SPEC 2.2). A neutral edit (``Edit.signal is None``) is a function of one
    tensor and mounts anywhere; an edit that reads recorded forward state
    (``Edit.signal == "forward"``: ``patch_from``, ``mean_ablate``,
    ``splice_module``, ``swap_with`` from a site) has no meaning on a
    gradient or autograd-node site.

    Parameters
    ----------
    site_target:
        Site target of this branch.
    helper_spec:
        Helper spec, if the hook came from a built-in helper.
    universe:
        Universe of the site branch.

    Raises
    ------
    InvalidArgumentError
        ``edit_signal_mismatch`` for an activation-only edit on a gradient or
        node site.
    """

    if helper_spec is None or universe == "activation":
        return
    if (
        universe == "node"
        and dict(helper_spec.metadata).get("requires_grad_output")
        and _targets_accumulate_grad(site_target)
    ):
        raise HelperMountError(
            f"{helper_spec.name} requires grad_output and cannot mount on AccumulateGrad prehooks."
        )
    if helper_spec.signal != "forward":
        return
    raise InvalidArgumentError(
        f"{helper_spec.name} reads recorded forward activations and cannot edit the "
        f"{'gradient' if universe == 'gradient' else 'autograd-node gradient'} at "
        f"{site_target!r}. Nothing was attached. Remedy: {EDIT_SIGNAL_MISMATCH_REMEDY}",
        code="edit_signal_mismatch",
        remedy=EDIT_SIGNAL_MISMATCH_REMEDY,
        argument="value_or_hook",
        verb=helper_spec.name,
        site=repr(site_target),
    )


#: The one remedy text of ``edit_signal_mismatch`` (SPEC 2.2).
EDIT_SIGNAL_MISMATCH_REMEDY = (
    "this edit reads activations; the address tl.grad(...) is a gradient; use a "
    "neutral edit (tl.scale, tl.zero_ablate, tl.clamp, tl.add_noise, tl.clip_norm or "
    "a callable)"
)


def _facet_selector_from_target(site_target: Any) -> FacetSelector | None:
    """Return a facet selector from a target object when possible.

    Parameters
    ----------
    site_target:
        Selector-like target.

    Returns
    -------
    FacetSelector | None
        Normalized facet selector, or ``None`` for ordinary targets.
    """

    if isinstance(site_target, FacetSelector):
        return site_target
    if isinstance(site_target, TargetSpec) and site_target.selector_kind == "facet":
        return _facet_selector_from_payload(site_target.selector_value)
    return None


def _facet_selector_from_payload(payload: Any) -> FacetSelector:
    """Build a ``FacetSelector`` from stored target payload data.

    Parameters
    ----------
    payload:
        TargetSpec selector payload.

    Returns
    -------
    FacetSelector
        Normalized semantic facet selector.
    """

    if isinstance(payload, Mapping):
        name = payload.get("name")
        head_index = payload.get("head_index")
        module_address = payload.get("module_address")
        return FacetSelector(
            None if name is None else str(name),
            head_index=None if head_index is None else int(head_index),
            module_address=None if module_address is None else str(module_address),
        )
    if isinstance(payload, str):
        return FacetSelector(payload)
    raise SiteResolutionError(f"Unsupported facet selector payload {payload!r}.")


def _resolve_facet_specs(log: Any, selector: FacetSelector) -> list[tuple[str, Any]]:
    """Resolve a semantic facet selector against captured module facet views.

    Parameters
    ----------
    log:
        Trace whose modules should be searched.
    selector:
        Facet selector to resolve.

    Returns
    -------
    list[tuple[str, Any]]
        Pairs of public facet names and ``FacetSpec`` objects.
    """

    from .._semantic.facets import Facet, FacetSpec

    resolved: list[tuple[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    facet_names = _DEFAULT_HEAD_FACET_NAMES if selector.name is None else (selector.name,)
    for record in _iter_facet_records(log):
        view = _scoped_facet_view(record, selector)
        if view is None:
            continue
        for facet_name in facet_names:
            if not view.has(facet_name):
                continue
            try:
                if selector.head_index is None:
                    value = view[facet_name]
                else:
                    value = view.head(selector.head_index)[facet_name]
            except (AttributeError, KeyError, RuntimeError, ValueError):
                continue
            spec = (
                value.spec
                if isinstance(value, Facet)
                else value
                if isinstance(value, FacetSpec)
                else None
            )
            if spec is None:
                continue
            dedupe_key = (id(spec.home), repr(spec.transforms), facet_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            resolved.append((facet_name, spec))
    return resolved


def _facet_scatter_hook(
    user_hook: HookCallable,
    spec: Any,
    *,
    facet_name: str,
    selector: FacetSelector | None = None,
) -> HookCallable:
    """Return a home-op hook that edits one facet slice.

    Parameters
    ----------
    user_hook:
        Normalized user hook that receives the facet slice.
    spec:
        FacetSpec used to read and scatter the slice.
    facet_name:
        Public facet name for diagnostics.
    selector:
        The facet selector narrowed to ``facet_name``; a donor-reading edit
        resolves it on its source trace (defaults to ``tl.facet(facet_name)``).

    Returns
    -------
    HookCallable
        Hook with the standard whole-output TorchLens contract.
    """

    if selector is None:
        selector = FacetSelector(facet_name)

    def _hook(out: torch.Tensor, *, ctx: HookContext) -> torch.Tensor:
        """Apply a user hook to a facet slice and scatter it into ``out``.

        Parameters
        ----------
        out:
            Full home-op output tensor.
        ctx:
            Hook context supplied by TorchLens.

        Returns
        -------
        torch.Tensor
            Full edited home-op output tensor.
        """

        from .runtime import validate_hook_output

        facet_slice = spec.apply_transforms(out)
        with _facet_fire_scope(spec, selector):
            edited_slice = user_hook(facet_slice, ctx=ctx)
        checked_slice = validate_hook_output(edited_slice, facet_slice, hook_context=ctx)
        return spec.scatter_update(out, checked_slice, mode="replace")

    _hook.__name__ = f"facet_{facet_name}_scatter_hook"
    # Marker consumed by _normalize_sticky_hook_specs: a facet_write hook spec whose
    # stored hook lacks this marker has LOST the slice-scatter wrapper and would write
    # the whole home tensor, so replay must refuse instead of firing it.
    _hook._tl_facet_scatter = True  # type: ignore[attr-defined]
    return _hook


def _dispatch_hook_pairs(
    hooks_or_site: Any,
    hook: HookInput | None,
    *,
    default_site_target: Any | None,
) -> list[tuple[Any, HookInput]]:
    """Parse supported hook input shapes into ``(site, hook)`` pairs.

    Parameters
    ----------
    hooks_or_site:
        Attach-hooks input object.
    hook:
        Optional hook paired with a site.
    default_site_target:
        Site target used for bare callable/helper input.

    Returns
    -------
    list[tuple[Any, HookInput]]
        Parsed site/hook pairs.
    """

    if hook is not None:
        if hasattr(hooks_or_site, "to_hook_pairs"):
            return list(hooks_or_site.to_hook_pairs(hook))
        return [(hooks_or_site, hook)]

    observer_site = getattr(hooks_or_site, "site", None)
    if observer_site is not None and _is_hook_like(hooks_or_site):
        return [(observer_site, hooks_or_site)]

    if _is_hook_like(hooks_or_site):
        if default_site_target is None:
            raise HookSiteCoverageError(
                "bare hook input has no site target; pass a site/hook pair or default_site_target"
            )
        return [(default_site_target, hooks_or_site)]

    if isinstance(hooks_or_site, Mapping):
        return [(site, hook_like) for site, hook_like in hooks_or_site.items()]

    if _is_single_pair(hooks_or_site):
        site_target, hook_like = hooks_or_site
        return [(site_target, hook_like)]

    if isinstance(hooks_or_site, Sequence) and not isinstance(hooks_or_site, (str, bytes)):
        pairs = []
        for item in hooks_or_site:
            if not _is_single_pair(item):
                raise HookSignatureError(
                    "hook sequence entries must be explicit (site, callable_or_helper) pairs"
                )
            site_target, hook_like = item
            pairs.append((site_target, hook_like))
        return pairs

    raise HookSignatureError(f"unsupported hook input shape: {type(hooks_or_site).__name__}")


def _is_hook_like(value: Any) -> bool:
    """Return whether an object can be normalized as a hook.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the object is callable or a helper spec.
    """

    return isinstance(value, Edit) or callable(value)


def _is_single_pair(value: Any) -> bool:
    """Return whether ``value`` is an unambiguous ``(site, hook)`` pair.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the object is a two-item tuple ending in a hook-like object.
    """

    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_site_like(value[0])
        and _is_hook_like(value[1])
    )


def _is_site_like(value: Any) -> bool:
    """Return whether ``value`` can name a future hook site.

    Parameters
    ----------
    value:
        Object to test.

    Returns
    -------
    bool
        Whether the value is accepted as a site target by the contract layer.
    """

    return isinstance(value, (BaseSelector, TargetSpec, str)) or hasattr(value, "layer_label")


__all__ = [
    "HookCallable",
    "HookContext",
    "HookDirection",
    "HookTiming",
    "NormalizedHookEntry",
    "expand_facet_hook_entries",
    "is_facet_target",
    "make_hook_context",
    "make_live_site_proxy",
    "live_selector_matches_site",
    "live_backward_selector_matches",
    "normalize_hook",
    "normalize_hook_plan",
]
