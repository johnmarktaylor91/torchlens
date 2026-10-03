"""``spec.bind(model)`` -- the capture-free bound intervention executor (F01).

Surgery memo section 3.3, amended lane brief foldA D12: lane 3 of the ONE-SPEC
design -- "really run, record nothing (~1x; hand it to anything that calls a
model)". The binding pairs one immutable :class:`InterventionSpec` with one
base model, changing neither. One engine underneath: WHERE terms evaluate
through the SAME capture-lifecycle selector evaluator the ``intervene=``
capture lane uses (``spec.match(ctx)`` over genuine ``RecordContext`` objects
built by the shared builder), actions lower through the SAME hook normalizer
(``normalize_hook``), payloads validate through the SAME ``_execute_hook``
gate, structural site keys mint through the SAME streaming live minter, and
every firing mints its record through the ONE ``build_fire_record`` builder.

Mechanism honesty (surgery memo section 4): the runtime intercepts each torch
function call AFTER it executes and replaces the value passed downstream --
the original computation always runs; every firing disclosure carries the
engine-set closed ``execution_effect`` value
``"values_replaced_after_execution"``. Execution removal exists in no lane.

Interception layer disclosure: op-level rules ride a ``TorchFunctionMode``
(the torch-function protocol -- the same Python-API layer the capture
wrappers decorate; interior redispatch is NOT re-intercepted, so op naming is
outermost-call). Module-boundary rules (plain ``tl.module``) ride real
``nn.Module`` forward hooks. In-place op edits substitute the RETURNED
handle; the mutated storage itself is not rewritten in this lane.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from .._errors import InvalidArgumentError
from .audit import EXECUTION_EFFECTS, build_fire_record, rules_payload
from .errors import BindingPreflightError, BindingRuntimeError
from .model_door import resolve_model_operand
from .spec import InterventionSpec

#: Attribute names whose access raises the training-surface teaching refusal
#: (surgery 3.3 item 7): a binding is NOT an ``nn.Module`` and never will be.
_TRAINING_SURFACE_NAMES: frozenset[str] = frozenset(
    {
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "state_dict",
        "load_state_dict",
        "to",
        "train",
        "eval",
        "cuda",
        "cpu",
        "half",
        "float",
        "double",
        "bfloat16",
        "requires_grad_",
        "zero_grad",
        "compile",
        "forward",
        "modules",
        "named_modules",
        "children",
        "named_children",
        "apply",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
    }
)

_ZERO_FIRE_POLICIES: frozenset[str] = frozenset({"error", "disclose"})

#: The engine-set execution-effect value for this lane (foldA D17).
_BIND_EXECUTION_EFFECT = "values_replaced_after_execution"


@dataclass(frozen=True)
class BindReport:
    """The out-of-band ledger for ONE bound call (surgery 3.3 item 3).

    Same shape class as the intervention transaction envelope with
    ``trace=None`` -- spec and model stamps, resolved static targets, per-rule
    fire counts, zero-fire rules, errors, and the cleanup verdict. No per-op
    claims are made: nothing was captured.
    """

    schema: str = "bind_report_v1"
    lane: str = "bind"
    door: str = "call"
    status: Literal["fired", "no_fire", "error"] = "no_fire"
    spec_digest: str = ""
    rules: tuple[dict[str, Any], ...] = ()
    model_class: str = ""
    model_training: bool = False
    grad_enabled: bool = True
    resolved_static_targets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fire_count: int = 0
    rule_fire_counts: dict[str, int] = field(default_factory=dict)
    fires: tuple[dict[str, Any], ...] = ()
    fire_records: tuple[Any, ...] = ()
    zero_fire_rule_ids: tuple[str, ...] = ()
    #: Canonical address -> the OTHER addresses the same module object is
    #: registered under (``named_modules(remove_duplicate=False)``). A rule
    #: anchored on the canonical name fires at EVERY call site of the shared
    #: module; alias spellings never resolve (bind_static_anchor_unresolved).
    module_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    execution_effect: str = _BIND_EXECUTION_EFFECT
    error: str | None = None
    cleanup: str = "removed"
    duration_s: float | None = None
    trace: None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the report as one plain envelope-shaped dictionary."""

        from dataclasses import asdict

        payload = asdict(self)
        payload["fire_records"] = self.fire_records
        return payload


def _selector_terms(selector: Any, *, negated: bool = False) -> list[tuple[str, Any, bool]]:
    """Walk one selector tree into ``(kind, node, negated)`` leaf terms.

    Parameters
    ----------
    selector:
        Selector or bare callable WHERE term.
    negated:
        Whether the walk is under an odd number of ``~`` operators (anchors
        under negation are trivially satisfiable and never required to
        resolve).
    """

    from .selectors import BaseSelector, CompositeSelector, NotSelector

    if isinstance(selector, NotSelector):
        return _selector_terms(selector.selector, negated=not negated)
    if isinstance(selector, CompositeSelector):
        terms: list[tuple[str, Any, bool]] = []
        for child in selector.selectors:
            terms.extend(_selector_terms(child, negated=negated))
        return terms
    if isinstance(selector, BaseSelector):
        return [(str(getattr(selector, "selector_kind", "")), selector, negated)]
    return [("predicate", selector, negated)]


def _module_anchor_addresses(rule: Any) -> list[str]:
    """Collect the non-negated module-address anchors of one rule's WHERE term."""

    anchors: list[str] = []
    for kind, node, negated in _selector_terms(rule.where):
        if negated:
            continue
        if kind in {"module", "in_module"}:
            anchors.append(str(node.selector_value))
        elif kind == "site":
            module_path = getattr(node, "module_path", None)
            if module_path:
                anchors.append(str(module_path))
    return anchors


def _bare_address(anchor: str) -> str:
    """Strip a trailing ``:pass`` qualifier from a module-address anchor."""

    base, sep, tail = anchor.rpartition(":")
    if sep and tail.isdigit():
        return base
    return anchor


def _is_boundary_rule(rule: Any) -> bool:
    """Whether a rule targets a module OUTPUT boundary (plain ``tl.module``)."""

    return getattr(rule.where, "selector_kind", None) == "module"


def _rule_contains_module_boundary(rule: Any) -> bool:
    """Whether any non-negated term of the WHERE tree is a ``module`` kind."""

    return any(
        kind == "module" and not negated for kind, _node, negated in _selector_terms(rule.where)
    )


class _RulePlan:
    """One rule's bind-time lowering: the normalized hook callable + stamps."""

    __slots__ = ("rule", "hook_callable", "helper_spec", "display_name", "boundary_targets")

    def __init__(self, rule: Any) -> None:
        """Lower one rule's action through the ONE hook normalizer.

        Raises
        ------
        BindingPreflightError
            ``bind_rule_unsupported`` when the rule's decision carries no
            executable hook payload.
        """

        from .hooks import normalize_hook
        from .types import HelperSpec

        decision = rule.decision
        if decision is None or decision.hook is None:
            raise BindingPreflightError(
                f"rule {rule.rule_id} ({rule.action_repr}) carries no executable "
                "hook payload; the capture-free lane can only run value-replacing "
                "actions",
                code="bind_rule_unsupported",
                remedy="use a value-replacing action (tl.scale, tl.noise, "
                "tl.replace_with, or a callable taking (out, *, hook))",
            )
        self.rule = rule
        self.helper_spec = decision.hook if isinstance(decision.hook, HelperSpec) else None
        self.hook_callable = normalize_hook(decision.hook, direction="forward")
        self.display_name = (
            self.helper_spec.helper_name
            if self.helper_spec is not None
            else getattr(decision.hook, "__qualname__", type(decision.hook).__name__)
        )
        self.boundary_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BindFireSite:
    """The coordinate of one fired bind target, as the fire ledger records it.

    Attributes
    ----------
    target_label:
        Pass-qualified target label (``"relu_1_2:1"`` / ``"blocks.7:2"``).
    site_key:
        L1 structural site key when minted (module address on boundary
        fires; ``None`` when underivable).
    container_path:
        Path of the fired tensor leaf inside the target's output structure.
    pass_index:
        One-based pass index of the fire.
    """

    target_label: str
    site_key: str | None
    container_path: tuple[Any, ...]
    pass_index: int


class _BindSession:
    """Mutable per-call runtime state (counters, stack, fires, minter)."""

    def __init__(self, binding: BoundInterventionExecutor, door: str) -> None:
        """Initialize fresh capture-parity counters for one bound call."""

        from .site_keys import LiveSiteKeyMinter

        self.binding = binding
        self.door = door
        self.minter = LiveSiteKeyMinter()
        self.event_counter = 0
        self.layer_counter = 0
        self.type_counter: defaultdict[str, int] = defaultdict(int)
        self.module_stack: list[Any] = []
        self.module_pass_counts: defaultdict[str, int] = defaultdict(int)
        self.root_passes = 0
        self.run_ctx: dict[str, Any] = {}
        self.rule_scratch: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        self.fires: list[dict[str, Any]] = []
        self.fire_records: list[Any] = []
        self.rule_fire_counts: dict[str, int] = {rule.rule_id: 0 for rule in binding.spec.rules}
        self.start = time.monotonic()

    def record_fire(
        self,
        plan: _RulePlan,
        site: _BindFireSite,
        *,
        replaced: bool,
        in_place_op: bool = False,
    ) -> None:
        """Mint one fire disclosure through the ONE builder and ledger it.

        ``site`` is the fired target's coordinate (label, site key, container
        path, pass index) as one frozen carrier.
        ``replaced`` is derived by the caller from object identity
        (``hooked is not out``), exactly as the capture lane derives it -- an
        identity hook fired but replaced nothing (AUD-CODE 3.7a).
        ``in_place_op`` discloses that the target was an in-place torch call
        (``mul_``, ``relu_``, ``F.relu(inplace=True)``): this lane substitutes
        the RETURNED handle only, so a caller that ignores the return and
        keeps the mutated storage never sees the edit.
        """

        previous_notes = tuple(self.run_ctx.get("ledger_notes", ()))
        record = build_fire_record(
            target_label=site.target_label,
            container_path=site.container_path,
            engine="bind",
            helper=plan.helper_spec,
            site_label=site.site_key,
            timing="post",
            direction="forward",
            helper_name=plan.display_name,
            run_ctx=self.run_ctx,
            previous_notes=previous_notes,
            replaced=replaced,
        )
        self.fire_records.append(record)
        self.rule_fire_counts[plan.rule.rule_id] += 1
        self.fires.append(
            {
                "rule_id": plan.rule.rule_id,
                "target": site.target_label,
                "site_key": site.site_key,
                "container_path": tuple(site.container_path),
                "pass_index": site.pass_index,
                "helper": plan.display_name,
                "execution_effect": _BIND_EXECUTION_EFFECT,
                "replaced": replaced,
                "in_place_op": in_place_op,
                "timestamp": record.timestamp,
            }
        )


class _BindDispatchMode:
    """The op-level interceptor (torch-function protocol; outermost-call).

    Built lazily as a ``TorchFunctionMode`` subclass instance so importing
    this module never requires torch.
    """

    def __new__(cls, session: _BindSession, plans: list[_RulePlan]) -> Any:
        """Build the mode instance bound to one session."""

        import torch
        from torch.overrides import TorchFunctionMode

        from ..capture.arg_positions import _normalize_func_name

        wrap_universe = _wrap_name_universe()

        class _Mode(TorchFunctionMode):
            """Session-scoped torch-function interceptor for one bound call."""

            def __torch_function__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                """Run the call, then offer its output to the op-level rules."""

                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                name = getattr(func, "__name__", None)
                if not name:
                    return out
                layer_type = _normalize_func_name(name)
                if layer_type not in wrap_universe:
                    return out
                if not _has_tensor(out, torch.Tensor):
                    return out
                in_place = _is_in_place_call(name, kwargs)
                return _consider_op(session, plans, (layer_type, name, out), in_place=in_place)

        return _Mode()


def _is_in_place_call(func_name: str, kwargs: dict[str, Any]) -> bool:
    """Whether one intercepted torch call mutates its input storage in place."""

    if kwargs.get("inplace") is True:
        return True
    return func_name.endswith("_") and not func_name.endswith("__")


def _has_tensor(value: Any, tensor_type: type) -> bool:
    """Cheap check that an output holds at least one top-level tensor."""

    if isinstance(value, tensor_type):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, tensor_type) for item in value)
    if isinstance(value, dict):
        return any(isinstance(item, tensor_type) for item in value.values())
    return False


_WRAP_UNIVERSE: frozenset[str] | None = None


def _wrap_name_universe() -> frozenset[str]:
    """The normalized names of every function the capture lane would wrap.

    Gating op consideration on this universe keeps ``tl.func(...)`` matching
    parity with the capture wrappers (the same public-API layer).
    """

    global _WRAP_UNIVERSE
    if _WRAP_UNIVERSE is None:
        from ..capture.arg_positions import _normalize_func_name
        from ..constants import get_orig_torch_funcs

        _WRAP_UNIVERSE = frozenset(
            _normalize_func_name(func_name) for _ns, func_name in get_orig_torch_funcs()
        )
    return _WRAP_UNIVERSE


def _consider_op(
    session: _BindSession,
    plans: list[_RulePlan],
    call: tuple[str, str, Any],
    *,
    in_place: bool = False,
) -> Any:
    """Evaluate + apply op-level rules against one executed torch call.

    Mirrors the capture lane's per-output context arithmetic (raw labels,
    per-type counters, event indexes) so WHERE semantics and live site keys
    agree with what a capture of the same forward would evaluate.
    """

    from ..backends.torch._ops_interventions import _iter_loggable_live_outputs
    from ..capture.predicates import build_op_record_context
    from .hooks import make_hook_context
    from .runtime import _execute_hook, _replace_tensor_outputs

    layer_type, func_name, out_orig = call
    outputs = list(_iter_loggable_live_outputs(out_orig, True))
    if not outputs:
        return out_orig
    binding = session.binding
    plans_by_rule = {plan.rule.rule_id: plan for plan in plans}
    replacements: dict[tuple[Any, ...], Any] = {}
    module_frame = session.module_stack[-1] if session.module_stack else None
    output_ordinal = 0
    for out, container_path, _container_spec in outputs:
        output_ordinal += 1
        raw_index = session.layer_counter + output_ordinal
        type_index = session.type_counter[layer_type] + output_ordinal
        raw_label = f"{layer_type}_{type_index}_{raw_index}_raw"
        tail = container_path[-1] if container_path else None
        ctx = build_op_record_context(
            kind="op",
            label=raw_label,
            raw_label=raw_label,
            raw_index=raw_index,
            layer_type=layer_type,
            type_index=type_index,
            func_name=func_name,
            parent_labels=(),
            tensor=out,
            output_index=tail if isinstance(tail, int) else None,
            is_bottom_level_func=True,
            module_stack=tuple(session.module_stack),
            history=(),
            op_counts=dict(session.type_counter),
            pass_index=max(session.root_passes, 1),
            event_index=session.event_counter + output_ordinal,
            step_index=session.event_counter + output_ordinal,
            capture_start_time=session.start,
            include_source_events=False,
            sample_id=None,
            address=getattr(module_frame, "address", None) if module_frame else None,
            module_type=getattr(module_frame, "module_type", None) if module_frame else None,
            module_pass_index=(getattr(module_frame, "pass_index", None) if module_frame else None),
        )
        try:
            rule = binding.spec.match(ctx)
        except InvalidArgumentError:
            raise
        except Exception as exc:
            raise BindingRuntimeError(
                f"rule WHERE evaluation failed at {raw_label} "
                f"(func {func_name!r}, module "
                f"{getattr(module_frame, 'address', None)!r}) during a bound "
                f"{session.door}",
                code="bind_rule_runtime_error",
                remedy="fix the predicate/selector to tolerate live capture "
                "contexts, or re-express the WHERE term structurally",
            ) from exc
        if rule is None:
            continue
        plan = plans_by_rule.get(rule.rule_id)
        if plan is None:
            # A module-boundary rule can never match an op context (boundary
            # rules apply at real module hooks); nothing to fire here.
            continue
        site_key = session.minter.key_for_context(ctx)
        hook_ctx = make_hook_context(
            name=plan.display_name,
            timing="post",
            direction="forward",
            layer_log={
                "label": raw_label,
                "layer_type": layer_type,
                "func_name": func_name,
                "site_key": site_key,
                "pass_index": max(session.root_passes, 1),
            },
            ctx=session.rule_scratch[rule.rule_id],
            run_ctx=session.run_ctx,
        )
        hooked = _execute_hook(plan.hook_callable, out, hook_ctx)
        session.record_fire(
            plan,
            _BindFireSite(
                target_label=raw_label,
                site_key=site_key,
                container_path=container_path,
                pass_index=max(session.root_passes, 1),
            ),
            replaced=hooked is not out,
            in_place_op=in_place,
        )
        if hooked is not out:
            replacements[container_path] = hooked
    session.layer_counter += len(outputs)
    session.type_counter[layer_type] += len(outputs)
    session.event_counter += len(outputs)
    if not replacements:
        return out_orig
    import torch

    if isinstance(out_orig, torch.Tensor):
        return replacements.get((), out_orig)
    return _replace_tensor_outputs(out_orig, replacements)


class _ArmedRuntime:
    """Atomic install + removal of ALL per-call runtime state (item 5).

    Installs the module-stack tracker hooks, boundary-rule hooks, the live
    site-key minter, and (only when op-level rules exist) the dispatch mode.
    Teardown runs on success AND on exception; a partial install tears down
    what it installed before re-raising.
    """

    def __init__(self, binding: BoundInterventionExecutor, session: _BindSession) -> None:
        """Stage the runtime pieces for one call without installing them."""

        self.binding = binding
        self.session = session
        self._handles: list[Any] = []
        self._minter_cm: Any = None
        self._mode: Any = None

    def __enter__(self) -> _ArmedRuntime:
        """Install the runtime atomically (teardown on partial failure)."""

        from .site_keys import _ACTIVE_LIVE_MINTER

        try:
            self._minter_token = _ACTIVE_LIVE_MINTER.set(self.session.minter)
            self._install_tracker_hooks()
            self._install_boundary_hooks()
            if self.binding._op_level_plans:
                self._mode = _BindDispatchMode(self.session, self.binding._op_level_plans)
                self._mode.__enter__()
        except BaseException:
            self._teardown()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Remove every installed piece; a teardown failure is typed."""

        failure = self._teardown()
        if failure is not None:
            self.session.binding._cleanup_verdict = f"failed: {failure!r}"
            if exc is None:
                raise BindingRuntimeError(
                    "bound-call runtime teardown failed; the base model may retain TorchLens hooks",
                    code="bind_cleanup_failed",
                    remedy="drop this binding, re-instantiate the model or "
                    "remove leftover forward hooks, and rebuild the binding "
                    "from the immutable spec",
                ) from failure
        else:
            self.session.binding._cleanup_verdict = "removed"

    def _teardown(self) -> BaseException | None:
        """Best-effort removal of everything installed; returns the first failure."""

        from .site_keys import _ACTIVE_LIVE_MINTER

        first: BaseException | None = None
        if self._mode is not None:
            try:
                self._mode.__exit__(None, None, None)
            except BaseException as exc:  # noqa: BLE001 -- teardown must continue
                first = first or exc
            self._mode = None
        for handle in self._handles:
            try:
                handle.remove()
            except BaseException as exc:  # noqa: BLE001 -- teardown must continue
                first = first or exc
        self._handles.clear()
        if getattr(self, "_minter_token", None) is not None:
            try:
                _ACTIVE_LIVE_MINTER.reset(self._minter_token)
            except BaseException as exc:  # noqa: BLE001 -- teardown must continue
                first = first or exc
            self._minter_token = None  # type: ignore[assignment]
        return first

    def _install_tracker_hooks(self) -> None:
        """Install the capture-parity module-stack tracker on every module."""

        from ..ir.predicate import ModuleStackFrame

        session = self.session
        root_id = id(self.binding.base_model)

        def _make_pre(address: str, module_type: str, module_id: int) -> Any:
            """Build the frame-pushing pre-hook for one module."""

            def _pre(module: Any, args: Any) -> None:
                """Push one stack frame; count per-address and root passes."""

                session.module_pass_counts[address] += 1
                if module_id == root_id:
                    session.root_passes += 1
                if address == "":
                    # The root module owns the pass count but is NOT a
                    # containing module: the capture minter's module axis
                    # holds submodule addresses only, so a root frame here
                    # minted ``s1|/fc1|...`` against capture's ``s1|fc1|...``
                    # for every submodule op (AUD-CODE 2.3d).
                    return
                session.module_stack.append(
                    ModuleStackFrame(
                        address=address,
                        module_type=module_type,
                        module_id=module_id,
                        pass_index=session.module_pass_counts[address],
                    )
                )

            return _pre

        def _make_post(module_id: int) -> Any:
            """Build the frame-popping post-hook for one module."""

            def _post(module: Any, args: Any, output: Any) -> None:
                """Pop this module's frame (LIFO; tolerant of unwound stacks)."""

                stack = session.module_stack
                for index in range(len(stack) - 1, -1, -1):
                    if stack[index].module_id == module_id:
                        del stack[index:]
                        break

            return _post

        for address, module in self.binding.base_model.named_modules():
            self._handles.append(
                module.register_forward_pre_hook(
                    _make_pre(address, type(module).__name__, id(module))
                )
            )
            self._handles.append(module.register_forward_hook(_make_post(id(module))))

    def _install_boundary_hooks(self) -> None:
        """Install module-output substitution hooks for boundary rules."""

        session = self.session
        binding = self.binding
        for plan in binding._boundary_plans:
            target_value = str(plan.rule.where.selector_value)
            for address in plan.boundary_targets:
                module = binding._modules_by_address[address]
                self._handles.append(
                    module.register_forward_hook(
                        _make_boundary_hook(session, plan, address, target_value)
                    )
                )


def _make_boundary_hook(
    session: _BindSession, plan: _RulePlan, address: str, target_value: str
) -> Any:
    """Build one module-output substitution forward hook for one rule/site."""

    def _boundary(module: Any, args: Any, output: Any) -> Any:
        """Replace the module's output leaves when the pass qualifier matches."""

        from ..backends.torch._ops_interventions import _iter_loggable_live_outputs
        from ..ir.selector_eval import module_address_matches
        from .hooks import make_hook_context
        from .runtime import _execute_hook, _replace_tensor_outputs

        pass_index = session.module_pass_counts[address]
        # A bare target address matches every pass; a pass-qualified target
        # ("blocks.7:2") matches only that pass -- the same candidate spelling
        # the capture lifecycle produces for containment matching.
        if not module_address_matches(f"{address}:{pass_index}", target_value):
            return None
        replacements: dict[tuple[Any, ...], Any] = {}
        for out, container_path, _spec in _iter_loggable_live_outputs(output, True):
            hook_ctx = make_hook_context(
                name=plan.display_name,
                timing="post",
                direction="forward",
                layer_log={
                    "label": f"{address}:{pass_index}",
                    "module_address": address,
                    "pass_index": pass_index,
                },
                ctx=session.rule_scratch[plan.rule.rule_id],
                run_ctx=session.run_ctx,
            )
            hooked = _execute_hook(plan.hook_callable, out, hook_ctx)
            session.record_fire(
                plan,
                _BindFireSite(
                    target_label=f"{address}:{pass_index}",
                    site_key=address,
                    container_path=container_path,
                    pass_index=pass_index,
                ),
                replaced=hooked is not out,
            )
            if hooked is not out:
                replacements[container_path] = hooked
        if not replacements:
            return None
        import torch

        if isinstance(output, torch.Tensor):
            return replacements.get(())
        return _replace_tensor_outputs(output, replacements)

    return _boundary


def _module_aliases(model: Any, modules_by_address: dict[str, Any]) -> dict[str, str]:
    """Map every alias address to the canonical name of the shared module.

    ``named_modules()`` de-duplicates: a module object registered under two
    attributes (``self.dec = self.enc``) is reported ONCE, under the first
    name. The other spellings are aliases -- no hook can distinguish the two
    call sites, so an anchor on an alias cannot resolve, and an anchor on
    the canonical name fires at every call site (AUD-CODE 3.7d).
    """

    canonical_by_id = {id(module): address for address, module in modules_by_address.items()}
    aliases: dict[str, str] = {}
    for address, module in model.named_modules(remove_duplicate=False):
        canonical = canonical_by_id.get(id(module))
        if canonical is not None and canonical != address:
            aliases[address] = canonical
    return aliases


def _alias_disclosure(aliases: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Invert the alias map into canonical address -> alias spellings."""

    disclosure: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        disclosure.setdefault(canonical, []).append(alias)
    return {canonical: tuple(names) for canonical, names in disclosure.items()}


def _lower_rules(
    spec: InterventionSpec,
    modules_by_address: dict[str, Any],
    aliases: dict[str, str] | None = None,
) -> tuple[list[_RulePlan], list[_RulePlan], dict[str, tuple[str, ...]]]:
    """Lower every rule and resolve its static anchors before ANY forward.

    Returns (boundary plans, op-level plans, per-rule resolved targets).

    Raises
    ------
    BindingPreflightError
        ``bind_rule_unsupported`` for boundary/op-level composites;
        ``bind_static_anchor_unresolved`` for stale module addresses.
    """

    aliases = aliases or {}
    boundary_plans: list[_RulePlan] = []
    op_level_plans: list[_RulePlan] = []
    resolved: dict[str, tuple[str, ...]] = {}
    unresolved: list[str] = []
    for rule in spec.rules:
        plan = _RulePlan(rule)
        hits: list[str] = []
        for anchor in _module_anchor_addresses(rule):
            bare = _bare_address(anchor)
            if bare in modules_by_address:
                hits.append(bare)
            elif bare in aliases:
                unresolved.append(
                    f"{rule.rule_id}: {anchor!r} is an alias of {aliases[bare]!r} (the same "
                    "module object registered under both names; named_modules() reports "
                    f"only {aliases[bare]!r}, and a rule anchored there fires at EVERY call "
                    "site of the shared module)"
                )
            else:
                unresolved.append(f"{rule.rule_id}: {anchor!r}")
        if _rule_contains_module_boundary(rule):
            if not _is_boundary_rule(rule):
                raise BindingPreflightError(
                    f"rule {rule.rule_id} composes a module OUTPUT-boundary "
                    "term (tl.module) with op-level terms; the capture-free "
                    "lane applies boundary rules at real module hooks and "
                    "cannot evaluate the composite at one site",
                    code="bind_rule_unsupported",
                    remedy="use plain tl.module(address) for the boundary "
                    "edit, or tl.in_module(address) for op-level containment",
                )
            plan.boundary_targets = tuple(dict.fromkeys(hits))
            boundary_plans.append(plan)
        else:
            op_level_plans.append(plan)
        resolved[rule.rule_id] = tuple(dict.fromkeys(hits)) or ("op-level",)
    if unresolved:
        raise BindingPreflightError(
            "static anchors did not resolve against the base model before "
            f"any forward: {', '.join(unresolved)}",
            code="bind_static_anchor_unresolved",
            remedy="fix the module addresses to names in "
            "model.named_modules() (an alias spelling resolves to its canonical "
            "name, which fires at every call site), or drop the stale rules "
            "from the spec",
        )
    return boundary_plans, op_level_plans, resolved


class BoundInterventionExecutor:
    """The bound intervention executor: one spec, one model, changing neither.

    Serial, non-reentrant, capture-free, and never an ``nn.Module``. The call
    is transparent (``bound(x)`` returns exactly the model's own output);
    ``generate()`` holds the runtime across the whole generation with
    pass-qualified rule semantics; the ledger is out-of-band on
    ``.last_report``; ``.spec``/``.base_model`` are read-only; runtime state
    installs and removes atomically on success or exception; zero-fire rules
    fail closed after the call by default (FOLD-A3); there is no binding
    serialization.
    """

    def __init__(self, spec: InterventionSpec, model: Any, *, on_zero_fire: str) -> None:
        """Preflight + lower the spec against the model (no forward runs).

        Use :meth:`InterventionSpec.bind`; this constructor is the engine.
        """

        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_base_model", model)
        object.__setattr__(self, "_on_zero_fire", on_zero_fire)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_last_report", None)
        object.__setattr__(self, "_cleanup_verdict", "removed")
        self._preflight()

    # ------------------------------------------------------------------
    # read-only surface (surgery 3.3 item 4)
    # ------------------------------------------------------------------
    @property
    def spec(self) -> InterventionSpec:
        """The immutable spec this binding applies (read-only)."""

        return self._spec

    @property
    def base_model(self) -> Any:
        """The base model this binding runs (read-only; never copied)."""

        return self._base_model

    @property
    def last_report(self) -> BindReport | None:
        """The most recent call's out-of-band ledger (``None`` before any call)."""

        return self._last_report

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse rebinding the read-only surface; allow internal state."""

        if name in {"spec", "base_model", "last_report", "_spec", "_base_model"}:
            raise AttributeError(
                f"BoundInterventionExecutor.{name} is read-only; build a new "
                "binding from the immutable spec instead"
            )
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        """Disclosure-first repr (read-only surface, lane, last status)."""

        status = self._last_report.status if self._last_report is not None else "never called"
        return (
            f"<bound intervention executor: {len(self._spec.rules)} rule(s) "
            f"{self._spec.spec_digest} over {type(self._base_model).__name__}; "
            f"capture-free live lane (values replaced after execution); "
            f".spec/.base_model read-only; last call: {status}>"
        )

    # ------------------------------------------------------------------
    # refusal surfaces (items 7 and 9)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Raise the training-surface teaching refusal for module-like access."""

        if name in _TRAINING_SURFACE_NAMES:
            raise BindingRuntimeError(
                f"a bound intervention executor has no {name!r}: it is NOT an "
                "nn.Module and never will be (no training lifecycle, no "
                "compile/shard/export/checkpoint claim). A hook binding and an "
                "owned trainable module are different artifacts",
                code="binding_training_surface",
                remedy="write a small wrapper nn.Module you own around "
                "binding.base_model, or use binding.base_model directly for "
                "training surfaces",
            )
        raise AttributeError(name)

    def __reduce__(self) -> Any:
        """Refuse binding serialization typed (item 9: one save path, one trust gate)."""

        raise BindingRuntimeError(
            "a bound intervention executor is never serialized; the spec is "
            "the keepable artifact and the binding is cheap to reconstruct",
            code="binding_serialization_unsupported",
            remedy="save the spec through the intervention-spec save path and "
            "rebuild with spec.bind(model) after loading",
        )

    # ------------------------------------------------------------------
    # preflight (item 8, bind time)
    # ------------------------------------------------------------------
    def _preflight(self) -> None:
        """Validate operands, resolve static anchors, lower every rule.

        Raises
        ------
        BindingPreflightError
            ``bind_spec_invalid`` / ``bind_spec_empty`` /
            ``bind_model_invalid`` / ``bind_zero_fire_policy_invalid`` /
            ``bind_static_anchor_unresolved`` / ``bind_rule_unsupported``.
        """

        import torch

        spec = self._spec
        model = self._base_model
        if not isinstance(spec, InterventionSpec):
            raise BindingPreflightError(
                f"bind() needs the immutable public InterventionSpec; received "
                f"{type(spec).__name__}",
                code="bind_spec_invalid",
                remedy="build the spec with tl.when(...) and call spec.bind(model)",
            )
        if not spec.rules:
            raise BindingPreflightError(
                "bind() refuses an empty spec: zero rules means zero fires by "
                "construction, and zero-fire rules fail closed",
                code="bind_spec_empty",
                remedy="add at least one tl.when(...) clause before binding",
            )
        if self._on_zero_fire not in _ZERO_FIRE_POLICIES:
            raise BindingPreflightError(
                f"on_zero_fire must be 'error' or 'disclose', got {self._on_zero_fire!r}",
                code="bind_zero_fire_policy_invalid",
                remedy="pass on_zero_fire='error' (fail-closed default) or 'disclose'",
            )
        if not isinstance(model, torch.nn.Module):
            hint = ""
            if callable(model) and getattr(model, "__self__", None) is not None:
                hint = (
                    " (bound methods are lane F41's root contract, not the "
                    "binding's: bind the owning nn.Module)"
                )
            raise BindingPreflightError(
                f"bind() needs an nn.Module base model; received {type(model).__name__}{hint}",
                code="bind_model_invalid",
                remedy="pass the plain nn.Module (spec.bind(model)); wrap bare "
                "functions in a small module you own",
            )
        modules_by_address = dict(model.named_modules())
        object.__setattr__(self, "_modules_by_address", modules_by_address)
        aliases = _module_aliases(model, modules_by_address)
        object.__setattr__(self, "_module_aliases", _alias_disclosure(aliases))
        boundary_plans, op_level_plans, resolved = _lower_rules(spec, modules_by_address, aliases)
        object.__setattr__(self, "_boundary_plans", boundary_plans)
        object.__setattr__(self, "_op_level_plans", op_level_plans)
        object.__setattr__(self, "_resolved_static_targets", resolved)

    # ------------------------------------------------------------------
    # execution (items 1, 2, 5, 8-call-time)
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the base model once with the runtime armed; return its output."""

        return self._run(self._base_model, args, kwargs, door="call")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Run the base model's real ``generate`` with the runtime held across it.

        Raises
        ------
        BindingRuntimeError
            ``bind_generate_unavailable`` when the base model has no callable
            ``generate``.
        """

        target = getattr(self._base_model, "generate", None)
        if not callable(target):
            raise BindingRuntimeError(
                f"base model {type(self._base_model).__name__} has no callable "
                "generate(); the binding holds the runtime across a real HF "
                "generate and cannot invent one",
                code="bind_generate_unavailable",
                remedy="call the binding directly (bound(x)), or wrap your "
                "generation loop around bound(x) yourself",
            )
        return self._run(target, args, kwargs, door="generate")

    def _run(self, target: Any, args: tuple, kwargs: dict, *, door: str) -> Any:
        """One serial bound call: arm, execute, settle, report."""

        if not self._lock.acquire(blocking=False):
            raise BindingRuntimeError(
                "this binding is already executing; v1 bindings are serial and "
                "non-reentrant (one call at a time, never from inside a hook)",
                code="binding_reentrant_call",
                remedy="wait for the active call to return, or make separate "
                "bindings from the same immutable spec for concurrent workers",
            )
        session = _BindSession(self, door)
        error: str | None = None
        output: Any = None
        try:
            try:
                with _ArmedRuntime(self, session):
                    output = target(*args, **kwargs)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
        finally:
            report = self._settle(session, door=door, error=error)
            object.__setattr__(self, "_last_report", report)
            self._lock.release()
        if self._on_zero_fire == "error" and report.zero_fire_rule_ids:
            names = ", ".join(report.zero_fire_rule_ids)
            raise BindingRuntimeError(
                f"rule(s) [{names}] never fired during this bound {door}; "
                "zero-fire rules fail closed after the call (the model's side "
                "effects may already have occurred; the full report is retained "
                "on .last_report)",
                code="bind_zero_fire",
                remedy="fix the WHERE terms so every rule fires, or opt into "
                "disclosure-only settlement with "
                "spec.bind(model, on_zero_fire='disclose')",
            )
        return output

    def _settle(self, session: _BindSession, *, door: str, error: str | None) -> BindReport:
        """Build the out-of-band ledger for one finished (or failed) call."""

        import torch

        zero_fire = tuple(
            rule_id for rule_id, count in session.rule_fire_counts.items() if count == 0
        )
        fire_count = len(session.fire_records)
        if error is not None:
            status: Literal["fired", "no_fire", "error"] = "error"
        else:
            status = "fired" if fire_count else "no_fire"
        if _BIND_EXECUTION_EFFECT not in EXECUTION_EFFECTS:
            raise RuntimeError("bind execution-effect constant left the closed vocabulary")
        return BindReport(
            door=door,
            status=status,
            spec_digest=self._spec.spec_digest,
            rules=rules_payload(self._spec),
            model_class=type(self._base_model).__qualname__,
            model_training=bool(getattr(self._base_model, "training", False)),
            grad_enabled=torch.is_grad_enabled(),
            resolved_static_targets=dict(self._resolved_static_targets),
            fire_count=fire_count,
            rule_fire_counts=dict(session.rule_fire_counts),
            fires=tuple(session.fires),
            fire_records=tuple(session.fire_records),
            zero_fire_rule_ids=zero_fire,
            module_aliases=dict(self._module_aliases),
            error=error,
            cleanup=self._cleanup_verdict,
            duration_s=time.monotonic() - session.start,
        )


def bind_spec_to_model(
    spec: InterventionSpec, model: Any, *, on_zero_fire: str = "error"
) -> BoundInterventionExecutor:
    """The bind door: funnel the model operand, then construct the executor.

    Parameters
    ----------
    spec:
        The immutable public spec.
    model:
        The base ``nn.Module``. A bound executor operand resolves through the
        ONE model-door funnel (OP2): arm (a) refuses typed; arm (b) would
        re-bind the same base model, which is refused here explicitly --
        bindings never nest.
    on_zero_fire:
        Zero-fire settlement policy (``"error"`` default per FOLD-A3).
    """

    resolution = resolve_model_operand(model, door="bind")
    if resolution.normalized:
        raise BindingPreflightError(
            "bind() received a bound intervention executor; bindings never "
            "nest -- one binding pairs one spec with one plain model",
            code="bind_model_invalid",
            remedy="bind the base model: spec.bind(binding.base_model), "
            "merging specs first if both should apply",
        )
    return BoundInterventionExecutor(spec, resolution.model, on_zero_fire=on_zero_fire)


__all__ = [
    "BindReport",
    "BoundInterventionExecutor",
    "bind_spec_to_model",
]
