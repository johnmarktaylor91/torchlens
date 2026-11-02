"""Wave-3 fills for the intervention verb of the behaviour net.

Rows, never files: the claimed features (``intervene``, ``fork_do``,
``bundle``, ``bind``) get ``after`` hooks that assert their full promise
through the public surface, four features join the axis (``hooks``,
``mean_ablate``, ``steer``, ``edit_chain``) and two toy cases join the case
registry. ``features.py`` and ``cases.py`` call :func:`extend_features` and
:func:`extend_toys` at their end, so this module imports neither at module
level (both registries are still being built when it loads).

The promises checked here, each through ``tl.*`` and the record fields the
glossary documents:

* an edit changes exactly the edited site and its descendants, stamps
  ``is_edited`` on the edited op and records a ``FireRecord``
  on it;
* ``fork_do`` halves the edited site, on the live and on the loaded trace;
* bundle members' outputs differ as declared and ``Bundle.compare`` puts the
  edited site at the top (zero ablation gives it relative distance 1.0) while
  every site outside its cone reads 0;
* user hooks under ``fn(out, *, ctx)`` fire once per site per pass with
  ``ctx.signal == "forward"``, their ``ctx.state`` scratch survives across
  the passes of one site and ``ctx.run_state`` across the run; a hook that
  raises refuses typed and leaves torch able to run and capture again;
* ``tl.mean_ablate(source=None, *, dim)`` matches a hand-computed mean for
  the self source, an axis and a ``Population``;
* ``tl.steer`` adds the scaled direction, on multi-pass sites of recurrent
  cases too;
* two edits on one site apply in order, an edit on a view site stays inside
  its cone, and a seeded stochastic edit reproduces;
* an intervened capture persists as documented: the loaded trace carries
  the same replaced-op set and fire records, its ``recipe`` is empty because
  ``trace.save()`` does not persist the intervention spec, and its replay
  refuses by the door ``run_intervention_spec_not_persisted``
  (:data:`RECIPE_PERSISTS` flips the check when the saved format decides to
  carry the spec).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

import torchlens as tl
from torchlens.errors import TorchLensError

if TYPE_CHECKING:
    from tests.behaviour.cases import Case
    from tests.behaviour.features import Feature
    from tests.behaviour.harness import Cell

SEED = 0
#: Whether ``trace.save()`` persists the intervention recipe. ``False`` is the
#: documented state (door ``run_intervention_spec_not_persisted`` in
#: ``torchlens/intervention/_staged_spec_persistence.py``); flip it when the
#: saved format decides to carry the spec and the loaded-trace checks assert
#: an equal recipe and a reproducing replay instead.
RECIPE_PERSISTS = False
_VIEW_FUNCS = frozenset(
    {
        "view",
        "reshape",
        "permute",
        "transpose",
        "t",
        "expand",
        "squeeze",
        "unsqueeze",
        "flatten",
        "__getitem__",
        "narrow",
        "select",
    }
)


def _not_applicable(reason: str) -> Any:
    from tests.behaviour.features import NotApplicable

    raise NotApplicable(reason)


def _error_code(exc: BaseException) -> str:
    """Return the contract code of a TorchLens error the way the harness reads it."""

    fields = getattr(exc, "fields", None) or {}
    return str(fields.get("code", "") or getattr(exc, "code", ""))


def _is_float(dtype: Any) -> bool:
    text = str(dtype)
    return "float" in text or "bfloat" in text


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))
    return bool(torch.equal(a, b))


def _informative(value: torch.Tensor) -> bool:
    """Whether an edit that scales or zeroes ``value`` can change it (a finite non-zero element)."""

    finite = torch.isfinite(value)
    return bool(finite.any()) and bool((value[finite] != 0).any())


# --- sites and cones ---------------------------------------------------------


def _ops_with_out(trace: Any) -> dict[str, torch.Tensor]:
    """Return the saved tensor payload of every op that holds one, by label."""

    return {
        op.label: op.out
        for op in trace.ops
        if isinstance(op.out, torch.Tensor) and op.out.numel() > 0
    }


def _no_pass(label: str) -> str:
    """Return the layer label of ``label`` (``conv_1_1:1`` reads ``conv_1_1``)."""

    head, sep, tail = label.rpartition(":")
    return head if sep and tail.isdigit() else label


def _descendants(trace: Any, label: str) -> set[str]:
    """Return the layer labels downstream of ``label`` (its cone), ``label`` excluded.

    Relations are spelled in no-pass layer labels, so the cone is too.
    """

    seen: set[str] = set()
    frontier = [label]
    while frontier:
        current = frontier.pop()
        for child in trace[current].children or ():
            child_label = child if isinstance(child, str) else getattr(child, "label", str(child))
            child_label = _no_pass(child_label)
            if child_label not in seen:
                seen.add(child_label)
                frontier.append(child_label)
    return seen


def _single_pass_site(cell: Cell, trace: Any) -> str:
    """Return a saved, single-pass, floating-point op label on ``trace``."""

    for op in trace.saved_ops:
        if (
            _is_float(op.dtype)
            and getattr(op.layer, "num_passes", 1) == 1
            and isinstance(op.out, torch.Tensor)
            and op.out.numel() > 0
        ):
            return op.label
    return _not_applicable("no saved single-pass floating-point op to edit on this capture")


def _leaf_site(trace: Any) -> str | None:
    """Return the first saved floating-point leaf site (an input or buffer, no func to replay)."""

    for op in trace.saved_ops:
        if not op.parents and _is_float(op.dtype) and isinstance(op.out, torch.Tensor):
            return op.label
    return None


def _interior_site(cell: Cell, trace: Any) -> str:
    """Return a saved, single-pass, floating-point op that has parents (a replayable func).

    Reads ``op.layer``, which a loaded trace's ops do not expose: choose on the
    live trace and pass the label to the loaded leg.
    """

    for op in trace.saved_ops:
        if (
            op.parents
            and _is_float(op.dtype)
            and getattr(op.layer, "num_passes", 1) == 1
            and isinstance(op.out, torch.Tensor)
            and op.out.numel() > 0
        ):
            return op.label
    return _single_pass_site(cell, trace)


def _multipass_site(trace: Any) -> str | None:
    """Return a saved floating-point op label on a layer that ran more than once, if any."""

    for op in trace.saved_ops:
        if (
            _is_float(op.dtype)
            and getattr(op.layer, "num_passes", 1) > 1
            and isinstance(op.out, torch.Tensor)
            and op.out.numel() > 0
        ):
            return op.label
    return None


def _view_site(trace: Any) -> str | None:
    """Return a saved floating-point view op label (``view``, ``__getitem__``, ...), if any."""

    for op in trace.saved_ops:
        if (
            op.func_name in _VIEW_FUNCS
            and _is_float(op.dtype)
            and getattr(op.layer, "num_passes", 1) == 1
            and isinstance(op.out, torch.Tensor)
            and op.out.numel() > 0
        ):
            return op.label
    return None


def _expect_cone_only(cell: Cell, base: Any, edited: Any, label: str, what: str) -> None:
    """Expect ``edited`` to differ from ``base`` only at ``label`` and inside its cone."""

    cone = _descendants(base, label)
    before, after = _ops_with_out(base), _ops_with_out(edited)
    for name, value in before.items():
        if name == label or _no_pass(name) in cone or name not in after:
            continue
        cell.expect(
            _same(value, after[name]), f"{what}: {name} outside the cone of {label} changed"
        )
    # ``trace.ops`` omits input records: a leaf-site edit is asserted by its caller.
    if label in before and label in after and _informative(before[label]):
        cell.expect(not _same(before[label], after[label]), f"{what}: {label} did not change")


def _marked_replaced(trace: Any) -> set[str]:
    """Return the labels marked replaced, over ``saved_ops`` (``trace.ops`` omits input records)."""

    rows = getattr(trace, "saved_ops", None)
    return {op.label for op in (trace.ops if rows is None else rows) if op.is_edited}


def _expect_edit_marks(
    cell: Cell, base: Any, edited: Any, label: str, helper: str, what: str
) -> None:
    """Expect the edit stamped on exactly ``label`` beyond what ``base`` already carried."""

    marked_before = _marked_replaced(base)
    marked_after = _marked_replaced(edited)
    cell.expect(
        marked_after == marked_before | {label},
        f"{what}: ops marked replaced {sorted(marked_after)[:8]} != "
        f"{sorted(marked_before | {label})[:8]}",
    )
    names = [getattr(record, "helper_name", None) for record in edited[label].interventions or ()]
    cell.expect(helper in names, f"{what}: no {helper!r} fire record on {label}; found {names}")


# --- claimed rows: intervene ------------------------------------------------


def _merge_intervene(cell: Cell, recipe: Any) -> None:
    """Set the capture-time recipe, joining one another feature already staged."""

    existing = cell.trace_kwargs.get("intervene")
    cell.trace_kwargs["intervene"] = recipe if existing is None else existing & recipe
    cell.trace_kwargs["on_zero_fire"] = "error"


def _float_target_labels(cell: Cell, trace: Any) -> set[str]:
    """Return the labels of the floating-point ops of the intervene row's function."""

    target = cell.facts.common_float_func
    return {op.label for op in trace.ops if op.func_name == target and _is_float(op.dtype)}


def _prepare_intervene(cell: Cell) -> None:
    """Scale the floating-point ops of the most common floating-point function.

    ``tl.func(name)`` matches every op of that name, integer ones included, and
    ``tl.scale`` on an integer tensor is a typed dtype refusal; the integer ops
    the plain capture saw under that name are excluded by their raw label (the
    ``<raw_label>_raw`` spelling is the one a live capture can match; a final
    label does not exist yet while the forward runs).
    """

    target = cell.facts.common_float_func
    if target is None:
        _not_applicable("no floating-point op to scale")
    plain = cell.facts.plain
    integer = sorted(
        f"{op.raw_label}_raw"
        for op in plain.ops
        if op.func_name == target and not _is_float(op.dtype)
    )
    selector: Any = tl.func(target)
    for raw_label in integer:
        selector = selector & ~tl.label(raw_label)
    _merge_intervene(cell, tl.when(selector, tl.scale(0.5)))


def _fire_facts(op: Any) -> list[tuple[Any, ...]]:
    return [
        (
            getattr(record, "helper_name", None),
            getattr(record, "site_label", None),
            getattr(record, "timing", None),
            getattr(record, "signal", None),
            getattr(record, "replaced", None),
            getattr(record, "engine", None),
        )
        for record in op.interventions or ()
    ]


def _rule_facts(recipe: Any) -> list[tuple[str, Any]]:
    """Return the comparable content of a frozen recipe: its targets, helper and fire count."""

    if recipe is None:
        return []
    helper = getattr(getattr(recipe, "helper", None), "helper_name", None)
    targets = [repr(target) for target in getattr(recipe, "targets", ())]
    return [(target, helper) for target in targets] + [
        ("records", len(getattr(recipe, "records", ())))
    ]


def _self_hook_outputs(cell: Cell, trace: Any, labels: set[str]) -> set[str]:
    """Return the ``labels`` a model's own forward hook may have stamped as replaced.

    A model's own ``register_forward_hook`` that returns another tensor is
    recorded as a replaced module output (``is_edited``) by design,
    the disclosure a runnable save and validation need. Only a model that
    registers such hooks itself gets the allowance, and only for ops no rule of
    the recipe fired on (they carry no ``scale`` fire record).
    """

    if not any(getattr(module, "_forward_hooks", None) for module in cell.model.modules()):
        return set()
    return {
        label for label in labels if "scale" not in [name for name, *_ in _fire_facts(trace[label])]
    }


def _recipe_is_empty(recipe: Any) -> bool:
    """Return whether a frozen recipe carries no targets and no fire records."""

    return not getattr(recipe, "targets", ()) and not getattr(recipe, "records", ())


def _after_intervene(cell: Cell) -> None:
    """Selected ops record a replaced value, and the edit persists through save and load.

    The persisted part follows the documented saved format: replaced-op set and
    fire records survive, the recipe does not (``loaded.recipe`` is empty) and
    the loaded trace's replay refuses by the typed door. :data:`RECIPE_PERSISTS`
    selects the other reading once the format carries the spec.
    """

    trace = cell.require_trace()
    want = _float_target_labels(cell, trace)
    got = {op.label for op in trace.ops if op.is_edited}
    hooked = any(feature.name == "hooks" for feature in cell.features)
    extra = got - want
    allowed = extra if hooked else _self_hook_outputs(cell, trace, extra)
    cell.expect(
        want <= got and extra <= allowed,
        f"ops recording a replaced value {sorted(got)[:8]} != selected {sorted(want)[:8]}",
    )
    for label in sorted(want):
        names = [name for name, *_ in _fire_facts(trace[label])]
        cell.expect("scale" in names, f"{label}: no scale fire record, found {names}")
    loaded = cell.loaded
    if loaded is None:
        return
    loaded_got = {op.label for op in loaded.ops if op.is_edited}
    cell.expect(loaded_got == got, f"loaded replaced-op set {sorted(loaded_got ^ got)[:8]} differs")
    for label in sorted(want):
        cell.expect(
            _fire_facts(trace[label]) == _fire_facts(loaded[label]),
            f"{label}: fire records differ after load",
        )
    live_rules, loaded_rules = _rule_facts(trace.recipe), _rule_facts(loaded.recipe)
    if RECIPE_PERSISTS:
        cell.expect(
            live_rules == loaded_rules,
            f"recipe differs after load: live {live_rules!r:.160} vs loaded {loaded_rules!r:.160}",
        )
    else:
        cell.expect(
            _recipe_is_empty(loaded.recipe),
            f"trace.save() does not persist the recipe, yet loaded.recipe holds {loaded_rules!r:.160}",
        )
    before = None if cell.case.has("stochastic") else cell.output_leaves(trace)
    _expect_loaded_replay(cell, loaded, before)
    if before is None:
        return
    torch.manual_seed(SEED)
    trace.run(cell.model, cell.fresh_inputs())
    cell.expect_same_leaves(before, cell.output_leaves(trace), "replay of the intervened trace")


def _expect_loaded_replay(cell: Cell, loaded: Any, before: list[Any] | None) -> None:
    """A loaded intervened trace replays as the saved format documents.

    Without a persisted recipe (:data:`RECIPE_PERSISTS` false) ``loaded.run``
    must refuse by the door ``run_intervention_spec_not_persisted``: a run that
    goes through would return un-intervened numbers under intervened records.
    With a persisted recipe it must reproduce the captured output (``before``;
    ``None`` for a stochastic case, whose output is not comparable).
    """

    torch.manual_seed(SEED)
    try:
        loaded.run(cell.model, cell.fresh_inputs())
    except TorchLensError as exc:
        if _error_code(exc) != "run_intervention_spec_not_persisted":
            raise
        cell.expect(
            not RECIPE_PERSISTS,
            "the loaded trace refuses run() as not persisted although the recipe persists",
        )
        return
    cell.expect(
        RECIPE_PERSISTS,
        "the loaded intervened trace ran without its recipe instead of refusing by "
        "run_intervention_spec_not_persisted",
    )
    if before is not None:
        cell.expect_same_leaves(before, cell.output_leaves(loaded), "replay of the loaded trace")


# --- claimed rows: fork_do ----------------------------------------------------


def _prepare_intervention_ready(cell: Cell) -> None:
    if cell.facts.single_pass_float_label is None:
        _not_applicable("no single-pass floating-point op to edit")
    cell.capture_fields["intervention_ready"] = True


def _halve_on_fork(cell: Cell, trace: Any, what: str, label: str | None = None) -> None:
    label = _interior_site(cell, trace) if label is None else label
    original = trace[label].out.detach().clone()
    fork = trace.fork()
    fork.do(tl.label(label), tl.scale(0.5))
    edited = fork[label].out
    cell.expect(
        torch.allclose(edited, original * 0.5, rtol=0, atol=0, equal_nan=True),
        f"{what}: fork.do(scale 0.5) did not halve {label}",
    )
    cell.expect(len(fork.ops) == len(trace.ops), f"{what}: the fork changed the op count")
    _expect_cone_only(cell, trace, fork, label, what)
    _expect_edit_marks(cell, trace, fork, label, "scale", what)
    cell.expect(
        _same(trace[label].out, original), f"{what}: the edit wrote through to the source trace"
    )


def _after_fork_do(cell: Cell) -> None:
    """A fork edit halves the site, stays inside its cone and marks the op (loaded leg: edit_chain)."""

    trace = cell.require_trace()
    _halve_on_fork(cell, trace, "live", _interior_site(cell, trace))


# --- claimed rows: bundle -----------------------------------------------------


def _after_bundle(cell: Cell) -> None:
    """Members differ as declared and ``Bundle.compare`` puts the edited site first."""

    trace = cell.require_trace()
    members = [trace] if cell.loaded is None else [trace, cell.loaded]
    bundle = tl.Bundle(members)
    cell.expect(len(bundle) == len(members), f"bundle has {len(bundle)} members")
    label = _interior_site(cell, trace)
    if not _informative(trace[label].out):
        _not_applicable(f"{label} holds no finite non-zero element to ablate")
    fork = trace.fork()
    fork.do(tl.label(label), tl.zero_ablate())
    pair = tl.Bundle({"base": trace, "edited": fork}, baseline="base")
    compared = pair.compare("base", "edited", metric="relative_l2")
    rows = {_no_pass(name): value for name, value in compared}
    site = _no_pass(label)
    cell.expect(site in rows, f"compare rows lack the edited site {site}: {sorted(rows)[:8]}")
    cell.expect(
        abs(rows[site] - 1.0) < 1e-5,
        f"zero ablation at {site} reads relative distance {rows[site]!r}, wanted 1.0",
    )
    cone = _descendants(trace, label)
    outside = {name: value for name, value in rows.items() if name != site and name not in cone}
    moved = sorted(name for name, value in outside.items() if value != 0)
    cell.expect(not moved, f"sites outside the cone of {site} differ: {moved[:8]}")
    cell.expect(
        all(rows[site] >= value for value in outside.values()),
        f"{site} is not ranked first among the sites outside its cone",
    )


# --- claimed rows: bind -------------------------------------------------------


def _after_bind_with(base_after: Callable[[Cell], None]) -> Callable[[Cell], None]:
    """Extend the bind row's current ``after`` (whichever fill holds it) with the report checks."""

    def _after_bind(cell: Cell) -> None:
        base_after(cell)
        _expect_bind_report(cell)

    return _after_bind


def _expect_bind_report(cell: Cell) -> None:
    """The bound executor reports its recipe and firing."""

    address = cell.facts.single_call_module
    spec = tl.when(tl.module(address), tl.scale(0.5))
    bound = spec.bind(cell.model)
    cell.expect(
        _rule_facts(bound.recipe) == _rule_facts(spec), "BoundModel.recipe is not the bound recipe"
    )
    with torch.no_grad():
        bound(*cell.positional_inputs(cell.fresh_inputs()))
    report = bound.last_report
    cell.expect(report is not None, "the bound call left no last_report")
    cell.expect(not getattr(report, "zero_fire_rule_ids", ()), "the bound rule fired at zero sites")


# --- new row: hooks -----------------------------------------------------------


def _second_float_func(cell: Cell) -> str | None:
    """Return the second most common floating-point function name (the intervene row has the first)."""

    counts = Counter(
        op.func_name
        for op in cell.facts.plain.ops
        if _is_float(op.dtype) and op.func_name != "identity"
    )
    ranked = [name for name, _ in counts.most_common()]
    return ranked[1] if len(ranked) > 1 else None


def _prepare_hooks(cell: Cell) -> None:
    target = _second_float_func(cell)
    if target is None:
        _not_applicable("no second floating-point function to hook")
    log: dict[str, Any] = {"fires": [], "states": [], "run_state": None}

    def hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        ctx.state["n"] = ctx.state.get("n", 0) + 1
        ctx.run_state["fires"] = ctx.run_state.get("fires", 0) + 1
        log["run_state"] = ctx.run_state
        if not any(state is ctx.state for state in log["states"]):
            log["states"].append(ctx.state)
        log["fires"].append((ctx.signal, ctx.state["n"], ctx.timing))
        return out + 1

    # The cell is the only object both hooks share; capture_fields would reach CaptureOptions.
    vars(cell)[_HOOKS_KEY] = (log, target)
    _merge_intervene(cell, tl.when(tl.func(target), hook))


_HOOKS_KEY = "_fill_intervention_hooks"


def _hook_fields(cell: Cell) -> tuple[dict[str, Any], str]:
    staged = vars(cell).get(_HOOKS_KEY)
    if staged is None:
        _not_applicable("hooks feature did not prepare")
    return staged


def _after_hooks(cell: Cell) -> None:
    """Hooks fire once per site per pass with the forward signal; scratch survives; a raise refuses typed."""

    trace = cell.require_trace()
    log, target = _hook_fields(cell)
    selected = [op for op in trace.ops if op.func_name == target]
    fires = log["fires"]
    cell.expect(
        len(fires) == len(selected),
        f"hook fired {len(fires)} times for {len(selected)} selected ops",
    )
    cell.expect(
        all(signal == "forward" for signal, _, _ in fires), "a forward hook saw another signal"
    )
    cell.expect(
        sum(state.get("n", 0) for state in log["states"]) == len(fires),
        "ctx.state scratch did not survive the pass with its fire counts",
    )
    cell.expect(
        log["run_state"] is not None and log["run_state"].get("fires") == len(fires),
        "ctx.run_state did not survive the run",
    )
    marked = {op.label for op in trace.ops if op.is_edited}
    cell.expect(
        {op.label for op in selected} <= marked,
        f"hooked ops not marked replaced: {sorted({op.label for op in selected} - marked)[:8]}",
    )
    _expect_raising_hook_refuses(cell, target)


def _expect_raising_hook_refuses(cell: Cell, target: str) -> None:
    """A hook that raises refuses typed, and torch runs and captures plainly afterwards."""

    def boom(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        raise RuntimeError("behaviour net: hook raised on purpose")

    torch.manual_seed(SEED)
    try:
        tl.trace(cell.model, cell.fresh_inputs(), intervene=tl.when(tl.func(target), boom))
    except TorchLensError:
        pass
    except Exception as exc:  # noqa: BLE001 - an untyped escape is the finding
        cell.expect(False, f"a raising hook escaped untyped: {type(exc).__name__}: {exc}")
    else:
        cell.expect(False, "a raising hook did not refuse the capture")
    from tests.behaviour.harness import _plain_forward

    if not cell.case.has("stochastic"):
        cell.expect_same_leaves(
            cell.plain_output, _plain_forward(cell), "plain forward after refusal"
        )
    torch.manual_seed(SEED)
    again = tl.trace(cell.model, cell.fresh_inputs())
    cell.expect(
        len(again.ops) == len(cell.facts.labels),
        f"capture after the refusal has {len(again.ops)} ops, plain {len(cell.facts.labels)}",
    )


# --- new row: mean_ablate -----------------------------------------------------


def _apply_on_fork(trace: Any, label: str, edit: Any) -> Any:
    fork = trace.fork()
    fork.do(tl.label(label), edit)
    return fork


def _after_mean_ablate(cell: Cell) -> None:
    """``tl.mean_ablate`` matches a hand-computed mean for self, axis and Population sources."""

    trace = cell.require_trace()
    label = _interior_site(cell, trace)
    out = trace[label].out.detach().clone()
    if not bool(torch.isfinite(out).all()):
        _not_applicable(f"{label} holds non-finite values; a mean is undefined")
    fork = _apply_on_fork(trace, label, tl.mean_ablate(dim=None))
    cell.expect(
        torch.allclose(fork[label].out, torch.zeros_like(out) + out.mean(), equal_nan=True),
        f"mean_ablate(dim=None) at {label} is not the scalar mean",
    )
    _expect_cone_only(cell, trace, fork, label, "mean_ablate(dim=None)")
    _expect_edit_marks(cell, trace, fork, label, "mean_ablate", "mean_ablate(dim=None)")
    if out.dim() > 0 and out.shape[0] > 1:
        fork = _apply_on_fork(trace, label, tl.mean_ablate(dim=0))
        cell.expect(
            torch.allclose(
                fork[label].out,
                torch.zeros_like(out) + out.mean(dim=0, keepdim=True),
                equal_nan=True,
            ),
            f"mean_ablate(dim=0) at {label} is not the axis-0 mean",
        )
    donors = tl.intervention.population(
        torch.stack([out, out * 3]), origin="behaviour net: two donors from the site itself"
    )
    fork = _apply_on_fork(trace, label, tl.mean_ablate(donors, dim=None))
    cell.expect(
        torch.allclose(fork[label].out, out * 2, equal_nan=True),
        f"mean_ablate(Population, dim=None) at {label} is not the elementwise member mean",
    )
    names = [name for name, *_ in _fire_facts(fork[label])]
    cell.expect("mean_ablate" in names, f"no mean_ablate fire record on {label}: {names}")


# --- new row: steer -----------------------------------------------------------


def _after_steer(cell: Cell) -> None:
    """``tl.steer`` adds the scaled direction at the site, on a multi-pass site when the case recurs."""

    trace = cell.require_trace()
    label = _multipass_site(trace) if cell.case.has("recurrent") else None
    if label is None:
        label = _interior_site(cell, trace)
    out = trace[label].out.detach().clone()
    direction = torch.ones_like(out)
    fork = _apply_on_fork(trace, label, tl.steer(direction, 2.0))
    cell.expect(
        torch.allclose(fork[label].out, out + 2.0, equal_nan=True),
        f"steer(ones, 2.0) at {label} did not add 2",
    )
    _expect_cone_only(cell, trace, fork, label, "steer")
    _expect_edit_marks(cell, trace, fork, label, "steer", "steer")
    weak = _apply_on_fork(trace, label, tl.steer(direction, 0.5))
    cell.expect(
        torch.allclose(weak[label].out, out + 0.5, equal_nan=True),
        f"steer(ones, 0.5) at {label} did not add 0.5",
    )


# --- new row: edit_chain ------------------------------------------------------


def _after_edit_chain(cell: Cell) -> None:
    """Ordered, view-site, seeded and leaf-site edits each keep the fork promise."""

    trace = cell.require_trace()
    label = _interior_site(cell, trace)
    out = trace[label].out.detach().clone()
    chained = trace.fork()
    chained.do(tl.label(label), tl.scale(0.5))
    chained.do(tl.label(label), tl.add(1.0))
    cell.expect(
        torch.allclose(chained[label].out, out * 0.5 + 1.0, equal_nan=True),
        f"scale then add at {label} is not (x * 0.5) + 1",
    )
    reversed_fork = trace.fork()
    reversed_fork.do(tl.label(label), tl.add(1.0))
    reversed_fork.do(tl.label(label), tl.scale(0.5))
    cell.expect(
        torch.allclose(reversed_fork[label].out, (out + 1.0) * 0.5, equal_nan=True),
        f"add then scale at {label} is not (x + 1) * 0.5",
    )
    records = _fire_facts(chained[label])
    cell.expect(
        [name for name, *_ in records][-2:] == ["scale", "add"],
        f"the fire records at {label} do not list the two edits in order: {records}",
    )
    _expect_cone_only(cell, trace, chained, label, "two edits")
    view = _view_site(trace)
    if view is not None:
        fork = _apply_on_fork(trace, view, tl.scale(0.5))
        cell.expect(
            torch.allclose(fork[view].out, trace[view].out * 0.5, equal_nan=True),
            f"fork.do(scale 0.5) did not halve the view site {view}",
        )
        _expect_cone_only(cell, trace, fork, view, "view-site edit")
    noisy_a = _apply_on_fork(trace, label, tl.add_noise(0.1, seed=3))
    noisy_b = _apply_on_fork(trace, label, tl.add_noise(0.1, seed=3))
    cell.expect(
        _same(noisy_a[label].out, noisy_b[label].out),
        f"add_noise(seed=3) at {label} did not reproduce across two forks",
    )
    seeds = [getattr(record, "seed", None) for record in noisy_a[label].interventions or ()]
    cell.expect(3 in seeds, f"the add_noise fire record at {label} does not carry seed 3: {seeds}")
    leaf = _leaf_site(trace)
    if leaf is not None:
        _halve_on_fork(cell, trace, "leaf site", leaf)
    if cell.loaded is not None:
        _halve_on_fork(cell, cell.loaded, "loaded trace", label)


# --- registry extension -------------------------------------------------------


def extend_features(rows: tuple[Feature, ...]) -> tuple[Feature, ...]:
    """Return ``rows`` with the claimed intervention rows filled and the new rows appended."""

    from tests.behaviour.features import MEDIUM, Feature

    fills = {
        "intervene": {"prepare": _prepare_intervene, "after": _after_intervene},
        "fork_do": {"after": _after_fork_do},
        "bundle": {"prepare": _prepare_intervention_ready, "after": _after_bundle},
    }
    filled = tuple(
        replace(row, after=_after_bind_with(row.after))
        if row.name == "bind"
        else replace(row, **fills[row.name])
        if row.name in fills
        else row
        for row in rows
    )
    float_only = frozenset({"float_output"})
    new_rows = (
        Feature(
            "hooks",
            MEDIUM,
            "User hooks at capture (intervene=tl.when(site, fn(out, *, ctx)))",
            changes=frozenset({"forward_match"}),
            requires=float_only,
            prepare=_prepare_hooks,
            after=_after_hooks,
        ),
        Feature(
            "mean_ablate",
            MEDIUM,
            "Mean ablation on a fork (tl.mean_ablate: self, axis, Population)",
            requires=float_only,
            prepare=_prepare_intervention_ready,
            after=_after_mean_ablate,
        ),
        Feature(
            "steer",
            MEDIUM,
            "Steering on a fork (tl.steer direction, strength)",
            requires=float_only,
            prepare=_prepare_intervention_ready,
            after=_after_steer,
        ),
        Feature(
            "edit_chain",
            MEDIUM,
            "Ordered, view-site and seeded edits on a fork (do twice, add_noise)",
            requires=float_only,
            prepare=_prepare_intervention_ready,
            after=_after_edit_chain,
        ),
    )
    return (*filled, *new_rows)


def extend_toys(rows: tuple[Case, ...]) -> tuple[Case, ...]:
    """Return ``rows`` with the intervention verb's toy cases appended."""

    from tests import example_models
    from tests.behaviour.cases import _toy

    def _positive_vec() -> torch.Tensor:
        return torch.rand(5, 5) + 1.0

    def _img() -> torch.Tensor:
        return torch.rand(2, 3, 8, 8)

    return (
        *rows,
        _toy("slice_operations", example_models.SliceOperations, _positive_vec, "views", "inplace"),
        _toy("gelu", example_models.GeluModel, _img, "views"),
    )
