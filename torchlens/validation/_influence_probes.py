"""Evidence probes for posthoc decisions whose perturbed output did not move.

When the perturbation check leaves an op's output unchanged, a few posthoc
excuses would pass the edge on the op's kind alone (a bool output, a dominated
extrema operand). Each probe here re-executes the op with the perturbed
parent's one slot replaced by values chosen to cross the op's decision
boundary, and returns:

* ``True``: some substitute moved the output, so the recorded edge transmits
  value and the excuse holds with evidence;
* ``False``: every substitute left the output pinned to the saved value, so
  the recorded parent has no value influence (the capture-bug class a blanket
  pass would bless) and the caller falls through to the failure;
* ``None``: the probe cannot run (no single perturbed slot, a non-tensor
  operand, a refused substitute), and the caller keeps its disclosed fallback.

The perturbed slot is located in the positional arguments or the keyword
arguments alike, and ``out=`` is never passed to the probe call: a keyword
spelling such as ``torch.eq(ids, other=pad)`` gets the same evidence as the
positional one, and a probe can never write into a saved output buffer.
Split from ``exemptions.py`` (file-cap row), which calls these from its
posthoc decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ._loaded_callables import refuse_pointer_keywords
from ._replay_warnings import replay_warnings_ignored

if TYPE_CHECKING:
    from .._data_classes.op import Op

#: Elementwise comparison spellings whose bool outputs admit the
#: threshold-straddle probe (R08): substituting the perturbed operand with
#: the comparand itself and its two adjacent values MUST flip some element
#: of any genuine two-operand comparison, so an output pinned to the saved
#: value under all three substitutions proves the recorded parent has no
#: value influence.
_ELEMENTWISE_COMPARISON_FUNCS = frozenset(
    {
        "gt",
        "greater",
        "lt",
        "less",
        "ge",
        "greater_equal",
        "le",
        "less_equal",
        "eq",
        "ne",
        "not_equal",
        "__gt__",
        "__lt__",
        "__ge__",
        "__le__",
        "__eq__",
        "__ne__",
    }
)

_COMPARISON_KEYWORDS = ("input", "other")
"""Keyword spellings of a comparison's two operands, in positional order."""

_PROBE_REFUSALS = (TypeError, ValueError, RuntimeError, NotImplementedError, IndexError)
"""Errors a legitimate op raises when it rejects a substitute: "probe cannot run".

Dtype, shape and value refusals, unimplemented substitute dtypes, and an
out-of-range substitute index (``IndexError`` from advanced indexing,
``aten::index`` / ``Tensor.__getitem__``, as in a vmapped mask builder).
Anything else (a poisoned replay callable, a torchlens internal error,
``KeyError`` included) is a capture bug and propagates instead of reverting to
an evidence-free pass.
"""

Slot = int | str
"""A positional index or a keyword name."""


@dataclass(frozen=True)
class _ProbeCall:
    """The op's recorded call, ``out=`` dropped, with the perturbed parent's slot."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    slot: Slot

    def value(self, slot: Slot) -> Any:
        """Return the recorded value in ``slot``."""

        return self.args[slot] if isinstance(slot, int) else self.kwargs[slot]

    def substituted(self, substitute: torch.Tensor) -> tuple[list[Any], dict[str, Any]]:
        """Return fresh call arguments with the perturbed slot replaced.

        Every retained tensor operand is cloned per call (r8 R08, fable MH):
        the probed universe includes IN-PLACE ops (``logical_and_``,
        ``bitwise_*_`` masks), and executing one against the record's retained
        ``saved_args`` would mutate the capture evidence itself.
        """

        args = [_fresh(item) for item in self.args]
        kwargs = {name: _fresh(item) for name, item in self.kwargs.items()}
        if isinstance(self.slot, int):
            args[self.slot] = substitute
        else:
            kwargs[self.slot] = substitute
        return args, kwargs


def _fresh(item: Any) -> Any:
    """Return a clone of ``item`` if it is a tensor, else ``item`` unchanged."""

    return item.clone() if isinstance(item, torch.Tensor) else item


def _flat_slot_of(layer: Op, label: str) -> Slot | None:
    """Return the one positional or keyword slot ``label`` fills, else None.

    Several slots (a self-comparison ``x == x``) would move together and
    prove nothing; a nested container member or the ``out=`` destination is
    not a value operand.
    """

    positions = getattr(layer, "parent_arg_positions", None) or {}
    slots: list[Any] = [k for k, v in (positions.get("args") or {}).items() if v == label]
    slots += [k for k, v in (positions.get("kwargs") or {}).items() if v == label]
    if len(slots) != 1 or isinstance(slots[0], tuple) or slots[0] == "out":
        return None
    return slots[0]


def _probe_call(layer: Op, layers_to_perturb: list[str]) -> _ProbeCall | None:
    """Return the probe call for one perturbed parent in exactly one flat slot."""

    slot = _flat_slot_of(layer, layers_to_perturb[0]) if len(layers_to_perturb) == 1 else None
    if slot is None:
        return None
    args = tuple(getattr(layer, "saved_args", None) or ())
    saved_kwargs = getattr(layer, "saved_kwargs", None) or {}
    kwargs = {name: value for name, value in saved_kwargs.items() if name != "out"}
    present = 0 <= slot < len(args) if isinstance(slot, int) else slot in kwargs
    return _ProbeCall(args, kwargs, slot) if present else None


def _substitutes_move_output(
    layer: Op,
    call: _ProbeCall,
    build_substitutes: Callable[[], tuple[torch.Tensor, ...] | None],
) -> bool | None:
    """Re-execute ``layer.func`` per substitute; report whether any moved the output."""

    func = getattr(layer, "func", None)
    saved_output = getattr(layer, "out", None)
    if func is None or not isinstance(saved_output, torch.Tensor):
        return None
    from .._utils.tensor_utils import tensor_nanequal

    # Raised OUTSIDE the probe refusals: a probe that cannot run keeps an exemption.
    refuse_pointer_keywords(layer, call.kwargs)
    try:
        with torch.no_grad(), replay_warnings_ignored():
            substitutes = build_substitutes()
            if substitutes is None:
                return None
            for substitute in substitutes:
                args, kwargs = call.substituted(substitute)
                probe_output = func(*args, **kwargs)
                if not isinstance(probe_output, torch.Tensor):
                    return None
                if not tensor_nanequal(probe_output, saved_output, allow_tolerance=False):
                    return True
    except _PROBE_REFUSALS:
        return None
    return False


def _comparison_operand_slots(call: _ProbeCall) -> tuple[Slot, Slot] | None:
    """Return the two operand slots of a comparison call, in positional order."""

    if set(call.kwargs) - set(_COMPARISON_KEYWORDS):
        return None
    if "input" in call.kwargs and call.args:
        return None
    slots: list[Slot] = list(range(len(call.args)))
    slots += [name for name in _COMPARISON_KEYWORDS if name in call.kwargs]
    if len(slots) != 2 or call.slot not in slots:
        return None
    return slots[0], slots[1]


def _straddle_substitutes(
    parent_saved: torch.Tensor, other: Any
) -> tuple[torch.Tensor, ...] | None:
    """Return the comparand and its two adjacent representable values."""

    if isinstance(other, torch.Tensor):
        base = other.detach().to(dtype=parent_saved.dtype).broadcast_to(parent_saved.shape)
    elif isinstance(other, (bool, int, float)):
        base = torch.full_like(parent_saved, other)
    else:
        return None
    if base.dtype == torch.bool or base.is_complex():
        return None
    if base.is_floating_point():
        if not bool(torch.isfinite(base).all()):
            return None
        below = torch.nextafter(base, torch.full_like(base, float("-inf")))
        above = torch.nextafter(base, torch.full_like(base, float("inf")))
        return base, below, above
    one = torch.ones_like(base)
    return base, base - one, base + one


def _bool_comparison_straddle_probe(layer: Op, layers_to_perturb: list[str]) -> bool | None:
    """Probe a bool comparison by straddling its comparand (R08).

    Re-executes the comparison with the perturbed operand replaced by the
    comparand itself and its two adjacent representable values. For any
    genuine elementwise comparison these produce at least two distinct
    outputs, so a moved output proves the edge transmits value (the original
    perturbation never crossed the threshold) and an output pinned under all
    three proves the recorded parent has no value influence. Keyword
    spellings (``other=``, ``input=``, ``out=``) are probed like positional
    ones.

    Returns
    -------
    bool | None
        ``True`` (edge transmits value), ``False`` (provably no influence),
        or ``None`` when the probe cannot run (non-comparison func, other
        keyword arguments, multi-slot/self-comparison operands, non-finite
        comparand, a refused substitute); the caller then falls back to the
        disclosed heuristic exemption.
    """

    if layer.func_name not in _ELEMENTWISE_COMPARISON_FUNCS:
        return None
    call = _probe_call(layer, layers_to_perturb)
    operand_slots = None if call is None else _comparison_operand_slots(call)
    if call is None or operand_slots is None:
        return None
    parent_saved = call.value(call.slot)
    other_slot = operand_slots[1] if operand_slots[0] == call.slot else operand_slots[0]
    other = call.value(other_slot)
    if not isinstance(parent_saved, torch.Tensor):
        return None
    return _substitutes_move_output(layer, call, lambda: _straddle_substitutes(parent_saved, other))


def _bool_probe_battery(parent_saved: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return dtype-directed substitutes chosen to flip a value-transmitting predicate.

    Sign / zero / extreme / non-finite / inversion coverage: ``isnan``/``isfinite``
    flip on the non-finite rows, ``logical_*`` and ``any``/``all`` on the
    all-False/all-True rows, ``signbit``-style predicates on the sign rows, and
    magnitude thresholds on the extreme rows.
    """

    if parent_saved.dtype == torch.bool:
        return (torch.zeros_like(parent_saved), torch.ones_like(parent_saved), ~parent_saved)
    if parent_saved.is_complex():
        return (
            torch.zeros_like(parent_saved),
            torch.ones_like(parent_saved),
            torch.full_like(parent_saved, complex(0.0, 1.0)),
            torch.full_like(parent_saved, complex(float("nan"), 0.0)),
        )
    if parent_saved.is_floating_point():
        return (
            torch.zeros_like(parent_saved),
            torch.ones_like(parent_saved),
            -torch.ones_like(parent_saved),
            -parent_saved,
            torch.full_like(parent_saved, float("nan")),
            torch.full_like(parent_saved, float("inf")),
            torch.full_like(parent_saved, float("-inf")),
            torch.full_like(parent_saved, torch.finfo(parent_saved.dtype).max / 2),
        )
    info = torch.iinfo(parent_saved.dtype)
    substitutes = [
        torch.zeros_like(parent_saved),
        torch.ones_like(parent_saved),
        torch.full_like(parent_saved, info.max),
        torch.full_like(parent_saved, info.min),
    ]
    if info.min < 0:
        substitutes.append(-torch.ones_like(parent_saved))
    return tuple(substitutes)


def _bool_predicate_influence_probe(layer: Op, layers_to_perturb: list[str]) -> bool | None:
    """Probe a NON-comparison bool-output op with a substitution battery (R08-2).

    The comparison family gets the threshold-straddle probe; the rest of the
    bool universe (``logical_*``, ``isnan``/``isfinite``, ``isclose``,
    ``any``/``all``, ``bitwise_*`` masks) gets :func:`_bool_probe_battery` in
    the perturbed slot, whether that slot is positional or a keyword and
    whatever other keywords (``atol=``, ``dim=``) the call carried. Any flip
    proves the recorded edge transmits value, and a battery-wide pin proves
    the recorded parent has no value influence; the caller then falls through
    to the ``perturbation_insensitive`` failure exactly like the comparison
    family.

    Returns
    -------
    bool | None
        ``True`` (edge transmits value), ``False`` (provably no influence),
        or ``None`` when the probe cannot run (comparison func -- the
        straddle owns those -- multi-slot operands, non-tensor parent/output,
        a refused substitute); the caller then falls back to the disclosed
        heuristic exemption.
    """

    if layer.func_name in _ELEMENTWISE_COMPARISON_FUNCS:
        return None
    call = _probe_call(layer, layers_to_perturb)
    if call is None:
        return None
    parent_saved = call.value(call.slot)
    if not isinstance(parent_saved, torch.Tensor):
        return None
    return _substitutes_move_output(layer, call, lambda: _bool_probe_battery(parent_saved))


def binary_extrema_crossing_probe(layer: Op, layers_to_perturb: list[str]) -> bool | None:
    """Probe a dominated binary ``max``/``min``/``maximum``/``minimum`` operand.

    The dominance excuse holds only while the perturbed operand stays on the
    dominated side. Pushing it past every element of the other operand (one
    representable step beyond the other's extreme for integers, a full
    magnitude for floats) must move the output of a genuine extrema op, so a
    pinned output proves the recorded parent does not feed the op.

    Returns
    -------
    bool | None
        ``True`` (edge transmits value), ``False`` (provably no influence),
        or ``None`` when no crossing value is representable (the other operand
        already holds the dtype's extreme or a non-finite value).
    """

    call = _probe_call(layer, layers_to_perturb)
    if call is None or not isinstance(call.slot, int) or call.slot > 1 or len(call.args) < 2:
        return None
    perturbed, other = call.args[call.slot], call.args[1 - call.slot]
    if not isinstance(perturbed, torch.Tensor) or not isinstance(other, torch.Tensor):
        return None
    upward = layer.func_name in ("max", "maximum")
    return _substitutes_move_output(layer, call, lambda: _beyond_extreme(perturbed, other, upward))


def _beyond_extreme(
    perturbed: torch.Tensor, other: torch.Tensor, upward: bool
) -> tuple[torch.Tensor, ...] | None:
    """Return ``perturbed`` filled with one value beyond every element of ``other``."""

    if perturbed.dtype == torch.bool or perturbed.is_complex() or other.numel() == 0:
        return None
    extreme = (other.max() if upward else other.min()).to(perturbed.dtype)
    if perturbed.is_floating_point():
        value: float | int | None = _float_beyond(extreme, upward)
    else:
        value = _int_beyond(int(extreme), torch.iinfo(perturbed.dtype), upward)
    return None if value is None else (torch.full_like(perturbed, value),)


def _float_beyond(extreme: torch.Tensor, upward: bool) -> float | None:
    """Return a finite float a full magnitude beyond ``extreme``, else None."""

    step = torch.clamp(extreme.abs(), min=1.0)
    beyond = extreme + step if upward else extreme - step
    return float(beyond) if bool(torch.isfinite(beyond)) else None


def _int_beyond(extreme: int, info: torch.iinfo, upward: bool) -> int | None:
    """Return the integer one step beyond ``extreme``, else None at the dtype's limit."""

    if extreme == (info.max if upward else info.min):
        return None
    return extreme + (1 if upward else -1)
