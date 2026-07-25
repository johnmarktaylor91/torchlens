"""Extensible receptive-field operation-rule registry.

The registry is process-local.  Registering or replacing a rule advances its
epoch so trace-owned receptive-field solutions can invalidate stale results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

from ..capture.arg_positions import _normalize_func_name


if TYPE_CHECKING:
    from ..data_classes.op import Op


RuleCallback: TypeAlias = Callable[..., object]
"""Callback optionally attached to a receptive-field rule result.

``map_index_set`` is the authoritative backward relation. A ``map_index_set_forward``
accelerator must be accompanied by an exhaustive small-case duality golden proving equality
with its membership transpose. A rule exposing only ``unit_box`` receives an inexact transposed
envelope in projective per-unit queries; authors needing exact forward support must provide one
of the index-set callbacks.
"""


@dataclass(frozen=True)
class _RuleResult:
    """Opaque rule-result specification consumed by the RF geometry engine."""

    kind: str
    values: Mapping[str, object]
    note: str | None = None
    map_index_set: RuleCallback | None = None
    map_index_set_forward: RuleCallback | None = None
    map_interval: RuleCallback | None = None
    unit_box: RuleCallback | None = None


class ReceptiveFieldRule(Protocol):
    """Protocol implemented by one receptive-field operation rule."""

    def __call__(self, context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Build the operation's geometry specification.

        Parameters
        ----------
        context:
            Captured-operation metadata and result constructors.

        Returns
        -------
        _RuleResult
            Opaque specification interpreted by the geometry engine.
        """


class ReceptiveFieldRuleContext:
    """Metadata and result constructors available to receptive-field rules."""

    def __init__(self, op: Op) -> None:
        """Create a context for one captured operation.

        Parameters
        ----------
        op:
            Captured operation whose rule is being evaluated.
        """

        self.op = op

    @property
    def in_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return captured input shapes, omitting unavailable inputs."""

        return tuple(tuple(shape) for shape in self.op.input_shapes if shape is not None)

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Return the captured output shape."""

        return tuple(self.op.shape)

    def cfg(self, key: str, default: object = None) -> object:
        """Return a salient config value, normalizing scalars against tuple defaults.

        Parameters
        ----------
        key:
            Salient configuration key.
        default:
            Value used when the operation did not record ``key``.

        Returns
        -------
        object
            Recorded value or default.  A scalar is expanded when ``default``
            is a tuple, preserving per-axis rule call sites.
        """

        value = self.op.func_config.get(key, default)
        if isinstance(default, tuple) and not isinstance(value, tuple):
            return (value,) * len(default)
        return value

    def arg(self, name_or_pos: str | int, default: object = None) -> object:
        """Return a recorded non-tensor argument by keyword/name or position.

        Parameters
        ----------
        name_or_pos:
            Argument name or positional index.
        default:
            Value used when the argument was not captured.

        Returns
        -------
        object
            Captured non-tensor argument or ``default``.
        """

        if isinstance(name_or_pos, str):
            if name_or_pos in self.op.non_tensor_kwargs:
                return self.op.non_tensor_kwargs[name_or_pos]
            try:
                name_or_pos = tuple(self.op.arg_names).index(name_or_pos)
            except ValueError:
                return default
        if 0 <= name_or_pos < len(self.op.non_tensor_pos_args):
            return self.op.non_tensor_pos_args[name_or_pos]
        return default

    def parent_role(self, parent_index: int) -> str | None:
        """Return the captured argument role for one parent, if available.

        Parameters
        ----------
        parent_index:
            Index into the operation's ordered parent list.

        Returns
        -------
        str | None
            ``"args:<index>"`` or ``"kwargs:<name>"`` when the parent is
            associated with a captured argument position.
        """

        if parent_index < 0 or parent_index >= len(self.op.parents):
            return None
        parent_label = self.op.parents[parent_index]
        positions = self.op.parent_arg_positions
        for position, label in positions.get("args", {}).items():
            if label == parent_label:
                return f"args:{position}"
        for position, label in positions.get("kwargs", {}).items():
            if label == parent_label:
                return f"kwargs:{position}"
        return None

    def window(
        self,
        *,
        kernel: object,
        stride: object = 1,
        padding: object = 0,
        dilation: object = 1,
        exact: bool = True,
        channel_dependency: str | None = None,
        note: str | None = None,
        **callbacks: RuleCallback,
    ) -> _RuleResult:
        """Construct a windowed-operation result specification."""

        values = {
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "exact": exact,
        }
        if channel_dependency is not None:
            values["channel_dependency"] = channel_dependency
        return self._result(
            "window",
            values,
            note,
            callbacks,
        )

    def window_edges(
        self,
        per_axis_edges: object,
        *,
        exact: bool,
        note: str | None = None,
        **callbacks: RuleCallback,
    ) -> _RuleResult:
        """Construct a raw two-edge window result specification."""

        return self._result(
            "window_edges", {"per_axis_edges": per_axis_edges, "exact": exact}, note, callbacks
        )

    def piecewise(
        self,
        segments: object,
        *,
        exact: bool = False,
        note: str | None = None,
        **callbacks: RuleCallback,
    ) -> _RuleResult:
        """Construct a piecewise mapping result specification."""

        return self._result("piecewise", {"segments": segments, "exact": exact}, note, callbacks)

    def passthrough(self, *, note: str | None = None, **callbacks: RuleCallback) -> _RuleResult:
        """Construct an identity-mapping result specification."""

        return self._result("passthrough", {}, note, callbacks)

    def full(
        self,
        axes: object = "all_nonmatched",
        *,
        exact: bool = True,
        note: str | None = None,
        **callbacks: RuleCallback,
    ) -> _RuleResult:
        """Construct a whole-extent dependence result specification."""

        return self._result("full", {"axes": axes, "exact": exact}, note, callbacks)

    def axis_map(
        self,
        out_to_parent_axis: Mapping[int, int],
        *,
        note: str | None = None,
        **callbacks: RuleCallback,
    ) -> _RuleResult:
        """Construct an exact output-to-parent axis mapping specification."""

        return self._result(
            "axis_map", {"out_to_parent_axis": dict(out_to_parent_axis)}, note, callbacks
        )

    def dissolve(self, *, note: str | None = None) -> _RuleResult:
        """Construct a grid-identity-dissolving result specification."""

        return self._result("dissolve", {}, note, {})

    def data_dependent(self, note: str) -> _RuleResult:
        """Construct a data-dependent result specification."""

        return self._result("data_dependent", {}, note, {})

    def unknown(self, note: str) -> _RuleResult:
        """Construct an unknown-geometry result specification."""

        return self._result("unknown", {}, note, {})

    def unsupported(self, note: str) -> _RuleResult:
        """Construct an unsupported-operation result specification."""

        return self._result("unsupported", {}, note, {})

    @staticmethod
    def _result(
        kind: str,
        values: Mapping[str, object],
        note: str | None,
        callbacks: Mapping[str, RuleCallback],
    ) -> _RuleResult:
        """Build one immutable opaque result specification."""

        allowed_callbacks = {
            "map_index_set",
            "map_index_set_forward",
            "map_interval",
            "unit_box",
        }
        unknown_callbacks = set(callbacks).difference(allowed_callbacks)
        if unknown_callbacks:
            names = ", ".join(sorted(unknown_callbacks))
            raise TypeError(f"Unsupported receptive-field rule callback(s): {names}.")
        return _RuleResult(
            kind=kind,
            values=MappingProxyType(dict(values)),
            note=note,
            map_index_set=callbacks.get("map_index_set"),
            map_index_set_forward=callbacks.get("map_index_set_forward"),
            map_interval=callbacks.get("map_interval"),
            unit_box=callbacks.get("unit_box"),
        )


_RF_RULES: dict[str, ReceptiveFieldRule] = {}
_RF_RULES_EPOCH = 0
_BUILTIN_RF_RULES: Mapping[str, ReceptiveFieldRule] | None = None


def register_rf_rule(
    *func_names: str, replace: bool = False
) -> Callable[[ReceptiveFieldRule], ReceptiveFieldRule]:
    """Register a receptive-field rule for one or more operation names.

    Parameters
    ----------
    *func_names:
        Captured operation names, normalized before registry lookup.
    replace:
        Whether existing registrations may be replaced.

    Returns
    -------
    collections.abc.Callable
        Decorator that registers and returns the supplied rule.

    Raises
    ------
    ValueError
        If no names are supplied, a name is invalid, or a duplicate is
        registered without ``replace=True``.
    """

    if not func_names:
        raise ValueError("register_rf_rule requires at least one function name.")
    normalized_names = tuple(_normalize_rule_name(name) for name in func_names)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("register_rf_rule received duplicate normalized function names.")

    def decorator(rule: ReceptiveFieldRule) -> ReceptiveFieldRule:
        """Register one rule and return it unchanged."""

        global _RF_RULES_EPOCH
        duplicates = tuple(name for name in normalized_names if name in _RF_RULES)
        if duplicates and not replace:
            names = ", ".join(duplicates)
            raise ValueError(f"Receptive-field rule already registered for: {names}.")
        for name in normalized_names:
            _RF_RULES[name] = rule
        _RF_RULES_EPOCH += 1
        return rule

    return decorator


def rules() -> Mapping[str, ReceptiveFieldRule]:
    """Return an immutable snapshot of the registered receptive-field rules."""

    return MappingProxyType(dict(_RF_RULES))


def _rf_rules_epoch() -> int:
    """Return the registry epoch used to invalidate cached RF solutions."""

    return _RF_RULES_EPOCH


def _install_builtin_rule_pack() -> None:
    """Record or restore the built-in rule pack without replacing custom rules.

    The first call follows the built-in modules' decorator registrations and
    records their canonical rules. Later imports are idempotent: missing
    built-ins are restored, while existing registrations (including intentional
    custom replacements) are left untouched. Restoring one or more entries
    advances the registry epoch once so cached trace solutions are invalidated.

    Returns
    -------
    None
        The process-global registry is updated only when a built-in entry is
        absent.
    """

    global _BUILTIN_RF_RULES, _RF_RULES_EPOCH
    if _BUILTIN_RF_RULES is None:
        _BUILTIN_RF_RULES = MappingProxyType(dict(_RF_RULES))
        return

    missing_rules = {
        name: rule for name, rule in _BUILTIN_RF_RULES.items() if name not in _RF_RULES
    }
    if missing_rules:
        _RF_RULES.update(missing_rules)
        _RF_RULES_EPOCH += 1


def _normalize_rule_name(func_name: str) -> str:
    """Validate and normalize one captured operation name."""

    if not isinstance(func_name, str) or not func_name:
        raise ValueError("Receptive-field rule names must be non-empty strings.")
    return _normalize_func_name(func_name)


__all__ = ["ReceptiveFieldRule", "ReceptiveFieldRuleContext", "register_rf_rule", "rules"]
