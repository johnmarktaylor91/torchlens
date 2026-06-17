"""Single Bundle type for intervention-ready TorchLens model logs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from .._trace_state import TraceState
from ._metrics import is_scalar_like, relative_l1_scalar, resolve_metric
from ._super.super_logs import (
    SuperBufferAccessor,
    SuperGradFnAccessor,
    SuperGradFnCallAccessor,
    SuperModuleAccessor,
    SuperModuleCallAccessor,
    SuperParamAccessor,
)
from ._super.super_op import (
    SuperLayerAccessor,
    SuperOp,
    SuperOpAccessor,
    TraceAccessor,
)
from ._topology.topology import Supergraph, build_supergraph
from .errors import (
    BaselineUndeterminedError,
    BundleMemberError,
    BundleRelationshipError,
)
from .resolver import resolve_sites
from .types import Relationship

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from torch import nn

    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


_RELATIONSHIP_RANK: dict[Relationship, int] = {
    Relationship.UNKNOWN: 0,
    Relationship.DIFF_MODEL: 0,
    Relationship.SAME_PARAM_SHAPES: 1,
    Relationship.SHARED_ARCHITECTURE: 1,
    Relationship.SHARED_GRAPH_DIFFERENT_INPUT: 2,
    Relationship.SHARED_GRAPH_SAME_INPUT: 3,
    Relationship.SAME_MODEL_OBJECT_AT_CAPTURE: 4,
    Relationship.SAME_OBJECT: 5,
}

_REQUIRED_RELATIONSHIPS: dict[str, Relationship] = {
    "node": Relationship.SAME_PARAM_SHAPES,
    "compare_at": Relationship.SHARED_GRAPH_SAME_INPUT,
    "most_changed": Relationship.SHARED_GRAPH_SAME_INPUT,
    "diff": Relationship.SHARED_GRAPH_SAME_INPUT,
}

_OP_LABEL_RE = re.compile(r"^.+_\d+_\d+$")
_BARE_LAYER_LABEL_RE = re.compile(r"^.+_\d+$")
_PASS_CALL_RE = re.compile(r"^(.+):(\d+)$")
_COMMON_PARAM_SUFFIXES = (".weight", ".bias")
_COMMON_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")
_BUNDLE_ACCESSOR_NAMES = (
    "ops",
    "layers",
    "modules",
    "params",
    "buffers",
    "grad_fns",
    "module_calls",
    "grad_fn_calls",
)


class AmbiguousLabelError(KeyError):
    """Raised when ``Bundle.at`` finds a label in multiple accessors."""


class _BundleAccessorClassView:
    """Class-level placeholder for Bundle accessor properties."""


class _BundleAccessorProperty:
    """Property that exposes Bundle accessors without inflating method counts."""

    def __init__(self, accessor_cls: type[Any]) -> None:
        """Initialize the property.

        Parameters
        ----------
        accessor_cls:
            Accessor class to instantiate for Bundle instances.
        """

        self._accessor_cls = accessor_cls
        self._class_view = _BundleAccessorClassView()

    def __get__(
        self,
        instance: "Bundle | None",
        owner: type["Bundle"] | None = None,
    ) -> Any:
        """Return a class view or an accessor on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Class view for class access, otherwise the instantiated accessor.
        """

        if instance is None:
            return self._class_view
        return self._accessor_cls(instance)


class _BundleStructuralProperty:
    """Descriptor exposing Bundle predicates without counting as a method."""

    def __init__(self, getter: Callable[["Bundle"], Any]) -> None:
        """Initialize the descriptor.

        Parameters
        ----------
        getter:
            Instance getter to call for Bundle instances.
        """

        self._getter = getter
        self.__doc__ = getter.__doc__

    def __get__(self, instance: "Bundle | None", owner: type["Bundle"]) -> Any:
        """Return the descriptor on classes or computed value on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Descriptor for class access, otherwise the computed property value.
        """

        if instance is None:
            return self
        return self._getter(instance)


class Bundle:
    """Flat container of Traces with relationship-gated operations.

    Parameters
    ----------
    members:
        Mapping of member names to ``Trace`` objects, sequence of logs,
        or sequence of ``(name, log)`` tuples.
    names:
        Optional names for a sequence of logs.
    baseline:
        Optional baseline member name or ``Trace`` reference.
    """

    def __init__(
        self,
        members: Mapping[str, "Trace"] | Sequence["Trace"] | Sequence[tuple[str, "Trace"]],
        *,
        names: Sequence[str] | None = None,
        baseline: str | "Trace" | None = None,
    ) -> None:
        """Initialize a flat bundle without eagerly building a supergraph."""

        parsed = self._parse_members(members, names=names)
        self._members: OrderedDict[str, Trace] = OrderedDict(parsed)
        self._supergraph: Supergraph | None = None
        self._capacity: int | None = None
        self._baseline_name: str | None = self._resolve_baseline_name(baseline)

    def __len__(self) -> int:
        """Return the number of bundle members.

        Returns
        -------
        int
            Number of member logs.
        """

        return len(self._members)

    def __iter__(self) -> Iterator["Trace"]:
        """Iterate member logs in insertion order.

        Returns
        -------
        Iterator[Trace]
            Iterator over member logs.
        """

        return iter(self._members.values())

    def __contains__(self, name: str) -> bool:
        """Return whether a member name is present.

        Parameters
        ----------
        name:
            Member name.

        Returns
        -------
        bool
            Whether the bundle contains ``name``.
        """

        return name in self._members

    def __repr__(self) -> str:
        """Return an informative one-line bundle representation."""

        baseline = self._baseline_name if self._baseline_name is not None else "None"
        return (
            f"Bundle(n_members={len(self)}, names={self.names!r}, "
            f"baseline={baseline!r}, structurally_consistent={self.is_structurally_consistent})"
        )

    def __getitem__(self, name: str) -> "Trace":
        """Return a member by name.

        Parameters
        ----------
        name:
            Member name.

        Returns
        -------
        Trace
            Matching member log.
        """

        if not isinstance(name, str):
            raise TypeError(f"Bundle indices must be member names, got {type(name).__name__}.")
        return self._members[name]

    def save(
        self,
        path: str | Path,
        *,
        level: str = "portable",
        overwrite: bool = False,
    ) -> None:
        """Save this bundle as a unified ``.tlspec`` directory.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        overwrite:
            Whether an existing destination may be replaced.
        """

        from .._io.tlspec import _TlSpecWriter

        _TlSpecWriter.write_bundle(
            bundle=self,
            path=path,
            save_level=level,
            overwrite=overwrite,
        )

    def __getattr__(self, name: str) -> Any:
        """Return budget-preserving Phase 8 helper custom_methods.

        Parameters
        ----------
        name:
            Requested attribute name.

        Returns
        -------
        Any
            Callable helper bound to this bundle.

        Raises
        ------
        AttributeError
            If ``name`` is not a dynamic bundle helper.
        """

        dynamic_custom_methods: dict[str, Callable[..., Any]] = {
            "aligned_pairs": _bundle_aligned_pairs,
            "compare": _bundle_compare,
            "delta_map": _bundle_delta_map,
            "norm_delta": _bundle_norm_delta,
            "output_delta": _bundle_output_delta,
            "show_diff": _bundle_show_diff,
        }
        helper = dynamic_custom_methods.get(name)
        if helper is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return helper.__get__(self, type(self))

    @property
    def names(self) -> list[str]:
        """Return member names in bundle order.

        Returns
        -------
        list[str]
            Member names.
        """

        return list(self._members)

    @property
    def members(self) -> dict[str, "Trace"]:
        """Return a shallow copy of the member mapping.

        Returns
        -------
        dict[str, Trace]
            Member mapping.
        """

        return dict(self._members)

    @property
    def traces(self) -> TraceAccessor:
        """Return dict-like access to member traces.

        Returns
        -------
        TraceAccessor
            Bundle trace accessor.
        """

        return TraceAccessor(dict(self._members))

    @property
    def ops(self) -> SuperOpAccessor:
        """Return cross-member access to pass-qualified Op labels.

        Returns
        -------
        SuperOpAccessor
            Bundle op accessor.
        """

        return SuperOpAccessor(self)

    @property
    def layers(self) -> SuperLayerAccessor:
        """Return cross-member access to aggregate Layer labels.

        Returns
        -------
        SuperLayerAccessor
            Bundle layer accessor.
        """

        return SuperLayerAccessor(self)

    modules = _BundleAccessorProperty(SuperModuleAccessor)
    buffers = _BundleAccessorProperty(SuperBufferAccessor)
    params = _BundleAccessorProperty(SuperParamAccessor)
    grad_fns = _BundleAccessorProperty(SuperGradFnAccessor)
    module_calls = _BundleAccessorProperty(SuperModuleCallAccessor)
    grad_fn_calls = _BundleAccessorProperty(SuperGradFnCallAccessor)

    @_BundleStructuralProperty
    def is_structurally_consistent(self) -> bool:
        """Return whether all members share the same graph-shape hash.

        Returns
        -------
        bool
            Whether every member has an equal ``graph_shape_hash`` value.
        """

        hashes = {getattr(member, "graph_shape_hash", None) for member in self._members.values()}
        return len(hashes) == 1

    @_BundleStructuralProperty
    def shared_op_labels(self) -> list[str]:
        """Return op labels present in every member.

        Returns
        -------
        list[str]
            Shared pass-qualified op labels, ordered by the first member.
        """

        return self._shared(lambda trace: getattr(trace, "op_labels", ()))

    @_BundleStructuralProperty
    def divergent_op_labels(self) -> list[str]:
        """Return op labels not present in every member.

        Returns
        -------
        list[str]
            Pass-qualified op labels present in some but not all members.
        """

        return self._divergent(lambda trace: getattr(trace, "op_labels", ()))

    @_BundleStructuralProperty
    def shared_layer_labels(self) -> list[str]:
        """Return layer labels present in every member.

        Returns
        -------
        list[str]
            Shared aggregate layer labels, ordered by the first member.
        """

        return self._shared(lambda trace: getattr(trace, "layer_labels", ()))

    @_BundleStructuralProperty
    def divergent_layer_labels(self) -> list[str]:
        """Return layer labels not present in every member.

        Returns
        -------
        list[str]
            Aggregate layer labels present in some but not all members.
        """

        return self._divergent(lambda trace: getattr(trace, "layer_labels", ()))

    @_BundleStructuralProperty
    def shared_module_addresses(self) -> list[str]:
        """Return module addresses present in every member.

        Returns
        -------
        list[str]
            Shared module addresses, ordered by the first member.
        """

        return self._shared(lambda trace: trace.modules.keys())

    @_BundleStructuralProperty
    def divergent_module_addresses(self) -> list[str]:
        """Return module addresses not present in every member.

        Returns
        -------
        list[str]
            Module addresses present in some but not all members.
        """

        return self._divergent(lambda trace: trace.modules.keys())

    @_BundleStructuralProperty
    def shared_param_names(self) -> list[str]:
        """Return parameter paths present in every member.

        Returns
        -------
        list[str]
            Shared parameter name paths, ordered by the first member.
        """

        return self._shared(lambda trace: trace.params.keys())

    @_BundleStructuralProperty
    def divergent_param_names(self) -> list[str]:
        """Return parameter paths not present in every member.

        Returns
        -------
        list[str]
            Parameter name paths present in some but not all members.
        """

        return self._divergent(lambda trace: trace.params.keys())

    @_BundleStructuralProperty
    def shared_buffer_names(self) -> list[str]:
        """Return buffer paths present in every member.

        Returns
        -------
        list[str]
            Shared buffer name paths, ordered by the first member.
        """

        return self._shared(lambda trace: trace.buffers.keys())

    @_BundleStructuralProperty
    def divergent_buffer_names(self) -> list[str]:
        """Return buffer paths not present in every member.

        Returns
        -------
        list[str]
            Buffer name paths present in some but not all members.
        """

        return self._divergent(lambda trace: trace.buffers.keys())

    @_BundleStructuralProperty
    def shared_grad_fn_labels(self) -> list[str]:
        """Return grad-fn labels present in every member.

        Returns
        -------
        list[str]
            Shared grad-fn labels, ordered by the first member.
        """

        return self._shared(lambda trace: trace.grad_fns.keys())

    @_BundleStructuralProperty
    def divergent_grad_fn_labels(self) -> list[str]:
        """Return grad-fn labels not present in every member.

        Returns
        -------
        list[str]
            Grad-fn labels present in some but not all members.
        """

        return self._divergent(lambda trace: trace.grad_fns.keys())

    @property
    def baseline_name(self) -> str | None:
        """Return the configured baseline member name, if any.

        Returns
        -------
        str | None
            Baseline member name.
        """

        return self._baseline_name

    @property
    def supergraph(self) -> Supergraph:
        """Return the lazily built bundle supergraph.

        Returns
        -------
        Supergraph
            Cached union graph for the bundle members.
        """

        return self._ensure_supergraph()

    def node(self, site: Any) -> SuperOp:
        """Return a view of one resolved site across all members.

        Parameters
        ----------
        site:
            Selector-like site query.

        Returns
        -------
        SuperOp
            Dict-keyed view over matching layer pass records.
        """

        self._require_relationship("node", _REQUIRED_RELATIONSHIPS["node"])
        self._ensure_supergraph()
        layer_members: dict[str, Op] = {}
        failures: dict[str, str] = {}
        for name, log in self._members.items():
            try:
                table = resolve_sites(log, site, max_fanout=1)
                layer_members[name] = cast("Op", table.first())
            except Exception as exc:  # noqa: BLE001 - rewrapped with member context
                failures[name] = str(exc)
        if failures:
            detail = "; ".join(f"{name}: {message}" for name, message in failures.items())
            raise BundleMemberError(f"site {site!r} failed to resolve for bundle members: {detail}")
        return SuperOp.from_members(site, layer_members)

    def at(self, label: str) -> Any:
        """Return the matching cross-member Super view for ``label``.

        Parameters
        ----------
        label:
            Label from any Bundle accessor family.

        Returns
        -------
        Any
            Matching ``Super*`` view.

        Raises
        ------
        AmbiguousLabelError
            If an unrecognized-format label is present in multiple accessors.
        KeyError
            If the label is absent from every accessor.
        TypeError
            If ``label`` is not a string.
        """

        if not isinstance(label, str):
            raise TypeError(f"Bundle.at labels must be strings, got {type(label).__name__}.")

        preferred_names = self._preferred_accessor_names(label)
        for accessor_name in preferred_names:
            try:
                return self._accessor_by_name(accessor_name)[label]
            except KeyError:
                continue

        remaining_names = [
            accessor_name
            for accessor_name in _BUNDLE_ACCESSOR_NAMES
            if accessor_name not in preferred_names
        ]
        matches = self._matching_accessor_names(label, remaining_names)
        if len(matches) == 1:
            return self._accessor_by_name(matches[0])[label]
        if len(matches) > 1:
            raise AmbiguousLabelError(self._ambiguous_label_message(label, matches))
        raise KeyError(self._missing_label_message(label))

    def _accessor_by_name(self, accessor_name: str) -> Any:
        """Return one label accessor by public Bundle attribute name.

        Parameters
        ----------
        accessor_name:
            Accessor attribute name.

        Returns
        -------
        Any
            Accessor object.
        """

        return getattr(self, accessor_name)

    def _preferred_accessor_names(self, label: str) -> list[str]:
        """Return format-preferred accessors for ``label`` in dispatch order.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        list[str]
            Accessor names to try before key-presence fallback.
        """

        call_base = self._pass_call_base(label)
        if call_base is not None:
            if self._looks_like_pass_qualified_op_label(call_base):
                return ["ops"]
            if self._looks_like_grad_fn_label(call_base):
                return ["grad_fn_calls"]
            return ["module_calls"]

        if self._looks_like_pass_qualified_op_label(label):
            return ["ops"]
        if self._looks_like_bare_layer_label(label):
            return ["layers"]
        if self._looks_like_module_address(label):
            return ["modules"]
        if self._looks_like_param_name(label):
            return ["params"]
        if self._looks_like_buffer_name(label):
            return ["buffers"]
        if self._looks_like_grad_fn_label(label):
            return ["grad_fns"]
        return []

    def _matching_accessor_names(
        self,
        label: str,
        accessor_names: Sequence[str],
    ) -> list[str]:
        """Return accessors that contain ``label``.

        Parameters
        ----------
        label:
            Candidate label.
        accessor_names:
            Accessor names to inspect.

        Returns
        -------
        list[str]
            Public accessor names where ``label`` resolves.
        """

        matches: list[str] = []
        for accessor_name in accessor_names:
            accessor = self._accessor_by_name(accessor_name)
            if label in accessor:
                matches.append(accessor_name)
        return matches

    def _missing_label_message(self, label: str) -> str:
        """Build a missing-label error message with accessor suggestions.

        Parameters
        ----------
        label:
            Missing label.

        Returns
        -------
        str
            Error message.
        """

        suggestions: list[str] = []
        seen: set[str] = set()
        for accessor_name in _BUNDLE_ACCESSOR_NAMES:
            accessor = self._accessor_by_name(accessor_name)
            for suggestion in accessor._suggest(label):
                if suggestion not in seen:
                    seen.add(suggestion)
                    suggestions.append(suggestion)
        if suggestions:
            suggestion_str = ", ".join(repr(suggestion) for suggestion in suggestions)
            return f"Label {label!r} not found. Did you mean {suggestion_str}?"
        return f"Label {label!r} not found."

    def _ambiguous_label_message(self, label: str, matches: Sequence[str]) -> str:
        """Build an ambiguity error message for ``Bundle.at``.

        Parameters
        ----------
        label:
            Ambiguous label.
        matches:
            Public accessor names that contain the label.

        Returns
        -------
        str
            Error message.
        """

        match_str = " and ".join(f"bundle.{name}" for name in matches)
        disambiguators = " or ".join(f"bundle.{name}[{label!r}]" for name in matches)
        return f"Label {label!r} matches {match_str}; use {disambiguators} explicitly."

    @staticmethod
    def _pass_call_base(label: str) -> str | None:
        """Return the base label for ``base:N`` call labels.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        str | None
            Base label when ``label`` has a numeric call suffix.
        """

        match = _PASS_CALL_RE.match(label)
        return match.group(1) if match is not None else None

    @staticmethod
    def _looks_like_pass_qualified_op_label(label: str) -> bool:
        """Return whether ``label`` has pass-qualified op label shape.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label resembles ``op_type_group_pass``.
        """

        return _OP_LABEL_RE.match(label) is not None

    @staticmethod
    def _looks_like_bare_layer_label(label: str) -> bool:
        """Return whether ``label`` has aggregate layer label shape.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label resembles ``op_type_group`` but not an op label.
        """

        return _BARE_LAYER_LABEL_RE.match(label) is not None and _OP_LABEL_RE.match(label) is None

    @staticmethod
    def _looks_like_module_address(label: str) -> bool:
        """Return whether ``label`` resembles a module address.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label has dotted or numeric module-address shape.
        """

        if ":" in label:
            return False
        if label.isdecimal():
            return True
        return "." in label and not (
            label.endswith(_COMMON_PARAM_SUFFIXES) or label.endswith(_COMMON_BUFFER_SUFFIXES)
        )

    @staticmethod
    def _looks_like_param_name(label: str) -> bool:
        """Return whether ``label`` resembles a parameter path.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label has a common parameter suffix.
        """

        return label.endswith(_COMMON_PARAM_SUFFIXES)

    @staticmethod
    def _looks_like_buffer_name(label: str) -> bool:
        """Return whether ``label`` resembles a buffer path.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label has a common buffer suffix.
        """

        return label.endswith(_COMMON_BUFFER_SUFFIXES)

    @staticmethod
    def _looks_like_grad_fn_label(label: str) -> bool:
        """Return whether ``label`` resembles a grad-fn label.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label has TorchLens backward grad-fn label shape.
        """

        return "_back_" in label

    def add(
        self,
        log_or_logs: "Trace | Sequence[Trace]",
        names: str | Sequence[str] | None = None,
    ) -> "Bundle":
        """Add one or more member logs and invalidate the cached supergraph.

        Parameters
        ----------
        log_or_logs:
            Model log or logs to add.
        names:
            Optional member name or names.

        Returns
        -------
        Bundle
            This bundle.
        """

        logs = self._coerce_trace_list(log_or_logs, arg_name="log_or_logs")
        member_names = self._coerce_optional_name_list(names, count=len(logs))
        for index, log in enumerate(logs):
            member_name = self._derive_name(
                log,
                name=member_names[index],
                index=len(self._members),
            )
            if member_name in self._members:
                raise ValueError(f"Bundle member names must be unique; duplicate {member_name!r}.")
            self._members[member_name] = log
        self._supergraph = None
        self._enforce_capacity()
        return self

    def remove(
        self, name_or_names: str | "Trace" | Sequence[str | "Trace"]
    ) -> "Trace | list[Trace]":
        """Remove and return one or more members by name or Trace object.

        Parameters
        ----------
        name_or_names:
            Member name, Trace object, or a list of either.

        Returns
        -------
        Trace | list[Trace]
            Removed member, or removed members for list input.
        """

        is_many = self._is_list_like(name_or_names)
        names = self._coerce_member_name_list(name_or_names)
        removed: list[Trace] = []
        for name in names:
            log = self._members.pop(name)
            removed.append(log)
            if self._baseline_name == name:
                self._baseline_name = None
        self._supergraph = None
        return removed if is_many else removed[0]

    def remove_except(self, keep: str | "Trace" | Sequence[str | "Trace"]) -> None:
        """Remove every member whose name is not listed in ``keep``.

        Parameters
        ----------
        keep:
            Member name, Trace object, or list of either to retain.

        Returns
        -------
        None
            The bundle is mutated in place.
        """

        keep_set = set(self._coerce_member_name_list(keep))
        unknown = keep_set - set(self._members)
        if unknown:
            raise KeyError(f"Unknown bundle member(s): {sorted(unknown)}")
        self._members = OrderedDict(
            (name, log) for name, log in self._members.items() if name in keep_set
        )
        if self._baseline_name is not None and self._baseline_name not in self._members:
            self._baseline_name = None
        self._supergraph = None

    @property
    def capacity(self) -> int | None:
        """Return the configured member capacity.

        Returns
        -------
        int | None
            Member capacity, or ``None`` when uncapped.
        """

        return self._capacity

    @capacity.setter
    def capacity(self, n: int | None) -> None:
        """Set member capacity with LRU-style eviction while preserving the baseline.

        Parameters
        ----------
        n:
            Maximum member count, or ``None`` to remove the cap.
        """

        if n is None:
            self._capacity = None
            return
        if n < 1:
            raise ValueError("Bundle capacity must be at least 1.")
        self._capacity = n
        self._enforce_capacity()

    def set_capacity(self, n: int | None) -> "Bundle":
        """Set member capacity and return this bundle.

        Parameters
        ----------
        n:
            Maximum member count, or ``None`` to remove the cap.

        Returns
        -------
        Bundle
            This bundle.
        """

        self.capacity = n
        return self

    def clear(self) -> None:
        """Remove all non-baseline members.

        Returns
        -------
        None
            The bundle is mutated in place.
        """

        if self._baseline_name is not None and self._baseline_name in self._members:
            baseline = self._members[self._baseline_name]
            self._members = OrderedDict([(self._baseline_name, baseline)])
        else:
            self._members.clear()
            self._baseline_name = None
        self._supergraph = None

    def do(self, *args: Any, **kwargs: Any) -> "Bundle":
        """Apply ``Trace.do`` to every member.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.do(*args, **kwargs)
        return self

    def fork(self, name: str | None = None) -> "Bundle":
        """Fork all member logs into a new bundle.

        Parameters
        ----------
        name:
            Optional suffix prefix for forked member names.

        Returns
        -------
        Bundle
            New bundle containing forked logs.
        """

        forked: OrderedDict[str, Trace] = OrderedDict()
        for member_name, member in self._members.items():
            fork_name = f"{name}_{member_name}" if name is not None else None
            forked[member_name] = member.fork(name=fork_name)
        return Bundle(forked, baseline=self._baseline_name)

    def attach_hooks(self, *args: Any, **kwargs: Any) -> "Bundle":
        """Apply ``Trace.attach_hooks`` to every member.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.attach_hooks(*args, **kwargs)
        return self

    def push(self, **kwargs: Any) -> "Bundle":
        """Push the edit downstream through all member logs.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.push(**kwargs)
        return self

    def replay(self, **kwargs: Any) -> "Bundle":
        """Deprecated alias for :meth:`push`.

        Returns
        -------
        Bundle
            This bundle.
        """

        from .._deprecations import warn_deprecated_alias

        warn_deprecated_alias("Bundle.replay", "Bundle.push")
        return self.push(**kwargs)

    def run(self, model: "nn.Module", x: Any = None, **kwargs: Any) -> "Bundle":
        """Run all member logs with a supplied model and input.

        Parameters
        ----------
        model:
            Model forwarded to each member.
        x:
            Forward input.

        Returns
        -------
        Bundle
            This bundle.
        """

        for member in self._members.values():
            member.run(model, x, **kwargs)
        return self

    def rerun(self, model: "nn.Module", x: Any = None, **kwargs: Any) -> "Bundle":
        """Deprecated alias for :meth:`run`.

        Parameters
        ----------
        model:
            Model forwarded to each member.
        x:
            Forward input.

        Returns
        -------
        Bundle
            This bundle.
        """

        from .._deprecations import warn_deprecated_alias

        warn_deprecated_alias("Bundle.rerun", "Bundle.run")
        return self.run(model, x, **kwargs)

    def apply(self, fn: Callable[["Trace"], Any]) -> dict[str, Any]:
        """Apply a function independently to each member.

        Parameters
        ----------
        fn:
            Callable receiving one member log.

        Returns
        -------
        dict[str, Any]
            Results keyed by member name.
        """

        return {name: fn(member) for name, member in self._members.items()}

    def joint_metric(self, fn: Callable[["Bundle"], Any]) -> Any:
        """Apply a function to the bundle as a whole.

        Parameters
        ----------
        fn:
            Callable receiving this bundle.

        Returns
        -------
        Any
            Return value from ``fn``.
        """

        return fn(self)

    def show(self, method: str = "graph", **kwargs: Any) -> dict[str, str | None]:
        """Render each bundle member graph as a member-keyed strip.

        Parameters
        ----------
        method:
            Display method forwarded to each member's :meth:`Trace.show`.
        **kwargs:
            Forwarded to each member's :meth:`Trace.show`. When an output
            path is supplied, member names are appended to produce one artifact
            per log. ``vis_opt='none'`` is accepted and returns without
            rendering, matching ``Trace.show``.

        Returns
        -------
        dict[str, str | None]
            Member-keyed render results. Values are DOT source strings when
            rendering occurs, or ``None`` for skipped ``vis_opt='none'`` calls.
        """

        if kwargs.get("vis_opt") == "none" or kwargs.get("vis_mode") == "none":
            return {name: None for name in self._members}

        base_outpath = kwargs.get("vis_outpath")
        results: dict[str, str | None] = {}
        for name, member in self._members.items():
            member_kwargs = dict(kwargs)
            if isinstance(base_outpath, str):
                member_kwargs["vis_outpath"] = f"{base_outpath}_{name}"
            if method == "repr":
                results[name] = repr(member)
            elif method == "html":
                results[name] = member._repr_html_()
            else:
                results[name] = member.draw(**member_kwargs)
        return results

    def compare_at(self, site: Any) -> torch.Tensor:
        """Return pairwise out differences at a site.

        Parameters
        ----------
        site:
            Selector-like site query.

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix.
        """

        self._require_relationship("compare_at", _REQUIRED_RELATIONSHIPS["compare_at"])
        return self.node(site).diff_pair()

    def most_changed(
        self,
        baseline: str | "Trace" | None = None,
        *,
        top_k: int = 10,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
    ) -> list[tuple[str, float]]:
        """Rank sites by out distance from a baseline.

        Parameters
        ----------
        baseline:
            Optional baseline override.
        top_k:
            Maximum rows to return.
        metric:
            Pairwise tensor metric.

        Returns
        -------
        list[tuple[str, float]]
            ``(site_label, score)`` rows sorted descending.
        """

        self._require_relationship("most_changed", _REQUIRED_RELATIONSHIPS["most_changed"])
        baseline_name = self._baseline_or_raise(baseline)
        baseline_log = self._members[baseline_name]
        metric_fn = resolve_metric(metric)
        scored: list[tuple[str, float]] = []
        for site in getattr(baseline_log, "layer_list", []):
            try:
                view = self.node(site.layer_label)
            except BundleMemberError:
                continue
            outs = {
                name: getattr(member, "out", None)
                for name, member in view.members.items()
                if getattr(member, "has_saved_activation", False)
            }
            base = outs.get(baseline_name)
            if not isinstance(base, torch.Tensor):
                continue
            distances: list[float] = []
            for name, out in outs.items():
                if name == baseline_name or not isinstance(out, torch.Tensor):
                    continue
                val = (
                    relative_l1_scalar(base, out) if is_scalar_like(base) else metric_fn(base, out)
                )
                distances.append(float(val.detach().item()))
            if distances:
                scored.append((str(site.layer_label), sum(distances) / len(distances)))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:top_k]

    def diff_pair(self, a: Any, b: Any) -> Any:
        """Return out differences between two members or at one site.

        Parameters
        ----------
        a:
            Site query or first member name.
        b:
            Optional second member name when ``a`` is a site.

        Returns
        -------
        Any
            Site-score rows for member pairs, or a SuperOp diff matrix.
        """

        self._require_relationship("diff", _REQUIRED_RELATIONSHIPS["diff"])
        if isinstance(a, str) and a in self._members and isinstance(b, str) and b in self._members:
            return self._diff_members(a, b)
        view = self.node(a)
        return view.diff_pair(other=b if isinstance(b, str) else None)

    def cluster(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for future bundle clustering.

        Raises
        ------
        NotImplementedError
            Always raised until v1+.
        """

        raise NotImplementedError("Bundle.cluster lands in v1+.")

    def help(self) -> str:
        """Return a per-member readiness summary.

        Returns
        -------
        str
            Human-readable readiness report.
        """

        lines = [f"Bundle ({len(self)} members):"]
        for name, member in self._members.items():
            parts = []
            if name == self._baseline_name:
                parts.append("baseline")
            pristine = (
                getattr(member, "_spec_revision", None) == 0
                and getattr(member, "state", None) is TraceState.PRISTINE
            )
            if pristine:
                parts.append("pristine")
            parts.append(f"intervention_ready={getattr(member, 'intervention_ready', False)}")
            parts.append(f"state={getattr(getattr(member, 'state', None), 'name', None)}")
            if getattr(member, "_spec_revision", 0) != 0:
                parts.append(f"spec_revision={getattr(member, '_spec_revision', 0)}")
            lines.append(f"  - {name}: {', '.join(parts)}")
        return "\n".join(lines)

    def relationship(self, a: str, b: str) -> Relationship:
        """Return the derived relationship for a pair of members.

        Parameters
        ----------
        a:
            First member name.
        b:
            Second member name.

        Returns
        -------
        Relationship
            Derived relationship.
        """

        return self._relationship_between(self._members[a], self._members[b])

    def _diff_members(self, left_name: str, right_name: str) -> list[tuple[str, float]]:
        """Return per-site out differences between two members.

        Parameters
        ----------
        left_name:
            Reference member name.
        right_name:
            Compared member name.

        Returns
        -------
        list[tuple[str, float]]
            ``(site_label, relative_l1)`` rows for common labels.
        """

        left_log = self._members[left_name]
        right_log = self._members[right_name]
        rows: list[tuple[str, float]] = []
        for left_site in getattr(left_log, "layer_list", []):
            label = str(left_site.layer_label)
            try:
                right_site = resolve_sites(right_log, label, max_fanout=1).first()
            except Exception:  # noqa: BLE001 - missing sites are not common sites
                continue
            left_out = getattr(left_site, "out", None)
            right_out = getattr(right_site, "out", None)
            if not isinstance(left_out, torch.Tensor) or not isinstance(
                right_out,
                torch.Tensor,
            ):
                continue
            value = relative_l1_scalar(left_out, right_out)
            rows.append((label, float(value.detach().item())))
        return rows

    @property
    def relationships(self) -> dict[tuple[str, str], Relationship]:
        """Return pairwise relationships for all member pairs.

        Returns
        -------
        dict[tuple[str, str], Relationship]
            Pairwise matrix keyed by member-name tuples.
        """

        matrix: dict[tuple[str, str], Relationship] = {}
        for left in self._members:
            for right in self._members:
                matrix[(left, right)] = self.relationship(left, right)
        return matrix

    @classmethod
    def _parse_members(
        cls,
        members: Mapping[str, "Trace"] | Sequence["Trace"] | Sequence[tuple[str, "Trace"]],
        *,
        names: Sequence[str] | None,
    ) -> list[tuple[str, "Trace"]]:
        """Normalize supported construction shapes.

        Returns
        -------
        list[tuple[str, Trace]]
            Ordered member pairs.
        """

        if isinstance(members, Mapping):
            if names is not None:
                raise ValueError("names= is not accepted when Bundle members are a mapping.")
            pairs = [(str(name), log) for name, log in members.items()]
        else:
            values = list(members)
            if names is not None:
                if len(names) != len(values):
                    raise ValueError("names length must match member count.")
                named_logs = cast("Sequence[Trace]", values)
                pairs = [(str(name), log) for name, log in zip(names, named_logs)]
            elif values and all(cls._is_name_log_tuple(value) for value in values):
                tuple_values = cast("Sequence[tuple[str, Trace]]", values)
                pairs = [(str(member_name), log) for member_name, log in tuple_values]
            else:
                log_values = cast("Sequence[Trace]", values)
                pairs = cls._dedupe_default_names(
                    [
                        (cls._derive_name(log, name=None, index=index), log)
                        for index, log in enumerate(log_values)
                    ]
                )
        if not pairs:
            raise ValueError("Bundle requires at least one Trace.")
        duplicate_names = sorted(
            {name for name, _ in pairs if [n for n, _ in pairs].count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(f"Bundle member names must be unique; duplicates: {duplicate_names}")
        return pairs

    @staticmethod
    def _is_list_like(value: Any) -> bool:
        """Return whether ``value`` should be treated as a list input.

        Parameters
        ----------
        value:
            Candidate input value.

        Returns
        -------
        bool
            Whether the value is a non-string sequence.
        """

        return isinstance(value, Sequence) and not isinstance(value, str)

    @classmethod
    def _coerce_trace_list(
        cls, value: "Trace | Sequence[Trace]", *, arg_name: str
    ) -> list["Trace"]:
        """Normalize a Trace-or-list input to a list.

        Parameters
        ----------
        value:
            Trace or sequence of Traces.
        arg_name:
            Argument name for error messages.

        Returns
        -------
        list[Trace]
            Normalized Trace list.
        """

        values = list(value) if cls._is_list_like(value) else [cast("Trace", value)]
        for item in values:
            if isinstance(item, str):
                raise TypeError(f"{arg_name} must contain Trace objects, not strings.")
        return cast(list["Trace"], values)

    @classmethod
    def _coerce_optional_name_list(
        cls,
        names: str | Sequence[str] | None,
        *,
        count: int,
    ) -> list[str | None]:
        """Normalize optional Bundle names to match a log count.

        Parameters
        ----------
        names:
            Optional name or names.
        count:
            Number of logs being added.

        Returns
        -------
        list[str | None]
            Per-log names.
        """

        if names is None:
            return [None] * count
        if isinstance(names, str):
            if count != 1:
                raise ValueError("A single Bundle name can only be used with one Trace.")
            return [names]
        name_list: list[str | None] = [str(name) for name in names]
        if len(name_list) != count:
            raise ValueError("names length must match added log count.")
        return name_list

    def _coerce_member_name_list(
        self,
        value: str | "Trace" | Sequence[str | "Trace"],
    ) -> list[str]:
        """Normalize Bundle member references to member names.

        Parameters
        ----------
        value:
            Member name, Trace object, or sequence of either.

        Returns
        -------
        list[str]
            Resolved member names.
        """

        values = list(value) if self._is_list_like(value) else [value]
        # _is_list_like narrows the runtime type (sequence -> its elements, scalar -> [value]),
        # but mypy can't follow that helper, so assert the element type the narrowing guarantees.
        return [self._coerce_member_name(cast("str | Trace", item)) for item in values]

    def _coerce_member_name(self, value: str | "Trace") -> str:
        """Resolve one Bundle member reference to a member name.

        Parameters
        ----------
        value:
            Member name or Trace object.

        Returns
        -------
        str
            Resolved member name.
        """

        if isinstance(value, str):
            return value
        for name, member in self._members.items():
            if member is value:
                return name
        raise KeyError("Trace is not a member of this Bundle.")

    @staticmethod
    def _dedupe_default_names(pairs: list[tuple[str, "Trace"]]) -> list[tuple[str, "Trace"]]:
        """Disambiguate automatically derived member names.

        Parameters
        ----------
        pairs:
            Derived name/log pairs from a sequence without explicit names.

        Returns
        -------
        list[tuple[str, Trace]]
            Pairs with ``_2``, ``_3`` suffixes for repeated names.
        """

        seen: dict[str, int] = {}
        deduped: list[tuple[str, Trace]] = []
        for member_name, log in pairs:
            count = seen.get(member_name, 0) + 1
            seen[member_name] = count
            if count == 1:
                deduped.append((member_name, log))
            else:
                deduped.append((f"{member_name}_{count}", log))
        return deduped

    @staticmethod
    def _is_name_log_tuple(value: Any) -> bool:
        """Return whether a value looks like a ``(name, log)`` pair.

        Returns
        -------
        bool
            Whether the value is a two-item name/log tuple.
        """

        return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)

    @staticmethod
    def _derive_name(log: "Trace", *, name: str | None, index: int) -> str:
        """Derive a member name from an explicit value or log metadata.

        Returns
        -------
        str
            Member name.
        """

        if name is not None:
            return str(name)
        log_name = getattr(log, "trace_label", None)
        if log_name:
            return str(log_name)
        return f"member_{index}"

    def _resolve_baseline_name(self, baseline: str | "Trace" | None) -> str | None:
        """Resolve a baseline constructor argument to a member name.

        Returns
        -------
        str | None
            Baseline member name.
        """

        if baseline is None:
            return self._auto_detect_baseline()
        if isinstance(baseline, str):
            if baseline not in self._members:
                raise KeyError(f"Unknown baseline member {baseline!r}.")
            return baseline
        for name, member in self._members.items():
            if member is baseline:
                return name
        raise KeyError("Baseline Trace is not a member of this Bundle.")

    def _auto_detect_baseline(self) -> str | None:
        """Auto-detect a pristine baseline when exactly one candidate exists.

        Returns
        -------
        str | None
            Baseline name or ``None`` when ambiguous/not needed yet.
        """

        candidates = [
            name
            for name, member in self._members.items()
            if getattr(member, "_spec_revision", None) == 0
            and getattr(member, "state", None) is TraceState.PRISTINE
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _baseline_or_raise(self, baseline: str | "Trace" | None) -> str:
        """Return a baseline name or raise for ambiguity.

        Returns
        -------
        str
            Baseline member name.
        """

        if baseline is not None:
            resolved = self._resolve_baseline_name(baseline)
            if resolved is not None:
                return resolved
        if self._baseline_name is not None:
            return self._baseline_name
        detected = self._auto_detect_baseline()
        if detected is not None:
            self._baseline_name = detected
            return detected
        raise BaselineUndeterminedError(
            "Bundle operation requires a baseline, but no unique pristine baseline exists."
        )

    def _ensure_supergraph(self) -> Supergraph:
        """Build and cache the supergraph lazily.

        Returns
        -------
        Supergraph
            Cached supergraph.
        """

        if self._supergraph is None:
            self._supergraph = build_supergraph(list(self._members.values()), list(self._members))
        return self._supergraph

    def _shared(self, key_fn: Callable[["Trace"], Sequence[Any]]) -> list[str]:
        """Return keys common to every member, ordered by the first member.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[str]
            Keys present in all members.
        """

        member_keys = self._member_key_lists(key_fn)
        key_sets = [set(keys) for keys in member_keys]
        shared = set.intersection(*key_sets)
        return [key for key in member_keys[0] if key in shared]

    def _divergent(self, key_fn: Callable[["Trace"], Sequence[Any]]) -> list[str]:
        """Return keys present in some members but not every member.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[str]
            Keys present in at least one member and absent from at least one member.
        """

        member_keys = self._member_key_lists(key_fn)
        key_sets = [set(keys) for keys in member_keys]
        shared = set.intersection(*key_sets)
        union = set.union(*key_sets)
        divergent = union - shared
        ordered: list[str] = []
        seen: set[str] = set()
        for keys in member_keys:
            for key in keys:
                if key in divergent and key not in seen:
                    ordered.append(key)
                    seen.add(key)
        return ordered

    def _member_key_lists(self, key_fn: Callable[["Trace"], Sequence[Any]]) -> list[list[str]]:
        """Return normalized per-member key lists with duplicates removed.

        Parameters
        ----------
        key_fn:
            Function returning candidate keys for one member trace.

        Returns
        -------
        list[list[str]]
            String keys for each member in member order.
        """

        member_keys: list[list[str]] = []
        for member in self._members.values():
            keys: list[str] = []
            seen: set[str] = set()
            for raw_key in key_fn(member):
                if raw_key is None:
                    continue
                key = str(raw_key)
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
            member_keys.append(keys)
        return member_keys

    def _enforce_capacity(self) -> None:
        """Evict oldest non-baseline members when above capacity.

        Returns
        -------
        None
            The bundle is mutated in place.
        """

        if self._capacity is None:
            return
        while len(self._members) > self._capacity:
            evictable = next(
                (name for name in self._members if name != self._baseline_name),
                None,
            )
            if evictable is None:
                break
            self._members.pop(evictable)
            self._supergraph = None

    def _require_relationship(self, operation: str, required: Relationship) -> None:
        """Raise if any member pair lacks the required relationship.

        Parameters
        ----------
        operation:
            Operation name for diagnostics.
        required:
            Minimum relationship.

        Returns
        -------
        None
            Returns only when compatible.
        """

        incompatible: list[tuple[str, str, Relationship]] = []
        names = list(self._members)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                relationship = self.relationship(left_name, right_name)
                if not self._relationship_satisfies(relationship, required):
                    incompatible.append((left_name, right_name, relationship))
        if incompatible:
            pairs = ", ".join(
                f"{left}/{right}={relationship.value}" for left, right, relationship in incompatible
            )
            raise BundleRelationshipError(
                f"Bundle operation {operation!r} requires {required.value}; "
                f"incompatible member pairs: {pairs}"
            )

    @staticmethod
    def _relationship_satisfies(actual: Relationship, required: Relationship) -> bool:
        """Return whether an actual relationship satisfies an operation gate.

        Returns
        -------
        bool
            Whether the gate should pass.
        """

        if required is Relationship.SAME_PARAM_SHAPES:
            return actual in {
                Relationship.SAME_OBJECT,
                Relationship.SAME_MODEL_OBJECT_AT_CAPTURE,
                Relationship.SHARED_GRAPH_SAME_INPUT,
                Relationship.SHARED_GRAPH_DIFFERENT_INPUT,
                Relationship.SAME_PARAM_SHAPES,
            }
        if required is Relationship.SHARED_GRAPH_SAME_INPUT:
            return actual in {
                Relationship.SAME_OBJECT,
                Relationship.SAME_MODEL_OBJECT_AT_CAPTURE,
                Relationship.SHARED_GRAPH_SAME_INPUT,
            }
        return _RELATIONSHIP_RANK[actual] >= _RELATIONSHIP_RANK[required]

    @classmethod
    def _relationship_between(cls, left: "Trace", right: "Trace") -> Relationship:
        """Derive relationship evidence for two model logs.

        Returns
        -------
        Relationship
            Highest-confidence relationship.
        """

        if left is right:
            return Relationship.SAME_OBJECT

        left_class = getattr(left, "model_class_qualname", None)
        right_class = getattr(right, "model_class_qualname", None)
        left_weight = cls._weight_fingerprint(left)
        right_weight = cls._weight_fingerprint(right)
        left_id = getattr(left, "model_object_id", None)
        right_id = getattr(right, "model_object_id", None)
        if (
            left_id is not None
            and left_id == right_id
            and left_class is not None
            and left_class == right_class
            and left_weight is not None
            and left_weight == right_weight
        ):
            return Relationship.SAME_OBJECT

        left_model = cls._weak_model(left)
        right_model = cls._weak_model(right)
        if (
            left_model is not None
            and left_model is right_model
            and left_class is not None
            and left_class == right_class
        ):
            return Relationship.SAME_MODEL_OBJECT_AT_CAPTURE

        left_graph = getattr(left, "graph_shape_hash", None)
        right_graph = getattr(right, "graph_shape_hash", None)
        left_input = getattr(left, "input_signature_hash", None)
        right_input = getattr(right, "input_signature_hash", None)
        if left_graph is not None and left_graph == right_graph:
            if left_input is not None and left_input == right_input:
                return Relationship.SHARED_GRAPH_SAME_INPUT
            return Relationship.SHARED_GRAPH_DIFFERENT_INPUT

        if left_weight is not None and left_weight == right_weight:
            return Relationship.SAME_PARAM_SHAPES
        if left_class is not None and left_class == right_class:
            return Relationship.SHARED_ARCHITECTURE
        if left_class is not None and right_class is not None and left_class != right_class:
            return Relationship.DIFF_MODEL
        return Relationship.UNKNOWN

    @staticmethod
    def _weight_fingerprint(log: "Trace") -> str | None:
        """Return the strongest available weight fingerprint.

        Returns
        -------
        str | None
            Fingerprint value.
        """

        return getattr(log, "param_hash_full", None) or getattr(
            log,
            "param_hash_quick",
            None,
        )

    @staticmethod
    def _weak_model(log: "Trace") -> Any | None:
        """Resolve a captured weak model reference.

        Returns
        -------
        Any | None
            Live model object, if available.
        """

        ref = getattr(log, "_source_model_ref", None)
        if ref is None:
            return None
        try:
            return ref()
        except TypeError:
            return None


def _metric_label(metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> str:
    """Return a stable label for a metric specifier.

    Parameters
    ----------
    metric:
        Metric name or callable.

    Returns
    -------
    str
        Human-readable metric label.
    """

    return metric if isinstance(metric, str) else getattr(metric, "__name__", "callable")


def _tensor_field(layer: Any, field: Literal["out", "grad"]) -> torch.Tensor | None:
    """Return a tensor field from a layer-like object.

    Parameters
    ----------
    layer:
        Layer or Op-like object.
    field:
        Tensor field to read.

    Returns
    -------
    torch.Tensor | None
        Tensor value when available.
    """

    value = getattr(layer, field, None)
    return value if isinstance(value, torch.Tensor) else None


def _distance_value(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Return a Python float distance between two tensors.

    Parameters
    ----------
    reference:
        Reference tensor.
    candidate:
        Compared tensor.
    metric:
        Metric name or callable.

    Returns
    -------
    float
        Scalar distance.
    """

    metric_fn = resolve_metric(metric)
    value = (
        relative_l1_scalar(reference, candidate)
        if is_scalar_like(reference)
        else metric_fn(reference, candidate)
    )
    return float(value.detach().item())


def _resolve_member_name(bundle: Bundle, member: str | "Trace" | None) -> str:
    """Resolve a member name or Trace reference within a bundle.

    Parameters
    ----------
    bundle:
        Bundle being queried.
    member:
        Member name, Trace reference, or ``None``.

    Returns
    -------
    str
        Resolved member name.
    """

    if member is None:
        return next(iter(bundle.names))
    if isinstance(member, str):
        if member not in bundle:
            raise KeyError(f"Unknown bundle member {member!r}.")
        return member
    for name, log in bundle.members.items():
        if log is member:
            return name
    raise KeyError("Trace is not a member of this Bundle.")


def _bundle_delta_map(
    self: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    *,
    baseline: str | "Trace" | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return per-node tensor deltas from a baseline trace.

    Parameters
    ----------
    metric:
        Metric name from ``torchlens.intervention._metrics`` or a callable.
    baseline:
        Baseline member name or log. Defaults to the configured baseline, then
        the first member.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of supergraph node name to member-name distance values. Members
        without a tensor at a node are omitted for that node.
    """

    baseline_name = (
        self._baseline_or_raise(baseline)
        if baseline is not None or self.baseline_name is not None
        else next(iter(self.names))
    )
    result: dict[str, dict[str, float]] = {}
    supergraph = self.supergraph
    for graph_node_label in supergraph.topological_order:
        node = supergraph.nodes[graph_node_label]
        reference_layer = node.layer_refs.get(baseline_name)
        if reference_layer is None:
            continue
        reference = _tensor_field(reference_layer, on)
        if reference is None:
            continue
        values: dict[str, float] = {}
        for member_name in self.names:
            layer = node.layer_refs.get(member_name)
            candidate = _tensor_field(layer, on) if layer is not None else None
            if candidate is None:
                continue
            values[member_name] = (
                0.0
                if member_name == baseline_name
                else _distance_value(
                    reference,
                    candidate,
                    metric,
                )
            )
        if values:
            result[graph_node_label] = values
    return result


def _bundle_norm_delta(
    self: Bundle,
    *,
    baseline: str | "Trace" | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return relative L2 deltas for every comparable bundle node.

    Parameters
    ----------
    baseline:
        Baseline member name or log.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-node relative L2 distances keyed by member name.
    """

    return _bundle_delta_map(self, "relative_l2", baseline=baseline, on=on)


def _output_layer_pairs(
    target_log: "Trace",
    candidate_log: "Trace",
) -> list[tuple[Any, Any]]:
    """Return paired output layers by output index.

    Parameters
    ----------
    target_log:
        Reference model log.
    candidate_log:
        Compared model log.

    Returns
    -------
    list[tuple[Any, Any]]
        Paired output layer-like objects.
    """

    target_labels = list(getattr(target_log, "output_layers", []) or [])
    candidate_labels = list(getattr(candidate_log, "output_layers", []) or [])
    if target_labels and candidate_labels:
        pairs: list[tuple[Any, Any]] = []
        for target_label, candidate_label in zip(target_labels, candidate_labels):
            try:
                pairs.append((target_log[target_label], candidate_log[candidate_label]))
            except (KeyError, IndexError):
                continue
        return pairs
    target_layers = list(getattr(target_log, "layer_list", []))
    candidate_layers = list(getattr(candidate_log, "layer_list", []))
    return [(target_layers[-1], candidate_layers[-1])] if target_layers and candidate_layers else []


def _bundle_output_delta(
    self: Bundle,
    target: str | "Trace",
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return output divergence for every member versus a target trace.

    Parameters
    ----------
    target:
        Target member name or ``Trace`` reference.
    metric:
        Metric name or callable.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Member-keyed output distance mapping.
    """

    target_name = _resolve_member_name(self, target)
    target_log = self[target_name]
    result: dict[str, dict[str, float]] = {}
    for member_name, member_log in self.members.items():
        output_values: dict[str, float] = {}
        for output_index, (target_layer, member_layer) in enumerate(
            _output_layer_pairs(target_log, member_log)
        ):
            reference = _tensor_field(target_layer, on)
            candidate = _tensor_field(member_layer, on)
            if reference is None or candidate is None:
                continue
            label = str(getattr(target_layer, "layer_label", f"output_{output_index}"))
            output_values[label] = (
                0.0 if member_name == target_name else _distance_value(reference, candidate, metric)
            )
        result[member_name] = output_values
    return result


def _bundle_motif_occurrences(self: Bundle) -> dict[str, list[tuple[str, str]]]:
    """Return repeated operation-equivalence motifs across bundle traces.

    Parameters
    ----------
    self:
        Bundle being inspected.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Operation-equivalence key to ``(member_name, layer_label)`` occurrences.
    """

    motifs: dict[str, list[tuple[str, str]]] = {}
    for member_name, member in self.members.items():
        for layer in getattr(member, "layer_list", []):
            key = getattr(layer, "equivalence_class", None)
            if not key:
                continue
            motifs.setdefault(str(key), []).append((member_name, str(layer.layer_label)))
    return {key: rows for key, rows in motifs.items() if len(rows) > 1}


def _bundle_compare(
    self: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    *,
    baseline: str | "Trace" | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, Any]:
    """Return a unified bundle comparison payload.

    Parameters
    ----------
    metric:
        Metric name or callable.
    baseline:
        Baseline member name or log. Defaults like :meth:`delta_map`.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, Any]
        Uniform payload with metric metadata, node deltas, output deltas, and
        repeated motif occurrences.
    """

    baseline_name = (
        self._baseline_or_raise(baseline)
        if baseline is not None or self.baseline_name is not None
        else next(iter(self.names))
    )
    return {
        "baseline": baseline_name,
        "metric": _metric_label(metric),
        "on": on,
        "nodes": _bundle_delta_map(self, metric, baseline=baseline_name, on=on),
        "outputs": _bundle_output_delta(self, baseline_name, metric=metric, on=on),
        "motifs": _bundle_motif_occurrences(self),
    }


def _alignment_score(left: Any, right: Any, left_index: int, right_index: int) -> float:
    """Return a conservative cross-architecture alignment score.

    Parameters
    ----------
    left:
        Left layer-like object.
    right:
        Right layer-like object.
    left_index:
        Topological index of ``left``.
    right_index:
        Topological index of ``right``.

    Returns
    -------
    float
        Heuristic score in ``[0, 1]``.
    """

    score = 0.0
    if getattr(left, "module", None) == getattr(right, "module", None):
        score += 0.35
    if getattr(left, "func_name", None) == getattr(right, "func_name", None):
        score += 0.35
    if getattr(left, "shape", None) == getattr(right, "shape", None):
        score += 0.2
    distance = abs(left_index - right_index)
    score += max(0.0, 0.1 - (distance * 0.01))
    return min(score, 1.0)


def _bundle_aligned_pairs(
    self: Bundle,
    left: str | "Trace" | None = None,
    right: str | "Trace" | None = None,
    *,
    min_score: float = 0.45,
) -> list[tuple[Any, Any]]:
    """Return best-match layer pairs across two bundle members.

    Alignment rules are intentionally conservative:

    1. Prefer exact module path and operation name matches.
    2. Use tensor shape and topological proximity to break ties.
    3. Pair each right-side layer at most once.

    Parameters
    ----------
    left:
        Left member name or log. Defaults to the first bundle member.
    right:
        Right member name or log. Defaults to the second bundle member.
    min_score:
        Minimum heuristic score required to emit a pair.

    Returns
    -------
    list[tuple[Any, Any]]
        Paired layer-like objects, ordered by the left trace.
    """

    names = self.names
    if len(names) < 2 and (left is None or right is None):
        raise ValueError("aligned_pairs requires at least two bundle members.")
    left_name = _resolve_member_name(self, left if left is not None else names[0])
    right_name = _resolve_member_name(self, right if right is not None else names[1])
    left_layers = list(getattr(self[left_name], "layer_list", []))
    right_layers = list(getattr(self[right_name], "layer_list", []))
    available_right = set(range(len(right_layers)))
    pairs: list[tuple[Any, Any]] = []
    for left_index, left_layer in enumerate(left_layers):
        best_index: int | None = None
        best_score = 0.0
        for right_index in available_right:
            score = _alignment_score(left_layer, right_layers[right_index], left_index, right_index)
            if score > best_score:
                best_index = right_index
                best_score = score
        if best_index is not None and best_score >= min_score:
            available_right.remove(best_index)
            pairs.append((left_layer, right_layers[best_index]))
    return pairs


def _bundle_show_diff(
    self: Bundle,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    layout: Literal["paired"] = "paired",
    **kwargs: Any,
) -> str:
    """Render a two-column bundle diff for a clean/intervention pair.

    Examples
    --------
    >>> trace = tl.trace(model, x, intervention_ready=True)
    >>> ablated = trace.fork("ablated")
    >>> ablated.do(tl.module("layer1.0.relu"), tl.zero_ablate())
    >>> bundle = tl.bundle({"clean": trace, "ablated": ablated}, baseline="clean")
    >>> bundle.show_diff(vis_outpath="bundle_diff_clean_vs_zero_relu")

    Parameters
    ----------
    metric:
        Metric forwarded to ``tl.viz.bundle_diff``.
    layout:
        Layout strategy forwarded to ``tl.viz.bundle_diff``.
    **kwargs:
        Additional renderer options forwarded unchanged.

    Returns
    -------
    str
        Graphviz DOT source for the rendered diff.
    """

    from ..visualization.bundle_diff import bundle_diff

    return bundle_diff(self, metric=metric, layout=layout, **kwargs)


__all__ = ["AmbiguousLabelError", "Bundle"]
