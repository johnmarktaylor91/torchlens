"""Resolve TorchLens intervention selectors against model logs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import importlib
import operator
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
import warnings

import torch

from .errors import (
    MultiMatchWarning,
    ReplayPreconditionError,
    SiteAmbiguityError,
    SiteResolutionError,
)
from .selectors import (
    BaseSelector,
    CompositeSelector,
    NotSelector,
    _classify_selector_direction,
    in_module,
)
from .types import FrozenTargetSpec, FunctionRegistryKey, TargetSpec
from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..ir.container_registry import Role

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.grad_fn import GradFn
    from torchlens.data_classes.layer import Layer
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

SelectorInput = BaseSelector | TargetSpec | FrozenTargetSpec | str
if TYPE_CHECKING:
    Site: TypeAlias = Op | Layer | GradFn
else:
    Site: TypeAlias = Any
DIRECTION_AGNOSTIC_KINDS = frozenset(
    {"label", "in_module", "module", "contains", "predicate", "func_transform"}
)


def function_registry_key_from_callable(func: Callable[..., Any]) -> FunctionRegistryKey:
    """Infer a portable registry key from a captured callable.

    Parameters
    ----------
    func:
        Callable captured during tracing.

    Returns
    -------
    FunctionRegistryKey
        Registry key using known namespaces where possible and import refs for
        custom callables.
    """

    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
    name = getattr(func, "__name__", qualname.rsplit(".", maxsplit=1)[-1])
    dispatch_kind: Literal["function", "dunder"] = (
        "dunder" if str(name).startswith("__") and str(name).endswith("__") else "function"
    )

    if module == "torch":
        return FunctionRegistryKey("torch", str(name), dispatch_kind)
    if module == "torch.nn.functional":
        return FunctionRegistryKey("torch.nn.functional", str(name), dispatch_kind)
    if module == "operator":
        return FunctionRegistryKey("operator", str(name), dispatch_kind)
    if module in {"torch._tensor", "torch.Tensor"} or (
        hasattr(torch.Tensor, str(name)) and "Tensor" in str(qualname)
    ):
        return FunctionRegistryKey("torch.Tensor", str(name), "method")

    import_path = f"{module}:{qualname}" if module else None
    return FunctionRegistryKey("custom", str(qualname), dispatch_kind, import_path=import_path)


def resolve_function_registry_key(key: FunctionRegistryKey) -> Callable[..., Any]:
    """Resolve a saved function registry key to a runtime callable.

    Parameters
    ----------
    key:
        Saved function registry key.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.

    Raises
    ------
    ReplayPreconditionError
        If the namespace or qualified name cannot be resolved.
    """

    try:
        if key.namespace == "torch":
            return cast(Callable[..., Any], getattr(torch, key.qualname))
        if key.namespace == "torch.Tensor":
            return cast(Callable[..., Any], getattr(torch.Tensor, key.qualname))
        if key.namespace == "torch.nn.functional":
            return cast(Callable[..., Any], getattr(torch.nn.functional, key.qualname))
        if key.namespace == "operator":
            return cast(Callable[..., Any], getattr(operator, key.qualname))
        if key.namespace == "custom":
            if not key.import_path:
                raise AttributeError("custom key is missing import_path")
            module_name, _, qualname = key.import_path.partition(":")
            module = importlib.import_module(module_name)
            obj: Any = module
            for part in qualname.split("."):
                obj = getattr(obj, part)
            if not callable(obj):
                raise TypeError(f"{key.import_path!r} resolved to non-callable {obj!r}")
            return cast(Callable[..., Any], obj)
    except (AttributeError, ImportError, TypeError) as exc:
        raise ReplayPreconditionError(f"Could not resolve function registry key {key!r}") from exc

    raise ReplayPreconditionError(f"Unknown function registry namespace {key.namespace!r}")


@dataclass(frozen=True)
class SiteTable:
    """Result of resolving a selector; ordered, indexable, dataframe-convertible."""

    _sites: tuple[Site, ...]
    query: Any | None = None

    def __len__(self) -> int:
        """Return the number of resolved sites.

        Returns
        -------
        int
            Number of layer-pass records in the table.
        """

        return len(self._sites)

    def __iter__(self) -> Iterator[Site]:
        """Iterate through resolved sites in execution order.

        Returns
        -------
        Iterator[Op]
            Iterator over layer-pass records.
        """

        return iter(self._sites)

    def __getitem__(self, idx: int | slice) -> "Site | SiteTable":
        """Return one site or a sliced site table.

        Parameters
        ----------
        idx:
            Integer index or slice.

        Returns
        -------
        Op | SiteTable
            Single layer pass for integer indexes, table for slices.
        """

        if isinstance(idx, slice):
            return SiteTable(self._sites[idx], query=self.query)
        return self._sites[idx]

    def __repr__(self) -> str:
        """Return a compact table representation.

        Returns
        -------
        str
            Summary including count and first labels.
        """

        count = len(self)
        if count == 0:
            return "SiteTable(0 sites)"
        labels = self.labels()
        if count == 1:
            return f"SiteTable(1 site: {labels[0]})"
        if count <= 3:
            return f"SiteTable({count} sites: {', '.join(labels)})"
        prefix = ", ".join(labels[:3])
        return f"SiteTable({count} sites: {prefix}, ... {labels[-1]})"

    def where(self, predicate: Callable[[Site], bool]) -> "SiteTable":
        """Filter the table with a predicate.

        Parameters
        ----------
        predicate:
            Callable receiving each layer-pass record.

        Returns
        -------
        SiteTable
            Filtered table in original execution order.
        """

        return SiteTable(tuple(site for site in self._sites if predicate(site)), query=self.query)

    def first(self) -> Site:
        """Return the first resolved site.

        Returns
        -------
        Op
            First layer-pass record.

        Raises
        ------
        SiteResolutionError
            If the table is empty.
        """

        if not self._sites:
            raise SiteResolutionError("SiteTable is empty; no first site is available.")
        return self._sites[0]

    def labels(self) -> tuple[str, ...]:
        """Return execution-order labels for resolved sites.

        Returns
        -------
        tuple[str, ...]
            Stable layer labels.
        """

        return tuple(
            str(
                getattr(
                    site,
                    "layer_label",
                    getattr(site, "label", "<unknown>"),
                )
            )
            for site in self._sites
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a pandas table describing resolved sites.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per resolved site.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [
            {
                "label": getattr(site, "label", None),
                "layer_label": getattr(site, "layer_label", None),
                "call_index": getattr(site, "call_index", None),
                "step_index": getattr(site, "step_index", None),
                "raw_index": getattr(site, "raw_index", None),
                "func_name": getattr(site, "func_name", getattr(site, "name", None)),
                "module": getattr(site, "module", None),
                "modules": getattr(site, "modules", ()),
                "output_of_module_calls": getattr(site, "output_of_module_calls", ()),
            }
            for site in self._sites
        ]
        return pd.DataFrame(rows)


def resolve_sites(
    log: "Trace",
    query: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> SiteTable:
    """Resolve a selector or accepted shorthand against a model log.

    Parameters
    ----------
    log:
        Model log whose captured layer-pass records should be searched.
    query:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed. ``None`` is rejected in this MVP.

    Returns
    -------
    SiteTable
        Ordered table of resolved sites.

    Raises
    ------
    SiteAmbiguityError
        If more sites resolve than ``max_fanout`` allows.
    SiteResolutionError
        If no site resolves or the query shape is unsupported.
    """

    if max_fanout is None:
        raise SiteResolutionError("max_fanout=None is not supported; pass an explicit integer.")
    if max_fanout < 1:
        raise SiteAmbiguityError("max_fanout must be at least 1.")
    if strict and isinstance(query, str):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector."
        )

    selector = _normalize_query(query)
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if not matched:
        raise SiteResolutionError(
            f"selector {query!r} matched 0 sites. Use log.find_sites(...) to discover labels."
        )
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    if len(matched) > 1:
        warnings.warn(
            f"selector {query!r} matched {len(matched)} sites and will fan out.",
            MultiMatchWarning,
            stacklevel=2,
        )
    return SiteTable(matched, query=query)


def find_sites(
    log: "Trace",
    query: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> SiteTable:
    """Find matching sites and return a table.

    Parameters
    ----------
    log:
        Model log whose captured layer-pass records should be searched.
    query:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed.

    Returns
    -------
    SiteTable
        Ordered table of resolved sites.
    """

    if max_fanout is None:
        raise SiteResolutionError("max_fanout=None is not supported; pass an explicit integer.")
    if max_fanout < 1:
        raise SiteAmbiguityError("max_fanout must be at least 1.")
    if strict and isinstance(query, str):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector."
        )

    selector = _normalize_query(query)
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    return SiteTable(matched, query=query)


def _iter_layer_ops(log: "Trace") -> Sequence["Op"]:
    """Return final layer ops from a completed model log.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    Sequence[Op]
        Execution-order layer-pass records.

    Raises
    ------
    SiteResolutionError
        If the model log has not completed postprocessing.
    """

    if not getattr(log, "_tracing_finished", False):
        raise SiteResolutionError("Sites can only be resolved after the forward pass is complete.")
    return log.layer_list


def _iter_sites(log: "Trace", direction: Literal["forward", "backward"]) -> Sequence[Site]:
    """Return candidate sites for the requested graph direction.

    Parameters
    ----------
    log:
        Model log to inspect.
    direction:
        Site universe to return.

    Returns
    -------
    Sequence[Site]
        Forward op sites or backward grad_fn_handle sites.
    """

    if direction == "forward":
        return _iter_layer_ops(log)
    if not getattr(log, "has_backward_pass", False):
        raise SiteResolutionError("Backward selectors require log_backward() to run first.")
    return tuple(log.grad_fn_logs.values())


def _resolve_unchecked(
    sites: Sequence[Site],
    query: SelectorInput,
    *,
    strict: bool,
) -> tuple[Site, ...]:
    """Resolve without zero/multi fanout validation.

    Parameters
    ----------
    sites:
        Layer-pass records to search.
    query:
        Selector input.
    strict:
        Whether strict portability rules are active.

    Returns
    -------
    tuple[Op, ...]
        Matching sites in execution order.
    """

    selector = _normalize_query(query)
    kind = selector.selector_kind
    value = selector.selector_value

    if strict and kind == "predicate":
        raise SiteResolutionError(
            "tl.where(...) predicate selectors are non-portable in strict mode."
        )

    if kind == "and" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        left_site_ids = {id(site) for site in _resolve_unchecked(sites, left, strict=strict)}
        right_site_ids = {id(site) for site in _resolve_unchecked(sites, right, strict=strict)}
        return tuple(
            site for site in sites if id(site) in left_site_ids and id(site) in right_site_ids
        )
    if kind == "or" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        left_site_ids = {id(site) for site in _resolve_unchecked(sites, left, strict=strict)}
        right_site_ids = {id(site) for site in _resolve_unchecked(sites, right, strict=strict)}
        return tuple(
            site for site in sites if id(site) in left_site_ids or id(site) in right_site_ids
        )
    if kind == "not" and isinstance(selector, NotSelector):
        excluded_site_ids = {
            id(site) for site in _resolve_unchecked(sites, selector.selector, strict=strict)
        }
        return tuple(site for site in sites if id(site) not in excluded_site_ids)

    if kind == "label":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "func":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "func_transform":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "module":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "output":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "output_at":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "input_at":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "contains":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "in_module":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "predicate":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind in {"grad_fn", "grad_fn_handle", "intervening", "label"}:
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))

    raise SiteResolutionError(f"Unsupported selector kind {kind!r}.")


def _selector_resolution_direction(query: SelectorInput) -> Literal["forward", "backward"]:
    """Pick the resolver search universe for a selector query.

    Parameters
    ----------
    query:
        Selector query to inspect recursively.

    Returns
    -------
    Literal["forward", "backward"]
        Direction whose site universe should be searched.
    """

    selector = _normalize_query(query)

    def _walk(sel: BaseSelector) -> tuple[bool, bool]:
        """Return whether a selector tree contains backward or forward selectors."""

        if isinstance(sel, CompositeSelector):
            has_back = False
            has_forward = False
            for child in sel.selectors:
                child_selector = _normalize_query(child)
                child_back, child_forward = _walk(child_selector)
                has_back = has_back or child_back
                has_forward = has_forward or child_forward
            return has_back, has_forward
        if isinstance(sel, NotSelector):
            return _walk(_normalize_query(sel.selector))
        direction = _classify_selector_direction(sel)
        return direction == "backward", direction == "forward"

    has_backward, has_forward = _walk(selector)
    if has_backward:
        return "backward"
    if has_forward:
        return "forward"
    return "forward"


def _resolve_site_kind(site: Site, kind: str, value: Any) -> bool:
    """Return whether one site matches a simple selector kind.

    Parameters
    ----------
    site:
        Candidate forward or backward site.
    kind:
        Selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    from torchlens.data_classes.grad_fn import GradFn

    if isinstance(site, GradFn):
        if kind in DIRECTION_AGNOSTIC_KINDS:
            return _resolve_grad_fn_forward_kind(site, kind, value)
        return _resolve_grad_fn_kind(site, kind, value)

    if kind == "label":
        return _label_matches(site, str(value))
    if kind == "func":
        if isinstance(value, dict):
            return site.func_name == value.get("name") and _output_matches(
                site, value.get("output")
            )
        return site.func_name == value
    if kind == "func_transform":
        if not bool(getattr(site, "is_transform", False)):
            return False
        if value is None:
            return True
        transform_kind = getattr(site, "transform_kind", None)
        if transform_kind is None:
            return False
        normalized_site = str(transform_kind).lower().replace("_", "").replace(".", "")
        normalized_value = str(value).lower().replace("_", "").replace(".", "")
        return normalized_site == normalized_value
    if kind == "output":
        return _output_matches(site, value)
    if kind == "output_at":
        return _output_path_matches(tuple(getattr(site, "container_path", ()) or ()), tuple(value))
    if kind == "input_at":
        return _input_path_matches(site, tuple(value))
    if kind == "module":
        return _module_output_matches(site, str(value))
    if kind == "contains":
        return str(value).lower() in str(site.layer_label).lower()
    if kind == "in_module":
        return bool(in_module(site, str(value)))
    if kind == "predicate":
        predicate, _name_hint = _predicate_payload(value)
        return bool(predicate(site))
    return False


def _resolve_grad_fn_forward_kind(site: "GradFn", kind: str, value: Any) -> bool:
    """Return whether a GradFn matches a forward selector through its op aliases.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    kind:
        Direction-agnostic selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches the paired op or a synthetic boundary alias
        sharing the same autograd ``grad_fn`` identity.
    """

    if site.op is not None and _resolve_site_kind(site.op, kind, value):
        return True
    for alias in _grad_fn_boundary_aliases(site):
        if _resolve_site_kind(alias, kind, value):
            return True
    return False


def _grad_fn_boundary_aliases(site: "GradFn") -> tuple[Site, ...]:
    """Return input/output alias ops sharing a GradFn identity with ``site``.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Site, ...]
        Synthetic input/output ops whose ``grad_fn_object_id`` equals the
        GradFn's object id, excluding the already-paired op.
    """

    trace = site.source_trace
    if trace is None:
        return ()
    paired_label = site.op_label
    aliases: list[Site] = []
    for layer in getattr(trace, "layer_list", ()):
        if getattr(layer, "layer_label", None) == paired_label:
            continue
        if getattr(layer, "grad_fn_object_id", None) != site.grad_fn_object_id:
            continue
        if not (getattr(layer, "is_input", False) or getattr(layer, "is_output", False)):
            continue
        aliases.append(layer)
    return tuple(aliases)


def _resolve_grad_fn_kind(site: "GradFn", kind: str, value: Any) -> bool:
    """Return whether one grad_fn_handle site matches a backward-only selector.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    kind:
        Backward selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    if kind == "intervening":
        return not site.has_op
    if kind == "label":
        return site.label == str(value)
    if kind == "grad_fn":
        payload = value if isinstance(value, dict) else {}
        type = payload.get("type")
        label_pattern = payload.get("grad_fn_label_pattern")
        is_custom = payload.get("is_custom")
        if type is not None and not _grad_fn_type_matches(site, str(type)):
            return False
        if label_pattern is not None and str(label_pattern) not in site.label:
            return False
        if is_custom is not None and bool(site.is_custom) is not bool(is_custom):
            return False
        return True
    return False


def _output_matches(site: Site, value: Any) -> bool:
    """Return whether a forward site matches an output index or role.

    Parameters
    ----------
    site:
        Candidate forward site.
    value:
        Output index or semantic role.

    Returns
    -------
    bool
        Whether the site matches the requested output.
    """

    if isinstance(value, int):
        return getattr(site, "multi_output_index", None) == value
    return getattr(site, "multi_output_name", None) == str(value)


def _output_path_matches(saved_path: tuple[Any, ...], requested_path: tuple[Any, ...]) -> bool:
    """Return whether a saved typed path matches a user path.

    Parameters
    ----------
    saved_path:
        Captured typed output path.
    requested_path:
        User path using plain indices/keys/field names.

    Returns
    -------
    bool
        Whether the paths address the same output leaf.
    """

    if len(saved_path) != len(requested_path):
        return False
    return all(
        _output_path_component_matches(saved_component, requested_component)
        for saved_component, requested_component in zip(saved_path, requested_path)
    )


def _output_path_component_matches(saved_component: Any, requested_component: Any) -> bool:
    """Return whether one typed path component matches a user component.

    Parameters
    ----------
    saved_component:
        Captured typed path component.
    requested_component:
        User path component.

    Returns
    -------
    bool
        Whether the components are equivalent.
    """

    if isinstance(saved_component, TupleIndex):
        return saved_component.index == requested_component
    if isinstance(saved_component, (DictKey, HFKey)):
        return saved_component.key == requested_component
    if isinstance(saved_component, (NamedField, DataclassField)):
        return saved_component.name == requested_component
    return saved_component == requested_component


def _input_path_matches(site: Site, requested_path: tuple[Any, ...]) -> bool:
    """Return whether a forward site consumes an input at ``requested_path``.

    Parameters
    ----------
    site:
        Candidate forward site.
    requested_path:
        User path using plain indices, keys, or field names.

    Returns
    -------
    bool
        Whether any input-container leaf occurrence matches the path.
    """

    trace = getattr(site, "source_trace", None)
    site_labels = {
        label
        for label in (
            getattr(site, "layer_label", None),
            getattr(site, "layer_label_raw", None),
            getattr(site, "_layer_label_raw", None),
        )
        if label is not None
    }
    for record in getattr(trace, "_containers", {}).values():
        for snapshot in getattr(record, "snapshots", ()) or ():
            if getattr(snapshot, "role", None) != Role.MODEL_INPUT:
                continue
            for occurrence in getattr(snapshot, "leaf_occurrences", ()) or ():
                if occurrence.producer_op_label not in site_labels:
                    continue
                if _output_path_matches(tuple(occurrence.path), requested_path):
                    return True
    for container in getattr(site, "input_containers", ()) or ():
        for occurrence in getattr(container, "leaf_occurrences", ()) or ():
            if _output_path_matches(tuple(occurrence.path), requested_path):
                return True
    return False


def _grad_fn_type_matches(site: "GradFn", requested: str) -> bool:
    """Return whether a grad_fn_handle type matches user spelling flexibly.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    requested:
        Requested type string.

    Returns
    -------
    bool
        Whether class name or normalized type matches.
    """

    lowered = requested.lower()
    normalized = lowered.removesuffix("backward0").removesuffix("backward")
    candidates = {
        site.class_name.lower(),
        site.type.lower(),
        site.label.lower(),
    }
    return lowered in candidates or normalized in candidates


def _normalize_query(query: SelectorInput) -> BaseSelector:
    """Normalize a query object to a selector.

    Parameters
    ----------
    query:
        Supported selector input.

    Returns
    -------
    BaseSelector
        Normalized selector object.

    Raises
    ------
    SiteResolutionError
        If the query shape is unsupported.
    """

    if isinstance(query, BaseSelector):
        return query
    if isinstance(query, str):
        from .selectors import contains

        return contains(query)
    if isinstance(query, TargetSpec):
        return _selector_from_spec(query.selector_kind, query.selector_value, query.metadata)
    if isinstance(query, FrozenTargetSpec):
        return _selector_from_spec(
            query.selector_kind,
            query.selector_value,
            dict(query.metadata),
        )
    raise SiteResolutionError(f"Unsupported site query {query!r}.")


def _selector_from_spec(kind: str, value: Any, metadata: dict[str, Any]) -> BaseSelector:
    """Build a selector from a target spec payload.

    Parameters
    ----------
    kind:
        Selector kind from a target spec.
    value:
        Selector payload.
    metadata:
        Selector metadata.

    Returns
    -------
    BaseSelector
        Selector matching the target spec.
    """

    from .selectors import (
        contains,
        facet,
        func,
        func_transform,
        grad_fn,
        head,
        intervening,
        label,
        module,
        output,
        where,
    )

    if kind == "label":
        return label(str(value))
    if kind == "func":
        if isinstance(value, dict):
            return func(str(value.get("name")), output=value.get("output"))
        return func(str(value))
    if kind == "func_transform":
        return func_transform(None if value is None else str(value))
    if kind == "module":
        return module(str(value))
    if kind == "output":
        return output(value)
    if kind == "contains":
        return contains(str(value))
    if kind == "in_module":
        from .selectors import in_module as make_in_module

        selector = make_in_module(str(value))
        if isinstance(selector, BaseSelector):
            return selector
    if kind == "facet":
        if isinstance(value, dict):
            name = value.get("name")
            head_index = value.get("head_index")
            if head_index is not None:
                return head(int(head_index), None if name is None else str(name))
            if name is not None:
                return facet(str(name))
        raise SiteResolutionError(f"Unsupported facet selector payload {value!r}.")
    if kind == "predicate" and callable(value):
        return where(value, name_hint=metadata.get("name_hint"))
    if kind == "grad_fn":
        payload = dict(value)
        return grad_fn(
            payload.get("type"),
            label=payload.get("grad_fn_label_pattern"),
            is_custom=payload.get("is_custom"),
        )
    if kind == "intervening":
        return intervening()
    if kind == "label":
        return label(str(value))
    if kind == "not":
        nested = _normalize_query(value)
        return ~nested
    raise SiteResolutionError(f"Unsupported target spec selector kind {kind!r}.")


def _label_matches(site: Any, label: str) -> bool:
    """Return whether a layer-pass record has a requested label.

    Parameters
    ----------
    site:
        Layer-pass record.
    label:
        Label to match.

    Returns
    -------
    bool
        Whether the label matches any exact label field or lookup key.
    """

    candidate_labels = (
        getattr(site, "layer_label", None),
        getattr(site, "label", None),
        getattr(site, "layer_label", None),
        getattr(site, "layer_label_short", None),
        getattr(site, "label_short", None),
        getattr(site, "layer_label_short", None),
        getattr(site, "_layer_label_raw", None),
    )
    return label in candidate_labels or label in getattr(site, "lookup_keys", ())


def _module_output_matches(site: Any, address: str) -> bool:
    """Return whether a layer pass is the output boundary for a module.

    Parameters
    ----------
    site:
        Layer-pass record.
    address:
        Module address or pass-qualified module label.

    Returns
    -------
    bool
        Whether the site exits the requested module.
    """

    module_ops = getattr(site, "output_of_module_calls", ())
    return any(_module_label_matches(module_pass, address) for module_pass in module_ops)


def _module_label_matches(module_pass: str, address: str) -> bool:
    """Return whether a module pass label matches an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified module label.
    address:
        Requested module address or pass label.

    Returns
    -------
    bool
        Whether the labels refer to the same module boundary.
    """

    module_address = module_pass.rsplit(":", 1)[0]
    return module_pass == address or module_address == address


def _predicate_payload(value: Any) -> tuple[Callable[[Any], bool], str | None]:
    """Validate and unpack a predicate selector payload.

    Parameters
    ----------
    value:
        Stored predicate payload.

    Returns
    -------
    tuple[Callable[[Op], bool], str | None]
        Predicate and optional name hint.

    Raises
    ------
    SiteResolutionError
        If the predicate payload is malformed.
    """

    if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
        predicate = value[0]
        name_hint = value[1] if value[1] is None or isinstance(value[1], str) else None
        return predicate, name_hint
    if callable(value):
        return value, None
    raise SiteResolutionError("tl.where(...) requires a callable predicate.")


__all__ = ["SiteTable", "_selector_resolution_direction", "find_sites", "resolve_sites"]
