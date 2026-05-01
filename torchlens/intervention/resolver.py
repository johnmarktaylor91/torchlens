"""Resolve TorchLens intervention selectors against model logs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import importlib
import operator
from typing import TYPE_CHECKING, Any, Literal, cast
import warnings

import torch

from .errors import (
    MultiMatchWarning,
    ReplayPreconditionError,
    SiteAmbiguityError,
    SiteResolutionError,
)
from .selectors import BaseSelector, CompositeSelector, NotSelector, in_module
from .types import FrozenTargetSpec, FunctionRegistryKey, TargetSpec

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.layer_pass_log import LayerPassLog
    from torchlens.data_classes.model_log import ModelLog

SelectorInput = BaseSelector | TargetSpec | FrozenTargetSpec | str


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

    _sites: tuple["LayerPassLog", ...]
    query: Any | None = None

    def __len__(self) -> int:
        """Return the number of resolved sites.

        Returns
        -------
        int
            Number of layer-pass records in the table.
        """

        return len(self._sites)

    def __iter__(self) -> Iterator["LayerPassLog"]:
        """Iterate through resolved sites in execution order.

        Returns
        -------
        Iterator[LayerPassLog]
            Iterator over layer-pass records.
        """

        return iter(self._sites)

    def __getitem__(self, idx: int | slice) -> "LayerPassLog | SiteTable":
        """Return one site or a sliced site table.

        Parameters
        ----------
        idx:
            Integer index or slice.

        Returns
        -------
        LayerPassLog | SiteTable
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

    def where(self, predicate: Callable[["LayerPassLog"], bool]) -> "SiteTable":
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

    def first(self) -> "LayerPassLog":
        """Return the first resolved site.

        Returns
        -------
        LayerPassLog
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

        return tuple(str(site.layer_label) for site in self._sites)

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
                "layer_label": site.layer_label,
                "layer_label_w_pass": site.layer_label_w_pass,
                "layer_label_no_pass": site.layer_label_no_pass,
                "pass_num": site.pass_num,
                "operation_num": site.operation_num,
                "creation_order": site.creation_order,
                "func_name": site.func_name,
                "containing_module": site.containing_module,
                "containing_modules": site.containing_modules,
                "module_passes_exited": site.module_passes_exited,
            }
            for site in self._sites
        ]
        return pd.DataFrame(rows)


def resolve_sites(
    log: "ModelLog",
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

    sites = tuple(_iter_layer_passes(log))
    matched = _resolve_unchecked(sites, query, strict=strict)
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
    log: "ModelLog",
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

    return resolve_sites(log, query, strict=strict, max_fanout=max_fanout)


def _iter_layer_passes(log: "ModelLog") -> Sequence["LayerPassLog"]:
    """Return final layer passes from a completed model log.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    Sequence[LayerPassLog]
        Execution-order layer-pass records.

    Raises
    ------
    SiteResolutionError
        If the model log has not completed postprocessing.
    """

    if not getattr(log, "_pass_finished", False):
        raise SiteResolutionError("Sites can only be resolved after the forward pass is complete.")
    return log.layer_list


def _resolve_unchecked(
    sites: Sequence["LayerPassLog"],
    query: SelectorInput,
    *,
    strict: bool,
) -> tuple["LayerPassLog", ...]:
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
    tuple[LayerPassLog, ...]
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
        left_sites = set(_resolve_unchecked(sites, left, strict=strict))
        right_sites = set(_resolve_unchecked(sites, right, strict=strict))
        return tuple(site for site in sites if site in left_sites and site in right_sites)
    if kind == "or" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        left_sites = set(_resolve_unchecked(sites, left, strict=strict))
        right_sites = set(_resolve_unchecked(sites, right, strict=strict))
        return tuple(site for site in sites if site in left_sites or site in right_sites)
    if kind == "not" and isinstance(selector, NotSelector):
        excluded_sites = set(_resolve_unchecked(sites, selector.selector, strict=strict))
        return tuple(site for site in sites if site not in excluded_sites)

    if kind == "label":
        return tuple(site for site in sites if _label_matches(site, str(value)))
    if kind == "func":
        return tuple(site for site in sites if site.func_name == value)
    if kind == "module":
        return tuple(site for site in sites if _module_output_matches(site, str(value)))
    if kind == "contains":
        lowered = str(value).lower()
        return tuple(site for site in sites if lowered in str(site.layer_label).lower())
    if kind == "in_module":
        return tuple(site for site in sites if bool(in_module(site, str(value))))
    if kind == "predicate":
        predicate, _name_hint = _predicate_payload(value)
        return tuple(site for site in sites if predicate(site))

    raise SiteResolutionError(f"Unsupported selector kind {kind!r}.")


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

    from .selectors import contains, func, label, module, where

    if kind == "label":
        return label(str(value))
    if kind == "func":
        return func(str(value))
    if kind == "module":
        return module(str(value))
    if kind == "contains":
        return contains(str(value))
    if kind == "in_module":
        from .selectors import in_module as make_in_module

        selector = make_in_module(str(value))
        if isinstance(selector, BaseSelector):
            return selector
    if kind == "predicate" and callable(value):
        return where(value, name_hint=metadata.get("name_hint"))
    if kind == "not":
        nested = _normalize_query(value)
        return ~nested
    raise SiteResolutionError(f"Unsupported target spec selector kind {kind!r}.")


def _label_matches(site: "LayerPassLog", label: str) -> bool:
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
        site.layer_label,
        site.layer_label_w_pass,
        site.layer_label_no_pass,
        site.layer_label_short,
        site.layer_label_w_pass_short,
        site.layer_label_no_pass_short,
        site.layer_label_raw,
    )
    return label in candidate_labels or label in site.lookup_keys


def _module_output_matches(site: "LayerPassLog", address: str) -> bool:
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

    module_passes = getattr(site, "module_passes_exited", ())
    return any(_module_label_matches(module_pass, address) for module_pass in module_passes)


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


def _predicate_payload(value: Any) -> tuple[Callable[["LayerPassLog"], bool], str | None]:
    """Validate and unpack a predicate selector payload.

    Parameters
    ----------
    value:
        Stored predicate payload.

    Returns
    -------
    tuple[Callable[[LayerPassLog], bool], str | None]
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


__all__ = ["SiteTable", "find_sites", "resolve_sites"]
