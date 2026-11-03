"""Bundle member operations: ``do``, ``fork``, ``run``, ``map``, ``show``, the chronology ledger and the effect table."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from ._lineage import BundleOperation, mint_bundle_id

if TYPE_CHECKING:
    from torch import nn

    from .._data_classes.trace import Trace
    from ..viz._drawing import Drawing
    from . import Bundle

__all__ = [
    "accepts_member_name",
    "bundle_do",
    "bundle_fork",
    "bundle_map",
    "bundle_run",
    "bundle_show",
    "effects_table",
    "record_bundle_operation",
]


def record_bundle_operation(
    self: Bundle,
    kind: str,
    *,
    member_names: Sequence[str] = (),
    params: Mapping[str, Any] | None = None,
) -> BundleOperation:
    """Append one chronology row to the container-side operation ledger."""

    previous = self._operations[-1] if self._operations else None
    row = BundleOperation(
        operation_id=mint_bundle_id(),
        seq=(previous.seq + 1) if previous is not None else 1,
        kind=kind,
        member_names=tuple(member_names),
        params=dict(params or {}),
        prev_operation_digest=(previous.operation_digest if previous is not None else None),
    )
    self._operations.append(row)
    return row


def bundle_do(self: Bundle, *args: Any, **kwargs: Any) -> Bundle:
    """Apply ``Trace.do`` to every member."""

    for member in self._members.values():
        member.do(*args, **kwargs)
    return self


def bundle_fork(self: Bundle, name: str | None = None) -> Bundle:
    """Fork all member logs into a new bundle (lineage survives the fork)."""

    from . import Bundle

    forked: OrderedDict[str, Trace] = OrderedDict()
    for member_name, member in self._members.items():
        fork_name = f"{name}_{member_name}" if name is not None else None
        forked[member_name] = member.fork(name=fork_name)
    child = Bundle(
        forked,
        baseline=self._baseline_name,
        relations=self._member_relations.rows,
        preserved_sections=self._preserved_sections,
    )
    child._forked_from_bundle_id = self._bundle_id
    self._record_bundle_operation(
        "fork",
        member_names=tuple(forked),
        params={"child_bundle_id": child._bundle_id},
    )
    child_row = child._record_bundle_operation(
        "fork",
        member_names=tuple(forked),
        params={"source_bundle_id": self._bundle_id},
    )
    child._member_construction = {
        member_name: {
            "origin": "forked",
            "source_bundle_id": self._bundle_id,
            "source_member": member_name,
            "operation_id": child_row.operation_id,
        }
        for member_name in forked
    }
    return child


def bundle_run(self: Bundle, model: nn.Module, x: Any = None, **kwargs: Any) -> Bundle:
    """Run all member logs with a supplied model and input."""

    for member in self._members.values():
        member.run(model, x, **kwargs)
    return self


def bundle_map(self: Bundle, fn: Callable[..., Any]) -> dict[str, Any]:
    """Apply a function independently to each member; released members read ``"unmeasured"``."""

    mode = _member_name_mode(fn)
    results: dict[str, Any] = {
        name: _call_with_member_name(fn, member, name, mode)
        for name, member in self._members.items()
    }
    for candidate in released_candidates(self):
        results.setdefault(candidate, "unmeasured")
    return results


def released_candidates(bundle: Bundle) -> list[str]:
    """Effect-table candidates with no retained member, sorted, each once.

    Parameters
    ----------
    bundle:
        Bundle whose effect tables are read.

    Returns
    -------
    list[str]
        Candidate ids a table records whose member was released, refused or
        failed; the ``__baseline__`` row is never listed.
    """
    released: set[str] = set()
    for table in bundle._effect_tables.values():
        for row in table.rows:
            if row.candidate_id == "__baseline__":
                continue
            if row.member_name is None or row.member_name not in bundle._members:
                released.add(row.candidate_id)
    return sorted(released)


def accepts_member_name(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` can take (member, name) rather than (member,) only."""

    return _member_name_mode(fn) != "none"


def _member_name_mode(fn: Callable[..., Any]) -> Literal["none", "positional", "keyword"]:
    """How ``Bundle.map`` hands ``fn`` the member's name, if at all.

    ``"positional"`` when the second positional parameter is REQUIRED (the
    documented ``lambda log, name: ...`` form) or ``fn`` takes ``*args``;
    ``"keyword"`` when ``fn`` declares a parameter called ``name`` that is not
    its first (``def f(log, *, name)``, ``def f(log, k=3, name=None)``);
    ``"none"`` otherwise, so a defaulted second parameter
    (``lambda log, k=3: ...``) keeps its default instead of receiving the
    name. Signature inspection failures (builtins, C callables) fall back to
    ``"none"``, never a guessed two-argument call that would raise
    mid-iteration.
    """

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return "none"
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    parameters = list(signature.parameters.values())
    positional = [parameter for parameter in parameters if parameter.kind in positional_kinds]
    if any(
        parameter.name == "name"
        and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        and parameter is not (positional[0] if positional else None)
        for parameter in parameters
    ):
        return "keyword"
    if len(positional) >= 2 and positional[1].default is inspect.Parameter.empty:
        return "positional"
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return "positional" if len(positional) <= 1 else "none"
    return "none"


def _call_with_member_name(
    fn: Callable[..., Any],
    member: Any,
    name: str,
    mode: Literal["none", "positional", "keyword"],
) -> Any:
    """Call ``fn`` on one member, passing its name the way ``mode`` says."""

    if mode == "positional":
        return fn(member, name)
    if mode == "keyword":
        return fn(member, name=name)
    return fn(member)


def bundle_show(self: Bundle, kind: str | None = None, **kwargs: Any) -> dict[str, Drawing | str]:
    """Render each bundle member graph as a member-keyed strip."""

    base_outpath = kwargs.get("path")
    results: dict[str, Drawing | str] = {}
    for name, member in self._members.items():
        member_kwargs = dict(kwargs)
        if isinstance(base_outpath, str):
            member_kwargs["path"] = f"{base_outpath}_{name}"
        if kind == "repr":
            results[name] = repr(member)
        elif kind == "html":
            results[name] = member._repr_html_()
        else:
            results[name] = member.draw(**member_kwargs)
    return results


def effects_table(self: Bundle, history_id: str | None = None) -> Any:
    """Serve the stored per-candidate effect table (``Bundle.effects``)."""

    from ..experiment._effects import bundle_effects

    return bundle_effects(self, history_id)
