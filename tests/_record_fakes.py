"""A ``SimpleNamespace`` stand-in for an Op or Layer that serves the concern groups.

Unit tests drive postprocess and cleanup helpers with light record fakes. Since
the fields.1 regroup those helpers read ``record.control_flow.branch_stack`` and
write the storage attribute (``_control_flow_branch_stack``), so a fake built
from the persisted names stores each moved field under its storage attribute
and serves the real group views over it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from torchlens._data_classes._record_groups import OpControlFlow, OpCost
from torchlens._data_classes._record_groups_autograd import OpAutograd
from torchlens._data_classes._record_groups_call import OpCall
from torchlens._data_classes._record_groups_graph import OpGraph
from torchlens._data_classes._record_groups_module_context import OpModuleContext
from torchlens._data_classes._record_groups_payload import OpPayload
from torchlens._data_classes._record_groups_source import OpSource
from torchlens._utils._record_group_table import (
    ACTIVE_GROUPS,
    LAYER_MEMBER_MOVES,
    OP_MEMBER_MOVES,
    op_attr,
)

#: ``(group, leaf) -> former flat name`` over both records (Op rows win).
_FLAT_NAME = {
    (group, leaf): old
    for moves in (LAYER_MEMBER_MOVES, OP_MEMBER_MOVES)
    for old, group, leaf in moves
    if group in ACTIVE_GROUPS
}


class FakeRecord(SimpleNamespace):
    """Record fake: persisted field names in, group views out."""

    def __init__(self, **fields: Any) -> None:
        """Store each field under the attribute the real Op keeps it in."""

        super().__init__(**{op_attr(name): value for name, value in fields.items()})

    @property
    def cost(self) -> OpCost:
        """The real cost group over this fake."""

        return OpCost(self)

    @property
    def control_flow(self) -> OpControlFlow:
        """The real control-flow group over this fake."""

        return OpControlFlow(self)

    @property
    def call(self) -> OpCall:
        """The real call group over this fake."""

        return OpCall(self)

    @property
    def source(self) -> OpSource:
        """The real source group over this fake."""

        return OpSource(self)

    @property
    def autograd(self) -> OpAutograd:
        """The real autograd group over this fake."""

        return OpAutograd(self)

    @property
    def module_context(self) -> OpModuleContext:
        """The real module-context group over this fake."""

        return OpModuleContext(self)

    @property
    def payload(self) -> OpPayload:
        """The real payload group over this fake."""

        return OpPayload(self)

    @property
    def graph(self) -> OpGraph:
        """The real graph group over this fake."""

        return OpGraph(self)


class _FlatGroupView:
    """``fake.<group>.<leaf>`` read from the fake's former flat attribute."""

    __slots__ = ("_group", "_record")

    def __init__(self, record: Any, group: str) -> None:
        """Bind the fake and the group name."""

        self._record = record
        self._group = group

    def __getattr__(self, leaf: str) -> Any:
        """Return the flat attribute the leaf replaced."""

        old = _FLAT_NAME.get((self._group, leaf))
        if old is None:
            raise AttributeError(leaf)
        return getattr(self._record, old)


class FlatGroups:
    """Mixin for hand-written duck-typed fakes that keep flat field names.

    A fake declaring ``func_name`` or ``saved_kwargs`` flat still answers the
    grouped spelling the library reads (``op.call.saved_kwargs``).
    """

    def __getattr__(self, name: str) -> Any:
        """Serve each applied concern group as a flat-attribute view."""

        if name in ACTIVE_GROUPS:
            return _FlatGroupView(self, name)
        raise AttributeError(name)


class FlatNamespace(FlatGroups, SimpleNamespace):
    """``SimpleNamespace`` fake with flat fields that also serves group spellings."""
