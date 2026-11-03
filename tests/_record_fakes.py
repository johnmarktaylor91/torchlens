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
from torchlens._data_classes._record_groups_call import OpCall
from torchlens._utils._record_group_table import op_attr


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
