"""The loadgroup stamp keeps ``xdist_group`` marks honoured on pytest 9.2+."""

from __future__ import annotations

from typing import Any

import pytest

nodeid_module = pytest.importorskip("_pytest.nodeid", reason="structured node ids are pytest 9.2+")


class _FakeItem:
    """The three item attributes the stamp reads, on a structured node id."""

    def __init__(self, nodeid: str, marks: list[Any]) -> None:
        self._id = nodeid_module.NodeId.parse(nodeid)
        self._marks = marks

    @property
    def nodeid(self) -> str:
        return str(self._id)

    def iter_markers(self, name: str) -> list[Any]:
        return [mark for mark in self._marks if mark.name == name]


@pytest.mark.xdist_group(name="xdist_groups_self_check")
def test_loadgroup_worker_sees_its_own_group(request: pytest.FixtureRequest) -> None:
    """Under ``-n N --dist loadgroup`` this very item must carry its suffix."""

    if not getattr(request.config.option, "loadgroup", False):
        pytest.skip("only meaningful on a --dist loadgroup worker")
    assert request.node.nodeid.endswith("@xdist_groups_self_check")
