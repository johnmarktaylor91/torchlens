"""The loadgroup stamp keeps ``xdist_group`` marks honoured on pytest 9.2+."""

from __future__ import annotations

from typing import Any

import pytest
from _xdist_groups import stamp_xdist_groups, xdist_group_suffix

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


def test_suffix_spells_xdist_names() -> None:
    by_name = _FakeItem("t.py::f[a]", [pytest.mark.xdist_group(name="cell").mark])
    positional = _FakeItem("t.py::f[a]", [pytest.mark.xdist_group("b").mark])
    two = _FakeItem(
        "t.py::f[a]", [pytest.mark.xdist_group("z").mark, pytest.mark.xdist_group("a").mark]
    )
    assert xdist_group_suffix(by_name) == "@cell"
    assert xdist_group_suffix(positional) == "@b"
    assert xdist_group_suffix(two) == "@a_z"
    assert xdist_group_suffix(_FakeItem("t.py::f", [pytest.mark.slow.mark])) is None


def test_stamp_writes_the_id_pytest_reads_once() -> None:
    marked = _FakeItem("tests/t.py::test_x[case-a-b]", [pytest.mark.xdist_group(name="g1").mark])
    plain = _FakeItem("tests/t.py::test_y", [])
    assert stamp_xdist_groups([marked, plain]) == 1
    assert marked.nodeid == "tests/t.py::test_x[case-a-b]@g1"
    assert plain.nodeid == "tests/t.py::test_y"
    # A second pass (or xdist's own write on a fixed release) never double-stamps.
    assert stamp_xdist_groups([marked, plain]) == 0
    assert marked.nodeid == "tests/t.py::test_x[case-a-b]@g1"


@pytest.mark.xdist_group(name="xdist_groups_self_check")
def test_loadgroup_worker_sees_its_own_group(request: pytest.FixtureRequest) -> None:
    """Under ``-n N --dist loadgroup`` this very item must carry its suffix."""

    if not getattr(request.config.option, "loadgroup", False):
        pytest.skip("only meaningful on a --dist loadgroup worker")
    assert request.node.nodeid.endswith("@xdist_groups_self_check")
