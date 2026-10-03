"""grind-p3 T11.6: hybrid registered containers keep one kind vocabulary.

The runtime structure-witness kind (``_container_kind``) checked the generic
kinds (dataclass / namedtuple / tuple / list / dict) BEFORE the registered
lookup, while the capture-side ``ContainerSpec`` builder checks the
registration FIRST. A HYBRID registered container (a registered dataclass or
namedtuple/tuple subclass) therefore recorded ``"registered"`` at capture but
reported its generic kind at run time, and an honest byte-identical run
false-DIVERGED with ``OUTPUT_STRUCTURE_MISMATCH``. Both sides now dispatch
registered-first.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import pytest
import torch

from torchlens._runnable_execution import _container_kind, _container_leaf_paths
from torchlens.ir.container import _CONTAINER_REGISTRY, register_container
from torchlens.ir.container_registry import _build_container_spec


@dataclasses.dataclass
class _HybridDataclass:
    """Dataclass container the user ALSO registers explicitly."""

    value: torch.Tensor


_HybridTuple = collections.namedtuple("_HybridTuple", ["value"])


class _HybridList(list):
    """List subclass the user ALSO registers explicitly."""


@pytest.fixture
def _registry() -> Any:
    """Isolate the container registry per test."""

    saved = dict(_CONTAINER_REGISTRY)
    _CONTAINER_REGISTRY.clear()
    try:
        yield _CONTAINER_REGISTRY
    finally:
        _CONTAINER_REGISTRY.clear()
        _CONTAINER_REGISTRY.update(saved)


def _register(cls: type, rebuild: Any) -> None:
    """Register ``cls`` with a one-child flatten."""

    register_container(
        cls,
        lambda box: ([box[0] if isinstance(box, (list, tuple)) else box.value], None),
        rebuild,
        state_complete=True,
    )


def test_registered_dataclass_reports_registered(_registry: Any) -> None:
    """Runtime kind for a registered dataclass matches capture's vocabulary."""

    _register(_HybridDataclass, lambda aux, children: _HybridDataclass(children[0]))
    box = _HybridDataclass(torch.ones(2))
    assert _container_kind(box) == "registered"
    spec = _build_container_spec(box)
    assert spec is not None and spec.kind == "registered"


def test_registered_namedtuple_reports_registered(_registry: Any) -> None:
    """Runtime kind for a registered namedtuple matches capture's vocabulary."""

    _register(_HybridTuple, lambda aux, children: _HybridTuple(children[0]))
    box = _HybridTuple(torch.ones(2))
    assert _container_kind(box) == "registered"
    spec = _build_container_spec(box)
    assert spec is not None and spec.kind == "registered"


def test_registered_list_subclass_reports_registered(_registry: Any) -> None:
    """Runtime kind for a registered list subclass matches capture's vocabulary."""

    _register(_HybridList, lambda aux, children: _HybridList(children))
    box = _HybridList([torch.ones(2)])
    assert _container_kind(box) == "registered"
    spec = _build_container_spec(box)
    assert spec is not None and spec.kind == "registered"


def test_registered_flatten_wins_for_hybrid_leaf_paths(_registry: Any) -> None:
    """Runtime leaf paths follow the registration's OWN flatten, like capture.

    A registered list subclass whose flatten selects a child subset used to be
    walked by generic indexing at run time (2 paths) while capture recorded
    the flatten view (1 child) -- an honest run then false-diverged on paths.
    """

    register_container(
        _HybridList,
        lambda box: ([box[0]], None),
        lambda aux, children: _HybridList(children),
        state_complete=True,
    )
    box = _HybridList([torch.ones(1), torch.zeros(1)])
    assert _container_leaf_paths(box) == ((0,),)
    spec = _build_container_spec(box)
    assert spec is not None and spec.kind == "registered" and spec.length == 1


def test_unregistered_generic_kinds_unchanged(_registry: Any) -> None:
    """Without a registration the generic vocabulary is untouched."""

    assert _container_kind(_HybridDataclass(torch.ones(2))) == "dataclass"
    assert _container_kind(_HybridTuple(torch.ones(2))) == "namedtuple"
    assert _container_kind([torch.ones(2)]) == "list"
    assert _container_kind((torch.ones(2),)) == "tuple"
    assert _container_kind({"x": torch.ones(2)}) == "dict"
    assert _container_kind(torch.ones(2)) == "tensor"
