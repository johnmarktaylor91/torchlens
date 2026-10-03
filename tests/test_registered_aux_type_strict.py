"""grind-p3 T11.9: registered-container aux is type-strict, subclasses refuse.

``_safe_aux`` admitted ``list``/``tuple`` SUBCLASSES through ``isinstance``
and encoded them as their plain base kind -- erasing the exact class and any
instance state (a namedtuple aux flattened to a bare ``tuple`` row), so two
registrations whose aux differ only in semantic sequence type produced
byte-identical witnesses. Exact-type nodes stay admitted; subclasses now
refuse typed (``registered_aux_unsafe``).
"""

from __future__ import annotations

import collections
from typing import Any

import pytest
import torch

from torchlens._input_walk import snapshot_input_boundary
from torchlens.ir.container import _CONTAINER_REGISTRY, register_container


class _Box:
    """Minimal registered container holding one tensor plus aux."""

    def __init__(self, value: torch.Tensor, aux: Any) -> None:
        self.value = value
        self.aux = aux


_AuxTuple = collections.namedtuple("_AuxTuple", ["mode"])


class _AuxList(list):
    """A list subclass carrying semantic type identity."""


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


def _register_box() -> None:
    """Register _Box with its live aux payload."""

    register_container(
        _Box,
        lambda box: ([box.value], box.aux),
        lambda aux, children: _Box(children[0], aux),
        state_complete=True,
    )


def _refusal_reasons(value: object) -> list[str]:
    """Snapshot one value and return its refusal reasons."""

    return [refusal["reason"] for refusal in snapshot_input_boundary(value)["refusals"]]


@pytest.mark.smoke
def test_exact_type_aux_stays_admitted(_registry: Any) -> None:
    """Plain list/tuple aux trees keep zero refusals and their exact kind."""

    _register_box()
    snap = snapshot_input_boundary(_Box(torch.ones(2), ("fast", 1, None)))
    assert snap["refusals"] == []
    (node,) = [n for n in snap["nodes"] if n["kind"] == "registered"]
    assert node["aux"][0] == "tuple"


def test_namedtuple_aux_refuses_typed(_registry: Any) -> None:
    """A namedtuple aux no longer launders into a bare tuple witness."""

    _register_box()
    reasons = _refusal_reasons(_Box(torch.ones(2), _AuxTuple("fast")))
    assert "registered_aux_unsafe" in reasons


def test_list_subclass_aux_refuses_typed(_registry: Any) -> None:
    """A list-subclass aux refuses instead of erasing its exact type."""

    _register_box()
    reasons = _refusal_reasons(_Box(torch.ones(2), _AuxList(["fast"])))
    assert "registered_aux_unsafe" in reasons


@pytest.mark.smoke
def test_nested_subclass_aux_refuses_typed(_registry: Any) -> None:
    """The exact-type rule applies at every nesting level of the aux tree."""

    _register_box()
    reasons = _refusal_reasons(_Box(torch.ones(2), ["outer", _AuxTuple("fast")]))
    assert "registered_aux_unsafe" in reasons
