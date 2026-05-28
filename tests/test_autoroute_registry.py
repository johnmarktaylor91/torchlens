"""Tests for the TorchLens auto-route registry."""

from __future__ import annotations

from typing import Any

import pytest
from torch import nn

import torchlens as tl
from torchlens.autoroute._registry import Registry


class _NoForwardModel(nn.Module):
    """Minimal model used for registry-dispatch tests."""

    def forward(self, x: Any) -> Any:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Arbitrary test payload.

        Returns
        -------
        Any
            The original payload.
        """

        return x


def _detector_one(model: Any, payload: Any, **kwargs: Any) -> str | None:
    """Return a sentinel for registry tests."""

    return "one"


def _detector_two(model: Any, payload: Any, **kwargs: Any) -> str | None:
    """Return a second sentinel for registry tests."""

    return "two"


def test_registry_mechanics_register_list_iter_unregister() -> None:
    """Register, list, iterate, and unregister detectors."""

    registry = Registry(kind="test")

    registry.register(name="one", priority=20)(_detector_one)
    registry.register(name="two", priority=10)(_detector_two)

    assert [detector.name for detector in registry.list()] == ["two", "one"]
    assert list(registry.iter_by_priority()) == [_detector_two, _detector_one]

    registry.unregister("two")

    assert [detector.name for detector in registry.list()] == ["one"]


def test_duplicate_name_registration_raises() -> None:
    """Registering an existing built-in name should fail closed."""

    with tl.autoroute.input.snapshot():
        with pytest.raises(ValueError, match="detector 'hf_text' is already registered"):
            tl.autoroute.input.register(name="hf_text", priority=10)(_detector_one)


def test_override_pattern_unregister_then_register_replacement() -> None:
    """Unregistering a built-in allows explicit replacement by name."""

    sentinel = object()
    calls: list[tuple[Any, Any, dict[str, Any]]] = []

    def replacement(model: Any, payload: Any, **kwargs: Any) -> object:
        """Return a sentinel while recording dispatch arguments."""

        calls.append((model, payload, kwargs))
        return sentinel

    with tl.autoroute.input.snapshot():
        tl.autoroute.input.unregister("hf_text")
        tl.autoroute.input.register(name="hf_text", priority=10)(replacement)

        model = _NoForwardModel()
        result = tl.trace(model, "hello", layers_to_save="none")

    assert result is sentinel
    assert calls[0][0] is model
    assert calls[0][1] == "hello"
    assert calls[0][2]["layers_to_save"] == "none"


def test_list_and_list_glob() -> None:
    """Input registry list supports bare and glob-filtered forms."""

    all_names = [detector.name for detector in tl.autoroute.input.list()]
    hf_names = [detector.name for detector in tl.autoroute.input.list("hf_*")]

    assert {"hf_text", "hf_multimodal", "hf_image"}.issubset(all_names)
    assert hf_names == ["hf_text", "hf_multimodal", "hf_image"]


def test_info_returns_detector_metadata() -> None:
    """Input registry info returns user-facing diagnostic metadata."""

    info = tl.autoroute.input.info("hf_text")

    assert info["name"] == "hf_text"
    assert info["priority"] == 10
    assert info["kind"] == "input"
    assert info["func"] in tl.autoroute.input.iter_by_priority()
    assert info["qualname"] == "hf_text"
    assert "trace_text" in info["doc"]


def test_info_missing_raises_keyerror() -> None:
    """Missing detector info raises ``KeyError``."""

    with pytest.raises(KeyError, match="no detector named 'missing'"):
        tl.autoroute.input.info("missing")


def test_priority_ordering_is_deterministic() -> None:
    """Priority ordering is sorted by priority, then registration order."""

    registry = Registry(kind="test")

    def first(model: Any, payload: Any, **kwargs: Any) -> None:
        """First registered detector."""

    def second(model: Any, payload: Any, **kwargs: Any) -> None:
        """Second registered detector."""

    def third(model: Any, payload: Any, **kwargs: Any) -> None:
        """Third registered detector."""

    def fourth(model: Any, payload: Any, **kwargs: Any) -> None:
        """Fourth registered detector."""

    def fifth(model: Any, payload: Any, **kwargs: Any) -> None:
        """Fifth registered detector."""

    registry.register(name="third", priority=20)(third)
    registry.register(name="first", priority=10)(first)
    registry.register(name="fourth", priority=20)(fourth)
    registry.register(name="second", priority=10)(second)
    registry.register(name="fifth", priority=30)(fifth)

    assert list(registry.iter_by_priority()) == [first, second, third, fourth, fifth]


def test_output_namespace_is_reserved() -> None:
    """The output auto-route namespace is importable but reserved."""

    with pytest.raises(NotImplementedError, match="reserved for a future sprint"):
        tl.autoroute.output.list()
