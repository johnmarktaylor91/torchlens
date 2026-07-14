"""Tests for transparent native forward adapters."""

from __future__ import annotations

from typing import Any

from menagerie.crawler.frameworks import NativeForwardAdapter


class _Native:
    """Tiny native-call fixture."""

    def __init__(self) -> None:
        """Initialize call capture."""

        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def apply(self, *args: object, **kwargs: object) -> object:
        """Record and return the exact native output object.

        Parameters
        ----------
        *args, **kwargs:
            Native call values.

        Returns
        -------
        object
            Exact sentinel output.
        """

        self.calls.append((args, kwargs))
        return args[0]


def test_forward_adapter_delegates_without_transforming_output() -> None:
    """Transparent forward invokes the recorded native call exactly once."""

    native = _Native()
    adapter = NativeForwardAdapter(native, original_framework="jax", call_method="apply")
    sentinel: Any = object()

    output = adapter.forward(sentinel, flag=True)

    assert output is sentinel
    assert native.calls == [((sentinel,), {"flag": True})]
    assert adapter.metadata.native_call_method == "apply"
