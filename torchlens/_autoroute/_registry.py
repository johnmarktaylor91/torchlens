"""Direction-agnostic priority registry for TorchLens auto-routing."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


def class_names_tuple(class_name: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize a ``class_name=`` registration filter to a tuple of patterns."""

    if class_name is None:
        return ()
    if isinstance(class_name, str):
        return (class_name,)
    return tuple(class_name)


def matches_class_name(patterns: tuple[str, ...], class_name: str | None) -> bool:
    """Return whether ``class_name`` passes a registration filter (empty matches all)."""

    if not patterns:
        return True
    if class_name is None:
        return False
    return any(fnmatch.fnmatchcase(class_name, pattern) for pattern in patterns)


@dataclass(frozen=True)
class Detector:
    """Registered detector metadata.

    Parameters
    ----------
    name:
        Unique detector name within one registry.
    priority:
        Sort key. Lower values run first.
    registration_order:
        Stable tiebreaker for detectors with equal priority.
    func:
        Detector callable.
    """

    name: str
    priority: int
    registration_order: int
    func: Callable[..., Any]


class Registry:
    """Generic priority-ordered registry of detector callables.

    Detectors are functions of signature ``(model, payload, **kwargs) -> Trace |
    InterpretedOutput | None``. Return ``None`` to indicate "this detector does
    not match"; return a non-None result to claim the dispatch.

    The registry is direction-agnostic: the ``kind`` field is purely for diagnostics.
    One instance serves input dispatch and one serves output decoding; the
    public registration doors are ``tl.inputs`` and ``tl.outputs``.
    """

    def __init__(self, kind: str) -> None:
        """Initialize an empty detector registry.

        Parameters
        ----------
        kind:
            Diagnostic registry name, such as ``"input"`` or ``"output"``.
        """

        self.kind = kind
        self._detectors: list[Detector] = []
        self._counter = 0

    def register(
        self, *, name: str, priority: int = 100
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that registers a detector function.

        Parameters
        ----------
        name:
            Unique detector name within this registry.
        priority:
            Sort key. Lower values run first.

        Returns
        -------
        Callable[[Callable[..., Any]], Callable[..., Any]]
            Decorator that registers and returns the original function.

        Raises
        ------
        ValueError
            If a detector with ``name`` is already registered.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Register ``func`` and return it unchanged.

            Parameters
            ----------
            func:
                Detector callable.

            Returns
            -------
            Callable[..., Any]
                The original detector callable.
            """

            if any(detector.name == name for detector in self._detectors):
                raise ValueError(
                    f"detector {name!r} is already registered on {self.kind} registry; "
                    f"call tl.{self.kind}s.unregister({name!r}) first to replace"
                )
            self._counter += 1
            self._detectors.append(
                Detector(
                    name=name,
                    priority=priority,
                    registration_order=self._counter,
                    func=func,
                )
            )
            self._detectors.sort(
                key=lambda detector: (detector.priority, detector.registration_order)
            )
            return func

        return decorator

    def unregister(self, name: str) -> None:
        """Remove a detector by name.

        Parameters
        ----------
        name:
            Registered detector name.

        Raises
        ------
        KeyError
            If no detector with ``name`` exists.
        """

        before = len(self._detectors)
        self._detectors = [detector for detector in self._detectors if detector.name != name]
        if len(self._detectors) == before:
            raise KeyError(f"no detector named {name!r} on {self.kind} registry")

    def iter_by_priority(self) -> Iterator[Callable[..., Any]]:
        """Yield registered detector functions in dispatch order.

        Yields
        ------
        Callable[..., Any]
            Registered detector callable.
        """

        for detector in self._detectors:
            yield detector.func

    def list(self, name_glob: str | None = None) -> list[Detector]:
        """List registered detectors, optionally filtered by glob.

        Parameters
        ----------
        name_glob:
            Optional shell-style glob pattern matched against detector names.

        Returns
        -------
        list[Detector]
            Matching detector metadata in dispatch order.
        """

        if name_glob is None:
            return list(self._detectors)
        return [
            detector for detector in self._detectors if fnmatch.fnmatch(detector.name, name_glob)
        ]

    def info(self, name: str) -> dict[str, Any]:
        """Return diagnostic metadata for a registered detector.

        Parameters
        ----------
        name:
            Registered detector name.

        Returns
        -------
        dict[str, Any]
            Detector metadata.

        Raises
        ------
        KeyError
            If no detector with ``name`` exists.
        """

        for detector in self._detectors:
            if detector.name == name:
                return {
                    "name": detector.name,
                    "priority": detector.priority,
                    "kind": self.kind,
                    "func": detector.func,
                    "qualname": getattr(detector.func, "__qualname__", None),
                    "doc": (detector.func.__doc__ or "").strip(),
                }
        raise KeyError(f"no detector named {name!r} on {self.kind} registry")

    @contextmanager
    def snapshot(self) -> Iterator[None]:
        """Temporarily snapshot registry state and restore it on exit.

        Yields
        ------
        None
            Context body execution point.
        """

        saved = list(self._detectors)
        saved_counter = self._counter
        try:
            yield
        finally:
            self._detectors = saved
            self._counter = saved_counter
