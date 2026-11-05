"""Generator context managers that never write onto the exception passing through.

``contextlib.contextmanager`` throws an in-flight exception into the generator
and, on Python 3.11 and later, assigns ``exc.__traceback__`` when the generator
re-raises it. A user exception whose ``__setattr__`` refuses writes is then
replaced by that ``AttributeError`` before it reaches the caller. The capture
path wraps the model's forward in several generator brackets, so each one uses
:func:`transparent_contextmanager`: the same enter, exit, suppression and
``StopIteration`` rules as ``contextlib``, minus the attribute write.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, cast

_Params = ParamSpec("_Params")
_Yield = TypeVar("_Yield")


class _TransparentGeneratorContext(Generic[_Yield]):
    """Drive one generator as a context manager without touching the exception."""

    __slots__ = ("_generator",)

    def __init__(self, generator: Iterator[_Yield]) -> None:
        """Hold the generator, started on enter.

        Parameters
        ----------
        generator:
            Generator that yields exactly once.
        """

        self._generator = cast(Generator[_Yield, None, None], generator)

    def __enter__(self) -> _Yield:
        """Run the generator up to its ``yield``.

        Returns
        -------
        _Yield
            The value the generator yields.
        """

        try:
            return next(self._generator)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(
        self, typ: type[BaseException] | None, value: BaseException | None, tb: Any
    ) -> bool:
        """Finish the generator, throwing any in-flight exception into it.

        Parameters
        ----------
        typ:
            Type of the in-flight exception, or ``None``.
        value:
            The in-flight exception, or ``None``.
        tb:
            Its traceback (unused: the exception keeps its own).

        Returns
        -------
        bool
            ``True`` only when the generator suppressed the exception.
        """

        if typ is None:
            try:
                next(self._generator)
            except StopIteration:
                return False
            raise RuntimeError("generator didn't stop")
        if value is None:
            value = typ()
        return self._throw(value)

    def _throw(self, value: BaseException) -> bool:
        """Throw ``value`` into the generator; report whether it was suppressed.

        Parameters
        ----------
        value:
            The in-flight exception.

        Returns
        -------
        bool
            ``True`` when the generator swallowed ``value``.
        """

        try:
            self._generator.throw(value)
        except StopIteration as exc:
            return exc is not value
        except RuntimeError as exc:
            # PEP 479 wraps a StopIteration thrown into a generator.
            if exc is value or (isinstance(value, StopIteration) and exc.__cause__ is value):
                return False
            raise
        except BaseException as exc:
            if exc is not value:
                raise
            return False
        raise RuntimeError("generator didn't stop after throw()")


def transparent_contextmanager(
    func: Callable[_Params, Iterator[_Yield]],
) -> Callable[_Params, _TransparentGeneratorContext[_Yield]]:
    """Decorate a one-yield generator like ``contextlib.contextmanager``.

    Parameters
    ----------
    func:
        Generator function that yields exactly once.

    Returns
    -------
    Callable
        Factory returning a context manager that never writes onto the
        exception passing through it.
    """

    @wraps(func)
    def _open(
        *args: _Params.args, **kwargs: _Params.kwargs
    ) -> _TransparentGeneratorContext[_Yield]:
        """Build the context manager for one use."""

        return _TransparentGeneratorContext(func(*args, **kwargs))

    return _open
