"""Arrays a forward can reach from outside itself: the closure-constant census.

The shared constants rule (SPEC section 2) records an array the forward used
that is neither an input, a parameter nor a recorded op output as a
``constant`` source op. An array produced *inside* the forward by an op no
wrapper saw is not a constant: it is the unwrapped-op gap and must trip the
unwrapped policy instead of being laundered into a saved constant. The two
are told apart by reachability before the forward runs: this module collects
every array reachable from the model object through closure cells, the
globals its code names, bound-method instances and small containers, to a
bounded depth. An unlabeled argument whose identity is in that census is a
closure constant; one that is not came from an unwrapped op.
"""

from __future__ import annotations

import functools
import inspect
import types
from collections.abc import Callable
from typing import Any

#: Container sizes above this are not scanned (a dataset is not a constant).
_MAX_CONTAINER_ITEMS = 64
#: How many function hops (closure cell -> inner function -> ...) are followed.
_MAX_FUNCTION_DEPTH = 3


def external_array_ids(model: object, is_tensor: Callable[[object], bool]) -> dict[int, Any]:
    """Arrays reachable from ``model`` before its forward runs, keyed by identity.

    Parameters
    ----------
    model:
        The traced callable: a function, a bound method, a ``functools.partial``
        or a callable object (an ``mlx.nn.Module`` or a plain class instance).
    is_tensor:
        Predicate for the framework's array type.

    Returns
    -------
    dict[int, Any]
        ``id(array) -> array`` for every reachable array. Holding the arrays
        keeps their identities valid for the capture's lifetime.
    """

    found: dict[int, Any] = {}
    visited_functions: set[int] = set()
    visited_containers: set[int] = set()

    def collect(value: Any, depth: int) -> None:
        if is_tensor(value):
            found[id(value)] = value
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in visited_containers or len(value) > _MAX_CONTAINER_ITEMS:
                return
            visited_containers.add(id(value))
            for item in value:
                collect(item, depth)
            return
        if isinstance(value, dict):
            if id(value) in visited_containers or len(value) > _MAX_CONTAINER_ITEMS:
                return
            visited_containers.add(id(value))
            for item in value.values():
                collect(item, depth)
            return
        if depth > 0 and callable(value) and not isinstance(value, type):
            visit_callable(value, depth - 1)

    def visit_code_names(function: types.FunctionType, depth: int) -> None:
        module_globals = getattr(function, "__globals__", None)
        if not isinstance(module_globals, dict):
            return
        pending = [function.__code__]
        seen_codes: set[int] = set()
        while pending:
            code = pending.pop()
            if id(code) in seen_codes:
                continue
            seen_codes.add(id(code))
            for name in code.co_names:
                if name in module_globals:
                    collect(module_globals[name], depth)
            pending.extend(const for const in code.co_consts if isinstance(const, types.CodeType))

    def visit_callable(value: Any, depth: int) -> None:
        if id(value) in visited_functions:
            return
        visited_functions.add(id(value))
        if isinstance(value, functools.partial):
            collect(value.args, depth)
            collect(value.keywords, depth)
            visit_callable(value.func, depth)
            return
        if inspect.ismethod(value):
            collect(vars(value.__self__) if hasattr(value.__self__, "__dict__") else {}, depth)
            visit_callable(value.__func__, depth)
            return
        if isinstance(value, types.FunctionType):
            for cell in value.__closure__ or ():
                try:
                    collect(cell.cell_contents, depth)
                except ValueError:  # empty cell
                    continue
            visit_code_names(value, depth)
            return
        # A callable object: its instance attributes and its class __call__.
        instance_dict = getattr(value, "__dict__", None)
        if isinstance(instance_dict, dict):
            collect(instance_dict, depth)
        call = inspect.getattr_static(type(value), "__call__", None)
        if isinstance(call, types.FunctionType):
            visit_callable(call, depth)

    visit_callable(model, _MAX_FUNCTION_DEPTH)
    return found


__all__ = ["external_array_ids"]
