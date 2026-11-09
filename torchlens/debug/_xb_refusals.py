"""Typed refusals of the module-level comparison in ``tl.debug``.

Each refusal has one raise site, so the refusal-site census classifies three rows.
"""

from __future__ import annotations

from typing import NoReturn

from .._errors import InvalidArgumentError
from ..backends._errors import BackendUnsupportedError


def refuse_align(problem: str, remedy: str) -> NoReturn:
    """Refuse an ``align=`` or ``module_map=`` value outside its domain.

    Raises
    ------
    InvalidArgumentError
        Code ``compare_align_invalid``.
    """

    raise InvalidArgumentError(problem, code="compare_align_invalid", remedy=remedy)


def refuse_backend(backend_a: str, backend_b: str, admitted: frozenset[str]) -> NoReturn:
    """Refuse a pair whose backend has not passed the module-recording checks.

    Raises
    ------
    BackendUnsupportedError
        Code ``backend_unsupported``; the message names the admitted backends.
    """

    names = ", ".join(sorted(admitted))
    raise BackendUnsupportedError(
        f"cannot compare a {backend_a!r} trace with a {backend_b!r} trace module by module: "
        "module-level comparison admits a backend only once its recorded module outputs pass "
        f"TorchLens's conformance checks; admitted today: {names}",
        code="backend_unsupported",
        remedy=f"compare two traces captured on the same backend, or traces from: {names}",
        backend_a=backend_a,
        backend_b=backend_b,
    )


def refuse_inconclusive(problem: str, remedy: str) -> NoReturn:
    """Refuse a module-level question that the two traces cannot answer.

    Raises
    ------
    InvalidArgumentError
        Code ``module_comparison_inconclusive``.
    """

    raise InvalidArgumentError(problem, code="module_comparison_inconclusive", remedy=remedy)
