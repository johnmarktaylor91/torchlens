"""The CPU Float8 allocation guard, importable from an unambiguous dotted path.

``tests/conftest.py`` and ``tests/backends/conftest.py`` are two DIFFERENT
files that both resolve to the bare top-level module name ``conftest`` under
pytest's default ``prepend`` import mode (neither ``tests/`` nor
``tests/backends/`` is a package). A consumer doing ``from conftest import
...`` gets whichever one Python's module cache (``sys.modules["conftest"]``)
happened to load first -- collection order, not file identity, decides which
one -- so the floor2 fix's ``from conftest import permit_cpu_float8_allocation``
in two test files broke collection the moment ``tests/backends/conftest.py``
(alphabetically first) won that race. ``tests/support/`` is a real package
(``__init__.py``), so ``support.fp8_guard`` names exactly one module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def permit_cpu_float8_allocation() -> Iterator[None]:
    """Temporarily relax forced determinism so a fresh Float8 CPU tensor can allocate.

    ``tests/conftest.py``'s ``_reset_rng_state`` forces
    ``torch.use_deterministic_algorithms(True)`` for every test (RNG
    reproducibility). torch 2.1/2.2's CPU ``fill_empty_deterministic_``
    kernel does not cover Float8 dtypes, so allocating any fresh Float8 CPU
    tensor (``.to(float8_dtype)``, ``torch.empty(dtype=float8_dtype)``, ...)
    under that forced determinism raises ``RuntimeError:
    "fill_empty_deterministic_" not implemented for 'Float8_...'`` -- a
    genuine torch CPU limitation (feature-detected as
    ``HAS_CPU_FLOAT8_DETERMINISTIC_FILL``), not a per-call bug. A no-op when
    the running torch covers it.
    """

    from torchlens.utils._torch_compat import get_cpu_float8_deterministic_fill_support

    if get_cpu_float8_deterministic_fill_support(force_probe=True):
        yield
        return
    was = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn_only)
