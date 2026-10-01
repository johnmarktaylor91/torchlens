"""Package-scoped R0 fixtures: one build+trace per (family, impl) per R0 block.

The deep sweep runs many assertions against each captured trace; caching the
capture keeps the whole R0 gate at seconds (memo section 10 PR row). Models
are ~100-225k params, inputs are 8 tokens -- the cache is a few MB.

The cache is PACKAGE-scoped and emptied when the session leaves this package
(FLOORLEAK). The historical session scope kept every family's model AND its
finished ``Trace`` alive for the rest of the process: a default capture's
saved activations carry autograd history, so each cached trace also pinned
its whole live forward graph (~100 gc-visible activation tensors for whisper
alone), and the process-wide live-holder tripwires in
``tests/test_brainpipe_capture_floor.py`` read red for every later test in
the session. A cache that outlives the tests that read it is a fixture leak,
not a capture leak.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch.nn as nn

from tests.real_model.r0.families import FAMILIES, FAMILY_BY_NAME, FamilySpec

# GATE-ID: R0_TRANSFORMERS_RUNTIME
#   kind: runtime-dependency gate (import availability, never version parsing)
#   executing legs: every tests.yml matrix row with a `transformers` matrix
#   pin (torch >= 2.4 rows; transformers 5.x floor), the latest-canary R0
#   step, the U/R0 train gates (repo venv). The three torch<2.4 floor rows
#   deliberately do not claim the sweep until B2's HF_FLOOR leg lands
#   per-version expectations. Enters the C1 gate-witness manifest.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every transformers-dependent item when the floor row has none installed.

    L8 floor fix: this gate originally only matched items whose PATH sat under
    ``tests/real_model/r0/``, so sibling files elsewhere in ``tests/`` that
    import ``tests.real_model.r0.families`` (and transitively transformers) at
    RUNTIME -- e.g. ``tests/test_sem_resid_a02.py`` -- crashed with
    ``ModuleNotFoundError`` on the floor rows instead of skipping cleanly. The
    registered ``real_model`` marker (``pyproject.toml``: "the R0 gate selects
    -m 'smoke or real_model'") is the actual session-wide signal a test needs
    this runtime, so match on it directly in addition to the directory check
    (hook implementations receive the full session item list regardless of
    which conftest registered them).
    """

    if importlib.util.find_spec("transformers") and importlib.util.find_spec("torchvision"):
        return
    if os.environ.get("TORCHLENS_REQUIRE_R0") == "1":
        raise pytest.UsageError(
            "GATE R0_TRANSFORMERS_RUNTIME: this leg claims the R0 sweep but"
            " transformers/torchvision are not importable; the leg's install is"
            " broken -- fix the env, never skip a claimed sweep."
        )
    marker = pytest.mark.skip(
        reason="GATE R0_TRANSFORMERS_RUNTIME: transformers/torchvision not"
        " installed; the R0 sweep executes on every transformers-pinned"
        " tests.yml row, the canary, and the train gates."
    )
    this_dir = os.path.dirname(__file__)
    for item in items:
        if str(item.path).startswith(this_dir) or item.get_closest_marker("real_model") is not None:
            item.add_marker(marker)


# (family, impl) parametrization ids for the sweep, e.g. "gpt2-sdpa".
FAMILY_IMPL_PARAMS = [
    pytest.param(spec.name, impl, id=f"{spec.name}-{impl}")
    for spec in FAMILIES
    for impl in spec.impls
]


@dataclass
class R0Capture:
    """One cached R0 capture with its coverage report."""

    spec: FamilySpec
    impl: str
    model: nn.Module
    input_args: tuple[Any, ...]
    input_kwargs: dict[str, Any]
    trace: Any
    coverage: Any


def r0_cache_lifetime() -> Iterator[dict[tuple[str, str], R0Capture]]:
    """Yield a fresh capture cache; empty it when the lifetime closes.

    The explicit ``clear()`` (rather than trusting the dict to die with the
    fixture value) releases every cached model and Trace even if a consumer
    kept a reference to the dict itself; ``gc.collect()`` then reclaims the
    autograd graphs the traces' saved activations kept alive, so the tests
    that follow start from the process baseline.
    """

    cache: dict[tuple[str, str], R0Capture] = {}
    try:
        yield cache
    finally:
        cache.clear()
        gc.collect()


@pytest.fixture(scope="package")
def r0_capture_cache() -> Iterator[dict[tuple[str, str], R0Capture]]:
    yield from r0_cache_lifetime()


@pytest.fixture()
def r0_capture(request: pytest.FixtureRequest, r0_capture_cache: dict) -> Any:
    """Factory: ``r0_capture(family, impl)`` -> cached :class:`R0Capture`."""

    import torchlens as tl
    from torchlens.semantic.coverage import facet_coverage

    def _get(family: str, impl: str) -> R0Capture:
        key = (family, impl)
        if key not in r0_capture_cache:
            spec = FAMILY_BY_NAME[family]
            if impl not in spec.impls:
                raise ValueError(f"{family} does not build under {impl!r}")
            with warnings.catch_warnings():
                # Shrunk-vocab configs re-print upstream token-id range
                # advisories; the families module sets ids in-range where the
                # class enforces them, and the rest are upstream chatter.
                warnings.simplefilter("ignore")
                model = spec.build(impl)
                input_args = spec.input_args()
                input_kwargs = spec.input_kwargs()
                trace = tl.trace(model, input_args, input_kwargs)
            r0_capture_cache[key] = R0Capture(
                spec=spec,
                impl=impl,
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                trace=trace,
                coverage=facet_coverage(trace),
            )
        return r0_capture_cache[key]

    return _get
