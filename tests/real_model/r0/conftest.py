"""Session-scoped R0 fixtures: one build+trace per (family, impl) per session.

The deep sweep runs many assertions against each captured trace; caching the
capture keeps the whole R0 gate at seconds (memo section 10 PR row). Models
are ~100-225k params, inputs are 8 tokens -- the cache is a few MB.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
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
        if str(item.path).startswith(this_dir):
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


@pytest.fixture(scope="session")
def r0_capture_cache() -> dict[tuple[str, str], R0Capture]:
    return {}


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
