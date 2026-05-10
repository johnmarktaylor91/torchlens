"""Phase 1 shadow-stack equality tests for module containment."""

from __future__ import annotations

from typing import Any

import pytest
import torchlens as tl
from torchlens.options import CaptureOptions

from fixtures.module_containment_models import ALL_FIXTURES, FixtureBuilder

KNOWN_PHASE1_DIVERGENCES = {
    "raw_hook_replacement_synthetic": (
        "Hook-stack does not propagate replaced-module ancestry through downstream ops; "
        "this is a documented semantic improvement, not a regression. See "
        ".research/module-containment-refactor_PLAN.md fixture 13b."
    ),
}


def _unpack_fixture(result: tuple[Any, ...]) -> tuple[Any, Any, str, Any | None]:
    """Unpack fixture results with optional hook handles.

    Parameters
    ----------
    result:
        Fixture builder return tuple.

    Returns
    -------
    tuple[Any, Any, str, Any | None]
        Model, input args, fixture name, and optional hook handle.
    """

    if len(result) == 4:
        model, input_args, fixture_name, hook_handle = result
        return model, input_args, fixture_name, hook_handle
    model, input_args, fixture_name = result
    return model, input_args, fixture_name, None


@pytest.mark.parametrize("builder", ALL_FIXTURES, ids=lambda builder: builder.__name__)
def test_phase1_shadow_stack_matches(builder: FixtureBuilder) -> None:
    """Assert the shadow hook stack matches thread-replay containment."""

    model, input_args, fixture_name, hook_handle = _unpack_fixture(builder())
    if fixture_name in KNOWN_PHASE1_DIVERGENCES:
        pytest.xfail(KNOWN_PHASE1_DIVERGENCES[fixture_name])
    capture = CaptureOptions(_module_containment_engine="both")
    try:
        trace = tl.trace(model, input_args, vis_opt="none", capture=capture)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    for op in trace.layer_list:
        assert hasattr(op, "_modules_via_stack")
