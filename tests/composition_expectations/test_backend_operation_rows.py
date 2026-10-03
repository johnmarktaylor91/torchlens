"""Operation-grain backend capability rows (compo row 0.7).

The five coarse ``BackendCapabilities`` flags gate ten public trace options
(``backends/_options.py``'s mapping -- the memo's migration inventory), so a
backend supporting ``intervene=`` but not ``halt=`` is inexpressible today.
The ledger references STABLE per-option row ids; the future flag split
refines ``operation_grain_capability_rows`` (an authority change), never the
ledger.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.compo


def test_every_backend_gets_a_row_per_gated_option() -> None:
    """10 gated options x every registered backend, unique stable row ids."""

    from torchlens.backends._options import (
        TRACE_OPTION_CAPABILITY_GATES,
        operation_grain_capability_rows,
    )
    from torchlens.backends.registry import registered_backend_specs

    specs = registered_backend_specs()
    assert specs, "no registered backends"
    all_ids: list[str] = []
    for spec in specs:
        rows = operation_grain_capability_rows(spec)
        assert [row.option_name for row in rows] == list(TRACE_OPTION_CAPABILITY_GATES)
        for row in rows:
            assert row.row_id == f"trace_option:{row.option_name}:{spec.name}"
            assert row.backend == spec.name
            all_ids.append(row.row_id)
    assert len(all_ids) == len(set(all_ids)), "capability row ids collide"
    print(f"\noperation-grain capability rows: {len(all_ids)} ({len(specs)} backends x 10 options)")


def test_rows_agree_with_the_coarse_authority() -> None:
    """Projected rows restate the coarse flags EXACTLY (today's one authority).

    When the flag split lands, this test is REWRITTEN alongside the authority
    (the split is allowed to refine values; it is not allowed to happen
    silently -- this equality pin is what makes the change visible).
    """

    from torchlens.backends._options import (
        TRACE_OPTION_CAPABILITY_GATES,
        operation_grain_capability_rows,
    )
    from torchlens.backends.registry import registered_backend_specs

    for spec in registered_backend_specs():
        for row in operation_grain_capability_rows(spec):
            assert row.coarse_flag == TRACE_OPTION_CAPABILITY_GATES[row.option_name]
            assert row.supported == bool(getattr(spec.capabilities, row.coarse_flag)), (
                f"{row.row_id}: operation-grain value diverged from the coarse "
                "authority without an authority change"
            )


@pytest.mark.smoke
def test_torch_backend_supports_the_gated_options() -> None:
    """The stable torch backend's rows read supported on the core gates."""

    from torchlens.backends._options import operation_grain_capability_rows
    from torchlens.backends.registry import registered_backend_specs

    torch_spec = next(spec for spec in registered_backend_specs() if spec.name == "torch")
    supported = {
        row.option_name for row in operation_grain_capability_rows(torch_spec) if row.supported
    }
    for option_name in ("intervene", "halt", "save_grads", "backward_ready"):
        assert option_name in supported, f"torch lost {option_name} support in the projection"
