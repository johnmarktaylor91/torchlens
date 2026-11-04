"""The loaded-rerun guard reads each persisted audit row by its kind.

A ``do()`` writes its canonical row as an ACT row, which carries no ``status``:
its ``sites`` list the site keys the edit fired on. A do that only staged its
hooks (``engine="set_only"``) therefore lists none and is value-neutral, while
the EVENT row family says what fired through ``status``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.intervention._intervened_rerun_guard import (
    _carries_intervened_values,
    refuse_unpersisted_intervention_rerun,
)
from torchlens.intervention.errors import EngineDispatchError


def _loaded_trace(*rows: dict[str, Any]) -> Any:
    layer = SimpleNamespace(is_edited=False, edits=None)
    spec = SimpleNamespace(records=())
    return SimpleNamespace(
        _layer_list=[layer], _intervention_spec=spec, intervention_audit=list(rows)
    )


def _act_row(sites: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "ACT",
        "edit": "scale",
        "selection_repr": "module('fc2')",
        "resolve_digest": "0" * 16,
        "sites": sites,
    }


@pytest.mark.parametrize(
    ("row", "intervened"),
    [
        (_act_row([]), False),
        (_act_row([{"relation": "exact", "selected": 1, "site_key": "('fc2',)"}]), True),
        ({"kind": "EVENT", "status": "no_fire"}, False),
        ({"kind": "EVENT", "status": "error"}, False),
        ({"kind": "EVENT", "status": "armed"}, False),
        ({"kind": "EVENT", "status": "fired"}, True),
        ({"kind": "EDGE", "selection_repr": "edge", "edit": "value"}, True),
        ({"kind": "PARAM", "selection_repr": "param", "edit": "value"}, True),
        ({"unknown": "shape"}, True),
    ],
)
def test_audit_row_kind_decides_whether_values_were_intervened(
    row: dict[str, Any], intervened: bool
) -> None:
    assert _carries_intervened_values(_loaded_trace(row)) is intervened


def test_staged_only_do_row_lets_the_loaded_plain_trace_rerun() -> None:
    """A do() that only staged hooks on a loaded plain trace does not make it intervened."""

    refuse_unpersisted_intervention_rerun(_loaded_trace(_act_row([])), "loaded_analysis")


def test_fired_act_row_on_a_loaded_trace_still_refuses_typed() -> None:
    trace = _loaded_trace(_act_row([{"relation": "exact", "selected": 1, "site_key": "('fc2',)"}]))
    with pytest.raises(EngineDispatchError) as excinfo:
        refuse_unpersisted_intervention_rerun(trace, "loaded_sparse")
    assert excinfo.value.fields["code"] == "run_intervention_spec_not_persisted"
