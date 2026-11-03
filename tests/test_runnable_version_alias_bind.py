"""Bind checks a direct version alias's recorded shape and dtype against the bound value.

Bind writes a produced value into each direct version-alias slot of it (``version_of``
and ``producer_slot_id`` both naming the producer). Before the check, a crafted file
could record such an alias at any shape or dtype and the run bound the producer's
value into it unchecked; the producer slot itself was always checked. These tests
redirect one reader to a crafted alias of the producer's slot and require a typed
divergence naming the alias when its recorded shape or dtype lies, and a verified
run when the alias is honest.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_tlspec_runnable_alloc_bomb import _tamper
from torchlens._runnable import RunnableErrorCode
from torchlens.errors import PathDivergenceError

_CAPTURE = tl.options.CaptureOptions(replayable=True)


class _ReluThenMean(nn.Module):
    """``mean`` reads the ``relu`` output; the tamper reroutes that read through an alias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (2, 4)
        return x.relu().mean() + x.sum()


def _build(tmp_path: Path) -> Path:
    trace = tl.trace(_ReluThenMean().eval(), torch.randn(2, 4), capture=_CAPTURE)
    bundle = tmp_path / "alias.tlspec"
    trace.save(str(bundle), level="runnable", include_weights=True)
    return bundle


def _alias_reader(**recorded: Any) -> Callable[[dict[str, Any]], None]:
    """Return a tamper that reroutes ``mean`` through a crafted direct alias of ``relu``.

    ``recorded`` overrides the alias slot's recorded fields (shape, rank, dtype).
    """

    def mutate(manifest: dict[str, Any]) -> None:
        run = manifest["run"]
        calls = {call["op_labels"][0].split("_")[0]: call for call in run["calls"]}
        producer_id = calls["relu"]["output_slot_ids"][0]
        reader = calls["mean"]
        slots = {slot["slot_id"]: slot for slot in run["tensor_slots"]}
        alias = copy.deepcopy(slots[producer_id])
        alias.update(
            slot_id=f"{producer_id}:alias",
            version_of=producer_id,
            producer_slot_id=producer_id,
            use_sites=[],
            **recorded,
        )
        for argument in reader["tensor_arguments"]:
            if argument["slot_id"] == producer_id:
                argument["slot_id"] = alias["slot_id"]
                alias["use_sites"].append(
                    {"call_id": reader["call_id"], "argument_path": argument["argument_path"]}
                )
        assert alias["use_sites"], "the reader does not read the producer slot"
        slots[producer_id]["use_sites"] = [
            site for site in slots[producer_id]["use_sites"] if site["call_id"] != reader["call_id"]
        ]
        run["tensor_slots"].append(alias)

    return mutate


@pytest.mark.parametrize(
    ("recorded", "code"),
    [
        ({"shape": [1], "rank": 1}, RunnableErrorCode.OUTPUT_SHAPE_MISMATCH),
        ({"shape": [2, 8]}, RunnableErrorCode.OUTPUT_SHAPE_MISMATCH),
        ({"dtype": "torch.float64"}, RunnableErrorCode.OUTPUT_DTYPE_MISMATCH),
    ],
    ids=["smaller-shape", "larger-shape", "dtype"],
)
def test_direct_alias_with_wrong_recorded_fields_diverges_at_bind(
    tmp_path: Path, recorded: dict[str, Any], code: RunnableErrorCode
) -> None:
    """An alias whose recorded shape or dtype disagrees with the bound value diverges typed."""

    bundle = _build(tmp_path)
    _tamper(bundle, _alias_reader(**recorded))
    with pytest.raises(PathDivergenceError) as caught:
        tl.load(str(bundle)).run(inputs=torch.randn(2, 4))
    assert caught.value.fields["code"] == code.value
    assert ":alias" in str(caught.value)


def test_honest_direct_alias_runs_verified(tmp_path: Path) -> None:
    """The same reroute through an alias recorded at the producer's shape and dtype runs."""

    bundle = _build(tmp_path)
    _tamper(bundle, _alias_reader())
    x = torch.randn(2, 4)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
    torch.testing.assert_close(result.output, _ReluThenMean()(x))
