"""Typed refusal for unhashable fact path/position COMPONENTS (r77 L1).

Round-76 (hon1 + free-roam, cross-lab) found the r75 L1 lane one nesting level shy:
``_fact_path_tuple`` validated only the CONTAINER type, so a foreign artifact whose
``path``/``position`` was a well-formed sequence carrying a nested list / mapping / slice
COMPONENT (all unhashable decoded literals) passed ``tuple(raw)`` and crashed the first
downstream ``set.add`` / ``dict`` lookup / ``in set`` membership check with a ``TypeError``
into the generic parse catch-all -- ``run_capability_unavailable`` with a stringified
traceback instead of the closed-vocabulary ``context_field_invalid`` analysis-only lane.
The inline ``position`` coercions shared the blind spot.

r77 requires every path/position COMPONENT to be a ``str``/``int`` scalar (the r67
structure-node-path precedent), routes the three former inline position coercions through
one validated helper, and extends the execution-side ``_fact_path_tuple_or_none`` belt the
same nesting level down. Fail-closed either way -- analysis-only load, typed ``.run()``
refusal, no verdict risk -- but always in the TYPED lane. Well-formed artifacts are
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, ReadinessStatus

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)

# Decoded-literal encodings whose DECODED component is unhashable: a sequence whose single
# component decodes to a nested list, an empty mapping, and a slice object respectively.
_NESTED_LIST_COMPONENT = {"kind": "list", "items": [{"kind": "list", "items": []}]}
_MAPPING_COMPONENT = {"kind": "list", "items": [{"entries": []}]}
_SLICE_COMPONENT = {
    "kind": "list",
    "items": [
        {
            "start": {"kind": "int", "value": 0},
            "stop": {"kind": "int", "value": 1},
            "step": {"kind": "int", "value": 1},
        }
    ],
}


class _MetaBranch(nn.Module):
    """Carries model_input_metadata + input_structure facts (leaf contiguity read)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the input leaf's contiguity."""

        if x.is_contiguous():
            return x * 2.0
        return x + 100.0


class _LiteralCarrier(nn.Module):
    """Carries model_input_literal facts (a non-tensor input leaf)."""

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Scale by the literal input."""

        return x * float(k)


def _save(model: nn.Module, inputs: Any, path: Path) -> Path:
    """Capture and save a runnable artifact with embedded weights."""

    trace = tl.trace(model, inputs, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return path


def _tamper_fact_field(
    path: Path, site_prefix: str, field_name: str, encoded_value: dict[str, Any]
) -> None:
    """Re-encode one decoded-fact field inside every matching control witness."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tampered = 0
    for witness in manifest["run"].get("control_witnesses", []):
        if not str(witness.get("site_label", "")).startswith(site_prefix):
            continue
        for entry in witness["observed_value"]["entries"]:
            if entry["key"].get("value") == field_name:
                entry["value"] = encoded_value
                tampered += 1
    assert tampered, f"no {site_prefix!r} witness carried field {field_name!r}"
    manifest_path.write_text(json.dumps(manifest))


def _assert_context_field_invalid_analysis_only(path: Path, run_inputs: Any) -> None:
    """Assert the typed analysis-only disposition: load OK, typed readiness, typed run."""

    loaded = tl.load(path)
    readiness = loaded.__dict__.get("_runnable_readiness")
    assert readiness is not None
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    codes = {diagnostic.code.value for diagnostic in readiness.diagnostics}
    assert "context_field_invalid" in codes, codes
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run(inputs=run_inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "encoded_value",
    [_NESTED_LIST_COMPONENT, _MAPPING_COMPONENT, _SLICE_COMPONENT],
    ids=["nested_list", "mapping", "slice"],
)
def test_r77_unhashable_metadata_path_component_refuses_typed(
    tmp_path: Path, encoded_value: dict[str, Any]
) -> None:
    """RED-now-fixed: an unhashable metadata-envelope path COMPONENT lands typed.

    Pre-fix the decoded ``[[]]`` / ``[{}]`` / ``[slice(...)]`` path passed the container
    check and crashed the downstream ``dict`` consumer into ``run_capability_unavailable``.
    """

    x = torch.randn(4, 4)
    path = _save(_MetaBranch().eval(), x, tmp_path / "meta_path.tlspec")
    _tamper_fact_field(path, "model_input_metadata:", "path", encoded_value)
    _assert_context_field_invalid_analysis_only(path, x.clone())


@pytest.mark.smoke
def test_r77_unhashable_metadata_position_component_refuses_typed(tmp_path: Path) -> None:
    """An unhashable metadata-fact POSITION component refuses typed (was ``in set`` crash)."""

    x = torch.randn(4, 4)
    path = _save(_MetaBranch().eval(), x, tmp_path / "meta_pos.tlspec")
    _tamper_fact_field(path, "model_input_metadata:", "position", _NESTED_LIST_COMPONENT)
    _assert_context_field_invalid_analysis_only(path, x.clone())


@pytest.mark.smoke
def test_r77_unhashable_literal_path_component_refuses_typed(tmp_path: Path) -> None:
    """The sibling literal-fact path lane refuses an unhashable component typed."""

    x = torch.randn(4, 4)
    path = _save(_LiteralCarrier().eval(), (x, 3), tmp_path / "lit_path.tlspec")
    _tamper_fact_field(path, "model_input_literal:", "path", _MAPPING_COMPONENT)
    _assert_context_field_invalid_analysis_only(path, (x.clone(), 3))


@pytest.mark.smoke
def test_r77_unhashable_literal_position_component_refuses_typed(tmp_path: Path) -> None:
    """The literal-fact POSITION coercion refuses an unhashable component typed."""

    x = torch.randn(4, 4)
    path = _save(_LiteralCarrier().eval(), (x, 3), tmp_path / "lit_pos.tlspec")
    _tamper_fact_field(path, "model_input_literal:", "position", _SLICE_COMPONENT)
    _assert_context_field_invalid_analysis_only(path, (x.clone(), 3))


@pytest.mark.smoke
def test_r77_unhashable_structure_position_component_refuses_typed(tmp_path: Path) -> None:
    """A structure-fact POSITION with an unhashable component refuses typed.

    The structure fact position feeds a ``dict`` key (``structure_expected``) after the
    shared path validator; the component check closes the same crash there.
    """

    x = torch.randn(4, 4)
    path = _save(_MetaBranch().eval(), x, tmp_path / "struct_pos.tlspec")
    _tamper_fact_field(path, "input_structure:", "position", _NESTED_LIST_COMPONENT)
    _assert_context_field_invalid_analysis_only(path, x.clone())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory",
    [
        lambda: (_MetaBranch().eval(), torch.randn(4, 4)),
        lambda: (_LiteralCarrier().eval(), (torch.randn(4, 4), 3)),
    ],
    ids=["metadata", "literal"],
)
def test_r77_untampered_artifacts_unchanged(
    tmp_path: Path, factory: Callable[[], tuple[nn.Module, Any]]
) -> None:
    """Zero collateral: well-formed artifacts still load and verify."""

    model, inputs = factory()
    path = _save(model, inputs, tmp_path / "clean.tlspec")
    run_inputs = (
        tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in inputs)
        if isinstance(inputs, tuple)
        else inputs.clone()
    )
    result = tl.load(path).run(inputs=run_inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
