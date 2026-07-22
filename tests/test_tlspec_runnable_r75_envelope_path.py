"""Typed refusal for malformed envelope/literal fact paths (r75 L1).

Round-74 (free-roam) found lane hygiene drift: a foreign artifact whose metadata-envelope
``path`` was literal-encoded as an int crashed ``tuple(fact.get("path", ()) or ())`` with a
``TypeError`` into the generic parse catch-all (``run_capability_unavailable`` with a
stringified traceback) instead of the explicit ``context_field_invalid`` closed-vocabulary
analysis-only lane its sibling ``position`` checks use. The same unguarded coercion existed
for literal-fact and structure-fact paths at parse and (shielded, because parse crashes
first) in three execution-side readers.

r75 routes every decoded fact path/position through one parse-side validator that refuses
non-sequences (including strings, which would silently shred into per-character components)
as ``context_field_invalid``, plus a fail-closed execution-side belt so the readers stay
total. Nothing untyped escapes; a well-formed artifact is untouched.
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


class _MetaBranch(nn.Module):
    """Carries a model_input_metadata envelope fact (leaf contiguity read)."""

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
    [
        {"kind": "int", "value": 7},
        {"kind": "str", "value": "arg0"},
    ],
    ids=["int_encoded", "str_encoded"],
)
def test_r75_malformed_metadata_envelope_path_refuses_typed(
    tmp_path: Path, encoded_value: dict[str, Any]
) -> None:
    """An int/str-encoded metadata envelope path lands in context_field_invalid, not the catch-all."""

    x = torch.randn(4, 4)
    path = _save(_MetaBranch().eval(), x, tmp_path / "meta.tlspec")
    _tamper_fact_field(path, "model_input_metadata:", "path", encoded_value)
    _assert_context_field_invalid_analysis_only(path, x.clone())


@pytest.mark.smoke
def test_r75_malformed_literal_fact_path_refuses_typed(tmp_path: Path) -> None:
    """The sibling literal-fact path lane refuses through the same typed disposition."""

    x = torch.randn(4, 4)
    path = _save(_LiteralCarrier().eval(), (x, 3), tmp_path / "literal.tlspec")
    _tamper_fact_field(path, "model_input_literal:", "path", {"kind": "int", "value": 7})
    _assert_context_field_invalid_analysis_only(path, (x.clone(), 3))


@pytest.mark.smoke
def test_r75_malformed_structure_fact_position_refuses_typed(tmp_path: Path) -> None:
    """A non-sequence structure-fact position refuses typed too (same unguarded class)."""

    x = torch.randn(4, 4)
    path = _save(_MetaBranch().eval(), x, tmp_path / "structure.tlspec")
    _tamper_fact_field(path, "input_structure:", "position", {"kind": "int", "value": 7})
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
def test_r75_untampered_artifacts_unchanged(
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
