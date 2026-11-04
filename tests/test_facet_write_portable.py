"""A facet-selector head ablation survives the default portable save.

``head_ablation_candidates`` returns ``tl.head(i, "v").in_module(m)`` selectors whose
edits write one facet slice through a scatter wrapper bound to the captured trace. The
recipe persists the request instead of the wrapper: the builtin edit, the narrowed
facet selector as the rule's site, the resolved home label and facet name, and the
write mask (tensor sidecar) with its sha256. Loading a trace re-resolves the selector
against the loaded trace and compares label, facet and mask digest; a write that no
longer reproduces refuses typed. A recipe carrying a facet rule is written at a newer
recipe format version, so a reader that predates it refuses typed instead of
installing the raw edit on the whole home tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import redigest
from torch import nn

import torchlens as tl
from torchlens._io.format3 import recipe as recipe_module
from torchlens.experiment import head_ablation_candidates

_MODULE = "transformer.h.1.attn"


def _gpt2() -> nn.Module:
    pytest.importorskip("transformers")
    sys.path.insert(0, ".")
    from tests.real_model.r0.families import build_gpt2

    return build_gpt2("eager").eval()


def _ids() -> torch.Tensor:
    return torch.randint(0, 100, (1, 8), generator=torch.Generator().manual_seed(0))


def _output(trace: Any) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _session(model: nn.Module, ids: torch.Tensor) -> tuple[Any, Any]:
    """Return (clean baseline, fork with head 0's value facet zero-ablated)."""

    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
    candidates = head_ablation_candidates(baseline, _MODULE, heads=model.config.n_head)
    session = baseline.fork()
    session.do(candidates["h0"], tl.zero_ablate())
    return baseline, session


def _coded(error: BaseException) -> Any:
    current: BaseException | None = error
    while current is not None:
        if getattr(current, "fields", {}).get("code"):
            return current
        current = current.__cause__ or current.__context__
    raise AssertionError(f"no coded error in the chain of {error!r}")


def _facet_rules(path: Path) -> list[dict[str, Any]]:
    data = json.loads((path / "intervention.json").read_text(encoding="utf-8"))
    return [rule for rule in data["rules"] if rule["metadata"].get("facet_write")]


def test_head_ablation_saves_by_default_and_reapplies_bit_identically(tmp_path: Path) -> None:
    """Default save, load, re-apply on the model: outputs are bit-identical."""

    model, ids = _gpt2(), _ids()
    baseline, session = _session(model, ids)
    assert not torch.equal(_output(session), _output(baseline))
    path = tmp_path / "head_ablation.tlspec"
    session.save(path)
    (rule,) = _facet_rules(path)
    assert rule["edit"]["name"] == "zero_ablate"
    assert rule["edit"]["portability"] == "builtin"
    assert len(rule["metadata"]["facet_write_mask_sha256"]) == 64
    loaded = tl.load(path)
    assert torch.equal(_output(loaded), _output(session))
    (stored,) = loaded._intervention_spec.hook_specs
    assert getattr(stored.hook, "_tl_facet_scatter", False)
    rerun = loaded.fork()
    rerun.run(model, ids)
    assert torch.equal(_output(rerun), _output(session))


def test_a_facet_write_that_no_longer_reproduces_refuses_typed(tmp_path: Path) -> None:
    """The stored selector re-resolves to another head: the mask digest differs."""

    model, ids = _gpt2(), _ids()
    _, session = _session(model, ids)
    path = tmp_path / "moved_head.tlspec"
    session.save(path)
    recipe_path = path / "intervention.json"
    data = json.loads(recipe_path.read_text(encoding="utf-8"))
    for rule in data["rules"]:
        if rule["metadata"].get("facet_write"):
            rule["selector"]["selector_value"]["head_index"] = 1
    recipe_path.write_text(json.dumps(data), encoding="utf-8")
    redigest(path, "intervention.json")
    with pytest.raises(Exception) as caught:
        tl.load(path)
    assert _coded(caught.value).fields["code"] == "bundle_load_failed"


def test_a_reader_that_predates_facet_rules_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipe carrying a facet rule is written at the newer recipe format version."""

    model, ids = _gpt2(), _ids()
    _, session = _session(model, ids)
    path = tmp_path / "newer.tlspec"
    session.save(path)
    data = json.loads((path / "intervention.json").read_text(encoding="utf-8"))
    assert data["format_version"] > 3
    monkeypatch.setattr(recipe_module, "READABLE_RECIPE_FORMAT_VERSIONS", (3,))
    with pytest.raises(Exception) as caught:
        tl.load(path)
    assert _coded(caught.value).fields["code"] == "bundle_load_failed"
