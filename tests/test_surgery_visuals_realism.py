"""Surgery visuals real-model gates (lane F43, foldA s6 realism row).

The fold's realism rule (toy-only validation is disqualifying) applied to
the rendering half: the census line and splice box are checked over a REAL
distilgpt2 live-intervened capture AND a real replay fork, PER LANE -- the
live lane's wording is false in the replay lane and vice versa, so each
lane's exported artifacts must carry exactly its own text -- plus a
ResNet-50 block splice.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.visualization.surgery_visuals import surgery_census

HF_HUB_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"

_CAPTURE = tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True)

_REPLAY_WORDING = "exit values were substituted; the interior was not replayed"
_LIVE_WORDING = "the original op ran; edited values replaced its output after execution"
_BANNED_VERBS = ("skipped", "deleted", "removed")


def _distilgpt2():
    """The cached distilgpt2 snapshot (skip when not fetched)."""

    transformers = pytest.importorskip("transformers")
    if not (HF_HUB_CACHE / "models--distilgpt2").exists():
        pytest.skip("distilgpt2 snapshot not cached; fetch once online.")
    return transformers.AutoModelForCausalLM.from_pretrained("distilgpt2").eval()


def _assert_no_banned_verbs(text: str) -> None:
    """No exported surgery wording may claim execution removal."""

    lowered = text.lower()
    for verb in _BANNED_VERBS:
        assert verb not in lowered, verb


def _render_with_sidecar(trace, tmp_path: Path, stem: str) -> tuple[str, str]:
    """Render the surgery lens focused on one block; return (svg, census)."""

    from torchlens.visualization.surgery_visuals import render_surgery

    outpath = str(tmp_path / stem)
    render_surgery(
        trace,
        vis_outpath=outpath,
        vis_fileformat="svg",
        vis_save_only=True,
    )
    svg = (tmp_path / f"{stem}.svg").read_text()
    census = (tmp_path / f"{stem}.census.txt").read_text()
    return svg, census


@pytest.mark.heavy
@pytest.mark.real_model
def test_distilgpt2_live_lane_census_and_splice_box(tmp_path) -> None:
    """GATE: real distilgpt2 LIVE-intervened capture, per-lane checked.

    The exported figure and its travelling census carry the live lane's
    wording and never the replay lane's -- the replay text is false here.
    """

    model = _distilgpt2()
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    log = tl.trace(
        model,
        ids,
        intervene=tl.when(tl.in_module("transformer.h.0") & tl.func("tanh"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    census_text = surgery_census(log).to_text()
    assert "lane capture: 1 transaction(s)" in census_text
    assert _LIVE_WORDING in census_text
    assert _REPLAY_WORDING not in census_text
    svg, sidecar = _render_with_sidecar(log, tmp_path, "distilgpt2_live")
    assert "surgery census" in svg
    assert _REPLAY_WORDING not in sidecar
    assert _LIVE_WORDING in sidecar
    _assert_no_banned_verbs(svg)
    _assert_no_banned_verbs(sidecar)


@pytest.mark.heavy
@pytest.mark.real_model
def test_distilgpt2_replay_fork_census_and_splice_box(tmp_path) -> None:
    """GATE: real distilgpt2 REPLAY fork, per-lane checked.

    The same model under the other lane: the replay wording appears, the
    live wording (false in this lane) never does.
    """

    model = _distilgpt2()
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    log = tl.trace(model, ids, capture=_CAPTURE)
    fork = log.fork()
    tanh_label = next(label for label in fork.op_labels if label.startswith("tanh"))
    fork.do(tl.units(tanh_label, [(0, 0, 0)]).resolve(fork), tl.zero_ablate())
    census_text = surgery_census(fork).to_text()
    assert "lane replay: 1 transaction(s)" in census_text
    assert _REPLAY_WORDING in census_text
    assert _LIVE_WORDING not in census_text
    svg, sidecar = _render_with_sidecar(fork, tmp_path, "distilgpt2_replay")
    assert "surgery census" in svg
    assert _REPLAY_WORDING in sidecar
    assert _LIVE_WORDING not in sidecar
    _assert_no_banned_verbs(svg)
    _assert_no_banned_verbs(sidecar)


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet50_block_splice_box_and_marks(tmp_path) -> None:
    """GATE: a ResNet-50 block splice renders splice boxes + fact marks."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet50(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    log = tl.trace(
        model,
        x,
        intervene=tl.when(
            tl.in_module("layer1.0") & tl.func("relu"),
            tl.splice_module(torch.nn.ReLU(), input="out"),
        ),
        capture=tl.options.CaptureOptions(intervention_ready=True, log_injections=True),
    )
    from torchlens.visualization.surgery_visuals import surgery_facts

    facts = surgery_facts(log)
    splice_rows = [
        mark for mark in facts.marks if mark.kind == "fire" and "splice_module" in mark.row
    ]
    assert len(splice_rows) >= 1
    assert all(mark.basis == "fact" for mark in splice_rows)
    host_marks = [mark for mark in facts.marks if mark.kind == "injection_host"]
    assert host_marks, "the spliced module's own ops must be recorded as injected"
    census_text = surgery_census(log).to_text()
    assert "splice_module" in census_text
    assert _LIVE_WORDING in census_text
    assert _REPLAY_WORDING not in census_text
    svg, sidecar = _render_with_sidecar(log, tmp_path, "resnet50_splice")
    assert "surgery census" in svg
    assert "splice_module" in sidecar
    _assert_no_banned_verbs(sidecar)
