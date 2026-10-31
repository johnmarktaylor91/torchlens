"""Explorer sign-off 5, built values: the D21 embedding cap and the D22 byte budget.

D21: giant embedding-class parameters reduce spine-only unless
``WatchSettings(sketch_embeddings=True)`` asks for their sketch, and the plan
table says so. D22: the disk history artifact has a byte budget (256 MiB by
default); the collector refuses typed BEFORE the block that would cross it,
and the artifact stays readable to the last committed block.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens.trackers as trk
from torchlens._observability import HistoryArtifactError, HistoryCollector, HistoryReader
from torchlens._observability._collector import (
    DEFAULT_ARTIFACT_BUDGET_BYTES,
    EMBEDDING_SKETCH_CAP_NUMEL,
    WatchSettings,
)


def _tiny_gpt2() -> nn.Module:
    """A real GPT-2 class with an embedding table at the D21 cap (no download)."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        vocab_size=EMBEDDING_SKETCH_CAP_NUMEL // 64,
        n_embd=64,
        n_layer=1,
        n_head=2,
        n_positions=16,
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config)


def _train_two_steps(settings: WatchSettings | None) -> tuple[trk.MemorySink, trk.WatchSession]:
    model = _tiny_gpt2()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    sink = trk.MemorySink()
    session = trk.watch(
        model,
        to=sink,
        signals=("gradients",),
        optimizer=optimizer,
        every=1,
        hist_every=1,
        settings=settings,
    )
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    for step in range(2):
        with session.step(step):
            model(ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    session.close()
    return sink, session


def test_embedding_table_is_spine_only_by_default() -> None:
    sink, session = _train_two_steps(WatchSettings())
    wte = "transformer.wte.weight"
    hist_tags = {h.tag for h in sink.histograms}
    assert not any(wte in tag for tag in hist_tags), "the embedding table must not sketch"
    assert any("transformer.h.0" in tag for tag in hist_tags), "other params still sketch"
    assert any(p.tag == f"gradients/norm/{wte}" for p in sink.scalars), "the spine still lands"
    assert "embedding-class: spine only" in session.plan_report()


def test_sketch_embeddings_opts_the_table_back_in() -> None:
    sink, session = _train_two_steps(WatchSettings(sketch_embeddings=True))
    assert any("transformer.wte.weight" in h.tag for h in sink.histograms)
    assert "embedding-class" not in session.plan_report()


def _linear_collector(path: Path, budget: int | None) -> tuple[HistoryCollector, nn.Module]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    collector = HistoryCollector(
        model,
        streams=("activation",),
        settings=WatchSettings(output_dir=str(path), artifact_budget_bytes=budget),
    )
    collector.discover(torch.randn(4, 8))
    collector.attach()
    return collector, model


def test_budget_refuses_before_the_crossing_block_and_artifact_stays_readable(
    tmp_path: Path,
) -> None:
    probe, _ = _linear_collector(tmp_path / "probe", None)
    bound = probe._block_bytes_bound
    probe.detach()
    assert bound > 0
    collector, model = _linear_collector(tmp_path / "run", 3 * bound)
    for step in range(3):
        with collector.step(step):
            model(torch.randn(4, 8))
    with pytest.raises(HistoryArtifactError) as info, collector.step(3):
        model(torch.randn(4, 8))
    assert info.value.fields["code"] == "history_artifact_budget_exceeded"
    assert "artifact_budget_bytes" in str(info.value.fields["remedy"])
    collector.detach()
    reader = HistoryReader(tmp_path / "run")
    assert [block.global_step for block in reader.step_blocks()] == [0, 1, 2]
    assert len(list(reader.observations())) == 3 * len(collector.site_catalog)


def test_default_budget_is_256_mib() -> None:
    assert WatchSettings().artifact_budget_bytes == DEFAULT_ARTIFACT_BUDGET_BYTES == 256 << 20
