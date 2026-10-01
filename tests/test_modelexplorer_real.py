"""Real-model Model Explorer regressions (memo section 5; heavy tier).

The id bug survived THREE rounds of toy validation -- capture-side
recurrence folding rescued every toy -- so the id-uniqueness regression, the
158/158 edge-resolution case, model-mode receipt discipline, and the worker
floors are pinned on real torchvision/config-built-transformers structures.
Floors live in test source; absolute counts are asserted only where the
panel measured them on THIS structure (torchvision resnet18 eval, weights
irrelevant to structure).
"""

from __future__ import annotations

import collections
from typing import Any

import pytest
import torch
from test_modelexplorer_assets import harness

import torchlens as tl

torchvision = pytest.importorskip("torchvision")

pytestmark = pytest.mark.heavy

requires_node = pytest.mark.skipif(
    not harness.node_available(), reason="worker oracle needs a Node runtime on PATH"
)


def _export(model: Any, inputs: torch.Tensor) -> dict[str, Any]:
    """Trace and export one model."""

    log = tl.trace(model, inputs)
    return tl.export.to_model_explorer_dict(log)


def _bare_site_key_collisions(model: Any, inputs: torch.Tensor) -> int:
    """Count nodes a bare-site-key id scheme would SILENTLY DROP (memo D2).

    Model Explorer keeps the first node per duplicate id and drops the rest,
    so the regression figure is ``sum(count - 1)`` over duplicated keys: 8 on
    ResNet-18 (relu reused twice per basic block), 32 on ResNet-50 (relu
    reused three times per bottleneck).
    """

    log = tl.trace(model, inputs)
    counts = collections.Counter(getattr(entry, "site_key", None) for entry in log.layer_list)
    return sum(count - 1 for key, count in counts.items() if key and count > 1)


def _namespace_depth_fraction(graph: dict[str, Any], depth: int) -> float:
    """Fraction of op nodes at module depth >= ``depth``."""

    nodes = graph["nodes"]
    deep = sum(
        1 for node in nodes if node["namespace"] and len(node["namespace"].split("/")) >= depth
    )
    return deep / len(nodes)


def _largest_layer_children(graph: dict[str, Any]) -> int:
    """Return the largest direct-children count across all namespaces."""

    children: dict[str, set[str]] = collections.defaultdict(set)
    for node in graph["nodes"]:
        namespace = node["namespace"]
        children[namespace].add(node["id"])
        components = namespace.split("/") if namespace else []
        for depth in range(1, len(components)):
            children["/".join(components[:depth])].add(components[depth])
    children[""].update(namespace.split("/")[0] for namespace in children if namespace)
    return max(len(members) for members in children.values())


@pytest.fixture(scope="module")
def resnet18_eval_payload() -> Any:
    """Export torchvision resnet18 in eval mode once per module."""

    model = torchvision.models.resnet18(weights=None).eval()
    log = tl.trace(model, torch.randn(1, 3, 64, 64))
    try:
        yield tl.export.to_model_explorer_dict(log)
    finally:
        log.cleanup()


def test_resnet18_reuse_duplicates_bare_site_keys() -> None:
    """The regression's precondition holds: 8 droppable reused-relu nodes."""

    model = torchvision.models.resnet18(weights=None).eval()
    assert _bare_site_key_collisions(model, torch.randn(1, 3, 64, 64)) == 8


def test_resnet18_eval_ids_unique_and_edges_resolve(
    resnet18_eval_payload: dict[str, Any],
) -> None:
    """THE id-uniqueness regression + the 158/158 edge-resolution case."""

    graph = resnet18_eval_payload["graphs"][0]
    ids = [node["id"] for node in graph["nodes"]]
    assert len(ids) == len(set(ids)) == 151
    assert sum(len(node.get("incomingEdges", [])) for node in graph["nodes"]) == 158
    report = tl.export.validate_model_explorer_payload(resnet18_eval_payload)
    assert report.ok, report.failures
    # Exactly ONE root-level non-boundary op: torch.flatten in
    # ResNet.forward -- a real root-module op, disclosed, never an anomaly
    # bucket (memo D19's counter stays a re-open tripwire).
    assert report.counters["stackless_root_ops"] == 1


def test_resnet18_train_mode_changes_receipts_not_integrity() -> None:
    """Train mode swings op counts (~40%); integrity floors hold anyway."""

    model = torchvision.models.resnet18(weights=None).train()
    payload = _export(model, torch.randn(1, 3, 64, 64))
    graph = payload["graphs"][0]
    ids = [node["id"] for node in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert len(ids) != 151, "train/eval mode must be a visible receipt axis"
    report = tl.export.validate_model_explorer_payload(payload)
    assert report.ok, report.failures


def test_resnet18_hierarchy_floors(resnet18_eval_payload: dict[str, Any]) -> None:
    """>=75% of ops at module depth >= 3; largest layer < 100 children."""

    graph = resnet18_eval_payload["graphs"][0]
    assert _namespace_depth_fraction(graph, 3) >= 0.75
    assert _largest_layer_children(graph) < 100


@requires_node
def test_resnet18_worker_floors(resnet18_eval_payload: dict[str, Any]) -> None:
    """Executed oracle: no silent loss, twins live, hierarchy renders."""

    rows = harness.run_worker_oracle(resnet18_eval_payload)
    harness.assert_no_silent_node_loss(resnet18_eval_payload, rows)
    stats = rows[0]["stats"]
    assert stats["identicalGroupCount"] > 0, "twin detection must engage (memo D3)"
    assert stats["identicalGroupNodes"] >= 16
    assert stats["maxNamespaceDepth"] >= 3
    assert stats["rootChildren"] <= 20, "collapsed first paint stays readable"


def test_resnet50_reuse_at_scale() -> None:
    """ResNet-50: 32 duplicate bare sites; ids stay unique; edges resolve."""

    model = torchvision.models.resnet50(weights=None).eval()
    inputs = torch.randn(1, 3, 64, 64)
    assert _bare_site_key_collisions(model, inputs) == 32
    payload = _export(model, inputs)
    graph = payload["graphs"][0]
    ids = [node["id"] for node in graph["nodes"]]
    assert len(ids) == len(set(ids))
    report = tl.export.validate_model_explorer_payload(payload)
    assert report.ok, report.failures


@requires_node
def test_config_built_gpt2_hierarchy_and_boundary_groups() -> None:
    """Config-built GPT-2: dot-split ModuleList levels, grouped outputs,
    block twins -- at zero network."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=12,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        n_positions=64,
        use_cache=True,
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    payload = _export(model, torch.randint(0, 128, (1, 8)))
    graph = payload["graphs"][0]
    namespaces = {node["namespace"] for node in graph["nodes"]}
    assert any(ns.startswith("transformer/h/0/attn") for ns in namespaces), (
        "the ModuleList container must reappear as a dot-split level"
    )
    rows = graph["groupNodeAttributes"]
    assert "transformer/h" in rows and "transformer/h/0" in rows
    output_nodes = [node for node in graph["nodes"] if node["namespace"] == "Outputs"]
    assert len(output_nodes) == 25, "24 KV-cache tensors + logits"
    report = tl.export.validate_model_explorer_payload(payload)
    assert report.ok, report.failures
    oracle_rows = harness.run_worker_oracle(payload)
    harness.assert_no_silent_node_loss(payload, oracle_rows)
    stats = oracle_rows[0]["stats"]
    assert stats["identicalGroupCount"] > 0, "twelve-block twins must engage"
    assert stats["rootChildren"] <= 10, "output markers group into one box"
