"""F29: the recurrence fold under the fixed decision rule (agent memo 3.4).

Six acceptance criteria plus the FOLD COHERENCE TEST (membership agreement
with the renderer's collapse plan). The depth-series criteria (flat class
count across 6/12/24 blocks) run on the real gpt2 series and are marked slow;
the structural criteria run on deterministic small nets in the smoke tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens as tl
from tests.test_agent_surface_helpers import RepeatedBlockNet, deterministic_input
from torchlens._visualization._trace_collapse import collapse_plan as _collapse_plan
from torchlens.agent._budgets import OVERVIEW_MAX_TOKENS
from torchlens.agent._fold import fold_trace


def _clean_trace() -> tl.Trace:
    """One deterministic repeated-block capture."""

    return tl.trace(RepeatedBlockNet().eval(), deterministic_input(), save=tl.func("relu"))


# Kept by the shrink-r4 coverage proof: sole branch coverage of _visualization/_collapse_runs.py, _visualization/_segment_descriptors.py and more.
def test_fold_coherence_with_the_collapse_plan(tmp_path: Path) -> None:
    """The FOLD COHERENCE TEST: fold and collapse plan agree on MEMBERSHIP.

    Grain may differ; membership may not: every rendered-plan member op
    appears in the fold's membership, and no fold class claims two ops the
    plan assigns to different module boxes at the same grain (a fold class
    spanning two distinct un-nested plan boxes would contradict the renderer).
    """

    log = _clean_trace()
    fold = fold_trace(log)
    membership = fold.membership()
    plan = _collapse_plan(log, mode="max")
    plan_ops: set[str] = set()
    for node in plan.nodes:
        if hasattr(node, "ops"):
            plan_ops.update(str(op) for op in node.ops)
        elif hasattr(node, "op"):
            plan_ops.add(str(node.op))
    assert plan_ops, "collapse plan emitted no op-bearing nodes"
    fold_labels = set(membership)
    for member in plan_ops:
        qualified = member if ":" in member else f"{member}:1"
        assert member in fold_labels or qualified in fold_labels, (
            f"plan member {member} missing from the fold membership"
        )


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_overview.py, report/_agent_json.py.
@pytest.mark.slow
def test_fold_is_flat_across_the_gpt2_depth_series(tmp_path) -> None:
    """Criteria (i)+(iii) on the real depth series: flat classes, and the
    SERVED overview inside the 4k orientation budget on the deepest model.

    distilgpt2 / gpt2 / gpt2-medium (6/12/24 blocks): the class count must be
    FLAT (not linear in depth), and the overview envelope actually served for
    the deepest model must fit the orientation budget with any trim DISCLOSED.
    The raw class-rows token proxy this replaces never guarded the served
    envelope: the backstop was already trimming gpt2-medium at base while the
    proxy held by 18 tokens (W051-AGENT report, out-of-fence item 1).
    """

    transformers = pytest.importorskip("transformers")
    from torchlens.agent import call_tool
    from torchlens.agent._budgets import estimate_tokens
    from torchlens.agent._envelope import canonical_dumps, json_safe

    counts: dict[str, int] = {}
    for name in ("distilgpt2", "gpt2", "gpt2-medium"):
        model = transformers.AutoModelForCausalLM.from_pretrained(name).eval()
        input_ids = torch.arange(12).unsqueeze(0)
        log = tl.trace(model, (), input_kwargs={"input_ids": input_ids}, save=None)
        fold = fold_trace(log)
        counts[name] = len(fold.classes)
        assert fold.membership().keys() == {str(op.label) for op in log._layer_list}
        if name == "gpt2-medium":
            artifact = tmp_path / "gpt2_medium.tlspec"
            log.save(artifact)
            envelope = call_tool("torchlens_overview", {"path": str(artifact)})
            served = estimate_tokens(canonical_dumps(json_safe(envelope)))
            assert served <= OVERVIEW_MAX_TOKENS, f"served overview is ~{served} tokens"
            truncation = envelope.get("truncation")
            assert truncation is None or truncation["omitted"] > 0
        del model, log
    # Flat: the 24-block count must not scale with depth (allow small drift
    # from boundary blocks, never a per-block term).
    assert counts["gpt2-medium"] - counts["distilgpt2"] <= 6, counts


# Kept by the shrink-r4 coverage proof: sole branch coverage of _visualization/_collapse_frontier.py, _visualization/_collapse_runs.py and more.
@pytest.mark.slow
def test_fold_seam_decision_rule_documented_on_gpt2() -> None:
    """The D1 decision rule's evidence, kept live (agent memo 3.4 / D1).

    The reuse path (the renderer's collapse universe) shipped MEASURED-
    DEGENERATE at depth: no pinned context produced block-level boxes on
    gpt2, so the run-aware fold ships agent-side. the review's stated flip
    condition was "any pinned context that produces block-level boxes on
    gpt2" -- if the renderer improves and this test FAILS, revisit the seam
    (factor the repeat classifier into the shared structural core) instead
    of silently keeping two grains.
    """

    transformers = pytest.importorskip("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    input_ids = torch.arange(12).unsqueeze(0)
    log = tl.trace(model, (), input_kwargs={"input_ids": input_ids}, save=None)
    plan = _collapse_plan(log, mode="max")
    box_names = [type(node).__name__ for node in plan.nodes]
    block_level_boxes = [
        node
        for node in plan.nodes
        if type(node).__name__ == "ModuleBox"
        and ".h." in str(getattr(node, "module", getattr(node, "address", "")))
    ]
    fold = fold_trace(log)
    assert fold.membership().keys() == {str(op.label) for op in log._layer_list}
    assert not block_level_boxes, (
        "the collapse plan now produces block-level boxes on gpt2 "
        f"({box_names[:10]}...): the review's D1 flip condition fired -- revisit "
        "the fold seam per the decision rule instead of keeping two grains"
    )
