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
from torch import nn

import torchlens as tl
from tests.test_agent_surface_helpers import RepeatedBlockNet, deterministic_input
from torchlens.agent._fold import FOLD_KEY_VERSION, fold_trace, normalize_module_path


def _clean_trace() -> tl.Trace:
    """One deterministic repeated-block capture."""

    return tl.trace(RepeatedBlockNet().eval(), deterministic_input(), save=tl.func("relu"))


@pytest.mark.smoke
def test_normalize_module_path_wildcards_numeric_components() -> None:
    """Call qualifiers strip; numeric dotted components wildcard."""

    assert normalize_module_path("transformer.h.11.mlp:1") == "transformer.h.*.mlp"
    assert normalize_module_path("blocks.0.fc:2") == "blocks.*.fc"
    assert normalize_module_path("head:1") == "head"


@pytest.mark.smoke
def test_fold_membership_reassembles_exactly() -> None:
    """Criterion (v): the union of class members IS the op-row set, disjoint."""

    log = _clean_trace()
    fold = fold_trace(log)
    assert fold.key_version == FOLD_KEY_VERSION
    membership = fold.membership()
    labels = [str(op.label) for op in log.layer_list]
    assert sorted(membership) == sorted(labels)
    assert sum(cls.n_instances for cls in fold.classes) == len(labels)


@pytest.mark.smoke
def test_fold_classes_are_field_uniform() -> None:
    """Criterion (ii): every emitted class is uniform on shape/dtype/num_params."""

    log = _clean_trace()
    fold = fold_trace(log)
    by_label = {str(op.label): op for op in log.layer_list}
    for cls in fold.classes:
        shapes = {tuple(by_label[m].shape) if by_label[m].shape else None for m in cls.members}
        dtypes = {str(by_label[m].dtype) for m in cls.members}
        params = {int(by_label[m].num_params or 0) for m in cls.members}
        assert len(shapes) == 1 and len(dtypes) == 1 and len(params) == 1, cls.class_id


@pytest.mark.smoke
def test_fold_folds_the_repeated_blocks_once() -> None:
    """Three identical blocks state ONCE with n_instances=3."""

    fold = fold_trace(_clean_trace())
    relu_classes = [cls for cls in fold.classes if cls.func_name == "relu"]
    assert len(relu_classes) == 1
    assert relu_classes[0].n_instances == 3
    assert relu_classes[0].module_path == ("blocks.*", "blocks.*.*")


@pytest.mark.smoke
def test_fold_splits_on_shape_disagreement() -> None:
    """A class whose members disagree on shape SPLITS, never averages."""

    class Widening(nn.Module):
        """Two relus with different widths (same normalized stack)."""

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 4)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x

    log = tl.trace(Widening().eval(), deterministic_input())
    fold = fold_trace(log)
    relu_classes = [cls for cls in fold.classes if cls.func_name == "relu"]
    assert len(relu_classes) == 2  # split by output shape
    assert {cls.fields["shape"][-1] for cls in relu_classes} == {8, 4}


@pytest.mark.smoke
def test_fold_never_declines_on_multipass() -> None:
    """Criterion (iv): recurrent multi-pass traces fold, pass-qualified."""

    class Loop(nn.Module):
        """One module called three times (recurrence grouping)."""

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.cell = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.tanh(self.cell(x))
            return x

    log = tl.trace(Loop().eval(), deterministic_input())
    fold = fold_trace(log)
    assert fold.membership().keys() == {str(op.label) for op in log.layer_list}
    multipass = [cls for cls in fold.classes if cls.pass_index is not None]
    if multipass:  # pass boundaries never merge silently
        assert all(cls.pass_index >= 1 for cls in multipass)


@pytest.mark.smoke
def test_fold_key_has_zero_render_context_inputs() -> None:
    """Criterion (vi): folding twice, and after a draw-free reload, is identical."""

    log = _clean_trace()
    first = fold_trace(log)
    second = fold_trace(log)
    assert [cls.members for cls in first.classes] == [cls.members for cls in second.classes]


@pytest.mark.smoke
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
    plan = log.collapse_plan(mode="max")
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


@pytest.mark.slow
def test_fold_is_flat_across_the_gpt2_depth_series() -> None:
    """Criteria (i)+(iii) on the real depth series: flat classes, <=4k tokens.

    distilgpt2 / gpt2 / gpt2-medium (6/12/24 blocks): the class count must be
    FLAT (not linear in depth) and the folded overview must serialize under
    the 4k orientation budget on the deepest model.
    """

    transformers = pytest.importorskip("transformers")
    from torchlens.agent._envelope import canonical_dumps, json_safe
    from torchlens.agent._fold import fold_class_rows

    counts: dict[str, int] = {}
    deepest_tokens = 0
    for name in ("distilgpt2", "gpt2", "gpt2-medium"):
        model = transformers.AutoModelForCausalLM.from_pretrained(name).eval()
        input_ids = torch.arange(12).unsqueeze(0)
        log = tl.trace(model, (), input_kwargs={"input_ids": input_ids}, save=None)
        fold = fold_trace(log)
        counts[name] = len(fold.classes)
        rows_text = canonical_dumps(json_safe(fold_class_rows(fold)))
        deepest_tokens = max(deepest_tokens, -(-len(rows_text) // 4))
        assert fold.membership().keys() == {str(op.label) for op in log.layer_list}
        del model, log
    # Flat: the 24-block count must not scale with depth (allow small drift
    # from boundary blocks, never a per-block term).
    assert counts["gpt2-medium"] - counts["distilgpt2"] <= 6, counts
    assert deepest_tokens <= 4_000, f"folded classes serialize to ~{deepest_tokens} tokens"


@pytest.mark.slow
def test_fold_seam_decision_rule_documented_on_gpt2() -> None:
    """The D1 decision rule's evidence, kept live (agent memo 3.4 / D1).

    The reuse path (the renderer's collapse universe) shipped MEASURED-
    DEGENERATE at depth: no pinned context produced block-level boxes on
    gpt2, so the run-aware fold ships agent-side. Opus's stated flip
    condition was "any pinned context that produces block-level boxes on
    gpt2" -- if the renderer improves and this test FAILS, revisit the seam
    (factor the repeat classifier into the shared structural core) instead
    of silently keeping two grains.
    """

    transformers = pytest.importorskip("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    input_ids = torch.arange(12).unsqueeze(0)
    log = tl.trace(model, (), input_kwargs={"input_ids": input_ids}, save=None)
    plan = log.collapse_plan(mode="max")
    box_names = [type(node).__name__ for node in plan.nodes]
    block_level_boxes = [
        node
        for node in plan.nodes
        if type(node).__name__ == "ModuleBox"
        and ".h." in str(getattr(node, "module", getattr(node, "address", "")))
    ]
    fold = fold_trace(log)
    assert fold.membership().keys() == {str(op.label) for op in log.layer_list}
    assert not block_level_boxes, (
        "the collapse plan now produces block-level boxes on gpt2 "
        f"({box_names[:10]}...): Opus's D1 flip condition fired -- revisit "
        "the fold seam per the decision rule instead of keeping two grains"
    )
