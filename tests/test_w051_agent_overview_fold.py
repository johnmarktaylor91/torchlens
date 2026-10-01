"""W051-AGENT: overview anomalies read the tri-state (AUD-CODE 2.5) and the
fold resolves edges + boundary roles (AUD-CODE 3.11b), incl. the realistic
fixture (AUD-CODE 4.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import (
    RepeatedBlockNet,
    deterministic_input,
    save_clean_artifact,
)
from tests.test_w051_agent_helpers import Loop, MiniTransformer, mini_ids
from torchlens.agent import call_tool
from torchlens.agent._fold import fold_class_rows, fold_trace
from torchlens.agent._overview import _anomalies_block


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The deterministic clean fixture artifact (module-scoped)."""

    return save_clean_artifact(tmp_path_factory.mktemp("w051_overview"))


@pytest.mark.smoke
def test_clean_capture_has_no_anomalies(clean: Path) -> None:
    """A COMPLETE capture with capture_verified=None (no ceiling) is HEALTHY."""

    envelope = call_tool("torchlens_overview", {"path": str(clean)})
    assert envelope["data"]["capture"]["capture_status"] == "complete"
    assert envelope["data"]["capture"]["capture_verified"] is None
    assert envelope["data"]["anomalies"] == []


@pytest.mark.smoke
def test_anomaly_block_reads_the_tri_state() -> None:
    """None never flags; an explicit False flags with its recorded reason."""

    class _Quiet:
        nonfinite_ops: list[str] = []

    healthy = _anomalies_block(_Quiet(), {"capture_verified": None})
    assert healthy == []
    ceilinged = _anomalies_block(
        _Quiet(), {"capture_verified": False, "capture_verification_reason": "mode_rescue_rerun"}
    )
    assert ceilinged == [{"kind": "capture_unverified", "detail": "mode_rescue_rerun"}]


def _expected_edges(log: tl.Trace, fold_membership: dict[str, str]) -> dict[str, set[str]]:
    """Derive each class's parent classes from the op records' own edges."""

    by_bare = {str(op.layer_label): fold_membership[str(op.label)] for op in log.layer_list}
    expected: dict[str, set[str]] = {}
    for op in log.layer_list:
        cls = fold_membership[str(op.label)]
        for parent in op.parents:
            spelled = str(parent)  # pass-qualified OR bare, both occur
            target = fold_membership[spelled] if spelled in fold_membership else by_bare[spelled]
            expected.setdefault(cls, set()).add(target)
    return expected


@pytest.mark.smoke
def test_fold_edges_resolve_on_single_pass_traces() -> None:
    """Bare single-pass edge labels resolve: parent/child classes are populated and exact."""

    log = tl.trace(RepeatedBlockNet().eval(), deterministic_input(), save=tl.func("relu"))
    fold = fold_trace(log)
    expected = _expected_edges(log, fold.membership())
    for cls in fold.classes:
        assert set(cls.parent_classes) == expected.get(cls.class_id, set()), cls.class_id
    interior = [cls for cls in fold.classes if cls.boundary is None]
    assert interior and all(cls.parent_classes and cls.child_classes for cls in interior)
    # Reciprocity: A lists B as a child iff B lists A as a parent.
    for cls in fold.classes:
        for child in cls.child_classes:
            target = next(c for c in fold.classes if c.class_id == child)
            assert cls.class_id in target.parent_classes


@pytest.mark.smoke
def test_fold_keeps_input_and_output_in_distinct_classes() -> None:
    """Same shape, same func_name='none', empty stack -- still two classes."""

    log = tl.trace(Loop().eval(), deterministic_input())
    fold = fold_trace(log)
    membership = fold.membership()
    input_class = membership[str(log.input_ops[0].label)]
    output_class = membership[str(log.output_ops[0].label)]
    assert input_class != output_class
    rows = {row["class_id"]: row for row in fold_class_rows(fold)}
    assert rows[input_class]["boundary"] == "input"
    assert rows[output_class]["boundary"] == "output"
    assert rows[input_class]["shape"] == rows[output_class]["shape"]
    assert all(
        "boundary" not in row for cid, row in rows.items() if cid not in (input_class, output_class)
    )
    # Multi-pass edges (pass-qualified spellings) keep resolving exactly.
    assert rows[output_class]["parent_classes"] and rows[input_class]["child_classes"]


@pytest.mark.heavy
def test_realistic_fixture_folds_flat_with_edges() -> None:
    """~108 ops fold to a flat class set with every interior edge resolved (4.9)."""

    log = tl.trace(MiniTransformer().eval(), mini_ids(), save=tl.func("softmax") | tl.func("gelu"))
    fold = fold_trace(log)
    n_ops = len(log.layer_list)
    assert n_ops > 100
    assert len(fold.classes) < n_ops // 2
    assert fold.membership().keys() == {str(op.label) for op in log.layer_list}
    block_classes = [
        cls for cls in fold.classes if any(p.startswith("blocks.*") for p in cls.module_path)
    ]
    assert block_classes and all(cls.n_instances % 4 == 0 for cls in block_classes)
    expected = _expected_edges(log, fold.membership())
    by_label = {str(op.label): op for op in log.layer_list}
    for cls in fold.classes:
        assert set(cls.parent_classes) == expected.get(cls.class_id, set()), cls.class_id
        # Empty parent classes iff NO member records a parent (a mid-forward
        # torch.arange is a legitimate source op); otherwise edges resolved.
        has_parents = any(by_label[m].parents for m in cls.members)
        assert bool(cls.parent_classes) == has_parents, cls.class_id
    assert sum(1 for cls in fold.classes if cls.parent_classes) > len(fold.classes) * 0.8
    outputs = {cls.class_id for cls in fold.classes if cls.boundary == "output"}
    assert len(outputs) == 2  # logits (rank 3) and the scalar loss (rank 0) split on rank
