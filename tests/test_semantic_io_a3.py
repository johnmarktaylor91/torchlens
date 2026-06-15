"""Sprint A3 semantic output table and surface tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.rendering import _render_raw_output


class _Config:
    """Small classifier config carrying explicit labels."""

    def __init__(self, labels: dict[int, str]) -> None:
        """Initialize config labels."""

        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)
        self.model_type = "tiny"
        self.architectures = ["TinyForImageClassification"]


class _Classifier(nn.Module):
    """Classifier returning fixed logits with optional label metadata."""

    def __init__(self, logits: torch.Tensor, labels: dict[int, str] | None = None) -> None:
        """Initialize fixed logits and optional config."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("_logits", logits)
        if labels is not None:
            self.config = _Config(labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits connected to ``x`` and a parameter."""

        return self._logits.to(x.device) + (x.sum() + self.weight) * 0


def _labels(count: int) -> dict[int, str]:
    """Return deterministic label metadata.

    Parameters
    ----------
    count:
        Number of labels.

    Returns
    -------
    dict[int, str]
        Integer label map.
    """

    return {index: f"class-{index}" for index in range(count)}


def _trace_batch_classifier() -> tl.Trace:
    """Trace a small batch classifier.

    Returns
    -------
    tl.Trace
        Trace with decoded batch top-k output.
    """

    logits = torch.tensor([[0.0, 5.0, 2.0, 1.0], [4.0, 0.0, 3.0, 1.0]])
    return tl.trace(_Classifier(logits, _labels(4)).eval(), torch.ones(2, 3))


def test_output_table_shape_top_n_and_batch_item_overrides() -> None:
    """``output_table`` exposes top-k rows with top_n and batch selectors."""

    trace = _trace_batch_classifier()

    table = trace.output_table(top_n=2)
    assert list(table.columns) == ["batch_item", "rank", "label", "prob"]
    assert table.shape == (4, 4)
    assert table["batch_item"].tolist() == [0, 0, 1, 1]
    assert table["rank"].tolist() == [1, 2, 1, 2]
    assert table["label"].tolist() == ["class-1", "class-2", "class-0", "class-2"]

    first_item = trace.output_table(top_n=1, batch_items=1)
    assert first_item[["batch_item", "rank", "label"]].to_dict(orient="records") == [
        {"batch_item": 0, "rank": 1, "label": "class-1"}
    ]

    explicit_item = trace.output_table(top_n=1, batch_items=[1])
    assert explicit_item[["batch_item", "rank", "label"]].to_dict(orient="records") == [
        {"batch_item": 1, "rank": 1, "label": "class-0"}
    ]


def test_decoded_output_uses_typed_batch_topk_representation() -> None:
    """Classification decoded output is stored as a typed batch top-k blob."""

    trace = _trace_batch_classifier()

    assert trace.decoded_output["kind"] == "batch_topk"
    assert trace.decode_output(top_n=1)["rows"] == [
        row for row in trace.decoded_output["rows"] if row["rank"] == 1
    ]


def test_render_raw_output_single_value_paths_are_byte_identical() -> None:
    """Existing raw-output render branches remain byte-identical."""

    html_prefix = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        '<TR><TD ALIGN="CENTER"><B>output</B></TD></TR>'
    )
    html_suffix = "</TABLE>>"
    assert _render_raw_output("done") == {
        "label": f'{html_prefix}<TR><TD ALIGN="CENTER">done</TD></TR>{html_suffix}',
        "tooltip": "done",
    }
    label_scores = [("cat", 0.9), ("dog", 0.1)]
    assert _render_raw_output(label_scores) == {
        "label": (
            f'{html_prefix}<TR><TD ALIGN="CENTER">cat 90%</TD></TR>'
            f'<TR><TD ALIGN="CENTER">dog 10%</TD></TR>{html_suffix}'
        ),
        "tooltip": repr(label_scores),
    }
    mapping = {"answer": "yes"}
    assert _render_raw_output(mapping) == {
        "label": f'{html_prefix}<TR><TD ALIGN="CENTER">answer: yes</TD></TR>{html_suffix}',
        "tooltip": repr(mapping),
    }


def test_render_raw_output_batch_topk_branch() -> None:
    """Typed batch top-k decoded output renders as an output table."""

    attrs = _render_raw_output(_trace_batch_classifier().decoded_output)

    assert attrs is not None
    assert "item 0" in attrs["label"]
    assert "class-1 93%" in attrs["label"]
    assert "batch_topk" in attrs["tooltip"]


def test_default_to_pandas_has_no_decoded_output_column_and_gated_column_is_output_only() -> None:
    """Decoded output summary does not leak into default per-op pandas export."""

    trace = _trace_batch_classifier()

    default_frame = trace.to_pandas()
    assert "decoded_output_summary" not in default_frame.columns

    gated_frame = trace.to_pandas(include_decoded_output_summary=True)
    assert "decoded_output_summary" in gated_frame.columns
    non_null = gated_frame[gated_frame["decoded_output_summary"].notna()]
    assert set(non_null["is_output"]) == {True}
    assert "class-1" in str(non_null.iloc[0]["decoded_output_summary"])


def test_summary_surfaces_output_postprocessing_and_output_level() -> None:
    """Summary includes output provenance and an output table level."""

    trace = _trace_batch_classifier()
    summary = trace.summary()
    output_summary = trace.summary(level="output")

    assert "Output postprocessing:" in summary
    assert "style=classification" in summary
    assert "preview: item 0:" in summary
    assert "Output Summary:" in output_summary
    assert "class-1" in output_summary


def test_summary_surfaces_undetected_output_style_hint() -> None:
    """Undetected output decode suggests the explicit style override."""

    trace = tl.trace(_Classifier(torch.zeros(1, 3)).eval(), torch.ones(1, 2))

    assert trace.decoded_output is None
    assert "undetected; pass output_style= to decode." in trace.summary()


def test_redecode_after_load_raises_clearly_when_logits_dropped(tmp_path: Path) -> None:
    """Loaded traces without retained logits reject larger re-decode requests clearly."""

    trace = _trace_batch_classifier()
    bundle_path = tmp_path / "decoded.tlspec"
    tl.save(trace, bundle_path)
    restored = tl.load(bundle_path)
    for output_op in restored.output_ops:
        output_op.out = None

    with pytest.raises(ValueError, match="re-decode unavailable"):
        restored.output_table(top_n=6)


def test_false_positive_output_detection_cases_fail_closed() -> None:
    """Segmentation, CIFAR-like, and bare 1000-wide regressors do not auto-decode."""

    cases: list[tuple[nn.Module, torch.Tensor]] = [
        (_Classifier(torch.zeros(1, 1000, 2, 2), _labels(1000)).eval(), torch.ones(1, 2)),
        (_Classifier(torch.zeros(1, 100)).eval(), torch.ones(1, 2)),
        (_Classifier(torch.zeros(2, 1000)).eval(), torch.ones(2, 2)),
    ]

    for model, inputs in cases:
        trace = tl.trace(model, inputs)
        assert trace.decoded_output is None


def test_output_table_recomputes_when_logits_are_retained() -> None:
    """Requests above captured top-k recompute from retained output logits."""

    logits = torch.tensor([[0.0, 6.0, 5.0, 4.0, 3.0, 2.0]])
    trace = tl.trace(_Classifier(logits, _labels(6)).eval(), torch.ones(1, 2))

    table = trace.output_table(top_n=6)

    assert table.shape == (6, 4)
    assert table["rank"].tolist() == [1, 2, 3, 4, 5, 6]
