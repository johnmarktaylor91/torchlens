"""Runnable Semantic I/O legibility demo.

This example is intentionally tiny and deterministic. It demonstrates the
review-day semantic I/O surface without downloading optional torchvision or
Hugging Face models.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import ResolvedPostprocessing, ResolvedPreprocessing


class TinyClassifier(nn.Module):
    """Small classifier with HF-style ``config.id2label`` metadata."""

    def __init__(self) -> None:
        """Initialize deterministic weights and semantic labels."""

        super().__init__()
        self.config = SimpleNamespace(
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            num_labels=3,
        )
        self.linear = nn.Linear(4, 3)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                )
            )
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for ``x``."""

        return self.linear(x)


def text_to_tensor(text: str) -> torch.Tensor:
    """Convert one comma-separated text row into a model-ready tensor.

    Parameters
    ----------
    text:
        Four comma-separated floating point values.

    Returns
    -------
    torch.Tensor
        Rank-2 tensor with one batch item.
    """

    values = [float(part.strip()) for part in text.split(",")]
    return torch.tensor([values], dtype=torch.float32)


def register_custom_decoder() -> None:
    """Register a demo output detector with a custom label bank."""

    @tl.autoroute.output.register(name="demo_custom_labels", priority=5)
    def demo_custom_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        """Resolve the demo-only ``output_style='demo_custom_labels'`` style."""

        if meta.get("output_style") != "demo_custom_labels":
            return None
        return ResolvedPostprocessing(
            source="example",
            identifier="demo_custom_labels",
            verified=True,
            config={"id2label": {0: "red", 1: "green", 2: "blue"}},
            description="demo custom label bank",
            style="classification",
            selected_output_head=meta.get("output_head"),
            label_source="example-local",
            label_source_version="demo-v1",
            confidence=1.0,
        )


def print_table(title: str, trace: tl.Trace, *, top_n: int = 2) -> None:
    """Print a compact decoded output table.

    Parameters
    ----------
    title:
        Section title.
    trace:
        Trace with decoded output rows.
    top_n:
        Maximum rank to print per batch item.
    """

    print(f"\n## {title}")
    print(trace.output_table(top_n=top_n).to_string(index=False))


def main() -> None:
    """Run the end-to-end Semantic I/O legibility demo."""

    model = TinyClassifier().eval()
    x = torch.tensor(
        [
            [0.1, 0.2, 2.0, 1.0],
            [1.5, 0.4, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )

    auto_trace = tl.trace(model, x)
    print_table("auto-detected output labels", auto_trace)
    print("\n## output summary")
    print(auto_trace.summary(level="output"))
    print("\n## decode_output(top_n=1)")
    print(auto_trace.decode_output(top_n=1))
    output_rows = auto_trace.to_pandas(include_decoded_output_summary=True)
    output_preview = output_rows.loc[output_rows["is_output"], ["label", "decoded_output_summary"]]
    print("\n## output-node pandas summary")
    print(output_preview.to_string(index=False))

    override_trace = tl.trace(model, x, output_style="classification")
    print_table("override: output_style='classification'", override_trace, top_n=1)
    print(f"override postprocessor: {override_trace.output_postprocessor.description}")

    register_custom_decoder()
    try:
        custom_trace = tl.trace(model, x, output_style="demo_custom_labels")
        print_table("custom decoder: output_style='demo_custom_labels'", custom_trace, top_n=1)
        print(f"custom postprocessor: {custom_trace.output_postprocessor.description}")
    finally:
        tl.autoroute.output.unregister("demo_custom_labels")

    text_trace = tl.trace(
        model,
        "0.1,0.2,2.0,1.0",
        transform=text_to_tensor,
        output_style="classification",
    )
    text_trace.input_preprocessor = ResolvedPreprocessing(
        source="example.transform",
        identifier="comma-vector-v1",
        verified=True,
        config={"format": "comma-separated floats", "shape": [1, 4]},
        description="comma-separated text -> rank-2 float tensor",
    )
    print("\n## original input + preprocessing provenance")
    print(f"raw_input: {text_trace.raw_input!r}")
    print(f"input_preprocessor: {text_trace.input_preprocessor.description}")
    with tempfile.TemporaryDirectory(prefix="torchlens_semantic_io_") as tmpdir:
        outpath = Path(tmpdir) / "semantic_io_input"
        text_trace.draw(
            vis_outpath=str(outpath),
            vis_save_only=True,
            vis_fileformat="svg",
            show_input_transform_summary=True,
        )
        print(f"input display SVG: {outpath}.svg")


if __name__ == "__main__":
    main()
