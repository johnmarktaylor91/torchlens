"""Save, load, and execute a sparse runnable ``.tlspec`` artifact."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class TinyClassifier(nn.Module):
    """Small deterministic model for the runnable artifact example."""

    def __init__(self) -> None:
        """Initialize deterministic classifier weights."""

        super().__init__()
        self.projection = nn.Linear(4, 2)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor([[1.0, 0.5, -0.5, 0.0], [-0.5, 0.0, 0.5, 1.0]])
            )
            self.projection.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return classifier logits."""

        return self.projection(inputs).relu()


def main() -> None:
    """Demonstrate sparse, embedded-state, and activation-attested execution."""

    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    trace = tl.trace(
        TinyClassifier().eval(),
        inputs,
        capture=CaptureOptions(
            layers_to_save="all",
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "classifier.tlspec"
        tl.save(
            trace,
            artifact,
            level="runnable",
            include_weights=True,
            include_activations=True,
        )
        loaded = tl.load(artifact)
        result = loaded.run(inputs=inputs, seed=7, on_divergence="raise")
        print(result.output)
        print(result.report.state_source.value)
        print(result.report.path_faithfulness.value)
        print(result.report.numeric_attestation.value)


if __name__ == "__main__":
    main()
