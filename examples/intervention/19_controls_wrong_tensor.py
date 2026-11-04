"""Catch an ablation aimed at the wrong tensor with baseline, direction and site controls.

What this demonstrates
----------------------
A small classifier multiplies its normalized hidden features by a ``gain`` that is
almost constant (about 3). Zero-ablating ``gain`` wipes out the logit margin, so
"gain carries the answer" looks like a large, significant effect. It is not: the
zero fill switches the whole hidden vector off, and any tensor on that path would
produce the same drop. ``run_controls`` reruns the claim with the mean and with
resampled rows of ``gain``; both leave the margin where it was, so the claim does
not survive.

The same controls on the tensor that does carry the answer (the normalized hidden
features) hold up: every baseline keeps the effect, projecting out the readout
direction beats random directions of the same norm, and the effect holds at the
pre-norm and post-norm sites alike.

The helper's public spelling is pending, so this example imports it from its
private module.

How to run
----------
``python examples/intervention/19_controls_wrong_tensor.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.mechinterp._controls import run_controls


class Gain(nn.Module):
    """A per-row scale that barely depends on the input: ``3 + 0.01 * tanh(w . x)``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the near-constant scale, one value per row."""

        return 3.0 + 0.01 * torch.tanh(self.proj(x))


class Classifier(nn.Module):
    """``readout(norm(embed(x)) * gain(x))``: the answer lives in the normalized features."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 16)
        self.norm = nn.LayerNorm(16)
        self.gain = Gain()
        self.readout = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.readout(self.norm(self.embed(x)) * self.gain(x))


def _train(model: Classifier, x: torch.Tensor, labels: torch.Tensor) -> None:
    """Fit the sign of the first input feature (deterministic, under a second)."""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()


def build() -> tuple[Classifier, torch.Tensor, torch.Tensor]:
    """Return the trained model, its 64 evaluation rows and their labels."""

    torch.manual_seed(0)
    model = Classifier()
    x = torch.randn(256, 4)
    labels = (x[:, 0] > 0).long()
    _train(model, x, labels)
    model.eval()
    return model, x[:64], labels[:64]


def margin_metric(labels: torch.Tensor):
    """The mean logit margin of the true class over the wrong one."""

    sign = labels.float() * 2 - 1

    def metric(trace) -> torch.Tensor:
        """Read the margin from a trace's output."""

        logits = trace.output_ops[-1].out
        return ((logits[:, 1] - logits[:, 0]) * sign).mean()

    return metric


def main() -> None:
    """Run the controls on the wrong tensor and on the right one."""

    model, x, labels = build()
    metric = margin_metric(labels)

    wrong = run_controls(model, x, tl.module("gain"), metric, axis=0)
    print(wrong)
    assert wrong.effect < 0  # the zero fill alone looks like a big drop
    assert not wrong.survives  # mean and resample show gain carries nothing

    direction = model.readout.weight[1] - model.readout.weight[0]
    right = run_controls(
        model,
        x,
        tl.module("norm"),
        metric,
        axis=0,
        direction=direction.detach(),
        feature_axis=-1,
        n_directions=39,
        sites={"pre-norm": tl.module("embed"), "post-norm": tl.module("norm")},
    )
    print()
    print(right)
    assert right.survives


if __name__ == "__main__":
    main()
