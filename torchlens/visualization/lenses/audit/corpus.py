"""The lens evidence corpus (themes memo section 8).

Toys are ADDITIONAL evidence, never substitutes for the real corpus; the
real members here are GUARDED builders (skipped where torchvision or the
checkpoint is unavailable) with the manifest naming the EXACT constructor
-- "a ViT" is at least three different stress cases.

Rendered artifacts (DOT + SVG + PNG) belong under sprint scratch, never
the repo.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CORPUS", "CorpusMember", "RunMeasurements", "build_run_manifest", "toy_builders"]


@dataclass(frozen=True)
class CorpusMember:
    """One corpus row: an exact constructor and its input recipe.

    Attributes
    ----------
    name:
        Manifest name (exact, never "a ViT").
    build:
        Zero-arg callable returning ``(model, example_input)``.
    kind:
        ``"toy"`` / ``"real"`` / ``"stress"``.
    tags:
        Which strata the member serves (``"multi_pass"``,
        ``"nonfinite"``, ``"attention"``, ``"filter"``, ...).
    notes:
        Why the member is in the corpus, from the memo.
    """

    name: str
    build: Callable[[], tuple[Any, Any]]
    kind: str = "toy"
    tags: tuple[str, ...] = ()
    notes: str = ""


def _two_node_minimal() -> tuple[Any, Any]:
    """The 2-node minimal member."""

    import torch
    import torch.nn as nn

    return nn.Linear(4, 2), torch.randn(1, 4)


def _residual_diamond() -> tuple[Any, Any]:
    """Residual diamond: one fork, two arms, one join."""

    import torch
    import torch.nn as nn

    class Diamond(nn.Module):
        """Fork/join residual toy."""

        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(8, 8)
            self.right = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, x: Any) -> Any:
            """Run the two-branch diamond and rejoin at the head."""

            return self.head(torch.relu(self.left(x)) + torch.sigmoid(self.right(x)))

    return Diamond(), torch.randn(2, 8)


def _lstm_cell_loop() -> tuple[Any, Any]:
    """Tied LSTMCell Python loop: multi-pass layers with varying per-pass fields."""

    import torch
    import torch.nn as nn

    class TiedLoop(nn.Module):
        """Six-step tied LSTMCell loop (fused kernels draw differently)."""

        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTMCell(4, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x: Any) -> Any:
            """Run the tied-weight recurrent loop over the sequence."""

            hidden = torch.zeros(x.shape[0], 8)
            cell_state = torch.zeros(x.shape[0], 8)
            for step in range(6):
                hidden, cell_state = self.cell(x[:, step], (hidden, cell_state))
            return self.head(hidden)

    return TiedLoop(), torch.randn(2, 6, 4)


def _nonfinite_chain() -> tuple[Any, Any]:
    """finite -> +Inf -> NaN chain with a clean side branch.

    Exercises the six-state channel: finite ops, a +Inf producer, a NaN
    producer (Inf - Inf), and -- captured with a selective ``save=`` -- a
    ``not_checked`` stratum on the unsaved branch.
    """

    import torch
    import torch.nn as nn

    class Blowup(nn.Module):
        """Submodule whose interior ops render as module boxes (stripe-safe)."""

        def forward(self, clean: Any) -> Any:
            """Blow the payload up into the full nonfinite state set."""

            positive = clean + 1.0  # strictly positive: division is pure +Inf
            blown_pos = positive / torch.zeros_like(positive)  # +Inf only
            blown_neg = -positive / torch.zeros_like(positive)  # -Inf only
            return blown_pos + blown_neg  # Inf + -Inf = NaN

    class NonfiniteChain(nn.Module):
        """Deterministic nonfinite factory (division by zero, Inf - Inf)."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.blow = Blowup()

        def forward(self, x: Any) -> Any:
            """Run clean ops, poison mid-chain, and propagate downstream."""

            clean = torch.relu(self.fc(x))
            poisoned = self.blow(clean)
            side = torch.sigmoid(clean)  # stays finite
            return poisoned.nan_to_num(0.0) + side

    return NonfiniteChain(), torch.randn(2, 4)


def _encdec_skips() -> tuple[Any, Any]:
    """Shape-changing encoder/decoder with skip connections (dims stressor)."""

    import torch
    import torch.nn as nn

    class EncDec(nn.Module):
        """Conv encoder/decoder with one skip join."""

        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
            self.enc2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
            self.dec1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
            self.dec2 = nn.ConvTranspose2d(8, 1, 2, stride=2)

        def forward(self, x: Any) -> Any:
            """Run the encoder/decoder with the long skip connection."""

            skip = torch.relu(self.enc1(x))
            deep = torch.relu(self.enc2(skip))
            up = torch.relu(self.dec1(deep)) + skip
            return self.dec2(up)

    return EncDec(), torch.randn(1, 1, 16, 16)


def _filtered_chain() -> tuple[Any, Any]:
    """Reshape-heavy chain/branch fixture for display-filter strata."""

    import torch
    import torch.nn as nn

    class Glue(nn.Module):
        """Chain with a reshape/permute/contiguous glue run between layers."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            """Run linears separated by reshape/permute/contiguous glue."""

            hidden = torch.relu(self.fc1(x))
            hidden = hidden.reshape(4, -1).permute(1, 0).contiguous()
            return self.fc2(hidden)

    return Glue(), torch.randn(2, 8)


def _attention_toy() -> tuple[Any, Any]:
    """Minimal real-attention member (transformer-lens subject)."""

    import torch
    import torch.nn as nn

    class TinyAttention(nn.Module):
        """One MultiheadAttention block plus an MLP head."""

        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.head = nn.Linear(8, 4)

        def forward(self, x: Any) -> Any:
            """Run one self-attention block and project through the head."""

            attended, _ = self.attn(x, x, x, need_weights=False)
            return self.head(attended)

    return TinyAttention(), torch.randn(2, 5, 8)


def _stress_diamond_10k() -> tuple[Any, Any]:
    """StressDiamond10k: deterministic ~10k-op renderer fixture (STRESS)."""

    import torch
    import torch.nn as nn

    class StressDiamond(nn.Module):
        """2,500 fork/join diamonds in sequence (~10k ops)."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            """Run the above-optimizer-ceiling stress chain of adds."""

            hidden = self.fc(x)
            for _ in range(2500):
                hidden = torch.relu(hidden) + torch.sigmoid(hidden)
            return hidden

    return StressDiamond(), torch.randn(1, 4)


def _torchvision_member(constructor_name: str) -> Callable[[], tuple[Any, Any]]:
    """Return a guarded random-init torchvision builder (geometry strata).

    Random init serves the STRUCTURAL strata (geometry, budget, coverage);
    the battery's real-weights legs are D03's, per the memo corpus rules.
    """

    def _build() -> tuple[Any, Any]:
        """Construct the random-init torchvision member and its input."""

        import importlib

        import torch

        torchvision_models = importlib.import_module("torchvision.models")
        model = getattr(torchvision_models, constructor_name)(weights=None)
        model.eval()
        return model, torch.randn(1, 3, 224, 224)

    _build.__name__ = f"build_{constructor_name}"
    return _build


def toy_builders() -> dict[str, Callable[[], tuple[Any, Any]]]:
    """Return the toy builder table (name -> builder)."""

    return {
        "two_node_minimal": _two_node_minimal,
        "residual_diamond": _residual_diamond,
        "lstm_cell_loop": _lstm_cell_loop,
        "nonfinite_chain": _nonfinite_chain,
        "encdec_skips": _encdec_skips,
        "filtered_chain": _filtered_chain,
        "attention_toy": _attention_toy,
    }


#: The corpus rows. Real members are torchvision random-init builders for
#: the structural strata; the HF real-checkpoint legs (gpt2, vit-base, t5,
#: clip, convit, the 2,802-op generation episode) are named in the gallery
#: spec and executed by D03 with pinned revisions.
CORPUS: tuple[CorpusMember, ...] = (
    CorpusMember("two_node_minimal", _two_node_minimal, "toy", ("minimal",)),
    CorpusMember("residual_diamond", _residual_diamond, "toy", ("structure",)),
    CorpusMember(
        "lstm_cell_loop",
        _lstm_cell_loop,
        "toy",
        ("multi_pass", "sequence"),
        notes="tied LSTMCell loop; varying and constant per-pass fields",
    ),
    CorpusMember(
        "nonfinite_chain",
        _nonfinite_chain,
        "toy",
        ("nonfinite", "debug"),
        notes="finite -> +Inf -> NaN with a clean branch; selective save adds not_checked",
    ),
    CorpusMember("encdec_skips", _encdec_skips, "toy", ("dims", "shapes")),
    CorpusMember("filtered_chain", _filtered_chain, "toy", ("filter",)),
    CorpusMember("attention_toy", _attention_toy, "toy", ("attention", "transformer")),
    CorpusMember(
        "torchvision_resnet18_random_init",
        _torchvision_member("resnet18"),
        "real",
        ("geometry", "coverage"),
        notes="structural strata only (random init); real-weights legs are D03's",
    ),
    CorpusMember(
        "torchvision_densenet121_random_init",
        _torchvision_member("densenet121"),
        "real",
        ("geometry", "large"),
        notes="917 ops: large but under the optimizer ceiling",
    ),
    CorpusMember(
        "stress_diamond_10k",
        _stress_diamond_10k,
        "stress",
        ("above_ceiling",),
        notes="the deterministic 10k-op above-ceiling fixture; never in smoke tests",
    ),
)


@dataclass(frozen=True)
class _ManifestVersions:
    """Version stamp block for one rendered artifact."""

    torch: str
    torchlens: str
    python: str
    graphviz_python: str | None = None


@dataclass(frozen=True)
class RunMeasurements:
    """Measured values one artifact run supplies to its manifest."""

    visible_count: int | None = None
    wall_seconds: float | None = None
    output_path: str | None = None
    extra: dict[str, Any] | None = None


def build_run_manifest(
    member: str,
    *,
    lens: str,
    skin: str | None,
    resolved_kwargs: dict[str, Any],
    measured: RunMeasurements | None = None,
) -> dict[str, Any]:
    """Build the per-artifact run manifest (memo section 8).

    Everything a reproduction needs: versions, the exact member name, the
    resolved row + skin, counts, wall time, output path. Deterministic
    fields only; the caller supplies the ``measured`` values.
    """

    import graphviz  # a hard core dependency (pyproject); never absent here
    import torch

    import torchlens

    measured = measured or RunMeasurements()
    versions = _ManifestVersions(
        # Spelled via torch.version: visualization holds no torch-privates
        # license row, and the arch-spine probe reads the usual dunder
        # version spelling on the torch module as a private touch.
        torch=str(torch.version.__version__),
        torchlens=getattr(torchlens, "__version__", "unknown"),
        python=platform.python_version(),
        graphviz_python=getattr(graphviz, "__version__", None),
    )
    manifest: dict[str, Any] = {
        "member": member,
        "lens": lens,
        "skin": skin,
        "resolved_settings": {name: repr(value) for name, value in sorted(resolved_kwargs.items())},
        "versions": versions.__dict__,
        "visible_count": measured.visible_count,
        "wall_seconds": measured.wall_seconds,
        "output_path": measured.output_path,
        "recorded_at_monotonic": time.monotonic(),
    }
    if measured.extra:
        manifest.update(measured.extra)
    return manifest
