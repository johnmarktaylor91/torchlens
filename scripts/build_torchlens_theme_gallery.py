#!/usr/bin/env python
"""Generate the TorchLens Dagua theme reference gallery."""

from __future__ import annotations

import json
import sys
import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from torchlens import log_forward_pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _example_models():
    return importlib.import_module("example_models")


DEFAULT_GALLERY_DIR = (
    ROOT / "tests" / "test_outputs" / "visualizations" / "torchlens_dagua_theme" / "final_gallery"
)
DEFAULT_REPORT_DIR = ROOT / "tests" / "test_outputs" / "reports" / "torchlens_dagua_theme"


@dataclass
class ReferenceSpec:
    key: str
    title: str
    structural_role: str
    description: str
    substitution_note: Optional[str]
    vis_mode: str
    vis_direction: str
    vis_nesting_depth: int
    builder_name: str
    input_description: str
    build: Callable[[], Tuple[Any, Any]]


def _simple_ff():
    example_models = _example_models()
    return example_models.SimpleFF().eval(), torch.rand(5, 5)


def _classic_cnn():
    example_models = _example_models()
    return example_models.ConvAutoencoder().eval(), torch.rand(1, 1, 28, 28)


def _resnet():
    example_models = _example_models()
    return example_models.ResidualBlockModel().eval(), torch.rand(1, 16, 16, 16)


def _unet():
    example_models = _example_models()
    return example_models.SmallUNet().eval(), torch.rand(1, 1, 32, 32)


def _transformer_encoder():
    example_models = _example_models()
    return example_models.TransformerEncoderModel().eval(), torch.rand(10, 2, 16)


def _transformer_decoder():
    example_models = _example_models()
    return example_models.TransformerDecoderModel().eval(), [
        torch.rand(8, 2, 16),
        torch.rand(10, 2, 16),
    ]


def _bilstm():
    example_models = _example_models()
    return example_models.BiLSTMModel().eval(), torch.rand(2, 6, 8)


def _simclr():
    example_models = _example_models()
    return example_models.SimCLRModel().eval(), [torch.rand(4, 16), torch.rand(4, 16)]


def _two_tower():
    example_models = _example_models()
    return example_models.TwoTowerRecommender().eval(), [torch.rand(4, 8), torch.rand(4, 8)]


def _monster():
    example_models = _example_models()
    return example_models.RandomGraphModel(
        target_nodes=120, nesting_depth=4, seed=42
    ).eval(), torch.rand(2, 64)


REFERENCE_SPECS: List[ReferenceSpec] = [
    ReferenceSpec(
        key="simple_ff",
        title="Simple feedforward",
        structural_role="sanity check",
        description="Minimal 3-op chain to judge the baseline rhythm of nodes, labels, and arrows.",
        substitution_note=None,
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=1000,
        builder_name="example_models.SimpleFF",
        input_description="torch.rand(5, 5)",
        build=_simple_ff,
    ),
    ReferenceSpec(
        key="classic_cnn",
        title="Classic CNN",
        structural_role="repetitive sequential depth",
        description="VGG-style stack for sequential convolutional depth and repeated module rhythm.",
        substitution_note="Uses example_models.ConvAutoencoder as the lightweight sequential-convolution substitute for a VGG-style stack.",
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=2,
        builder_name="example_models.ConvAutoencoder",
        input_description="torch.rand(1, 1, 28, 28)",
        build=_classic_cnn,
    ),
    ReferenceSpec(
        key="resnet",
        title="ResNet-style residual network",
        structural_role="skip connections",
        description="Residual topology for long skip edges and repeated parameterized blocks.",
        substitution_note="Uses example_models.ResidualBlockModel as the lightweight residual substitute.",
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=3,
        builder_name="example_models.ResidualBlockModel",
        input_description="torch.rand(1, 16, 16, 16)",
        build=_resnet,
    ),
    ReferenceSpec(
        key="unet",
        title="U-Net",
        structural_role="encoder-decoder with lateral skips",
        description="Nested encoder-decoder with lateral merge edges and mirrored stages.",
        substitution_note=None,
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=3,
        builder_name="example_models.SmallUNet",
        input_description="torch.rand(1, 1, 32, 32)",
        build=_unet,
    ),
    ReferenceSpec(
        key="transformer_encoder",
        title="Transformer encoder",
        structural_role="attention stack",
        description="Encoder-only transformer block stack with repeated attention and normalization motifs.",
        substitution_note=None,
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=3,
        builder_name="example_models.TransformerEncoderModel",
        input_description="torch.rand(10, 2, 16)",
        build=_transformer_encoder,
    ),
    ReferenceSpec(
        key="transformer_encoder_decoder",
        title="Encoder-decoder transformer",
        structural_role="cross-attention two-stream structure",
        description="Two-stream transformer with decoder-side cross-attention.",
        substitution_note="Uses example_models.TransformerDecoderModel as a lightweight T5-style structural substitute.",
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=3,
        builder_name="example_models.TransformerDecoderModel",
        input_description="[torch.rand(8, 2, 16), torch.rand(10, 2, 16)]",
        build=_transformer_decoder,
    ),
    ReferenceSpec(
        key="bilstm",
        title="Bidirectional LSTM",
        structural_role="recurrent network",
        description="Temporal recurrence with hidden-state threading and repeated cell structure.",
        substitution_note="Uses bidirectional LSTM rather than a plain LSTM because it stresses recurrent clarity more aggressively.",
        vis_mode="rolled",
        vis_direction="leftright",
        vis_nesting_depth=2,
        builder_name="example_models.BiLSTMModel",
        input_description="torch.rand(2, 6, 8)",
        build=_bilstm,
    ),
    ReferenceSpec(
        key="dual_subgraph",
        title="Dual-subgraph contrastive model",
        structural_role="GAN/dual-subgraph substitute",
        description="Twin shared-weight branches for contrastive learning, standing in for a generator/discriminator style split.",
        substitution_note="Uses example_models.SimCLRModel as the dual-subgraph substitute because it cleanly exposes mirrored branches and a merge objective.",
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=2,
        builder_name="example_models.SimCLRModel",
        input_description="[torch.rand(4, 16), torch.rand(4, 16)]",
        build=_simclr,
    ),
    ReferenceSpec(
        key="multi_input",
        title="Multi-input recommender",
        structural_role="heterogeneous input streams merging",
        description="Parallel user/item towers with a late interaction point.",
        substitution_note="Uses example_models.TwoTowerRecommender as the multi-input structural reference.",
        vis_mode="unrolled",
        vis_direction="leftright",
        vis_nesting_depth=2,
        builder_name="example_models.TwoTowerRecommender",
        input_description="[torch.rand(4, 8), torch.rand(4, 8)]",
        build=_two_tower,
    ),
    ReferenceSpec(
        key="monster",
        title="Monster graph",
        structural_role="100+ layer stress test",
        description="Large nested random graph model for clutter, hierarchy, and scale resilience.",
        substitution_note="Uses example_models.RandomGraphModel(target_nodes=120, nesting_depth=4) as the lightweight production-scale stress test.",
        vis_mode="rolled",
        vis_direction="leftright",
        vis_nesting_depth=2,
        builder_name="example_models.RandomGraphModel",
        input_description="torch.rand(2, 64)",
        build=_monster,
    ),
]


def _render_one(spec: ReferenceSpec, gallery_dir: Path) -> Dict[str, Any]:
    model, input_args = spec.build()
    log = log_forward_pass(model, input_args, layers_to_save=None)
    outpath = gallery_dir / f"{spec.key}.png"
    try:
        log.render_graph(
            vis_renderer="dagua",
            vis_theme="torchlens",
            vis_mode=spec.vis_mode,
            direction=spec.vis_direction,
            vis_nesting_depth=spec.vis_nesting_depth,
            vis_save_only=True,
            vis_fileformat="png",
            vis_outpath=str(outpath),
        )
        audit = log.visualization_field_audit().to_dict()
        result = {
            "status": "ok",
            "key": spec.key,
            "title": spec.title,
            "structural_role": spec.structural_role,
            "description": spec.description,
            "substitution_note": spec.substitution_note,
            "builder_name": spec.builder_name,
            "input_description": spec.input_description,
            "vis_mode": spec.vis_mode,
            "vis_direction": spec.vis_direction,
            "vis_nesting_depth": spec.vis_nesting_depth,
            "output_path": str(outpath),
            "num_nodes": len(log.layer_labels),
            "is_recurrent": bool(log.is_recurrent),
            "has_conditional_branching": bool(log.has_conditional_branching),
            "audit": audit,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "key": spec.key,
            "title": spec.title,
            "structural_role": spec.structural_role,
            "description": spec.description,
            "substitution_note": spec.substitution_note,
            "builder_name": spec.builder_name,
            "input_description": spec.input_description,
            "vis_mode": spec.vis_mode,
            "vis_direction": spec.vis_direction,
            "vis_nesting_depth": spec.vis_nesting_depth,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        log.cleanup()
    return result


def build_gallery(
    gallery_dir: Path = DEFAULT_GALLERY_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Dict[str, Any]:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = [_render_one(spec, gallery_dir) for spec in REFERENCE_SPECS]
    manifest = {
        "theme": "torchlens",
        "renderer": "dagua",
        "gallery_dir": str(gallery_dir),
        "reports_dir": str(report_dir),
        "references": results,
    }

    (gallery_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "# TorchLens Dagua Theme Gallery",
        "",
        "This gallery is the maintained 10-architecture reference set for the `torchlens` Dagua theme.",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item['title']}",
                "",
                f"- Key: `{item['key']}`",
                f"- Structural role: {item['structural_role']}",
                f"- Builder: `{item['builder_name']}`",
                f"- Input: `{item['input_description']}`",
                f"- Render mode: `{item['vis_mode']}` / `{item['vis_direction']}` / depth `{item['vis_nesting_depth']}`",
                f"- Status: `{item['status']}`",
            ]
        )
        if item["status"] == "ok":
            lines.extend(
                [
                    f"- Nodes: {item['num_nodes']}",
                    f"- Output: `{Path(item['output_path']).name}`",
                ]
            )
        else:
            lines.append(f"- Error: `{item['error']}`")
        if item["substitution_note"]:
            lines.append(f"- Substitution: {item['substitution_note']}")
        if item["status"] == "ok":
            lines.extend(["", f"![{item['title']}]({Path(item['output_path']).name})", ""])
        else:
            lines.extend(
                ["", "_Render failed for this reference model in the current environment._", ""]
            )

    (gallery_dir / "README.md").write_text("\n".join(lines))
    return manifest


if __name__ == "__main__":
    build_gallery()
