# SOURCE: vendored from eliahuhorwitz/3D-ADS @ main (feature_extractors/features.py)
# https://github.com/eliahuhorwitz/3D-ADS
# Paper: "Back to the Feature: Classical 3D Features are (Almost) All You Need for
# 3D Anomaly Detection" (Horwitz & Hoshen, VAND Workshop @ CVPR 2023).
#
# 3D-ADS's own architectural contribution is a PatchCore-style memory-bank anomaly detector
# built on top of (a) an off-the-shelf pretrained CNN backbone for RGB features and (b)
# classical (non-neural) FPFH point-cloud descriptors for the depth/3D branch -- the memory
# bank, coreset subsampling, and re-weighting logic are all non-parametric (numpy/sklearn),
# not an nn.Module forward graph. The one real, traceable neural architecture in the repo is
# this `Model` class: a thin wrapper around a timm feature-extraction backbone (used
# unmodified by both `RGBInetFeatures` and `DepthInetFeatures` in feature_extractors/*.py).
# Vendored verbatim (only the timm `pretrained=True` call is defaulted to False here so the
# tiny build below is hermetic / does not require a network fetch of ImageNet weights).
from __future__ import annotations

import torch
import timm


class Model(torch.nn.Module):
    def __init__(
        self,
        device,
        backbone_name="wide_resnet50_2",
        out_indices=(2, 3),
        checkpoint_path="",
        pool_last=False,
        pretrained=False,
    ):
        super().__init__()
        # Determine if to output features.
        kwargs = {"features_only": True if out_indices else False}
        if out_indices:
            kwargs.update({"out_indices": out_indices})

        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        return features

    def freeze_parameters(self, layers, freeze_bn=False):
        """Freeze resent parameters. The layers which are not indicated in the layers list are freeze."""

        layers = [str(layer) for layer in layers]
        # Freeze first block.
        if "1" not in layers:
            if hasattr(self.backbone, "conv1"):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, "bn1"):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, "layer1"):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False

        # Freeze second block.
        if "2" not in layers:
            if hasattr(self.backbone, "layer2"):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False

        # Freeze third block.
        if "3" not in layers:
            if hasattr(self.backbone, "layer3"):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False

        # Freeze fourth block.
        if "4" not in layers:
            if hasattr(self.backbone, "layer4"):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False

        # Freeze last FC layer.
        if "-1" not in layers:
            if hasattr(self.backbone, "fc"):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False

        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()


def build_3dads_model() -> Model:
    """Build the real 3D-ADS RGB/depth deep feature extractor (wide_resnet50_2 backbone,
    features_only, out_indices=(2,3) -- exactly as used by RGBInetFeatures/DepthInetFeatures)
    at random init for TorchLens tracing."""
    model = Model(
        device="cpu", backbone_name="wide_resnet50_2", out_indices=(2, 3), pretrained=False
    )
    model.eval()
    return model


def example_input_3dads_model() -> torch.Tensor:
    """A single RGB (or depth-rendered-as-RGB) image tile, as fed to the backbone in
    RGBInetFeatures.add_sample_to_mem_bank / DepthInetFeatures.add_sample_to_mem_bank."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("3D-ADS", "build_3dads_model", "example_input_3dads_model", 2023, "vendored-pytorch"),
]
