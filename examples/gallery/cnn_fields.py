"""Geometric fields and input interventions on torchvision ResNet-18.

Run ``python examples/gallery/cnn_fields.py --out DIR``. Rules below are fixed
before measurement. Background: Araujo, Norris and Sim, Distill 2019,
https://doi.org/10.23915/distill.00021. A geometric hull is potential support;
the gradient-support equality in this entry is an additional empirical test.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import _gallery as gallery
import torch
from torch import nn
from torchvision import models

import torchlens as tl

CLAIM = "CNN geometric influence and interventions; Araujo et al., Distill 2019, doi:10.23915/distill.00021"
SEED = 17
SIDE = 224
IOU_MIN = 0.95
PATCH = 4
OUTSIDE_DELTA = 0.0


# region: plain
def advance(field: tuple[int, int, int], module: nn.Module) -> tuple[int, int, int]:
    """Propagate size, jump and first-centre through a convolution or pool.

    Parameters
    ----------
    field, module:
        Incoming geometry and the spatial module.

    Returns
    -------
    tuple
        Receptive size, jump and first-centre in input pixel coordinates.
    """
    size, jump, start = field
    kernel = module.kernel_size
    stride = module.stride
    padding = module.padding
    k = kernel if isinstance(kernel, int) else kernel[0]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    return size + (k - 1) * jump, jump * s, start + ((k - 1) // 2 - p) * jump


def analytic(model: nn.Module, layer: str, *, serial: bool = False) -> tuple[int, int, int]:
    """Compute ResNet geometry independently, or the deliberately wrong leaf chain.

    Parameters
    ----------
    model, layer, serial:
        ResNet, target stage, and whether to treat registered leaves as serial.

    Returns
    -------
    tuple
        Size, jump and first-centre.
    """
    field = advance(advance((1, 1, 0), model.conv1), model.maxpool)
    for stage_index in range(1, int(layer[-1]) + 1):
        for block in getattr(model, f"layer{stage_index}"):
            incoming = field
            field = advance(advance(incoming, block.conv1), block.conv2)
            if block.downsample is not None:
                skip = advance(incoming, block.downsample[0])
                if serial:
                    field = advance(field, block.downsample[0])
                else:
                    assert skip[1:] == field[1:]
                    field = (max(field[0], skip[0]), *field[1:])
    return field


def feature(model: nn.Module, x: torch.Tensor, layer: str) -> torch.Tensor:
    """Extract a module output with a plain hook.

    Parameters
    ----------
    model, x, layer:
        Model, input batch and module address.

    Returns
    -------
    torch.Tensor
        Module output, preserving autograd.
    """
    values: list[torch.Tensor] = []

    def collect(_module: nn.Module, _args: Any, output: torch.Tensor) -> None:
        """Collect one module output."""
        values.append(output)

    hook = model.get_submodule(layer).register_forward_hook(collect)
    try:
        model(x)
    finally:
        hook.remove()
    return values[0]


def gradient_mask(model: nn.Module, x: torch.Tensor, layer: str, unit: tuple) -> torch.Tensor:
    """Measure the spatial gradient support without TorchLens.

    Parameters
    ----------
    model, x, layer, unit:
        Model, input, target module and complete element index.

    Returns
    -------
    torch.Tensor
        Spatial boolean support mask.
    """
    leaf = x.detach().clone().requires_grad_(True)
    grad = torch.autograd.grad(feature(model, leaf, layer)[unit], leaf)[0]
    return grad.ne(0).any(dim=(0, 1))


# endregion


def iou(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return support intersection divided by union.

    Parameters
    ----------
    left, right:
        Boolean masks.

    Returns
    -------
    float
        IoU, or zero when the union is empty.
    """
    union = int((left | right).sum())
    return float((left & right).sum()) / union if union else 0.0


def draw_fields(
    x: torch.Tensor,
    spatial: torch.Tensor,
    empirical: torch.Tensor,
    projected: torch.Tensor,
    changed: torch.Tensor,
    out: Path,
) -> None:
    """Save the field overlay and projective support as a diagnostic figure.

    Parameters
    ----------
    x, spatial, empirical, projected, changed, out:
        Input, geometric and measured masks, and output folder. Single seeded run;
        no uncertainty intervals. Colour limits and stimulus order are fixed.
    """
    import matplotlib.pyplot as plt

    image = x[0].permute(1, 2, 0).detach().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(image)
    axes[0].contour(spatial.numpy(), levels=[0.5], colors=["black"], linewidths=2)
    axes[0].imshow(empirical.numpy(), cmap="Blues", alpha=0.3, vmin=0, vmax=1)
    axes[0].set_title(f"Input field / gradient support\nIoU = {iou(spatial, empirical):.4f}")
    axes[1].imshow(projected.numpy(), cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Projective field\nblack = potential influence")
    axes[2].imshow(changed.numpy(), cmap="Greys", vmin=0, vmax=1)
    axes[2].set_title(f"Changed downstream units\nIoU = {iou(projected, changed):.4f}")
    for axis in axes:
        axis.set_xlabel("Horizontal position (pixels or feature units)")
        axis.set_ylabel("Vertical position")
    fig.suptitle("ResNet-18: geometric and observed influence (single seeded input)")
    fig.savefig(out / "cnn_fields.png", dpi=160)
    fig.savefig(out / "cnn_fields.svg")
    plt.close(fig)


# region: torchlens
def measure(
    model: nn.Module, x: torch.Tensor, layer: str, channel: int, out: Path
) -> gallery.Entry:
    """Measure the fixed oracles and controls and reload the saved trace.

    Parameters
    ----------
    model, x, layer, channel, out:
        Pinned model/input, target, channel and artifact directory.

    Returns
    -------
    gallery.Entry
        Complete measurement record.
    """
    trace = tl.trace(model, x, save="all")
    target = trace[layer]
    source = trace.input_ops[0]
    h, w = target.out.shape[-2:]
    unit = (0, channel, h // 2, w // 2)
    box = target.receptive_field.at(unit[-2:])
    inside = torch.zeros_like(x, dtype=torch.bool)
    inside[box.slices()] = True
    outside = ~inside
    size, jump, start = analytic(model, layer)
    lo = max(0, start + unit[-1] * jump - size // 2)
    hi = min(SIDE, start + unit[-1] * jump + size // 2 + 1)
    expected = torch.zeros((SIDE, SIDE), dtype=torch.bool)
    expected[lo:hi, lo:hi] = True
    spatial = inside.any(dim=(0, 1))
    empirical = gradient_mask(model, x, layer, unit)
    baseline = float(target.out[unit])
    outside_fork = trace.fork()
    outside_fork.do(tl.units(source.label, outside), tl.zero_ablate())
    outside_delta = abs(float(outside_fork[layer].out[unit]) - baseline)
    inside_fork = trace.fork()
    inside_fork.do(tl.units(source.label, inside), tl.zero_ablate())
    inside_delta = abs(float(inside_fork[layer].out[unit]) - baseline)
    patch = torch.zeros_like(inside)
    patch[:, :, SIDE // 2 : SIDE // 2 + PATCH, SIDE // 2 : SIDE // 2 + PATCH] = True
    projected = source.projective_field.of(patch, within=layer)
    projected_mask = next(iter(projected)).mask.any(dim=(0, 1))
    plain_changed = feature(model, x.masked_fill(patch, 0), layer).ne(target.out).any(dim=(0, 1))
    outside_patch = torch.zeros_like(inside)
    outside_patch[:, :, :PATCH, :PATCH] = True
    is_outside = not bool((outside_patch & inside).any())
    control_fork = trace.fork()
    control_fork.do(tl.units(source.label, outside_patch), tl.zero_ablate())
    control_delta = abs(float(control_fork[layer].out[unit]) - baseline)
    serial = analytic(model, layer, serial=True)
    serial_lo = max(0, serial[2] + unit[-1] * serial[1] - serial[0] // 2)
    serial_hi = min(SIDE, serial[2] + unit[-1] * serial[1] + serial[0] // 2 + 1)
    entry = gallery.Entry(
        "cnn_fields",
        CLAIM,
        {
            "seed": SEED,
            "shape": list(x.shape),
            "arch": "resnet18",
            "weights": "IMAGENET1K_V1",
            "mode": "eval",
            "dtype": "float32",
            "layer": layer,
            "channel": channel,
            "unit": list(unit),
        },
    )
    entry.oracles = {
        "analytic_box": gallery.Check(
            "exact clipped box and exact status",
            {
                "expected": [lo, hi],
                "measured": [[a.clipped_start, a.clipped_stop] for a in box.axes[-2:]],
                "status": box.status.value,
            },
            box.exact and torch.equal(spatial, expected),
        ),
        "gradient_support": gallery.Check(
            f"IoU >= {IOU_MIN}", iou(spatial, empirical), iou(spatial, empirical) >= IOU_MIN
        ),
        "outside_ablation": gallery.Check(
            "nonempty outside mask; unit delta == 0 float32",
            {"delta": outside_delta, "pixels": int(outside.any(dim=(0, 1)).sum())},
            bool(outside.any()) and outside_delta == OUTSIDE_DELTA,
        ),
        "inside_ablation": gallery.Check("unit delta > 0", inside_delta, inside_delta > 0),
        "projective_support": gallery.Check(
            f"IoU >= {IOU_MIN}",
            iou(projected_mask, plain_changed),
            iou(projected_mask, plain_changed) >= IOU_MIN,
        ),
    }
    entry.controls = {
        "serial_module_chain": gallery.Check(
            "registered-leaf chain gives a different clipped box",
            {"serial": [serial_lo, serial_hi], "dag": [lo, hi]},
            (serial_lo, serial_hi) != (lo, hi),
        ),
        "same_size_outside_patch": gallery.Check(
            "4x4 corner patch is outside; unit delta == 0",
            {"outside": is_outside, "delta": control_delta},
            is_outside and control_delta == 0,
        ),
    }
    draw_fields(x, spatial, empirical, projected_mask, plain_changed, out)
    trace.annotations["gallery"] = {"layer": layer, "unit": list(unit), "baseline": baseline}
    artifact = out / "cnn_fields.tlspec"
    trace.save(artifact, level="portable", overwrite=True)

    def replay(loaded: Any) -> dict[str, Any]:
        """Replay the outside-field intervention on the reloaded trace."""
        fork = loaded.fork()
        fork.do(tl.units(source.label, outside), tl.zero_ablate())
        delta = abs(float(fork[layer].out[unit]) - baseline)
        return {"matched": delta == OUTSIDE_DELTA, "outside_unit_delta": delta}

    entry.artifact = gallery.artifact_record(artifact, replay)
    entry.verbs = ("capture", "select", "read", "do", "compare", "save")
    entry.architectures = ("ResNet-18",)
    return entry


# endregion


def main() -> None:
    """Run the gallery entry and print its stamped verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--quick", action="store_true", help="same headline input; no extra architectures"
    )
    parser.add_argument(
        "--layer", choices=("layer1", "layer2", "layer3", "layer4"), default="layer3"
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--arch", choices=("resnet18",), default="resnet18")
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    x = torch.randn(1, 3, SIDE, SIDE)
    try:
        entry = measure(model, x, args.layer, args.channel, args.out)
    except Exception as exc:
        traceback.print_exc()
        entry = gallery.Entry(
            "cnn_fields",
            CLAIM,
            {
                "seed": SEED,
                "arch": args.arch,
                "weights": "IMAGENET1K_V1",
                "layer": args.layer,
                "channel": args.channel,
            },
        )
        entry.oracles["execution"] = gallery.Check(
            "measurement completes without a capture gap",
            {"type": type(exc).__name__, "message": str(exc)},
            False,
        )
    entry.lines = {
        name: gallery.count_region_lines(Path(__file__), name) for name in ("torchlens", "plain")
    }
    entry.lines["reference"] = "not measured"
    gallery.report(entry, args.out, "cpu", started)


if __name__ == "__main__":
    main()
