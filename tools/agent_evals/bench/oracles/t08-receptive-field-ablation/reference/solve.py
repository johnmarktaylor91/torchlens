import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

def load(inputs):
    inputs = Path(inputs)
    cfg = json.loads((inputs / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("task_model", inputs / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module

def geometry(unit, ignore_padding=False):
    from torch.nn import functional as F
    # Inverse connectivity via transposed convolutions with positive kernels.
    stages = [("relu1", 64, 64, 5, 1, 2, 1), ("pool1", 64, 32, 2, 2, 0, 1),
              ("relu2", 32, 16, 3, 2, 1, 1), ("relu3", 16, 16, 3, 1, 2, 2)]
    stop = next(i for i, stage in enumerate(stages) if stage[0] == unit["layer"])
    size = stages[stop][2]
    mask = torch.zeros(1, 1, size, size)
    mask[0, 0, unit["row"], unit["column"]] = 1
    for _, ins, outs, kernel, stride, padding, dilation in reversed(stages[:stop + 1]):
        extra = ins - ((outs - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1)
        p = 0 if ignore_padding else padding
        mask = F.conv_transpose2d(mask, torch.ones(1, 1, kernel, kernel), stride=stride,
                                  padding=p, dilation=dilation, output_padding=extra)
        mask = mask[:, :, :ins, :ins]
        mask = F.pad(mask, (0, ins-mask.shape[-1], 0, ins-mask.shape[-2]))
        mask = (mask > 0).float()
    return mask[0, 0].numpy().astype(bool)


def run(inputs, outputs, variant=None):
    cfg, module = load(inputs)
    m = module.build_model(cfg["weight_seed"])
    x = torch.from_numpy(module.make_inputs(cfg["stimulus_seed"])).requires_grad_(True)
    logits, acts = m(x, return_intermediates=True)
    baseline = logits[:, 0].detach()
    geometric, gradients, effects = [], [], []
    for unit in cfg["units"]:
        field = geometry(unit, variant == "padding_ignored")
        selected = acts[unit["layer"]][:, unit["channel"], unit["row"], unit["column"]]
        if variant == "single_input":
            selected = selected[:1]
        grad = torch.autograd.grad(selected.sum(), x, retain_graph=True)[0]
        if variant == "single_input":
            grad = grad[:1]
        gradients.append((grad.abs().mean(0)[0] > cfg["gradient_threshold"]).numpy())
        geometric.append(field)
        with torch.no_grad():
            if variant == "unit_ablation":
                def zero_unit(mod, args, out):
                    result = out.clone()
                    result[:, unit["channel"], unit["row"], unit["column"]] = 0
                    return result
                handle = getattr(m, unit["layer"]).register_forward_hook(zero_unit)
                try:
                    changed = m(x)[:, 0]
                finally:
                    handle.remove()
            else:
                ablated = x.detach().clone()
                ablated[:, :, torch.from_numpy(field)] = 0
                changed = m(ablated)[:, 0]
            effects.append((changed - baseline).mean().item())
    geo, grad = np.stack(geometric), np.stack(gradients)
    effect = np.array(effects, dtype=np.float32)
    if variant == "transpose":
        geo, grad = geo.transpose(0, 2, 1), grad.transpose(0, 2, 1)
    if variant == "stimulus_order":
        geo, grad, effect = geo[::-1], grad[::-1], effect[::-1]
    outputs = Path(outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez(outputs / "geometric.npz", masks=geo)
    np.savez(outputs / "gradient.npz", masks=grad)
    np.savez(outputs / "ablation.npz", effects=effect)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
