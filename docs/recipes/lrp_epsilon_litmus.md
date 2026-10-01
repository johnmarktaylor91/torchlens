# The epsilon-LRP litmus recipe

TorchLens deliberately ships **no LRP rule registry, no composites, no
canonizers, and no architecture support table** (ruled twice). Layer-wise
Relevance Propagation is a family of research rules whose maintained catalogs
belong to the LRP ecosystem; what TorchLens supplies is the **mechanism** —
the forward/backward pairing store `tl.attribution.SiteStash` — and this
recipe, written cleanroom from the published equations (Bach et al.;
Montavon et al.; the LRP-0 == input-x-gradient equivalence is Ancona et al.).
The recipe is a *litmus*: it runs epsilon-LRP end to end on real
architectures, emits a per-site conservation ledger, and refuses honestly at
the exact op where its rules stop.

The code below is transcribed verbatim into
`tests/test_attrib_methods_lrp.py`, where the blocking rows run it per
commit; keep the two in lockstep.

## The convention

Relevance rides the backward pass as `c` with `R = value * c` at every
tensor. Under this convention the **native backward already implements the
standard LRP treatment** for ReLU (relevance passes through active units),
max pooling (winner-take-all), average pooling (uniform split), reshape /
flatten, dropout in eval mode, and **concatenation — a concat splits
relevance by slice with no free parameter** (one extra HONEST rule, not a
hidden opinion). Hooks are therefore mounted ONLY on the parameterized
`Linear` / `Conv2d` sites, where the epsilon rule replaces the gradient:

```
s     = R_out / (z + epsilon * sign(z))        # sign(0) = +1
c_in  = (dz/dx)^T s                            # one autograd re-derivation
R_in  = x * c_in
```

Bias and epsilon **absorb** relevance; the ledger discloses the per-site
absorption rather than calling it exact conservation.

## The recipe

```python
import torch
from torch import Tensor, nn
from torchlens.attribution import AttributionError, SiteStash
from typing import Any


def epsilon_lrp(
    model: nn.Module,
    inputs: Tensor,
    target_index: int,
    epsilon: float = 1e-6,
) -> tuple[Tensor, list[dict[str, Any]], dict[str, int]]:
    """Cleanroom epsilon-LRP through paired forward/backward hooks."""

    stash = SiteStash()
    ledger: list[dict[str, Any]] = []
    handles = []
    in_backward = {"flag": False}

    def make_forward_hook(site):
        def forward_hook(module, args, output):
            if in_backward["flag"]:
                return  # the backward-rule recompute must not restash
            if len(args) != 1 or not isinstance(args[0], Tensor):
                raise AttributionError(
                    f"site {stash.label_of(module)!r} fired with an "
                    "unexpected argument tuple; the epsilon rule covers "
                    "single-tensor-input Linear/Conv2d sites"
                )
            stash.mark_firing(module)
            stash.stash(module, "input", args[0].detach())
            stash.stash(module, "output", output.detach())
        return forward_hook

    def make_backward_hook(site):
        def backward_hook(module, grad_input, grad_output):
            if in_backward["flag"]:
                # The rule's own recompute backward passes through natively:
                # the autograd trick WANTS dz/dx with the substituted s.
                return None
            # Every gradient tuple slot is explicitly guarded: the recipe
            # never leans on a selector distinction.
            if len(grad_output) != 1 or grad_output[0] is None:
                raise AttributionError(
                    f"site {stash.label_of(module)!r} received a "
                    f"{len(grad_output)}-slot grad_output tuple; the recipe "
                    "expects exactly one used output"
                )
            if len(grad_input) != 1:
                raise AttributionError(
                    f"site {stash.label_of(module)!r} received a "
                    f"{len(grad_input)}-slot grad_input tuple; the recipe "
                    "expects exactly one tensor input"
                )
            x = stash.fetch(module, "input")
            z_saved = stash.fetch(module, "output")
            c_out = grad_output[0]
            in_backward["flag"] = True
            try:
                with torch.enable_grad():
                    x_live = x.clone().requires_grad_(True)
                    z = module(x_live)
                    stabilizer = epsilon * torch.where(
                        z.detach() >= 0,
                        torch.ones_like(z.detach()),
                        -torch.ones_like(z.detach()),
                    )
                    s = (z_saved * c_out) / (z_saved + stabilizer)
                    c_in = torch.autograd.grad(z, x_live, grad_outputs=s.detach())[0]
            finally:
                in_backward["flag"] = False
            output_relevance = float((z_saved * c_out).sum())
            input_relevance = float((x * c_in).sum())
            delta = input_relevance - output_relevance
            ledger.append(
                {
                    "site": stash.label_of(module),
                    "rule": "epsilon",
                    "grad_slots": len(grad_input),
                    "input_relevance_sum": input_relevance,
                    "output_relevance_sum": output_relevance,
                    "conservation_delta_abs": abs(delta),
                    "conservation_delta_rel": abs(delta)
                    / max(abs(output_relevance), 1e-30),
                    "dtype": str(z_saved.dtype),
                    "device": str(z_saved.device),
                    "nonfinite": not bool(torch.isfinite(c_in).all()),
                }
            )
            return (c_in,)
        return backward_hook

    covered = (nn.Linear, nn.Conv2d)
    for label, module in model.named_modules():
        if list(module.children()):
            continue
        if any(True for _ in module.parameters(recurse=False)) and not isinstance(
            module, covered
        ):
            raise AttributionError(
                f"epsilon-LRP litmus covers Linear and Conv2d sites only; "
                f"uncovered parameterized op at {label!r} "
                f"({type(module).__name__}). Generic identity is forbidden: "
                "write an explicit rule or stop here"
            )
        if isinstance(module, covered):
            stash.register_site(module, label)
            handles.append(module.register_forward_hook(make_forward_hook(module)))
            handles.append(module.register_full_backward_hook(make_backward_hook(module)))
    # In-place activations (nn.ReLU(inplace=True) and friends) would mutate
    # the hook-wrapped site outputs, which autograd forbids; flipping the
    # flag is semantics-preserving (same values, out of place) and restored.
    inplace_flags = [
        module for module in model.modules() if getattr(module, "inplace", False) is True
    ]
    try:
        for module in inplace_flags:
            module.inplace = False
        leaf = inputs.detach().clone().requires_grad_(True)
        model.eval()
        output = model(leaf)
        seed = torch.zeros_like(output)
        seed[..., target_index] = 1.0
        output.backward(gradient=seed)
        relevance = (leaf * leaf.grad).detach()
    finally:
        for handle in handles:
            handle.remove()
        for module in inplace_flags:
            module.inplace = True
    return relevance, ledger, stash.leftovers()
```

## What each ledger row records

Site label and firing pairing (via `SiteStash`), the rule, the gradient
tuple slot count (explicitly guarded), the input/output relevance sums, the
absolute and relative conservation delta (epsilon + bias **absorption**,
disclosed), dtype/device, and a nonfinite flag. Stashes never fetched are
returned as `leftovers` — disclosed, never silently dropped.

## Subjects, tiered by size

- **Blocking, no meaningful cost:** the bias-free ReLU self-oracle
  (LRP-0 == input × gradient, exact conservation — the Ancona et al.
  equivalence, measured 3.7e-09 by the design panel); the captum
  `LRP(EpsilonRule)` shared-MLP exact row (in the captum oracle ledger); the
  FULL `squeezenet1_1` ledger (~5 MB — smaller than several test fixtures —
  zero BatchNorm, no residual add, exactly one concat per Fire module;
  fetched once, stated openly, even in blocking CI).
- **Periodic / download tier:** `vgg16` (~553 MB) as the canonical rendered
  figure — image, relevance heatmap, ledger. The recipe is not called
  complete until that row has run in the periodic job.
- **Disclosed frontier, asserted nowhere:** a `resnet18` residual block. The
  residual add is exactly where LRP becomes architecture-opinionated — how
  relevance splits across a skip connection is a modeling CHOICE
  (composites, canonizers), and that doorway stays closed here. Shown so
  nobody mistakes silence for coverage.

## Friction record (what this recipe surfaced, honestly)

1. Full backward hooks forbid downstream in-place mutation of hooked
   outputs, so real torchvision models (`nn.ReLU(inplace=True)`) need the
   documented inplace-flag flip.
2. The rule's autograd re-derivation re-enters the site's own hooks; both
   hooks carry an explicit re-entrancy guard.
3. Gradient tuple slots are guarded explicitly at every site — the recipe
   must not lean on any engine selector distinction while the engine-defect
   conversation (grad_output fire labels, `grad_kind` vocabulary, the
   `wrt=`/`param_grad` slot filter) is resolved by its owning lane.
4. Relevance seeding is a one-hot on the LOGITS; seeding a probability
   would silently change the question.
5. `sign(0)` must be pinned (+1 here) or dead units divide by zero.
6. Conservation drift compounds across depth in float32; the ledger
   discloses per-site deltas so the reader sees WHERE absorption happens
   instead of one end-to-end number hiding it.

## The closure principle

TorchLens supplies the mechanism (`SiteStash`, the hook discipline, the
ledger shape); the epsilon rule above is the whole shipped catalog, and it
refuses at the exact op label where it stops. Further relevance rules
(alpha-beta, z-plus, gamma), composites, and canonizers are the maintained
territory of dedicated LRP libraries — port this recipe's discipline there
rather than asking TorchLens to referee research rules.
