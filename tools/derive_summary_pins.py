"""THE summary pin-derivation script (F08; summary memo section 4, binding).

Numbers discipline: every count and total in the summary memo's tables is
RE-DERIVED at build time by this ONE committed script against pinned
versions -- the panel's tables establish structure, never byte-final
constants. The derivation basis is the IDENTITY PARTITION (via C02's
FactCore/compute face), never a label sweep.

Zero-network by construction: HF architectures are CONFIG-BUILT (a default
``GPT2Config()`` IS the gpt2-124M architecture; parameter counts, tie
topology, FLOPs, and ladder row counts are architecture facts independent
of weight values) and torchvision models load ``weights=None``.

Run:  python tools/derive_summary_pins.py [model ...]
Emits a JSON object per model on stdout; the F08 pin tests import
``derive_pins`` directly so the tests and this script share one basis.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import torch


def _summary_facts(model: torch.nn.Module, example: Any) -> dict[str, Any]:
    """Capture once and derive every pinned fact from the one substrate."""

    import torchlens as tl
    from torchlens.report._factcore import factcore
    from torchlens.report._summary_ladder import _ModelIndex, build_hybrid, build_tree, resolve_view

    trace = tl.trace(model, example)
    try:
        core = factcore(trace)
        index = _ModelIndex(trace)
        ladder = {depth: len(build_tree(index, depth)) for depth in range(1, index.max_depth + 1)}
        view = resolve_view(trace)
        return {
            "params_declared": core.params.total,
            "params_per_path": core.params.per_path_total,
            "params_executed": core.params.executed,
            "params_unexecuted": core.params.unexecuted,
            "tied_groups": [list(group) for group in core.params.tied_groups],
            "flops_forward_fma2": int(core.compute.partition_total),
            "macs_forward": int(core.compute.macs_total),
            "compute_ops": core.counts.compute_ops,
            "hybrid_rows": len(build_hybrid(index)),
            "tree_rows_by_depth": ladder,
            "default_rung": view.rung,
            "default_depth": view.depth,
            "default_rows": view.body_row_count,
            "default_folds": view.fold_groups,
            "disclosure": view.disclosure,
        }
    finally:
        trace.cleanup()
        tl.release_model(model)


def _gpt2() -> dict[str, Any]:
    """gpt2-124M architecture, config-built, PINNED 16-token input."""

    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config()).eval()
    return _summary_facts(model, torch.arange(16, dtype=torch.long).unsqueeze(0))


def _bert_base() -> dict[str, Any]:
    """bert-base architecture, config-built, 12-token input."""

    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    model = BertModel(BertConfig()).eval()
    return _summary_facts(model, torch.arange(12, dtype=torch.long).unsqueeze(0))


def _torchvision(name: str, size: int = 224) -> Callable[[], dict[str, Any]]:
    """A torchvision derivation thunk (weights=None, eval, one input)."""

    def derive() -> dict[str, Any]:
        """Build and derive one torchvision architecture."""

        from torchvision import models

        torch.manual_seed(0)
        model = getattr(models, name)(weights=None).eval()
        return _summary_facts(model, torch.randn(1, 3, size, size))

    return derive


#: The derivation catalog (subset of the memo's 18-model gate that derives
#: at ZERO network; pretrained-checkpoint rows join via the R1 venue).
CATALOG: dict[str, Callable[[], dict[str, Any]]] = {
    "gpt2": _gpt2,
    "bert-base": _bert_base,
    "resnet18": _torchvision("resnet18"),
    "resnet50": _torchvision("resnet50"),
    "vgg16": _torchvision("vgg16"),
    "vgg19": _torchvision("vgg19"),
    "densenet121": _torchvision("densenet121"),
    "mobilenet_v3_large": _torchvision("mobilenet_v3_large"),
    "inception_v3": _torchvision("inception_v3", size=299),
}


def derive_pins(name: str) -> dict[str, Any]:
    """Derive the pin facts for one catalog model."""

    if name not in CATALOG:
        raise KeyError(f"unknown pin model {name!r}; catalog: {sorted(CATALOG)}")
    return CATALOG[name]()


def main(argv: list[str]) -> int:
    """Derive and print pins for the requested (or all) catalog models."""

    names = argv or sorted(CATALOG)
    result: dict[str, Any] = {}
    for name in names:
        result[name] = derive_pins(name)
        print(f"== {name}", file=sys.stderr)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
