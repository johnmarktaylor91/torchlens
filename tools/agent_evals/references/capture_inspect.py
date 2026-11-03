"""Reference solution: capture and inspect.

Docs used: docs/for-ai-agents.md (Predicate language; Common recipes), docs/reference/glossary.md
(Predicate: the save= list form and 'all'; trace.ops counting).
"""

import json

from model import build_model, make_input

import torchlens as tl

trace = tl.trace(build_model(), make_input(), save="all")
relu_sites = trace.find_sites(tl.func("relu"))
shapes = [list(site.out.shape) for site in relu_sites]
output = trace[trace.output_layers[0]].out

with open("result.json", "w", encoding="utf-8") as handle:
    json.dump(
        {"relu_count": len(shapes), "relu_shapes": shapes, "output_sum": float(output.sum())},
        handle,
    )
