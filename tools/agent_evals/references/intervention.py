"""Reference solution: a capture-time intervention.

Docs used: docs/for-ai-agents.md (Common recipes: "Run a capture-time intervention" with
tl.when(..., tl.zero_ablate()); tl.module(...) site selector), docs/reference/glossary.md
(Site selectors).
"""

import json

from model import build_model, make_input

import torchlens as tl

trace = tl.trace(
    build_model(),
    make_input(),
    save="all",
    intervene=tl.when(tl.module("act1"), tl.zero_ablate()),
)
output = trace[trace.output_layers[0]].out

with open("result.json", "w", encoding="utf-8") as handle:
    json.dump({"output": output.tolist()}, handle)
