"""Reference solution: save and load.

Docs used: docs/for-ai-agents.md (The agent surface: trace.save(path) then point the tools at
the artifact), docs/reference/error_refusal_contract.md (agent_artifact_unreadable remedy names
trace.save(path)), docs/reference/glossary.md (trace.ops counting: trace.num_ops).
"""

import json

from model import build_model, make_input

import torchlens as tl

trace = tl.trace(build_model(), make_input(), save="all")
trace.save("run.tlspec")
loaded = tl.load("run.tlspec")
first_relu = loaded.find_sites(tl.func("relu")).first()

with open("result.json", "w", encoding="utf-8") as handle:
    json.dump({"num_ops": loaded.num_ops, "first_relu_sum": float(first_relu.out.sum())}, handle)
