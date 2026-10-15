"""Reference solution: a documented refusal.

Docs used: docs/quickstart.md (The three rungs: mixing a real input and input_size= refuses
input_rung_conflict), docs/reference/error_refusal_contract.md (branch on exc.fields["code"];
every refusal carries exc.fields["remedy"]).
"""

import json

from model import build_model, make_input

import torchlens as tl

try:
    tl.trace(build_model(), make_input(), input_size=(4, 6))
except tl.errors.TorchLensError as exc:
    record = {
        "code": exc.fields["code"],
        "remedy": exc.fields["remedy"],
        "exception_type": type(exc).__name__,
    }
else:
    record = {"code": None, "remedy": None, "exception_type": None}

with open("result.json", "w", encoding="utf-8") as handle:
    json.dump(record, handle)
