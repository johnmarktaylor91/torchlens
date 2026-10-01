# Weights-free capture (structure-only on the meta device)

*Every spelling on this page is DOCUMENTED-UNSTABLE pending the naming
sprint; the semantics are the contract.*

TorchLens admits meta-device models under the structure-only contract
(decision point D8, granted 2026-08-26): an 855-byte config file plus a
laptop yields the complete op graph, module nesting, shape/dtype hypotheses,
parameter geometry, and a resolvable intervention-site table for a
Llama-3.1-8B-class model — in seconds, on CPU, with **no tensor values
anywhere**.

Wording discipline: this is **zero tensor-payload storage and no real
forward** — never "zero cost". Construction, Python records, CPU time, and
RSS are real (measured: ~8-15 s and under 1 GB for 7B-class transformers;
one op-dense hybrid architecture measured >400 s — there is no "any 7B in
seconds" promise).

## The four faces

```python
import torch
import torchlens as tl
from torchlens.options import CaptureOptions
from transformers import AutoConfig, AutoModelForCausalLM

# Face 1 — the explicit power path (the only face that takes tl.trace):
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
ids = torch.zeros(1, 8, dtype=torch.int64, device="meta")
t = tl.trace(model, input_kwargs={"input_ids": ids},
             capture=CaptureOptions(structure_only=True))

# Face 2 — the zero-payload-storage summary rung (quickstart ladder rung 4):
tl.summary(model, input_size=(1, 8))   # auto-selects the same contract; the
                                       # auto-set is recorded in provenance

# Face 3 — drawing (same capture, existing render stack, banner mandatory):
t.draw()

# Face 4 — the audit-only plan check (nnsight-scan parity; executable=false):
report = t.check_plan([("relu_1_2", torch.empty(1, 8, 4096, device="meta"))])
```

## Admission (the entry matrix)

Meta state/inputs are admitted **if and only if structure-only is in
force**, and only for a UNIFORM substrate: every input tensor leaf meta AND
every registered parameter/buffer meta. Mixed cells refuse typed in both
directions (`structure_only_substrate_mismatch`, sides named); a REAL tensor
minted mid-forward (a stale pre-wrap factory reference, a `device="cpu"`
literal) refuses through the same family at your source line. Meta without
the flag keeps the historical refusal. Fake/functional/symbolic/sparse/
quantized/DTensor state stays refused; failed captures restore all state.

## What the gate proves — and what it cannot

The entry gate proves substrate coherence, **never shape truth**. A meta
kernel that lies about shapes is not gate-detectable BY DESIGN: every
value-bearing claim is born a HYPOTHESIS, and only discharge promotes it.

## Corroborating a weights-free trace (the discharge recipe)

1. **Build BOTH twins before capturing either** (construction order matters:
   whichever model is built before TorchLens's first wrap holds pre-wrap
   function references, and the preflight refuses the pair typed). Same
   config fingerprint/revision, same torch/transformers versions, `eval()`
   on both (`from_config` defaults to TRAIN; dropout alone refutes), same
   input shape/dtype/`use_cache`/mask policy.
2. Capture the meta twin structure-only; explore, draw, plan, save under the
   HYPOTHESIS banner.
3. Later, on real hardware: an ordinary COMPLETE capture of the real twin.
4. `d = structure_trace.discharge_against(real_trace)`. The comparable-twins
   preflight refuses typed on any mismatch (refuse is not refute — nothing
   registers); equal digests license the positional per-claim join.
5. CORROBORATED promotes (only this call may); REFUTED names the first
   contradiction structurally (record counts and positions before any
   "digests differ" fallback) and hypothesis consumers refuse.

Measured at PR scale: distilgpt2 real-weights vs meta twin — 291/291
records, digests equal, 873 claims CORROBORATED; BERT config twin — 307/307,
921 claims; resnet50 — 389/389 with all 106 declared BatchNorm buffer writes
present and sourced; Llama-2-7B geometry — 6,738,415,616 declared parameters
exactly, 225 linear records (7 per decoder layer x 32 + head).

## The HF on-ramp order

1. **Config-only construction under `torch.device("meta")`** (shown above):
   needs one JSON file and no optional dependency. Capture AFTER leaving the
   construction context.
2. `accelerate.init_empty_weights()` — covered by the optional on-ramp row
   (`tests/test_weightsfree_accelerate.py`; needs the accelerate extra).
3. Literal `from_pretrained(..., device_map="meta")` — advertised only after
   its offline integration row passes (verification queue).

Passing a tokenizer's `attention_mask` refuses typed at the third-party
value branch (the mask, not the architecture, is what refuses); the
input-IDs-only spelling works. The text bridge does not synthesize
mask-omitted inputs (FORK F1, branch A).

## The evidence envelope

Every structure-only capture carries ONE immutable envelope
(`Trace.structure_evidence`, persisted, load-validated fail-closed):
capture mode, substrate, `values_available: false`, outcome, factory-device
policy, ambient-mode provenance, wrap generation, input plan, and the claim
table. Every surface — repr, summary, slices, draw captions, exports, agent
JSON, MCP — renders the claim ladder from it: HYPOTHESIS says shapes are
unproven; CORROBORATED names the discharge and still says no payloads exist;
REFUTED is visually stronger and hypothesis-tolerant consumers refuse.
Loaded traces are HYPOTHESIS again: the discharge registry is session-only.

Measurement-shaped output refuses
(`structure_only_measurements_unsupported`): timings and allocator peaks are
measurements a value-free capture never made. Geometry bytes and FLOPs
render as explicit hypothesis estimates — `2.5 KB (estimated)` is true and
useful where `0 B` would be false.

## Trust sentence

A weights-free trace tells you exactly what it knows and exactly how to find
out the rest: structure is observed, values are absent, shapes are
hypotheses until one real capture of the same twin corroborates them — and
the artifact remembers which it is.

## Acknowledgments

torchview (mert-kurttutan) made zero-memory meta-device drawing a familiar
expectation (`model.to("meta")` is their mechanism and their trap — TorchLens
never moves your model; a real loaded model asking for a weights-free
drawing refuses and teaches config-only construction). nnsight / NDIF
pioneered the scan-before-execute posture; our plan check persists what
their scan evaporates, with typed claim status until discharge. Hugging Face
Accelerate's `init_empty_weights()` and the community's meta-initialization
conventions are consumed, not competed with. calflops showed the value of
config-based weights-free estimation.

Full capability rows: `docs/reference/structure_only_capabilities.md`.
