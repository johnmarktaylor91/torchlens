# TorchLens quickstart: the input ladder

Every TorchLens entry verb -- `tl.trace`, `tl.summary`, and the one-call
render facade -- accepts its input the same three ways. Learn one call shape
and you own every surface.

## The three rungs

```python
import torch, torchvision.models as models, torchlens as tl

model = models.resnet18(weights="IMAGENET1K_V1").eval()

log = tl.trace(model, x)                             # rung 1: your real input (best)
log = tl.trace(model, input_size=(1, 3, 224, 224))   # rung 2: your shape, random values; disclosed
log = tl.trace(model)                                # rung 3: inferred shape + random values; disclosed, or a teach
log = tl.trace(lm, "The quick brown fox")            # HF models: a string IS a real input
```

The rungs are exclusive by law: a real input XOR `input_size=` XOR nothing.
Mixing them refuses with a typed error (`input_rung_conflict`) before any
forward runs. A failed explicit size never falls through to inference, and a
failed inference never falls through to a stock tensor -- TorchLens does not
guess twice.

**Rung 1 (gold).** Caller-authoritative: TorchLens guesses nothing. "Gold"
means the tool did not synthesize anything -- it does not certify that your
data is meaningful. For Hugging Face models, a prompt string is a gold
input: it is tokenized with the model's own tokenizer, and the tokenizer
provenance is recorded on the trace.

**Rung 2 (`input_size=`).** One flat positive-integer tuple is one input
shape (batch included exactly as written). A sequence of shape tuples is
several positional tensors. A mapping binds forward keyword names to
shapes -- this is how multi-input models are reachable:

```python
log = tl.trace(clip, input_size={
    "input_ids": (1, 16),
    "pixel_values": (1, 3, 224, 224),
    "attention_mask": (1, 16),
})
```

Values are synthesized from a **local seed-0 generator** (your global RNG
streams are never advanced, and the model is never moved). Dtype and value
recipes are fail-closed static facts: an unambiguous embedding entry
permits vocab-bounded integer ids, an unambiguous float entry permits
uniform `[0, 1)` values, and ambiguity refuses with a teach naming
`torchlens.quickstart.InputSpec(shape, dtype=..., low=..., high=...)` as
the override. TorchLens never discovers dtypes by running failed forwards.

**Rung 3 (nothing).** Shape inference probes the model (bounded, disclosed
in a probe diary) and the surface consumes the exact verified capture --
the cost is "N probes + 1 verified capture", never a second capture after
verification. When inference cannot help, the refusal teaches the exact
next command, in order: pass the real call; pass `input_size=` (including
the mapping form); call `torchlens.debug.infer_input_shape` directly with
hints. For Hugging Face models the refusal says the better answer first: a
real prompt string.

## What synthesized inputs mean (and do not mean)

> Successful inference proves that this exact synthesized call ran under the
> stated execution policy and its capture verified. It does not prove
> canonical preprocessing, representative values, a stable graph for other
> data, or another input's control-flow path.

Synthesized values can change the **graph**, not just the numbers:
same-shaped inputs with different values may take different control-flow
paths (measured: detection models differ by several ops between two random
inputs). Every synthesized-input trace therefore carries a persistent
provenance record -- origin, shapes, dtypes, recipe, seed, strategy, probe
diary -- that survives `tl.save`/`tl.load`, so a synthesized trace can never
masquerade as a real one after a handoff.

Two guardrails follow from the record:

- **Derived-semantics claims refuse.** Decoded labels and top-k tables
  (`log.decode_output()`, `log.output_table()`) raise a typed refusal
  (`nongold_semantics_unavailable`) on a synthesized-value trace: they would
  be statements about random numbers presented as statements about data.
  Structure, shapes, parameter facts, and measured costs all stay
  first-class.
- **The first raw value read warns once.** Reading `log[...].out` on a
  synthesized-value trace warns once per trace object
  (`nongold_raw_value_read`) -- the numbers are real measurements of a
  forward pass over random data. Filter the warning by its code to
  acknowledge it.

## The one-call render facade

```python
from torchlens.user_funcs import render   # tl.render lands with the surface sweep

result = render(model, x)                            # your real input
result = render(model, input_size=(1, 3, 224, 224))  # declared shape
result = render(model)                               # inferred
```

`render` runs ONE metadata-only capture pinned to eval + no-grad -- training
flags, RNG state, and normalization buffers are restored afterwards and the
restoration is verified with a state-dict hash (`result.receipt`) -- then
renders through the existing renderer with `collapse="auto"` as its default.
This facade is the only place auto-collapse is a default: `Trace.draw()`
keeps `collapse="none"`.

The result is detached (`RenderResult`): it owns the DOT source, the
rendered bytes or written path, the input provenance, and `save(path)`
re-renders without retaining the capture. In notebooks it displays inline
and writes nothing; in a plain script a bare call writes a collision-safe
`<ModelClass>-graph.pdf` and prints the path (never a shared filename, never
a silent overwrite). No surface ever auto-opens a viewer; pass `view=True`
to opt in. Pass `return_trace=True` to keep the underlying trace (on the
inferred rung this is the exact verified capture); cleanup then transfers to
`result.close()` or a `with` block.

Curated dials: `theme=`, `collapse=`, `module=`, `depth=`, `file=`,
`format=`, `view=`, `return_trace=`. Everything else passes through to
`Trace.draw`, and a collision with a curated dial refuses typed
(`render_kwarg_collision`) -- for full renderer control, drop to
`tl.trace(...).draw(...)`.

## Lazy modules

A model holding un-materialized lazy parameters (`nn.LazyLinear` and
friends) works on the gold and declared rungs: the parameters materialize
during the ONE captured forward -- exactly as they would on a direct call --
and the inventory, totals, and metadata invariants report the real
post-materialization geometry. The zero-argument rung refuses BEFORE
probing (a lazy module accepts any width, so a probe would silently install
its guess in your model), and intervention-ready captures refuse typed
(`state_baseline_unavailable`) because a pending parameter has no bytes to
witness. The taught two-line remedy always works:

```python
with torch.no_grad():
    model(x)   # materialize outside capture, then re-run the verb
```

Un-materialized lazy BUFFERS (`nn.LazyBatchNorm2d` before any forward)
refuse at entry with the same teach (`lazy_uninitialized`).

## Execution policy per surface

`tl.summary` and `render` pin eval + no-grad and restore everything they
touch (verified, disclosed). `tl.trace` is the power surface: it keeps YOUR
modes -- a train-mode capture really does update BatchNorm running
statistics, and TorchLens warns once per process
(`batchnorm_train_stats_mutated`) naming `model.eval()`. The zero-argument
trace is necessarily the eval-mode trace that inference verified; asking for
incompatible capture options recaptures from the inferred input with your
options instead (disclosed in the provenance record).
