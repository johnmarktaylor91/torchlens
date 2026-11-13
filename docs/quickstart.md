# TorchLens quickstart: three ways to give an input

Every TorchLens entry function -- `tl.trace`, `tl.summary`, and the one-call
picture `tl.draw` -- accepts its input the same three ways. Learn one call
shape and it works everywhere.

## The three ways

```python
import torch, torchvision.models as models, torchlens as tl

model = models.resnet18(weights="IMAGENET1K_V1").eval()
x = torch.rand(1, 3, 224, 224)  # stands in for your real, preprocessed batch

log = tl.trace(model, x)                             # 1. your real input (best)
log = tl.trace(model, input_shape=(1, 3, 224, 224))   # 2. your shape, random values; disclosed
log = tl.trace(model)                                # 3. inferred shape + random values; disclosed,
                                                     #    or an error that names the fix
log = tl.trace(lm, "The quick brown fox")            # HF models: a string IS a real input
# (`lm` here is any Hugging Face language model you have loaded.)
```

The three are exclusive: a real input, or `input_shape=`, or nothing. Passing
a real input together with `input_shape=` raises a typed error
(`input_rung_conflict`) before any forward runs. A failed explicit size never
falls through to inference, and a failed inference never falls through to a
stock tensor -- TorchLens does not guess twice.

**1. A real input.** You supply it, so TorchLens guesses nothing. This
means only that the tool synthesized nothing -- it does not certify that your
data is meaningful. For Hugging Face models, a prompt string is a real
input: it is tokenized with the model's own tokenizer, and the tokenizer
provenance is recorded on the trace. The same string (or PIL image, or a
modality dict such as `{"text": ..., "image": ...}`) is a real input wherever
a model input is taken: `trace.run`, `tl.record`, `tl.recording.dry_run`,
`tl.validation.validate_backward_pass` and the `tl.debug` helpers resolve it exactly as
`tl.trace` does.

**2. A declared shape (`input_shape=`).** One flat positive-integer tuple is one input
shape (batch included exactly as written). A sequence of shape tuples is
several positional tensors. A mapping binds forward keyword names to
shapes -- this is how multi-input models are reachable:

```python
log = tl.trace(clip, input_shape={
    "input_ids": (1, 16),
    "pixel_values": (1, 3, 224, 224),
    "attention_mask": (1, 16),
})
```

Values are synthesized from a **local seed-0 generator** (your global RNG
streams are never advanced, and the model is never moved). Dtype and value
recipes are fail-closed static facts: an unambiguous embedding entry
permits vocab-bounded integer ids, an unambiguous float entry permits
uniform `[0, 1)` values, and an ambiguous entry raises an error that names
`torchlens.types.InputSpec(shape, dtype=..., low=..., high=...)` as
the override. TorchLens never discovers dtypes by running failed forwards.

**3. No input (shape inferred).** Shape inference probes the model (bounded,
disclosed in a probe diary) and the call uses the exact verified capture --
the cost is "N probes + 1 verified capture", never a second capture after
verification. When inference cannot help, the error names the next command
to try, in order: pass the real call; pass `input_shape=` (including the
mapping form); call `torchlens.debug.guess_input_shape` directly with
hints. For Hugging Face models the error gives the better answer first: a
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
diary -- that survives `Trace.save`/`tl.load`, so a synthesized trace can never
masquerade as a real one after a handoff.

Two guardrails follow from the record:

- **Derived-meaning reads raise.** Decoded labels and top-k tables
  (`log.decode_output()`, `log.output_table()`) raise a typed error
  (`nongold_semantics_unavailable`) on a synthesized-value trace: they would
  be statements about random numbers presented as statements about data.
  Structure, shapes, parameter facts, and measured costs all stay
  first-class.
- **The first raw value read warns once.** Reading `log[...].out` on a
  synthesized-value trace warns once per trace object
  (`nongold_raw_value_read`) -- the numbers are real measurements of a
  forward pass over random data. Filter the warning by its code to
  acknowledge it.

## The one-call picture: `tl.draw`

```python
import torchlens as tl

result = tl.draw(model, x)                            # your real input
result = tl.draw(model, input_shape=(1, 3, 224, 224))  # declared shape
result = tl.draw(model)                               # inferred
```

`tl.draw` runs ONE metadata-only capture pinned to eval + no-grad -- training
flags, RNG state, and normalization buffers are restored afterwards and the
restoration is verified with a state-dict hash (`result.receipt`) -- then
renders through the existing renderer with `collapse="auto"` as its default.
`tl.draw` is the only place auto-collapse is a default: `Trace.draw()`
keeps `collapse="none"`.

The result is a detached `tl.viz.Drawing`: it owns the DOT source, the
rendered bytes or written path, the input provenance, and `save(path)`
re-renders without retaining the capture (`save(path, editable=True)` writes
the DOT source instead). In notebooks it displays inline and writes nothing;
in a plain script a bare call writes a collision-safe `<ModelClass>-graph.pdf`
and prints the path (never a shared filename, never a silent overwrite). No
surface ever auto-opens a viewer; pass `open_viewer=True` to opt in. To keep
the trace, call `tl.trace(model, x)` and draw it yourself.

Curated dials: `theme=`, `collapse=`, `module=`, `depth=`, `path=`,
`file_format=`, `open_viewer=`, the spellings `Trace.draw` itself uses.
Everything else passes through to `Trace.draw`; for full renderer control,
drop to `tl.trace(...).draw(...)`.

## Lazy modules

A model holding un-materialized lazy parameters (`nn.LazyLinear` and
friends) works with a real input or a declared shape: the parameters
materialize during the ONE captured forward -- exactly as they would on a
direct call -- and the inventory, totals, and metadata invariants report the
real post-materialization geometry. A call with no input raises BEFORE
probing (a lazy module accepts any width, so a probe would silently install
its guess in your model), and intervention-ready captures raise
`state_baseline_unavailable` because a pending parameter has no bytes to
witness. The two-line fix the error names always works:

```python
with torch.no_grad():
    model(x)   # materialize outside capture, then re-run the verb
```

Un-materialized lazy BUFFERS (`nn.LazyBatchNorm2d` before any forward)
raise at entry with the same fix (`lazy_uninitialized`).

A lazy module the forward itself builds (`self.lazy = nn.LazyLinear(4)`
inside `forward`) is registered when the forward first consumes its
materialized parameters, and the trace discloses the creation as a
`parameter_created_in_forward` row in
`trace.annotations["capture_advisories"]`; see
[Limitations](reference/limitations.md).

## Execution policy per surface

`tl.summary` and `tl.draw` pin eval + no-grad and restore everything they
touch (verified, disclosed). `tl.trace` is the full-control entry: it keeps YOUR
modes -- a train-mode capture really does update BatchNorm running
statistics, and TorchLens warns once per process
(`batchnorm_train_stats_mutated`) naming `model.eval()`. The zero-argument
trace is necessarily the eval-mode trace that inference verified; asking for
incompatible capture options recaptures from the inferred input with your
options instead (disclosed in the provenance record).
