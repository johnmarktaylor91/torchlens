# Frequently asked questions

Short answers, each with a link to the page that covers the topic in full.

## What distinguishes TorchLens from other packages?

Most tools for looking inside a PyTorch model attach hooks to modules, so they see what each module
returns. TorchLens records every tensor operation the forward runs, including the functions and
tensor methods called inside a module's `forward`, and links each operation to the operations that
produced its inputs. The result is the full graph of the run with every intermediate value, and
several features follow from having that graph:

- **Receptive and projective fields.** Which input positions can affect a unit, and which units an
  input position can affect, are read off the recorded graph (`layer.receptive_field`,
  `layer.projective_field`), so skip connections and merges are counted, where a formula built
  from layer sizes can misstate them
  ([receptive and projective fields](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/receptive_projective_fields.md)).
- **Validation.** Each operation's function, inputs and output are saved, so `tl.validate(model, x)`
  re-runs every saved operation on its saved inputs and checks that it reproduces the saved output.
- **Editing.** `trace.do(site, edit)` can change any operation, not only a module output. Without
  the model, it propagates the change by replaying only the operations downstream of the edit in
  the saved graph ([intervention reference](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/intervention_api.md)).
- **Rolled drawings.** The graph records when one module runs several times, as in a loop, so
  `trace.draw(rolled=True)` draws it once instead of once per pass.
- **Comparing runs.** Two traces line up operation by operation, so a `tl.Bundle` of them gives the
  difference at every operation both runs share with `bundle.compare(a, b)`, one row per pass.

## How does TorchLens work under the hood?

The first capture replaces every public torch function (the `torch` namespace,
`torch.nn.functional`, tensor methods and operators) with a thin wrapper around the original.
Outside a capture each wrapper checks one flag and calls the original; during a capture it also
records the call. After the forward, a postprocessing step turns those records into operations,
groups them into layers and modules, and builds the `Trace`.
[How TorchLens works](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/how-it-works.md) walks through each stage.

## Does it work with my model?

Very likely, if its forward runs as ordinary eager PyTorch. TorchLens records what actually runs,
so custom Python code, if/else branches, loops and early exits are captured as the path the forward
took for the inputs you gave, and a module that runs several times appears once per pass. Hugging
Face models are traced like any other `nn.Module`; pass keyword inputs with `input_kwargs=`. The
[Model Menagerie](https://modelmenagerie.ai) runs a capture-and-replay check on thousands of
cataloged models across the major architecture families
([the same three lines on six architectures](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/gallery/any_architecture.md)). If a model
fails, `tl.report.doctor(model=model, inputs=x)` lists the settings known to break or not yet
tested. The exceptions are covered below: TorchScript and `torch.export`, and sharded models.

## Does tracing change my model's outputs?

No: TorchLens wraps torch functions but calls the originals with the same arguments, and
`tl.validate(model, x)` checks that a fresh forward reproduces the traced output. Two things to
know. Random operations such as dropout draw from a seed TorchLens sets for each capture (fix it
with `capture=tl.options.CaptureOptions(seed=...)`), and your own random-number state is restored
afterwards. And a model in training mode updates state such as BatchNorm running statistics on
every forward, traced or not, so call `.eval()` first unless you want that. If TorchLens has to
run the forward a second time to recover a stale reference to a torch function, the trace is
marked unverified and names the re-run as the reason
([capture completeness](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#capture-completeness-and-python-call-routes)).

## How much slower is tracing, and how much memory does it use?

A full trace keeps the output of every operation and costs about 14 times a plain forward on CPU
(ResNet-18 13.8x, GPT-2 14.8x on the benchmark host). A trace of ResNet-18 at 224 pixels keeps
about 31 MiB of activations and one of GPT-2 small on 32 tokens about 96 MiB; both grow with batch
size. You pay less by keeping less: `save=` takes layer names, module addresses or a selector
such as `tl.func("relu")`, `stop_after=` ends the forward early, and `tl.record(model, x,
save=...)` is a lighter recording path: saving nothing and stopping a quarter of the way through,
it took 0.84x the time of a plain forward on ResNet-18. `storage=tl.to_disk(path)` streams saved activations
to disk; `CaptureOptions(inference_only=True)` or `CaptureOptions(detach_saved_activations=True)`
stops the trace from keeping the autograd graph alive; `trace.cleanup()` frees a trace you are
done with. See the [performance guide](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/performance.md) for the full tables and
[memory and object lifetime](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#memory-payloads-and-object-lifetime).

## Does it work with torch.compile, TorchScript or torch.export?

With `torch.compile`, yes: pass the compiled model and TorchLens records the original eager module.
On torch 2.6 and later, compiled regions run eagerly during the capture, so the trace is complete
and the compiled caches stay intact; on older torch, a compiled region reached mid-capture is
skipped with a warning and the trace is marked unverified. TorchScript modules and `torch.export`
programs are refused, because they do not run as ordinary Python: trace the eager `nn.Module`
before scripting or exporting it. Details are in the
[limitations catalog](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#capture-entry-and-execution-contexts).

## Does it work on a GPU?

Yes. Capture runs on whatever device the model and inputs are on, and saved activations stay on
that device by default, so a CUDA capture spends GPU memory;
`capture=tl.options.CaptureOptions(output_device="cpu")` keeps them in host memory instead.
Continuous integration runs the test suite on CPU only. A nightly canary captures a small model
on Apple-silicon MPS and saves and reloads it; CUDA, and device-specific behavior beyond that
canary, is checked by local testing, not in CI ([tested platforms](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#ci-attested-platforms)).

## Can I trace large models or models spread over several GPUs?

TorchLens captures one forward in one process. `DataParallel` models are unwrapped and captured
as one model; `DistributedDataParallel` models are captured per rank, and `tl.merge_ranks([...])`
merges the rank traces; FSDP, tensor-parallel and pipeline-parallel models are refused, so capture
an unsharded copy or one stage at a time
([multi-GPU extraction](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/multi_gpu_extraction.md)). Accelerate CPU and disk
offload, 8-bit and 4-bit bitsandbytes layers and PEFT adapters are verified on small models on
CPU; a model split over several GPUs with `device_map` and models of 7B parameters or more are not
verified yet ([deployment envelope](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/deployment_envelope.md)). To extract
activations over a dataset on several GPUs, run one `tl.extract_dataset` per GPU on its own share
of the data.

## Can I use it inside a training loop?

Yes. Saved outputs stay connected to the autograd graph by default, and
`capture=tl.options.CaptureOptions(backward_ready=True)` declares a training capture (it keeps that
connection on every save path and is what the training tutorial uses), so a loss built from saved
outputs backpropagates into the model: auxiliary losses, probes on a frozen backbone, losses at
several layers, and distillation. The
[training tutorial](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/training_tutorial.ipynb)
works through each pattern. A full trace costs many forwards' worth of time, so in a loop save
only the layers the loss needs.

## Does it work with threads?

A capture belongs to the thread that started it, so keep the model's tensor work on that thread.
If tensor work runs on another thread, or the number of Python threads changes during the
forward, you still get a trace, but it is marked unverified: `trace.provenance.verified` is
`False` and `trace.provenance.verification_reason` names the cause (for example
`owner_thread_tripwire_changed`). A capture also refuses to start inside a `DataLoader` worker
process or from inside another capture. See
[capture completeness](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#capture-completeness-and-python-call-routes).

## Does it work with JAX, TensorFlow or other frameworks?

TorchLens is built for PyTorch eager. JAX, TensorFlow, MLX, Paddle and tinygrad have their own
backends: pass `backend="jax"` (or `"tf"`, `"paddle"`, `"tinygrad"`) to `tl.trace`, and MLX
modules are routed to their backend automatically. Each needs its framework installed (for
example `pip install "torchlens[jax]"`), covers a smaller set of features, and refuses with a
typed error anything outside its capability table. See
[backend support](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/backends.md).

## Coming from an earlier TorchLens (1.x or 2.x)

Many names changed, and the old spellings were removed outright rather than kept as deprecated
aliases, so old code fails with an error instead of quietly doing something different. The
[removed-spellings ledger](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/deprecations.md)
maps each old name to its replacement (for example, the 1.x capture function and the object it
returned are now `tl.trace` and `Trace`), and
[TorchLens 2.0 API changes](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/v2.0_api_changes.md)
lists where each former top-level name lives now. Files saved by torchlens 2.37 and earlier do
not load in 2.38 or later; [MIGRATIONS.md](https://github.com/johnmarktaylor91/torchlens/blob/main/MIGRATIONS.md)
gives the remedy for each kind of file.

## Will the API change again?

The [glossary](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/glossary.md)
fixes the public meaning of every name. Names the glossary marks *unstable* may change without
notice. A release that removes or renames a name lists it in the
[CHANGELOG](https://github.com/johnmarktaylor91/torchlens/blob/main/CHANGELOG.md) and in the
[removed-spellings ledger](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/deprecations.md).
For saved files, a newer release loads files written by older releases of the same saved format
and refuses, with a typed error, anything it cannot read
([MIGRATIONS.md](https://github.com/johnmarktaylor91/torchlens/blob/main/MIGRATIONS.md)).

## I use another package. How does TorchLens compare?

The README's [Related tools](https://github.com/johnmarktaylor91/torchlens#related-tools) section
says how TorchLens sits beside the tools people use most. For a direct translation, the
[coming-from pages](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/index.md)
each do one task in both tools, with a table of equivalent calls: for example, coming from
[TransformerLens](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_transformerlens.md),
[forward hooks](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_hooks.md)
or [Flax's `tabulate`](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_flax_tabulate.md).

## Will you add feature X to TorchLens?

If it works inside the network, probably: recording, selecting, editing, comparing and drawing what
a forward pass runs are in scope, so open an
[issue](https://github.com/johnmarktaylor91/torchlens/issues) or a pull request. Analysis of the
activations afterwards, such as decoding, RSA, dimensionality reduction and sparse autoencoders,
stays with the packages built for it (scikit-learn, rsatoolbox, SAELens), and so does training
interventions (pyvene); TorchLens hands them plain tensors and tables (`trace.to_pandas()`,
`tl.extract_dataset`). The README's
[Related tools](https://github.com/johnmarktaylor91/torchlens#related-tools) section and the
[connectors to other tools](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/bridges.md) say which tool does what. Where TorchLens already
ships something in these areas, it goes this far:

- **Analysis.** Descriptive quantities only: streaming statistics including an incremental PCA
  (`tl.stats`), RDMs, classical MDS and Procrustes alignment (`tl.repgeom`), linear CKA
  (`tl.stats.cka`), and descriptive RSA, ridge and CKA scores against brain data
  (`tl.neuro.compare`). Statistical inference (p-values, noise ceilings, model comparison) stays
  with rsatoolbox, scikit-learn and similar tools. It collects the features a probe or decoder
  needs but trains neither, and it trains no sparse autoencoder.
- **Interventions.** It applies what you learned elsewhere: `tl.subspace` applies a basis you
  supply, and the pyreft connector loads a trained LoReFT intervention. The
  [DAS recipe](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/field-guide/das_gpt2.md) fits its distributed alignment search basis with a
  plain PyTorch loop. For steering, `tl.intervention.fit_direction` (marked unstable) computes a
  direction from paired runs by mean difference, first principal axis or logistic regression.

Other jobs, such as training-data attribution and adversarial-attack suites, are listed with the
tool to use instead in [what TorchLens does not do](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/non_goals.md).

## How do I cite TorchLens?

Cite the paper: Taylor, J., Kriegeskorte, N. Extracting and visualizing hidden activations and
computational graphs of PyTorch models with TorchLens. *Sci Rep* 13, 14375 (2023).
<https://doi.org/10.1038/s41598-023-40807-0>. The repository's
[CITATION.cff](https://github.com/johnmarktaylor91/torchlens/blob/main/CITATION.cff) has the same
entry in machine-readable form, and GitHub's "Cite this repository" button exports it as BibTeX.

## Where can I ask a question or report a problem?

Ask questions in
[GitHub Discussions](https://github.com/johnmarktaylor91/torchlens/discussions/categories/q-a),
where other users can find the answer later. Report bugs on the
[issues page](https://github.com/johnmarktaylor91/torchlens/issues); for a model-specific failure,
include the output of `print(tl.report.doctor(model=model, inputs=x).to_markdown())`. Report
security vulnerabilities privately, as
[SECURITY.md](https://github.com/johnmarktaylor91/torchlens/blob/main/SECURITY.md) describes.
