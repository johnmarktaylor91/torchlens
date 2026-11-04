# Frequently asked questions

Short answers, each with a link to the page that covers the topic in full.

<!-- Links point at the repository on GitHub because most target pages are not
published on the site yet (docs/_site_map.yaml); switch a link to a relative
path when its page gets publish: true. -->

## Coming from an earlier TorchLens (1.x or 2.x)

Many names changed, and the old spellings were removed outright rather than kept as deprecated
aliases, so old code fails with an error instead of quietly doing something different. The
[removed-spellings ledger](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/deprecations.md)
maps each old name to its replacement (for example, the 1.x capture function and the object it
returned are now `tl.trace` and `Trace`), and
[TorchLens 2.0 API changes](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/v2.0_api_changes.md)
lists where each former top-level name lives now. Files saved by torchlens 2.x do not load in
TorchLens 3; [MIGRATIONS.md](https://github.com/johnmarktaylor91/torchlens/blob/main/MIGRATIONS.md)
gives the remedy for each kind of file.

## Will the API change again?

[JMT: confirm] TorchLens follows semantic versioning, and the
[glossary](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/glossary.md)
fixes the public meaning of every 3.x name. Names the glossary marks *unstable* may still change
without a deprecation period. For saved files, a newer 3.x release loads files written by older
3.x releases and refuses, with a typed error, anything it cannot read
([MIGRATIONS.md](https://github.com/johnmarktaylor91/torchlens/blob/main/MIGRATIONS.md)).

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
`DataParallel` and `DistributedDataParallel` models are unwrapped and captured as one model,
while FSDP and other sharded models are refused
([multi-GPU extraction](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/multi_gpu_extraction.md)).
Continuous integration runs on CPU only, so CUDA and MPS behavior is checked by local testing,
not in CI ([tested platforms](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#ci-attested-platforms)).

## A trace uses too much memory. What can I do?

By default a trace keeps the output of every operation. Keep only what you need with `save=`
(layer names, module addresses or a selector such as `tl.func("relu")`), stop the forward early
with `stop_after=`, or stream saved activations to disk with `storage=tl.to_disk(path)`. For
forward-only analysis, `CaptureOptions(inference_only=True)` or
`CaptureOptions(detach_saved_activations=True)` stops the trace from keeping the autograd graph
alive, and `trace.cleanup()` frees a trace you are done with straight away. See
[memory and object lifetime](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#memory-payloads-and-object-lifetime)
and the [performance guide](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/performance.md).

## Does it work with threads?

A capture belongs to the thread that started it, so keep the model's tensor work on that thread.
If tensor work runs on another thread, or the number of Python threads changes during the
forward, you still get a trace, but it is marked unverified: `trace.provenance.verified` is
`False` and `trace.provenance.verification_reason` names the cause (for example
`owner_thread_tripwire_changed`). A capture also refuses to start inside a `DataLoader` worker
process or from inside another capture. See
[capture completeness](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/limitations.md#capture-completeness-and-python-call-routes).

## Does it work with JAX, TensorFlow or other frameworks?

PyTorch eager is the stable backend. JAX, TensorFlow, MLX, Paddle and tinygrad have preview
backends: pass `backend="jax"` (or `"tf"`, `"paddle"`, `"tinygrad"`) to `tl.trace`, and MLX
modules are routed to their backend automatically. Each needs its framework installed (for
example `pip install "torchlens[jax]"`), covers a smaller set of features, and refuses with a
typed error anything outside its capability table. See
[backend support](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/backends.md).

## I use another package. How does TorchLens compare?

The README's [Related tools](https://github.com/johnmarktaylor91/torchlens#related-tools) section
says how TorchLens sits beside the tools people use most. For a direct translation, the
[coming-from pages](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/index.md)
each do one task in both tools, with a table of equivalent calls: for example, coming from
[TransformerLens](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_transformerlens.md),
[forward hooks](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_hooks.md)
or [Flax's `tabulate`](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/migration/from_flax_tabulate.md).

## What does TorchLens not do?

It records, edits and checks what an eager PyTorch forward runs; jobs such as training-data
attribution, adversarial-attack suites and retraining-based metrics are left to other tools.
[What TorchLens does not do](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/non_goals.md)
names each one and the tool to use instead.

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
