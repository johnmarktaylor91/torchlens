# Classics corpus

A coverage-chosen sample of hand-built "classics": historical and unusual neural-network
architectures written directly in PyTorch (McCulloch-Pitts nets, Hopfield and Boltzmann machines,
ART, early CNNs and RNNs, through modern attention, state-space, graph, and generative models).
Each was built for the Model Menagerie (https://modelmenagerie.ai), whose full classics battery
runs downstream of TorchLens. This directory keeps the subset that exercises the
most TorchLens capture paths for the least test time, so TorchLens itself stays honest about being
complete across architectures.

The corpus holds 207 entries (13 smoke-subset entries among them) from 198 model files.

## What each entry is checked for

`tests/test_classics_corpus.py` runs, per entry:

1. capture with `tl.trace`;
2. a portable `.tlspec` save and load, comparing the op labels;
3. forward-replay validation with metadata invariants (`tl.validate(scope="forward")`);
4. runnable resolver readiness: zero unresolved or ambiguous torch registry keys (the release gate
   in `docs/reference/runnable_tlspec_contract.md`, section 13).

That full check is the comprehensive tier: it covers every entry and carries `slow` (weekly CI).
The smoke subset also runs forward validation alone under the `smoke` marker; its size is set by
the smoke tier's aggregate family budget (`tests/conftest.py`, about 10 s per parametrized family
at load factor 1), so it holds the cheapest entries that still reach the fragile paths.
`pytest tests/test_classics_corpus.py` runs both; `classics_resolver_coverage_report()` in
`tests/test_tlspec_resolver_coverage.py` produces the release report over the whole corpus.

## How the entries were chosen

Every classic that imports only torch, numpy and the standard library (2,144 files holding 3,506
entries) was a candidate. The census on 2026-10-01 traced 2,744 of those entries on CPU workers
(the first entry of every file and 606 further variants), and each trace yielded a set of coverage
features: op function names, built-in module types, multi-pass (recurrent) layers, conditional
branches, in-place ops, output dtypes, parameter dtypes, input and output container shapes,
buffers and buffer writes, tied or multiply-used parameters, frozen parameters, tensors created
inside `forward`, scalar-bool ops, rank-0 and rank-5+ tensors, module reuse and depth, model size,
and era. Entries that failed any of the four checks above were not eligible; each such failure is a
TorchLens finding to root-cause, and a fixed model may be added later. Entries slower than 20 s,
entries using `max_unpool` (no deterministic kernel, and the suite runs with
`torch.use_deterministic_algorithms(True)`), files over the 2,000-line test-module cap, and one
model whose validation verdict differed between a fresh and a warm process were left out too.

The smoke subset was picked first: a greedy cover weighted toward rare and fragile features
(recurrence, conditionals, in-place ops, unusual dtypes and containers, shared parameters), by
coverage gain per measured validation second, within the family budget. The comprehensive tier then
extends it greedily, by coverage gain per square root of cost: one witness for every feature
reachable under the cost cap, a second witness for every feature, then a dearer pass for features
that only slower models show. The selection covers 435 of
the 442 features seen in the eligible pool; the seven left out (`bernoulli`, `eq`, `hardsigmoid`,
`mish` and `nn.Mish`, `tensor_split`, and graphs of 3,000 or more ops) appear only in models too
slow for this tier.

## Provenance and editing rules

Every file under `models/` is byte-identical to `menagerie/classics/<module>.py` at public
TorchLens commit `bac76673d`, pinned by sha256 in `manifest.json` (`files`). Do not edit a model
file in place: a model that fails a check is a TorchLens finding to root-cause, never a reason to
change the model or loosen a check. `models/` is excluded from ruff for the same reason.

Each `manifest.json` entry records the entry id, file, constructor and input factory names, tier,
era, year, the features the census saw, and why it was chosen.
