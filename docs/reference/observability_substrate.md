# Observability substrates

`torchlens.observability` is the shared substrate the observe / checks /
trackers / explorer / snoop / torchnative surfaces build on. Every spelling in
it is DOCUMENTED-UNSTABLE pending naming-session ratification; access is
`import torchlens.observability` (root-facade routing is a later registration).

## The per-step history schema

Four normalized layers, one artifact:

- **RunRecord** -- schema version, run/segment identity with resume ancestry,
  model fingerprint, package versions, the histogram descriptor, per-stream
  cadences, budgets, device, rank/world.
- **SiteRecord** -- stable STRUCTURAL identity; labels are display metadata
  only. Topology drift ends a series (`ended_step`/`ended_reason`) and starts
  a new one -- disclosure, never re-binding.
- **StepBlockRecord** -- the atomic training coordinate. Both step spellings
  exist and stamp provenance: the explicit `step()` transaction is
  authoritative (and required for resume), carrying its per-step
  optimizer-truth disclosures (applied/scale/unscaled/clipped/micro-batches)
  as one frozen `StepTruth`; the implicit `optimizer=` spelling records
  `unknown` for the unscale/clip status it cannot determine.
- **ObservationRecord** -- `(step, site, stream, phase)` plus a closed
  presence vocabulary (`observed | not_scheduled | site_absent | unsupported |
  capture_failed | budget_dropped`), the spine, the sketch, and disclosures.
  Missing is never zero: a non-observed presence may not carry payloads.
  Gradient streams must stamp `grad_scale` (`scaled | unscaled | unknown`).

Merging: integer counts merge EXACTLY (rank merge, micro-batch folds, time
coarsening); floating fields merge with a documented stable pairwise algorithm
(Chan moments) and are never called exact.

## The two kernels

- **Spine** -- the always-on scalar spine: exact counts (total / finite /
  zero / negative / NaN / +/-inf), finite min/max/absmax, sum, sum-squares,
  sum-abs, stable-merge moments. `num, min, max, sum, sum_squares` map
  verbatim onto TensorBoard `add_histogram_raw` arguments.
- **Histogram** -- ONE immutable universal SIGNED log2 grid: defaults
  `bins_per_octave=4`, window `[2^-48, 2^16]` per sign; specials (exact zero,
  per-side under/overflow, NaN, +/-inf) stay exact and render as annotated
  bands. The full descriptor `(base, bins_per_octave, lo_exp, hi_exp, signed,
  encoding)` travels in the manifest, making resolution a VALUE change
  forever. There is no rebinning path; descriptor mismatch at merge refuses
  `sketch_descriptor_mismatch`.

Both implement the `torchlens.stats` `StreamingStat` protocol and are
re-exported as `tl.stats.Spine` / `tl.stats.Histogram`, so `tl.aggregate`
gets dataset-mode histograms free. Neither consumes RNG; half/bf16 inputs
reduce in float32-or-wider with the reduction dtype recorded; complex dtypes
refuse `stat_dtype_unsupported`.

## The artifact

Disk-first, append-only, versioned directory (`torchlens.history.v1`):
`manifest.json` + `catalog.json` + `index.json` + checksummed NumPy chunk
files, no pickle. A chunk exists only once its row enters the index (temp +
fsync + rename), so a killed writer is readable to the last committed chunk;
a committed chunk that later fails its sha256 refuses
`history_artifact_corrupt` -- recovery never silently drops committed data.

The RAM ring (default 256 committed StepBlocks) is a cache when the disk
archive is on (LRU eviction, not an event). RAM-only mode has three EXPLICIT
policies: `refuse` (default -- typed refusal BEFORE the next scheduled block,
naming the remedies), `drop_oldest` (with counters), `coarsen` (exact
pairwise count-preserving merges carrying `[step_lo, step_hi]` spans).
No automatic tier shedding exists anywhere.

## The module-tier collector (Route A)

`HistoryCollector(model, sites=, streams=, cadence=, settings=)`: one real
discovery forward catalogs the selected module outputs while returning the
user's actual result; persistent observers then reduce DETACHED views under
`no_grad` at module return. The collector never enters fastlog and needs no
capture-core change. The what-to-watch surface stays flat; the plumbing knobs
(sketch cadence + descriptor, run/segment identity, output dir, RAM
capacity/policy, event stream, update sample size) ride the frozen
`WatchSettings` dataclass.

- Transfer invariant: per-site reductions are fixed-width device vectors in a
  preallocated staging buffer; host transfers are batched per phase per
  device, independent of site count -- never a per-site `.item()`.
- The plan table prints REDUCED ELEMENTS PER LOGGED STEP alongside bytes,
  with per-stream cadence; zero-match selections refuse at plan time.
- The update channel is a deterministic seeded fp32 subvector of
  `min(numel, 4096)` indices per parameter (private RNG, never the model's);
  full coverage degrades gracefully to EXACT with `estimated=False`. A
  skipped optimizer step records `skipped` and NO fake update.
- `activation_grad` rides the F24 observer seam, not this collector.

Lifecycle purity is a tested contract: every hook handle is removed on every
exit path, and a bitwise same-seed observed run equals the unobserved control
(losses, parameters, buffers incl. train-mode BatchNorm running stats, RNG
state).

## The observer/check slot

`EventStream` is the bounded immutable event stream checks (F23) and trackers
(F26) both consume; neither repeats the tensor scan. `ObserverEvent` kinds
are `scalar | histogram_counts | verdict | unavailable` (`unavailable` is
mandatory vocabulary and needs a reason); every gradient scalar carries
`stage` + `scale_provenance`. A throwing subscriber cannot corrupt
collection: failures are caught, warned about once, and the subscriber is
disabled after eight consecutive failures.

## Spans, regions, the ONE session engine

`SpanRegistry` registers spans AT ENTRY (identity-carrying, escaped/bounded
names) across three altitudes (`op` / `grad_fn_fire` / `aten`) with owner
classification including the mandatory `torchlens_internal` bucket; leaked
spans are closed at the session boundary WITH disclosure. Timestamps are
monotonic and never projected onto Kineto's clock.

`region(name, **scalar_metadata)` is the user bracket on that same stack:
occurrence + parent ids, inert without a consumer; under an active session it
records a `user_region` span (and a `torch.profiler.record_function` bracket);
during a TorchLens capture it rides the SHIPPED `observers.span` record
surface -- one span vocabulary, never a parallel stack.

`torchlens.observability.session(...)` is the ONE activation knob: owned mode
creates and closes `torch.profiler.profile`; borrowed mode adds markers
inside a caller-owned profiler and never steps or closes it; nested sessions
refuse `profiler_session_nested`; success, halt, and exception paths restore
every marker and the active-session slot. The kernel-enrolled
`profiler_doors` registry holds exactly one entry, and the dependency test
(`tests/test_obs_substrate_spans.py::TestOneProfilerDoor`) forbids a second
`torch.profiler.profile` construction site (the kernel_telemetry legacy seam
was burned down by F27/W2.1: its ATen activation now routes through this
door) and any parallel grad_fn node-hook stack. Session results carry the
five-state availability lattice (`not_requested | unavailable | empty |
partial | joined`).

## The Kineto join (F27, on this substrate)

`torchlens.observability.native_profile(model, x)` runs ONE save-nothing
capture under the owned session and joins device events to captured ops by
runtime correlation IDs only (names are display metadata): in-memory
`_KinetoEvent` extraction behind the feature-detected `_torch_compat`
boundary (bounded chrome-stream fallback, path disclosed), same-thread
innermost containment in one O(N log N) sweep, launches counted once (a
multi-owner launch is an `ambiguous` group, never split or duplicated),
interval-union device-busy time, typed TorchLens-internal work excluded
from part (a) of the two-part accounting and owned in part (b), and a
NAMED residual. Consumers: `hot_path(by="device_time")`,
`draw(color_by=/size_by="device_time")`, the profile device columns, and
the native chrome artifact + exact-ID mapping sidecar. An unjoined capture
refuses explicit device-time requests typed (`device_time_unavailable`);
the >= 95% real-GPU acceptance gates (C-KINETO) run on the cluster via
`tools/gpu_gates/kineto_join_gate.py`.
