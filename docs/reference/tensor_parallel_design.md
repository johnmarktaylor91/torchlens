# Tensor-parallel capture: the planned answer (design note, not built)

Status: decided direction, dated 2026-10-29. Nothing here is implemented. Ruling
arch.7 (2026-10-13) chose "design note now; build in a follow-up": this page
records the answer so the typed refusal points somewhere, and the build is
scheduled with later work.

## Today

A model that holds `DTensor` or `ShardedTensor` state, or has active
tensor-parallel hooks or styles, is refused at capture entry with
`tl.errors.DistributedCaptureUnsupportedError` (finding kinds `dtensor` and
`tensor_parallel`; `tl.compat.report()` shows the same rows). The reason is
fidelity, not effort: DTensor's real work happens under `__torch_dispatch__`,
below the `__torch_function__` layer TorchLens wraps, so an unguarded capture
records rank-local view and reshape shims instead of the real ops and reports
zero parameters. The workaround stays the same until the build lands: capture a
rank-local dense copy of the module (build it unsharded, or materialize the
logical tensors with `DTensor.full_tensor()` first).

## The planned answer: inference-only gather-on-read

1. **Inference only.** The first build captures forward passes under
   `inference_only=True` semantics. Backward capture, interventions, replay and
   rank merging of tensor-parallel captures stay refused with their own typed
   codes until each is proven separately.
2. **Record the shard, read the logical tensor.** Each captured activation that
   is a `DTensor` is stored as this rank's local shard plus its placement
   record (device mesh, placements, logical shape, local shape), the same dual
   geometry the refusal already reports for parameters. Reading an activation
   (`op.out` and every read built on it) gathers the logical tensor on demand
   instead of at capture time, so a capture that is never read costs no extra
   collectives.
3. **Logical identity.** Op shapes, parameter counts and module records use
   the logical geometry, and every record says which rank captured it and how
   its tensor was placed, so nothing reads a shard as if it were the whole
   tensor.
4. **Reference design.** nnsight 0.8 added tensor-parallel support, and
   transformers 5.16 made tensor parallelism DTensor-native. The build checks
   its gathered values against a dense single-device capture of the same
   model.

## Constraints the build must settle

- A gather is a collective: every rank of the mesh has to take part, in the
  same order. Reads therefore either run on all ranks together (SPMD reads) or
  the capture gathers once at an explicit point (at save, or an explicit
  materialize step) while the process group is alive. A loaded trace never
  gathers.
- Fidelity comes first: the op graph recorded under DTensor must be shown to
  match the dense graph op for op (the condition the current refusal names)
  before any tensor-parallel capture is admitted.
- Memory: a gathered read allocates the full logical tensor on the reading
  rank; the read must say so and respect the capture's save budget.

## Not planned

Training-time tensor parallelism, FSDP and pipeline parallelism stay refused;
they are outside this note.
