## Brainpipe (F20; spellings DOCUMENTED-UNSTABLE)

`torchlens.brainpipe` is the memory-planned whole-model extraction planner:
`extraction_plan` (warm-up + two measured probes, per-site linear byte fits with
batch-invariant detection, live transform measurement, full input-signature keying),
`plan.table()` (one row per requested site plus measured peak pairs, exact pass
counts, and budget arithmetic; over-budget refuses `extraction_plan_over_budget`,
drifted signatures refuse `extraction_plan_signature_mismatch`, non-trace engines
refuse `extraction_engine_unsupported`), `plan.run()` (the single-pass door onto
`extract_dataset`), `export_npz` (consolidated / per-stimulus; net2brain mode writes
their lexicographic naming contract), and `parse_bytes`. Companion surfaces:
`Trace.forward_peak_memory_pair` (session-time live/resident pair, backend named),
`tl.repgeom.rdm` keyword widening (manhattan/compute_device/row_chunk_size/
output_device/dtype/condensed/batched), descriptive `tl.repgeom.rdm_compare`,
`tl.stats.cka`/`CKA` `device=`/`dtype=`, geometry evolution over transformed
payloads with `result.payload_basis`, and lazy-buffer (`LazyBatchNorm*`) capture
without pre-materialization. Seam ledger: `docs/reference/brain_benchmarking_seam.md`.
