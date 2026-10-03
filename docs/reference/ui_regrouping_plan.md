# Op surface regrouping plan (FORK-2 — PLAN ONLY, no implementation)

Status: decided PLAN-ONLY (2026-07); this document is the M13
deliverable of docs/reference/trace_core_design.md section 5. Nothing here
is implemented, scheduled, or authorized for implementation — it is the
design of record for the day the fork is exercised.

## Problem

`Op` exposes ~195 public attributes flat. The M5-M9 columnar re-plumbing
made the STORAGE cheap (a two-word facade over one row), but the *surface*
remains a single namespace where everyday fields (`out`, `shape`, `parents`,
`func_name`) sit beside specialist facts (RNG snapshots, FLOP counters,
autograd introspection, conditional role stacks, storage bookkeeping).
Tab-completion and docs pay the cost.

## Decided shape

Everyday fields stay top-level, unchanged and unaliased. Specialist facts
gain grouped, lazily-constructed sub-views (one per domain), each a small
read-through facade over the SAME row — no storage change, no schema change,
no FIELD_ORDER change:

| group | representative fields |
|---|---|
| `op.timing` | `func_duration`, `func_time_started/finished`, capture timestamps |
| `op.autograd` | `grad_fn*`, `has_grad`, gradient memory, backward links |
| `op.geometry` | shapes/strides/dtypes beyond `shape`/`dtype`, memory numbers |
| `op.provenance` | `code_context`, call stack, var names, source links |
| `op.storage` | save/eviction/blob-ref bookkeeping, payload policies |
| `op.control_flow` | conditional roles/stacks/children, terminal-bool facts |
| `op.module_context` | module stacks, entry/exit metadata beyond `modules` |

## Compatibility rules (LOCKED for the eventual implementation)

1. Grouped views are ADDITIVE. Every existing flat name keeps working for a
   full deprecation cycle; flat specialist names emit a deprecation warning
   only after the grouped surface has shipped one stable release.
2. `FIELD_ORDER`, `.tlspec` state, pickle state, and `to_pandas` columns are
   UNCHANGED — grouping is presentation, never schema.
3. Group views are lazy one-per-op facades reading the same row cells;
   identity is not guaranteed across reads (`op.timing is op.timing` may be
   False) and mutation goes through the same descriptor/overlay path as the
   flat name.
4. The grouped/flat split is declared in ONE table (extending the
   `StorageBinding` axes with a `ui_group` column) so the compiler generates
   both surfaces and the lockstep tests police them; no hand-maintained
   duplicate lists.
5. Rollout gate: a real-IDE completion session (the M0/M13 gate) comparing
   completion noise before/after, plus the DX oracle.

## Explicitly out

- Removing or renaming any flat field (separate authorization).
- Grouping on `Layer`/`Trace` (revisit only after the Op rollout soaks).
- Any `.tlspec`/manifest vocabulary change.
