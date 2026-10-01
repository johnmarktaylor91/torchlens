# torchlens architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.


## Reference obligations

Referenced sections retain their instructions. Before changes, read the relevant module, subsystem, public surface and architecture reference. Paths in reference text are relative to the repository root.

## What This Is

TorchLens extracts outs and metadata from backend-resolved captures. PyTorch eager capture is the
stable default; MLX, JAX, tinygrad, Paddle, and TensorFlow are technical-preview backends. `import torchlens`
exposes the public API, but torch wrapping is lazy: the first torch capture
prepares the model and calls `wrap_torch()` from `backends/torch/`.

## Architecture Overview

Required reference: [Architecture Overview](../docs/agent-reference/package/architecture-overview.md).

## Top-Level Modules

Required reference: [Top-Level Modules](../docs/agent-reference/package/top-level-modules.md).

## Subpackages

Required reference: [Subpackages](../docs/agent-reference/package/subpackages.md).

## Key Concepts

Required reference: [Key Concepts](../docs/agent-reference/package/key-concepts.md).

## Files in This Directory

Required reference: [Files in This Directory](../docs/agent-reference/package/files-in-this-directory.md).

## Attribute Conventions

- TorchLens metadata attached to user/model objects lives under `obj._tl`.
- Permanent module metadata uses `_tl.address` and `_tl.module_type`.
- Session tensor/parameter metadata is cleaned per capture; callable wrapper markers also live
  under `_tl`.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state.

## Public Surface

Required reference: [Public Surface](../docs/agent-reference/package/public-surface.md).

## Constants as Ordering Spec

FIELD_ORDER tuples define canonical serialized and display field sets. When adding a field,
update the class definition, the appropriate FIELD_ORDER constant, metadata tests, and any
`to_pandas()`/summary surface that should expose it.

## Critical Invariants

1. `_state.py` has no outgoing torchlens imports except the sanctioned `errors._base`
   leaf — a RUNTIME import (its classes are base classes, e.g. `ReentrantTraceError`);
   only the TYPE_CHECKING block below it is typing-only (cycle-safe by construction;
   documented in `_state.py`).
2. `_ensure_model_prepared()` is the lazy wrapping chokepoint; do not reintroduce import-time
   torch namespace mutation.
3. RNG state capture/restore must happen before `active_logging()`.
4. Internal torch ops during capture must be wrapped in `pause_logging()`.
5. Module suffixes are appended to `equivalence_class` at op creation before loop detection.
6. There is no `postprocess_fast()` orchestrator; refresh captures run the full `postprocess()`
   entry point (see `postprocess/AGENTS.md`, "Refresh Projection").
7. `backward_ready=True` must preserve user `requires_grad` and reject detach/disk conflicts.
8. Portable I/O must reject unsafe paths/symlinks and unsupported tensor variants.

## Newer 2.x Subsystems

Required reference: [Newer 2.x Subsystems](../docs/agent-reference/package/newer-2-x-subsystems.md).

## Package Layout Policy

- `io` is the public portable I/O facade; `_io` owns bundle, manifest, codec, and lazy-load internals.
- `errors` is the public exception facade; `_errors.py` is legacy internal exception plumbing to fold in later.
- `debug` is the public diagnostics toolbox; private debug helpers should stay local to their owning modules.
- `visualization` owns graph rendering and layout; `viz` owns image and plot primitives used by renderers.

## Conditional Branch Attribution

- Step 5 builds AST file indexes, classifies terminal bools, materializes dense
  `conditional_records`, runs backward flood, attributes forward arm edges, then derives
  legacy THEN/ELIF/ELSE views.
- Primary structures are `Trace.conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, and `conditional_arm_children`.
- Graphviz renders IF/THEN/ELIF/ELSE labels; dagua conditional support remains more
  limited than Graphviz.

## Release Safety

Semantic-release uses `scripts/no_major_parser.py` plus commit hooks to block accidental
major bumps. For docs-only work use `docs(...)` or `chore(...)` and never add major-bump
markers to commit messages, PR text, or committed docs.
