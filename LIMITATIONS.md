# TorchLens Limitations

TorchLens observes eager PyTorch execution by wrapping PyTorch call sites and recording what
actually runs. Its substrate identity is therefore "PyTorch eager execution plus TorchLens
metadata", not a separate static IR, compiler graph, or symbolic model format.

## Single-Threaded By Design

TorchLens is single-threaded by design. Capture depends on process-global logging state and
ordered operation counters, so running captures concurrently in multiple Python threads or child
processes can corrupt ordering assumptions. Use TorchLens from the main process.

## Dynamic Control Flow

Eager dynamic control flow is a feature, not a limitation. TorchLens records the branch, loop, and
module behavior that occurred for the concrete input you supplied. It does not claim to enumerate
branches that did not execute.

## Bundle Diff Rendering

`Bundle.show_diff()` is static for v1. Interactive bundle comparison belongs in
`torchlens.viewer` later.

## Compatibility Truth Table

Phase 13 will fill in the full `tl.compat.report` truth table. Until then, known-broken rows are
tracked structurally here so downstream documentation has a stable cross-reference:

| Row | Status | Notes |
| --- | --- | --- |
| `torch.compile` primary capture | Known broken | Scope row; see Phase 13 `tl.compat.report`. |
| Static graph export parity | Known broken | TorchLens records eager execution, not static export IR. |
| Concurrent capture | Known broken | Conflicts with the single-threaded design above. |
| Optional bridge without its extra | Known broken | Install the matching extra before using bridge adapters. |

See `ROADMAP.md` for planned follow-up work. That file is scheduled for Phase 15.
