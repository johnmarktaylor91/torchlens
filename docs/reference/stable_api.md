# Stable API

TorchLens 3.x promises that the names on this page keep their spelling and meaning
across 3.x releases. A change to any of them is a major-version change and is listed in
`MIGRATIONS.md`. Everything else in the package is preview: it works and is tested, but
its spelling or behaviour may change in a minor release, and the change is listed in
`CHANGELOG.md`.

## The stable core

| Area | Names |
|---|---|
| Capture | `tl.trace(model, x, ...)` and the `tl.Trace` it returns |
| Records and lookups | `tl.Trace`, `tl.Layer`, `tl.Op`, modules; `trace["label"]`, `trace.layers`, `trace.ops`, `trace.modules`, and `layer.out` |
| Choosing what to save | the `save=` argument and the selectors `tl.func`, `tl.module`, `tl.label` (with `tl.label_contains`, `tl.label_matches`, `tl.in_module`) and their boolean combinations |
| Drawing | `trace.draw(...)` and `tl.draw(model, x, ...)` |
| Intervention | `trace.fork()` and `trace.do(site, edit)` |
| Comparing runs | `tl.Bundle` |
| Saving and loading | `trace.save(path)` and `tl.load(path)` |
| Checking a capture | `tl.validate(model, x)` |
| Reading activations | `tl.extract(model, x, layers)` (many layers) and `tl.pluck(model, x, layer)` (one layer) |
| Lightweight recording | `tl.record(model, x, save=...)` and the `tl.Recording` it returns |

Error and warning codes listed in `docs/reference/error_refusal_contract.md` and
`docs/reference/warning_contract.md` are also stable: code that branches on
`error.fields["code"]` keeps working across 3.x.

## Preview

Every other public name is preview, including the backends other than PyTorch, the
experiment helpers (`tl.sweep`, `tl.steer`, ablation and projection edits), the
appliance and bridge namespaces, and anything a docstring or the glossary marks
experimental. Use them freely; pin a TorchLens version if you depend on their exact
spelling.
