# autoroute/ - Implementation Guide

Auto-routing registries for model INPUT preprocessing and OUTPUT decoding
(lazy public submodule, `tl._autoroute`). The output registry runs twice: the
automatic, option-free pass at the end of every capture (`decode_outputs_for_trace`,
live outputs, `meta["style"] is None`) and the read-time door
`Trace.decode_output(style=, head=, tokenizer=)` (`decode_trace_output`, retained
output tensors keyed by head). Public spellings of the read-time door live in
`_decode_spellings.py`; `CaptureOptions` has no decode field.

## Files

| File / dir | Purpose |
|------------|---------|
| `__init__.py` | Exposes exactly the two direction namespaces: `input` and `output` |
| `_registry.py` | Direction-agnostic priority `Registry` + frozen `Detector` metadata; glob (`fnmatch`) matching |
| `_builtin_input.py` / `_builtin_output.py` | Built-in detector implementations registered into the two namespaces |
| `input/__init__.py`, `output/__init__.py` | Per-direction public registry surfaces — identical 9-name `__all__`: `Detector`, `Registry`, `register`, `unregister`, `list`, `info`, `iter_by_priority`, `snapshot` |
| `data/` | Bundled label data (`imagenet1k_labels.json`; see its README) |

## Gotchas

- `Detector` is a frozen dataclass; registration is priority-ordered and
  iterated via `iter_by_priority`.
- `snapshot()` is the context-manager save/restore spelling — tests that
  mutate a registry must restore through it rather than hand-editing.
- `list` in the per-direction `__all__` shadows the builtin name by design;
  import the namespace (`from torchlens._autoroute import output`), not the
  bare names.

## Tests

`tests/test_autoroute_registry.py`, `tests/test_autoroute_output_a2.py`.
