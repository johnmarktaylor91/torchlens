# Activation Transform Persistence

TorchLens persistence distinguishes portable recipe state from executable Python callables.

## Runtime Behavior

- `activation_transform` and `gradient_postfunc` can be ordinary Python callables during capture.
- Callable outputs can be saved as tensor payloads when the selected save level stores them.
- The callable object itself is not made portable unless it is represented by a supported built-in
  helper or import reference.

## Portable Save Policy

For portable `.tlspec/` saves:

| Item | Portable behavior |
| --- | --- |
| Built-in helpers | Persisted as structured recipe data. |
| Tensor values produced by transforms | Persisted when included by the save level. |
| Opaque Python callables | Dropped from portable recipe execution. |
| Callable display text | `repr(callable)` is retained for audit/readback context when available. |

This is the callable-drop / repr-keep behavior: a portable artifact should not execute unknown
Python code, but it should still explain which callable was present at capture time.

## Audit or Local Execution

Use an audit or executable-with-callables level only when the receiving environment is trusted and
has the same callable definitions importable. Portable artifacts are the default for sharing across
machines or repositories.
