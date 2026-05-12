# Post-backward Megasprint P4 Done

## Summary

Implemented P4 stack-append surgical gaps on `codex/post-backward-megasprint`.

- Added `AppendStreamingNotSupportedError` and hard-rejected append rerun while `_out_writer` or `_out_sink` is active.
- Added `AppendStateValidationWarning` and validator skip/warn behavior for appended Trace-like inputs.
- Clarified no-helper grad append failure messaging.
- Added `append_history` provenance for appended chunks.
- Reset `is_appended`, `_append_sequence_id`, and `append_history` on non-append rerun.

## Commits

- `423b5ee fix(intervention): reject append=True on streaming-active trace`
- `2271de7 fix(validation): backward validator handles stacked traces`
- `28b69a3 docs(intervention): clarify AppendBatchDependenceError`
- `2be4241 fix(model-log): clear is_appended on non-append rerun`

## Verification

Full gate output is in the final implementation response.
