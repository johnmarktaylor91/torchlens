## Resolution

| Item | Status | Evidence |
|---|---|---|
| R4-F1 `source_loading_enabled` incomplete for full `FuncCallLocation` surface | PARTIAL | v5 does fix the specific round-4 gaps: D17 now enumerates all seven lazy accessors, requires preinitialized disabled-source values, and clears `_frame_func_obj` at construction to avoid the previously identified pickle risk; the strengthened `SaveSourceContextOffModel` also adds full-accessor and pickle-round-trip coverage. See `.project-context/plans/if-else-attribution/plan.md:67,71-92,421-423`. But the newly specified disabled-source contract now says `code_context == ""`, which conflicts with the current `Optional[List[str]]` API and current `__getitem__`/`__len__` behavior in `torchlens/data_classes/func_call_location.py:176-181,253-263` plus existing tests at `tests/test_metadata.py:891-897,971-1000`. That new mismatch means the full-surface contract is still not internally coherent. |
| R4-F2 `attribute_op` subsection contradicts D14 v2 | RESOLVED | The operative algorithm now matches D14 v2 exactly: scope resolution falls back to `(filename, code_firstlineno, func_name)` only on a unique match, and otherwise fails closed with no "smallest containing scope" heuristic. See `.project-context/plans/if-else-attribution/plan.md:64,281-289,573`. |
| R4-F3 `source_context == []` type contradicts current `str` API | RESOLVED | v5 aligns the disabled-source contract and the strengthened test with the current `str`-typed `source_context` surface by specifying `source_context == ""` rather than `[]`. See `.project-context/plans/if-else-attribution/plan.md:67,84-85,421-423,574` and `torchlens/data_classes/func_call_location.py:185-191`. |

## New-in-v5 findings

### Finding 1
- Severity: medium
- Description: the new disabled-source accessor spec now contradicts the current `code_context` API and still leaves `__getitem__` behavior underspecified.
- Why it matters: D17 and the strengthened disabled-source test now require `loc.code_context == ""` under `save_source_context=False` (`.project-context/plans/if-else-attribution/plan.md:67,84-85,421-423`). That does not match the current public surface, where `code_context` is `Optional[List[str]]`, existing tests assert `list | None`, `__getitem__` returns source lines / slices, and disabled source today is represented via `code_context is None` with `IndexError` from `__getitem__` (`torchlens/data_classes/func_call_location.py:62,81,176-181,253-263`; `tests/test_metadata.py:891-897,954-1000`). As written, an implementation worker can either preserve the current API or follow the new plan text, but not both. The same section also says `__getitem__` should "return the pre-initialized empty values" without defining whether integer indexing should raise, return `""`, or whether slicing should return `[]` or `""`.
- Concrete fix: make one explicit compatibility choice and use it consistently in D17, the data-model block, and `SaveSourceContextOffModel`. The conservative choice is: keep `source_context=""`, keep `code_context=None` when source loading is disabled, keep `__len__ == 0`, and specify that disabled `__getitem__` preserves the current `IndexError` behavior. If the API is intentionally changing instead, then the plan needs to say so explicitly and update the type contract, dunder semantics, and compatibility notes together.

## Summary: verdict = YELLOW

v5 cleanly resolves the round-4 contradiction in `attribute_op` and the `source_context` type mismatch, and it also closes the `_frame_func_obj` pickle-risk gap that made round 4 red. The remaining issue is narrower but still real: the new disabled-source state is not yet specified in a way that is compatible with the current `code_context`/`__getitem__` contract. That is a clear-cut spec fix, not a rethink, so this is no longer RED, but it is not GREEN either.

Counts: 0 blocker / 0 high / 1 medium / 0 low
