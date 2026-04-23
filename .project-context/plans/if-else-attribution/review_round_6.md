## Resolution

| Item | Status | Evidence |
|---|---|---|
| R5-F1 disabled-source contract for `code_context` / `__getitem__` | RESOLVED | v6 now aligns all three normative sections with the current `FuncCallLocation` no-source contract. D17 specifies `code_context=None`, `source_context="None"` (literal string), and verbatim dunder behavior: `len==0`, `__getitem__` raises `IndexError`, `__repr__` ends with `"code: source unavailable"`; see `.project-context/plans/if-else-attribution/plan.md:67`. The `FuncCallLocation` data-model comment repeats the same disabled-source state, including `code_context=None` and `loc[i]` raising `IndexError`; see `.project-context/plans/if-else-attribution/plan.md:71-94`. The strengthened `SaveSourceContextOffModel` row matches that contract exactly, asserting `loc.code_context is None`, `loc.source_context == "None"`, and `loc[0]` raises `IndexError`; see `.project-context/plans/if-else-attribution/plan.md:426`. This matches the live implementation at `torchlens/data_classes/func_call_location.py:149-150,176-178,253-257`. No contradiction remains across D17, the data-model block, and the test row. |

## New-in-v6

No new findings.

## Summary: verdict = GREEN

v6 resolves the round-5 medium cleanly and is implementable as written. The disabled-source contract is now internally consistent and matches the current `FuncCallLocation` API and dunder semantics.

Counts: 0 blocker / 0 high / 0 medium / 0 low
