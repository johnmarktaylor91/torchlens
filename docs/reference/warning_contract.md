# Warning contract (S-18)

TorchLens warnings that steer user decisions carry a stable machine-readable
`code` in `warning.fields["code"]` and a structured remedy in
`warning.fields["remedy"]` (derived at the one `TorchLensWarning` chokepoint
from the authored `Remedy: ...` message tail, exactly like the error base).
Consumers branch on `fields["code"]`, never on message text.

This table is the closed vocabulary of CONTRACTED warning codes. The
alias-resolving warn-site census and the lockstep gates live in
`tests/composition_expectations/` (S-18): every `code=` passed at a
`warnings.warn` construction site must have a row here, every row here must
resolve to a live warn site, and the uncoded-site count is a monotone
ratchet burning down to zero. A warning attached to a wrong or unusable
result is a GAP, not a disclosure (compo memo D3); silently overriding an
explicit user request is a typed conflict, never a warning.

| Code | Warning | Remedy class |
|---|---|---|
| `rerun_zero_fire` | A rerun hook plan entry fired at zero sites on the new inputs; the rerun completed but those interventions were silent no-ops (the flagship silent-wrongness signal, compo memo 5.4) | Resolve the target sites against the rerun trace (`trace.resolve_sites`) before re-applying, or route the edit through the push engine (`fork().do(...)`), which validates sites at plan time |

Adding or renaming a warning code updates this table, the
`docs/reference/error_refusal_contract.md` code table (the package-wide
`code=` literal scanner reads that one), and the S-18 lockstep baseline in
the same change.
