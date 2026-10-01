# Composition ledger (generated -- do not edit)

Regenerate: update `tests/composition_expectations/ledger.py`, then
copy the output of `render_projection()` over this file (the
lockstep test prints the drift). State, oracle, risk, and gap teeth
render TOGETHER so SUPPORTED cannot hide an unusable cell.

| Row | Family | State | Oracle | Risk tags | Axes | Owner/Evidence |
|---|---|---|---|---|---|---|
| CELL-FLAGSHIP-RERUN-NOOP | intervention | KNOWN-GAP | DIFFERENTIAL | silent_noop, lane_asymmetry | selector=postprocess-label; engine=rerun | A04 (due PB1) |
| CELL-SAVE-ALL-REMEDY-CURRENCY | capture-options | KNOWN-GAP | REFUSAL | remedy_currency | save=all; remedy=currency | A06 (due PB1) |
| CELL-REPLAY-PRECONDITION-CODELESS | diagnostics | KNOWN-GAP | REFUSAL | facade_teaching | error=ReplayPreconditionError; capture=default | compo(S-17) (due S-17 ratchet burn-down) |
| CELL-SITE-RESOLUTION-ZERO-MATCH-CODELESS | diagnostics | KNOWN-GAP | REFUSAL | facade_teaching | error=SiteResolutionError; match=zero | compo(S-17) (due S-17 ratchet burn-down) |
| CELL-ZERO-MATCH-LANE-ASYMMETRY | lane-parity | KNOWN-GAP | DIFFERENTIAL | lane_asymmetry, silent_noop | lane=selector-vs-plan; match=zero | A04 (due PB1) |
| CELL-DRAW-WRONG-KNOB-REFUSAL | render | KNOWN-GAP | REFUSAL | facade_teaching | kwarg=vis_mode; view=False | C05 (due PB2a) |
| CELL-FACADE-NO-DID-YOU-MEAN | entry | KNOWN-GAP | REFUSAL | facade_teaching | attr=explain|utils|bridge; surface=module-facade | A10 (due PB1) |
| CELL-STORAGE-STREAMING-SILENT-CONFLICT | capture-options | KNOWN-GAP | DIFFERENTIAL | accepted_then_discarded | storage=to_disk; streaming=explicit | A06 (due PB1) |
| CELL-MCP-CACHE-KEY-SYMLINK | agent | KNOWN-GAP | INVARIANT | untrusted_input | path=symlinked; cache=keyed-by-path | A09 (due PB2a) |
| CELL-ENV-PARSER-BYPASS | environment | KNOWN-GAP | INVARIANT | facade_teaching | vars=3-bypassing; parser=closed_bool_env | A10 (due PB1) |
| CELL-PARTIALTRACE-AGENT-VERBS | product-surface | KNOWN-GAP | CONTENT-FLOOR | container_shape | product=PartialTrace; verb=agent-doors | A09 (due PB1) |
| CELL-MULTIPASS-BARE-LABEL-REFUSAL | intervention | REFUSES-TYPED-TEACHING | REFUSAL | pass_resolution | label=bare; layer=multi-pass | tests/test_replay_pass_qualified.py |
| CELL-FORK-RAW-WRITE-SHARED-STORAGE | product-surface | SUPPORTED-WITH-DISCLOSURE | DISCLOSURE | shared_storage | write=raw-inplace; payload=shared-storage | tests/composition_expectations/test_galleries.py::test_raw_fork_writes_reach_the_shared_storage |

KNOWN-GAP rows: 11 of 13 (transitional; each carries owner, issue, deadline, reproducer, auto-probe).
