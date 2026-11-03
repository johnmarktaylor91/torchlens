# Composition matrix

<!-- Generated from tests/composition_expectations/ledger.py by
render_public_matrix(); do not edit by hand. The lockstep test
(tests/composition_expectations/test_contract_lockstep.py) fails on drift. -->

Which feature pairings have been exercised together, and with what result.
Every pairing in the composition ledger appears here. A pairing is listed as
`supported` only when a test proves it; `refuses with a typed error and a remedy`
means the combination is rejected on purpose, with the remedy in the message;
`not yet verified` means the pairing is known and tracked but no test yet proves
its behaviour, so do not rely on it. The evidence column names the test that
proves a verified row.

| Pairing | Family | Operations | Settings | Status | Evidence |
|---|---|---|---|---|---|
| A postprocess-label selector under the rerun engine | intervention | trace_option:intervene:torch, fork.do | selector=postprocess-label; engine=rerun | refuses with a typed error and a remedy | tests/test_rerun_push_head_ablation_parity.py |
| The save-all refusal names a current remedy | capture-options | tl.trace | save=all; remedy=currency | not yet verified | - |
| Replay precondition errors carry a code | diagnostics | fork.do | error=ReplayPreconditionError; capture=default | not yet verified | - |
| A selector matching no site refuses with a code | diagnostics | fork.do | error=SiteResolutionError; match=zero | not yet verified | - |
| Zero-match handling agrees between capture and plan | lane-parity | tl.trace, fork.do | lane=selector-vs-plan; match=zero | not yet verified | - |
| Trace.draw refuses a removed keyword | render | Trace.draw | kwarg=vis_mode; view=False | not yet verified | - |
| A mistyped top-level name gets a suggestion | entry | tl.__getattr__ | attr=explain|utils|bridge; surface=module-facade | not yet verified | - |
| Disk storage with explicit streaming | capture-options | tl.trace, trace_option:storage:torch | storage=to_disk; streaming=explicit | not yet verified | - |
| MCP overview cache under a symlinked path | agent | bridge.mcp.load_overview | path=symlinked; cache=keyed-by-path | not yet verified | - |
| Every boolean environment variable uses the closed parser | environment | closed_bool_env | vars=3-bypassing; parser=closed_bool_env | not yet verified | - |
| Agent verbs on a partial trace | product-surface | Trace.to_agent_json, trace.explain | product=PartialTrace; verb=agent-doors | not yet verified | - |
| A bare label on a multi-pass layer | intervention | fork.do | label=bare; layer=multi-pass | refuses with a typed error and a remedy | tests/test_replay_pass_qualified.py |
| Raw in-place writes on a fork reach shared storage | product-surface | Trace.fork | write=raw-inplace; payload=shared-storage | supported, with a disclosure | tests/composition_expectations/test_galleries.py::test_raw_fork_writes_reach_the_shared_storage |
| Summary pass counts across ASCII and Unicode | report-surface | Trace.summary, tl.summary | charset=ascii-vs-unicode; layer=multi-pass; view=ladder | supported | F08: tests/test_summary_rebuild_charset.py::test_dual_charset_goldens + tests/test_summary_rebuild_ladder.py::test_multipass_one_row_at_module_grain (byte-exact dual-charset goldens; one row owns all pass events; ascii == degrade(unicode)) |

Rows: 14; not yet verified: 10.

The matrix covers the hand-declared pairings of the composition ledger. Generated
pairings (the matrix, sweep and gallery generators in the test tree) are checked by
their own tests and are not listed here. A `not yet verified` row moves to a verified
state only through a test; the ledger refuses a verified state without evidence.
