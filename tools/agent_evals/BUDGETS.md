# Agent-surface budgets: method and measured basis

The numbers in `torchlens/agent/_budgets.py` are configuration, set from this measurement and
revisited when it changes. None of them is a guaranteed API limit: each result's `limits` block
carries the values a call used, and a trimmed response always says so in `truncation`.

## Method

`measure_budgets.py` runs over the classics test corpus (`tests/classics_corpus/`, all tiers).
For each entry it builds the model and input with the corpus loader, captures with
`tl.trace(model, inputs, save="all")`, saves a `.tlspec`, and then:

- calls each tool's handler directly, so the token backstop trims nothing, and records the
  response size in estimated tokens (the surface's own estimator, characters / 4, rounded up):
  `torchlens_overview` (folded and manifest), `torchlens_dump` (`overview`, `graph`, `full`),
  `torchlens_query_sites` (no query: every site), `torchlens_payload_stats` (every saved site);
- records the plain `tl.report.explain(trace)` size with no token budget;
- for the row tools, records the rows returned at the row cap, the uncapped row total, and the
  mean and largest per-row token cost;
- reads the manifest's declared payload bytes: the largest single payload and the artifact total;
- calls each tool once more through `call_tool` (the served path) and records whether the served
  response was trimmed.

Per budget the summary reports the median, 90th percentile and maximum over the entries, and how
many entries exceed the configured value. Where one budget covers several calls, each entry
contributes its largest value. Run it with:

```bash
python tools/agent_evals/measure_budgets.py --tier all --out budgets.json
```

Token counts are estimates (characters / 4), not a tokenizer's count. The corpus is small
hand-built models (median 50 ops, largest 573); large pretrained models produce longer graphs and
larger payloads, so the byte ceilings are memory protection that this corpus does not reach.

## Measured basis (2026-10-14, 205 of 207 entries)

| Budget | Configured | Measured quantity | Median | p90 | Max |
| --- | ---: | --- | ---: | ---: | ---: |
| `ORIENTATION_MAX_TOKENS` (explain) | 4,000 | explain report tokens | 361 | 388 | 914 |
| `OVERVIEW_MAX_TOKENS` (folded overview) | 12,000 | folded overview tokens | 3,510 | 10,745 | 21,313 |
| `ROW_TOOL_MAX_TOKENS` (row-tool backstop) | 25,000 | `dump full` tokens at the old row cap of 200 | 8,326 | 23,727 | 60,817 |
| `ROW_TOOL_MAX_TOKENS` (row-tool backstop) | 25,000 | largest row-tool response at the row cap of 150 | 8,326 | 21,093 | 48,551 |
| `DEFAULT_MAX_ROWS` (row cap) | 150 | per-row tokens, mean per call (query rows) | 120 | 142 | 332 (one row) |
| `MAX_BLOB_BYTES` | 64 MiB | largest declared payload | 10 KB | 558 KB | 12.8 MB |
| `MAX_CALL_BYTES` | 128 MiB | whole-artifact declared payload | 145 KB | 4.7 MB | 106 MB |

What changed on this measurement:

- The folded overview had shared the 4,000-token orientation default and passes it from about
  43 ops, so the served overview was trimmed on 86 of 205 entries. It now has its own
  12,000-token default, the same as `torchlens_api_map`; 11 of 205 are still trimmed, with the
  trim disclosed and a next call named.
- The row cap was 200. At the measured per-row cost a full 200-row page passed the
  25,000-token backstop, so the backstop rather than the row cap was choosing the page: the
  first page lost rows to the backstop on 16 entries (`dump full` 16, `dump graph` 10,
  `query_sites` 15, `payload_stats` 0). At 150 rows that falls to 9 entries (9, 2, 5, 0); the
  rows still reach the caller through `data.next`. More entries now take a second page (38
  against 18), which is the row cap doing its job.
- The explain default (4,000) and both byte ceilings stay: the explain report peaks at 914
  tokens, and the largest declared payloads sit far below 64 MiB and 128 MiB.

Two entries (NeoDTI, ADL-MVDR) are not measured: the folded overview fails on them while
listing the trace's input operations, because each has an input layer with two passes.
