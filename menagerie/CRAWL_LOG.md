# Menagerie Crawl Log

This is the durable record of **when the menagerie roster was last exhaustively crawled** for new
architecture families. Discovery agents should read it (and its machine-readable companion
`data/crawl_history.json`) at the start of a sweep so they can focus on what is genuinely new.

> **Last exhaustive crawl: 2026-06-18.**
> A new sweep should PRIORITIZE architecture families first published **after 2026-06-18** (the recent
> big-conference / journal / arXiv frontier) — while remaining free to surface anything missed in ANY
> earlier era, language, or field. The machine-readable source of truth is
> `menagerie/data/crawl_history.json` -> `last_exhaustive_crawl`.

## How to use this when running a sweep
1. Read `last_exhaustive_crawl` from `data/crawl_history.json`.
2. Seed candidates from everything published since then: `python -m menagerie.discover_crawler`
   (it defaults its date window to the last-crawl date).
3. Run the adversarial sweep prompt in `DISCOVER_MODELS.md` (recency-first, but every-axis and
   non-English).
4. Fold finds into the catalog / `classics/` per `DISCOVER_MODELS.md`.
5. **Append a new entry to `data/crawl_history.json` and bump `last_exhaustive_crawl`** to the sweep date,
   and add a one-line row below.

## History
| Date | Type | Result (models / families / classics / verified) |
| --- | --- | --- |
| 2026-06-18 | Exhaustive multi-axis sweep + cross-lab adversarial red-team (non-English included) | 10,742 / 3,746 / 312 / 6,127 |
