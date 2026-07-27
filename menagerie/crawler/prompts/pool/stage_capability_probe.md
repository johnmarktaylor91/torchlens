## CAPABILITY PROBE -- prove this author path can actually research

You are not authoring a model. The crawler's strict preflight (`doctor`) is checking that
the author path it is about to run 28,482 models through genuinely has live web research
tools -- `WebSearch`, `web_search_exa`, and `web_fetch_exa` -- and not merely their names
in a help text. A campaign started without them produces ungrounded proposals for a month.

The nonce and the challenge package in JOB FACTS were derived seconds ago from a fresh
random value, so nothing about this can be precomputed or cached. Answer it by **actually
using each tool**.

### Do exactly this

1. **`web_fetch_exa`** the challenge metadata URL from JOB FACTS, exactly as given. Keep
   the returned document body verbatim. Read out of it:
   - `info.version` -- the current released version string;
   - `last_serial` -- the integer release serial.
2. **`WebSearch`** for the same package's current version (for example
   `<package> pypi latest version`). Keep at least two ranked results, from at least two
   different hosts, with their URLs and titles.
3. **`web_search_exa`** for the same thing. Keep at least two results, at least one of
   which carries a substantial block of page text (Exa returns document content, not just
   a one-line snippet). Do not reuse the `WebSearch` result list -- run the actual query.

All three must independently agree on the version. That agreement on an unpredictable live
value is the proof; a disagreement means one of them did not really run.

### Write the evidence

Write ONE JSON object to the exact `required_output_path` from JOB FACTS. Echo the nonce
in every place shown:

```json
{
  "challenge_id": "<challenge_id from JOB FACTS>",
  "tools": {
    "WebSearch": {
      "nonce": "<nonce>", "observed_at": "<ISO-8601 Z, when you ran it>",
      "query": "...", "reported_version": "X.Y.Z",
      "results": [{"url": "https://...", "title": "...", "excerpt": "..."},
                  {"url": "https://...", "title": "...", "excerpt": "..."}]
    },
    "web_search_exa": {
      "nonce": "<nonce>", "observed_at": "<ISO-8601 Z>",
      "query": "...", "reported_version": "X.Y.Z",
      "results": [{"url": "https://...", "title": "...", "text": "<>=200 chars of real page text>"},
                  {"url": "https://...", "title": "...", "text": "..."}]
    },
    "web_fetch_exa": {
      "nonce": "<nonce>", "observed_at": "<ISO-8601 Z>",
      "url": "<the exact challenge metadata URL>",
      "content": "<the verbatim document body you received, >=800 chars>",
      "content_sha256": "<sha256 of exactly that content string>",
      "reported_version": "X.Y.Z",
      "reported_last_serial": 12345678
    }
  }
}
```

### Non-negotiable

- **Report only what the tools actually returned.** The pool validates the fetched
  document against its own declared digest, requires the reported version and serial to
  literally appear in it, requires the two search tools to have returned different result
  lists, and requires all three timestamps inside the probe window. Anything you did not
  genuinely observe will fail one of those checks -- and a probe that fails is the correct,
  useful answer. A fabricated pass would send a month-long campaign into the field with a
  research capability nobody verified.
- If a tool is **missing, unconfigured, disconnected, or erroring**, say so plainly and
  write no evidence for it. That is exactly the finding the doctor is looking for.
- Do not write anywhere except `required_output_path`. Do not touch the repository.
- Report the exact number of tool calls you made.
