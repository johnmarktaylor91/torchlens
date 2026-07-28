## CAPABILITY PROBE -- prove this author path can actually research

You are not authoring a model. The crawler's strict preflight (`doctor`) is checking that
the author path it is about to run 28,482 models through genuinely has live web research
tools -- `WebSearch`, `web_search_exa`, and `web_fetch_exa` -- and not merely their names
in a help text. A campaign started without them produces ungrounded proposals for a month.

The nonce and the challenge package in JOB FACTS were derived seconds ago from a fresh
random value, so nothing about this can be precomputed or cached. Answer it by **actually
using each tool**.

### First: find the tools' real names

`WebSearch`, `web_search_exa`, and `web_fetch_exa` are **canonical** names, and only that
suffix is stable. The Exa tools arrive over MCP, and the name they are actually *registered*
under carries a namespace prefix that depends on how this session was launched -- for example
`mcp__exa__web_search_exa` under an explicit `--mcp-config`, or
`mcp__plugin_everything-claude-code_exa__web_search_exa` when the settings tree loads them from
a plugin. Look through the tool names you actually have and **match on the suffix**: any
registered name ending in a canonical name after a `__`, `.`, `:`, or `/` separator *is* that
tool. Call it under its registered name, and record that spelling as `registered_tool_name`.

**A name mismatch is NOT a missing tool, and reporting one as missing is a false failure.**
This probe is the gate on a month-long campaign; failing it because a namespace changed would
block a perfectly good author path. Search the registered names before you conclude anything.
The reverse error is worse: a tool you never actually called is not a tool you have.

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
      "registered_tool_name": "WebSearch",
      "nonce": "<nonce>", "observed_at": "<ISO-8601 Z, when you ran it>",
      "query": "...", "reported_version": "X.Y.Z",
      "results": [{"url": "https://...", "title": "...", "excerpt": "..."},
                  {"url": "https://...", "title": "...", "excerpt": "..."}]
    },
    "web_search_exa": {
      "registered_tool_name": "<the exact name you called, e.g. mcp__exa__web_search_exa>",
      "nonce": "<nonce>", "observed_at": "<ISO-8601 Z>",
      "query": "...", "reported_version": "X.Y.Z",
      "results": [{"url": "https://...", "title": "...", "text": "<>=200 chars of real page text>"},
                  {"url": "https://...", "title": "...", "text": "..."}]
    },
    "web_fetch_exa": {
      "registered_tool_name": "<the exact name you called, e.g. mcp__exa__web_fetch_exa>",
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
- The evidence keys above are the canonical names; keying an entry by its registered name
  instead is accepted and resolved to the same tool. Either way, `registered_tool_name` must
  be the exact spelling you really called.
- If a tool is **missing, unconfigured, disconnected, permission-blocked, or erroring** --
  after you have searched the registered names for its canonical suffix -- say so plainly,
  name the spelling you tried and the exact error, and write no evidence for it. That is
  exactly the finding the doctor is looking for.
- Do not write anywhere except `required_output_path`. Do not touch the repository.
- Report the exact number of tool calls you made.
