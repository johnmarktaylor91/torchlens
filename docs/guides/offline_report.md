# The single-file offline HTML report

*(F16, treescope memo decision 4. Option spellings are [UI-SPRINT] /
DOCUMENTED-UNSTABLE.)*

```python
tl.export.html(log, "report.html")
```

writes ONE self-contained HTML file: no network, no CDN, no sibling files,
no Python needed to read it. Capture on a cluster, `scp` the file, open from
`file://`. The report is first-class for FAILED captures too — a shippable
failure report is the artifact for "it broke on the cluster".

The content doors are the flat keywords `arrays=` and `graph=`; emission
plumbing (redaction, byte-stability, renderer depth, thumbnail budgets)
rides the frozen `options=tl.export.ReportOptions(...)` bundle.

## Anatomy

1. **Sticky banner** — identity, capture outcome, and a mandatory embed
   disclosure ("array values EMBEDDED" / "no array values embedded" /
   "share_safe: every embedded value stripped").
2. **Capture honesty** — every disclosure from the shared honesty facts.
3. **Summary** — the summary lane's typed table, embedded, never rebuilt.
4. **Graph** — the REAL module-collapsed graph SVG with pan/zoom (wheel +
   drag; scrolls natively with JavaScript disabled), collapse depth
   disclosed. If the real layout cannot render, a `report_graph_fallback`
   warning fires and a clearly-labeled schematic canvas takes its place —
   the report never silently pretends a grid is the real geometry.
5. **Modules** — bounded module index.
6. **Ops** — one metadata row per op with stable anchors (`id="op-<label>"`).
7. **Arrays** — see policy below.
8. **Manifest** — schema id, TorchLens/torch versions, counts, array/graph
   ledgers (shown vs omitted), budgets, capture-honesty facts, timestamp
   (suppressed by `ReportOptions(deterministic=True)`).

## Array policy

Default `arrays="frontier"`: metadata for every op, but truncated thumbnails
ONLY for the diagnostic frontier — each minimal flagged site (first op whose
output went nonfinite with no flagged parents), its direct parents (the last
clean values), its direct children, plus intervention sites with one-hop
context. NaN propagates, so "all flagged ops" is unbounded (one injected NaN
flags 65% of distilgpt2's ops); the frontier IS the clean-to-dirty
transition. **Healthy captures embed zero array bytes by default.**

- `arrays="flagged"` — debugging preset: every flagged op, frontier-priority
  ordered, HARD-CAPPED by the same budgets with exact `K flagged, M shown`
  disclosure.
- `arrays=<tl.Selection>` — the general form: thumbnails for the resolved
  selection's sites.
- `arrays="none"` — metadata only.
- `options=tl.export.ReportOptions(share_safe=True)` — strips EVERY
  embedded value including the frontier.

Budgets (`ReportOptions(max_array_sites=..., max_array_bytes=...)`, plus
the thumbnail cell budget) are enforced BEFORE payloads are written;
omissions land in the manifest.

## Graph options

- `graph="collapsed"` (default) — module-collapsed real graph, depth
  disclosed in the manifest; over-budget renders collapse harder, then omit
  with disclosure.
- `graph="flat"` — try the uncollapsed layout first (same byte ladder).
- `graph="none"` — omit, disclosed.
- `ReportOptions(vis_call_depth=...)` — forwarded to the renderer and
  disclosed.

The full flat graph is never linked as a sibling file (one-file contract);
use the separate `tl.export.svg` door when you want it.

## Security and privacy

All text is escaped (embedded JSON escapes `<`, so `</script>` cannot break
out); absolute home paths are scrubbed to `~`; no foreign `_repr_html_` is
ever embedded; a restrictive CSP meta tag ships in the head
(`default-src 'none'`); the only scripts are clipboard + pan/zoom, and the
document reads fully without them.
