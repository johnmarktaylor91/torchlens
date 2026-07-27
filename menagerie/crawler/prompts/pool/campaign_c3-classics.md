## CAMPAIGN c3-classics -- unregistered classics (author tier: opus-5)

~5,487 historically important architectures with **no prior public implementation**. This
is the campaign the whole atlas is for, and the only one dispatched at the top tier. These
rows are here precisely because a cheap pass cannot do them: the source material is an
old paper, sometimes a thesis, sometimes a scan, often not in English, and the forward
pass has to be recovered from prose, a diagram, and a table.

Take the time. The budget is ~25 minutes per model and it is meant to be spent.

### The ladder is still the ladder

R4_REIMPLEMENT is the *expected* rung here, but it is still the **fourth** rung. Before
you take it, establish and record that you looked for:

- any surviving code by the original authors, under any name;
- a third-party implementation in any framework, including ones that predate PyTorch;
- code embedded in a thesis, a supplement, a course repository, or a reproduction study;
- the original (frequently non-English) publication, whose appendix routinely specifies
  what the translated summary omits.

Rung 3 beats rung 4: if real code exists anywhere and merely cannot run, transcribe it
faithfully rather than rebuilding it from prose.

### What a faithful reimplementation means

Every material forward choice must be **specified by the primary material and cited to
it**: layer sequence and counts, channel and unit dimensions, kernel sizes, strides,
padding, activations, normalization, connectivity and skip structure, recurrence and
state, initialization, input semantics, and output behavior.

If the paper does not specify something material, you do **not** choose it. Record it as a
sufficiency gap, keep the verbatim text that leaves it open, and let the result be
`R5_SKIP` or a documented gap. An architecture that is 90% cited and 10% invented is not a
historical reconstruction; it is a fabrication wearing a citation.

A gist, an abstract, a blog summary, a name, or a generic block diagram is **never**
sufficient.

### Evidence discipline at this tier

- Quote the specifying sentence or table cell verbatim for each material choice, with page
  or section locator, and map it to the code you stage.
- Where the paper is ambiguous and the ambiguity is *resolvable* from a later erratum,
  a follow-up paper, or the authors' own subsequent work, cite that explicitly as the
  resolution rather than folding it in silently.
- Where two readings are both defensible, say so, pick the one the text better supports,
  and record the alternative. The Codex checker will read the same page.

### Do not

- Do not fill a gap to make the model constructible. `BLOCKED` at the missing choice is
  the correct answer.
- Do not fetch pretrained weights; there are none and looking for them wastes the budget.
- Do not shrink a fixed architectural dimension for convenience. Batch 1 and the smallest
  *source-valid* variable dimensions only.
