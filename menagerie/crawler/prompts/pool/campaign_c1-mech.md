## CAMPAIGN c1-mech -- library-first residual tier (author tier: sonnet)

6,968 models. Most of them do live in a maintained public zoo -- `timm`,
`segmentation_models_pytorch`, `transformers`, `ultralytics`, `diffusers`, `mmdetection`,
`torchvision`, and a long tail of smaller registries. But this partition is drawn by
*exclusion*, not by evidence: it is every roster row whose `zoo` string is not one of four
reserved literals, and it carries 716 distinct zoo strings. Roughly one row in six is not
an R1 library row.

**The `zoo` label will not tell you which.** On a sampled adjudication of this exact
partition, about 45% of the rows whose zoo string looked hand-built, taxonomic, or
repo-shaped turned out to have a real installed-library constructor after all -- and the
converse happens too. `zoo` records where a row was harvested, never where the
architecture is defined. Establish the rung; do not inherit it.

### R1 is the most common rung, not the required one

- When the architecture ships materially unmodified in a maintained library, the proposal
  is that library's own declarative recipe: exact constructor, exact entrypoint name, exact
  keyword arguments, every pretrained/weights/checkpoint flag explicitly disabled.
- Establish with evidence that the installed class **is** the published architecture: same
  paper, same variant, same configuration. A same-named entrypoint that is actually a
  different variant is the characteristic failure of this class.
- Pin the library version and the exact symbol path. "timm has it" is not a source;
  `timm==<version>` plus `timm/models/<file>.py::<symbol>` is.

### R2 is a normal, successful outcome of this campaign

An R2_VENDOR proposal here is a **win, not an escalation and not a downgrade.** A large
share of this tier's non-R1 rows are ordinary vendor-repo work: the upstream repository
exists, it runs, and the only thing R1 lacked was a packaged release. Author it in tier:

- Pin the repository and the exact revision the broker resolved, and name the exact model
  file you read.
- Stage a typed adapter defining `build_model()` and `make_dummy_call(seed, device)`, per
  the canonical author prompt. That adapter is genuinely more work than a declarative
  recipe. That extra work is the expected shape of an R2, not a signal that the model
  belongs to another tier.
- The fidelity bar does not move one millimetre. R2 means **the real upstream code at a
  pinned revision**. Cheaper routing never buys a cheaper source: an "approximation"
  written because the real code was inconvenient is slop and is forbidden outright.

The ladder is still evaluated in order. R1 wins whenever it genuinely applies; reach R2
when it genuinely does not.

### `needs-higher-tier` means reasoning depth, not workflow cost

Reserve `NEEDS_HIGHER_TIER` (stage 1) and `BLOCKED(stage=author,
reason_code=needs-higher-tier)` (after fetch) for the small tail where a *smarter
adjudicator* is the thing actually missing -- typically an R3_PORT or R4_REIMPLEMENT whose
variant fidelity cannot be settled at this tier. That tail is on the order of a hundred
rows in this campaign, not a thousand.

Promotion sends the model to C3/Opus at a larger grant. A merely-laborious R2 does not need
a smarter model, so promoting one buys nothing and costs a great deal. Concretely, none of
these is a higher-tier signal on its own:

- the row is not in a library;
- the source is a research repository rather than a package;
- the adapter, the vendoring, or the plumbing is tedious;
- the session is running long.

### Do not

- Do not write a from-scratch reimplementation of something a library or a repository
  already ships. If the real source exists, use the real source.
- Do not silently escalate a hard model in place. This campaign's author identity is frozen
  to sonnet for its entire run; the typed arms above are the only escalation, and they
  carry the bounded research summary.
- Do not treat a non-PyTorch-looking `zoo` (Paddle, ONNX, darknet) as a verdict. Such a row
  is frequently a downstream export of a real PyTorch repository; look for that repository
  before concluding anything about the framework.

### Budget shape

An R1 typically lands in a few minutes and well inside the tool-call grant. An R2
legitimately costs more, and is expected to. Spend the difference on the source, not on
re-deriving what the broker already resolved for you: the frozen manifest's revisions and
digests are machine facts to be cited verbatim.

If the wall deadline is genuinely approaching, the typed budget response is a `BLOCKED`
result naming exactly what you could not establish -- that flows through the terminal
gate and can be requeued. It is not `needs-higher-tier`, which is a claim about the model
rather than about the clock.
