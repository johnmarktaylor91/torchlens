## CAMPAIGN c2-disco -- discovered PyTorch tail (author tier: sonnet)

~14,798 models discovered by crawling the literature and public repositories. This is the
campaign's biggest class and its most uneven: some rows are a well-maintained repo away
from R2, and a large fraction are dead links, renamed projects, one-off research dumps, or
things that were never a trainable network at all.

### The work is triage, and triage is a real finding

Most of the value here is deciding *correctly and with evidence* which of these is true:

- **R2_VENDOR** -- the upstream repository exists and runs; pin the revision and the exact
  model file.
- **ENV_SETUP** -- the code exists but needs a dependency set the assigned intent does not
  have. Say which packages, precisely. This feeds the requeue path; a vague "needs deps"
  wastes the finding.
- **R3_PORT** -- real code exists but genuinely cannot run on either planned target.
  Apple/CUDA incompatibility is a **deferral**, not a licence to port.
- **R5_SKIP** -- there is no usable code *and* no description detailed enough to specify
  the forward pass, or it is not a real trainable neural architecture. Record the searches
  you ran, where you looked, in which languages, and what you found -- a skip without a
  documented search is not a skip, it is a shrug.

### Before you conclude "nothing exists"

Check the non-obvious axes: the project's earlier name, the first author's personal or lab
page, thesis appendices, a non-English original paper, a framework port by a third party,
and archived/forked copies of a deleted repository. A large share of this campaign's
apparent dead ends are renames.

### Do not

- Do not write a from-scratch approximation to make a row "work". If real source exists,
  use it; if it does not, a faithful reimplementation is allowed **only** from a detailed
  primary description, and otherwise the answer is `R5_SKIP`.
- Do not escalate a hard model in place. This campaign is frozen to sonnet; emit a typed
  `BLOCKED` so the row can be requeued into `c3-classics`.

### Budget shape

Target roughly 6 minutes. Bound the search: two or three genuinely different angles, then
commit to a triage verdict with the evidence you have. An honest, well-evidenced
`R5_SKIP` is worth far more than a fifteen-minute hunt that ends in a guess.
