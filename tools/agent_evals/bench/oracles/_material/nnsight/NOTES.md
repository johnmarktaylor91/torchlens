# nnsight arm material: provenance

Source: the maintainers' own agent skill, `plugins/nnsight/skills/nnsight/SKILL.md` in
https://github.com/ndif-team/skills at commit `c8ef0ab7d3a051424ff5fab4ad1fba1b07a6f819`
(committed 2026-10-19), snapshot taken 2026-10-30, kept verbatim in `raw/SKILL.md`
(11,473 bytes). `raw/llms.txt` is https://nnsight.net/llms.txt on the same date.
Licence: MIT (ndif-team/skills, Copyright 2026 The NDIF Team; nnsight itself MIT,
Northeastern University).

What the benchmark authors did to fit the template (8,192 bytes):

- Re-ordered the maintainers' text under the template headings; the examples,
  the "five things that break agent-written nnsight code" and the run-method table
  are the maintainers' wording, lightly shortened.
- Trimmed: the NDIF remote-execution section, the references table (ten files
  that ship with the skill), the related-skills list beyond a one-line pointer,
  and the `check_env.py` script.
- Added (marked here, not in the file): the two-line patch-from-another-run example
  in "Changing intermediate values", the steering line inside the `tracer.iter`
  example, the `.source` sentence in "Reading intermediate values", the
  vector-Jacobian sentence in "Gradients and attribution", and the determinism
  sentence. These were written from the skill's own references and the nnsight
  documentation, not from any other toolkit.
- The skill's companion `references/*.md` files and the technique skills
  (`activation-patching`, `causal-tracing`, ...) are not included: every arm gets
  one file of the same size. The harness may offer them as an explicitly reported
  "extended material" arm if wanted.

Pitfall parity (30 Oct 2026, see `../PITFALL-PARITY.md`): added the final-norm sentence
in "Reading intermediate values", the batched vector-Jacobian sentence in "Gradients and
attribution", the batch-size sentence in "Batching, timing and determinism" and pitfall 8
(in-place overwrites), so this file covers the same task traps as the plain-PyTorch file.
