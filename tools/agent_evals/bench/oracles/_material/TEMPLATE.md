# Arm material template

Every arm of the benchmark gives its subject agent exactly one context file built
on this template, so the benchmark measures the toolkit, not the quality of one
context file. The toolkit of arm A is plain PyTorch; the others are one
interpretability library each, used on top of plain PyTorch.

Rules:

- Size: at most 8,192 bytes (`wc -c`), about 1,500 to 2,000 tokens. Smaller is allowed;
  padding is not.
- Sections, in this order and with these headings. A section that does not apply
  says so in one line.
  1. `# <Toolkit>: agent notes` and a two-line statement of what the toolkit is for.
  2. `## Setup`: install and load a model (one code block).
  3. `## Reading intermediate values`: how to get an activation at a named place,
     several at once, and from inside a module's computation when it is not a
     module output.
  4. `## Changing intermediate values`: replace, add to or zero a value during a
     run; patch from another run; intervene during generation.
  5. `## Gradients and attribution`: a gradient with respect to an intermediate
     value; gradient times activation; a Jacobian or vector-Jacobian product.
  6. `## Batching, timing and determinism`: running many inputs or variants in one
     pass; keeping the overhead low; seeds and float32.
  7. `## Pitfalls`: the five to eight mistakes that most often break code written
     with the toolkit.
  8. `## Where to look`: at most five pointers (documentation URL, module paths,
     one inspection command).
- Code examples use GPT-2 small or a small torchvision CNN, on CPU, and are runnable.
- No comparison with any other toolkit, no marketing.
- Material taken from a toolkit's own maintainers is pinned to a dated commit or
  URL in the arm's `NOTES.md`, together with its licence and a list of what the
  benchmark authors trimmed or added to fit this template.
