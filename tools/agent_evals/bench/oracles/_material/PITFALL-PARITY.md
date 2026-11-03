# Pitfall parity across arm files

The plain-PyTorch file was written by the task authors after the tasks, so its
pitfalls line up with the tasks' traps. To keep that from favouring one arm, every
arm file must cover the same task-relevant traps, each in its own toolkit's terms.
Check a new arm file (including the library-under-test's own index) against this
table before any run uses it; a missing row is added to that file and recorded in
its `NOTES.md` as an addition by the benchmark authors.

| Trap | Plain PyTorch | nnsight | TransformerLens |
|---|---|---|---|
| In-place ops overwrite a kept value; clone | Pitfall 1 | Pitfall 8 (added) | Pitfall 8 (added) |
| Module outputs can be tuples | Pitfall 3 | Pitfall 3 | Reading (added, for unwrapped models) |
| Last hidden state is after the final norm | Pitfall 4 | Reading (added) | Reading (added) |
| Values with no module or hook boundary | Reading | Reading (`.source`) | Reading |
| Many vector-Jacobian products in one backward | Gradients | Gradients (added) | Gradients (added) |
| Gradient through a frozen model needs a graph root | Gradients, Pitfall 6 | Gradients (`.grad` inside `backward()`) | Gradients (example fixed) |
| Padding shifts positions | Pitfall 5 | Batching | Batching, Pitfall 1 (BOS) |
| Generation: cached steps see only the new position | Changing, Pitfall 8 | Changing (`tracer.iter`) | Changing |
| Interventions left installed | Pitfall 7 | not applicable (scoped to a trace) | Pitfall 5 |
| Same batch size as the reference (about 1e-5 kernel drift) | Batching | Batching (added) | Batching (added) |

Checked 30 Oct 2026 against the files at this commit. Every file stays within the
template's 8,192 bytes.
