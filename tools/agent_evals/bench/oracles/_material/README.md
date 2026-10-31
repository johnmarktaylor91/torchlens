# Arm material

One context file per arm, all built on `TEMPLATE.md` and all at most 8,192 bytes:

| Arm | File | Bytes | Source |
|---|---|---|---|
| A: plain PyTorch | `arm-a-pytorch.md` | see `wc -c` | written by the benchmark authors |
| nnsight | `nnsight/nnsight.md` | see `wc -c` | maintainers' agent skill, pinned; `nnsight/NOTES.md` |
| TransformerLens | `transformerlens/transformerlens.md` | see `wc -c` | assembled from public docs; `transformerlens/NOTES.md` |

Raw snapshots live beside each file under `raw/` and are not given to agents.
Material for any further library arm follows the same template and size and
records its provenance in a `NOTES.md` the same way.

`PITFALL-PARITY.md` lists the task traps every arm file must cover and where each
file covers them; check any new arm file against it before a run uses it.
