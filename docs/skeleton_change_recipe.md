# Recipe: Change a Model's Skeleton, Keep Your Analyses Aligned

You wrapped a module in an adapter, swapped a block for a variant, or inserted
a new layer -- and you have saved TorchLens analyses (intervention recipes,
site addresses, notes keyed to layer labels) that were authored against the
model as it used to be. This page is the supported workflow for carrying those
analyses across the edit.

One thing to be clear about up front: **no TorchLens verb edits your model.**
The skeleton change happens in your own code, on your own `nn.Module`, with
ordinary PyTorch -- assign a submodule, wrap a block, replace an attribute.
TorchLens's job starts after the edit: trace the edited model, then re-address
your saved analyses honestly, telling you exactly what still resolves, what
merely got renumbered, and what genuinely moved.

The workflow below runs end to end on a real GPT-2 model (the real
`transformers` classes, small config, no downloads). Every code block is
copy-runnable in order. The same flow works unchanged on pretrained
checkpoints -- swap the config-built model for
`GPT2LMHeadModel.from_pretrained("distilgpt2")`.

## Why labels break and site keys don't

TorchLens gives every captured op two kinds of address:

- a **recorded label** like `linear_1_121` -- readable, but its numbering is
  minted per capture. Insert one op anywhere upstream and labels renumber.
- a **structural site key** like `s1|lm_head|linear||1` (`op.site_key`) -- the
  op's structural position: which module site it ran in, its op type, and its
  occurrence ordinal within that call. Site keys do not move unless the site
  itself moves.

The compatibility checker joins your saved recipe to a fresh capture
**site-key-first**: labels are display-only disclosure, the structural key is
the join identity. A skeleton edit that renumbers every downstream label
without moving your target is reported as compatible label drift, never as a
lost target.

## Step 0: the skeleton change (your code, not TorchLens)

Build the original model and the edited variant. Here the edit wraps the
second GPT-2 transformer block in a user-owned adapter wrapper -- the classic
"insert an adapter without touching upstream code" move. Build both variants
before tracing anything.

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    n_layer=2,
    n_head=2,
    n_embd=64,
    vocab_size=512,
    n_positions=64,
    use_cache=False,
    bos_token_id=0,
    eos_token_id=0,
)
torch.manual_seed(0)
model = GPT2LMHeadModel(config).eval()


class BlockAdapter(nn.Module):
    """Run the wrapped GPT-2 block, then add a residual adapter."""

    def __init__(self, inner: nn.Module, width: int) -> None:
        super().__init__()
        self.inner = inner
        self.adapter = nn.Linear(width, width)

    def forward(self, *args, **kwargs):
        outputs = self.inner(*args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        hidden = hidden + self.adapter(hidden)
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden


torch.manual_seed(0)
edited = GPT2LMHeadModel(config).eval()
edited.transformer.h[1] = BlockAdapter(edited.transformer.h[1], config.n_embd)

input_ids = torch.randint(
    0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(1)
)
```

The edit is one assignment: `edited.transformer.h[1] = BlockAdapter(...)`.
Everything that used to live at module address `transformer.h.1.<child>` now
lives at `transformer.h.1.inner.<child>`, and a brand-new
`transformer.h.1.adapter` linear exists.

## Step 1: author the recipe with structural addresses, save it

Trace the original model and attach your analysis recipe with a **structural
selector** -- `tl.intervention.site(...)` addresses by module path, op type,
and occurrence, never by recorded numbering. Then save the recipe to disk.

```python
import torchlens as tl

recipe = tl.when(tl.intervention.site(module_path="lm_head"), tl.scale(0.5))
original_log = tl.trace(model, input_ids)
original_log.attach_hooks(recipe, confirm_mutation=True)
tl.io.save_intervention(original_log, "recipe.tlspec")
```

Two authoring choices matter for later alignment:

- **Address structurally.** `tl.intervention.site(module_path="lm_head")` is a
  structural address, valid on any capture of any model that has that site.
  A label selector like `"linear_1_121"` is a recorded address -- it names
  this capture's numbering and cannot survive renumbering.
- **Attach through the spec door.** `attach_hooks(recipe)` records your WHERE
  expression in the saved artifact, so a later capture can re-resolve it. The
  capture-time predicate door (`tl.trace(..., intervene=...)`) instead lowers
  the fired selector into per-site *label* targets before saving -- the saved
  spec discloses this as `spec_derived` -- and a label-lowered recipe cannot
  re-align after a skeleton edit, because the drifted labels no longer
  resolve. If you plan to carry a recipe across model edits, author it
  through `attach_hooks`.

During tracing you may see a `ScalarEscapeWarning` pointing into
`transformers` masking code; it is a capture-honesty disclosure from the
library internals and unrelated to this workflow.

## Step 2: trace the edited model and align

Trace the edited model, load the saved recipe, and ask the compatibility
checker whether the recipe still targets this graph.

```python
edited_log = tl.trace(edited, input_ids)
spec = tl.io.load_intervention_spec("recipe.tlspec")
compat = tl.validation.check_spec_compat(spec, edited_log)
print(compat.outcome)
print(compat.site_diff)
```

This prints:

```text
COMPATIBLE_WITH_CONFIRMATION
site-level diff:
  matched   s1|lm_head|linear||1
```

The verdict is `COMPATIBLE_WITH_CONFIRMATION`, not `EXACT`, because the graph
did change (the adapter added ops), and not `FAIL`, because the recipe's
target did not move: the saved site key matched exactly. The per-selector diff
carries the receipt -- the label was renumbered by the edit, the site key was
not:

```python
drift_rows = [
    row
    for row in compat.diff.selector_resolution_diffs.values()
    if row.get("label_drift_only")
]
for row in drift_rows:
    print("labels renumbered:", row["saved_labels"], "->", row["resolved_labels"])
    print("site keys unchanged:", row["resolved_site_keys"])
```

```text
labels renumbered: ['linear_1_99'] -> ['linear_2_101']
site keys unchanged: ['s1|lm_head|linear||1']
```

(Your exact label numbers depend on the `transformers` version; the site key
does not.) `label_drift_only` rows are disclosure, not incompatibility: the
checker confirmed every saved structural target on the new capture. Confirm
the drift and reuse the recipe on the edited capture.

Had your selector been broader -- say a type-wide `site(op_type="linear")` --
the diff would instead show `+ new` lines naming the adapter's own linear,
because the recipe now matches a site it never saw at save time. That is the
same honesty in the other direction: confirm before you reuse.

## Step 3: when the target itself moved

Now the failure case. Add a second rule that targets *inside* the block that
got wrapped -- the MLP down-projection of block 1 -- and save the updated
recipe.

```python
inner_rule = tl.when(
    tl.intervention.site(module_path="transformer.h.1.mlp.c_proj", op_type="addmm"),
    tl.zero_ablate(),
)
original_log.attach_hooks(inner_rule, confirm_mutation=True)
tl.io.save_intervention(original_log, "recipe.tlspec", overwrite=True)
spec = tl.io.load_intervention_spec("recipe.tlspec")

refusal = None
try:
    tl.validation.check_spec_compat(spec, edited_log)
except tl.intervention.GraphShapeMismatchError as exc:
    refusal = exc
print("refused:", refusal)
```

```text
refused: Saved spec's graph_shape_hash doesn't match target log; refusing to
apply at executable level.
```

On the edited model no module lives at `transformer.h.1.mlp.c_proj` anymore
(it moved to `transformer.h.1.inner.mlp.c_proj`), so the saved selector cannot
resolve, and the checker refuses to bless an executable recipe whose target
vanished from a changed graph. This is a typed refusal
(`GraphShapeMismatchError`), not a silent partial application.

To see exactly what the edit moved, compare the two captures' site keys --
they are ordinary data on every op:

```python
def all_site_keys(log):
    """Collect every op-level structural site key in a capture."""
    keys = set()
    for label in log.layer_labels:
        for op in log[label].ops:
            if isinstance(op.site_key, str):
                keys.add(op.site_key)
    return keys


moved_out = sorted(all_site_keys(original_log) - all_site_keys(edited_log))
moved_in = sorted(all_site_keys(edited_log) - all_site_keys(original_log))
print(f"{len(moved_out)} old addresses gone, {len(moved_in)} new addresses")
for key in moved_out[:3]:
    print("  -", key)
for key in moved_in[:3]:
    print("  +", key)
```

```text
39 old addresses gone, 41 new addresses
  - s1|transformer/transformer.h.1/transformer.h.1.attn/transformer.h.1.attn.c_attn|addmm||1
  - s1|transformer/transformer.h.1/transformer.h.1.attn/transformer.h.1.attn.c_attn|view||1
  - s1|transformer/transformer.h.1/transformer.h.1.attn/transformer.h.1.attn.c_attn|view||2
  + s1|transformer/transformer.h.1/transformer.h.1.adapter|linear||1
  + s1|transformer/transformer.h.1/transformer.h.1.inner/transformer.h.1.inner.attn/transformer.h.1.inner.attn.c_attn|addmm||1
  + s1|transformer/transformer.h.1/transformer.h.1.inner/transformer.h.1.inner.attn/transformer.h.1.inner.attn.c_attn|view||1
```

Every vanished address and every new address sits inside `transformer.h.1` --
the diff proves the edit was local to the wrapped block. Ops outside the
block, including everything downstream of it, kept their site keys.

Re-authoring is one line: discover the target's new address on the edited
capture, write the rule against it, and check again. A recipe checked against
the same capture it was saved from comes back `EXACT`.

```python
print(
    edited_log.find_sites(
        tl.intervention.site(
            module_path="transformer.h.1.inner.mlp.c_proj", op_type="addmm"
        )
    )
)
recipe_v2 = tl.when(
    tl.intervention.site(
        module_path="transformer.h.1.inner.mlp.c_proj", op_type="addmm"
    ),
    tl.zero_ablate(),
)
edited_log.attach_hooks(recipe_v2, confirm_mutation=True)
tl.io.save_intervention(edited_log, "recipe_v2.tlspec")
spec_v2 = tl.io.load_intervention_spec("recipe_v2.tlspec")
compat_v2 = tl.validation.check_spec_compat(spec_v2, edited_log)
print(compat_v2.outcome)
```

```text
SiteTable(1 site: addmm_8_93)
EXACT
```

## Notes and boundaries

- **Reading a refused diff.** On an executable-level spec, an unresolvable
  target on a changed graph raises the typed refusal above rather than
  returning a verdict. To *preview* the full matched/new/missing diff for a
  recipe you already suspect has moved targets, save a copy of the spec at
  `level="audit"` -- audit-level compatibility checks report `FAIL` with the
  complete `TargetManifestDiff` instead of raising.
- **Applying a confirmed recipe.** After a `COMPATIBLE_WITH_CONFIRMATION`
  verdict, applying the loaded spec to new captures is the ordinary
  intervention workflow; see [intervention_api.md](intervention_api.md) and
  the save-level table in
  [intervention_explainers.md](intervention_explainers.md).
- **Spelling stability.** `op.site_key`, `tl.intervention.site(...)`, and the
  compatibility checker's field spellings are DOCUMENTED-UNSTABLE pending
  naming-session ratification; the workflow itself is the supported path.
- **This page is executable.** The realism test
  `tests/test_skeleton_recipe_docs.py` runs every code block above, in order,
  against the real `transformers` GPT-2 classes on every test run, and
  asserts the verdicts and diffs shown here -- if this page drifts from the
  shipped behavior, that test fails.
