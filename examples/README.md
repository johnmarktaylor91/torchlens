# TorchLens examples

Runnable notebooks and scripts, from one-screen tasks to full experiments. New to
TorchLens? Start with the learning path in [`notebooks/learn/`](../notebooks/learn/README.md),
then come here.

## Folders

| Folder | What it holds | How long |
| --- | --- | --- |
| [`quick/`](quick/README.md) | One-screen recipes: pluck an activation, find the first NaN, render the graph, extract layers or a whole dataset, ablate one layer, save and load a trace, build an RDM. | Under 30 seconds each, offline on a CPU. |
| [`workflows/`](workflows/README.md) | Full analyses: ablation grids, causal tracing, paired-prompt patching, steering, custom hooks, fork and replay, neuroscience extraction, TransformerLens migration. | Longer reads; each runs offline on a CPU. |
| [`methods/`](methods/README.md) | Recent interpretability methods run end to end: tuned lens, distributed alignment search, function vectors, CLIP head readouts, CKA and RSA, attribution graphs with transcoders, and more. | Seconds to about ten minutes on a laptop CPU; some download a public checkpoint on first run. |
| [`replications/`](replications/README.md) | Scripts that re-test published findings on GPT-2 small (induction heads, IOI name movers, causal tracing) and print a verdict against a rule fixed before the first run. | Downloads GPT-2 small (about 550 MB) on first use. |
| [`intervention/`](intervention/README.md) | Numbered Python scripts for the intervention API, from a first ablation to publishing a reproducible artifact. | Small local models unless a script says otherwise. |
| [`demos/`](demos/) | The bundle-diff demo: a clean-versus-ablated ResNet-18 comparison notebook and its rendered SVG. | Random weights, no download. |
| [`provider_package/`](provider_package/README.md) | A minimal installable provider distribution to copy for your own export target. | Install with `pip install -e examples/provider_package`. |

## Single scripts

Each runs with `python examples/<script>.py`.

| Script | What it shows |
| --- | --- |
| [`aten_ops_recording.py`](aten_ops_recording.py) | Recording the ATen operations behind each captured op, and the refusal when recording was off. |
| [`auto_collapse_sample_renders.py`](auto_collapse_sample_renders.py) | A small set of smart-collapse graph renders written to a temporary folder for visual review. |
| [`capture_unification_selective.py`](capture_unification_selective.py) | Minimal examples of the unified capture path with selective saving. |
| [`debug_utilities.py`](debug_utilities.py) | The trace debugging helpers. |
| [`jacobian_lens_demo.py`](jacobian_lens_demo.py) | Fitting a Jacobian lens for GPT-2 small from public TorchLens calls and reading it out (downloads GPT-2 small). |
| [`quick/paris_vs_rome.py`](quick/paris_vs_rome.py) | Patching each GPT-2 small attention head from a France prompt into an Italy prompt to find the heads that make the model answer " Paris" (downloads GPT-2 small). |
| [`runnable_tlspec.py`](runnable_tlspec.py) | Saving, loading and executing a sparse runnable `.tlspec` artifact. |
| [`semantic_io_legibility_demo.py`](semantic_io_legibility_demo.py) | Semantic input and output labels on a tiny deterministic model, with no downloads. |
| [`treescope_cards_gallery.py`](treescope_cards_gallery.py) | Rendering the treescope card fragments and offline reports for the reachable fixtures into one folder with an index page. |
| [`vmap_masked_toy_transformer.py`](vmap_masked_toy_transformer.py) | Tracing a toy transformer block that builds its attention mask with `torch.vmap`. |
