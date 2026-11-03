# TorchLens Recipes

These notebooks are self-contained intervention recipes. Most build small local
models and run offline, so they work in CI or with `papermill` without network
access. A recipe that needs a public checkpoint says so in its description
("Downloads ...", with the size): it fetches a pinned revision from the Hugging
Face Hub on first run (GPT-2 small, ViT-tiny, CLIP ViT-B/32, a transcoder set) and
still finishes in minutes on a laptop CPU.

| Recipe | Difficulty | Description |
| --- | --- | --- |
| [`causal_trace_recipe.ipynb`](causal_trace_recipe.ipynb) | Intermediate | Clean-versus-corrupt activation patching, per-site sweeps with `trace.find_sites`, output-divergence scoring, and `tl.viz.causal_trace_heatmap`. |
| [`contrastive_direction_recipe.ipynb`](contrastive_direction_recipe.ipynb) | Beginner | RepEng-style positive-minus-negative activation direction with replay-time steering. |
| [`measure_faithfulness_recipe.ipynb`](measure_faithfulness_recipe.ipynb) | Intermediate | ROAD-style ranked removal, output-drop, locality checks, and `tl.validate(scope="intervention")`. |
| [`trainable_intervention_recipe.ipynb`](trainable_intervention_recipe.ipynb) | Advanced | Gradient-based learning of a trainable activation intervention parameter. |
| [`generation_step_iterator_recipe.ipynb`](generation_step_iterator_recipe.ipynb) | Intermediate | Per-step hook planning for generation loops that mirror `model.generate()`. |
| [`train_tuned_lens_recipe.ipynb`](train_tuned_lens_recipe.ipynb) | Intermediate | Train a tuned lens from a disk-mode `tl.extract_dataset` artifact and apply it through `logit_lens(lens=...)`, which checks the recorded model identity. |
| [`eap_ig_circuit_recipe.ipynb`](eap_ig_circuit_recipe.ipynb) | Intermediate | Edge attribution patching with integrated gradients (Syed, Rager and Conmy 2023, arXiv:2310.10348; Hanna, Pezzelle and Belinkov 2024, arXiv:2403.17806) on GPT-2 small's IOI task: `torchlens.mechinterp.edge_scores` ranks every component-graph edge from one gradient pass, the name movers of Wang et al. 2022 surface at the top, and `patch_edges` checks the circuit's faithfulness and `circuit_performance_ratio` in one call per kept fraction. Downloads GPT-2 small (about 550 MB); about ten minutes on a laptop CPU, most of it in the faithfulness replays. |
| [`vision_logit_lens_recipe.ipynb`](vision_logit_lens_recipe.ipynb) | Beginner | The logit lens for vision models (Vilas, Schaumloeffel and Roig 2023, arXiv:2310.18969; Gandelsman, Efros and Steinhardt 2024, arXiv:2310.05916): a ViT-tiny classifier's per-block class votes through its own head, and CLIP's open-vocabulary readout of every vision block against captured label prompts, both validated against the model's logits. Downloads ViT-tiny and CLIP ViT-B/32 (about 620 MB); under a minute. |
| [`leace_concept_erasure_recipe.ipynb`](leace_concept_erasure_recipe.ipynb) | Intermediate | LEACE concept scrubbing (Belrose et al. 2023, arXiv:2306.03819): fit the closed-form eraser of subject number at every GPT-2 small block in turn, bind the twelve `tl.when` rules to the model as one live executor, and watch a linear probe fall to chance and subject-verb agreement collapse, while one block alone or a random direction per block changes nothing. Downloads GPT-2 small (about 550 MB); under a minute. |

The Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in Language Models") is a
documentation recipe rather than a notebook: [`docs/recipes/jacobian_lens.md`](../../docs/recipes/jacobian_lens.md) fits
one from public TorchLens primitives, and [`examples/jacobian_lens_demo.py`](../jacobian_lens_demo.py) runs it on GPT-2 small.
The other recipe pages (SAEs, TCAV, LRP, feature visualisation, dissection, ACDC, tuned lens, Grad-CAM and more) are listed
in [`docs/recipes/`](../../docs/recipes/).
