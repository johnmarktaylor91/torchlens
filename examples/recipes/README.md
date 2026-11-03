# TorchLens Recipes

These notebooks are self-contained intervention recipes. They use small local
models only, so they can run in CI or with `papermill` without network
downloads.

| Recipe | Difficulty | Description |
| --- | --- | --- |
| [`causal_trace_recipe.ipynb`](causal_trace_recipe.ipynb) | Intermediate | Clean-versus-corrupt activation patching, per-site sweeps with `trace.find_sites`, output-divergence scoring, and `tl.viz.causal_trace_heatmap`. |
| [`contrastive_direction_recipe.ipynb`](contrastive_direction_recipe.ipynb) | Beginner | RepEng-style positive-minus-negative activation direction with replay-time steering. |
| [`measure_faithfulness_recipe.ipynb`](measure_faithfulness_recipe.ipynb) | Intermediate | ROAD-style ranked removal, output-drop, locality checks, and `tl.validate(scope="intervention")`. |
| [`trainable_intervention_recipe.ipynb`](trainable_intervention_recipe.ipynb) | Advanced | Gradient-based learning of a trainable activation intervention parameter. |
| [`generation_step_iterator_recipe.ipynb`](generation_step_iterator_recipe.ipynb) | Intermediate | Per-step hook planning for generation loops that mirror `model.generate()`. |
| [`train_tuned_lens_recipe.ipynb`](train_tuned_lens_recipe.ipynb) | Intermediate | Train a tuned lens from a disk-mode `tl.extract_dataset` artifact and apply it through `logit_lens(lens=...)`, which checks the recorded model identity. |
| [`das_causal_basis_recipe.ipynb`](das_causal_basis_recipe.ipynb) | Advanced | Distributed alignment search (Geiger et al., 2024): learn a rotation through a bound hook, then test the causal basis with `tl.subspace` and `tl.patch_from` against a random direction and whole-layer patching. |
| [`function_vectors_recipe.ipynb`](function_vectors_recipe.ipynb) | Advanced | Function vectors (Todd et al., 2024) on GPT-2 small: score every attention head's causal indirect effect with bound hooks, sum the top heads into a task vector, and add it zero-shot with `tl.steer` to turn country prompts into capitals. |
| [`clip_head_attention_recipe.ipynb`](clip_head_attention_recipe.ipynb) | Intermediate | Per-head attention maps of CLIP ViT-B/32's vision tower from the `pattern` facet, rebuilt from the fused attention kernel, and which heads send the class token's attention to the object. Downloads the public `openai/clip-vit-base-patch32` checkpoint (about 600 MB). |
| [`clip_text_decomposition_recipe.ipynb`](clip_text_decomposition_recipe.ipynb) | Advanced | Gandelsman et al. (ICLR 2024): split CLIP's image embedding into per-head `result` terms, mean-ablate layers, and name colour and shape heads with TextSpan. Downloads the public `openai/clip-vit-base-patch32` checkpoint (about 600 MB). |
