# TorchLens Intervention Worked Examples

Each script is runnable with `python examples/intervention/<name>.py` and uses a
small local PyTorch model unless noted otherwise.

1. `01_first_five_minutes.py` - minimal capture, discovery, fork, zero-ablation, and replay.
2. `02_exact_site_after_discovery.py` - discover labels with `find_sites`, then target an exact site.
3. `03_activation_patching_paired_prompt.py` - clean-versus-corrupted activation patching.
4. `04_sticky_hooks_multiple_engines.py` - reuse sticky hooks with replay and rerun.
5. `05_set_vs_attach_hooks.py` - static `set` replacements versus sticky hook recipes.
6. `06_chunked_batching.py` - append compatible chunks for memory-constrained evaluation.
7. `07_bundle_comparison.py` - compare multiple `ModelLog` objects in a `Bundle`.
8. `08_live_post_hooks_during_capture.py` - run hooks during the original capture.
9. `09_submodule_discover_first.py` - use `tl.in_module` for submodule-scoped discovery.
10. `10_post_hoc_replay_generation_trace.py` - replay over a captured generation-style trace.
11. `11_sae_attachment.py` - attach an SAE-style `nn.Module` with `splice_module`.
12. `12_linear_probe_attachment.py` - collect a linear-probe readout from a hook.
13. `13_paired_prompt_3plus.py` - compare three or more prompt variants in a `Bundle`.
14. `14_per_position_steering.py` - steer with a per-position direction tensor.
15. `15_publishing_for_reproducibility.py` - save a portable `.tlspec/` and load it back.
16. `16_pearl_style_tl_do.py` - use the top-level `tl.do` shortcut.
