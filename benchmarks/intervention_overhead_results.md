# Intervention Overhead Results

Model: `TinyMLP` with dimensions 8 -> 16 -> 16 -> 8, batch size 32.

Budget reference: PLAN.md v5.2 section 13.12 is qualitative. It requires inactive overhead to stay behind existing cheap gates, replay to scale with cone size, rerun to use pre-normalized hook dispatch, fork to stay shallow, and Bundle supergraph construction to remain lazy.

| Benchmark | Mean seconds | Ratio | Notes |
|---|---:|---:|---|
| Baseline forward | 0.000057 | 1.00x | PyTorch only |
| log_forward_pass(intervention_ready=False) | 0.032300 | 564.60x | TorchLens capture |
| log_forward_pass(intervention_ready=True) | 0.007983 | 0.25x | Incremental vs non-ready capture |
| replay(hook=zero relu) | 0.001117 | 19.52x | Saved-DAG propagation |
| rerun(model, x) | 0.012192 | 213.12x | Full forward through hooks |
| Bundle.node() lazy supergraph build | 0.000300 | n/a | First node access builds supergraph |

Budget misses: none automatically enforced; review ratios against the qualitative v5.2 budget.
