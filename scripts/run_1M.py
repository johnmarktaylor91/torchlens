"""Standalone script: render a 1M-node graph with ELK stress + topo seeding."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch
from example_models import RandomGraphModel
from torchlens import log_forward_pass

outdir = os.path.join("tests", "test_outputs", "visualizations", "large")
os.makedirs(outdir, exist_ok=True)

# Phase 1: Model construction
t0 = time.time()
model = RandomGraphModel(target_nodes=1000000, seed=42)
x = torch.randn(2, 64)
t1 = time.time()
print(f"Phase 1 — Model construction: {t1 - t0:.1f}s", flush=True)

# Phase 2: log_forward_pass (logging + postprocessing)
ml = log_forward_pass(model, x, layers_to_save=None, detect_loops=False)
t2 = time.time()
print(f"Phase 2 — log_forward_pass:    {t2 - t1:.1f}s ({len(ml)} layers)", flush=True)

# Phase 3: Render
ml.render_graph(
    vis_opt="unrolled",
    vis_nesting_depth=1000,
    vis_outpath=os.path.join(outdir, "elk_1M"),
    save_only=True,
    vis_fileformat="svg",
    vis_node_placement="elk",
)
t3 = time.time()
print(f"Phase 3 — ELK render:          {t3 - t2:.1f}s", flush=True)

total = t3 - t0
print(f"Total: {total:.1f}s ({total / 60:.1f} min)", flush=True)

svg_path = os.path.join(outdir, "elk_1M.svg")
if os.path.exists(svg_path):
    size_mb = os.path.getsize(svg_path) / 1024 / 1024
    print(f"Output: {svg_path} ({size_mb:.0f} MB)")

ml.cleanup()
