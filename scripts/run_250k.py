"""Standalone script: render a 250k-node graph with ELK."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch
from example_models import RandomGraphModel
from torchlens import show_model_graph

outdir = os.path.join("tests", "test_outputs", "visualizations", "large")
os.makedirs(outdir, exist_ok=True)

model = RandomGraphModel(target_nodes=250000, seed=42)
x = torch.randn(2, 64)

print("Starting 250k ELK render...", flush=True)
t0 = time.time()
show_model_graph(
    model,
    x,
    vis_node_placement="elk",
    vis_fileformat="svg",
    save_only=True,
    vis_outpath=os.path.join(outdir, "elk_250k"),
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
