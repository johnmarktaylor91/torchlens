"""Render a large random graph with ELK layout.

Usage:
    python scripts/render_large_graph.py NUM_NODES [OPTIONS]

Examples:
    python scripts/render_large_graph.py 250000
    python scripts/render_large_graph.py 1000000 --format png --seed 123
    python scripts/render_large_graph.py 50000 --outdir /tmp/graphs
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch
from example_models import RandomGraphModel
from torchlens import log_forward_pass


def main():
    parser = argparse.ArgumentParser(description="Render a large random graph with ELK layout.")
    parser.add_argument("num_nodes", type=int, help="Target number of nodes in the graph")
    parser.add_argument("--format", default="svg", help="Output format (default: svg)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--outdir",
        default=os.path.join("tests", "test_outputs", "visualizations", "large"),
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    label = f"elk_{args.num_nodes // 1000}k"

    t0 = time.time()
    print(f"Building RandomGraphModel with target_nodes={args.num_nodes}...", flush=True)
    model = RandomGraphModel(target_nodes=args.num_nodes, seed=args.seed)
    x = torch.randn(2, 64)
    print(f"Model constructed ({time.time() - t0:.1f}s)", flush=True)

    # Log forward pass WITHOUT rendering — collect metadata only.
    ml = log_forward_pass(model, x, layers_to_save=None, verbose=True)
    print(f"Forward pass logged ({time.time() - t0:.1f}s)", flush=True)

    # Free model parameters and autograd graphs before the memory-heavy ELK render.
    del model, x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Render from ModelLog metadata (model tensors no longer in memory).
    ml.render_graph(
        vis_mode="rolled",
        vis_outpath=os.path.join(args.outdir, label),
        vis_save_only=True,
        vis_fileformat=args.format,
        vis_node_placement="elk",
    )

    total = time.time() - t0
    print(f"Total: {total:.1f}s ({total / 60:.1f} min)", flush=True)

    out_path = os.path.join(args.outdir, f"{label}.{args.format}")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Output: {out_path} ({size_mb:.1f} MB)")

    ml.cleanup()


if __name__ == "__main__":
    main()
