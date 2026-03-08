"""Render a large random graph with ELK layout.

Usage:
    python scripts/render_large_graph.py NUM_NODES [OPTIONS]

Examples:
    python scripts/render_large_graph.py 250000
    python scripts/render_large_graph.py 1000000 --format png --seed 123
    python scripts/render_large_graph.py 50000 --outdir /tmp/graphs
"""

import argparse
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

    # Phase 1: Model construction
    t0 = time.time()
    model = RandomGraphModel(target_nodes=args.num_nodes, seed=args.seed)
    x = torch.randn(2, 64)
    t1 = time.time()
    print(f"Phase 1 — Model construction: {t1 - t0:.1f}s", flush=True)

    # Phase 2: log_forward_pass (logging + postprocessing)
    ml = log_forward_pass(model, x, layers_to_save=None, detect_loops=False, verbose=True)
    t2 = time.time()
    print(f"Phase 2 — log_forward_pass:    {t2 - t1:.1f}s ({len(ml)} layers)", flush=True)

    # Phase 3: Render
    ml.render_graph(
        vis_opt="unrolled",
        vis_nesting_depth=1000,
        vis_outpath=os.path.join(args.outdir, label),
        save_only=True,
        vis_fileformat=args.format,
        vis_node_placement="elk",
    )
    t3 = time.time()
    print(f"Phase 3 — ELK render:          {t3 - t2:.1f}s", flush=True)

    total = t3 - t0
    print(f"Total: {total:.1f}s ({total / 60:.1f} min)", flush=True)

    out_path = os.path.join(args.outdir, f"{label}.{args.format}")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Output: {out_path} ({size_mb:.1f} MB)")

    ml.cleanup()


if __name__ == "__main__":
    main()
