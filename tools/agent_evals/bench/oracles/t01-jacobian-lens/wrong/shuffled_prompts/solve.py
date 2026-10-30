"""Seeded wrong solution: shuffled_prompts (the reference with one deliberate error)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference"))
import solve as ref  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--outputs", required=True)
a = p.parse_args()
ref.run(a.inputs, a.outputs, variant="shuffled_prompts")
