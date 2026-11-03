"""Control solution: writes nothing. Every grader must fail it."""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--outputs", required=True)
a = p.parse_args()
os.makedirs(a.outputs, exist_ok=True)
