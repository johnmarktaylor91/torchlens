import argparse
import importlib.util
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--outputs", required=True)
a = p.parse_args()
s = importlib.util.spec_from_file_location("reference", Path(__file__).resolve().parents[2] / "reference" / "solve.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.run(a.inputs, a.outputs, variant="off_by_one")
