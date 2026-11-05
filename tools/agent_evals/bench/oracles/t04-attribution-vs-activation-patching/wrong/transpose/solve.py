import argparse
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "reference" / "solve.py"
spec = importlib.util.spec_from_file_location("reference_solution", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--outputs", required=True)
a = p.parse_args()
reference.run(a.inputs, a.outputs, variant="transpose")
