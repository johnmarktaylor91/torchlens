"""Reference solution for the harness stub: a forward hook on ``fc2``."""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True)
parser.add_argument("--outputs", required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("model", os.path.join(args.inputs, "model.py"))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
seed = json.load(open(os.path.join(args.inputs, "config.json")))["seed"]
model = model_module.build_model()
captured = []
model.fc2.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
with torch.no_grad():
    model(model_module.make_inputs(seed))
os.makedirs(args.outputs, exist_ok=True)
np.savez(os.path.join(args.outputs, "outputs.npz"), out=torch.cat(captured).numpy())
