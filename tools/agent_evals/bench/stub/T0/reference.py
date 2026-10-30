"""Reference solution for the harness stub: a forward hook on ``fc2``."""

import argparse

import torch
from model import build_model, make_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs.pt")
args = parser.parse_args()
model = build_model()
captured = []
model.fc2.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
with torch.no_grad():
    model(make_inputs(args.seed))
torch.save(torch.cat(captured), args.out)
