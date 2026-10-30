"""Seeded wrong solution: saves ``fc1`` instead of ``fc2``."""

import argparse

import torch
from model import build_model, make_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs.pt")
args = parser.parse_args()
model = build_model()
with torch.no_grad():
    torch.save(model.fc1(make_inputs(args.seed)), args.out)
