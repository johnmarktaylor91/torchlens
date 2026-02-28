#!/usr/bin/env python3
"""Check FLOPs coverage against the full PyTorch op list.

Usage:
    python scripts/check_flops_coverage.py

Reports which ops from ORIG_TORCH_FUNCS are covered by the FLOPs module
and which are uncovered, grouped by category.
"""

from torchlens.constants import ORIG_TORCH_FUNCS
from torchlens.flops import ELEMENTWISE_FLOPS, SPECIALTY_HANDLERS, ZERO_FLOPS_OPS

all_names = {name for _, name in ORIG_TORCH_FUNCS}
covered = ZERO_FLOPS_OPS | set(ELEMENTWISE_FLOPS) | set(SPECIALTY_HANDLERS)

# Only count ops that are actually in the PyTorch op list
covered_in_pytorch = covered & all_names
uncovered = all_names - covered

# Categorize uncovered ops
private_ops = sorted(op for op in uncovered if op.startswith("_"))
dunder_ops = sorted(op for op in uncovered if op.startswith("__"))
private_ops = sorted(op for op in uncovered if op.startswith("_") and not op.startswith("__"))
public_ops = sorted(op for op in uncovered if not op.startswith("_"))

print(f"Total PyTorch ops:  {len(all_names)}")
print(
    f"Covered by FLOPs:   {len(covered_in_pytorch)} ({len(covered_in_pytorch) / len(all_names):.1%})"
)
print(f"Uncovered:          {len(uncovered)} ({len(uncovered) / len(all_names):.1%})")
print(f"Extra (not in PT):  {len(covered - all_names)}")
print()

if public_ops:
    print(f"Uncovered public ops ({len(public_ops)}):")
    for op in public_ops:
        print(f"  {op}")
    print()

if dunder_ops:
    print(f"Uncovered dunder ops ({len(dunder_ops)}):")
    for op in dunder_ops:
        print(f"  {op}")
    print()

if private_ops:
    print(f"Uncovered private ops ({len(private_ops)}):")
    for op in private_ops:
        print(f"  {op}")
