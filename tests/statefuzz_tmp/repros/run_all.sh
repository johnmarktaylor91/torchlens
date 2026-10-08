#!/bin/bash
cd "$(dirname "$0")"
for f in f1_deepcopy_alias.py f2_loaded_rerun_drops_spec.py f3_output_node_double_edit.py f4_pass_qualified_module.py b0_family_doors.py f6_rng_reseed.py; do
  echo "===== $f"; "$P" "$f" 2>&1 | grep -v "^\s*$" | tail -25
done
