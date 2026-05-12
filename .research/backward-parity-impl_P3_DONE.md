# Backward-Parity P3 Done

Final implementation SHA: 20730ba.

Time spent: about 1.5 hours.

## Changes

- Added combined forward/backward Graphviz rendering in
  `torchlens/visualization/rendering.py`.
- Added `Trace.draw_combined(...)`, `torchlens.visualization.draw_combined(...)`,
  and top-level `torchlens.draw_combined(...)`.
- Added module-aware backward node clustering, intervening grad_fn placement
  modes (`upstream`, `outside`, `downstream`, `own`), AccumulateGrad module
  attribution via `_grad_fn_param_refs`, and dashed correspondence edges.
- Added `tests/test_combined_visualization.py`.
- Added gallery/sample PDF:
  `tests/test_outputs/visualizations/combined/linear_relu_combined.pdf`.
- Added drop sample:
  `/home/jtaylor/.claude/drops/backward-parity-P3-sample.pdf`.
- Updated `CHANGELOG.md` under Unreleased.

## Commits

- `b369fab feat(viz): render combined forward backward graph`
- `5f069d5 test(viz): cover combined visualization modes`
- `20730ba chore(changelog): record P3 combined visualization`
- `chore(research): record P3 implementation done`

Note: the requested five logical P3 commits were compressed to three commits
because the public API, merged renderer, module clustering, and intervening
helpers were tightly coupled in `rendering.py` and had to stay green together.

## Test Results

- `pytest tests/test_combined_visualization.py -v -x --tb=short`
  - 10 passed, 1 warning.
- `mypy torchlens/`
  - Success: no issues found in 199 source files.
- `pytest tests/ -m smoke -x --tb=short`
  - 194 passed, 1 skipped, 2160 deselected, 118 warnings.
- `ruff check torchlens/visualization/rendering.py torchlens/data_classes/model_log.py torchlens/user_funcs.py torchlens/visualization/__init__.py torchlens/__init__.py tests/test_combined_visualization.py --fix`
  - All checks passed.
- `ruff check . --fix`
  - Blocked by pre-existing syntax error in
    `notebooks/torchlens_in_10_minutes.ipynb`, cell 27:
    `from torchvision import `.

## Assumptions

- `intervening_cluster` follows the dispatch prompt API:
  `upstream`, `outside`, `downstream`, `own`.
- Backward clustering is implemented only for combined visualization;
  `draw_backward()` remains unchanged.
- `vis_mode="rolled"` remains explicitly unsupported for combined rendering.

## Controversial Choices

- Top-level `tl.draw_combined(...)` follows the existing moved-name pattern used
  by `tl.draw_backward(...)`, so it forwards to `torchlens.visualization` and
  emits the standard deprecation warning.
- AccumulateGrad attribution uses the plan-required `co_parent_params`
  ambiguity predicate, even though multi-parameter modules can populate that
  field in existing traces.

## Concerns

- Full-repo ruff is not green until the pre-existing notebook syntax error is
  fixed outside P3 scope.
- Large-model combined graph layout density remains expectedly high.

## Knowledge

- Existing module cluster construction places nodes through subgraphs mostly via
  edges; combined rendering adds an optional `nodes` bucket to the same cluster
  accumulator so backward nodes can be declared inside module clusters.
- For submodule output ops, `output_of_modules[0]` gives the containing module
  address and `output_of_module_calls` supplies the unrolled call key.

Ready for P4: yes, subject to the unrelated notebook ruff blocker.
