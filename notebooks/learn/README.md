# Learn TorchLens

Twelve notebooks, in order. Start with `00_quickstart.ipynb` (five minutes). Each later notebook
adds one idea and names the earlier ones it uses. After setup, each runs on a laptop CPU in about
a minute on tiny local models from `learn_models.py` beside the notebooks; no model weights are
downloaded.

## How to run

From the repository root, in a fresh virtual environment (JupyterLab opens in your browser; pick
the kernel of this venv if it asks):

```bash
python3 -m venv .venv-learn && source .venv-learn/bin/activate
python -m pip install -e ".[viz]" -r notebooks/learn/requirements-learn.txt
dot -V || brew install graphviz        # drawings need the Graphviz binary; apt install graphviz on Debian/Ubuntu
jupyter lab notebooks/learn/00_quickstart.ipynb
```

Each notebook imports `learn_models.py` from its own folder, so open the notebooks from
`notebooks/learn/` (JupyterLab does this when launched as above). Rendered pictures go to a
temporary directory, nothing is written beside the notebooks.

Levels: 0 one call; 1 read like a dict; 2 one more keyword on the same call; 3 compose pieces;
4 objects for scale and record; 5 extend.

| # | Notebook | Level | Starting point |
|---|---|---|---|
| 00 | `00_quickstart.ipynb` | 0 to 2 | You have a model and want to see inside it |
| 01 | `01_capture_and_look.ipynb` | 0 | A model in hand |
| 02 | `02_address_and_read.ipynb` | 1 | You want a number out |
| 03 | `03_see_it.ipynb` | 1 | The summary is too long to read |
| 04 | `04_keep_what_you_need.ipynb` | 2 | Memory or time ran out |
| 05 | `05_change_and_rerun.ipynb` | 2 to 3 | Does this layer matter? |
| 06 | `06_gradients.ipynb` | 2 | What drives this unit? |
| 07 | `07_compare_runs.ipynb` | 3 to 4 | You ran it twice |
| 08 | `08_semantic_shortcuts.ipynb` | 3 to 5 | You keep typing "head 3's query" |
| 09 | `09_scale.ipynb` | 4 | Thousands of batches |
| 10 | `10_beyond_one_forward.ipynb` | 4 to 5 | Your field's experiment |
| 11 | `11_dynamic_graphs.ipynb` | 4 | The graph changes between forwards |

## Run in Colab

Each notebook opens with an "Open in Colab" badge (the notebook on the `main` branch of
`johnmarktaylor91/torchlens`). Its first code cell installs TorchLens from `main` and fetches
`learn_models.py` beside the notebook; run from a checkout, that cell does nothing.

## Stored outputs

The notebooks are committed with their outputs, so GitHub renders them as run. After changing a
notebook, or when the API changes, regenerate the outputs in a fresh environment from the
repository root (the nightly CI job only checks that every cell runs; it does not diff outputs):

```bash
python scripts/run_notebooks.py --write-outputs notebooks/learn/*.ipynb
```

The script executes each notebook top to bottom and writes the outputs back with machine
paths scrubbed. `python scripts/run_notebooks.py notebooks/learn/*.ipynb` runs them without
writing, which is what the nightly job does.

The three-line recipe cards live in `examples/quick/`,
the longer workflows in `examples/workflows/`, and the API spec in `docs/reference/glossary.md`.
