# Building Age Classification

This project classifies building-age groups from long-term Landsat spectral
observations for Madrid and Amsterdam. The analysis is organized as four
notebooks:

1. `1-Introduction.ipynb` explains the problem and proposed approaches.
2. `2-Reading_Data.ipynb` loads and explores both city datasets.
3. `3-Preprocessing.ipynb` cleans the observations, engineers features, and
   writes `data/preprocessed/preprocessed_data.pkl`.
4. `4-Modelling.ipynb` trains and evaluates the Random Forest and few-shot
   prototype models using Notebook 3's output.

Run commands from the repository root. The bundled data files must be valid
Parquet files supplied by the project environment.

## Running the notebooks

The notebooks must be run in order because Notebook 4 consumes the artifact
created by Notebook 3. Notebook 1 is informational but is included in the
execution check:

```bash
python3 -m scripts.run_notebooks
```

Executed copies are written to `artifacts/executed-notebooks/`, leaving the
source notebooks unchanged. To choose another output directory:

```bash
python3 -m scripts.run_notebooks --output-dir artifacts/notebook-runs
```

For faster script-based runs of the same implemented workflows:

```bash
python3 -m scripts.run_notebook2
python3 -m scripts.run_notebook3
python3 -m scripts.run_notebook4
```

To run the controlled Amsterdam few-shot comparison (5 folds x 3 repeats):

```bash
python3 -m scripts.run_few_shot_evaluation
```

This compares baseline, spectral, temporal, and full feature sets across
5, 25, 50, 100, and 200 labelled Amsterdam pixels per class using the same
folds and support-selection seeds for Random Forest, logistic regression, and
prototype classification. Results are written to
`artifacts/few_shot_evaluation.csv`. The default final plot uses the full
feature set and Random Forest and is written to
`artifacts/few_shot_f1_vs_log2_budget.png`. Choose another comparison for the
plot with `--feature-set` and `--method`.

## Memory profiling

Use the repository helper so Memray output does not accumulate in the project
root:

```bash
python3 -m scripts.profile_memory notebooks
python3 -m scripts.profile_memory notebook3 --label notebook3
```

Reports are written to `memory_profiles/` and ignored by Git. You can choose a
different directory with `--output-dir`.

## Tests

```bash
python3 -m pytest -q
```
