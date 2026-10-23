<<<<<<< HEAD
# Building Age Classification

## Project overview

CHECK OUT track.md FOR A DETAILED REPORT ON HOW THE MODEL EVOLVED OVER TIME

This project classifies building-age groups from long-term multispectral satellite observations. It uses annual pixel-level Landsat data from Madrid and Amsterdam to:

- assign building-age classes from estimated construction years;
- select valid spectral observations and remove known atmospheric outliers;
- fill time-series gaps and calculate spectral indices;
- construct per-pixel temporal features;
- train and evaluate Random Forest and prototype-based models;
- measure transfer from labelled Madrid data to Amsterdam.

The original analysis is documented in four notebooks:

1. `1-Introduction.ipynb` — project context and motivation
2. `2-Reading_Data.ipynb` — data loading and exploratory analysis
3. `3-Preprocessing.ipynb` — preprocessing and feature engineering
4. `4-Modelling.ipynb` — modelling and evaluation

The notebooks remain available as explanatory references. The executable workflows are now represented by scripts and reusable modules.

## Project structure

## Project structure

```text
.
├── data/
│   ├── madrid_train.parquet
│   ├── amsterdam_data.parquet
│   └── preprocessed/
│       └── preprocessed_data.pkl
├── artifacts/                         # Generated notebook and evaluation outputs
├── memory_profiles/                   # Generated Memray reports
├── 1-Introduction.ipynb
├── 2-Reading_Data.ipynb
├── 3-Preprocessing.ipynb
├── 4-Modelling.ipynb
├── load_data.py
├── feature_engineering.py
├── model_training.py
├── preprocessing_io.py
├── utils.py
├── visualization.py
├── scripts/
│   ├── run_notebook2.py
│   ├── run_notebook3.py
│   ├── run_notebook4.py
│   ├── run_notebooks.py
│   ├── run_few_shot_evaluation.py
│   └── profile_memory.py
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_load_data.py
│   ├── test_model_training.py
│   ├── test_notebook3_contract.py
│   └── test_preprocessing_io.py
├── METRIC_LEARNING_APPROACH.md
├── pyproject.toml
├── poetry.lock
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
└── track.md
```

`artifacts/`, `memory_profiles/`, and generated files under
`data/preprocessed/` are local outputs and are ignored by Git. The
`run_notebooks.py`, `run_few_shot_evaluation.py`, and `profile_memory.py`
helpers write to these project-local directories rather than to the caller's
working directory or a shared VM folder.

## Requirements

- Python 3.12 or newer
- pandas
- NumPy
- Matplotlib
- scikit-learn
- pytest
- a Parquet reader supported by pandas, such as `pyarrow`
- optional: Jupyter or JupyterLab for running the notebooks
- optional: pre-commit for Git hook automation

## Installation

### Using Poetry

From the project root:

```bash
poetry install
poetry shell
```

If the environment is externally managed, use the Python environment supplied by the VM administrator instead of installing packages globally.

### Using an existing virtual environment

Activate the environment first, then install the project dependencies using the environment's approved package-management process. Verify the interpreter:

```bash
python3 -c "import sys; print(sys.executable)"
```

The interpreter must be the one containing the project dependencies.

## Running the executable workflows

Run commands from the project root so that relative paths such as `data/` resolve correctly.

### Notebook 2 workflow

```bash
python3 -m scripts.run_notebook2
```

This loads both cities, creates flattened views, performs exploratory temporal and pixel-statistics analysis, runs quality checks, and produces the notebook's intentional plots.

To use the opt-in PyArrow-backed loader:

```bash
python3 -m scripts.run_notebook2 --arrow-backend
```

### Notebook 3 workflow

```bash
python3 -m scripts.run_notebook3
```

This performs preprocessing and feature engineering, then writes:

```text
data/preprocessed/preprocessed_data.pkl
```

To run the opt-in PyArrow-backed loading experiment without overwriting the
standard preprocessing output:

```bash
python3 -m scripts.run_notebook3 \
  --arrow-backend \
  --output data/preprocessed/preprocessed_data_arrow.pkl
```

This uses pandas' PyArrow dtype backend and stores the repeated `city` metadata
as a categorical column. The default workflow remains unchanged. The Arrow
output can be compared with `data/preprocessed/preprocessed_data.pkl` before
adopting the option for regular runs.

### Notebook 4 workflow

Run Notebook 3 first, then:

```bash
python3 -m scripts.run_notebook4
```

This loads the preprocessed arrays, runs cross-validation, trains the final model, evaluates zero-shot transfer, and runs few-shot prototype experiments.

To evaluate the Arrow-backed Notebook 3 output without replacing the baseline:

```bash
python3 -m scripts.run_notebook4 \
  --input data/preprocessed/preprocessed_data_arrow.pkl
```

The feature-engineering workflow also uses targeted NumPy pre-allocation when
selecting the first valid observation for each spectral band. It fills one
result array in place rather than creating a new array for every observation
slot. This does not change the model features or predictions. Sparse matrices
are not used because the current continuous feature matrix is not sparse.

## Running tests

Run the automated tests from the project root:

```bash
python3 -m pytest -q
```

The tests use small in-memory fixtures where possible, so they validate function contracts without rerunning the full multi-million-row pipelines.

## Pre-commit hooks

The optional `.pre-commit-config.yaml` configures:

- Black formatting;
- debug-statement checks;
- end-of-file and trailing-whitespace checks;
- YAML validation;
- pytest;
- clearing saved Jupyter notebook outputs.

Install the hooks in the active environment:

```bash
python3 -m pre_commit install
```

Run them manually:

```bash
python3 -m pre_commit run --all-files
```

The notebook hook intentionally rewrites staged notebooks by removing saved outputs. Review and restage those changes before retrying a commit.

## Automated memory profiling

Install Memray in the active VM environment if it is not already available:

```bash
python3 -m pip install memray
```

Use the profiling helper to run a workflow under Memray and generate a binary
recording, terminal reports, and an interactive flamegraph in one command:

```bash
python3 -m scripts.profile_memory notebook3 \
  --arrow-backend \
  --preprocessed-output data/preprocessed/preprocessed_data_arrow.pkl \
  --label notebook3_arrow
```

Reports are written to `memory_profiles/` by default:

```text
memory_profiles/notebook3_arrow.bin
memory_profiles/notebook3_arrow_stats.txt
memory_profiles/notebook3_arrow_tree.txt
memory_profiles/notebook3_arrow.html
```

Other workflows can be profiled with:

```bash
python3 -m scripts.profile_memory notebook2 --arrow-backend
python3 -m scripts.profile_memory notebook4 \
  --input data/preprocessed/preprocessed_data_arrow.pkl \
  --label notebook4_arrow
```

Use `--output-dir` to store reports elsewhere. Open the generated `.html`
flamegraph through the JupyterLab file browser.

### Profiling after a new feature

From the project root, run the tests first and then use a new label so the
new reports do not overwrite earlier measurements:

```bash
cd ~/Building_Age_Classification
python3 -m pytest -q
python3 -m scripts.profile_memory notebook3 \
  --arrow-backend \
  --preprocessed-output data/preprocessed/preprocessed_data_arrow.pkl \
  --label notebook3_after_new_feature
```

If the change affects Notebook 3 preprocessing, use the newly generated
preprocessed file when profiling Notebook 4:

```bash
python3 -m scripts.profile_memory notebook4 \
  --input data/preprocessed/preprocessed_data_arrow.pkl \
  --label notebook4_after_new_feature
```

For a change limited to Notebook 2, profile it directly:

```bash
python3 -m scripts.profile_memory notebook2 \
  --arrow-backend \
  --label notebook2_after_new_feature
```

Each run creates a binary recording, `_stats.txt`, `_tree.txt`, and an HTML
flamegraph under `memory_profiles/`. Compare the new reports with the
previously labelled reports to identify changes in peak memory, allocation
stacks, runtime, and line-level hotspots.

## Python packaging

`pyproject.toml` is the project metadata and dependency declaration file. It defines the package name, version, supported Python version, runtime dependencies, and Poetry build backend. `poetry.lock` records the resolved dependency versions for reproducible Poetry installations.

The current project is organized as importable modules and executable scripts. The notebooks are documentation and analysis references rather than the only location of the implementation.
=======
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
>>>>>>> 1f08cd24aa45034ad2c59c3b59ca614389b6600f
