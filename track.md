# Project update track

## 2026-10-23

### Repository hygiene and execution

- Added a repository-wide `.gitignore` for Python caches, Jupyter checkpoints,
  generated artifacts, preprocessing output, and Memray reports.
- Added `scripts/run_notebooks.py` to execute Notebooks 1-4 in dependency order
  and write executed copies to `artifacts/executed-notebooks/`.
- Added `scripts/profile_memory.py` so Memray binaries, statistics, tree
  reports, and flamegraphs are written under `memory_profiles/` rather than
  the repository root.
- Updated the documentation to use Linux commands with `python3`.

### Controlled Amsterdam few-shot evaluation

- Added `cross_validate_few_shot()` with leakage-safe repeated stratified
  validation:
  - 5 folds x 3 repeats;
  - support selection only from each fold's training partition;
  - untouched validation partitions;
  - budgets of 5, 25, 50, 100, and 200 samples per class.
- Added comparisons for Random Forest, logistic regression, and prototype
  classification.
- Added four reproducible feature configurations:
  - `baseline`: overall features plus availability indicators;
  - `spectral`: baseline plus spectral-index features;
  - `temporal`: overall, early, late, year-over-year, and availability features;
  - `full`: all existing feature groups.
- Added `scripts/run_few_shot_evaluation.py`, which writes the comparison table
  to `artifacts/few_shot_evaluation.csv`.

### Notebook 1 presentation output

- Added a dedicated F1-versus-budget plot with standard-deviation error bars.
- The x-axis uses `log2(samples per class)` as required by Notebook 1.
- The default plot uses the full feature set and Random Forest and is written
  to `artifacts/few_shot_f1_vs_log2_budget.png`.
- The feature set and method can be selected with `--feature-set` and
  `--method`.

### Evaluation findings

- The controlled comparison showed a substantial improvement over the
  baseline at 25 Amsterdam samples per class:
  - baseline Random Forest: approximately 0.515 macro-F1;
  - temporal features: approximately 0.631 macro-F1;
  - full features: approximately 0.637-0.644 macro-F1 depending on the exact
    estimator configuration.
- Additional Random Forest weighting experiments produced only marginal and
  inconsistent gains, so no further model-tuning change was adopted.
- The current results are development evidence rather than a formally
  untouched final score because Amsterdam validation results were inspected
  during feature and model selection.

### Validation

- Test suite increased from 8 to 10 tests.
- Latest test result: `10 passed`.
- Plot generation was smoke-tested successfully and produced both CSV and PNG
  artifacts.
