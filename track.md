# Project change history

This document records how the building-age classification project evolved,
including the original notebook workflow, refactoring, memory-management work,
evaluation changes, and the safeguards added for running on a shared Linux VM.

## 1. Original notebook project

The project began as a four-notebook analysis:

1. `1-Introduction.ipynb` documented the hackathon task, the four age classes,
   the Madrid-to-Amsterdam transfer goal, and the required deliverables.
2. `2-Reading_Data.ipynb` loaded the two city datasets and explored coverage,
   class distributions, spectral bands, time series, and data quality.
3. `3-Preprocessing.ipynb` selected valid observations, removed haze-affected
   rows, interpolated time-series gaps, computed spectral indices, and created
   per-pixel features.
4. `4-Modelling.ipynb` trained a Random Forest on Madrid, evaluated transfer to
   Amsterdam, and ran prototype-based few-shot trials.

The original model used six Landsat bands:

```text
Blue, Green, Red, NIR, SWIR1, SWIR2
```

The project assigned four city-aware age classes from
`weighted_mean_year`. Madrid and Amsterdam have different historical event
boundaries, while the 1984-2004 and 2004-2024 boundaries are shared.

The first recorded baseline achieved approximately:

```text
Madrid macro-F1: 0.5345 +/- 0.0013
```

At this stage, the primary workflow was notebook-oriented and relative paths
such as `./data` were assumed to be resolved from the project directory.

## 2. Refactoring into reusable Python modules

The next phase moved repeated notebook logic into importable modules:

- `load_data.py` handles Parquet loading and age-class assignment.
- `feature_engineering.py` handles valid-observation selection, flattening,
  gap filling, spectral indices, temporal statistics, and feature columns.
- `model_training.py` contains Random Forest, prototype, cross-validation, and
  few-shot helpers.
- `preprocessing_io.py` serializes and loads preprocessed arrays.
- `utils.py` stores shared bands, labels, class events, colours, and paths.
- `visualization.py` contains exploratory, diagnostic, spatial, and modelling
  plots.
- `scripts/run_notebook2.py`, `run_notebook3.py`, and `run_notebook4.py`
  provide script equivalents of the main notebook stages.

The preprocessing contract became a 60-column feature matrix containing:

- overall band and index means;
- overall band and index standard deviations;
- early-period band statistics;
- late-period band statistics;
- year-over-year change statistics;
- early and late observation indicators.

Automated tests replaced several manual checks. The test suite covered age
class assignment, observation selection, spectral-index calculation, feature
column contracts, prototype predictions, preprocessing serialization, and the
Notebook 3 output contract.

The refactoring initially left generated Python bytecode, Jupyter checkpoints,
and preprocessed data in version control. Those artifacts were later addressed
by repository hygiene changes.

## 3. Memory-management experiments

The project then investigated memory usage for the large annual satellite
tables. The work included:

- alternative loading and preprocessing experiments;
- an Arrow-backed preprocessing variant;
- Memray recordings for Notebook 2, Notebook 3, and Notebook 4;
- Memray binary recordings, statistics, allocation trees, and flamegraphs;
- changes to valid-observation selection to reduce repeated temporary arrays.

The experiments confirmed that the workflows process millions of rows and
that Notebook 2 can open many Matplotlib figures. A warning such as:

```text
More than 20 figures have been opened
```

is a resource warning rather than a model failure, but it indicates that long
interactive runs can retain substantial memory.

Memray files were initially written directly in the repository root, and some
were also written under checkpoint and test directories. This made it easy to
mistake profiling output for source code and caused repository clutter.

## 4. Test and repository hygiene improvements

The project added automated pre-commit checks for formatting, debug
statements, YAML validity, trailing whitespace, tests, and notebook output
clearing.

A repository `.gitignore` was then restored and expanded to cover:

- Python caches and test caches;
- Jupyter checkpoints;
- virtual environments;
- generated preprocessing output;
- executed notebook copies;
- `artifacts/`;
- `memory_profiles/`;
- Memray binaries, HTML reports, statistics, and tree files;
- common editor and operating-system files.

The ignore patterns cover both underscore and hyphenated Memray names, for
example:

```text
memray_notebook3.bin
memray-flamegraph-memray_notebook4_arrow.html
```

The project documentation was updated for Linux with `python3` commands.

## 5. Notebook execution and profiling helpers

`scripts/run_notebooks.py` was added to execute the four notebooks in their
dependency order. It writes executed copies to:

```text
artifacts/executed-notebooks/
```

The source notebooks are not modified by this command.

`scripts/profile_memory.py` was added to run a selected workflow under Memray
and generate:

```text
<label>.bin
<label>_stats.txt
<label>_tree.txt
<label>.html
```

under:

```text
memory_profiles/
```

The profiler later required a command-line correction. Memray does not accept
an interpreter followed by `-m` as a positional script target. The correct
module form is:

```text
memray run -o output.bin -m scripts.run_notebooks
```

The helper was updated accordingly.

## 6. Controlled Amsterdam few-shot evaluation

The original Notebook 4 prototype experiment selected random Amsterdam
support pixels from the complete dataset and evaluated on the remaining
pixels. That procedure was useful for exploration, but it did not strictly
protect validation folds from support selection.

The controlled evaluation introduced:

- 5-fold stratified cross-validation;
- 3 repeats;
- 15 validation scores for every feature set, method, and budget;
- support selection only from each fold's training portion;
- untouched validation portions;
- budgets of 5, 25, 50, 100, and 200 labelled pixels per class;
- controlled support-selection seeds;
- Random Forest, logistic regression, and prototype methods.

The new runner is:

```bash
python3 -m scripts.run_few_shot_evaluation
```

It writes:

```text
artifacts/few_shot_evaluation.csv
```

The feature comparison uses the existing engineered groups:

| Configuration | Feature groups |
|---|---|
| Baseline | `overall + indicators` |
| Spectral | `overall + indices + indicators` |
| Temporal | `overall + early + late + yoy + indicators` |
| Full | all existing groups |

This is an evaluation-layer feature ablation. It compares already engineered
columns; it does not claim that new raw spectral or temporal measurements were
added to the preprocessing stage.

## 7. Feature-evaluation findings

The controlled comparison showed that temporal information was substantially
more transferable than the baseline reflectance summaries. At 25 Amsterdam
samples per class, the development results were approximately:

| Configuration | Random Forest macro-F1 |
|---|---:|
| Baseline | 0.515 |
| Temporal | 0.631 |
| Full | 0.637-0.644 |

The exact result varies slightly with estimator count and the selected
configuration, so these values are development measurements rather than a
single final leaderboard claim.

Additional Random Forest experiments tested feature subsampling, class
weighting, tree depth, and minimum leaf sizes. The changes beyond the full
feature configuration were small and inconsistent across label budgets, so no
additional model-tuning change was adopted.

The existing Notebook 4 workflow produces different values because it uses
prototype classification, ten random trials, and the remaining Amsterdam
pixels as the query set. For example, it reported approximately:

```text
Amsterdam zero-shot: 0.4433
Amsterdam 25-shot prototype: 0.5986 +/- 0.0176
Amsterdam 200-shot prototype: 0.6150 +/- 0.0036
```

Those numbers should not be compared directly with the controlled
fold-isolated Random Forest results.

## 8. Notebook 1 presentation output

The controlled evaluator now generates the required Amsterdam curve:

```text
artifacts/few_shot_f1_vs_log2_budget.png
```

The default plot shows the full feature set with Random Forest. It contains:

- `log2(samples per class)` on the x-axis;
- mean macro-F1 on the y-axis;
- standard-deviation error bars across the 15 validation scores.

Another feature set or method can be selected with:

```bash
python3 -m scripts.run_few_shot_evaluation \
  --feature-set temporal \
  --method logistic_regression
```

## 9. Repository-root path hardening

The shared-VM use case exposed an important limitation: relative paths depend
on the process working directory. If a notebook, Jupyter kernel, or profiling
command was launched from a shared directory, output could be created there.

The path-hardening update introduced:

- `REPO_ROOT` derived from the source file location;
- `resolve_repo_path()` for relative project paths;
- repository-root defaults for data, preprocessing, artifacts,
  executed notebooks, and Memray reports;
- explicit repository-root working directories for notebook and Memray
  subprocesses;
- a test that changes the caller's working directory and verifies that paths
  still resolve to the project.

The intended private output locations are:

```text
<project>/data/preprocessed/
<project>/artifacts/
<project>/memory_profiles/
```

This protects the Python runner defaults, but directly running old notebooks
or legacy scripts that contain `./data` can still depend on the launch
directory. The supported shared-VM workflow is to run the module-based
scripts from the project installation and use `umask 077`.

## 10. VM conflict and profiling issues

When the project was copied into a Linux VM with several users sharing
resources, local VM work was mixed with Git history and generated files. This
caused:

- divergent local and remote histories;
- unresolved README and checkpoint conflicts;
- Python files containing `<<<<<<<`, `=======`, and `>>>>>>>` markers;
- `SyntaxError: invalid decimal literal` when Python interpreted a commit hash
  as source code;
- `IndentationError` after a manual partial edit of `profile_memory.py`;
- an initial Memray invocation error because the interpreter command was
  passed as a positional target.

The safe recovery sequence is:

1. abort an incomplete rebase, merge, or cherry-pick;
2. remove or resolve all conflict markers;
3. compile the affected Python files;
4. run the tests;
5. run the pipeline only after the checks pass.

Useful checks are:

```bash
grep -RIn --exclude-dir=.git \
  -E '^(<<<<<<<|=======|>>>>>>>)' \
  --include='*.py' --include='*.ipynb' --include='*.md' .

python3 -m py_compile scripts/profile_memory.py
python3 -m pytest -q
```

## 11. Current supported workflow

From the private project directory on Linux:

```bash
cd ~/Building_Age_Classification
umask 077

python3 -m pytest -q
python3 -m scripts.run_notebook2
python3 -m scripts.run_notebook3
python3 -m scripts.run_notebook4
python3 -m scripts.run_few_shot_evaluation
```

To execute the notebooks themselves:

```bash
python3 -m scripts.run_notebooks
```

To profile a workflow:

```bash
python3 -m scripts.profile_memory notebooks
```

Before running on a shared VM, verify that no conflict markers remain and
confirm the VM's TCP listeners separately. The path hardening prevents the
supported Python runners from writing to a shared output directory, but it does
not secure a globally exposed Jupyter server or other network service.

## 12. Open limitations

- The final presentation and written justification from Notebook 1 are not
  generated automatically.
- The current feature expansion compares existing feature groups rather than
  adding new raw predictors.
- Amsterdam results used during feature selection are development evidence;
  a final frozen evaluation should be run after the feature set, method,
  hyperparameters, and seeds are fixed.
- The direct notebook cells still contain some relative paths and should be
  executed through `scripts.run_notebooks.py` when path isolation matters.
- The plotting code can retain many Matplotlib figures during exploratory
  runs; closing figures would reduce memory pressure further.