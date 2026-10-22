import numpy as np

from model_training import (
    cross_validate_few_shot,
    prototype_predict,
    prototype_predict_proba,
)
from visualization import plot_few_shot_budget_results


def test_prototype_predict_returns_nearest_class_for_each_query():
    X_support = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_support = np.array([1, 2])
    X_query = np.array([[1.0, 1.0], [9.0, 9.0]])

    predictions = prototype_predict(X_support, y_support, X_query)

    assert predictions.shape == (2,)
    assert predictions.tolist() == [1, 2]


def test_prototype_predict_proba_returns_normalized_class_probabilities():
    X_support = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_support = np.array([1, 2])
    X_query = np.array([[1.0, 1.0], [9.0, 9.0]])

    classes, probabilities = prototype_predict_proba(
        X_support,
        y_support,
        X_query,
    )

    assert classes.tolist() == [1, 2]
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_cross_validate_few_shot_uses_requested_budgets_and_scores():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = np.repeat([1, 2, 3, 4], 20)

    results = cross_validate_few_shot(
        X,
        y,
        {"baseline": np.arange(4)},
        methods=("prototype",),
        budgets=(2,),
        cv_config={"folds": 2, "repeats": 1},
        seed=42,
    )

    assert len(results) == 1
    assert results[0]["feature_set"] == "baseline"
    assert results[0]["samples_per_class"] == 2
    assert len(results[0]["scores"]) == 2
    assert 0.0 <= results[0]["mean_macro_f1"] <= 1.0


def test_few_shot_plot_writes_log2_budget_figure(tmp_path):
    import pandas as pd

    results = pd.DataFrame(
        {
            "feature_set": ["full", "full"],
            "method": ["random_forest", "random_forest"],
            "samples_per_class": [5, 25],
            "mean_macro_f1": [0.5, 0.6],
            "std_macro_f1": [0.02, 0.01],
        }
    )
    output = tmp_path / "few_shot.png"

    plot_few_shot_budget_results(
        results,
        "full",
        "random_forest",
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
