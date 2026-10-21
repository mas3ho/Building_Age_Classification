import numpy as np

from model_training import prototype_predict, prototype_predict_proba


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
