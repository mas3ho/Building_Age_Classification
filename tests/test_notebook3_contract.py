import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from scripts.run_notebook3 import (
    build_preprocessed_output,
    standardize_city_features,
)


def test_standardize_city_features_preserves_feature_contract():
    feature_columns = ["feature_a", "feature_b"]
    features = {
        "Madrid": pd.DataFrame(
            {
                "px_key": [1, 2],
                "py_key": [10, 20],
                "age_class": [1, 2],
                "feature_a": [0.0, 1.0],
                "feature_b": [10.0, 20.0],
            }
        ),
        "Amsterdam": pd.DataFrame(
            {
                "px_key": [3],
                "py_key": [30],
                "age_class": [1],
                "feature_a": [0.5],
                "feature_b": [15.0],
            }
        ),
    }

    standardized = standardize_city_features(features, feature_columns)
    output = build_preprocessed_output(
        features,
        standardized,
        feature_columns,
    )

    assert output["X_madrid"].shape == (2, 2)
    assert output["X_amsterdam"].shape == (1, 2)
    assert output["X_madrid"].shape[1] == len(output["feature_names"])
    assert len(output["y_madrid"]) == len(output["X_madrid"])
    assert len(output["y_amsterdam"]) == len(output["X_amsterdam"])
    assert output["pixel_ids_madrid"].shape == (2, 2)
    assert output["pixel_ids_amsterdam"].shape == (1, 2)
    assert np.isfinite(output["X_madrid"]).all()
