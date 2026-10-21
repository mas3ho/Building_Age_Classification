import numpy as np
import pandas as pd

from feature_engineering import (
    compute_spectral_indices,
    get_best_obs_band,
    select_feature_columns,
)


def test_get_best_obs_band_selects_first_valid_observation():
    data = pd.DataFrame(
        {
            "Blue_1": [100, 200],
            "Blue_2": [110, 220],
            "qa_valid_1": [False, True],
            "qa_valid_2": [True, True],
        }
    )

    result = get_best_obs_band(data, "Blue")

    assert result.tolist() == [110, 200]


def test_compute_spectral_indices_adds_finite_index_columns():
    data = pd.DataFrame(
        {
            "Blue": [100.0],
            "Green": [200.0],
            "Red": [150.0],
            "NIR": [300.0],
            "SWIR1": [250.0],
            "SWIR2": [275.0],
        }
    )

    result = compute_spectral_indices(data)
    index_columns = {"NDVI", "NDBI", "UI", "MNDWI", "BSI"}

    assert index_columns.issubset(result.columns)
    assert np.isfinite(result[list(index_columns)].to_numpy()).all()
    assert set(data.columns) == {
        "Blue",
        "Green",
        "Red",
        "NIR",
        "SWIR1",
        "SWIR2",
    }


def test_select_feature_columns_has_expected_model_contract():
    columns = select_feature_columns()

    assert len(columns) == 60
    assert len(columns) == len(set(columns))
    assert "Blue_mean" in columns
    assert "NDVI_mean" in columns
    assert "has_early_data" in columns
    assert "has_late_data" in columns
