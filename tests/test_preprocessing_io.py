import numpy as np

from preprocessing_io import load_preprocessed_data, save_preprocessed_data


def test_preprocessed_data_round_trip(tmp_path):
    output = {
        "X_madrid": np.array([[1.0, 2.0]]),
        "y_madrid": np.array([1]),
        "feature_names": ["feature_a", "feature_b"],
    }
    output_path = tmp_path / "preprocessed" / "data.pkl"

    saved_path = save_preprocessed_data(output, output_path)
    loaded = load_preprocessed_data(saved_path)

    assert saved_path == output_path
    assert saved_path.exists()
    assert np.array_equal(loaded["X_madrid"], output["X_madrid"])
    assert np.array_equal(loaded["y_madrid"], output["y_madrid"])
    assert loaded["feature_names"] == output["feature_names"]
