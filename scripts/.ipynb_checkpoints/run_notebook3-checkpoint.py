from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    build_pixel_features,
    compute_spectral_indices,
    fill_time_series_gaps,
    feature_column_groups,
    make_flat_view,
    period_coverage,
)
from load_data import load_city_datasets
from preprocessing_io import save_preprocessed_data
from utils import CLASS_COLORS, CLASS_LABELS
from visualization import plot_feature_class_distribution


def standardize_city_features(
    features,
    feature_columns,
    fit_city="Madrid",
    transform_city="Amsterdam",
):
    """Fit a scaler on one city and apply it to a second city's features."""
    scaler = StandardScaler()
    fit_features = features[fit_city]
    transform_features = features[transform_city]
    X_fit = scaler.fit_transform(fit_features[feature_columns].astype(float))
    X_transform = scaler.transform(
        transform_features[feature_columns].astype(float)
    )
    return {
        "X_madrid": X_fit,
        "y_madrid": fit_features["age_class"].to_numpy(dtype=int),
        "X_amsterdam": X_transform,
        "y_amsterdam": transform_features["age_class"].to_numpy(dtype=int),
        "scaler": scaler,
    }


def build_preprocessed_output(features, standardized, feature_groups):
    """Assemble the Notebook 3 output contract for Notebook 4."""
    if isinstance(feature_groups, list):
        feature_groups = {"features": feature_groups}
    feature_columns = [
        column
        for group in feature_groups.values()
        for column in group
    ]
    return {
        **standardized,
        "feature_names": feature_columns,
        "feat_groups": feature_groups,
        "pixel_ids_madrid": features["Madrid"][["px_key", "py_key"]].values,
        "pixel_ids_amsterdam": features["Amsterdam"][["px_key", "py_key"]].values,
    }


def main():
    raw = load_city_datasets(Path("data"))
    features = {}
    for name, data in raw.items():
        flat = make_flat_view(data)
        indicators = period_coverage(flat)
        flat = fill_time_series_gaps(flat)
        flat = compute_spectral_indices(flat)
        city_features = build_pixel_features(flat).merge(
            indicators,
            on=["px_key", "py_key"],
            how="left",
        )
        features[name] = city_features
        print(
            f"{name}: {len(flat):,} rows after preparation "
            f"({flat['year'].nunique()} years)"
        )
    feature_groups = feature_column_groups()
    feature_cols = [
        column
        for group in feature_groups.values()
        for column in group
    ]
    plot_feature_class_distribution(
        features,
        CLASS_LABELS,
        CLASS_COLORS,
    )
    standardized = standardize_city_features(features, feature_cols)
    X_madrid = standardized["X_madrid"]
    X_amsterdam = standardized["X_amsterdam"]
    print(f"Madrid: {X_madrid.shape}, NaN check: {not np.isnan(X_madrid).any()}")
    print(f"Amsterdam: {X_amsterdam.shape}, NaN check: {not np.isnan(X_amsterdam).any()}")

    output = build_preprocessed_output(features, standardized, feature_groups)
    save_path = save_preprocessed_data(
        output,
        Path("data") / "preprocessed" / "preprocessed_data.pkl",
    )
    print(f"Saved to: {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
