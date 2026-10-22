"""Run the controlled Amsterdam few-shot feature comparison."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model_training import cross_validate_few_shot
from preprocessing_io import load_preprocessed_data


FEATURE_SET_GROUPS = {
    "baseline": ("overall", "indicators"),
    "spectral": ("overall", "indices", "indicators"),
    "temporal": ("overall", "early", "late", "yoy", "indicators"),
    "full": ("overall", "indices", "early", "late", "yoy", "indicators"),
}


def build_feature_sets(feature_names, feature_groups):
    positions = {name: index for index, name in enumerate(feature_names)}
    return {
        feature_set: np.asarray(
            [
                positions[column]
                for group in groups
                for column in feature_groups[group]
            ]
        )
        for feature_set, groups in FEATURE_SET_GROUPS.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Amsterdam few-shot feature sets with repeated CV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "preprocessed" / "preprocessed_data.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "few_shot_evaluation.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    data = load_preprocessed_data(args.input)
    feature_sets = build_feature_sets(data["feature_names"], data["feat_groups"])
    results = cross_validate_few_shot(
        data["X_amsterdam"],
        data["y_amsterdam"],
        feature_sets,
        cv_config={"folds": 5, "repeats": 3},
        model_params={
            "n_estimators": args.n_estimators,
            "class_weight": "balanced",
            "random_state": args.seed,
            "n_jobs": -1,
        },
        seed=args.seed,
    )
    table = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "scores"
            }
            for result in results
        ]
    )
    table = table.sort_values(
        ["samples_per_class", "feature_set", "method"]
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
