from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from model_training import (
    cross_validate_random_forest,
    run_prototype_trials,
    train_random_forest,
)
from preprocessing_io import load_preprocessed_data
from utils import CLASS_LABELS
from visualization import (
    plot_confusion_matrix,
    plot_cross_validation_results,
    plot_feature_importance,
    plot_prototype_results,
    plot_spatial_accuracy,
    plot_spatial_uncertainty,
)


RF_PARAMS = dict(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
CLASS_LABELS = {
    "Madrid": {1: "Pre-1960", 2: "1960-1984", 3: "1984-2004", 4: "2004-2024"},
    "Amsterdam": {1: "Pre-1945", 2: "1945-1984", 3: "1984-2004", 4: "2004-2024"},
}


def main():
    data = load_preprocessed_data(
        Path("data") / "preprocessed" / "preprocessed_data.pkl"
    )
    X_madrid, y_madrid = data["X_madrid"], data["y_madrid"]
    X_amsterdam, y_amsterdam = data["X_amsterdam"], data["y_amsterdam"]
    pixel_ids_madrid = data["pixel_ids_madrid"]
    pixel_ids_amsterdam = data["pixel_ids_amsterdam"]
    feature_names = data["feature_names"]
    print(f"Madrid: {X_madrid.shape} ({len(np.unique(y_madrid))} classes)")
    print(f"Amsterdam: {X_amsterdam.shape} ({len(np.unique(y_amsterdam))} classes)")
    print(f"Features: {len(feature_names)}")

    folds, repeats = 5, 5
    cv = cross_validate_random_forest(
        X_madrid,
        y_madrid,
        {"folds": folds, "repeats": repeats},
        RF_PARAMS,
    )
    scores = cv["scores"]
    oof_true = cv["oof_true"]
    oof_pred = cv["oof_pred"]
    plot_cross_validation_results(
        scores,
        oof_true,
        oof_pred,
        CLASS_LABELS["Madrid"],
    )
    plot_spatial_uncertainty(
        pixel_ids_madrid[cv["oof_indices"]],
        cv["oof_proba"],
        oof_true,
        "Madrid CV (out-of-fold)",
    )
    plot_spatial_accuracy(
        pixel_ids_madrid[cv["oof_indices"]],
        oof_true,
        oof_pred,
        "Madrid CV (out-of-fold)",
    )
    print(f"\n{folds}-fold x {repeats}-repeat CV on Madrid")
    for fit_num, score in enumerate(scores, 1):
        print(f"  Fit {fit_num:3d}/{folds * repeats}: macro-F1 = {score:.4f}")
    print(f"Mean macro F1: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    final_model = train_random_forest(X_madrid, y_madrid, RF_PARAMS)
    madrid_pred = final_model.predict(X_madrid)
    print(f"In-sample accuracy: {accuracy_score(y_madrid, madrid_pred):.4f}")

    importances = final_model.feature_importances_
    plot_feature_importance(importances, feature_names)
    for rank in np.argsort(importances)[::-1][:10]:
        print(f"{feature_names[rank]}: {importances[rank]:.4f}")

    zero_pred = final_model.predict(X_amsterdam)
    zero_proba = final_model.predict_proba(X_amsterdam)
    zero_f1 = f1_score(y_amsterdam, zero_pred, average="macro")
    print(f"\nZero-shot Amsterdam macro F1: {zero_f1:.4f}")
    print(
        classification_report(
            y_amsterdam,
            zero_pred,
            labels=sorted(np.unique(y_amsterdam)),
            target_names=[f"Class {c}" for c in sorted(np.unique(y_amsterdam))],
            digits=4,
        )
    )
    plot_confusion_matrix(
        y_amsterdam,
        zero_pred,
        CLASS_LABELS["Amsterdam"],
        f"Zero-shot transfer - Amsterdam (macro F1={zero_f1:.3f})",
    )
    plot_spatial_uncertainty(
        pixel_ids_amsterdam,
        zero_proba,
        y_amsterdam,
        "Amsterdam zero-shot",
    )
    plot_spatial_accuracy(
        pixel_ids_amsterdam,
        y_amsterdam,
        zero_pred,
        "Amsterdam zero-shot",
    )

    shots_per_class = [5, 10, 25, 50, 100, 200]
    results = run_prototype_trials(
        X_amsterdam,
        y_amsterdam,
        {"shots_per_class": shots_per_class, "trials": 10},
    )
    for shots in shots_per_class:
        print(
            f"{shots:4d} shots/class: macro-F1 = "
            f"{np.mean(results[shots]):.4f} +/- {np.std(results[shots]):.4f}"
        )
    plot_prototype_results(results, zero_f1, np.mean(scores))

    print("\nNotebook 4 modelling pipeline completed successfully.")


if __name__ == "__main__":
    main()
