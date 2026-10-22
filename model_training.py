import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


def prototype_predict(X_support, y_support, X_query):
    classes = np.unique(y_support)
    prototypes = np.stack(
        [X_support[y_support == cls].mean(axis=0) for cls in classes]
    )
    distances = np.linalg.norm(
        X_query[:, None, :] - prototypes[None, :, :], axis=2
    )
    return classes[distances.argmin(axis=1)]


def prototype_predict_proba(X_support, y_support, X_query):
    classes = np.unique(y_support)
    prototypes = np.stack(
        [X_support[y_support == cls].mean(axis=0) for cls in classes]
    )
    distances = np.linalg.norm(
        X_query[:, None, :] - prototypes[None, :, :], axis=2
    )
    negative_distances = -distances
    negative_distances -= negative_distances.max(axis=1, keepdims=True)
    probabilities = np.exp(negative_distances)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return classes, probabilities


def cross_validate_random_forest(
    X,
    y,
    cv_config=None,
    model_params=None,
):
    """Evaluate a Random Forest with repeated stratified cross-validation."""
    cv_config = cv_config or {"folds": 5, "repeats": 5}
    folds = cv_config["folds"]
    repeats = cv_config["repeats"]
    params = model_params or {
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=42,
    )
    scores = []
    oof_true = []
    oof_pred = []
    oof_proba = []
    oof_indices = []
    for train_idx, valid_idx in splitter.split(X, y):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[valid_idx])
        probabilities = model.predict_proba(X[valid_idx])
        scores.append(f1_score(y[valid_idx], predictions, average="macro"))
        oof_true.extend(y[valid_idx])
        oof_pred.extend(predictions)
        oof_proba.extend(probabilities)
        oof_indices.extend(valid_idx)
    return {
        "scores": np.asarray(scores),
        "oof_true": np.asarray(oof_true),
        "oof_pred": np.asarray(oof_pred),
        "oof_proba": np.asarray(oof_proba),
        "oof_indices": np.asarray(oof_indices),
    }


def train_random_forest(X, y, model_params=None):
    """Fit a Random Forest on all supplied labelled features."""
    params = model_params or {
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    return model.fit(X, y)


def run_prototype_trials(
    X,
    y,
    trial_config=None,
    seed=42,
):
    """Measure few-shot prototype macro-F1 for each requested label budget."""
    from sklearn.metrics import f1_score

    trial_config = trial_config or {"shots_per_class": [5, 10, 25, 50, 100, 200], "trials": 10}
    shots_per_class = trial_config["shots_per_class"]
    trials = trial_config["trials"]
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    results = {shots: [] for shots in shots_per_class}
    for shots in shots_per_class:
        for _ in range(trials):
            support = []
            for cls in classes:
                class_indices = np.where(y == cls)[0]
                support.extend(
                    rng.choice(
                        class_indices,
                        min(shots, len(class_indices)),
                        replace=False,
                    )
                )
            support = np.asarray(support)
            query = np.ones(len(y), dtype=bool)
            query[support] = False
            prediction = prototype_predict(X[support], y[support], X[query])
            results[shots].append(
                f1_score(y[query], prediction, average="macro", zero_division=0)
            )
    return results


def cross_validate_few_shot(
    X,
    y,
    feature_sets,
    methods=("random_forest", "logistic_regression", "prototype"),
    budgets=(5, 25, 50, 100, 200),
    cv_config=None,
    model_params=None,
    seed=42,
):
    """Evaluate few-shot Amsterdam support selection without validation leakage."""
    cv_config = cv_config or {"folds": 5, "repeats": 3}
    folds = cv_config["folds"]
    repeats = cv_config["repeats"]
    params = model_params or {
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )
    classes = np.unique(y)
    results = []

    for feature_name, feature_indices in feature_sets.items():
        feature_indices = np.asarray(feature_indices)
        for method in methods:
            for budget in budgets:
                scores = []
                for split_index, (train_idx, valid_idx) in enumerate(
                    splitter.split(X, y)
                ):
                    rng = np.random.default_rng(seed + split_index * 1000 + budget)
                    support_idx = []
                    for cls in classes:
                        class_train_idx = train_idx[y[train_idx] == cls]
                        if len(class_train_idx) < budget:
                            raise ValueError(
                                f"Budget {budget} exceeds class {cls} training "
                                f"count ({len(class_train_idx)})"
                            )
                        support_idx.extend(
                            rng.choice(class_train_idx, budget, replace=False)
                        )
                    support_idx = np.asarray(support_idx)
                    X_support = X[support_idx][:, feature_indices]
                    X_valid = X[valid_idx][:, feature_indices]
                    y_support = y[support_idx]

                    if method == "random_forest":
                        model = RandomForestClassifier(**params)
                        model.fit(X_support, y_support)
                        predictions = model.predict(X_valid)
                    elif method == "logistic_regression":
                        model = LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=seed,
                        )
                        model.fit(X_support, y_support)
                        predictions = model.predict(X_valid)
                    elif method == "prototype":
                        predictions = prototype_predict(
                            X_support,
                            y_support,
                            X_valid,
                        )
                    else:
                        raise ValueError(f"Unsupported few-shot method: {method}")

                    scores.append(
                        f1_score(
                            y[valid_idx],
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    )

                results.append(
                    {
                        "feature_set": feature_name,
                        "method": method,
                        "samples_per_class": budget,
                        "mean_macro_f1": float(np.mean(scores)),
                        "std_macro_f1": float(np.std(scores)),
                        "scores": np.asarray(scores),
                    }
                )

    return results
