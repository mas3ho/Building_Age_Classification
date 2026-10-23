from pathlib import Path

import pandas as pd

from utils import CLASS_EVENTS, resolve_repo_path


def assign_age_class(
    data: pd.DataFrame,
    events: dict[str, int] = CLASS_EVENTS,
) -> pd.DataFrame:
    out = data.copy()
    if "age_class" not in out:
        out["age_class"] = pd.NA

    for city, event_year in events.items():
        mask = out["city"] == city
        if mask.any() and "weighted_mean_year" in out:
            bins = [-float("inf"), event_year, 1984, 2004, float("inf")]
            out.loc[mask, "age_class"] = pd.cut(
                out.loc[mask, "weighted_mean_year"],
                bins=bins,
                labels=[1, 2, 3, 4],
            ).astype("Int64")
    return out


def load_city_data(path: Path | str, assign_classes: bool = True) -> pd.DataFrame:
    """Load one city's Parquet data and optionally assign age classes."""
    data = pd.read_parquet(path)
    return assign_age_class(data) if assign_classes else data


def load_city_datasets(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load and label both city datasets from one data directory."""
    data_dir = resolve_repo_path(data_dir)
    madrid_path = data_dir / "madrid_train.parquet"
    amsterdam_path = data_dir / "amsterdam_data.parquet"
    return {
        "Madrid": load_city_data(madrid_path),
        "Amsterdam": load_city_data(amsterdam_path),
    }
