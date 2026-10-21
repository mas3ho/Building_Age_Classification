import numpy as np
import pandas as pd

from utils import BASE_BANDS, BLUE_MAX

PID = ["px_key", "py_key"]


def get_best_obs_band(data: pd.DataFrame, band: str) -> np.ndarray:
    values = np.full(len(data), np.nan)
    for slot in [1, 2, 3]:
        band_col = f"{band}_{slot}"
        valid_col = f"qa_valid_{slot}"
        if band_col not in data.columns or valid_col not in data.columns:
            continue
        valid = data[valid_col].fillna(False).to_numpy()
        values = np.where(
            np.isnan(values) & valid,
            data[band_col].fillna(np.nan).to_numpy(),
            values,
        )
    return values


def make_flat_view(data: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in [
            "pixel_id",
            "city",
            "year",
            "px_key",
            "py_key",
            "weighted_mean_year",
            "age_class",
        ]
        if column in data.columns
    ]
    out = data[keep].copy()
    for band in BASE_BANDS:
        out[band] = get_best_obs_band(data, band)
    out = out.dropna(subset=BASE_BANDS)

    before = len(out)
    out = out[out["Blue"] <= BLUE_MAX].copy()
    removed = before - len(out)
    if removed:
        city = data["city"].iloc[0] if "city" in data.columns else ""
        print(f"  [{city}] Removed {removed:,} haze-contaminated rows (Blue > {BLUE_MAX:,})")
    return out


def period_coverage(flat: pd.DataFrame) -> pd.DataFrame:
    pixels = flat[PID].drop_duplicates()
    early = (
        flat[flat["year"] <= 2003]
        .drop_duplicates(PID)
        .assign(has_early_data=1.0)
    )
    late = (
        flat[flat["year"] >= 2004]
        .drop_duplicates(PID)
        .assign(has_late_data=1.0)
    )
    return (
        pixels.merge(early[PID + ["has_early_data"]], on=PID, how="left")
        .merge(late[PID + ["has_late_data"]], on=PID, how="left")
        .fillna(0.0)
    )


def fill_time_series_gaps(flat: pd.DataFrame) -> pd.DataFrame:
    years = range(int(flat["year"].min()), int(flat["year"].max()) + 1)
    pixels = flat[PID + ["city", "weighted_mean_year", "age_class"]].drop_duplicates(PID)
    grid = pixels.assign(_key=1).merge(
        pd.DataFrame({"year": years, "_key": 1}), on="_key"
    ).drop(columns="_key")
    merged = grid.merge(
        flat[PID + ["year"] + BASE_BANDS],
        on=PID + ["year"],
        how="left",
    ).sort_values(PID + ["year"])

    def interpolate(series: pd.Series) -> pd.Series:
        return series.interpolate(method="linear").bfill().ffill()

    for band in BASE_BANDS:
        merged[band] = merged.groupby(PID)[band].transform(interpolate)
    return merged


def compute_pixel_stats(flat: pd.DataFrame) -> pd.DataFrame:
    ordered = flat.sort_values(PID + ["year"]).copy()
    group = ordered.groupby(PID, sort=False)
    stats = pd.concat(
        [
            group[BASE_BANDS].mean().add_suffix("_mean"),
            group[BASE_BANDS].std().add_suffix("_std"),
        ],
        axis=1,
    )
    differences = ordered.groupby(PID)[BASE_BANDS].diff()
    stats = pd.concat(
        [
            stats,
            differences.groupby([ordered["px_key"], ordered["py_key"]]).mean()
            .add_suffix("_mean")
            .add_prefix("d_"),
            differences.groupby([ordered["px_key"], ordered["py_key"]]).std()
            .add_suffix("_std")
            .add_prefix("d_"),
        ],
        axis=1,
    )
    labels = ordered.groupby(PID, sort=False)["age_class"].first()
    return stats.join(labels).reset_index()


def find_best_pixels(
    flat: pd.DataFrame,
    age_class: int,
    n: int = 3,
) -> list[tuple[int, int]]:
    pixels = flat[flat["age_class"] == age_class].drop_duplicates(PID)
    if pixels.empty:
        return []
    selected = pixels.sort_values("weighted_mean_year").iloc[
        np.linspace(0, len(pixels) - 1, min(n, len(pixels))).round().astype(int)
    ]
    return list(selected[PID].itertuples(index=False, name=None))


def get_pixel_ts(
    flat: pd.DataFrame,
    pixel_id: tuple[int, int],
    band: str,
) -> pd.DataFrame:
    px, py = pixel_id
    return flat[(flat["px_key"] == px) & (flat["py_key"] == py)][
        ["year", band, "weighted_mean_year", "age_class"]
    ].sort_values("year")


def get_yearly_means(flat: pd.DataFrame, band: str) -> pd.DataFrame:
    return (
        flat.groupby(["year", "age_class"])[band]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
