from pathlib import Path


BASE_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BLUE_MAX = 15_000
CLASS_EVENTS = {"Amsterdam": 1945, "Madrid": 1960}
CLASS_COLORS = {
    1: "#4393c3",
    2: "#2ca25f",
    3: "#fd8d3c",
    4: "#de2d26",
}
CLASS_LABELS = {
    "Amsterdam": {
        1: "Pre-1945",
        2: "1945-1984",
        3: "1984-2004",
        4: "2004-2024",
    },
    "Madrid": {
        1: "Pre-1960",
        2: "1960-1984",
        3: "1984-2004",
        4: "2004-2024",
    },
}


def default_data_paths(data_dir: Path | str = "data") -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    return data_dir / "madrid_train.parquet", data_dir / "amsterdam_data.parquet"


def feature_columns() -> list[str]:
    from feature_engineering import feature_column_groups

    groups = feature_column_groups()
    return [column for group in groups.values() for column in group]
