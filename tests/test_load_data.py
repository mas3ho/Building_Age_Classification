import pandas as pd

from load_data import assign_age_class
from utils import REPO_ROOT, default_data_paths, resolve_repo_path


def test_assign_age_class_uses_city_specific_boundaries():
    data = pd.DataFrame(
        {
            "city": ["Madrid", "Madrid", "Amsterdam", "Amsterdam"],
            "weighted_mean_year": [1950, 1990, 1940, 1990],
        }
    )

    result = assign_age_class(data)

    assert result["age_class"].tolist() == [1, 3, 1, 3]
    assert "age_class" not in data.columns


def test_relative_project_paths_are_anchored_to_repository_root():
    assert resolve_repo_path("artifacts") == REPO_ROOT / "artifacts"
    assert default_data_paths()[0] == REPO_ROOT / "data" / "madrid_train.parquet"
