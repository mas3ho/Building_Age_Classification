import pandas as pd

from load_data import assign_age_class


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
