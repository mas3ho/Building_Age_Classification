import pickle
from pathlib import Path


def save_preprocessed_data(output, output_path: Path | str) -> Path:
    """Serialize the prepared feature arrays and metadata to a pickle file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(output, handle)
    return path


def load_preprocessed_data(input_path: Path | str) -> dict:
    """Load the serialized feature arrays and metadata."""
    with Path(input_path).open("rb") as handle:
        return pickle.load(handle)
