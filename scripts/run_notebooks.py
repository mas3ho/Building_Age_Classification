"""Execute the project notebooks in dependency order."""

import argparse
import subprocess
import sys
from pathlib import Path


NOTEBOOKS = (
    "1-Introduction.ipynb",
    "2-Reading_Data.ipynb",
    "3-Preprocessing.ipynb",
    "4-Modelling.ipynb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute all project notebooks in dependency order."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "executed-notebooks",
        help="Directory for executed notebook copies.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for notebook in NOTEBOOKS:
        output = args.output_dir / notebook
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--ExecutePreprocessor.cwd=.",
            "--output",
            str(output),
            notebook,
        ]
        print("$", " ".join(command))
        subprocess.run(command, check=True)

    print(f"Executed notebooks written to {args.output_dir}")


if __name__ == "__main__":
    main()
