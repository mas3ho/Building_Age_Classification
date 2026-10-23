"""Execute the project notebooks in dependency order."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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
        default=REPO_ROOT / "artifacts" / "executed-notebooks",
        help="Directory for executed notebook copies.",
    )
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for notebook in NOTEBOOKS:
        output = args.output_dir / notebook
        notebook_path = REPO_ROOT / notebook
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            f"--ExecutePreprocessor.cwd={REPO_ROOT}",
            "--output",
            str(output),
            str(notebook_path),
        ]
        print("$", " ".join(command))
        subprocess.run(command, check=True, cwd=REPO_ROOT)

    print(f"Executed notebooks written to {args.output_dir}")


if __name__ == "__main__":
    main()
