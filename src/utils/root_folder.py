"""finds the root project of the project"""

from pathlib import Path


def find_project_root(marker="environment.yml"):
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory.")


if __name__ == "__main__":
    print(find_project_root())
