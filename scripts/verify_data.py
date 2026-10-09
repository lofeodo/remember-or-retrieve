"""Verify the corpus and golden set against data/PROVENANCE.json. Exits 1 on any mismatch."""

import sys

from remember_or_retrieve.data import verify_data


def main() -> int:
    problems = verify_data()
    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print("OK: data files match PROVENANCE.json")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
