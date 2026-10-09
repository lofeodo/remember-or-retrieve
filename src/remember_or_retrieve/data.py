"""Load the corpus and golden set, and verify them against the provenance manifest."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from remember_or_retrieve.config import REPO_ROOT, load_config, resolve_path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_title: str
    url: str
    section_path: str
    heading_anchor: str
    token_count: int
    source_type: str
    text: str


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    type: str | None  # "single_hop", "multi_hop", or None for unanswerable
    answerable: bool
    gold_chunk_ids: tuple[str, ...]
    gold_answer: str


def load_corpus(path: str | Path | None = None) -> list[Chunk]:
    path = path or resolve_path(load_config()["data"]["corpus_path"])
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def load_golden_set(path: str | Path | None = None) -> list[GoldenQuestion]:
    path = path or resolve_path(load_config()["data"]["golden_set_path"])
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [GoldenQuestion(**{**q, "gold_chunk_ids": tuple(q["gold_chunk_ids"])}) for q in raw]


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_data(root: str | Path = REPO_ROOT) -> list[str]:
    """Check every file in PROVENANCE.json against its recorded SHA-256.

    Returns a list of problems; empty means everything matches.
    """
    root = Path(root)
    manifest = json.loads((root / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))
    problems = []
    for rel, info in manifest["files"].items():
        file = root / rel
        if not file.exists():
            problems.append(f"{rel}: missing")
        elif sha256_of(file) != info["sha256"]:
            problems.append(f"{rel}: SHA-256 does not match PROVENANCE.json")
    return problems
