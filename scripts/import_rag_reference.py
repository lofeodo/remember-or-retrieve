"""Import the recorded rag-with-receipts golden-set run as a slim, committed reference.

One-off: reads results/eval_report.json from a local rag-with-receipts checkout (path given as
the argument), drops chunk text (it resolves from our corpus by chunk_id), writes
data/eval/rag_reference.json and records it in data/PROVENANCE.json. The reference lets the
scorer be validated offline: replaying its verdicts must reproduce the published numbers.

Usage: python scripts/import_rag_reference.py PATH_TO_rag-with-receipts
"""

import json
import subprocess
import sys
from pathlib import Path

from remember_or_retrieve.config import REPO_ROOT
from remember_or_retrieve.data import load_corpus, sha256_of

OUT = REPO_ROOT / "data" / "eval" / "rag_reference.json"
PROVENANCE = REPO_ROOT / "data" / "PROVENANCE.json"


def slim(result: dict) -> dict:
    gen = result["generated"]
    judge = result["judge"]
    grounding = result["grounding"]
    return {
        "question_id": result["question"]["id"],
        "answer": gen["answer"],
        "answerable": gen["answerable"],
        "citations": [{"chunk_id": c["chunk_id"], "claim": c["claim"]} for c in gen["citations"]],
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "judge_verdict": judge["verdict"] if judge else None,
        "correctness_label": result["correctness_label"],
        "citation_verdicts": [
            {k: v for k, v in cv.items()}
            for cv in (grounding["citation_verdicts"] if grounding else [])
        ],
        "error": result["error"],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    upstream = Path(sys.argv[1])
    src = upstream / "results" / "eval_report.json"
    report = json.loads(src.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    corpus = {c.chunk_id for c in load_corpus()}
    rows = [slim(r) for r in report["results"]]
    missing = {c["chunk_id"] for r in rows for c in r["citations"] if c["chunk_id"] not in corpus}
    if missing:
        print(f"FAIL: cited chunk ids not in our corpus: {sorted(missing)}")
        return 1
    out = {
        "source": "rag-with-receipts results/eval_report.json",
        "source_commit": commit,
        "judge_model": "claude-sonnet-5",
        "summary": report["summary"],
        "results": rows,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prov["files"]["data/eval/rag_reference.json"] = {
        "source_path": "results/eval_report.json (slimmed: chunk text dropped)",
        "source_commit": commit,
        "sha256": sha256_of(OUT),
        "count": len(rows),
    }
    with open(PROVENANCE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"OK: {len(rows)} rows -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
