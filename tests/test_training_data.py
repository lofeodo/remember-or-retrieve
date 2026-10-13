"""Checks over the frozen, committed training data in data/training/."""

import json
from collections import Counter

import pytest

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.data import load_corpus, load_golden_set, sha256_of
from remember_or_retrieve.datagen import prompts, validate

OUT = resolve_path(load_config()["training_data"]["out_dir"])
pytestmark = pytest.mark.skipif(not (OUT / "MANIFEST.json").exists(), reason="data not frozen yet")


def rows(name):
    with open(OUT / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_manifest_matches_files():
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        assert sha256_of(OUT / name) == info["sha256"], name
        assert len(rows(name)) == info["rows"], name
    assert manifest["spend_total_usd"] < 7  # roadmap's Claude line for steps 3 to 10
    assert manifest["template_hashes"] == prompts.template_hashes()


def test_sft_rows():
    sft = rows("sft_train.jsonl") + rows("sft_val.jsonl")
    assert len(sft) >= 4500
    assert len({r["id"] for r in sft}) == len(sft)
    corpus_ids = {c.chunk_id for c in load_corpus()}
    kinds = Counter(r["kind"] for r in sft)
    assert set(kinds) == set(prompts.KINDS)
    assert kinds["single_hop"] > kinds["multi_hop"] > 0
    assert 0.08 <= kinds["unanswerable"] / len(sft) <= 0.2
    for r in sft:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == prompts.SFT_SYSTEM_PROMPT
        assert set(r["chunk_ids"]) <= corpus_ids
        answer = r["messages"][2]["content"]
        assert (answer == prompts.ABSTAIN_PHRASE) == (r["kind"] == "unanswerable")
        if r["kind"] != "unanswerable":
            assert r["reference_facts"] and r["evidence"]


def test_dpo_rows():
    dpo = rows("dpo.jsonl")
    assert len(dpo) >= 1400
    assert len({r["id"] for r in dpo}) == len(dpo)
    assert set(Counter(r["rejected_kind"] for r in dpo)) == {
        "hallucinated",
        "verbose",
        "fabricated_abstention",
    }
    for r in dpo:
        assert r["chosen"] != r["rejected"] and r["rejected"].strip()
        assert [m["role"] for m in r["prompt_messages"]] == ["system", "user"]
        if r["rejected_kind"] == "fabricated_abstention":
            assert r["chosen"] == prompts.ABSTAIN_PHRASE


def test_grpo_rows():
    grpo = rows("grpo_prompts.jsonl")
    assert len(grpo) >= 300
    assert len({r["id"] for r in grpo}) == len(grpo)
    for r in grpo:
        assert r["answerable"] == (r["kind"] != "unanswerable")
        assert (r["reference_answer"] == prompts.ABSTAIN_PHRASE) == (not r["answerable"])


def test_no_golden_question_in_training_data_exact():
    """Cheap exact/normalized check that runs in CI. The embedding check is
    scripts/check_leakage.py."""
    gold = {validate.alnum(g.question) for g in load_golden_set()}
    qs = [r["messages"][1]["content"] for r in rows("sft_train.jsonl") + rows("sft_val.jsonl")]
    qs += [r["question"] for r in rows("grpo_prompts.jsonl")]
    qs += [r["prompt_messages"][1]["content"] for r in rows("dpo.jsonl")]
    assert not [q for q in qs if validate.alnum(q) in gold]


def test_grpo_prompts_do_not_repeat_sft_questions():
    sft = {validate.alnum(r["messages"][1]["content"]) for r in rows("sft_train.jsonl")}
    sft |= {validate.alnum(r["messages"][1]["content"]) for r in rows("sft_val.jsonl")}
    assert not [r for r in rows("grpo_prompts.jsonl") if validate.alnum(r["question"]) in sft]
