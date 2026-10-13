"""Offline tests for the training-data pipeline (no API calls, no model downloads)."""

import hashlib
import json

import numpy as np

from remember_or_retrieve.config import load_config
from remember_or_retrieve.data import load_corpus
from remember_or_retrieve.datagen import llm, prompts, sample, validate
from remember_or_retrieve.datagen.leakage import dedup, find_leaks


def hash_embedder(texts):
    """Deterministic bag-of-words embedding: similar word sets give high cosine."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in validate.norm_words(t):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# ---- constants and schemas -------------------------------------------------------------------
def test_constants():
    assert prompts.KINDS == ("single_hop", "multi_hop", "unanswerable")
    assert prompts.ABSTAIN_PHRASE in prompts.SFT_SYSTEM_PROMPT
    assert set(prompts.template_hashes()) >= {"generator_system", "schemas"}


# ---- validators ------------------------------------------------------------------------------
def test_evidence_matching_ignores_spacing_and_punctuation():
    chunk = "| Diamond | High alch: 1,200 coins |"
    assert validate.evidence_in("Diamond High alch: 1,200 coins", chunk)
    assert not validate.evidence_in("Ruby High alch: 800 coins", chunk)


def test_single_hop_gate():
    chunk = "Runite ore requires level 85 Mining and gives 125 experience."
    ok = {
        "question": "What Mining level is needed for runite ore?",
        "answer": "Level 85 Mining.",
        "evidence": "Runite ore requires level 85 Mining",
        "reference_facts": ["85"],
    }
    assert validate.single_hop_gate(ok, chunk) is None
    assert validate.single_hop_gate({**ok, "answer": "Level 90 Mining."}, chunk) is not None
    assert (
        validate.single_hop_gate({**ok, "evidence": "Runite ore needs level eighty"}, chunk)
        == "evidence_not_verbatim"
    )
    assert (
        validate.single_hop_gate({**ok, "question": "According to the text, what level?"}, chunk)
        == "refers_to_excerpt"
    )


def test_multi_hop_gate_and_solver():
    a, b = "Kruk has a max hit of 37.", "The Jungle Demon has a max hit of 32."
    row = {
        "feasible": True,
        "question": "By how much does Kruk's max hit exceed the Jungle Demon's?",
        "answer": "37 - 32 = 5",
        "evidence_a": "Kruk has a max hit of 37",
        "evidence_b": "Jungle Demon has a max hit of 32",
        "reference_facts": ["5"],
    }
    assert validate.multi_hop_gate(row, a, b) is None
    assert validate.multi_hop_gate({**row, "answer": "99 - 61 - 7 = 5"}, a, b) is not None
    assert validate.solver_agrees(["5"], "Kruk 37, demon 32: difference is 5")
    assert not validate.solver_agrees(["5"], "The difference is 6")


# ---- leakage and dedup -----------------------------------------------------------------------
GOLDEN = [
    "Which attack style has weapons with the largest range of possible attack ranges?",
    "Are monkey archers aggressive?",
]


def test_leak_exact_and_paraphrase_caught_unrelated_passes():
    train = [
        "Are monkey archers aggressive?",  # exact
        "are Monkey archers aggressive??",  # normalization
        "Are the monkey archers aggressive to players?",  # near paraphrase
        "What level of Mining is required for runite ore?",  # unrelated
    ]
    leaks = find_leaks(train, GOLDEN, hash_embedder, cosine=0.7, jac=0.5)
    assert set(leaks) == {0, 1, 2}
    assert 3 not in leaks


def test_dedup_keeps_first_occurrence():
    qs = [
        "What is the max hit of Kruk?",
        "What is the max hit of Kruk exactly?",
        "Runite ore level?",
    ]
    assert dedup(qs, hash_embedder, cosine=0.85, jac=0.9) == {1}


# ---- sampling --------------------------------------------------------------------------------
def test_generation_requests_are_deterministic_and_well_formed():
    cfg = load_config()
    chunks = load_corpus()
    a, b = (
        sample.build_generation_requests(chunks, cfg),
        sample.build_generation_requests(chunks, cfg),
    )
    assert [r.custom_id for r in a] == [r.custom_id for r in b]
    assert [r.user for r in a] == [r.user for r in b]
    ids = {c.chunk_id for c in chunks}
    assert len({r.custom_id for r in a}) == len(a)
    for r in a:
        assert all(i in ids for i in r.meta["chunk_ids"])
        assert len(r.custom_id) <= 64 and r.custom_id.replace("-", "").replace("_", "").isalnum()
    sh = [r for r in a if r.meta["split"] == "sft" and r.meta["kind"] == "single_hop"]
    assert len(sh) == len(chunks)
    mh = [r.meta["chunk_ids"] for r in a if r.meta["kind"] == "multi_hop"]
    assert len({tuple(sorted(p)) for p in mh}) == len(mh)  # distinct pairs
    td = cfg["training_data"]
    models = {r.meta["kind"]: r.model for r in a}
    assert models["multi_hop"] == td["generator_model"]
    assert models["single_hop"] == td["cheap_model"]


# ---- llm wrapper -----------------------------------------------------------------------------
def test_params_have_no_forced_tool_or_sampling():
    r = llm.Request("x-1", "claude-sonnet-5-5", "sys", "user", prompts.SOLVE_SCHEMA)
    p = llm.build_params(r)
    assert "tool_choice" not in p and "temperature" not in p
    assert p["thinking"] == {"type": "between_tools"}
    assert p["output_config"]["format"]["type"] == "json_schema"
    h = llm.build_params(llm.Request("x-2", "claude-haiku-5-5", "s", "u", {}))
    assert h["thinking"] == {"type": "disabled"}


def test_estimate_and_budget_guard(tmp_path):
    prices = {"claude-sonnet-5-5": {"input": 2.0, "output": 10.0}}
    reqs = [
        llm.Request(f"r-{i}", "claude-sonnet-5-5", "s" * 3600, "u", {}, est_out=1000)
        for i in range(10)
    ]
    est = llm.estimate_cost(reqs, prices, 0.5)
    assert 0.055 < est["usd"] < 0.07  # about 10 * (1000 in * 2 + 1000 out * 10) / 1e6 * 0.5
    stage = llm.Stage(tmp_path, "t")
    try:
        llm.submit(object(), stage, reqs, 0.01, prices, 0.5)
    except llm.BudgetExceeded:
        pass
    else:
        raise AssertionError("budget guard did not trigger")
    assert not stage.batch_path.exists()


def test_make_client_requires_key_and_never_prints_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    try:
        llm.make_client()
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_stage_roundtrip(tmp_path):
    stage = llm.Stage(tmp_path, "s")
    reqs = [llm.Request("a-1", "m", "s", "u", {"type": "object"}, meta={"k": 1})]
    stage.save_requests(reqs)
    assert stage.load_requests() == reqs
    stage.results_path.write_text(json.dumps({"custom_id": "a-1", "data": {}}) + "\n")
    assert set(stage.load_results()) == {"a-1"}
