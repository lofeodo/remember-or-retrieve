"""Offline judge tests: a fake client stands in for the API (no key, no network)."""

from types import SimpleNamespace

import pytest

from remember_or_retrieve.datagen.llm import BudgetExceeded
from remember_or_retrieve.scoring import judge as judge_mod
from remember_or_retrieve.scoring.judge import Judge

PRICES = {"input": 2.0, "output": 10.0}


class FakeMessages:
    def __init__(self, verdict="correct", stop_reason="tool_use", with_block=True):
        self.calls = []
        self.verdict, self.stop_reason, self.with_block = verdict, stop_reason, with_block

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            [
                SimpleNamespace(
                    type="tool_use", input={"verdict": self.verdict, "reasoning": "because"}
                )
            ]
            if self.with_block
            else [SimpleNamespace(type="text", text="no tool")]
        )
        return SimpleNamespace(
            content=content,
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


def make(tmp_path, **kw):
    fake = FakeMessages(
        **{k: kw.pop(k) for k in list(kw) if k in ("verdict", "stop_reason", "with_block")}
    )
    client = SimpleNamespace(messages=fake)
    return Judge(client, "claude-sonnet-5", PRICES, tmp_path / "cache.jsonl", **kw), fake


def test_request_matches_upstream_shape(tmp_path):
    j, fake = make(tmp_path)
    v = j.score("Q?", "gold", "cand")
    assert v.verdict == "correct" and not v.cached
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"] == judge_mod.SYSTEM_PROMPT
    assert call["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
    assert call["tools"] == [judge_mod.VERDICT_TOOL]
    assert call["messages"][0]["content"] == (
        "Question: Q?\n\nGold reference answer: gold\n\nCandidate answer: cand"
    )
    assert "temperature" not in call


def test_cache_hit_skips_the_api_and_survives_restart(tmp_path):
    j, fake = make(tmp_path)
    j.score("Q?", "gold", "cand")
    assert j.score("Q?", "gold", "cand").cached
    assert len(fake.calls) == 1
    j2, fake2 = make(tmp_path)
    assert j2.score("Q?", "gold", "cand").cached and not fake2.calls


def test_fresh_bypasses_cache_and_does_not_write_it(tmp_path):
    j, fake = make(tmp_path)
    j.score("Q?", "gold", "cand")
    j.score("Q?", "gold", "cand", fresh=True)
    assert len(fake.calls) == 2
    assert len(j.cache_path.read_text().splitlines()) == 1


def test_changed_answer_is_a_cache_miss(tmp_path):
    j, fake = make(tmp_path)
    j.score("Q?", "gold", "cand")
    j.score("Q?", "gold", "different")
    assert len(fake.calls) == 2


def test_budget_guard_aborts_before_any_call(tmp_path):
    j, fake = make(tmp_path, max_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        j.score_many([("Q?", "gold", "cand")] * 3)
    assert not fake.calls


def test_cached_items_do_not_count_against_the_budget(tmp_path):
    j, fake = make(tmp_path, max_usd=0.0001)
    j.score("Q?", "gold", "cand")  # direct call is not budget-guarded
    out = j.score_many([("Q?", "gold", "cand")])
    assert out[0].cached


def test_refusal_maps_to_incorrect(tmp_path):
    j, _ = make(tmp_path, stop_reason="refusal", with_block=False)
    assert j.score("Q?", "gold", "cand").verdict == "incorrect"


def test_failures_come_back_as_exceptions_in_order(tmp_path):
    j, _ = make(tmp_path, with_block=False)
    out = j.score_many([("Q1", "g", "c"), ("Q2", "g", "c")])
    assert all(isinstance(o, RuntimeError) for o in out)


def test_max_tokens_is_an_error(tmp_path):
    j, _ = make(tmp_path, stop_reason="max_tokens")
    with pytest.raises(RuntimeError):
        j.score("Q?", "gold", "cand")


def test_spend_is_tracked(tmp_path):
    j, _ = make(tmp_path)
    j.score("Q?", "gold", "cand")
    assert j.spent_usd == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6)


def test_prompt_is_the_upstream_prompt():
    assert judge_mod.SYSTEM_PROMPT.startswith("You are grading a question-answering system")
    assert judge_mod.SYSTEM_PROMPT.endswith("never respond with plain text.")
    assert judge_mod.VERDICT_TOOL["input_schema"]["properties"]["verdict"]["enum"] == [
        "correct",
        "partially_correct",
        "incorrect",
    ]
