"""LLM judge for correctness: the rag-with-receipts judge, replicated so scores are comparable.

Same model, system prompt, tool schema, and user-message format as upstream (see the
reference note in docs/roadmap.md, step 4). No temperature is set, so verdicts are not fully
deterministic; the cache below makes a re-score of unchanged answers free and repeatable, and
step 4's validation measures the run-to-run noise.

The API key comes from the environment (datagen.llm.make_client); nothing here prints it.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from remember_or_retrieve.datagen.llm import BudgetExceeded, estimate_tokens, usage_cost
from remember_or_retrieve.scoring.schema import CORRECT, INCORRECT, PARTIAL

TOOL_NAME = "submit_verdict"

SYSTEM_PROMPT = """You are grading a question-answering system for the Old School RuneScape
Wiki. You will be given a question, a gold reference answer, and a candidate answer
produced by the system. Judge only whether the candidate answer is factually correct
relative to the gold reference answer - do not penalize differences in phrasing, length,
or extra correct detail not present in the reference.

Verdicts:
- "correct": the candidate answer's key facts match the gold reference answer.
- "partially_correct": the candidate answer gets some key facts right but misses or
  garbles others (common on multi-hop questions requiring several facts synthesized
  together).
- "incorrect": the candidate answer's key facts contradict or are unrelated to the gold
  reference answer.

Always respond by calling the submit_verdict tool - never respond with plain text."""

VERDICT_TOOL = {
    "name": TOOL_NAME,
    "description": "Submit the correctness verdict for the candidate answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [CORRECT, PARTIAL, INCORRECT],
                "description": "The correctness verdict.",
            },
            "reasoning": {
                "type": "string",
                "description": "A short explanation for the verdict.",
            },
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + json.dumps(VERDICT_TOOL, sort_keys=True)).encode()
).hexdigest()[:12]

EST_OUT_TOKENS = 120  # observed upstream: ~100 output tokens per verdict


def build_message(question: str, gold_answer: str, candidate: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Gold reference answer: {gold_answer}\n\n"
        f"Candidate answer: {candidate}"
    )


@dataclass
class Verdict:
    verdict: str
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class Judge:
    def __init__(
        self,
        client,
        model: str,
        prices: dict,
        cache_path: str | Path,
        max_tokens: int = 1024,
        max_usd: float = 1.0,
        workers: int = 4,
    ):
        self.client = client
        self.model = model
        self.prices = {"input": prices["input"], "output": prices["output"]}
        self.cache_path = Path(cache_path)
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.workers = workers
        self.spent_usd = 0.0
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    @classmethod
    def from_config(cls, client, scoring_cfg: dict, cache_dir: Path) -> "Judge":
        j = scoring_cfg["judge"]
        return cls(
            client,
            j["model"],
            j["prices_usd_per_mtok"],
            cache_dir / "judge_cache.jsonl",
            j["max_tokens"],
            j["max_usd"],
            j["workers"],
        )

    def key(self, question: str, gold: str, candidate: str) -> str:
        raw = json.dumps([self.model, PROMPT_HASH, question, gold, candidate], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, encoding="utf-8") as f:
            return {(d := json.loads(line))["key"]: d for line in f if line.strip()}

    def estimate_usd(self, triples: list[tuple[str, str, str]]) -> float:
        """Heuristic cost of the given calls (cache is the caller's concern)."""
        tokens_in = sum(
            estimate_tokens(SYSTEM_PROMPT + json.dumps(VERDICT_TOOL) + build_message(*t))
            for t in triples
        )
        out = len(triples) * EST_OUT_TOKENS
        return (tokens_in * self.prices["input"] + out * self.prices["output"]) / 1e6

    def _call(self, question: str, gold: str, candidate: str) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_message(question, gold, candidate)}],
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        if resp.stop_reason == "refusal":
            return {
                "verdict": INCORRECT,
                "reasoning": "Judge model refused to score this answer.",
                **usage,
            }
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"judge hit max_tokens ({self.max_tokens}) before finishing")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            raise RuntimeError("judge returned no tool_use block")
        return {"verdict": block.input["verdict"], "reasoning": block.input["reasoning"], **usage}

    def score(self, question: str, gold: str, candidate: str, fresh: bool = False) -> Verdict:
        """Judge one answer. `fresh` bypasses the cache (used to measure run-to-run noise)."""
        k = self.key(question, gold, candidate)
        if not fresh and k in self._cache:
            d = self._cache[k]
            return Verdict(d["verdict"], d["reasoning"], 0, 0, cached=True)
        d = self._call(question, gold, candidate)
        prices = {self.model: self.prices}
        cost = usage_cost(self.model, d["input_tokens"], d["output_tokens"], prices, 1.0)
        with self._lock:
            self.spent_usd += cost
            if not fresh:
                self._cache[k] = {"key": k, **d}
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(self._cache[k], ensure_ascii=False) + "\n")
        return Verdict(d["verdict"], d["reasoning"], d["input_tokens"], d["output_tokens"])

    def score_many(
        self, triples: list[tuple[str, str, str]], fresh: bool = False
    ) -> list[Verdict | Exception]:
        """Judge many (question, gold, candidate) triples, in order.

        Refuses before any paid call if the estimate for the uncached calls exceeds max_usd.
        A per-item failure comes back as the exception (the caller labels it `error`) so one
        bad call does not lose the batch.
        """
        paid = triples if fresh else [t for t in triples if self.key(*t) not in self._cache]
        est = self.estimate_usd(paid)
        if est > self.max_usd:
            raise BudgetExceeded(f"judge estimate ${est:.3f} exceeds max_usd ${self.max_usd:.2f}")

        def one(t):
            try:
                return self.score(*t, fresh=fresh)
            except Exception as e:  # noqa: BLE001 - keep the rest of the batch
                return e

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, triples))
