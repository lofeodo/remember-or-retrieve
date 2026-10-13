"""Grounding checker: the rag-with-receipts NLI + overlap logic, applied to any system's answer.

`check_pairs` is the upstream primitive: per (chunk, claim) pair, softmax NLI probabilities in a
fixed (entailment, contradiction, neutral) column order resolved from the model's own id2label,
then
  contradiction_prob >= 0.5                     -> contradicted
  entailment_prob >= 0.5 or overlap >= 0.3      -> grounded
  otherwise                                     -> ungrounded

`check_answer` is the head-to-head definition (decided in step 4): split the answer into
sentence claims and test each against the question's gold chunks, so a closed-book model that
cites nothing is judged on the same evidence as the RAG pipeline. A claim is grounded if any
gold chunk grounds it, else contradicted if any contradicts it, else ungrounded.

Known limits, reported rather than tuned: the NLI model reads flattened wiki tables poorly, and
a multi-hop claim that needs two chunks at once can read as ungrounded against either alone.
"""

import re
from typing import Protocol

import numpy as np

from remember_or_retrieve.data import Chunk
from remember_or_retrieve.scoring.schema import (
    CONTRADICTED,
    GROUNDED,
    UNGROUNDED,
    ClaimVerdict,
)

_TOKEN = re.compile(r"\w+")
# A sentence ends at . ! ? followed by space and an uppercase letter, digit, or opening mark.
# "1,200." and "2.5 hours" stay whole because the split needs whitespace after the stop.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class NliScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) pairs -> (n, 3) probs: entailment, contradiction, neutral."""
        ...


def breadcrumb(page_title: str, section_path: str) -> str:
    """Same context string rag-with-receipts prepends to a chunk (ingestion/utils.breadcrumb)."""
    return f"{page_title} > {section_path}" if section_path else page_title


def premise(chunk: Chunk) -> str:
    return breadcrumb(chunk.page_title, chunk.section_path) + "\n\n" + chunk.text


def token_overlap(claim: str, chunk_text: str) -> float:
    """|claim tokens in chunk| / |claim tokens|; 0.0 for an empty claim (as upstream)."""
    claim_tokens = set(_TOKEN.findall(claim.lower()))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & set(_TOKEN.findall(chunk_text.lower()))) / len(claim_tokens)


def split_claims(answer: str) -> list[str]:
    """Sentence-level claims. Newlines are boundaries; decimals and thousands commas survive."""
    claims = []
    for line in answer.splitlines():
        claims.extend(s.strip() for s in _SENTENCE_END.split(line.strip()))
    return [c for c in claims if _TOKEN.search(c)]


class CrossEncoderScorer:
    """sentence-transformers CrossEncoder with the label order resolved from the model itself."""

    def __init__(self, model_name: str, revision: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # heavy: imported lazily

        # No trust_remote_code: the checkpoint's own config and weights only.
        kwargs = {"revision": revision} if revision else {}
        self.model = CrossEncoder(model_name, **kwargs)
        self.batch_size = batch_size
        id2label = {int(i): name.lower() for i, name in self.model.model.config.id2label.items()}
        wanted = {"entailment", "contradiction", "neutral"}
        if set(id2label.values()) != wanted:
            raise ValueError(f"unexpected NLI labels {id2label}; expected {sorted(wanted)}")
        by_name = {name: i for i, name in id2label.items()}
        self.columns = [by_name["entailment"], by_name["contradiction"], by_name["neutral"]]
        self.id2label = id2label

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, 3))
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        return np.asarray(raw)[:, self.columns]


def label_for(entailment: float, contradiction: float, overlap: float, thresholds: dict) -> str:
    if contradiction >= thresholds["contradiction_threshold"]:
        return CONTRADICTED
    if (
        entailment >= thresholds["entailment_threshold"]
        or overlap >= thresholds["overlap_threshold"]
    ):
        return GROUNDED
    return UNGROUNDED


class GroundingChecker:
    def __init__(self, scorer: NliScorer, thresholds: dict):
        self.scorer = scorer
        self.thresholds = thresholds

    @classmethod
    def from_config(cls, grounding_cfg: dict) -> "GroundingChecker":
        scorer = CrossEncoderScorer(
            grounding_cfg["nli_model"],
            grounding_cfg.get("nli_revision"),
            grounding_cfg["batch_size"],
        )
        return cls(scorer, grounding_cfg)

    def check_pairs(self, pairs: list[tuple[Chunk, str]]) -> list[ClaimVerdict]:
        """Upstream-compatible: one verdict per (chunk, claim)."""
        if not pairs:
            return []
        scores = self.scorer.score([(premise(c), claim) for c, claim in pairs])
        out = []
        for (chunk, claim), row in zip(pairs, scores, strict=True):
            ent, con = float(row[0]), float(row[1])
            overlap = token_overlap(claim, chunk.text)
            out.append(
                ClaimVerdict(
                    claim=claim,
                    label=label_for(ent, con, overlap, self.thresholds),
                    chunk_id=chunk.chunk_id,
                    entailment_prob=ent,
                    contradiction_prob=con,
                    overlap_score=overlap,
                )
            )
        return out

    def check_answer(self, answer: str, gold_chunks: list[Chunk]) -> list[ClaimVerdict]:
        """Head-to-head grounding: every sentence claim against the gold chunks.

        Returns [] when there is nothing to check (empty answer or no gold chunks), which the
        metrics treat as not applicable.
        """
        claims = split_claims(answer)
        if not claims or not gold_chunks:
            return []
        pairs = [(chunk, claim) for claim in claims for chunk in gold_chunks]
        verdicts = self.check_pairs(pairs)
        n = len(gold_chunks)
        return [self._decide(verdicts[i * n : (i + 1) * n]) for i in range(len(claims))]

    @staticmethod
    def _decide(per_chunk: list[ClaimVerdict]) -> ClaimVerdict:
        grounded = [v for v in per_chunk if v.label == GROUNDED]
        if grounded:
            return max(grounded, key=lambda v: v.entailment_prob)
        contradicted = [v for v in per_chunk if v.label == CONTRADICTED]
        if contradicted:
            return max(contradicted, key=lambda v: v.contradiction_prob)
        return max(per_chunk, key=lambda v: v.entailment_prob)
