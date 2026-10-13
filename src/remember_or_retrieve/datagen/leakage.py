"""Leakage check against the golden set, and near-duplicate removal inside the training set.

Only the golden QUESTIONS are excluded. The facts they cover overlap with the training data by
design (the model is meant to know the corpus); the README says so.
"""

from collections.abc import Callable

import numpy as np

from remember_or_retrieve.datagen.validate import alnum, jaccard, ngrams

Embedder = Callable[[list[str]], np.ndarray]  # returns L2-normalized rows


def bge_embedder(model_name: str = "BAAI/bge-large-en-v1.5") -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )

    return embed


def nearest_golden(
    train_qs: list[str], golden_qs: list[str], embed: Embedder
) -> tuple[np.ndarray, np.ndarray]:
    """For each training question: max cosine to any golden question, and which golden index."""
    sims = embed(train_qs) @ embed(golden_qs).T
    return sims.max(axis=1), sims.argmax(axis=1)


def find_leaks(
    train_qs: list[str],
    golden_qs: list[str],
    embed: Embedder,
    cosine: float,
    jac: float,
    sims: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[int, tuple[str, int, float]]:
    """Indices of training questions to drop -> (reason, golden index, score)."""
    best, arg = sims if sims is not None else nearest_golden(train_qs, golden_qs, embed)
    gold_norm = [alnum(g) for g in golden_qs]
    gold_grams = [ngrams(g) for g in golden_qs]
    leaks: dict[int, tuple[str, int, float]] = {}
    for i, q in enumerate(train_qs):
        qn = alnum(q)
        if qn in gold_norm:
            leaks[i] = ("exact", gold_norm.index(qn), 1.0)
            continue
        if best[i] >= cosine:
            leaks[i] = ("cosine", int(arg[i]), float(best[i]))
            continue
        qg = ngrams(q)
        for j, gg in enumerate(gold_grams):
            if (s := jaccard(qg, gg)) >= jac:
                leaks[i] = ("ngram", j, s)
                break
    return leaks


def dedup(qs: list[str], embed: Embedder, cosine: float, jac: float) -> set[int]:
    """Indices to drop so no two kept questions are near-duplicates (first occurrence wins)."""
    if not qs:
        return set()
    emb = embed(qs)
    grams = [ngrams(q) for q in qs]
    drop: set[int] = set()
    block = 512
    for start in range(0, len(qs), block):
        sims = emb[start : start + block] @ emb.T
        for r in range(sims.shape[0]):
            i = start + r
            if i in drop:
                continue
            for j in np.nonzero(sims[r, i + 1 :] >= cosine)[0] + i + 1:
                drop.add(int(j))
            # n-gram check against nearby candidates only (cosine screen keeps this cheap)
            for j in np.nonzero(sims[r, i + 1 :] >= 0.7)[0] + i + 1:
                if int(j) not in drop and jaccard(grams[i], grams[int(j)]) >= jac:
                    drop.add(int(j))
    return drop
