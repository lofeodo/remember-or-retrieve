"""Deterministic sampling of chunks and chunk pairs, and construction of generation requests."""

import random
import re
from collections import defaultdict

from remember_or_retrieve.data import Chunk
from remember_or_retrieve.datagen import prompts
from remember_or_retrieve.datagen.llm import Request


def multi_hop_candidates(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Candidate chunk-id pairs: same page (different sections), and cross-page where one
    chunk's text mentions the other's page title. Order is deterministic."""
    by_page: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_page[c.page_title].append(c)
    pairs: set[tuple[str, str]] = set()
    for page_chunks in by_page.values():
        for i, a in enumerate(page_chunks):
            for b in page_chunks[i + 1 : i + 4]:  # nearby chunks only
                if a.section_path != b.section_path:
                    pairs.add((a.chunk_id, b.chunk_id))
    titles = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in by_page if len(t) >= 4}
    for a in chunks:
        for t, pat in titles.items():
            if t != a.page_title and pat.search(a.text):
                b = by_page[t][0]
                pairs.add(tuple(sorted((a.chunk_id, b.chunk_id))))
    return sorted(pairs)


def clusters(chunks: list[Chunk], size: int) -> list[list[Chunk]]:
    """Consecutive groups of `size` chunks within each page."""
    by_page: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_page[c.page_title].append(c)
    out = []
    for page_chunks in by_page.values():
        for i in range(0, len(page_chunks), size):
            out.append(page_chunks[i : i + size])
    return out


def build_generation_requests(chunks: list[Chunk], cfg: dict) -> list[Request]:
    """All stage-1 requests: SFT and GRPO, single-hop, multi-hop, unanswerable."""
    td = cfg["training_data"]
    rng = random.Random(td["seed"])
    gen = td["generator_model"]
    cheap = td["cheap_model"]
    by_id = {c.chunk_id: c for c in chunks}
    reqs: list[Request] = []

    # Single-hop, SFT: every chunk, count by source type.
    for c in chunks:
        n = td["pairs_per_chunk"][c.source_type]
        reqs.append(
            Request(
                custom_id=f"sh-sft-{len(reqs):05d}",
                model=cheap,
                system=prompts.GENERATOR_SYSTEM,
                user=prompts.single_hop_user(c, n, fresh_angle=False),
                schema=prompts.SINGLE_HOP_SCHEMA,
                est_out=130 * n,
                max_tokens=4096,
                meta={"split": "sft", "kind": "single_hop", "chunk_ids": [c.chunk_id]},
            )
        )
    # Single-hop, GRPO: a random subset of chunks, fresh angle.
    g = td["grpo_single_hop"]
    for c in rng.sample(chunks, g["chunks"]):
        reqs.append(
            Request(
                custom_id=f"sh-grpo-{len(reqs):05d}",
                model=cheap,
                system=prompts.GENERATOR_SYSTEM,
                user=prompts.single_hop_user(c, g["pairs_per_chunk"], fresh_angle=True),
                schema=prompts.SINGLE_HOP_SCHEMA,
                est_out=130 * g["pairs_per_chunk"],
                max_tokens=4096,
                meta={"split": "grpo", "kind": "single_hop", "chunk_ids": [c.chunk_id]},
            )
        )

    # Multi-hop: distinct pairs, SFT then GRPO.
    cands = multi_hop_candidates(chunks)
    rng.shuffle(cands)
    want = td["multi_hop_requests"]
    if len(cands) < want["sft"] + want["grpo"]:
        raise ValueError(f"only {len(cands)} multi-hop candidates, need {want}")
    picked = {"sft": cands[: want["sft"]], "grpo": cands[want["sft"] : want["sft"] + want["grpo"]]}
    for split, pairs in picked.items():
        for a_id, b_id in pairs:
            reqs.append(
                Request(
                    custom_id=f"mh-{split}-{len(reqs):05d}",
                    model=gen,
                    system=prompts.GENERATOR_SYSTEM,
                    user=prompts.multi_hop_user(by_id[a_id], by_id[b_id]),
                    schema=prompts.MULTI_HOP_SCHEMA,
                    effort="medium",
                    thinking="adaptive",
                    est_out=450,
                    max_tokens=4096,
                    meta={"split": split, "kind": "multi_hop", "chunk_ids": [a_id, b_id]},
                )
            )

    # Unanswerable: one request per cluster (SFT: all clusters; GRPO: a random subset).
    u = td["unanswerable"]
    all_clusters = clusters(chunks, u["cluster_size"])
    grpo_idx = set(rng.sample(range(len(all_clusters)), u["grpo_clusters"]))
    for i, cl in enumerate(all_clusters):
        for split, n in (("sft", u["sft_questions_per_cluster"]), ("grpo", 2)):
            if split == "grpo" and i not in grpo_idx:
                continue
            reqs.append(
                Request(
                    custom_id=f"un-{split}-{len(reqs):05d}",
                    model=cheap,
                    system=prompts.GENERATOR_SYSTEM,
                    user=prompts.unanswerable_user(cl, n),
                    schema=prompts.UNANSWERABLE_SCHEMA,
                    est_out=40 * n,
                    meta={
                        "split": split,
                        "kind": "unanswerable",
                        "chunk_ids": [c.chunk_id for c in cl],
                    },
                )
            )
    return reqs
