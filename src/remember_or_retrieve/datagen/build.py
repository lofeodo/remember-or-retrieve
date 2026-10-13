"""Orchestration of the training-data pipeline. State lives in the untracked raw_dir so every
stage resumes instead of re-paying. Stages: gen -> (local gates) -> verify -> (assemble) ->
dpo -> freeze."""

import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.data import Chunk, load_corpus, load_golden_set, sha256_of
from remember_or_retrieve.datagen import llm, prompts, sample, validate
from remember_or_retrieve.datagen.leakage import bge_embedder, dedup, find_leaks, nearest_golden

ABBR = {"single_hop": "sh", "multi_hop": "mh", "unanswerable": "un"}


class Pipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.td = self.cfg["training_data"]
        self.raw = resolve_path(self.td["raw_dir"])
        self.out = resolve_path(self.td["out_dir"])
        self.chunks = load_corpus()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.prices = self.td["prices_usd_per_mtok"]
        self.discount = self.td["batch_discount"]
        self._embed = None

    # ---- helpers -------------------------------------------------------------------------
    def embed(self, texts):
        if self._embed is None:
            self._embed = bge_embedder(self.td["leakage"]["embedding_model"])
        return self._embed(texts)

    def stage(self, name: str) -> llm.Stage:
        return llm.Stage(self.raw, name)

    def _jsonl_write(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _jsonl_read(self, path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _stats_path(self) -> Path:
        return self.raw / "stats.json"

    def _stats(self) -> dict:
        p = self._stats_path()
        return json.loads(p.read_text()) if p.exists() else {"spend_usd": {}, "drops": {}}

    def _save_stats(self, s: dict) -> None:
        self._stats_path().parent.mkdir(parents=True, exist_ok=True)
        self._stats_path().write_text(json.dumps(s, indent=1, sort_keys=True))

    # ---- stage 1: generation ---------------------------------------------------------------
    def gen_requests(self) -> list[llm.Request]:
        return sample.build_generation_requests(self.chunks, self.cfg)

    def estimate(self, requests: list[llm.Request]) -> dict:
        return llm.estimate_cost(requests, self.prices, self.discount)

    def submit(self, name: str, requests: list[llm.Request]) -> str:
        client = llm.make_client()
        return llm.submit(
            client, self.stage(name), requests, self.td["max_usd"], self.prices, self.discount
        )

    def collect(self, name: str, wait: bool = True) -> dict:
        client = llm.make_client()
        res = llm.collect(client, self.stage(name), self.prices, self.discount, wait=wait)
        if res.get("status") == "ended":
            s = self._stats()
            s["spend_usd"][name] = res["usd"]
            self._save_stats(s)
        return res

    # ---- local gates over stage-1 results ----------------------------------------------------
    def gate_generated(self) -> list[dict]:
        """Turn raw generation results into candidate rows, applying the programmatic gates.
        Writes candidates.jsonl and returns the rows."""
        results = self.stage("gen").load_results()
        drops: Counter = Counter()
        cands: list[dict] = []
        for _, r in sorted(results.items()):
            m = r["meta"]
            kind, split = m["kind"], m["split"]
            if r["data"] is None:
                drops[f"{kind}:unparseable_or_error"] += 1
                continue
            ids = m["chunk_ids"]
            if kind == "single_hop":
                text = self.by_id[ids[0]].text
                for p in r["data"]["pairs"]:
                    if reason := validate.single_hop_gate(p, text):
                        drops[f"{kind}:{reason}"] += 1
                        continue
                    cands.append(
                        {
                            "split": split,
                            "kind": kind,
                            "question": p["question"].strip(),
                            "answer": p["answer"].strip(),
                            "evidence": [p["evidence"]],
                            "reference_facts": p["reference_facts"],
                            "chunk_ids": ids,
                        }
                    )
            elif kind == "multi_hop":
                d = r["data"]
                if reason := validate.multi_hop_gate(
                    d, self.by_id[ids[0]].text, self.by_id[ids[1]].text
                ):
                    drops[f"{kind}:{reason}"] += 1
                    continue
                cands.append(
                    {
                        "split": split,
                        "kind": kind,
                        "question": d["question"].strip(),
                        "answer": d["answer"].strip(),
                        "evidence": [d["evidence_a"], d["evidence_b"]],
                        "reference_facts": d["reference_facts"],
                        "chunk_ids": ids,
                    }
                )
            else:
                for q in r["data"]["questions"]:
                    if not (10 <= len(q) <= 400) or validate._BANNED.search(q):
                        drops[f"{kind}:question_shape"] += 1
                        continue
                    cands.append(
                        {
                            "split": split,
                            "kind": kind,
                            "question": q.strip(),
                            "answer": prompts.ABSTAIN_PHRASE,
                            "evidence": [],
                            "reference_facts": [],
                            "chunk_ids": ids,
                        }
                    )
        for i, c in enumerate(cands):
            c["cand_id"] = f"c{i:06d}"
        self._jsonl_write(self.raw / "candidates.jsonl", cands)
        s = self._stats()
        s["drops"]["local_gates"] = dict(drops)
        s["candidates"] = dict(Counter(f"{c['split']}:{c['kind']}" for c in cands))
        self._save_stats(s)
        return cands

    # ---- stage 2: verification ---------------------------------------------------------------
    def _chunk_embeddings(self) -> np.ndarray:
        path = self.raw / "chunk_emb.npy"
        if path.exists():
            return np.load(path)
        emb = self.embed([c.text for c in self.chunks])
        np.save(path, emb)
        return emb

    def verify_requests(self) -> list[llm.Request]:
        cands = self._jsonl_read(self.raw / "candidates.jsonl")
        ver = self.td["cheap_model"]
        un = [c for c in cands if c["kind"] == "unanswerable"]
        top5: dict[str, list[Chunk]] = {}
        if un:
            q_emb = self.embed([c["question"] for c in un])
            sims = q_emb @ self._chunk_embeddings().T
            for c, row in zip(un, sims, strict=True):
                top5[c["cand_id"]] = [self.chunks[i] for i in np.argsort(-row)[:5]]
        reqs = []
        for c in cands:
            if c["kind"] == "multi_hop":
                reqs.append(
                    llm.Request(
                        custom_id=f"vs-{c['cand_id']}",
                        model=ver,
                        system=prompts.SOLVER_SYSTEM,
                        user=prompts.solve_user(
                            c["question"], [self.by_id[i] for i in c["chunk_ids"]]
                        ),
                        schema=prompts.SOLVE_SCHEMA,
                        effort="medium",
                        thinking="adaptive",
                        est_out=250,
                        max_tokens=4096,
                        meta={"cand_id": c["cand_id"], "check": "solve"},
                    )
                )
            elif c["kind"] == "unanswerable":
                reqs.append(
                    llm.Request(
                        custom_id=f"va-{c['cand_id']}",
                        model=ver,
                        system=prompts.SOLVER_SYSTEM,
                        user=prompts.answerable_user(c["question"], top5[c["cand_id"]]),
                        schema=prompts.ANSWERABLE_SCHEMA,
                        est_out=15,
                        meta={"cand_id": c["cand_id"], "check": "answerable"},
                    )
                )
        return reqs

    # ---- assemble: verification, dedup, leakage, trim, split ----------------------------------
    def assemble(self) -> dict:
        td, lk = self.td, self.td["leakage"]
        cands = self._jsonl_read(self.raw / "candidates.jsonl")
        results = self.stage("verify").load_results()
        by_cand = {r["meta"]["cand_id"]: r for r in results.values()}
        s = self._stats()
        drops: Counter = Counter()
        kept = []
        for c in cands:
            if c["kind"] == "single_hop":
                kept.append(c)
                continue
            r = by_cand.get(c["cand_id"])
            if r is None or r["data"] is None:
                drops[f"{c['kind']}:verify_missing"] += 1
            elif c["kind"] == "multi_hop":
                if validate.solver_agrees(c["reference_facts"], r["data"]["answer"]):
                    kept.append(c)
                else:
                    drops["multi_hop:solver_disagrees"] += 1
            elif r["data"]["answerable"]:
                drops["unanswerable:actually_answerable"] += 1
            else:
                kept.append(c)

        golden = [g.question for g in load_golden_set()]
        qs = [c["question"] for c in kept]
        leaks = find_leaks(
            qs,
            golden,
            self.embed,
            lk["cosine"],
            lk["jaccard"],
            nearest_golden(qs, golden, self.embed),
        )
        drops["leakage:golden"] = len(leaks)
        kept = [c for i, c in enumerate(kept) if i not in leaks]

        # Near-duplicates: SFT first (it is the bigger pool), then GRPO against everything kept.
        kept.sort(key=lambda c: (c["split"] != "sft", c["cand_id"]))
        dup = dedup(
            [c["question"] for c in kept], self.embed, lk["dedup_cosine"], lk["dedup_jaccard"]
        )
        drops["dedup"] = len(dup)
        kept = [c for i, c in enumerate(kept) if i not in dup]

        rng = random.Random(td["seed"])
        final = {"sft": [], "grpo": []}
        targets = {"sft": td["sft_targets"], "grpo": td["grpo_targets"]}
        short = {}
        for split in ("sft", "grpo"):
            for kind in prompts.KINDS:
                pool = [c for c in kept if c["split"] == split and c["kind"] == kind]
                rng.shuffle(pool)
                want = targets[split][kind]
                if len(pool) < want:
                    short[f"{split}:{kind}"] = f"{len(pool)} < target {want}"
                final[split] += pool[:want]
        s["drops"]["assemble"] = dict(drops)
        s["shortfalls"] = short
        self._save_stats(s)
        for split in final:
            for n, c in enumerate(final[split]):
                c["id"] = f"{split}-{ABBR[c['kind']]}-{n:05d}"
        self._jsonl_write(self.raw / "assembled_sft.jsonl", final["sft"])
        self._jsonl_write(self.raw / "assembled_grpo.jsonl", final["grpo"])
        return {
            "sft": dict(Counter(c["kind"] for c in final["sft"])),
            "grpo": dict(Counter(c["kind"] for c in final["grpo"])),
            "drops": dict(drops),
            "shortfalls": short,
        }

    # ---- stage 3: DPO rejected answers -------------------------------------------------------
    def dpo_requests(self) -> list[llm.Request]:
        rng = random.Random(self.td["seed"] + 1)
        sft = self._jsonl_read(self.raw / "assembled_sft.jsonl")
        t = self.td["dpo_targets"]
        answerable = [c for c in sft if c["kind"] != "unanswerable"]
        unans = [c for c in sft if c["kind"] == "unanswerable"]
        rng.shuffle(answerable)
        rng.shuffle(unans)
        # Over-ask by 20% to leave room for the post-checks below.
        pick = {
            "hallucinated": answerable[: int(t["hallucinated"] * 1.2)],
            "verbose": answerable[
                int(t["hallucinated"] * 1.2) : int(t["hallucinated"] * 1.2)
                + int(t["verbose"] * 1.2)
            ],
            "fabricated_abstention": unans[: int(t["fabricated_abstention"] * 1.2)],
        }
        reqs = []
        for kind, rows in pick.items():
            for c in rows:
                reqs.append(
                    llm.Request(
                        custom_id=f"dp-{kind[:4]}-{len(reqs):05d}",
                        model=self.td["cheap_model"],
                        system=prompts.REJECTED_SYSTEM,
                        user=prompts.rejected_user(
                            kind,
                            c["question"],
                            c["answer"],
                            [self.by_id[i] for i in c["chunk_ids"]],
                        ),
                        schema=prompts.REJECTED_SCHEMA,
                        est_out=150,
                        max_tokens=2048,
                        meta={"sft_id": c["id"], "rejected_kind": kind},
                    )
                )
        return reqs

    def assemble_dpo(self) -> dict:
        sft = {c["id"]: c for c in self._jsonl_read(self.raw / "assembled_sft.jsonl")}
        results = self.stage("dpo").load_results()
        drops: Counter = Counter()
        rows = {k: [] for k in self.td["dpo_targets"]}
        for _, r in sorted(results.items()):
            m = r["meta"]
            kind, c = m["rejected_kind"], sft[m["sft_id"]]
            rej = (r["data"] or {}).get("rejected", "").strip()
            if not rej:
                drops[f"{kind}:empty"] += 1
            elif rej == c["answer"]:
                drops[f"{kind}:same_as_chosen"] += 1
            elif kind == "hallucinated" and validate.solver_agrees(c["reference_facts"], rej):
                drops[f"{kind}:still_correct"] += 1
            elif kind == "verbose" and not (
                len(rej) >= 2 * len(c["answer"])
                and validate.solver_agrees(c["reference_facts"], rej)
            ):
                drops[f"{kind}:not_verbose_or_lost_facts"] += 1
            elif kind == "fabricated_abstention" and any(
                p in rej.lower()
                for p in ("don't have", "do not have", "not sure", "cannot", "can't", "unknown")
            ):
                drops[f"{kind}:abstains"] += 1
            else:
                rows[kind].append(
                    {
                        "prompt_messages": [
                            {"role": "system", "content": prompts.SFT_SYSTEM_PROMPT},
                            {"role": "user", "content": c["question"]},
                        ],
                        "chosen": c["answer"],
                        "rejected": rej,
                        "rejected_kind": kind,
                        "chunk_ids": c["chunk_ids"],
                    }
                )
        out = []
        for kind, want in self.td["dpo_targets"].items():
            out += rows[kind][:want]
        for n, row in enumerate(out):
            row["id"] = f"dpo-{n:05d}"
        self._jsonl_write(self.raw / "assembled_dpo.jsonl", out)
        s = self._stats()
        s["drops"]["dpo"] = dict(drops)
        self._save_stats(s)
        return {"rows": dict(Counter(r["rejected_kind"] for r in out)), "drops": dict(drops)}

    # ---- freeze -----------------------------------------------------------------------------
    def freeze(self) -> dict:
        """Write the committed files and MANIFEST.json from the assembled raw files."""
        rng = random.Random(self.td["seed"] + 2)
        sft = self._jsonl_read(self.raw / "assembled_sft.jsonl")
        grpo = self._jsonl_read(self.raw / "assembled_grpo.jsonl")
        dpo = self._jsonl_read(self.raw / "assembled_dpo.jsonl")
        sft_rows = [
            {
                "id": c["id"],
                "kind": c["kind"],
                "messages": [
                    {"role": "system", "content": prompts.SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": c["question"]},
                    {"role": "assistant", "content": c["answer"]},
                ],
                "chunk_ids": c["chunk_ids"],
                "evidence": c["evidence"],
                "reference_facts": c["reference_facts"],
            }
            for c in sft
        ]
        rng.shuffle(sft_rows)
        n_val = round(len(sft_rows) * self.td["sft_targets"]["val_fraction"])
        val, train = sft_rows[:n_val], sft_rows[n_val:]
        grpo_rows = [
            {
                "id": c["id"],
                "kind": c["kind"],
                "question": c["question"],
                "reference_answer": c["answer"],
                "reference_facts": c["reference_facts"],
                "answerable": c["kind"] != "unanswerable",
                "chunk_ids": c["chunk_ids"],
            }
            for c in grpo
        ]
        files = {
            "sft_train.jsonl": train,
            "sft_val.jsonl": val,
            "dpo.jsonl": dpo,
            "grpo_prompts.jsonl": grpo_rows,
        }
        self.out.mkdir(parents=True, exist_ok=True)
        manifest_files = {}
        for name, rows in files.items():
            self._jsonl_write(self.out / name, rows)
            manifest_files[name] = {
                "sha256": sha256_of(self.out / name),
                "rows": len(rows),
                "by_kind": dict(Counter(r.get("kind") or r.get("rejected_kind") for r in rows)),
            }
        s = self._stats()
        manifest = {
            "created": date.today().isoformat(),
            "generator_model": self.td["generator_model"],
            "cheap_model": self.td["cheap_model"],
            "template_hashes": prompts.template_hashes(),
            "corpus_sha256": sha256_of(resolve_path(self.cfg["data"]["corpus_path"])),
            "golden_sha256": sha256_of(resolve_path(self.cfg["data"]["golden_set_path"])),
            "leakage": self.td["leakage"],
            "spend_usd": s["spend_usd"],
            "spend_total_usd": round(sum(s["spend_usd"].values()), 4),
            "drops": s["drops"],
            "shortfalls": s.get("shortfalls", {}),
            "files": manifest_files,
        }
        (self.out / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return manifest
