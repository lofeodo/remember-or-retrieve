# Roadmap — Remember or Retrieve

The project's paper trail: the approved step checklist, and under "Step detail" each
step's approved plan, kept current with real deviations as the step is implemented.
The locked decisions and the workflow rules this roadmap follows live in
[`CLAUDE.md`](../CLAUDE.md).

Approved 2026-10-09. This replaces the original "Candidate step breakdown" (11 steps),
which was a starting point only. Each step is one feature branch and goes through its own
plan-then-approve cycle (workflow rules 2 to 8 in `CLAUDE.md`) before any of it is implemented. "Free"
means no cloud or API spend.

- [ ] **Step 1 — Repo scaffold** (`feat/scaffold`, free)
- [ ] **Step 2 — Compute plan, GCP bootstrap, GPU quota** (`feat/compute-plan`, free)
- [ ] **Step 3 — Training data generation** (`feat/training-data`, ~$2 to $4 Claude API)
- [ ] **Step 4 — Scoring harness and reward functions** (`feat/scoring-harness`, under $1 judge calls)
- [ ] **Step 5 — Distributed training pipeline: SFT, LoRA vs full, DPO** (`feat/training-pipeline`, free)
- [ ] **Step 6 — GRPO stage** (`feat/grpo`, free)
- [ ] **Step 7 — JAX training stage** (`feat/jax-stage`, free)
- [ ] **Step 8 — Training infrastructure and model registry** (`feat/training-infra`, under $1)
- [ ] **Step 9 — The training run** (`feat/training-run`, ~$11 to $28, the main cost)
- [ ] **Step 10 — Comparison harness and API baselines** (`feat/comparison-harness`, ~$1 to $2)
- [ ] **Step 11 — Serving infrastructure** (`feat/serving-infra`, free)
- [ ] **Step 12 — Serving on GKE** (`feat/gke-serving`, ~$2 to $5)
- [ ] **Step 13 — User-facing demo** (`feat/demo-ui`, ~$0)
- [ ] **Step 14 — README** (`feat/readme`, free)

## What each step is, and why it is its own step

1. **Repo scaffold.** Package skeleton, `pyproject.toml`, config, lint and test setup, CI,
   license, README stub, and the corpus and golden set brought in pinned to a
   `rag-with-receipts` commit with checksums. Its own step because every later branch
   builds on it.
2. **Compute plan, GCP bootstrap, GPU quota.** Picks the base model, GPU tier and count,
   parallelization strategy (DDP, FSDP, or both), and expected wall-clock, with a cost
   table for approval. Creates the project, enables APIs, sets billing alerts at $20 and
   $50 through Terraform, and files the GPU quota request. Its own step because everything
   paid depends on it and the quota wait should overlap the free steps.
3. **Training data generation.** Claude-generated Q/A pairs from the 955 chunks
   (single-hop, multi-hop, and out-of-corpus questions so the model learns to abstain),
   the DPO preference pairs targeting hallucination and verbosity, and the GRPO prompt
   set. Includes an automated leakage check against the 55 golden questions. Its own step
   because it is the only build-time LLM dependency and its output is frozen before
   training.
4. **Scoring harness and reward functions.** One way to score any system on the golden
   set: the judge and grounding checker reused from `rag-with-receipts`, plus a fast local
   reward for GRPO. Validated by re-scoring the RAG project's recorded answers and
   checking the result against its published numbers. Its own step because training,
   registry, and comparison all depend on identical scoring.
5. **Distributed training pipeline.** Ray Train over PyTorch DDP/FSDP for SFT, the
   LoRA-vs-full comparison, and DPO, with per-stage profiling (tokens/sec, memory per
   device) and the distribution checks (ranks started, every GPU busy, one coherent loss
   curve). Code only, smoke-tested on a tiny model locally. No real checkpoint is produced
   here.
6. **GRPO stage.** The RL stage on top of DPO: group rollouts, reward from step 4,
   rollout-throughput profiling, prompt count and group size costed. Code only,
   smoke-tested. Separate from step 5 because rollouts are a different bottleneck and a
   different failure mode.
7. **JAX training stage.** A JAX LoRA fine-tune of a small model on the same training
   data (see the Framework coverage row in `CLAUDE.md`'s Locked decisions). Code only, smoke-tested.
   Separate because it is a second framework with its own dependencies.
8. **Training infrastructure and model registry.** Terraform for the GPU machines, the
   Ray cluster, networking, and the checkpoint bucket, plus the Vertex AI Model Registry
   logging code. Validated with `terraform plan` and a dummy registry entry. A rehearsal
   of a few minutes on the cheapest GPUs, to prove the cluster and multi-rank launch work
   before the real window, is to be proposed in this step's plan.
9. **The training run.** The one paid window: provision, verify it is really distributed,
   take the single-GPU reference number, run SFT, LoRA vs full, DPO, GRPO, and the JAX
   stage, score and register every checkpoint, promote the winner, `terraform destroy`,
   confirm teardown here.
10. **Comparison harness and API baselines.** The golden set through the live
    `rag-with-receipts` API, a Mistral model, and a Cohere model, all scored by step 4,
    plus the client for the fine-tuned endpoint tested against a stub.
11. **Serving infrastructure.** Terraform for the GKE cluster, the autoscaling GPU node
    pool, and the vLLM deployment manifests. `terraform plan` only.
12. **Serving on GKE.** Provision, deploy the winner, run the fine-tuned leg of the
    harness through the live endpoint, benchmark latency and throughput, exercise
    autoscaling, record answers for the demo, destroy, confirm teardown here.
13. **User-facing demo.** Side-by-side page on Cloud Run (scales to zero). The fine-tuned
    side uses recorded answers unless a live endpoint fits the budget. That tradeoff is
    raised in this step's plan.
14. **README.** To the README bar in `CLAUDE.md`, including the Slurm gap and PPO as the road not
    taken.

## Changes from the candidate breakdown

| Change | Reason |
|---|---|
| New step 2: compute plan, GCP bootstrap, GPU quota | A new GCP project starts with zero GPU quota and approval can take days. The request needs the GPU type, which needs the model size. So the model and GPU decision moves ahead of training code instead of being made when training is planned. |
| New step 4: scoring harness | Every checkpoint must be scored on the golden set as it is produced, and GRPO's reward reuses the same scorers. Both need the harness to exist before the GPU window. |
| Training split into code (5, 6, 7) and one paid run (9) | All stages share one GPU window. Every stage has to be written and smoke-tested on a tiny model before the meter starts. |
| Registry merged into training infra (8) | "Logged as it is produced" means the registry must exist before the run. |
| Comparison harness (10) moved ahead of GKE serving (12) | The harness must be ready to fire the moment the GPU node pool is up. The three API baselines get scored for real before any GKE cost. |
| New step 7: JAX stage | JAX coverage was decided in favor on 2026-10-09 (see Locked decisions in `CLAUDE.md`). |

## Budget envelope (provisional)

Prices are from memory and are re-checked against live GCP pricing in step 2, which turns
this into a real estimate for approval before anything is provisioned. Claude, Mistral,
and Cohere API spend counts against the budget.

| Item | Low | High |
|---|---|---|
| Claude API: data generation and judging (steps 3, 4, 9, 10) | $3 | $7 |
| Training window, PyTorch stages (step 9) | $10 | $20 |
| JAX stage, same window (step 9) | $1 | $8 |
| Infra rehearsal (step 8) | $0 | $1 |
| Mistral and Cohere APIs (step 10) | under $1 | $1 |
| GKE serving window (step 12) | $2 | $5 |
| **Total** | **~$16** | **~$42** |

The $20 target is only reachable with spot GPUs and a model at the small end of the 13B
to 34B range. A realistic landing zone is $25 to $35, inside the $50 cap.

**Spend to date: $0.** Nothing has been provisioned.

## Known risks

- **GPU quota** on a new project is the main schedule risk, and a denial would force a
  different GPU tier. This is why step 2 comes so early.
- **"Full" fine-tune of a 13B+ model** needs roughly 100 GB or more of GPU memory. Step 2
  decides whether that means FSDP across large GPUs or a defined partial unfreeze, and
  says so openly.
- **Grounding for a closed-book model.** The RAG grounding checker tests a claim against a
  cited chunk. The fine-tuned model cites nothing, so step 4 has to define what its
  grounding score is measured against.
- **The live RAG API** is gated by a demo key and limited to 10 requests per 5 minutes per
  IP. Step 10 needs the key and a paced run or a temporary limit change.
- **Leakage.** Training questions cover the same facts as the golden set by design. Only
  the golden questions themselves are excluded. The README will say this plainly.
- **Local tooling.** Terraform is not installed on the development machine, and its Python
  is 3.14 (`rag-with-receipts` used 3.11). Steps 1 and 2 settle both.

## Step detail

Each step's approved plan is copied here in full when it is approved, then kept current
with real deviations as the step is implemented.

*No step has been planned yet.*
