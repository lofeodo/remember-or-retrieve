# Roadmap — Remember or Retrieve

The project's paper trail: the approved step checklist, and under "Step detail" each
step's approved plan, kept current with real deviations as the step is implemented.
The locked decisions and the workflow rules this roadmap follows live in
[`CLAUDE.md`](../CLAUDE.md).

Approved 2026-10-09. This replaces the original "Candidate step breakdown" (11 steps),
which was a starting point only. Each step is one feature branch and goes through its own
plan-then-approve cycle (workflow rules 2 to 8 in `CLAUDE.md`) before any of it is implemented. "Free"
means no cloud or API spend.

**Standing instruction: end-of-work report.** Every time work on something finishes
(a task, a commit batch, a step), report two things, in short bullet points, no
long prose:

- **Done:** what was just completed.
- **Left:** everything remaining in the current step, listed explicitly. If nothing is
  left and the step is complete, say so plainly ("Nothing left. Step N is complete.").

- [x] **Step 1 — Repo scaffold** (`feat/scaffold`, free)
- [x] **Step 2 — Compute plan, GCP bootstrap, GPU quota** (`feat/compute-plan`, free)
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

**Superseded by step 2:** with live spot prices and the 8B model, the estimate is ~$23 low,
~$34 expected, ~$49 high (full table in [`compute-plan.md`](compute-plan.md)). The $20 target
is not reachable for a genuinely distributed run; the $50 cap holds in every case.

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

### Step 1 — Repo scaffold (`feat/scaffold`, free)

Plan approved 2026-10-09. Status: complete (implemented as planned, no scope deviations).

**Context.** Every later branch builds on this one, so it stays small and free: package
skeleton, tooling, CI, and the corpus and golden set brought in from `rag-with-receipts`
with provenance and checksums. No cloud, no API spend, no training dependencies.

**What exploration found (changes the wording above).**

- The golden set is committed upstream as `data/eval/qa_pairs.json` (55 items, fields
  `id, question, type, answerable, gold_chunk_ids, gold_answer`; 35 single_hop, 15
  multi_hop, 5 unanswerable with `type: null` and `answerable: false`). Pinning to a
  `rag-with-receipts` commit works for it.
- The corpus is **not** committed upstream. `data/processed/chunks.jsonl` (955 lines,
  1.2 MB) is gitignored and regenerated from the live wiki, so "pinned to a commit" is
  impossible and re-fetching could drift. It is therefore copied in from the local
  `rag-with-receipts` checkout, its SHA-256 recorded, and committed here. All 55
  `gold_chunk_ids` resolve in it.
- Upstream `HEAD` is `6b8d032`. It has no CI.
- Python 3.11 and 3.14 are both installed locally; the ML stack is not reliably available
  on 3.14, so the project targets 3.11 like `rag-with-receipts`. `uv`, `terraform`, and
  `ruff` were not installed; Terraform is left to steps 2 and 8.

**Deliverables.**

- `pyproject.toml` (src layout, `requires-python >=3.11`, runtime dep `pyyaml`, extra
  `dev` = pytest + ruff; heavy extras are added by the step that needs them).
- `src/remember_or_retrieve/`: `config.py` (`load_config`), `data.py` (`load_corpus`,
  `load_golden_set`, `verify_data` recomputing SHA-256 against the manifest).
- `config/config.yaml`; `data/corpus/chunks.jsonl`; `data/eval/golden_set.json` (renamed
  from `qa_pairs.json`); `data/PROVENANCE.json`; `data/NOTICE.md` (OSRS Wiki, CC BY-NC-SA
  3.0).
- `scripts/verify_data.py`; `tests/test_data.py`, `tests/test_config.py`.
- `.github/workflows/ci.yml` (ubuntu, Python 3.11: ruff check, ruff format --check,
  pytest); `.gitignore`; MIT `LICENSE`; README stub.

**Tests (offline, fast).** Corpus: 955 chunks, unique ids, required fields. Golden set: 55
items, unique ids, 35/15/5 split, every `gold_chunk_id` in the corpus, unanswerable items
have none. Checksums match the manifest and a tampered copy fails. Config loads and
resolves the data paths.

**Decisions made in the plan.** Python 3.11 with plain `venv` and pip; MIT license; golden
set renamed `golden_set.json`; minimal dependencies now.

**Verification.** `ruff check`, `ruff format --check`, `pytest` green, `verify_data.py`
exits 0, CI passes on GitHub. Spend stays $0; nothing paid exists, so no teardown.

**Implementation notes (small additions beyond the plan).**

- Added `.gitattributes` (`data/** -text`, LF elsewhere). Without it, git's CRLF conversion
  on Windows would change the data files' bytes and break the SHA-256 check against Linux CI.
- CI runs on pull requests and pushes to `main`, and also runs `scripts/verify_data.py`.
- Local results: ruff clean, 12 tests passing, `verify_data.py` OK.
- Recorded checksums: `chunks.jsonl` `2b245aed...`, `golden_set.json` `ec89e698...`
  (full values in `data/PROVENANCE.json`).
- Spend: $0. Nothing paid was provisioned, so there is no teardown to confirm.

### Step 2 — Compute plan, GCP bootstrap, GPU quota (`feat/compute-plan`, free)

Plan approved 2026-10-09. Status: complete (see deviations).

**Outcome and deviations from the plan.**

- Part A: `docs/compute-plan.md`, `compute:` and `budget:` config, consistency test.
  Live prices made a 14B full fine-tune too tight against the cap, so the user chose
  Qwen3-8B on 4x A100 40GB (`a2-highgpu-4g`); `CLAUDE.md`'s Base model row was amended.
  Catalog-verified spot price $8.816/hr matched the mirror.
- Part B: project `remember-or-retrieve` created under `daniel.lofeodo@gmail.com`; the one
  open billing account linked; named gcloud config `remember-or-retrieve` created with
  `--no-activate` (default config untouched). No account switch or new login was needed.
- Deviation: the first APIs were enabled with `gcloud` before Terraform, so Terraform only
  adopted them. `cloudbilling.googleapis.com` was added later (needed for catalog prices).
- Deviation: the billing account is in **CAD**, and budget creation fails with a 400 in any
  other currency. Budgets are CAD 28 (~USD 20) and CAD 70 (~USD 50) at an assumed 1.40
  USD to CAD rate (`usd_to_cad` variable). All real spend is billed in CAD.
- An unrelated account-wide CAD 20 "billing budget" already exists on the billing account;
  it was left alone.
- Quota requests filed (spot A100 40GB: 4 GPUs; spot CPUs: 48; plus 1 L4 for serving):
  us-central1 A100 and CPUs **granted in full** automatically; L4 granted;
  us-east1 and europe-west4 A100 granted 1 of 4 so far, still pending, and not needed
  unless us-central1 has no spot capacity.
- Spend: $0. Nothing billable was created, so there is no teardown to confirm.

**Context.** Everything paid depends on the model size, GPU tier and count, and
parallelism strategy, and a new GCP project starts with zero GPU quota (approval can take
days; new Gmail-account projects are sometimes denied). This step decides the compute plan
on paper, then creates the project and files the quota request early so the wait overlaps
steps 3 to 8. Cost of this step: $0 (running total $0 of the $20 target and $50 cap).

**Heads-up rule for the GCP project.** The default gcloud login is
`latentspacemail@gmail.com`, used for unrelated parallel work, and is never changed. Part B
uses a separate named gcloud configuration (`remember-or-retrieve`) passed explicitly as
`--configuration=remember-or-retrieve` on every command. `gcloud auth login` is interactive,
so the user runs it. The user is told before any GCP command runs. Part A makes no GCP
contact.

**Part A — Compute plan (no cloud contact).**

1. Verify live prices from cloud.google.com (A100 80GB, H100, L4; spot and on-demand;
   per region). Third-party figures are not trusted.
2. Pick the model against the cost table. Recommendation: a ~14B instruct (Qwen family,
   Apache-2.0; exact checkpoint verified on Hugging Face). 14B keeps full fine-tuning
   feasible via FSDP with a short window; 32B roughly doubles GPU-hours and pushes toward
   the $50 cap.
3. Parallelism for 14B: full fine-tune needs roughly 16 bytes/param (~220 GB with optimizer
   state), so FSDP full-shard across 4x A100-80GB (320 GB); LoRA under DDP; DPO and GRPO as
   LoRA-on-FSDP or DDP as measured. Orchestrated by Ray Train. A single-GPU reference run on
   the same tier for the throughput comparison.
4. Cost table (low/high, spot vs on-demand) for SFT, LoRA vs full, DPO, GRPO (prompt count x
   group size stated), the JAX stage (2 to 4B Gemma), the single-GPU reference, and
   setup/idle overhead, with running total vs $20 and $50. Includes a cut order: GKE window
   first, then GRPO size, never the distributed setup.
5. Fallback tiers if A100-80GB quota is denied: 8x A100-40GB, 8x L4 (LoRA only), H100, each
   with its changed model size and cost.
6. Outputs: `docs/compute-plan.md` (decision record and cost table), a `compute:` section in
   `config/config.yaml`, and a small test that it loads and the budget numbers are
   consistent (target <= cap). Update this roadmap's budget envelope.

**Part B — GCP bootstrap (after go-ahead).**

1. Install Terraform (pinned version).
2. The user runs `gcloud auth login` for `daniel.lofeodo@gmail.com`; the named configuration
   is created.
3. Create project `remember-or-retrieve`, link billing (the billing account ID is never
   printed or committed), enable APIs: compute, container, aiplatform, secretmanager,
   storage, billingbudgets, cloudresourcemanager, serviceusage.
4. Terraform in `infra/bootstrap/`: billing budget with alerts at $20 and $50 and API
   enablement as code; account ID and email via untracked tfvars or env; `*.tfvars` and
   state gitignored. `terraform plan` shown before `apply`.
5. Quota requests: spot A100-80GB (4, ideally 8) in 2 to 3 regions with matching vCPU quota,
   plus one serving GPU for GKE later. Request IDs and status recorded here (no secrets).
6. If Google denies on a free-trial account, upgrading to a paid account is the usual fix;
   this is flagged to the user, not done automatically.

**Security.** No credentials read or committed. Billing account ID and alert email stay out
of the repo. Only the listed APIs; no service-account keys created; no public endpoints; no
IAM grants beyond the user's own ownership.

**Paid / teardown.** Nothing billable is created (budgets and quotas are free). Spend $0,
nothing to tear down.

**Verification.** `pytest`, `ruff check`, `ruff format --check` pass and CI is green;
`terraform validate` and `terraform plan` clean; after apply, the budgets list shows the $20
and $50 alerts and the enabled APIs list shows the APIs; quota requests visible with status;
`gcloud config configurations list` shows `default` still on `latentspacemail@gmail.com`.
