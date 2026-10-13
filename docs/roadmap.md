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
- [x] **Step 3 — Training data generation** (`feat/training-data`, $4.98 Claude API)
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

**Spend to date: $4.98** (step 3, Claude API only; ledger in the step 3 outcome). Nothing has been provisioned.

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

### Step 3 — Training data generation (`feat/training-data`, ~$2 to $4 Claude API)

Plan approved 2026-10-13. Status: complete (see outcome and deviations).

###### Context

Steps 5 to 9 (SFT, LoRA vs full, DPO, GRPO, JAX) all train on data that does not exist yet. This step builds
it: Claude-generated Q/A pairs from the 955 corpus chunks, the DPO preference pairs, and the GRPO prompt
set, then freezes them behind an automated leakage check against the 55 golden questions. Claude is a
build-time dependency only (never at inference). Step 4 (scoring) and step 5 (training) consume the file
formats defined here, so the schemas are part of the deliverable.

Facts from exploration that shape the design:
- `rag-with-receipts` has **no** bulk-generation, batch-API, retry, dedup, or leakage code. It calls
  `anthropic.Anthropic()` directly with forced tool output (`submit_answer`), no `temperature` (Sonnet 5
  family rejects sampling params), and `ANTHROPIC_API_KEY` from the environment. Everything here is new
  code; I reuse its conventions (forced tool schemas, env-var key, config-driven model id) and its embedder
  choice (`BAAI/bge-large-en-v1.5`, normalized, so dot product = cosine).
- Golden set style (hand-written, terse): 35 single-hop lookups, 15 multi-hop with arithmetic shown
  ("11,000 x 85 XP = 935,000"), 5 out-of-corpus questions. Training data mimics that style and mix, never the
  questions themselves.
- Corpus: 955 chunks, 110 pages, 512 table / 376 prose / 67 infobox, ~256k tokens, ~268 tokens/chunk.
- The RAG side answers as `{answerable, answer, citations}`. The fine-tuned model is closed-book and cites
  nothing, so its target is plain text; abstention is a fixed phrase (below). How step 4 scores grounding for
  a closed-book answer is that step's problem; this step supplies the `evidence` and `reference_facts`
  fields it will need.

Decisions confirmed with the user: **Sonnet 5.5 via the Batch API**, **~6,000 SFT pairs**,
**embedding + n-gram leakage filter that drops near-duplicate questions** (facts may overlap with the golden
set by design; only the questions are excluded, and the README says so).

#### Outputs (frozen, committed)

All under `data/training/`, JSONL, UTF-8, LF (covered by the `data/** -text` rule in `.gitattributes`):

| File | Rows | Schema |
|---|---|---|
| `sft_train.jsonl`, `sft_val.jsonl` | ~5,700 / ~300 | `id, kind, messages[system,user,assistant], chunk_ids, evidence, reference_facts` |
| `dpo.jsonl` | ~1,500 | `id, prompt_messages, chosen, rejected, rejected_kind (hallucinated\|verbose\|fabricated_abstention), chunk_ids` |
| `grpo_prompts.jsonl` | ~800 (step 6 subsamples) | `id, kind, question, reference_answer, reference_facts, answerable, chunk_ids` |
| `MANIFEST.json` | n/a | per-file SHA-256, row counts by kind, generator model id, prompt-template hashes, date, token usage and cost, leakage thresholds and results |

SFT mix (target of ~6,000): ~3,300 single-hop (about 3.5 per chunk, more for table/infobox chunks),
~1,500 multi-hop, ~900 abstention (15%, so the model learns to say it does not know). `kind` is
`single_hop | multi_hop | unanswerable`, matching the golden set's types.
`reference_facts` is a short list of strings (numbers, entity names) a correct answer must contain; step 4's
fast local GRPO reward matches on it.
Abstention target is one fixed phrase: "I don't have information about that in my Old School RuneScape Wiki
knowledge." (constant in code, so step 4 can detect it exactly.)
System prompt for SFT rows is a short closed-book OSRS assistant instruction (constant, in `prompts.py`).
Rows store plain chat messages; the Qwen3 chat template (thinking disabled) is applied at training time in
step 5, not baked in here.

Splits: random 95/5 train/val by pair. Val measures format and loss, not generalization (memorization is the
goal and the facts overlap by design); the README does not present it as held-out accuracy.
DPO and GRPO prompts: DPO draws its prompts from the SFT pool (rejected answers are new); GRPO prompts are
**freshly generated** with a different angle/seed so RL is not rewarded for pure recall of SFT rows.

#### Generation pipeline

New package `src/remember_or_retrieve/datagen/`:

- `llm.py`: thin Anthropic wrapper. Builds Batch API requests (forced tool output, no sampling params),
  submits, polls, collects, and writes raw results to untracked `data/raw/` (gitignored) so a crashed run
  resumes instead of re-paying. Cost accounting from `usage` with prices read from config; a `--max-usd`
  guard aborts before submit if the pre-computed token estimate exceeds it. API key from
  `ANTHROPIC_API_KEY` only; the code checks the variable is set, never prints it. Client injectable so tests
  use a fake.
- `prompts.py`: all templates, plus a hash of each recorded in the manifest. Chunk text is wrapped in
  delimiters and the model is told it is data, not instructions (wiki text is untrusted input); output is
  forced through a JSON-schema tool, so injected text cannot change the format.
- `sample.py`: chunk selection and pairing. Single-hop: per chunk, N scaled by `source_type` and
  `token_count`. Multi-hop: chunk pairs from the same page (neighbouring sections) plus cross-page pairs
  where one chunk mentions the other's page title; ~1,500 pairs sampled with a fixed seed. Abstention: for
  each page cluster, Claude proposes plausible OSRS questions the shown chunks do not answer.
- `validate.py`: programmatic gates. (1) Schema. (2) `evidence` quotes must be substrings of the cited
  chunk(s) after whitespace normalization. (3) Every number in a single-hop answer appears in the evidence;
  multi-hop answers must show the arithmetic and every operand must appear in the evidence. (4) Abstention
  questions are verified by a second batch call: given the bge top-5 nearest chunks, Claude must answer
  "not answerable" or the row is dropped (so "out-of-corpus" really is). (5) Multi-hop answers get an
  independent second-pass solve (batch, cheap); disagreement drops the row. (6) Near-duplicate removal within
  the training set (bge cosine and 5-gram Jaccard).
- `leakage.py`: embeds every training question and the 55 golden questions with bge-large (CPU is fine for
  ~8k short strings), drops any training question with cosine >= threshold or normalized 5-gram Jaccard >=
  threshold against any golden question, or an exact normalized match. Starting thresholds 0.88 and 0.5 are
  **calibrated, not assumed**: the script prints the top nearest training question per golden question and
  the similarity histogram, I read the borderline cases by eye, and the final thresholds and the number
  dropped are recorded in the manifest. A hard gate (`scripts/check_leakage.py`, also a test over the
  committed files) fails if any frozen row is over threshold or equals a golden question.
- `build.py`: orchestration. CLI `scripts/gen_training_data.py` with subcommands
  `pilot | estimate | submit | collect | filter | freeze | report`.

DPO rejected-answer construction (one Claude batch call per prompt, given the gold answer and chunks):
`hallucinated` (fluent, confident, subtly wrong specifics, ~700), `verbose` (correct but padded, hedging,
restating the question, ~500), `fabricated_abstention` (a confident invented answer to an unanswerable
question, ~300). Chosen is the concise gold answer (or the abstention phrase). Stated limitation: these are
**off-policy** synthetic rejecteds, not samples from the SFT model (which does not exist until the paid
window); the README says so, and step 9 can optionally add on-policy pairs if budget allows.

Dependencies (pinned, new optional extra `data`): `anthropic`, `numpy`, `sentence-transformers` (CPU torch).
Added to `pyproject.toml`; `dev` extra unchanged. Config: new `training_data:` section in
`config/config.yaml` (generator model id, per-kind targets, seeds, thresholds, output paths, `max_usd`,
per-million-token prices). Model id and prices are verified with the claude-api reference at implementation
time, not hardcoded from memory.

#### Cost

Estimated **$2 to $4** (matches the roadmap): ~955 single-hop calls, ~1,500 multi-hop, ~300 abstention
proposals, plus verification and rejected-answer calls; mostly ~1k tokens in, ~0.5k out each, at Batch API
50% pricing for Sonnet 5.5. The exact figure is computed by `estimate` from real token counts (the Anthropic
count-tokens endpoint is free) and shown before any paid submit. Running total: **$0 now, ~$2 to $4 after this
step, against the $20 target and $50 cap** (inside the roadmap's $3 to $7 Claude line). If the estimate
exceeds $5, I stop and ask before submitting.

Process gate: a **pilot** of ~20 chunks through the synchronous API (<$0.15) is printed for you to eyeball
(answer style vs the golden set, arithmetic shown, abstention phrasing) before the full batch runs. You
approve the printed estimate before `submit`.

#### Security

- Credential: `ANTHROPIC_API_KEY` env var only; never read from or echoed, no `.env` opened, not in config,
  tests, logs, or manifest. Step 4 and 10 reuse the same variable.
- No new public endpoint, IAM role, or cloud resource. No GCP contact at all.
- Prompt-injection surface: corpus text is untrusted; mitigated by delimiting, forced tool schema, and
  validation gates above. Raw API responses stay in the gitignored `data/raw/`.
- Licensing: the corpus is CC BY-NC-SA 3.0 and the generated data is derived from it. `data/NOTICE.md` is
  extended to say so, and the README later notes the resulting weights are non-commercial/ShareAlike.
- Review `git diff --staged` for secrets before every commit.

#### Tests (offline, fast; the one real API call is marked `slow`)

- Schema and constants (`ABSTAIN_PHRASE`, kind names) and golden-compatible `kind` values.
- `validate.py`: evidence-substring check passes/fails correctly; number-in-evidence gate; arithmetic
  operand gate; dedup.
- `leakage.py`: a paraphrase and an exact copy of a golden question are caught, an unrelated question passes
  (small embedding model fixture or precomputed vectors so CI stays light).
- `sample.py`: deterministic with a seed; kind mix and per-page coverage match targets; every chunk id exists.
- `llm.py`: fake client; resume skips already-collected requests; `--max-usd` guard aborts; the key is never
  logged.
- Frozen-data tests over the committed files: row counts and mix within tolerance, ids unique, every
  `chunk_id` in the corpus, no golden question present, SHA-256 matches `MANIFEST.json`, DPO chosen != rejected.
- `ruff check`, `ruff format --check`, `pytest` green in CI (CI installs only the `dev` extra plus what the
  frozen-data and unit tests need; the embedding test is `slow` or uses fixed vectors).

#### Branch and workflow

On approval: copy this plan into `docs/roadmap.md` under "Step 3" and mark the step in progress, then branch
`feat/training-data`; small commits in order: config + deps + schemas; sampling; prompts + llm wrapper;
validators; leakage; pilot (output not committed); full run; frozen data + manifest + tests; roadmap
update. Any deviation (a threshold that needed changing, a kind count that came back different, real cost)
goes into the roadmap entry. Nothing is committed directly to `main`.

#### Verification

1. `pytest`, `ruff check`, `ruff format --check` pass locally and in CI.
2. Pilot output read by hand and approved before the full batch.
3. `scripts/gen_training_data.py report` prints counts by kind, validation drop rates per gate, leakage drops,
   and actual spend (vs estimate).
4. `scripts/check_leakage.py` exits 0 on the frozen files and exits non-zero on a copy with one golden
   question injected (checked once manually).
5. `scripts/verify_data.py` plus the new manifest check pass. Spot-read ~30 random rows per kind.
6. Spend recorded in the roadmap. Nothing billable remains running, so there is no teardown to confirm
   (stated explicitly in the roadmap entry).

#### Open items I will settle during implementation (not blocking approval)

- Exact leakage thresholds (calibrated as above).
- Exact per-kind counts after validation drops (target totals ~6,000 SFT, ~1,500 DPO, ~800 GRPO prompts; if a
  gate drops far more than expected I regenerate that kind rather than lower the bar).

#### Outcome and deviations from the plan (recorded 2026-10-13)

**Result.** `data/training/`: `sft_train.jsonl` 4,841 + `sft_val.jsonl` 255 (5,096 SFT rows: 4,000
single-hop, 422 multi-hop, 674 abstention), `dpo.jsonl` 1,500 (700 hallucinated, 500 verbose, 300
fabricated-abstention), `grpo_prompts.jsonl` 404 (275 / 62 / 67), `MANIFEST.json`. The leakage gate
(`scripts/check_leakage.py`) reports 0 leaks over 7,000 training questions and fails when a golden
question is injected. 30 tests pass.

**Spend ledger (USD, real, billed at batch rates except the pilots).**

| Item | Cost |
|---|---|
| Pilots (sync calls, standard rate) | 0.282 |
| Generation batch (2,685 requests) | 4.294 (estimate 3.70) |
| Verification batch (1,695 requests) | 0.246 |
| DPO rejected-answer batch (1,800 requests) | 0.160 |
| **Step 3 total** | **4.98** |

Running project total: $4.98 of the $20 target and $50 cap. This is above the plan's $2 to $4 but
inside the roadmap's $3 to $7 Claude line. No cloud resource was created, so there is no teardown.

**Deviations from the approved plan.**

- **Cost was re-planned after the first estimate.** The plan's $2 to $4 assumed Sonnet 5.5 for
  everything. The first estimate was $13.10, so (with the user's approval) single-hop, abstention,
  verification, and DPO-rejected calls use **Haiku 5.5** (`cheap_model`); only multi-hop generation
  uses Sonnet 5.5. The estimator was recalibrated against real pilot usage (chars/2.6) and still came in
  16% under the actual generation cost.
- **Structured outputs, not forced tools.** Sonnet 5.5 rejects forced `tool_choice` with a 400, so
  every call uses `output_config.format` (JSON schema). Thinking is "off" via `between_tools` on
  Sonnet and `disabled` on Haiku; multi-hop and the multi-hop solver run adaptive at medium effort.
- **Multi-hop pairing was redesigned after the pilots.** Random same-page pairs produced contrived
  questions. Candidates are now adjacent same-page chunks plus cross-page links through specific page
  titles (mentioned in at most 25 chunks, one best-overlap partner), and both chunks must contain at
  least 6 distinct numbers. That left 1,147 candidates, so multi-hop requests were cut from 2,000/300
  to 950/150. Yield after all gates is about 40%. **Result: 422 SFT multi-hop rows, not 1,500 (and 62
  GRPO, not 200).** The model sees less multi-hop arithmetic in training than the plan intended; this
  is a known weakness to look for in the step 9 multi-hop scores. Single-hop was raised to 4,000
  (7/6/5 pairs per table/infobox/prose chunk) to partly compensate.
- **SFT is 5,096 rows, not ~6,000**; abstention is 674 (13%), GRPO is 404, not ~800.
- **Operand check is approximate.** For multi-hop, "every operand must appear in evidence" is
  implemented as: at most 2 numbers in the answer may be absent from the evidence and question (the
  result and one intermediate). The independent Haiku solve is the second line of defence (27 rows
  dropped for disagreement).
- **Leakage threshold calibrated by hand: cosine 0.85, not 0.88.** At 0.88, real paraphrases of golden
  questions survived (0.877 and 0.878). 0.85 drops 120 questions in total, plus an exact/5-gram check.
  The generator reproduced several golden questions verbatim (cosine 1.000), which is expected since the
  golden set was written from the same pages.
- **Estimation used a heuristic**, not the free count-tokens endpoint (3,000+ calls were not worth it).

**Caveats to carry into the README.**

- **Abstention is "not in this corpus", not "not true".** Some abstention questions are real OSRS facts
  that are simply outside the 955 chunks (for example the God Wars Dungeon requirement). 280 candidates
  that the verifier judged answerable from the nearest chunks were dropped, but the model will still be
  trained to decline facts that exist on the wiki beyond the corpus.
- **DPO rejected answers are off-policy** (written by Haiku, not sampled from the SFT model).
- **The validation split measures format and loss, not generalization**; facts overlap with training
  by design, and with the golden set (only the questions are excluded).
- Some multi-hop questions are still somewhat contrived even after the pairing fix.
