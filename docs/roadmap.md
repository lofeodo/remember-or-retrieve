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
- [x] **Step 4 — Scoring harness and reward functions** (`feat/scoring-harness`, $0.29 judge calls)
- [x] **Step 5 — Distributed training pipeline: SFT, LoRA vs full, DPO** (`feat/training-pipeline`, free)
- [x] **Step 6 — GRPO stage** (`feat/grpo`, free)
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

**Spend to date: $5.27** (step 3 $4.98 plus step 4 $0.29, Claude API only; ledgers in each step's outcome). Nothing has been provisioned.

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

### Step 4 — Scoring harness and reward functions (`feat/scoring-harness`, under $1 judge calls)

Plan approved 2026-10-13. Status: complete (see outcome and deviations).

#### Context

Every later step needs one identical way to score an answer. Step 6 (GRPO) needs a fast local reward,
step 8 logs every checkpoint's golden-set score to the registry, step 9 scores each checkpoint as it is
produced, and step 10 compares the fine-tune, the live RAG API, Mistral and Cohere. If each used its own
scorer the head-to-head would not be apples to apples. This step builds that one scorer and proves it by
re-scoring `rag-with-receipts`' recorded answers and reproducing its published numbers. Code and data
only: no GPU, no GCP, no training. Spend: judge calls only (~$0.3 to $0.6 estimated). Running total:
$4.98 now, ~$5.3 to $5.6 after, against the $20 target and $50 cap.

#### What exploration found (shapes the design)

- **Recorded results**: `rag-with-receipts/results/eval_report.json` holds, for all 55 questions, the RAG
  answer, `answerable` flag, citations (claim + chunk), judge verdict, NLI citation verdicts, and a
  `summary`. Published: accuracy **0.83636 (46/55)**, single-hop 1.0 (n=35), multi-hop 0.4667 (n=15),
  labels correct 42 / partially_correct 1 / incorrect 3 / incorrectly_abstained 4 / correct_abstention 4 /
  incorrectly_answered 1, grounded 0.948, contradicted 0.052, fully-grounded answers 0.909 (77 citations,
  44 answers). Stale numbers in that repo's CLAUDE.md (0.80, 96.0%) are ignored.
- **Judge**: `claude-sonnet-5`, forced `submit_verdict` tool (`correct | partially_correct | incorrect` +
  reasoning), no temperature, so non-deterministic. Sees only question, gold answer, candidate answer.
  The system prompt is short and is replicated verbatim.
- **Abstention** for RAG is its own `answerable` flag (no judge call): expected-no/said-no =
  `correct_abstention`, expected-yes/said-no = `incorrectly_abstained`, expected-no/said-yes =
  `incorrectly_answered`. Only `correct` and `correct_abstention` count; `partially_correct` is wrong.
- **Grounding**: `cross-encoder/nli-deberta-v3-base`, softmax, columns reordered by `id2label`, premise =
  breadcrumb + chunk text, hypothesis = the citation's claim; `contradicted` if p(contra) >= 0.5, else
  `grounded` if p(entail) >= 0.5 or word-overlap >= 0.3, else `ungrounded`. Per citation, not per answer.
  Runs on CPU or GPU.
- **Not available upstream**: per-question latency and cost (aggregates only); chunk text is not in the
  live API response. The corpus is already here, so chunk text resolves locally by `chunk_id`.
- **Already here to reuse**: `datagen/llm.py` (`make_client`, `usage_cost`, `Request`/`sync_call`, budget
  guard, key-from-env), `datagen/validate.py` (`numbers`, `alnum`, `norm_words`, `solver_agrees`),
  `prompts.ABSTAIN_PHRASE` / `SFT_SYSTEM_PROMPT`, `data.load_golden_set` / `load_corpus`.

#### Decisions (confirmed with the user)

1. **Judge = `claude-sonnet-5`**, same model and prompt as `rag-with-receipts`, so the re-score check is
   meaningful and every system is graded identically.
2. **Grounding for every system = answer vs the golden question's gold chunks.** Each answer is split into
   sentence-level claims; the same NLI checker runs each claim against the gold chunks (a claim is
   supported if any gold chunk grounds it, contradicted if any contradicts and none grounds). Same evidence
   for the fine-tune, RAG, Mistral and Cohere. The RAG's native citation-grounding stays as a separate,
   clearly labelled number and is never mixed into the head-to-head. Unanswerable questions have no gold
   chunks, so they are scored by abstention only (an answer there counts as a hallucination).

#### Design

New package `src/remember_or_retrieve/scoring/`:

- `schema.py`: the one input format every system produces, JSONL
  `{question_id, answer, abstained (optional, system's own flag), citations (optional), latency_s,
  input_tokens, output_tokens, cost_usd}`. Plus `ScoredAnswer` and `Summary` dataclasses. Steps 9 and 10
  write this format; adapters live with them, except the RAG-report adapter below.
- `abstention.py`: `is_abstention(answer)`: exact normalized match or prefix of `ABSTAIN_PHRASE` (the
  fine-tune's trained phrase), used when a system has no flag. If a system supplies its own flag (RAG), the
  flag wins. Labels are the RAG project's six, so tables line up.
- `judge.py`: `Judge.score(question, gold, answer)` with the verbatim prompt, tool schema, user-message
  format, `max_tokens=1024`, refusal -> `incorrect`, errors -> `error` label (as upstream). Reuses
  `llm.make_client`/`usage_cost`. Results cached on disk (gitignored `data/raw/scoring/`) keyed by hash of
  (question, gold, answer, judge model, prompt hash), so re-scoring unchanged answers is free and a crashed
  run resumes. `--max-usd` guard before any paid call. Concurrency small (threads) to respect rate limits.
- `grounding.py`: `GroundingChecker` replicating upstream: load CrossEncoder (pinned revision, no
  `trust_remote_code`), `apply_softmax`, reorder columns by `id2label` (asserted at load, not assumed),
  breadcrumb + text premise, thresholds from config (0.5 / 0.5 / 0.3). `check_pairs(pairs)` is the
  upstream-compatible primitive; `check_answer(answer, gold_chunks)` adds sentence splitting and the
  any-gold-chunk aggregation. Heavy imports lazy; `sentence-transformers` is in a new `eval` extra.
- `metrics.py`: `summarize(scored, golden)` reproducing upstream's formulas: accuracy over non-error rows,
  per-type accuracy, label counts, abstention precision/recall, grounded / contradicted / ungrounded rates
  and fully-grounded-answer rate (answers with no claims skipped, per upstream), latency p50/p95 and cost
  totals when present.
- `reward.py`: fast, local, no network, no GPU, for GRPO. `reward(row, completion) -> RewardBreakdown`
  with components, weighted from config:
  - *correct*: fraction of `reference_facts` present (reuse `validate.numbers`/`alnum`, same matching as
    `solver_agrees`), full credit only if all present; multi-hop gets credit for the final numbers.
  - *abstention*: unanswerable row -> +1 for the exact phrase, 0 otherwise (and the hallucination
    penalty bites); answerable row -> abstaining scores 0 and takes a penalty.
  - *hallucination proxy*: numbers in the completion that appear in neither the reference answer, the
    question, nor the cited chunks' text are penalized per unsupported number (cheap stand-in for the NLI
    checker; running DeBERTa on every rollout would cost more than it buys).
  - *verbosity*: smooth penalty above a length budget set from the SFT answer length distribution.
  Output in a bounded range; the breakdown dict is returned so step 6 can log components. NLI as an
  optional extra reward term is left off by default; step 6 decides if rollout budget allows it.
- `rag_reference.py`: adapter that loads the committed reference (below) into `ScoredAnswer`s with the RAG's
  recorded verdicts, and into the shared answer schema for re-judging.
- `scripts/score.py` (argparse, same style as `gen_training_data.py`, paid subcommands marked): 
  `replay` (free: recompute the summary from recorded verdicts), `judge --answers F --out F` (PAID),
  `ground --answers F`, `score` (all of it for one answers file, writes scored JSON + summary),
  `rag-check` (the validation below), `reward-check`.
- Config: new `scoring:` section in `config/config.yaml`: judge model, judge price per Mtok (sonnet-5
  $2 in / $10 out, verified with the claude-api reference at implementation time), NLI model + revision +
  thresholds, `max_usd`, cache dir, reward weights and length budget, bounds for the validation checks.
- Dependencies: new `eval` extra (`anthropic==1.13.0` already pinned, `sentence-transformers`, `numpy`),
  versions pinned. CI keeps installing only `dev`; tests needing the NLI model are `slow`.

**Reference data (committed, public).** `data/eval/rag_reference.json`: a slimmed copy of
`rag-with-receipts/results/eval_report.json` per question (id, answer, `answerable` flag, citations as
`{chunk_id, claim}`, judge verdict, NLI citation verdicts, summary), with chunk text dropped (it resolves
from our corpus; the plan checks all cited ids exist). Recorded in `data/PROVENANCE.json` with the upstream
commit, source path and SHA-256, and covered by `verify_data`. Contains no keys and no demo key.

#### Validation (the point of the step)

1. **Replay, deterministic, free, runs in CI.** Recomputing the summary from the recorded verdicts must give
   exactly 0.83636 / 46 of 55, single-hop 1.0, multi-hop 0.4667, the six label counts, grounded 0.948,
   contradicted 0.052, fully-grounded 0.909, 77 citations / 44 answers. Proves `metrics.py` and the
   abstention logic.
2. **NLI re-implementation (`slow`, local).** Recompute NLI on the 77 recorded (claim, chunk) citation
   pairs with our `GroundingChecker`: labels identical on all 77 and probabilities within 1e-3 of the
   recorded ones. This is the empirical label-order check, the same discipline as upstream's NLI
   check, and it also fails loudly if the model revision drifts.
3. **Judge re-score (PAID, ~$0.15 per pass).** Re-judge the stored RAG answers (~46 answerable-and-answered)
   twice. Report verdict agreement with the recorded verdicts and run-to-run agreement (the noise floor).
   Pass criteria, set before running: accuracy within 0.055 (3 questions) of 0.836 on each pass, and
   per-question agreement >= 90% (single-hop must stay ~1.0; the known hard cases are multi-hop). Any
   disagreement is listed and read by hand. If it fails, I stop and report rather than loosen the bar.
   The measured judge noise becomes the +/- band step 10 attaches to every comparison.
4. **Gold-chunk grounding on the RAG answers (informational).** Run decision 2's checker on the 44
   answered RAG answers and report it beside the native 0.948. It is a different definition and no match
   is expected; the number is a baseline and the difference is documented. Sentence splitting and
   known NLI weakness on flattened tables are noted, not tuned (thresholds untouched).
5. **Reward sanity, free.** (a) On all 1,500 DPO pairs, reward(chosen) > reward(rejected) for at least the
   expected share per `rejected_kind`; list failures by kind. (b) The reference answer scores near max on
   every GRPO prompt and the abstention phrase scores max only on unanswerable rows. (c) Reward vs judge on
   the 46 RAG answers where `gold_answer` contains numbers: report agreement. (d) Adversarial cases:
   empty string, abstain-everything, padded answer with the right number, confident wrong number. None may
   outscore a concise correct answer. A reward that fails (d) is fixed before step 6, not after.

#### Tests (offline, fast)

Plain pytest functions, fakes not mocks, per repo convention. Schema round trip; `is_abstention`
variants; metrics replay equals published numbers; judge cache hit/miss, `--max-usd` abort, refusal and
error mapping, key never logged (fake client); grounding aggregation with a fake NLI scorer (any-gold-chunk
logic, empty answer, sentence splitting); reward components, bounds and the adversarial cases; reference
file checksum and all cited chunk ids exist; the DPO chosen-vs-rejected check as a test over committed
data. NLI model test marked `slow`. `ruff check` and `ruff format --check` green in CI.

#### Security

- Key: `ANTHROPIC_API_KEY` env var only, checked for presence, never read from files or printed. No `.env`
  is opened.
- No endpoint, IAM role, GCP contact, or cloud resource. Mistral/Cohere/RAG API calls belong to step 10.
- Candidate answers are untrusted text going into the judge. The judge prompt is kept verbatim for
  comparability, so I'm not adding delimiters; the output is a forced enum, which limits what an injection
  can do. Noted in the README limitations.
- Model weights: NLI model pinned by revision, no `trust_remote_code`, no unpickling.
- The reference file is checked (`git diff --staged`) for keys and demo-key strings before commit.

#### Branch and workflow

On approval: copy this plan into `docs/roadmap.md` under Step 4 and mark it in progress; branch
`feat/scoring-harness`. Small commits, in order: config + deps + schema; reference data + provenance;
abstention + metrics + replay test; judge (cache, guard); grounding; reward; CLI; validation runs and
results recorded; roadmap outcome. Deviations (a threshold that had to change, a failed pass criterion, real
cost) go into the roadmap entry. Nothing goes to `main` directly.

#### Cost and teardown

Judge: ~46 calls x 3 passes (two re-scores plus any rerun after a fix) at ~$0.003 each, ~$0.5 worst case;
estimated **$0.3 to $0.6**. The CLI prints the estimate and requires `--max-usd` (default $1) before any
paid call. Running total ~$5.3 to $5.6 of $20 target / $50 cap. Nothing billable is created, so no
teardown to confirm (stated in the roadmap entry).

#### Verification

1. `pytest`, `ruff check`, `ruff format --check` pass locally and in CI; `verify_data.py` passes with the
   new reference file.
2. `scripts/score.py replay` reproduces the published numbers exactly.
3. `slow` NLI test passes on the 77 recorded pairs.
4. Judge re-score meets the pre-set criteria; per-question disagreements listed and read.
5. Reward sanity checks (a to d) pass; results table recorded in the roadmap.
6. Spend ledger recorded in the roadmap.

#### Open items settled during implementation (not blocking)

- Confirm `claude-sonnet-5` is still callable and its price; if retired, stop and ask.
- Read upstream `ingestion/utils.breadcrumb` to replicate the premise exactly.
- Sentence-splitter choice (a simple regex that keeps decimals and "1,200"-style numbers intact) and the
  length budget for the verbosity term (from SFT answer lengths).
- Reward weights: start from the config defaults and adjust only to fix a failed sanity check, recording
  what changed.

#### Outcome and deviations from the plan (recorded 2026-10-13)

**Result.** `src/remember_or_retrieve/scoring/` (schema, abstention, metrics, judge, grounding,
reward, pipeline, rag_reference), `scripts/score.py` (`replay | score | rag-judge | rag-nli |
rag-ground | reward-check`), `scripts/import_rag_reference.py`, and `data/eval/rag_reference.json`
(pinned to `rag-with-receipts` commit `6b8d032`, SHA-256 in `PROVENANCE.json`). 81 tests pass in CI
mode; the `slow` NLI test passes locally.

**Validation, against the pass criteria set before running.**

| Check | Result |
|---|---|
| Replay of the recorded run | Exact: 0.83636 (46/55), single-hop 1.0, multi-hop 0.4667, all six label counts, grounded 0.948, contradicted 0.052, fully grounded 0.909, 77 citations / 44 answers. Our abstention logic plus the recorded judge verdict rebuilds upstream's label on all 55 questions. |
| NLI re-implementation | 77 of 77 labels identical, max probability difference 1.1e-5. Label order read from the model's own config: `{0: contradiction, 1: entailment, 2: neutral}`. Model revision pinned to `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`. |
| Judge re-score (2 fresh passes, 46 answered) | Accuracy 0.8364 on both passes (delta +0.0000 against the bar of 0.055). Agreement with recorded verdicts 45/46 = 97.8% (bar 90%). The one difference is q048, `incorrect` to `partially_correct`, both counted wrong. Run-to-run agreement 100%. |
| Gold-chunk grounding on the RAG answers (informational) | 85.9% grounded over 85 claims / 46 answers (single-hop 89.3%, multi-hop 79.3%), contradicted 9.4%, ungrounded 4.7%, fully grounded answers 82.6%. Native citation grounding was 94.8%. Different definition, no match expected. |
| Reward vs DPO pairs | reward(chosen) > reward(rejected) on 691/700 hallucinated (98.7%, bar 97%), 493/500 verbose (98.6%, bar 97%), 300/300 fabricated abstention (bar 100%). Misses are pairs where the chosen answer itself lacks a text fact. |
| Reward vs GRPO prompts | Reference answers score >= 0.9 on 400/404; the abstention phrase scores 1.0 only on unanswerable prompts. |
| Reward adversarial cases | Empty, abstain-everything, confident wrong number, padded-with-right-number, number spray, and a stuffed hallucinated fact all score below a concise correct answer. |
| Reward vs judge on RAG answers | The numeric-fact match agrees with the judge on 31/34 answers whose gold answer has numbers. The 3 misses are gold-format quirks ("04-Jan-01" vs "4 January 2001", "1" vs "One"). |

**Spend ledger (USD).** Judge re-score, two passes of 46 calls: **0.293** (estimate was 0.3 to 0.6).
Running project total: **$5.27** of the $20 target and $50 cap. Nothing billable was created, so
there is no teardown to confirm.

**Deviations from the approved plan.**

- **Soft matching for text facts in the reward.** Exact substring matching gave a paraphrased correct
  answer partial credit (8 of 404 GRPO reference answers fell below 0.9). Non-numeric facts now count
  as present at >= 60% word recall, proportional below. The 4 remaining rows are data quirks in the
  frozen GRPO set: the reference answer omits one of its own facts (3 rows) or a fact contains a U+FFFD
  character (1 row). The frozen data was not edited; the test allows up to 2%. Step 6 can drop them.
- **Number-spray guard added** (`stray_number`): a short answer listing every number from the evidence
  would otherwise get full correctness credit with no hallucination penalty.
- **Reward range is [-1, 1]**, not [0, 1], so GRPO's group-relative advantage still separates two bad
  completions.
- **Judge and CLI used the sync API**, as upstream does, with a disk cache keyed by (model, prompt hash,
  question, gold, candidate). `--passes 2` re-judges with the cache bypassed.
- **The reference file is a slim copy** (answers, citations without chunk text, recorded verdicts). Its
  `PROVENANCE.json` entry keeps the file's existing CRLF line endings.
- `test_data.py::test_missing_file_reported` now expects 3 manifest files (it hard-coded 2).

**Caveats to carry into the README.**

- **The judge is not deterministic** and sees only question, gold answer and candidate. It was stable here
  (100% run-to-run on 46 answers), but that is a small sample; step 10 should attach the measured noise
  to every comparison and re-run if a headline gap is within one or two questions.
- **Gold-chunk grounding is a stricter, different measure** than the RAG's native citation grounding. A
  claim that needs two chunks together can read as ungrounded against either alone, and the NLI model
  reads flattened tables poorly (thresholds untouched, as upstream). It is reported for all four systems
  the same way and never mixed with the native number.
- **Numeric-fact reward cannot read format variants** (dates, number words), so it is a training signal,
  not a grader. Final checkpoint scores come from the judge.
- Candidate answers are untrusted text in the judge prompt, kept verbatim for comparability; the forced
  enum output limits what an injection can do.

### Step 5 — Distributed training pipeline: SFT, LoRA vs full, DPO (`feat/training-pipeline`, free)

Plan approved 2026-10-13. Status: complete (see outcome and deviations).

#### Context

Steps 3 and 4 produced the frozen training data (`data/training/`: 4,841 + 255 SFT rows, 1,500 DPO pairs,
404 GRPO prompts) and the scorer. Nothing trains yet. This step writes, and proves correct, all the training
code for the PyTorch stages that run in the one paid GCP window (step 9): SFT, the LoRA-vs-full comparison,
and DPO, orchestrated by Ray Train over DDP and FSDP2, with per-stage profiling and the distribution checks
that CLAUDE.md requires ("ranks started, every GPU busy, one coherent loss curve").

Code only. No GPU rental, no GCP, no API spend, no real checkpoint (that is step 9). GRPO is step 6, JAX is
step 7, Terraform and the registry are step 8. **Cost: $0. Running total stays $5.27 of the $20 target and
$50 cap.** Nothing billable is created, so there is no teardown to confirm.

#### What exploration found (shapes the design)

- Compute plan is fixed: `Qwen/Qwen3-8B`, `a2-highgpu-4g` (4x A100 40GB, spot), **full fine-tune = FSDP
  full-shard (~25 GB/GPU for states, then activations), LoRA / DPO / GRPO = DDP**, all under Ray Train.
  `config.yaml` `compute:` already carries this.
- Data schemas (step 3): SFT rows hold `messages[system,user,assistant]`; DPO rows hold `prompt_messages`,
  `chosen`, `rejected`, `rejected_kind`. Chat template is applied at training time with thinking disabled.
  Abstention is the fixed phrase in `datagen/prompts.ABSTAIN_PHRASE`. Rows are short (~10-word answers plus a
  ~60-token system prompt), so one SFT epoch is only ~0.5M tokens. Memorization is the goal, so several epochs;
  epoch count is config, final value chosen in step 9 from the rehearsal numbers.
- Dev machine: Windows 11, Python 3.11 venv, **CPU-only torch 2.14.1 and transformers 5.19, no Ray, no peft,
  no TRL**, RTX 3060 12 GB unused, Docker installed (WSL2 backend). Ray multi-worker on native Windows is
  experimental, so the smoke test runs on Linux (Docker locally, ubuntu in CI).
- Reusable: `config.load_config/resolve_path`, `data.load_golden_set`, `scoring.schema.read_answers/write`
  (answers file format step 9 scores), `datagen.prompts.SFT_SYSTEM_PROMPT/ABSTAIN_PHRASE`, `scripts/*` CLI style
  (argparse subcommands), `slow` pytest marker, fakes-not-mocks test convention.

#### Decisions (confirmed with the user)

1. **Hybrid loops.** Own PyTorch loop for SFT / LoRA / full (explicit DDP and FSDP2 wrapping, visible gradient
   all-reduce, built-in profiling and rank checks). **TRL `DPOTrainer`** for DPO, run under Ray Train. If TRL's
   Ray/Accelerate integration misbehaves with the pinned versions, the fallback is a ~40-line own DPO loss in
   the same loop; any switch is recorded here as a deviation.
2. **Smoke test on Linux**, not native Windows: 2 Ray CPU workers, gloo backend, tiny randomly initialised
   Qwen3-shaped model, run in a local Docker container and in a new CI job.

#### Design

New package `src/remember_or_retrieve/training/`:

- `data.py` — loads SFT/DPO JSONL; applies the Qwen3 chat template with `enable_thinking=False`; builds
  **assistant-only loss labels** (prompt tokens masked to -100, `<|im_end|>` kept as a target so the model
  learns to stop); length-bucketed dynamic padding; `DistributedSampler` sharding with a per-epoch seed.
  Tokenizer injectable so unit tests use a fake one.
- `model.py` — loads the model in bf16 with SDPA attention, `trust_remote_code=False`, safetensors only,
  pinned HF revision. `build_lora(model, cfg)` via peft (all linear projections, r/alpha from config).
  `tiny_model(cfg)` builds the smoke-test model from a small `Qwen3Config` (random init, real tokenizer).
- `parallel.py` — `wrap_ddp(model)` and `wrap_fsdp2(model)` (`torch.distributed.fsdp.fully_shard` per decoder
  layer, bf16 params, fp32 gradient reduction, optional activation checkpointing and CPU offload flag).
  Checkpointing: LoRA saves adapters only; full saves a gathered bf16 state dict on rank 0 (safetensors), with
  a sharded DCP save path for resume. Resume from the latest Ray checkpoint so a spot preemption costs minutes,
  not the stage.
- `sft.py` — the training loop for `--mode full|lora`: cosine schedule with warmup, gradient accumulation
  (`no_sync` under DDP), grad clipping, bf16 autocast on GPU, periodic val loss, global-mean loss via
  all-reduce.
- `dpo.py` — TRL `DPOTrainer` on the SFT checkpoint with LoRA; reference = adapter-disabled policy (no second
  model copy, which matters at 40 GB). Converts `dpo.jsonl` to TRL's conversational prompt/chosen/rejected.
  Takes `--init-from <checkpoint>` since which SFT variant wins is decided in step 9.
- `profiling.py` — one `StageProfile` schema for every stage, written as `profile.json` next to the
  checkpoint: tokens/sec (global and per device, non-pad tokens), step-time p50/p95 (CUDA events), **peak
  memory per device** (`max_memory_allocated`, reset per stage), trainable vs total params, estimated MFU
  against the A100 bf16 peak, wall-clock including model load, and an optional short `torch.profiler` window
  (compute vs NCCL time) saved as a trace. Per-rank values gathered with `all_gather_object`. Step 6 adds
  rollout throughput to the same schema.
- `checks.py` — **the distribution checks**, each an assertion that fails loudly (same discipline as step 4's
  NLI label-order check):
  1. *Ranks started:* every rank reports `{rank, world_size, host, pid, device}`; world size equals the
     requested workers, pids and devices unique per host.
  2. *Gradients really sync:* after one backward, the max-min of each rank's gradient checksum is 0 (DDP);
     under FSDP2, parameters are sharded `DTensor`s and each rank holds ~1/world of the total numel.
  3. *Weights stay identical across ranks* (DDP) after N optimizer steps.
  4. *Every GPU busy:* on GPU, a `pynvml` sampler per rank records utilization over the run; passes only if
     every device's median utilization exceeds a threshold (skipped on CPU, covered in step 9's real run).
  5. *One coherent loss curve:* per-rank losses within tolerance of the all-reduced mean at every step.
  6. *Equivalence to single-process:* a 2-rank run with the same global batch and seed matches the 1-rank
     loss curve within tolerance (DDP and FSDP2). This is the strongest proof the run is one training job,
     and the same code path doubles as the single-GPU reference run (`--num-workers 1`) for throughput.
- `launch.py` — Ray Train `TorchTrainer` (`ScalingConfig(num_workers, use_gpu)`, `RunConfig(storage_path,
  FailureConfig(max_failures))`), NCCL backend on GPU and gloo on CPU, dashboard off. `train_loop_per_worker`
  dispatches on stage. Storage path is a config value so step 8 can point it at the GCS bucket.
- `generate.py` — greedy answer generation for the 55 golden questions with the same system prompt, merging
  LoRA adapters if needed, writing the **scoring answers format** (`scoring.schema`) so step 9 only has to run
  `scripts/score.py score`. Small, but it makes each stage's output scoreable the moment it exists.
- `scripts/train.py` (`sft | dpo | check | generate`, flags `--mode full|lora`, `--strategy ddp|fsdp`,
  `--num-workers`, `--smoke`) and `scripts/compare_stages.py` (reads `profile.json` files into the
  LoRA-vs-full table: trainable params, tokens/sec/GPU, peak GB, wall-clock; golden score filled in step 9).
- Config: new `training:` section in `config/config.yaml` (lr, batch sizes, epochs, max length, warmup, LoRA
  r/alpha/targets, DPO beta, per-stage strategy, profiling toggles, smoke model shape, checkpoint dir, the
  HF revision pin, check tolerances). Paths relative to repo root as elsewhere.
- Dependencies: new `train` extra, exact pins verified against PyPI at implementation time: `torch`,
  `transformers`, `peft`, `trl`, `accelerate`, `ray[train]`, `pynvml`, `safetensors`. The pins are chosen
  together so Qwen3, FSDP2, TRL and Ray Train agree; the dev extra and the main CI job are unchanged.

#### Tests

- **Fast, run in the existing CI job (no torch needed, fakes):** label masking, chat-template/format
  round trip with a fake tokenizer, sampler sharding (no overlap, full coverage), bucketing, schedule, profile
  schema, check logic given synthetic per-rank reports (a deliberately desynced report must fail), DPO row
  conversion over the committed file, every SFT row tokenizes within max length.
- **Smoke, new CI job `smoke-distributed` (ubuntu, CPU torch, `pip install -e ".[train]"`) and the same
  command in a local Docker container:** 2-worker Ray + gloo run of SFT full (FSDP2), SFT LoRA (DDP) and DPO
  for a few steps on the tiny model against real data slices; all six distribution checks pass; the 2-rank
  vs 1-rank equivalence holds; checkpoint save, kill-and-resume continues the same loss curve; `generate`
  writes a valid answers file that `scoring` loads. Marked `slow` locally; the CI job runs `-m slow` for
  this file only.
- `ruff check` and `ruff format --check` stay green.

#### Security

No credentials involved, no endpoint, no IAM, no GCP contact. Supply chain: exact pins; model and tokenizer
loaded with `trust_remote_code=False` from a **pinned HF revision**, safetensors only, `torch.load` only with
`weights_only=True`, no unpickling of downloaded files. Ray dashboard disabled and bound to localhost so no
open port. The new CI job gets `contents: read` only and no secrets. The Qwen3 download is public and needs
no token (Gemma's gated token belongs to step 7).

#### Branch and workflow

On approval: copy this plan into `docs/roadmap.md` under Step 5 and mark it in progress; branch
`feat/training-pipeline`. Small commits in order: config + `train` extra; data/collation/masking; model +
tiny model; profiling; parallel wrappers; SFT loop; distribution checks; Ray launch; DPO; generate + CLI;
smoke tests + CI job; roadmap outcome. Deviations (TRL fallback, pinned versions that had to change, a check
that needed a looser tolerance) go into the roadmap entry. Nothing is committed to `main` directly.

#### Verification

1. `pytest`, `ruff check`, `ruff format --check` green locally and in CI, plus the new smoke job green.
2. Local Docker run of the smoke suite: all six distribution checks pass; 2-rank loss curve matches 1-rank
   for DDP and FSDP2; resume works.
3. A deliberately broken run (one rank not syncing, or a world size mismatch) makes the checks fail, shown once.
4. `profile.json` for each smoke stage has tokens/sec, step times and peak memory per rank (CPU memory proxy
   locally; real GPU numbers come in step 9).
5. `scripts/compare_stages.py` renders the LoRA-vs-full table from the smoke profiles.
6. Roadmap entry records the plan, deviations, and "Spend: $0, nothing billable, no teardown".

#### Not in this step / known limits

- Real throughput, memory and the GPU-utilization check can only be proven on the A100s in step 9; this step
  proves correctness and that the measurement code runs. The 8B memory plan (FSDP ~25 GB/GPU plus activations,
  with the 8-bit-optimizer and partial-unfreeze fallbacks in `compute-plan.md`) is validated there, not here.
- FSDP2 (not FSDP1) is used; if the pinned torch's FSDP2 + Ray Train combination fails in the smoke test, FSDP1
  is the recorded fallback.
- Optional follow-up if wanted: a real-tokenizer single-GPU LoRA run on the local RTX 3060 (needs CUDA torch);
  deliberately left out as it adds install cost for little proof beyond the tiny-model path.

#### Outcome and deviations from the plan (recorded 2026-10-13)

**Result.** `src/remember_or_retrieve/training/` (data, model, parallel, sft, dpo, profiling, checks,
launch, generate, smoke), `scripts/train.py` (`sft | dpo | check | generate`),
`scripts/compare_stages.py`, `docker/smoke.Dockerfile`, a `training:` config section, a `train` extra,
and a `smoke-distributed` CI job. 100 tests pass in the dev-only job; the 10 distributed smoke tests
(2-worker Ray and multiprocessing runs on a tiny Qwen3-shaped model, CPU, gloo) pass in Docker and in
a clean-container replay of both CI jobs. Not yet run on GitHub Actions itself.

**Verification against the plan.**

| Check | Result |
|---|---|
| Ranks started, gradients synced, weights identical (DDP), parameters sharded (FSDP2), one coherent loss | pass in every SFT and DPO smoke run |
| 2-rank vs 1-rank loss curve (full/FSDP2, full/DDP, LoRA/DDP) | agree; worst relative difference 1e-7 (full), within 5e-3 (LoRA, dropout RNG differs by split) |
| Resume after simulated preemption (FSDP2, stop at step 3) | continues the uninterrupted curve |
| Deliberately unsynced run | caught by the gradient-checksum check (test) |
| DPO on top of an SFT adapter; Ray Train path; `generate` output loads in `scoring` | pass |
| GPU-busy check | skipped on CPU by design; measured in the paid run |

Smoke profile table (CPU, 2 workers, 10M-parameter model; shows the code runs, not real numbers):
sft_full FSDP 402 tok/s/device, sft_lora DDP 542, dpo 375; LoRA trains 131k of 9.9M parameters.

**Deviations from the approved plan.**

- **DPO is an own loop, not TRL.** TRL 1.15 routes `DPOTrainer` through a fused LM-head kernel that
  needs Triton, so it cannot run on CPU and the stage would have reached the paid window untested.
  This is the fallback the plan named. `trl` and `accelerate` are not dependencies. Reference policy is
  the adapter-disabled model, as planned.
- **Max lengths set from the data.** The real tokenizer showed SFT rows up to 293 tokens (p99 210) and
  DPO pairs up to 442, so `max_length` is 320 (SFT) and 512 (DPO), not 256 and 320. A test fails on any
  row that does not fit. One SFT epoch is about 628k tokens.
- **Loss-coherence check redefined.** Each rank sees different examples, so its own-slice loss cannot
  match the mean to 1e-3. The check is: every rank observes the identical all-reduced loss (1e-6), and
  each local loss stays within a 0.75 band of it.
- **Batch size is global.** Config uses `global_batch_size` (64) with derived accumulation, and the loss
  is normalised by the global token count, so 1 and N ranks are the same training job.
- **FSDP2 master weights are fp32** (about 16 bytes/parameter, roughly 33 GB per GPU on 4x40 GB), not the
  12 bytes the compute plan assumed, to avoid bf16 update loss. This is tight on A100 40GB. Step 9 must
  check memory first and fall back to the compute plan's options (8-bit optimizer, partial unfreeze,
  80GB machine) if needed. Not testable before the paid window.
- **Resume state lives in the output directory** (`resume/`), not in Ray's checkpoint store. Step 8 must
  make it durable (bucket sync) or a spot preemption loses it with the VM.
- **Ray dashboard is off**, which makes Ray log harmless "State API unavailable" messages. Ray 2.59
  enables token auth by default.
- Rank 0 loads the model on every rank (about 4 x 32 GB host RAM for the fp32 full run; the machine has
  340 GB). Meta-device loading is a possible optimisation, not needed.

**Caveats.**

- Real tokens/sec, peak GPU memory, MFU, communication share and GPU utilisation are only measured in
  step 9. The measurement code runs and is tested, but its GPU branches (NVML sampling, CUDA memory,
  NCCL) are untested here.
- The torch profiler trace is written per rank next to the checkpoint.

**Spend.** $0. Nothing billable was created, so there is no teardown to confirm. Running total stays
$5.27 of the $20 target and $50 cap.

### Step 6 — GRPO stage (`feat/grpo`, free)

Plan approved 2026-10-13. Status: complete (see outcome and deviations).

#### Context

Steps 3 to 5 produced the frozen data (`data/training/grpo_prompts.jsonl`, 404 rows), the scorer
and fast reward (`scoring/reward.py`, `RewardScorer`), and the distributed SFT / LoRA / DPO code.
No RL code exists. This step writes and proves the GRPO stage that runs after DPO in the one paid
window (step 9): group rollouts, reward from step 4, DDP learner, KL to the DPO policy, and
rollout-throughput profiling. Code only, smoke-tested on the tiny model. No GPU rental, no GCP, no
API spend, no real checkpoint. **Cost: $0. Running total stays $5.27 of the $20 target and $50 cap.**
Nothing billable is created, so there is no teardown to confirm.

Separate from step 5 because rollouts are a different bottleneck (generation, not gradients) and a
different failure mode (reward hacking, collapse).

#### What exploration found

- `RewardScorer(cfg, chunk_text)` is a class (not a function). `.rewards(rows, completions)` gives one
  float in [-1, 1]; `RewardBreakdown` gives the components to log. Pure Python, runs inside any rank.
- 4 of 404 GRPO rows score below 0.9 on their own reference answer (frozen-data quirks, noted in
  step 4). The loader drops rows whose reference scores < 0.9 rather than editing frozen data.
- **Checkpoint chain gap.** DPO saves only a LoRA adapter trained on top of a merged SFT base, and
  `model.load_trained` merges an adapter onto the plain pretrained base. Loading a DPO adapter
  alone therefore silently drops SFT. GRPO needs the whole chain (SFT -> DPO -> GRPO).
- DPO has no resume; SFT's resume is built into `train_sft`. Under DDP, rollouts must call
  `net.module.generate` in eval mode with `use_cache=True` (gradient checkpointing sets it False).
- `profiling.py`'s docstring already reserves "step 6 adds rollout throughput to the same schema".

#### Decisions (defaults; change at approval)

1. **Rollout engine: Hugging Face `generate` under DDP**, own GRPO loop (same hybrid pattern as DPO:
   no TRL, no new dependencies). It runs on the CPU smoke test, so it is proven before the paid
   window. vLLM stays in the project for serving (steps 11 and 12); using it for rollouts too would
   add GPU-only code, per-step weight sync from the LoRA, and VRAM sharing on a 40 GB card, none
   testable before the meter starts. README names it as the road not taken for training.
2. **Scale: 16 prompts x 8 samples per step, ~88 steps** (4 epochs over ~360 training prompts;
   ~40 held out). 128 completions per step, ~450k generated tokens in total. Config values, so the
   step 8 rehearsal and step 9 can scale up or down.

#### Algorithm

- **Policy init** = merged DPO model (chain-loaded), fresh LoRA on top (r/alpha from `training.lora`,
  **LoRA dropout 0** for GRPO so rollout and learner probabilities match). **Reference** =
  adapter-disabled model, the same no-second-copy trick as DPO (matters at 40 GB).
- **Per step:** a global batch of 16 prompts is sharded across ranks (4 per rank, same
  `data.global_batches` / `rank_slice` helpers). Each rank samples G=8 completions per prompt
  (temperature 1.0, `max_new_tokens` 96; answers p99 is 42 words), scores them with
  `RewardScorer`, and computes group-normalised advantages `(r - mean) / (std + eps)`. A prompt's
  whole group lives on one rank, so advantages need no communication. Zero-variance groups
  contribute nothing and their share is logged.
- **Update:** one gradient step per rollout batch (on-policy), so the importance ratio is 1 and no
  old-policy pass or clipping is needed; this is stated rather than hidden. Loss = `-A * logp`
  per completion token plus `beta * KL(k3 estimator vs reference)`, beta 0.04, normalised by the
  **global** completion-token count (all-reduced) so 1 and N ranks are the same training job, as in
  SFT/DPO. Micro-batched with `parallel.accumulate`, grad clip, `lr_at` cosine schedule, AdamW on
  LoRA params, lr 1e-5.
- **Held-out eval:** ~40 prompts (stratified by kind, seeded) scored greedily at start, end, and every
  N steps. This is a training-signal curve only; the golden-set judge in step 9 is the real measure.
- **Resume:** adapters + optimizer via `parallel.save_resume_state(trainable_only=True)` every
  `checkpoint_every` steps plus a small `trainer_state.json` (step). Prompt order and per-step
  sampling seeds derive from (seed, step, rank), so resume is deterministic.

#### Design

Changes under `src/remember_or_retrieve/training/` unless noted:

- `grpo.py` (new): `load_prompts`, `rollout` (batched sampling, left-pad, trim at `<|im_end|>`,
  restores `use_cache`/train mode), `group_advantages`, `grpo_loss` (pure, unit-tested),
  `learn` (takes rollouts as input, so 1-vs-N equivalence is testable with injected fixed
  rollouts), `train_grpo` (loop, checks, profile, `run.json`).
- `model.py`: `load_trained` follows a chain. DPO and GRPO write a small `base.json`
  (`{"init_from": ...}`) beside their adapter; the loader recursively builds base -> merge -> merge.
  `dpo.py` gets a 3-line change to write it. Existing SFT/DPO behaviour unchanged.
- `data.py`: `encode_grpo_prompt` (reuses `tok.prompt_ids` with `SFT_SYSTEM_PROMPT`, same prompt
  as `generate.py`).
- `profiling.py`: backward-compatible `RankStats.rollout_s` / `rollout_tokens` / `reward_s`;
  `build_profile` adds a `rollout` block: generated tokens/s (global and per device),
  completions/s, rollout share of step time, reward time, mean and p95 completion length, learner
  tokens/s, peak memory per rank. Rollout throughput is the named RL bottleneck in `CLAUDE.md`.
- `checks.py`: reuse ranks, gradient/weight checksums, loss finite, GPU busy. Add
  `check_rollout_health` (rewards finite; at least some groups have non-zero variance; completions
  not all truncated or all identical).
- `launch.py`: `grpo` branch in `_ray_loop`; `smoke.py`: `grpo_job`.
- `scripts/train.py`: `grpo --init-from DIR` subcommand (stage name `grpo`, `cmd_stage` shape kept);
  `scripts/compare_stages.py`: rollout columns.
- `config/config.yaml`: `training.grpo` (group_size, prompts_per_step, epochs, beta, lr,
  temperature, max_new_tokens, heldout, eval_every, checkpoint_every), `training.strategy.grpo: ddp`.
  Smoke override: 2 prompts per rank, G=2, 4 steps.
- Logged per step into `run.json` and Ray report: mean reward and each `RewardBreakdown`
  component, KL, completion length, abstention rate on answerable vs unanswerable prompts,
  zero-variance group fraction, truncation rate.

#### Tests

- **Fast (CI `test` job, torch-gated like `test_training_loops.py`):** `group_advantages` (known
  values, zero-variance group is zero, mean zero per group); `grpo_loss` against a hand-computed
  case, KL non-negative and zero at the reference; prompt loader drops the 4 bad rows and the
  held-out split is deterministic and disjoint; `load_trained` chain resolution with a fake
  `base.json` layout; profile rollout block from synthetic stats; `check_rollout_health` fails
  on all-equal rewards / non-finite.
- **Slow smoke (`smoke-distributed` job, Linux, 2 workers, gloo):** GRPO runs end to end on the
  tiny model after SFT-LoRA -> DPO (exercises the chain); all distribution checks pass; learner
  with injected rollouts matches 1-rank vs 2-rank loss curve within `equivalence_rel_tol`;
  kill-and-resume continues; Ray Train path; `generate` loads the GRPO adapter and writes a valid
  answers file. Rollout sampling is per-rank random, so equivalence is claimed for the learner
  only, and the plan says so.
- `ruff check`, `ruff format --check` green.

#### Security

No credentials, endpoints, IAM, or GCP contact. No new dependencies (no `trl`, no `vllm`).
Same model-loading rules: `trust_remote_code=False`, pinned revision, safetensors only,
`torch.load(weights_only=True)`. The reward is offline and deterministic. Prompts are our own
frozen data, but completions are untrusted model output: they go only into the reward function and
JSON logs, never executed.

#### Branch and workflow

On approval: copy this plan into `docs/roadmap.md` under Step 6 and mark it in progress; branch
`feat/grpo`. Small commits: config + chain loading (with DPO `base.json`); prompts/data; advantages
and loss (pure); rollout; learner and loop; profiling and checks; launch + CLI; smoke tests + CI;
roadmap outcome. Deviations go into the roadmap entry. Nothing to `main` directly.

#### Verification

1. `pytest`, `ruff check`, `ruff format --check` green locally and in CI; smoke job green.
2. Docker smoke suite: GRPO passes all checks, SFT -> DPO -> GRPO chain loads the full stack, resume
   works, 1-vs-2 rank learner equivalence holds.
3. Deliberately broken run (one rank not syncing) fails the checks, shown once.
4. `profile.json` for the smoke GRPO run contains the rollout block; `compare_stages.py` renders it.
5. Reward sanity on the real prompts: a stub "policy" returning each reference answer averages
   >= 0.9; one returning the abstention phrase averages near 0 (guards against a reward that
   cannot separate good from bad before it costs GPU time).
6. Roadmap entry records plan, deviations, "Spend: $0, nothing billable, no teardown".

#### Cost carried forward (for step 9, estimated, not measured)

~88 steps at an estimated 6 to 15 s/step plus model load: **~0.2 to 0.5 GPU-hours, ~$1.8 to $4.4**
on 4x A100 40GB spot, at or below the compute plan's 0.4 to 0.8 hour line. If the step 8
rehearsal shows rollouts dominating, the cut order is fewer steps or a smaller group, never the
distributed setup.

#### Known limits

- Real rollout tokens/s, memory and the GPU branches (NCCL, `generate` on 8B with LoRA, NVML)
  are only measured in step 9. HF generate with an unmerged adapter is slower than merged
  weights or vLLM; it is measured, not assumed.
- Only ~360 distinct RL prompts (62 multi-hop in total), so the policy can overfit them; the golden
  set judge decides whether GRPO beat DPO, and the README reports it either way.
- The reward is a proxy (numeric facts, abstention, unsupported numbers, length), not the judge.
  Reward hacking is watched through the logged components, not assumed absent.
- On-policy single update means no PPO-style clipping; this is the intended GRPO simplification
  at this scale, not a PPO implementation.

#### Outcome and deviations from the plan (recorded 2026-10-13)

**Result.** `training/grpo.py` (rollouts, group advantages, loss, learner, loop), `training.grpo`
config, `base.json` adapter-chain loading in `model.py` (DPO and GRPO write it), a `rollout` block
in `profile.json`, `check_rollout_health`, a `grpo` CLI subcommand and Ray branch, and rollout
columns in `scripts/compare_stages.py`. No new dependencies. 112 tests pass in the default job;
all 14 distributed smoke tests pass in Docker (Linux, CPU, 2 Ray/gloo workers). Not yet run on
GitHub Actions itself.

**Verification against the plan.**

| Check | Result |
|---|---|
| SFT -> DPO -> GRPO on the tiny model, 2 ranks | all checks pass (ranks, gradients synced, weights identical, metrics identical, loss finite, rollout health); KL is 0 at step 0 |
| Adapter chain | `load_trained` on the DPO adapter differs from the same adapter without `base.json`, so SFT really is in the stack; the GRPO adapter records the DPO adapter as its base |
| Learner, fixed rollouts, 1 rank vs 2 | same loss, KL and gradient checksum (rel 1e-5 and 1e-4); gradient spread across ranks 0 |
| Resume after simulated preemption (stop at step 2) | continues the uninterrupted reward curve exactly (abs 1e-6) |
| Ray Train path, `generate` on the result | pass |
| Reward sanity on the real prompts | 400 of 404 rows kept (4 dropped, as predicted), 360 train / 40 held out (27 single-hop, 6 multi-hop, 7 unanswerable); reference answers average reward 1.0; abstain-everything scores -0.5 on answerable and +1.0 on unanswerable prompts |
| Deliberately unsynced run | caught by the existing gradient-checksum test |
| GPU-busy, NCCL, 8B `generate` with LoRA | not testable here; measured in step 9 |

Smoke profile (CPU, 2 workers, 10M-parameter model; shows the code runs, not real numbers): 244
generated tok/s/device, rollout 8% of step time.

**Deviations from the approved plan.**

- **The smoke runs use a synthetic reward** (`smoke.SyntheticReward`, a CRC of the text, passed as
  `reward_fn`). A randomly initialised model never produces a correct fact, so the project's real
  reward is flat for it and `check_rollout_health` correctly fails. The synthetic reward exercises
  advantages, learner and checks with a signal; the real reward is covered by step 4's tests and the
  sanity numbers above.
- **`check_rollout_health` takes `require_stops`** and the smoke run turns it off: a random tiny
  model never emits `<|im_end|>`, so every completion is truncated. Real runs require that not all
  completions are cut off.
- **A latent bug from step 5 was fixed: LoRA resume.** Loading a trainable-only resume checkpoint
  demanded the frozen base keys (`strict` default). Step 5 only tested full-FSDP resume. The options
  now use `strict=False` when only trainable parameters are saved. SFT-LoRA resume benefits too.
- **`loss_coherent` is not used for GRPO.** Group advantages sum to zero, so the loss hovers near
  zero and a relative per-rank band is meaningless. Replaced by `metrics_identical` (every rank saw
  the same reward curve) plus the learner equivalence test.
- **Held-out reward is logged at step 0, every 20 steps and at the end** (`heldout_reward` in
  `run.json`), greedy, 40 prompts.
- **Rollout engine and scale were the plan defaults** (HF `generate`; 16 prompts x 8, ~88 steps).
  The user asked what vLLM would otherwise be used for: it stays in the project for serving
  (steps 11 and 12).

**Caveats.**

- **Rollout throughput, GPU memory and NCCL behaviour are unmeasured.** The estimate in the plan
  (0.2 to 0.5 GPU-hours, about $1.8 to $4.4) is a guess until step 8's rehearsal and step 9.
  HF `generate` with an unmerged LoRA is the slowest reasonable engine; if it dominates, the cut
  order is fewer steps or a smaller group.
- **Only 360 distinct RL prompts (about 62 multi-hop in total before the split).** The golden-set
  judge, not the training reward, decides whether GRPO beat DPO.
- **Resume state lives in the output dir**, as for SFT. Step 8 must make it durable.

**Spend.** $0. Nothing billable was created, so there is no teardown to confirm. Running total
stays $5.27 of the $20 target and $50 cap.
