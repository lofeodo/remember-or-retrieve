# CLAUDE.md — Remember or Retrieve

Portfolio project. Fine-tune an open model (SFT, LoRA, DPO), trained once via distributed
training on rented GCP GPUs, to answer questions about the same Old School RuneScape Wiki
corpus used in `rag-with-receipts`, then
benchmark it head to head against that project's RAG pipeline on the same 55-question
golden set. The project's name is the question it answers: does the model remember the
answer (it's baked into the fine-tuned weights) or does it retrieve it (looked up at
query time)? The point of the project is the comparison itself: report honestly where
fine-tuning wins and where RAG wins, not "fine-tuning is better." Deployed with a public,
presentable demo on GCP. This file is the project's single source of truth: a roadmap of
steps, each with its own approved plan recorded in place, kept in sync with the real state
of the codebase at all times.

## Why this project exists

`rag-with-receipts` already proved retrieval-augmented generation on this corpus,
rigorously. This project asks the natural follow-up question a RAG project invites: when
would you fine-tune instead, and what do you actually give up or gain by doing so? Reusing
the same corpus and the same golden evaluation set means the comparison is apples to apples,
not two unrelated benchmarks compared informally.

This project is also deliberately shaped to demonstrate post-training and distributed
training engineering specifically, the kind of work described in model post-training and
applied AI engineering roles at frontier labs: writing scalable training software, using
real distributed training infrastructure and frameworks (not just a bigger single machine),
running the full post-training stack through an RL stage rather than stopping at SFT/DPO,
and measuring and reporting performance (throughput, memory, cost) rather than only
accuracy. Every decision below that looks like "more than strictly necessary for a
portfolio RAG comparison" (the RL stage, the framework choice, the profiling) is there for
that reason, not scope creep for its own sake.

## Locked decisions

| Area | Decision |
|---|---|
| Budget | **$20 is the target, $50 is the hard cap.** Lower is always better than higher, and $20 is what to aim for when sizing the model and the GPU plan, but spending up to $50 is acceptable if that's what a genuinely distributed run and a short GKE serving window actually cost. Every paid resource (GPU rental, a GKE node pool, Vertex AI Model Registry, anything with a dollar cost) must have its estimated cost stated and approved before it's provisioned, running total against both the $20 target and the $50 ceiling, and torn down immediately once its measurement is done. There is no local-training fallback in this version of the plan (see Base model and Training infrastructure below), so essentially the whole training budget comes out of one deliberately bounded GCP GPU window — pick the model size and GPU count against this constraint explicitly, not assumed. If the training run and the GKE serving step together would cross $50, cut the GKE serving step down to the shortest possible live window before cutting the model size or faking the distributed-training setup, since both of those are the actual point of this change. |
| GCP project | **Decided 2026-10-09:** a new, dedicated `remember-or-retrieve` project under the `daniel.lofeodo@gmail.com` account, not the existing `rag-with-receipts` project, so spend is cleanly attributable to this budget and a `terraform destroy` here can never touch the live RAG service. The development machine's default gcloud login is a different account (`latentspacemail@gmail.com`) used for unrelated work in parallel: do not switch the default account. Use a separate named gcloud configuration for this project, and only create it when step 2 is underway and the user says so. |
| Infrastructure as code | Terraform, covering everything that costs money or takes more than a couple of commands to stand up: the GKE cluster and its GPU node pool, the storage bucket for checkpoints, and the model registry if it needs its own resources. This also doubles as a budget safeguard: a `terraform destroy` after each paid step guarantees nothing paid is left running by accident, which matters more here than in the other two projects given how little room this budget has for a provisioning mistake. |
| Corpus and eval set | Reused as-is from `rag-with-receipts`: the 955-chunk OSRS corpus and the 55-question golden set (35 single-hop, 15 multi-hop, 5 unanswerable). No changes to either without a documented reason. |
| Base model | **No longer capped at 1-8B.** Since training now happens once, distributed, on rented GCP GPUs rather than locally, pick the largest instruct model that fits comfortably inside the training-run cost estimate below (realistically something in the 13-34B range, depending on what the costed GPU plan actually supports) rather than defaulting to a small model out of local-hardware habit. The exact size is decided when this step is planned, against a real cost estimate, not assumed here. |
| Training data | Question/answer pairs generated from the corpus chunks using Claude at build time only (never at inference time), the same "LLM at training time, zero at inference" split as the clinical guideline compiler project. Held strictly separate from the 55-question golden set, which is eval-only and never used for training. |
| Training infrastructure | **One fine-tuning run, not two.** All training (SFT, LoRA, DPO, RL) happens on rented GCP GPUs, provisioned via Terraform, and it must be genuinely distributed, not just "running on a machine that happens to have several GPUs." Renting a multi-GPU instance and running an ordinary single-process training script on it is **not** distributed training, even if multiple GPUs are present and billed. The run has to actually use a parallelization strategy that splits the work across devices and synchronizes it: data parallelism (PyTorch DDP, with gradient all-reduce) and, if the chosen model is large enough that it doesn't comfortably fit on one GPU for full fine-tuning, model/parameter sharding (FSDP or DeepSpeed ZeRO) on top of or instead of DDP. Which of these applies depends on the model size picked in this step, and should be decided together with it, not as an afterthought. Before trusting the run, verify it's actually distributed the way Step 6 of `rag-with-receipts` verified its NLI label order empirically rather than assuming it: confirm multiple ranks/processes actually started, that GPU utilization shows up on every device during training (not just one), and that the loss curve looks like one coherent run rather than several independent ones. There is no separate local run and no separate disconnected "distributed training benchmark" for its own sake: the distributed run itself produces the real checkpoints that go on to the registry and serving, and its own timing/throughput numbers (vs. a quick single-GPU reference run on the same hardware tier, if that's affordable within the budget) are reported as a real measurement. This is the project's biggest cost item, so the GPU count, tier, parallelization strategy, and expected training wall-clock time must all be estimated and approved before provisioning, per the budget row above. |
| Distributed training framework | **Ray** (Ray Train) orchestrating the PyTorch DDP/FSDP job across the rented GPUs, rather than a bare `torchrun` launch script. This is a deliberate choice, not just "use PyTorch's own launcher": Ray is the orchestration framework named explicitly in the kind of post-training role this project is aimed at, and using it (rather than only the lower-level PyTorch primitives) is itself a signal worth having in the README and the code, not just in a bullet point. **Slurm**, the other infrastructure named in those roles, is a deliberate gap this project does not close: standing up a real Slurm cluster is HPC-scheduler infrastructure disproportionate to a single-tenant budget project, and faking it would be worse than naming the gap plainly. Say this openly in the README rather than quietly using Ray and hoping nobody asks about Slurm. |
| Framework coverage (JAX/XLA) | **Decided 2026-10-09: add JAX, on rented GPUs.** The main training stack stays PyTorch end to end (DDP/FSDP, Ray Train, Hugging Face TRL for SFT/LoRA/DPO/GRPO). On top of it, one real JAX component: a JAX LoRA fine-tune of a small model (2B to 4B class, most likely a Gemma, which has a maintained JAX implementation) on the same training data, run in the same rented GPU window as the PyTorch stages, split across the GPUs with JAX's own multi-device data parallelism, XLA-compiled, and profiled against the PyTorch SFT stage. It is a smaller model than the main one because there is no cheap, maintained JAX path for loading a 13B+ instruct model's pretrained weights; say so in the README rather than implying parity. The exact model is verified when the step is planned, not assumed here. Estimated cost: $1 to $3 on spot GPUs, $3 to $8 on demand (roughly 20 to 40 extra minutes of the training window, no second provisioning). It has its own code step (step 7) and runs inside the training run (step 9). |
| Performance profiling | Every training stage (SFT, LoRA/full comparison, DPO, RL) is profiled, not just timed end to end: tokens/sec, memory per device, and for the RL stage specifically, rollout generation throughput (since that's usually the actual bottleneck in RL post-training, not the gradient step). This is a direct answer to "performance optimization" being named explicitly in the target roles, and it reuses the same measured, honestly-reported discipline as `rag-with-receipts`'s per-stage latency work rather than introducing a new style of reporting. |
| Fine-tuning methods | SFT first (establishes the baseline capability on the larger model), then LoRA specifically compared against a fuller fine-tune (the point is to be able to speak to the LoRA-vs-full tradeoff from a real measurement, not just the term), then DPO on top, using preference pairs built to specifically target hallucination and verbosity, the same failure modes the RAG project's grounding checker measures. **Then an RL post-training stage on top of DPO** (see the row below) — the project should not stop at SFT/DPO, since going through an actual RL regime is specifically what separates "fine-tuned a model" from "did post-training the way a frontier lab does it." All four stages happen within the same rented GPU window rather than across separate sessions, to avoid paying to re-provision multiple times. |
| RL post-training | **GRPO (Group Relative Policy Optimization), not PPO, as the default choice.** GRPO needs no separate learned reward model or value network, which matters a lot at this budget: it generates a group of candidate completions per prompt and scores them directly, so the reward signal can be the project's own existing scorers (correctness against the training-time Q/A pairs, plus the grounding/hallucination checker's signal from `rag-with-receipts`, reused rather than reinvented) instead of training a whole separate reward model from scratch. PPO is the textbook alternative and worth naming in the README as the road not taken, with the real reason (reward-model cost, more moving parts, more rollouts) stated plainly rather than hand-waved. RL rollouts are the most compute-hungry part of the whole project (multiple generations per prompt, every stage), so this step's prompt count and group size must be scoped tightly and costed explicitly before it runs, smaller and cheaper than it would be in a non-budget-constrained setting, and that tradeoff should be stated openly rather than hidden. |
| Model registry | Every checkpoint (base, SFT, LoRA, DPO, RL) logged with its golden-set eval score, not just the final one. MLflow or Vertex AI Model Registry (prefer Vertex, since the project is already on GCP and it avoids standing up a separate MLflow server). The best-performing checkpoint is explicitly promoted; the others stay in the registry as a record, not deleted. |
| Serving | The winning checkpoint served via vLLM or TGI behind a GPU-backed deployment on GKE (a real node pool with GPU scheduling and autoscaling), not Cloud Run — this is the project's specific vehicle for genuine Kubernetes experience, so it should not be quietly replaced with something simpler partway through. **This is the other real cost against the budget** — a GPU node pool bills by the hour whether or not it's being queried, so plan the smallest GPU type that can serve the chosen model, provision with Terraform, run the comparison harness (step 9) immediately, and destroy the node pool right after rather than leaving it up for the demo to poll live. If keeping a live GPU endpoint running for the public demo would push the total past $50, the user-facing interface should call a cached/recorded comparison instead of a live GPU endpoint, and this tradeoff should be raised explicitly in that step's plan rather than decided silently. |
| Comparison baseline | The already-deployed `rag-with-receipts` Cloud Run service, called over its real API, not reimplemented or mocked. **Also benchmark a Mistral model and a Cohere model** on the same 55-question golden set, called through their own APIs as off-the-shelf baselines (no fine-tuning on either, to keep the budget low — API calls for 55 questions are cheap, GPU rental for two more fine-tuning runs would not be). This widens the comparison beyond just the home-grown fine-tune vs. Anthropic-backed RAG, and shows familiarity with more than one model provider. *Assumption flagged: if either model was meant as a third fine-tuning target rather than an off-the-shelf comparison point, say so before this step is planned — fine-tuning a second or third base model would need its own cost and scope discussion against the budget cap.* |
| User-facing interface | A small, clean web frontend, hosted on GCP, that lets a visitor ask a question and see both answers side by side (fine-tuned model vs RAG pipeline) along with which one the eval says tends to win on that question type. Public, presentable, no login required. |
| Final step | A rewritten README aimed at both technical and non-technical visitors (recruiters as well as engineers): short sections, no long paragraphs, visual wherever possible (Mermaid diagrams, badges, charts, a banner), everything skimmable at a glance. See "README bar" below. |

## Workflow rules — read this before writing any code

This project is run in discrete, user-approved steps. Do not skip ahead of this process,
even if a later step seems obvious or quick.

1. **Before any code is written**, produce a roadmap: a numbered list of steps that
   together cover the whole project (data generation through the README), each with a
   one or two sentence description of what it accomplishes and why it is its own step
   rather than folded into another one. Each roadmap step corresponds 1:1 to a feature
   branch. Present this roadmap to the user before doing anything else. The "Candidate
   step breakdown" below is a starting point for this roadmap, not the roadmap itself —
   refine, reorder, split, or merge it as the real design calls for, and say so.
   *(Done 2026-10-09: the approved roadmap is the "Roadmap" section below, which replaced
   the candidate breakdown and records what changed from it.)*
2. **To start a step:** the user says so explicitly. Enter plan mode. Write a detailed,
   in-depth design and implementation plan for that step, and that step only — do not
   plan ahead into future steps. Present the plan for the user's approval.
3. **Once approved:** copy the approved plan, in full, into the roadmap document under
   that step's entry, so the roadmap carries both the checklist and the actual design
   record for every step that has been planned. This is the project's paper trail —
   treat it as something a reader could reconstruct the whole project's history from.
4. **Each step is implemented on its own feature branch** (`feat/<step-name>`), with
   small, continual commits rather than one large batched push.
5. **While a step is in progress**, keep its entry in the roadmap current. If the real
   implementation deviates from the approved plan (a library doesn't do what was
   expected, a number comes back different from what was assumed, a scope decision
   changes), update the roadmap's record for that step to say so, the same way
   `rag-with-receipts`'s CLAUDE.md documents real deviations rather than quietly
   overwriting the original plan. The roadmap must always be a correct, current
   description of what is actually in the codebase, never just a snapshot of what was
   originally intended.
6. **When a step is fully complete**, check it off in the roadmap and tell the user
   explicitly before starting the next step's plan. Do not start implementing the next
   step on your own initiative.
7. **The user will then prompt you to begin planning the next step**, and the cycle
   repeats from rule 2.
8. Never implement a step that has not gone through this plan-then-approve cycle, even a
   small one. If a step turns out to be trivial, say so in the plan rather than skipping
   the plan.
9. **Any step that will spend real money** must state its estimated dollar cost in the
   plan, running total against both the $20 target and the $50 ceiling included, before
   the user approves it. After
   that step finishes, confirm in the roadmap that the paid resource was torn down
   (mirror the explicit teardown confirmations in `rag-with-receipts`'s CLAUDE.md), and
   never leave a billable resource running "just in case" between steps.

## Roadmap

Approved 2026-10-09. This replaces the original "Candidate step breakdown" (11 steps),
which was a starting point only. Each step is one feature branch and goes through its own
plan-then-approve cycle (workflow rules 2 to 8) before any of it is implemented. "Free"
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

### What each step is, and why it is its own step

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
   data (see the Framework coverage row in Locked decisions). Code only, smoke-tested.
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
14. **README.** To the README bar below, including the Slurm gap and PPO as the road not
    taken.

### Changes from the candidate breakdown

| Change | Reason |
|---|---|
| New step 2: compute plan, GCP bootstrap, GPU quota | A new GCP project starts with zero GPU quota and approval can take days. The request needs the GPU type, which needs the model size. So the model and GPU decision moves ahead of training code instead of being made when training is planned. |
| New step 4: scoring harness | Every checkpoint must be scored on the golden set as it is produced, and GRPO's reward reuses the same scorers. Both need the harness to exist before the GPU window. |
| Training split into code (5, 6, 7) and one paid run (9) | All stages share one GPU window. Every stage has to be written and smoke-tested on a tiny model before the meter starts. |
| Registry merged into training infra (8) | "Logged as it is produced" means the registry must exist before the run. |
| Comparison harness (10) moved ahead of GKE serving (12) | The harness must be ready to fire the moment the GPU node pool is up. The three API baselines get scored for real before any GKE cost. |
| New step 7: JAX stage | JAX coverage was decided in favor on 2026-10-09 (see Locked decisions). |

### Budget envelope (provisional)

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

### Known risks

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


## README bar (for the final step)

The audience includes people who will not read code and have limited time. Every section
should be skimmable in a few seconds. Concretely:

- No paragraph longer than two or three sentences.
- A Mermaid diagram for the pipeline and for the fine-tuning stages (SFT → LoRA/full →
  DPO → RL), not a wall of text describing them.
- Badges for stack and status (Python version, model, Ray, GKE, license, live demo link).
- A results table or small chart for the head-to-head comparison, not prose describing
  the numbers.
- A short banner or header image if one can be made cleanly; skip it rather than force a
  bad one.
- The honest headline finding (where fine-tuning wins, where RAG wins) stated in one or
  two lines near the top, not buried at the bottom after the methodology.
