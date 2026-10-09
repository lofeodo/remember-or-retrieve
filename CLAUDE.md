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
| Infrastructure as code | Terraform, covering everything that costs money or takes more than a couple of commands to stand up: the GKE cluster and its GPU node pool, the storage bucket for checkpoints, and the model registry if it needs its own resources. This also doubles as a budget safeguard: a `terraform destroy` after each paid step guarantees nothing paid is left running by accident, which matters more here than in the other two projects given how little room this budget has for a provisioning mistake. |
| Corpus and eval set | Reused as-is from `rag-with-receipts`: the 955-chunk OSRS corpus and the 55-question golden set (35 single-hop, 15 multi-hop, 5 unanswerable). No changes to either without a documented reason. |
| Base model | **No longer capped at 1-8B.** Since training now happens once, distributed, on rented GCP GPUs rather than locally, pick the largest instruct model that fits comfortably inside the training-run cost estimate below (realistically something in the 13-34B range, depending on what the costed GPU plan actually supports) rather than defaulting to a small model out of local-hardware habit. The exact size is decided when this step is planned, against a real cost estimate, not assumed here. |
| Training data | Question/answer pairs generated from the corpus chunks using Claude at build time only (never at inference time), the same "LLM at training time, zero at inference" split as the clinical guideline compiler project. Held strictly separate from the 55-question golden set, which is eval-only and never used for training. |
| Training infrastructure | **One fine-tuning run, not two.** All training (SFT, LoRA, DPO, RL) happens on rented GCP GPUs, provisioned via Terraform, and it must be genuinely distributed, not just "running on a machine that happens to have several GPUs." Renting a multi-GPU instance and running an ordinary single-process training script on it is **not** distributed training, even if multiple GPUs are present and billed. The run has to actually use a parallelization strategy that splits the work across devices and synchronizes it: data parallelism (PyTorch DDP, with gradient all-reduce) and, if the chosen model is large enough that it doesn't comfortably fit on one GPU for full fine-tuning, model/parameter sharding (FSDP or DeepSpeed ZeRO) on top of or instead of DDP. Which of these applies depends on the model size picked in this step, and should be decided together with it, not as an afterthought. Before trusting the run, verify it's actually distributed the way Step 6 of `rag-with-receipts` verified its NLI label order empirically rather than assuming it: confirm multiple ranks/processes actually started, that GPU utilization shows up on every device during training (not just one), and that the loss curve looks like one coherent run rather than several independent ones. There is no separate local run and no separate disconnected "distributed training benchmark" for its own sake: the distributed run itself produces the real checkpoints that go on to the registry and serving, and its own timing/throughput numbers (vs. a quick single-GPU reference run on the same hardware tier, if that's affordable within the budget) are reported as a real measurement. This is the project's biggest cost item, so the GPU count, tier, parallelization strategy, and expected training wall-clock time must all be estimated and approved before provisioning, per the budget row above. |
| Distributed training framework | **Ray** (Ray Train) orchestrating the PyTorch DDP/FSDP job across the rented GPUs, rather than a bare `torchrun` launch script. This is a deliberate choice, not just "use PyTorch's own launcher": Ray is the orchestration framework named explicitly in the kind of post-training role this project is aimed at, and using it (rather than only the lower-level PyTorch primitives) is itself a signal worth having in the README and the code, not just in a bullet point. **Slurm**, the other infrastructure named in those roles, is a deliberate gap this project does not close: standing up a real Slurm cluster is HPC-scheduler infrastructure disproportionate to a single-tenant budget project, and faking it would be worse than naming the gap plainly. Say this openly in the README rather than quietly using Ray and hoping nobody asks about Slurm. |
| Framework coverage (JAX/XLA) | **Open question, not decided here.** The training stack above is PyTorch end to end (DDP/FSDP, Ray Train, Hugging Face TRL for SFT/LoRA/DPO/GRPO). Some post-training roles also name JAX and XLA/MLIR explicitly. Adding a real JAX-based component (even a small one, like reimplementing the RL reward scoring or one training stage in JAX) is possible but is genuinely extra scope and extra cost, not a small add-on, and would need its own planned step and its own cost estimate against the budget. Flagging this here rather than silently deciding either way: if JAX exposure matters enough to be worth the added scope, say so before the training step is planned so it can be designed in from the start rather than retrofitted. |
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

## Candidate step breakdown (starting point for the real roadmap)

1. **Repo scaffold and this file.**
2. **Training data generation** — Claude-generated Q/A pairs from the corpus chunks, kept
   strictly separate from the golden eval set.
3. **Terraform for the training infrastructure** — the rented/distributed GCP GPU setup
   (instance type, count, networking) codified before any training happens, so it can be
   provisioned and torn down with one command and the run doesn't idle on the clock.
4. **Distributed SFT, LoRA, and DPO — one costed training run** — base model picked and
   its cost estimated against the budget, then SFT, the LoRA-vs-full comparison, and DPO
   all run within that same provisioned GPU window, orchestrated with Ray Train over
   PyTorch DDP/FSDP, with real single- vs multi-GPU throughput and memory numbers
   captured as part of this run rather than as a separate benchmark.
5. **RL post-training (GRPO)** — the reward signal built from the project's own
   correctness and grounding scorers rather than a separately trained reward model,
   group rollouts sized tightly against the budget, rollout throughput profiled
   specifically since that's the likely bottleneck. Runs in the same provisioned GPU
   window as step 4, torn down once this step is also done.
6. **Model registry** — every checkpoint (including the RL one) logged with its eval
   score as it's produced, not retrofitted at the end.
7. **Terraform for the serving infrastructure** — the GKE cluster, GPU node pool, and
   checkpoint storage bucket, codified separately from step 3's training infra since the
   two are provisioned and torn down at different times.
8. **Serving on GKE** — the winning checkpoint behind a GPU-backed deployment,
   provisioned via step 7's Terraform, benchmarked, then torn down.
9. **Comparison harness** — running the same 55-question golden set through the
   fine-tuned model, the live `rag-with-receipts` API, and the Mistral and Cohere
   off-the-shelf baselines, all scored the same way.
10. **User-facing interface** — the side-by-side demo, hosted on GCP, using recorded
    comparison results rather than a live GPU endpoint if keeping one running would
    exceed the budget.
11. **README polish** — see below.

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
