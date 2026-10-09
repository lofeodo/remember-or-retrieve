# Compute plan

Decision record for step 2. Locks the model, GPUs, parallelism strategy, and cost estimate
that steps 3 to 12 build against. Prices are us-central1 USD/hr from a third-party mirror
of GCP's catalog (gcloud-compute.com; GCP's own pricing pages are JavaScript-rendered and
could not be fetched). They are re-verified against the Cloud Billing Catalog once the
project exists.

## Decision

| Item | Choice |
|---|---|
| Base model | `Qwen/Qwen3-8B` (8.2B, Apache-2.0, post-trained, `transformers>=4.51`) |
| Training machine | `a2-highgpu-4g`: 4x A100 40GB, spot ~$8.82/hr (on-demand ~$14.69) |
| Full fine-tune | PyTorch FSDP full-shard across the 4 GPUs, via Ray Train |
| LoRA, DPO, GRPO | DDP (base fits on one 40GB GPU), via Ray Train |
| JAX stage | Gemma 2B to 4B class on the same 4 GPUs (model chosen in step 7; Gemma is gated on Hugging Face and needs a license-accepted token) |
| Region | us-central1 first, then us-east1 and europe-west4 |

## Why 8B, not the 13B to 34B in CLAUDE.md

Priced against real spot rates, a 14B full fine-tune needs ~220 GB with optimizer state,
which means 4x A100 80GB (`a2-ultragpu-4g`, spot ~$12.17/hr). That puts the expected
project total near $44 and the high case above the $50 cap. 8B needs ~130 GB in fp32
states, or ~100 GB with bf16 params/grads and fp32 Adam states, which fits 4x40GB
(160 GB) and keeps the high case under the cap. The user chose 8B on 2026-10-09 and
`CLAUDE.md`'s Base model row is amended to say so.

## Memory plan for the 8B full fine-tune

- FSDP full-shard, bf16 params and grads, fp32 Adam states: ~12 bytes/param, ~98 GB
  total, ~25 GB per GPU before activations.
- Gradient checkpointing and a small micro-batch to fit activations in the remaining ~15 GB.
- If it still does not fit: 8-bit optimizer states, then a defined partial unfreeze (and
  say so), then the 80GB fallback machine.

## GPU-hours and cost (4x A100 40GB, spot ~$8.82/hr)

| Stage | Low h | High h |
|---|---|---|
| Provision, setup, distribution checks (ranks, every GPU busy, one loss curve) | 0.25 | 0.5 |
| Single-GPU reference run on the same tier | 0.15 | 0.3 |
| SFT, full fine-tune (FSDP) | 0.2 | 0.4 |
| SFT, LoRA (DDP) | 0.15 | 0.3 |
| DPO | 0.2 | 0.4 |
| GRPO (small: prompt count x group size set in step 6) | 0.4 | 0.8 |
| JAX LoRA stage | 0.33 | 0.67 |
| Scoring every checkpoint on the golden set | 0.15 | 0.3 |
| Buffer, including one spot preemption restart | 0.1 | 0.4 |
| **Total hours** | **~1.9** | **~4.1** |
| **Training window cost** | **~$17** | **~$36** |

Expected window ~2.8 hr, ~$25. On-demand for the same window would be ~$28 to ~$60, so
the run uses spot with checkpoints to the bucket, and on-demand only as a deliberate,
costed fallback.

## Whole-project estimate

| Item | Low | Expected | High |
|---|---|---|---|
| Claude API (data generation, judging) | $3 | $5 | $7 |
| Training window (above) | $17 | $25 | $36 |
| Infra rehearsal (step 8, a few minutes on 4x L4 spot, ~$2.40/hr) | $0.5 | $0.75 | $1 |
| Mistral and Cohere API baselines | $1 | $1.5 | $2 |
| GKE serving window (step 12; 8B bf16 is ~16 GB, fits one L4, price unverified) | $1 | $2 | $3 |
| **Total** | **~$23** | **~$34** | **~$49** |

Against the budget: the $20 target is not reachable with a genuinely distributed run at
these prices (low case ~$23); the $50 cap holds in every case, with thin margin in the
high case.

## Cut order if spend runs ahead of the estimate

1. Shorten the GKE serving window (record answers for the demo, as CLAUDE.md allows).
2. Shrink GRPO (fewer prompts, smaller group).
3. Drop the single-GPU reference to a few steps rather than a full run.

Never cut: the distributed setup, or the distribution checks.

## Fallback tiers if A100 40GB quota is denied

| Tier | Machine | Spot $/hr | Consequence |
|---|---|---|---|
| A100 80GB | `a2-ultragpu-4g` | ~12.17 | Same plan, window ~$23 to ~$50; expected total ~$43 |
| 8x L4 24GB | 2x `g2-standard-48` or `g2-standard-96` | ~2.4 per 4 GPUs | LoRA/DDP only; full fine-tune infeasible; LoRA-vs-full becomes a smaller-model measurement |
| H100 | `a3-highgpu-8g` | ~52.96 | Too expensive for this budget; not used |

## Quota to request (part B)

- Spot A100 40GB: 4 GPUs (ask for 8 for headroom) in each of us-central1, us-east1,
  europe-west4, plus matching vCPU quota (48 per 4-GPU machine).
- One serving GPU (L4) for GKE.
- New projects on Gmail accounts are sometimes denied; if so, upgrading to a paid billing
  account is the usual fix and is raised with the user rather than done automatically.
