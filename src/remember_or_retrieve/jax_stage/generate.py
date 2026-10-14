"""Answer the golden questions with a trained adapter, in the scoring answers format.

Greedy decoding with the exact prompt SFT trained on (system prompt, Gemma 4 chat format). The
decode loop is deliberately simple: a fixed-width token buffer re-run through the model each step
(no KV cache), so XLA compiles exactly one program. That is slow per token but there are only 55
short questions, and it keeps the stage's generation path small enough to verify.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from remember_or_retrieve.data import load_golden_set
from remember_or_retrieve.datagen.prompts import SFT_SYSTEM_PROMPT
from remember_or_retrieve.jax_stage import model as model_lib
from remember_or_retrieve.jax_stage import train as train_lib
from remember_or_retrieve.jax_stage.tokenization import load_tokenizer
from remember_or_retrieve.scoring.schema import SystemAnswer, write_answers


def greedy_decode(
    lm,
    variables,
    prompts: list[list[int]],
    *,
    max_new: int,
    end_id: int,
    pad_id: int,
    width_multiple: int = 32,
) -> list[list[int]]:
    """Greedy continuation of each prompt (left-padded to a common width), up to `end_id`."""
    prompt_w = train_lib.pad_to_multiple(max(len(p) for p in prompts), width_multiple)
    total = prompt_w + max_new
    buf = np.full((len(prompts), total), pad_id, np.int32)
    for i, p in enumerate(prompts):
        buf[i, prompt_w - len(p) : prompt_w] = p

    @jax.jit
    def next_token(variables, tokens, pos):
        logits = lm.apply(variables, tokens=tokens).logits
        return jnp.argmax(logits[:, pos, :], axis=-1)

    done = np.zeros(len(prompts), bool)
    out: list[list[int]] = [[] for _ in prompts]
    for t in range(max_new):
        nxt = np.asarray(next_token(variables, jnp.asarray(buf), prompt_w + t - 1))
        for i in range(len(prompts)):
            if done[i]:
                continue
            if int(nxt[i]) == end_id:
                done[i] = True
            else:
                out[i].append(int(nxt[i]))
        buf[:, prompt_w + t] = np.where(done, pad_id, nxt)
        if done.all():
            break
    return out


def answer_questions(
    lm, variables, tok, questions: list[dict], *, max_new: int = 96, batch_size: int = 8
) -> list[SystemAnswer]:
    answers: list[SystemAnswer] = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i : i + batch_size]
        prompts = [
            tok.prompt_ids(
                [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": q["question"]},
                ]
            )
            for q in chunk
        ]
        t0 = time.perf_counter()
        outs = greedy_decode(
            lm,
            variables,
            prompts,
            max_new=max_new,
            end_id=tok.end_token_id,
            pad_id=tok.pad_token_id,
        )
        latency = (time.perf_counter() - t0) / len(chunk)
        for q, p, ids in zip(chunk, prompts, outs, strict=True):
            answers.append(
                SystemAnswer(
                    question_id=q["id"],
                    answer=tok.decode(ids),
                    latency_s=latency,
                    input_tokens=len(p),
                    output_tokens=len(ids),
                    cost_usd=None,
                )
            )
    return answers


def run(cfg: dict, adapter_dir: Path, out_path: Path, *, smoke: bool = False, limit=None) -> int:
    """Load base + adapter, answer the golden set, write the scoring answers file."""
    jcfg = cfg["jax"]
    dtype = jnp.float32 if smoke else model_lib.DTYPES[jcfg["model"]["dtype"]]
    tok = load_tokenizer(cfg)
    lm, base_model = model_lib.build_model(cfg, smoke=smoke)
    seed = jcfg["sft"]["seed"]
    if smoke:
        base = model_lib.init_base_random(base_model, seed, dtype=dtype)
    else:
        base = model_lib.load_base_checkpoint(cfg, dtype)
    target = model_lib.init_lora(lm, seed)
    lora = train_lib._checkpointer().restore(Path(adapter_dir).resolve() / "adapter", target)
    variables = model_lib.merge(base, lora, dtype)
    questions = [{"id": q.id, "question": q.question} for q in load_golden_set()]
    if limit:
        questions = questions[:limit]
    answers = answer_questions(lm, variables, tok, questions)
    write_answers(out_path, answers)
    return len(answers)
