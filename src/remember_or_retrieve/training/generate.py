"""Answer the golden questions with a trained checkpoint, in the scoring answers format.

Greedy decoding with the exact prompt SFT trained on (system prompt, thinking-disabled template),
so each checkpoint's output is scoreable by `scripts/score.py score` the moment it exists.
"""

import time

import torch

from remember_or_retrieve.datagen.prompts import SFT_SYSTEM_PROMPT
from remember_or_retrieve.scoring.schema import SystemAnswer
from remember_or_retrieve.training import data


def generate_answers(
    model, tok, questions: list[dict], device, max_new_tokens: int = 96, batch_size: int = 16
) -> list[SystemAnswer]:
    """`questions` are golden-set items (`id`, `question`). Latency is per batch, split evenly."""
    model.to(device).eval()
    im_end = tok.tok.convert_tokens_to_ids(data.IM_END)
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
        width = max(len(p) for p in prompts)
        ids = [[tok.pad_token_id] * (width - len(p)) + p for p in prompts]  # left-pad
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(mask, device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=im_end,
                pad_token_id=tok.pad_token_id,
            )
        latency = (time.perf_counter() - t0) / len(chunk)
        for q, p, seq in zip(chunk, prompts, out[:, width:].tolist(), strict=True):
            n_out = next((j for j, t in enumerate(seq) if t == im_end), len(seq))
            text = tok.tok.decode(seq[:n_out], skip_special_tokens=True).strip()
            answers.append(
                SystemAnswer(
                    question_id=q["id"],
                    answer=text,
                    latency_s=latency,
                    input_tokens=len(p),
                    output_tokens=n_out,
                    cost_usd=None,
                )
            )
    return answers
