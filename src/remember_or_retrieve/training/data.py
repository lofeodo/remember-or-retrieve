"""Training data for SFT and DPO: tokenization, loss masks, and world-size-independent batching.

No torch import here, so the fast CI job can test all of it with a fake tokenizer.

Batching is deterministic in (seed, epoch) and does not depend on the world size: the same
*global* batch is formed whether 1 or 4 ranks run, then split across ranks. Together with a
loss normalised by the global token count, that makes a 1-rank and a 2-rank run the same
training job, which is what the equivalence check in `checks.py` verifies.
"""

import json
import random
from pathlib import Path
from typing import Protocol

IGNORE_INDEX = -100
IM_END = "<|im_end|>"


class Tokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module uses (a fake in unit tests)."""

    pad_token_id: int | None

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        """Chat-template ids for `messages` plus the assistant generation prompt."""

    def encode(self, text: str) -> list[int]: ...


class HFTokenizer:
    """Adapter from a transformers tokenizer: Qwen3 template with thinking disabled."""

    def __init__(self, tok):
        self.tok = tok
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tok.encode(text, add_special_tokens=False)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_sft_row(tok: Tokenizer, row: dict, max_length: int) -> dict:
    """Tokenize one SFT row. Only the assistant answer (and its closing `<|im_end|>`) is trained on.

    The prompt is the template's generation prompt, exactly what inference sees, so there is no
    train/serve skew in the thinking-disabled prefix.
    """
    messages = row["messages"]
    assert messages[-1]["role"] == "assistant", row["id"]
    prompt = tok.prompt_ids(messages[:-1])
    target = tok.encode(messages[-1]["content"] + IM_END)
    ids = prompt + target
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds max_length {max_length}")
    labels = [IGNORE_INDEX] * len(prompt) + target
    return {"id": row["id"], "input_ids": ids, "labels": labels}


def encode_sft(tok: Tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    return [encode_sft_row(tok, r, max_length) for r in rows]


def collate(examples: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest example; returns plain lists (the loop turns them into tensors)."""
    width = max(len(e["input_ids"]) for e in examples)
    ids, labels, mask = [], [], []
    for e in examples:
        pad = width - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * pad)
        labels.append(e["labels"] + [IGNORE_INDEX] * pad)
        mask.append([1] * len(e["input_ids"]) + [0] * pad)
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def global_batches(n: int, global_batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Shuffled global batches for one epoch. The trailing partial batch is dropped so every
    batch splits evenly; the order depends only on (seed, epoch), never on the world size."""
    order = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    full = n // global_batch_size
    return [order[i * global_batch_size : (i + 1) * global_batch_size] for i in range(full)]


def rank_slice(batch: list[int], lengths: list[int], rank: int, world_size: int) -> list[int]:
    """This rank's share of a global batch. Sorting by length then dealing round-robin gives every
    rank a similar length mix, so no rank idles waiting on the longest sequences."""
    if len(batch) % world_size:
        raise ValueError(f"global batch {len(batch)} is not divisible by world size {world_size}")
    ordered = sorted(batch, key=lambda i: (lengths[i], i))
    return ordered[rank::world_size]


def micro_batches(indices: list[int], micro_batch_size: int) -> list[list[int]]:
    return [indices[i : i + micro_batch_size] for i in range(0, len(indices), micro_batch_size)]


def accum_steps(global_batch_size: int, micro_batch_size: int, world_size: int) -> int:
    per_rank = global_batch_size // world_size
    if global_batch_size % world_size or per_rank % micro_batch_size:
        raise ValueError(
            f"global batch {global_batch_size} must be a multiple of "
            f"micro batch {micro_batch_size} x world size {world_size}"
        )
    return per_rank // micro_batch_size


def to_trl_dpo(row: dict) -> dict:
    """A frozen DPO row in TRL's conversational preference format."""
    return {
        "prompt": row["prompt_messages"],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }
