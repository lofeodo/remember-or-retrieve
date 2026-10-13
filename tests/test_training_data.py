"""Fast tests for training/data.py: masking, batching, and the committed rows. No torch needed."""

import pytest

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.training import data

IM_END_ID = 1


class FakeTok:
    """Char-level tokenizer; `<|im_end|>` is one token."""

    pad_token_id = 0

    def prompt_ids(self, messages):
        return [2] + [ord(c) for m in messages for c in m["content"]] + [3]

    def encode(self, text):
        body = text.removesuffix(data.IM_END)
        ids = [ord(c) for c in body]
        return ids + [IM_END_ID] if text.endswith(data.IM_END) else ids


def row(i, answer="ok"):
    return {
        "id": f"r{i}",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_labels_mask_prompt_and_keep_im_end():
    ex = data.encode_sft_row(FakeTok(), row(1, "yes"), 100)
    n_prompt = len(FakeTok().prompt_ids(row(1)["messages"][:-1]))
    assert ex["labels"][:n_prompt] == [data.IGNORE_INDEX] * n_prompt
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert ex["labels"][-1] == IM_END_ID  # the model is trained to stop


def test_too_long_row_raises():
    with pytest.raises(ValueError):
        data.encode_sft_row(FakeTok(), row(1, "x" * 50), 10)


def test_collate_pads_and_masks():
    a = data.encode_sft_row(FakeTok(), row(1, "a"), 100)
    b = data.encode_sft_row(FakeTok(), row(2, "longer answer"), 100)
    out = data.collate([a, b], pad_id=0)
    assert len({len(x) for x in out["input_ids"]}) == 1
    assert out["labels"][0][-1] == data.IGNORE_INDEX
    assert out["attention_mask"][0][-1] == 0 and out["attention_mask"][1][-1] == 1


def test_global_batches_independent_of_world_size_and_deterministic():
    a = data.global_batches(100, 16, seed=7, epoch=0)
    assert a == data.global_batches(100, 16, seed=7, epoch=0)
    assert a != data.global_batches(100, 16, seed=7, epoch=1)
    assert len(a) == 6 and all(len(b) == 16 for b in a)
    assert len({i for b in a for i in b}) == 96  # no repeats; remainder dropped


@pytest.mark.parametrize("world", [1, 2, 4])
def test_rank_slices_partition_the_global_batch(world):
    lengths = [(i * 7) % 13 + 5 for i in range(64)]
    batch = list(range(0, 64, 2))[:16]
    parts = [data.rank_slice(batch, lengths, r, world) for r in range(world)]
    assert sorted(i for p in parts for i in p) == sorted(batch)
    assert all(len(p) == 16 // world for p in parts)


def test_rank_slice_balances_lengths():
    lengths = list(range(16))
    parts = [data.rank_slice(list(range(16)), lengths, r, 2) for r in (0, 1)]
    sums = [sum(lengths[i] for i in p) for p in parts]
    assert abs(sums[0] - sums[1]) <= 8


def test_accum_steps():
    assert data.accum_steps(64, 16, 4) == 1
    assert data.accum_steps(64, 16, 1) == 4
    with pytest.raises(ValueError):
        data.accum_steps(64, 16, 3)


def test_dpo_pair_shares_prompt_and_masks_it():
    row = {
        "id": "d1",
        "prompt_messages": [{"role": "user", "content": "q"}],
        "chosen": "a",
        "rejected": "bb",
    }
    out = data.encode_dpo_row(FakeTok(), row, 100)
    n_prompt = len(FakeTok().prompt_ids(row["prompt_messages"]))
    for side in ("chosen", "rejected"):
        assert out[side]["input_ids"][:n_prompt] == out["chosen"]["input_ids"][:n_prompt]
        assert out[side]["labels"][:n_prompt] == [data.IGNORE_INDEX] * n_prompt
        assert out[side]["labels"][-1] == IM_END_ID
    assert len(out["rejected"]["input_ids"]) == len(out["chosen"]["input_ids"]) + 1


def test_committed_sft_and_dpo_rows_have_the_expected_shape():
    cfg = load_config()
    out = resolve_path(load_config()["training_data"]["out_dir"])
    sft = data.read_jsonl(out / "sft_train.jsonl")
    assert sft and all(r["messages"][-1]["role"] == "assistant" for r in sft)
    dpo = data.read_jsonl(out / "dpo.jsonl")
    assert all(r["chosen"] != r["rejected"] for r in dpo)
    sft_cfg = cfg["training"]["sft"]
    assert sft_cfg["global_batch_size"] % sft_cfg["micro_batch_size"] == 0
