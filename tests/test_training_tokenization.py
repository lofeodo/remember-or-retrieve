"""Every frozen training row fits the configured max length under the real Qwen3 tokenizer.

Downloads only the tokenizer (a few MB). Slow because it needs the network.
"""

import pytest

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.training import data

pytestmark = pytest.mark.slow


def test_all_sft_and_dpo_rows_fit_max_length_and_end_with_im_end():
    pytest.importorskip("transformers")
    from remember_or_retrieve.training import model as model_lib

    cfg = load_config()
    tcfg = cfg["training"]
    tok = model_lib.load_tokenizer(tcfg["model"])
    out = resolve_path(cfg["training_data"]["out_dir"])
    longest = 0
    for name in ("sft_train.jsonl", "sft_val.jsonl"):
        rows = data.read_jsonl(out / name)
        enc = data.encode_sft(tok, rows, tcfg["max_length"])  # raises if any row is too long
        longest = max(longest, max(len(e["input_ids"]) for e in enc))
        assert all(e["labels"][-1] == tok.tok.convert_tokens_to_ids(data.IM_END) for e in enc)
        assert all(any(t != data.IGNORE_INDEX for t in e["labels"]) for e in enc)
    assert longest <= tcfg["max_length"]
    for r in data.read_jsonl(out / "dpo.jsonl"):
        data.encode_dpo_row(tok, r, tcfg["dpo"]["max_length"])
