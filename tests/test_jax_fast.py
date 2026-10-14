"""Fast tests for the JAX stage's pure logic. No JAX needed (runs in the 3.11 `test` job)."""

import importlib.util
from pathlib import Path

import pytest

from remember_or_retrieve.jax_stage import checks, tokenization
from remember_or_retrieve.training import data, profiling

END_ID = tokenization.END_OF_TURN_ID


class FakeInner:
    """Char-level stand-in for gm.text.Gemma4Tokenizer (ids are code points)."""

    def encode(self, text, *, add_bos=False):
        ids = [ord(c) for c in text]
        return ([tokenization.BOS_ID] + ids) if add_bos else ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_prompt_uses_gemma4_turns_with_native_system_role():
    text = tokenization.render_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]
    )
    assert text == "<|turn>system\nS<turn|>\n<|turn>user\nQ?<turn|>\n<|turn>model\n"


def test_answer_target_ends_with_the_single_end_of_turn_id():
    tok = tokenization.GemmaTokenizer(FakeInner())
    assert tok.encode("hi" + tokenization.END_OF_TURN) == [ord("h"), ord("i"), END_ID]
    assert tok.encode("hi") == [ord("h"), ord("i")]


def test_sft_row_trains_on_answer_and_stop_token_only():
    tok = tokenization.GemmaTokenizer(FakeInner())
    row = {
        "id": "r1",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "ok"},
        ],
    }
    ex = data.encode_sft_row(tok, row, 10_000, tok.end_token)
    n_prompt = len(tok.prompt_ids(row["messages"][:-1]))
    assert ex["input_ids"][0] == tokenization.BOS_ID
    assert ex["labels"][:n_prompt] == [data.IGNORE_INDEX] * n_prompt
    assert ex["labels"][n_prompt:] == [ord("o"), ord("k"), END_ID]


def test_decode_cuts_at_end_of_turn():
    tok = tokenization.GemmaTokenizer(FakeInner())
    assert tok.decode([ord(" "), ord("a"), END_ID, ord("z")]) == "a"


def test_check_devices_flags_missing_or_repeated_devices():
    reports = [{"id": 0, "platform": "cpu"}, {"id": 1, "platform": "cpu"}]
    assert checks.check_devices(reports, 2)["devices"] == [0, 1]
    with pytest.raises(checks.DistributionCheckError):
        checks.check_devices(reports, 4)
    with pytest.raises(checks.DistributionCheckError):
        checks.check_devices([reports[0], reports[0]], 2)


def test_check_batch_sharded_requires_an_even_split_of_the_batch_axis():
    ok = checks.check_batch_sharded([(2, 64), (2, 64)], (4, 64))
    assert ok["shard_rows"] == [2, 2]
    with pytest.raises(checks.DistributionCheckError):
        checks.check_batch_sharded([(4, 64), (4, 64)], (4, 64))  # replicated, not split
    with pytest.raises(checks.DistributionCheckError):
        checks.check_batch_sharded([(3, 64), (1, 64)], (4, 64))  # uneven
    with pytest.raises(checks.DistributionCheckError):
        checks.check_batch_sharded([(2, 32), (2, 32)], (4, 64))  # split the wrong axis


def test_desynced_replicas_fail_the_identical_check():
    with pytest.raises(checks.DistributionCheckError):
        checks.check_checksums_equal([10.0, 10.001], "LoRA parameter replicas", rel_tol=0.0)
    assert checks.check_checksums_equal([10.0, 10.0], "x", rel_tol=0.0) == 0.0


def test_jax_profile_renders_in_the_comparison_table():
    spec = importlib.util.spec_from_file_location(
        "compare_stages", Path(__file__).parents[1] / "scripts" / "compare_stages.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ranks = [
        profiling.RankStats(rank=r, device="gpu", step_times_s=[1.0], step_tokens=[100])
        for r in range(2)
    ]
    p = profiling.build_profile(
        "jax_lora",
        ranks,
        total_params=7_500_000_000,
        trainable_params=50_000_000,
        meta={"framework": "jax", "strategy": "jit_data_parallel"},
        wall_clock_s=3.0,
    )
    lines = mod.table([p]).splitlines()
    assert "jax" in lines[2] and "jit_data_parallel" in lines[2]


def test_lr_schedule_matches_the_pytorch_stage():
    jax = pytest.importorskip("jax")  # noqa: F841
    from remember_or_retrieve.jax_stage.train import lr_at, pad_to_multiple

    assert lr_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert lr_at(9, 100, 10, 1.0) == pytest.approx(1.0)
    assert lr_at(100, 100, 10, 1.0) == pytest.approx(0.1)  # decays to 10% of peak
    assert pad_to_multiple(65, 64) == 128 and pad_to_multiple(64, 64) == 64
