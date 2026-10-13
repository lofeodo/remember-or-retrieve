"""Fast GRPO tests. The pure parts (advantages, held-out split, rollout health, profile block,
chain file) run without torch; the loss and prompt-loading tests need it."""

import json

import pytest

from remember_or_retrieve.training import checks, data, profiling


def test_group_advantages_are_zero_mean_per_group_and_flat_groups_are_zero():
    adv = data.group_advantages([1.0, 0.0, 0.5, 0.5, 0.2, 0.9], group_size=2)
    assert adv[0] == pytest.approx(1.0) and adv[1] == pytest.approx(-1.0)
    assert adv[2:4] == [0.0, 0.0]  # no spread, no signal
    assert adv[4] == pytest.approx(-1.0) and adv[5] == pytest.approx(1.0)
    four = data.group_advantages([0.0, 1.0, 2.0, 3.0], group_size=4)
    assert sum(four) == pytest.approx(0.0, abs=1e-9)
    assert sum(a * a for a in four) / 4 == pytest.approx(1.0)  # unit variance


def test_group_advantages_reject_ragged_input():
    with pytest.raises(ValueError):
        data.group_advantages([1.0, 2.0, 3.0], group_size=2)


def _rows():
    kinds = ["single_hop"] * 60 + ["multi_hop"] * 20 + ["unanswerable"] * 20
    return [{"id": f"r{i}", "kind": k} for i, k in enumerate(kinds)]


def test_heldout_split_is_deterministic_disjoint_stratified_and_exact():
    rows = _rows()
    train, held = data.split_heldout(rows, 10, seed=7)
    assert (train, held) == data.split_heldout(rows, 10, seed=7)
    assert len(held) == 10 and len(train) == 90
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert {r["kind"] for r in held} == {"single_hop", "multi_hop", "unanswerable"}
    assert data.split_heldout(rows, 10, seed=8)[1] != held
    assert data.split_heldout(rows, 0, seed=7) == (rows, [])


def test_encode_completion_trains_only_on_the_completion():
    ex = data.encode_completion([1, 2, 3], [4, 5])
    assert ex["input_ids"] == [1, 2, 3, 4, 5]
    assert ex["labels"] == [data.IGNORE_INDEX] * 3 + [4, 5]


def test_rollout_health_passes_a_learning_run_and_fails_dead_ones():
    ok = checks.check_rollout_health([0.1, 0.2], [0.5, 0.25], [0.0, 0.1])
    assert ok["steps"] == 2
    with pytest.raises(checks.DistributionCheckError, match="no learning signal"):
        checks.check_rollout_health([0.1, 0.2], [1.0, 1.0], [0.0, 0.0])
    with pytest.raises(checks.DistributionCheckError, match="length limit"):
        checks.check_rollout_health([0.1, 0.2], [0.5, 0.5], [1.0, 1.0])
    with pytest.raises(checks.DistributionCheckError, match="non-finite"):
        checks.check_rollout_health([float("nan")], [0.5], [0.0])


def test_profile_gets_a_rollout_block_only_for_rl_stages():
    def rank(r, rollout):
        s = profiling.RankStats(
            rank=r, device="cpu", step_times_s=[2.0, 2.0], step_tokens=[100, 100]
        )
        if rollout:
            s.rollout_s, s.reward_s = [1.0, 1.5], [0.1, 0.1]
            s.rollout_tokens, s.rollout_completions = [200, 200], [16, 16]
            s.completion_lengths = [10] * 32
        return s

    meta = {"total_params": 10, "trainable_params": 1, "meta": {}, "wall_clock_s": 5.0}
    rl = profiling.build_profile("grpo", [rank(0, True), rank(1, True)], **meta)["rollout"]
    assert rl["generated_tokens"] == 800
    assert rl["generated_tokens_per_s_global"] == pytest.approx(800 / 2.5)
    assert rl["generated_tokens_per_s_per_device"] == pytest.approx(800 / 2.5 / 2)
    assert rl["completions_per_s_global"] == pytest.approx(64 / 2.5)
    assert rl["rollout_fraction_of_step"] == pytest.approx(2.5 / 4.0)
    assert rl["completion_tokens_mean"] == 10
    sft = profiling.build_profile("sft_lora", [rank(0, False)], **meta)
    assert "rollout" not in sft


def test_record_rollout_skips_warmup_like_step_times():
    torch = pytest.importorskip("torch")
    timer = profiling.StepTimer(0, 1, torch.device("cpu"))
    for _ in range(3):
        timer.begin()
        timer.record_rollout(1.0, 0.1, [5, 5])
        timer.end(10)
    assert len(timer.stats.step_times_s) == len(timer.stats.rollout_s) == 2


def test_chain_file_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from remember_or_retrieve.training import model as model_lib

    model_lib.write_chain(tmp_path, "some/sft/adapter")
    assert json.loads((tmp_path / model_lib.CHAIN_FILE).read_text())["init_from"] == (
        "some/sft/adapter"
    )
    model_lib.write_chain(tmp_path, None)
    assert json.loads((tmp_path / model_lib.CHAIN_FILE).read_text())["init_from"] is None


# ------------------------------------------------------------- need torch


def test_grpo_loss_matches_a_hand_computation_and_kl_is_zero_at_the_reference():
    torch = pytest.importorskip("torch")
    from remember_or_retrieve.training import grpo

    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]])
    adv = torch.tensor([2.0, -1.0])
    mask = torch.tensor([[True, True], [True, False]])
    loss, kl = grpo.grpo_loss(logp, logp.clone(), adv, mask, beta=0.5)
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    # -A*logp summed over masked tokens: -(2*-1) - (2*-2) - (-1*-0.5) = 2 + 4 - 0.5
    assert loss.item() == pytest.approx(5.5)

    ref = logp - 0.3  # policy above the reference on every token
    loss2, kl2 = grpo.grpo_loss(logp, ref, adv, mask, beta=0.5)
    k3 = (torch.exp(torch.tensor(-0.3)) + 0.3 - 1).item()
    assert kl2.item() == pytest.approx(3 * k3, rel=1e-5) and kl2.item() > 0
    assert loss2.item() == pytest.approx(5.5 + 0.5 * 3 * k3, rel=1e-5)


def test_the_policy_gradient_raises_the_probability_of_positive_advantage():
    torch = pytest.importorskip("torch")
    from remember_or_retrieve.training import grpo

    logits = torch.zeros(2, 3, requires_grad=True)
    logp = torch.log_softmax(logits, -1)[:, 0:1]  # both samples are token 0
    adv = torch.tensor([1.0, 1.0])
    loss, _ = grpo.grpo_loss(logp, logp.detach(), adv, torch.ones(2, 1, dtype=torch.bool), 0.0)
    loss.backward()
    assert (logits.grad[:, 0] < 0).all()  # descent on the loss increases token 0's logit


def test_prompt_loader_drops_unscorable_rows_and_holds_out_deterministically():
    pytest.importorskip("torch")
    from remember_or_retrieve.config import load_config
    from remember_or_retrieve.training import grpo

    cfg = load_config()
    scorer = grpo.make_scorer(cfg)
    train, held = grpo.load_prompts(cfg, scorer)
    assert len(held) == cfg["training"]["grpo"]["heldout"]
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert 395 <= len(train) + len(held) < 404  # the quirky rows are dropped, not all of them
    floor = cfg["training"]["grpo"]["min_reference_reward"]
    assert all(scorer(r, r["reference_answer"]).total >= floor for r in train + held)
    assert (train, held) == grpo.load_prompts(cfg, scorer)


def test_make_rollouts_trims_at_end_of_turn_flags_truncation_and_groups_advantages():
    pytest.importorskip("torch")
    from remember_or_retrieve.scoring.reward import RewardBreakdown
    from remember_or_retrieve.training import grpo

    IM = 99

    class Tok:
        class tok:  # noqa: N801
            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids)

    class Scorer:
        def __call__(self, row, text):
            return RewardBreakdown(total=float(len(text.split())))

    comps = [([7, 8, IM], False), ([7, 8, 9, 10], True), ([5, IM], False), ([5, IM], False)]
    out = grpo.make_rollouts([[1], [2]], comps, [{"id": "a"}, {"id": "b"}], Scorer(), Tok, 2, IM)
    assert [r.text for r in out] == ["7 8", "7 8 9 10", "5", "5"]
    assert [r.truncated for r in out] == [False, True, False, False]
    assert out[0].prompt_ids == [1] and out[2].prompt_ids == [2]
    assert out[0].advantage < 0 < out[1].advantage  # the longer text scored higher
    assert out[2].advantage == out[3].advantage == 0.0  # equal rewards, no signal
