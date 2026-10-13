"""Distributed smoke tests: real multi-process runs of every stage on a tiny model, on CPU.

Needs Linux (gloo + Ray); run in CI's `smoke-distributed` job or in Docker locally:
  docker build -f docker/smoke.Dockerfile -t ror-smoke .
  docker run --rm --shm-size=2g -v "$PWD":/work -w /work -e PYTHONPATH=/work/src ror-smoke \
      pytest -m slow tests/test_training_smoke.py
"""

import json
import sys

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="needs Linux (gloo, Ray)"),
]

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from remember_or_retrieve.config import load_config  # noqa: E402
from remember_or_retrieve.scoring.schema import read_answers  # noqa: E402
from remember_or_retrieve.training import checks, launch, smoke  # noqa: E402
from remember_or_retrieve.training import model as model_lib  # noqa: E402

TOL = load_config()["training"]["checks"]["equivalence_rel_tol"]


def all_passed(record: dict) -> None:
    failed = {k: v for k, v in record["checks"].items() if v.get("passed") is False}
    assert not failed, failed


@pytest.mark.parametrize(
    ("mode", "strategy", "expected_checks"),
    [
        ("lora", "ddp", {"gradients_synced", "weights_identical", "loss_coherent"}),
        ("full", "fsdp", {"parameters_sharded", "loss_coherent"}),
        ("full", "ddp", {"gradients_synced", "weights_identical"}),
    ],
)
def test_two_ranks_pass_every_check_and_match_one_rank(tmp_path, mode, strategy, expected_checks):
    one = launch.spawn_cpu(smoke.sft_job, 1, mode, strategy, str(tmp_path / "w1"))
    two = launch.spawn_cpu(smoke.sft_job, 2, mode, strategy, str(tmp_path / "w2"))
    all_passed(two)
    assert expected_checks <= set(two["checks"])
    assert two["world_size"] == 2 and one["world_size"] == 1
    checks.check_equivalence(one["global_losses"], two["global_losses"], TOL)
    profile = json.loads((tmp_path / "w2" / "profile.json").read_text())
    assert profile["world_size"] == 2 and profile["tokens_per_s_global"] > 0
    assert len(profile["peak_mem_gb_per_rank"]) == 2


def test_fsdp_really_shards_and_ddp_does_not_need_to(tmp_path):
    rec = launch.spawn_cpu(smoke.sft_job, 2, "full", "fsdp", str(tmp_path))
    detail = rec["checks"]["parameters_sharded"]["detail"]
    assert detail["dtensor_params"] > 0 and len(detail["local_numels"]) == 2


def test_resume_after_a_simulated_preemption_continues_the_same_curve(tmp_path):
    whole = launch.spawn_cpu(smoke.sft_job, 2, "full", "fsdp", str(tmp_path / "whole"))
    out = str(tmp_path / "interrupted")
    part = launch.spawn_cpu(smoke.sft_job, 2, "full", "fsdp", out, {"stop_after": 3})
    assert part["interrupted"] and part["steps"] == 3
    rest = launch.spawn_cpu(smoke.sft_job, 2, "full", "fsdp", out, {"resume": True})
    assert not rest["interrupted"] and rest["steps"] == whole["steps"]
    checks.check_equivalence(whole["global_losses"], rest["global_losses"], TOL)


def test_dpo_two_ranks_all_checks_pass(tmp_path):
    rec = launch.spawn_cpu(smoke.dpo_job, 2, str(tmp_path))
    all_passed(rec)
    assert rec["global_losses"][0] == pytest.approx(
        0.6931, abs=1e-3
    )  # policy == reference at step 0
    assert (tmp_path / "adapter" / "adapter_config.json").exists()


def test_dpo_on_top_of_an_sft_adapter(tmp_path):
    launch.spawn_cpu(smoke.sft_job, 2, "lora", "ddp", str(tmp_path / "sft"))
    rec = launch.spawn_cpu(
        smoke.dpo_job, 2, str(tmp_path / "dpo"), str(tmp_path / "sft" / "adapter")
    )
    all_passed(rec)


def _sft_dpo(tmp_path):
    launch.spawn_cpu(smoke.sft_job, 2, "lora", "ddp", str(tmp_path / "sft"))
    launch.spawn_cpu(smoke.dpo_job, 2, str(tmp_path / "dpo"), str(tmp_path / "sft" / "adapter"))
    return tmp_path / "dpo" / "adapter"


def test_grpo_on_top_of_sft_and_dpo_passes_every_check_and_profiles_rollouts(tmp_path):
    dpo_adapter = _sft_dpo(tmp_path)
    rec = launch.spawn_cpu(smoke.grpo_job, 2, str(tmp_path / "grpo"), str(dpo_adapter))
    all_passed(rec)
    assert {"gradients_synced", "weights_identical", "rollout_health"} <= set(rec["checks"])
    assert rec["steps"] == 4 and rec["world_size"] == 2
    assert rec["log"][0]["kl"] == pytest.approx(0.0, abs=1e-6)  # policy == reference at step 0
    assert set(rec["heldout_reward"]) >= {"0", "4"}
    profile = json.loads((tmp_path / "grpo" / "profile.json").read_text())
    assert profile["rollout"]["generated_tokens_per_s_global"] > 0
    assert 0 < profile["rollout"]["rollout_fraction_of_step"] < 1
    adapter = tmp_path / "grpo" / "adapter"
    chain = json.loads((adapter / "base.json").read_text())
    assert chain["init_from"] == str(dpo_adapter)


def test_the_adapter_chain_loads_the_whole_stack_not_just_the_last_adapter(tmp_path):
    import shutil

    dpo_adapter = _sft_dpo(tmp_path)
    tcfg = load_config()["training"]
    args = (tcfg["model"],)
    stack = model_lib.load_trained(*args, str(dpo_adapter), tcfg["smoke"], tcfg["sft"]["seed"])
    orphan = tmp_path / "orphan"
    shutil.copytree(dpo_adapter, orphan)
    (orphan / "base.json").unlink()
    alone = model_lib.load_trained(*args, str(orphan), tcfg["smoke"], tcfg["sft"]["seed"])
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        a, b = stack(input_ids=ids).logits, alone(input_ids=ids).logits
    assert not torch.allclose(a, b, atol=1e-6)  # SFT really is in the stack


def test_grpo_learner_is_the_same_job_on_one_rank_and_two(tmp_path):
    one = launch.spawn_cpu(smoke.grpo_learner_job, 1)
    two = launch.spawn_cpu(smoke.grpo_learner_job, 2)
    assert two["grad_spread"] == pytest.approx(0.0, abs=1e-9)
    assert one["loss"] == pytest.approx(two["loss"], rel=1e-5)
    assert one["kl"] > 0 and one["kl"] == pytest.approx(two["kl"], rel=1e-5)
    assert one["grad_checksum"] == pytest.approx(two["grad_checksum"], rel=1e-4)


def test_grpo_resume_after_a_simulated_preemption_continues_the_same_run(tmp_path):
    dpo_adapter = _sft_dpo(tmp_path)
    init = str(dpo_adapter)
    whole = launch.spawn_cpu(smoke.grpo_job, 2, str(tmp_path / "whole"), init)
    out = str(tmp_path / "interrupted")
    part = launch.spawn_cpu(smoke.grpo_job, 2, out, init, {"stop_after": 2})
    assert part["interrupted"] and part["steps"] == 2
    rest = launch.spawn_cpu(smoke.grpo_job, 2, out, init, {"resume": True})
    assert not rest["interrupted"] and rest["steps"] == whole["steps"]
    got = [r["reward"] for r in rest["log"]]
    assert got == pytest.approx([r["reward"] for r in whole["log"]], abs=1e-6)


def test_an_unsynced_run_is_caught():
    with pytest.raises(Exception, match="not synchronised|exitcode|Process"):
        launch.spawn_cpu(smoke.unsynced_job, 2)


def test_ray_train_runs_sft_dpo_grpo_and_generate_writes_scoring_answers(tmp_path):
    cfg = load_config()
    for stage, mode in [("sft_lora", "lora"), ("dpo", "lora"), ("grpo", "lora")]:
        out = tmp_path / stage
        launch.run_ray(
            {
                "stage": stage,
                "mode": mode,
                "strategy": None,
                "cfg": cfg,
                "out_dir": str(out),
                "smoke": True,
                "init_from": (str(tmp_path / "dpo" / "adapter") if stage == "grpo" else None),
            },
            num_workers=2,
            use_gpu=False,
            storage_path=str(out / "ray"),
            name=stage,
        )
        rec = json.loads((out / "run.json").read_text())
        assert rec["world_size"] == 2
        all_passed(rec)

    from remember_or_retrieve.data import load_golden_set
    from remember_or_retrieve.scoring.schema import write_answers
    from remember_or_retrieve.training import generate
    from remember_or_retrieve.training import model as model_lib

    tcfg = cfg["training"]
    tok = model_lib.load_tokenizer(tcfg["model"])
    net = model_lib.load_trained(
        tcfg["model"], str(tmp_path / "sft_lora" / "adapter"), tcfg["smoke"], tcfg["sft"]["seed"]
    )
    golden = [{"id": q.id, "question": q.question} for q in load_golden_set()[:3]]
    answers = generate.generate_answers(
        net, tok, golden, torch.device("cpu"), max_new_tokens=4, batch_size=2
    )
    path = tmp_path / "answers.jsonl"
    write_answers(path, answers)
    back = read_answers(path)
    assert [a.question_id for a in back] == [q["id"] for q in golden]
    assert all(a.output_tokens is not None for a in back)
