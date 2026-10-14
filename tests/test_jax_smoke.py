"""Slow smoke tests for the JAX stage: a tiny random-init Gemma 4 on 2 virtual CPU devices.

Run in Python 3.12 with the `jax` extra (docker/jax.Dockerfile, CI job `smoke-jax`):
  XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -m slow tests/test_jax_smoke.py
The real Gemma 4 tokenizer is downloaded anonymously from the public bucket.
"""

import pytest

pytestmark = pytest.mark.slow
jax = pytest.importorskip("jax")

from remember_or_retrieve.config import load_config  # noqa: E402
from remember_or_retrieve.jax_stage import checks, generate, smoke, train  # noqa: E402
from remember_or_retrieve.jax_stage.tokenization import load_tokenizer  # noqa: E402
from remember_or_retrieve.scoring.schema import read_answers  # noqa: E402

needs_two_devices = pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="set XLA_FLAGS=--xla_force_host_platform_device_count=2 before importing jax",
)


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["jax"]["profiling"]["trace_steps"] = 1  # exercise the jax.profiler window too
    return c


@pytest.fixture(scope="module")
def tok(cfg):
    return load_tokenizer(cfg)


@needs_two_devices
def test_two_devices_train_with_all_checks(cfg, tok, tmp_path):
    rec = train.train(cfg, num_devices=2, out_dir=tmp_path, smoke=True, tokenizer=tok)
    results = rec["checks"]
    for name in ("devices_started", "batch_sharded", "replicas_identical", "loss_decreasing"):
        assert results[name]["passed"], results[name]
    assert results["replicas_identical"]["detail"]["spread"] == 0.0
    assert results["batch_sharded"]["detail"]["shard_rows"] == [2, 2]
    profile = __import__("json").loads((tmp_path / "profile.json").read_text())
    assert profile["world_size"] == 2 and profile["framework"] == "jax"
    assert profile["tokens_per_s_global"] > 0 and profile["compile_s_est"] >= 0
    assert profile["trainable_params"] < profile["total_params"]


@needs_two_devices
def test_two_devices_match_one_device(cfg, tok, tmp_path):
    result = smoke.equivalence(cfg, tmp_path, 2, tokenizer=tok)
    assert result["worst_rel_diff"] < cfg["jax"]["checks"]["equivalence_rel_tol"]


@needs_two_devices
def test_resume_continues_the_same_curve(cfg, tok, tmp_path):
    full = train.train(cfg, num_devices=2, out_dir=tmp_path / "full", smoke=True, tokenizer=tok)
    part = tmp_path / "part"
    first = train.train(cfg, num_devices=2, out_dir=part, smoke=True, tokenizer=tok, stop_after=4)
    assert first["interrupted"] and first["steps"] == 4
    second = train.train(cfg, num_devices=2, out_dir=part, smoke=True, tokenizer=tok, resume=True)
    assert second["steps"] == full["steps"]
    for a, b in zip(full["global_losses"], second["global_losses"], strict=True):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a))


@needs_two_devices
def test_unsynced_replica_is_caught(cfg, tok, tmp_path):
    with pytest.raises(checks.DistributionCheckError, match="replicas"):
        train.train(
            cfg, num_devices=2, out_dir=tmp_path, smoke=True, tokenizer=tok, perturb_replica=True
        )


@needs_two_devices
def test_generate_writes_scoreable_answers(cfg, tok, tmp_path):
    train.train(cfg, num_devices=2, out_dir=tmp_path, smoke=True, tokenizer=tok)
    out = tmp_path / "answers.jsonl"
    n = generate.run(cfg, tmp_path, out, smoke=True, limit=3)
    answers = read_answers(out)
    assert n == len(answers) == 3
    assert all(a.question_id and a.input_tokens > 0 for a in answers)


def test_lora_wrapper_adds_adapters(cfg):
    """`gm.nn.LoRA` adds nothing on Gemma 4; ours must add adapters that start as a no-op."""
    import jax.numpy as jnp
    import numpy as np

    from remember_or_retrieve.jax_stage import model as model_lib

    lm, base_model = model_lib.build_model(cfg, smoke=True)
    base = model_lib.init_base_random(base_model, 0)
    lora = model_lib.init_lora(lm, 0)
    assert model_lib.count_params(lora) > 0
    tokens = jnp.array([[0, 0, 5, 6, 7, 8]], jnp.int32)
    ref = base_model.apply(base, tokens=tokens).logits
    out = lm.apply(model_lib.merge(base, lora), tokens=tokens).logits
    assert float(jnp.abs(ref - out).max()) == 0.0  # B = 0 at init: identical to the base model
    short = lm.apply(model_lib.merge(base, lora), tokens=tokens[:, 2:]).logits
    np.testing.assert_allclose(
        np.asarray(out[:, 2:]), np.asarray(short), atol=1e-5
    )  # pad-invariant
