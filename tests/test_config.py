from remember_or_retrieve.config import load_config, resolve_path


def test_config_loads_with_data_section():
    cfg = load_config()
    assert cfg["corpus"]["license"] == "CC BY-NC-SA 3.0"
    assert {"corpus_path", "golden_set_path", "provenance_path"} <= set(cfg["data"])


def test_resolve_path_is_under_repo_root():
    path = resolve_path(load_config()["data"]["corpus_path"])
    assert path.is_absolute()
    assert path.name == "chunks.jsonl"


def test_compute_plan_is_consistent_with_budget():
    cfg = load_config()
    compute, budget = cfg["compute"], cfg["budget"]
    assert compute["spot_usd_per_hour"] < compute["on_demand_usd_per_hour"]
    assert compute["train_gpu_count"] >= 2  # distributed means more than one device
    assert budget["target_usd"] <= budget["expected_usd"] <= budget["cap_usd"]


def test_jax_stage_config_is_consistent():
    cfg = load_config()
    jax_cfg = cfg["jax"]
    assert cfg["compute"]["jax_model"] == jax_cfg["model"]["name"]
    sft = jax_cfg["sft"]
    per_device = sft["global_batch_size"] // cfg["compute"]["train_gpu_count"]
    assert sft["global_batch_size"] % cfg["compute"]["train_gpu_count"] == 0
    assert per_device % sft["micro_batch_size"] == 0  # whole accumulation steps on 4 devices
    assert jax_cfg["max_length"] == cfg["training"]["max_length"]  # same SFT rows, same bound
