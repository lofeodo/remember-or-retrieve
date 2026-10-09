from remember_or_retrieve.config import load_config, resolve_path


def test_config_loads_with_data_section():
    cfg = load_config()
    assert cfg["corpus"]["license"] == "CC BY-NC-SA 3.0"
    assert {"corpus_path", "golden_set_path", "provenance_path"} <= set(cfg["data"])


def test_resolve_path_is_under_repo_root():
    path = resolve_path(load_config()["data"]["corpus_path"])
    assert path.is_absolute()
    assert path.name == "chunks.jsonl"
