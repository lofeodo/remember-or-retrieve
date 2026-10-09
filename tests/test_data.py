import json
import shutil
from collections import Counter

import pytest

from remember_or_retrieve.config import REPO_ROOT
from remember_or_retrieve.data import load_corpus, load_golden_set, verify_data


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def golden():
    return load_golden_set()


def test_corpus_size_and_unique_ids(corpus):
    assert len(corpus) == 955
    assert len({c.chunk_id for c in corpus}) == 955


def test_corpus_fields_populated(corpus):
    for c in corpus:
        assert c.text.strip()
        assert c.token_count > 0
        assert c.url.startswith("https://oldschool.runescape.wiki/")


def test_golden_size_and_unique_ids(golden):
    assert len(golden) == 55
    assert len({q.id for q in golden}) == 55


def test_golden_type_split(golden):
    counts = Counter(q.type for q in golden)
    assert counts == {"single_hop": 35, "multi_hop": 15, None: 5}


def test_unanswerable_questions_have_no_gold_chunks(golden):
    unanswerable = [q for q in golden if not q.answerable]
    assert len(unanswerable) == 5
    assert all(q.type is None and not q.gold_chunk_ids for q in unanswerable)


def test_answerable_questions_have_gold_chunks(golden):
    assert all(q.gold_chunk_ids for q in golden if q.answerable)


def test_every_gold_chunk_exists_in_corpus(corpus, golden):
    ids = {c.chunk_id for c in corpus}
    missing = [c for q in golden for c in q.gold_chunk_ids if c not in ids]
    assert missing == []


def test_data_matches_provenance():
    assert verify_data() == []


def test_tampered_copy_fails_verification(tmp_path):
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data")
    golden = tmp_path / "data" / "eval" / "golden_set.json"
    golden.write_text(golden.read_text(encoding="utf-8") + " ", encoding="utf-8")
    problems = verify_data(tmp_path)
    assert len(problems) == 1 and "golden_set.json" in problems[0]


def test_missing_file_reported(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copy(REPO_ROOT / "data" / "PROVENANCE.json", tmp_path / "data")
    assert len(verify_data(tmp_path)) == 2
    assert json.loads((tmp_path / "data" / "PROVENANCE.json").read_text())["files"]
