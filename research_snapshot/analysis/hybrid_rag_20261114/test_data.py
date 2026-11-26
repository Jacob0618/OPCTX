"""Pure local tests; run `python -m pytest .../test_data.py -q`."""
from __future__ import annotations

from collections import Counter
import copy
import importlib.util
import socket
import sys
from unittest.mock import patch

import numpy as np
import pytest

from analysis.hybrid_rag_20261114 import data as d


def session(sid="s1", text="hello world", date="2024-01-01", speaker="User"):
    return {"session_id": sid, "date": date,
            "turns": [{"speaker": speaker, "dia_id": "", "text": text, "caption": ""}]}


def test_long_turn_overlaps_and_has_complete_word_coverage():
    s = session(text=" ".join(f"word{i}" for i in range(1000)))
    chunks = d.build_chunks([s])
    assert len(chunks) > 4
    assert chunks[0]["content_word_start"] == 0
    assert chunks[-1]["content_word_end"] == 1000
    for i, c in enumerate(chunks):
        assert c["word_count"] == len(c["text"].split()) <= 256
        assert c["text"].startswith("[Session s1 at 2024-01-01]\nUser (T1): ")
        assert c["source_ids"] == ["s1:T1"]
        if i:
            assert chunks[i-1]["content_word_end"] - c["content_word_start"] == 32
            assert c["overlap_content_words"] == 32


def test_session_boundaries_empty_sessions_and_caption_provenance():
    a = session("1", "a " * 400, speaker="Alice")
    a["turns"][0].update(dia_id="D1:1", caption="a red cabinet")
    b = session("2", "b " * 400, date="2024-02-02", speaker="Bob")
    empty = {"session_id": "3", "date": "", "turns": []}
    chunks = d.build_chunks([a, b, empty])
    assert all(c["session_id"] in {"1", "2"} for c in chunks)
    for c in chunks:
        assert ("Alice" in c["text"]) != ("Bob" in c["text"])
        assert ("2024-01-01" in c["text"]) != ("2024-02-02" in c["text"])
    assert "shared image: a red cabinet" in " ".join(c["text"] for c in chunks)
    assert any(c["source_ids"] == ["D1:1"] for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_many_short_turns_count_repeated_labels_in_budget():
    s = session()
    s["turns"] = [{"speaker": "Two Word Name", "text": f"t{i}", "dia_id": f"D1:{i}"}
                  for i in range(150)]
    chunks = d.build_chunks([s])
    assert len(chunks) > 1
    assert all(c["word_count"] <= 256 for c in chunks)
    assert chunks[-1]["content_word_end"] == 150


def test_invalid_metadata_and_duplicate_session_fail_not_loop():
    with pytest.raises(ValueError):
        d.build_chunks([session(date="date " * 300)])
    with pytest.raises(ValueError):
        d.build_chunks([session(), session()])
    with pytest.raises(ValueError):
        d.build_chunks([session()], overlap_words=256)


def test_locomo_gold_separate_and_original_question_index():
    raw = [{"sample_id": "conv-x", "conversation": {"session_1": [
        {"speaker": "Alice", "dia_id": "D1:1", "text": "source content"}],
        "session_1_date_time": "DATE"}, "qa": [
        {"question": "ignore", "category": 5},
        {"question": "QUERY_SENTINEL", "category": 1, "answer": "GOLD_SENTINEL"}]}]
    c = d.parse_dataset("locomo", raw)[0]
    assert c["questions"][0]["question_id"] == "conv-x-q001"
    chunks = d.build_chunks(c["sessions"])
    assert "QUERY_SENTINEL" not in str(chunks) and "GOLD_SENTINEL" not in str(chunks)
    other = copy.deepcopy(raw)
    other[0]["qa"][1].update(question="changed", answer="changed")
    assert chunks == d.build_chunks(d.parse_dataset("locomo", other)[0]["sessions"])


def test_lme_preserves_actual_session_dates_roles_and_question_date_once():
    raw = [{"question_id": "qid_abs", "question_type": "knowledge-update",
            "question": "Where?", "question_date": "ASK_DATE", "answer": "GOLD",
            "haystack_dates": ["DATE_A", "DATE_B"],
            "haystack_session_ids": ["sid-a", "sid-b"],
            "haystack_sessions": [[{"role": "user", "content": "earlier"}],
                                  [{"role": "assistant", "content": "later"}]]}]
    c = d.parse_dataset("lme", raw)[0]
    q = c["questions"][0]
    assert q["question"] == "Where? (question asked on ASK_DATE)"
    assert q["raw_question"] == "Where?" and q["is_abstention"]
    chunks = d.build_chunks(c["sessions"])
    assert len(chunks) == 2
    assert "DATE_A" in chunks[0]["text"] and "DATE_B" in chunks[1]["text"]
    assert "User (T1): earlier" in chunks[0]["text"]
    assert "Assistant (T1): later" in chunks[1]["text"]
    assert "GOLD" not in str(chunks) and "ASK_DATE" not in str(chunks)


def test_bm25_scores_match_frozen_original_and_empty_query():
    spec = importlib.util.spec_from_file_location("_hybrid_test_frozen_graph",
        d.WORKSPACE / "creamem3/frozen_sources/locomo81_graph_memory_v2.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    chunks = [{"id": f"C{i}", "text": text} for i, text in enumerate([
        "the cabinet cabinet is blue", "a white cabinet", "brown dog", "", "CABINET_2 blue 123"])]
    ours, reference = d.BM25Index(chunks), original.BM25Index()
    for c in chunks:
        reference.add(c["id"], c["text"])
    for query in ["cabinet", "cabinet cabinet", "blue 123", "???", "", "notmatched"]:
        baseline = dict(reference.search(query, len(chunks)))
        np.testing.assert_array_equal(ours.scores(query), [baseline.get(c["id"], 0.) for c in chunks])


def test_duplicate_lme_session_ids_preserve_both_dates_and_contents():
    row = {"question_id": "q", "question": "Which?", "question_date": "NOW",
           "question_type": "knowledge-update", "answer": "GOLD",
           "haystack_session_ids": ["same", "same"],
           "haystack_dates": ["EARLIER", "LATER"],
           "haystack_sessions": [[{"role": "user", "content": "first"}],
                                 [{"role": "user", "content": "second"}]]}
    parsed = d.parse_dataset("lme", [row])[0]
    assert [s["session_id"] for s in parsed["sessions"]] == ["same::occ1", "same::occ2"]
    assert [s["source_session_id"] for s in parsed["sessions"]] == ["same", "same"]
    chunks = d.build_chunks(parsed["sessions"])
    assert chunks[0]["text"] == "[Session same at EARLIER]\nUser (T1): first"
    assert chunks[1]["text"] == "[Session same at LATER]\nUser (T1): second"
    assert chunks[0]["source_ids"] != chunks[1]["source_ids"]
    assert d.BRANCH_LIMITS == {"locomo": 40, "lme": 240}


def test_rrf_union_caps_ties_and_positive_bm25_only():
    chunks = [{"id": f"C{i}", "text": str(i)} for i in range(4)]
    picked, trace = d.hybrid_candidates(chunks, [4,3,2,1], [0,0,2,1], 2, 3)
    assert trace["dense_ids"] == ["C0", "C1"]
    assert trace["bm25_ids"] == ["C2", "C3"]
    assert trace["union_count_before_cap"] == 4
    assert [c["id"] for c in picked] == ["C0", "C2", "C1"]
    picked, trace = d.hybrid_candidates(chunks, [1,1,1,1], [0,0,0,0], 3, 2)
    assert [c["id"] for c in picked] == ["C0", "C1"] and trace["bm25_ids"] == []
    assert d.hybrid_candidates([], [], [], 120, 120)[0] == []
    with pytest.raises(ValueError):
        d.hybrid_candidates(chunks, [float("nan")]*4, [0]*4, 2, 2)


def test_cosine_metric_and_invalid_vectors():
    np.testing.assert_allclose(d.cosine_scores([[2,0],[0,3],[-1,0]], [5,0]), [1,0,-1])
    assert d.cosine_scores([], [1,0]) == []
    with pytest.raises(ValueError):
        d.cosine_scores([[0,0]], [1,0])
    with pytest.raises(ValueError):
        d.cosine_scores([[1,2,3]], [1,0])


@pytest.mark.parametrize("name,contexts,questions", [("locomo",10,1540),("lme",500,500)])
def test_full_dataset_census_no_gold_fields_in_memory(name, contexts, questions):
    def blocked(*args, **kwargs):
        raise AssertionError("Network attempted in pure data test")
    with patch("socket.socket.connect", blocked), patch("socket.create_connection", blocked):
        rows = d.load_dataset(name)
        assert len(rows) == contexts
        qids = [q["question_id"] for c in rows for q in c["questions"]]
        assert len(qids) == len(set(qids)) == questions
        for c in rows:
            assert set(c) == {"context_id", "dataset", "sessions", "questions"}
            for s in c["sessions"]:
                assert set(s) == {"session_id", "source_session_id", "date", "turns"}
                for t in s["turns"]:
                    assert set(t) == {"speaker", "role", "dia_id", "source_id", "text", "caption"}
            chunks = d.build_chunks(c["sessions"])
            assert chunks and len({x["id"] for x in chunks}) == len(chunks)
            assert all(0 < x["word_count"] <= 256 for x in chunks)
        if name == "locomo":
            assert Counter(q["category_name"] for c in rows for q in c["questions"]) == {
                "single": 841, "multi": 282, "temporal": 321, "open": 96}
