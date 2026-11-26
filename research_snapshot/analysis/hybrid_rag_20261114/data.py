"""Offline raw-dialogue data, session-bounded windows, and Hybrid-RAG retrieval.

No model client, credential, Writer fact, answer, or Judge is used by chunking
or retrieval. The only I/O is reading the explicitly requested dataset file.
256 words counts the entire rendered chunk, including date/speaker labels.
The 32-word overlap counts source-content words, not repeated metadata labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DATASET_PATHS = {
    "locomo": WORKSPACE / "data/locomo/locomo10.json",
    "lme": WORKSPACE / "data/longmemeval/longmemeval_s.json",
}
CATEGORIES = {1: "multi", 2: "temporal", 3: "open", 4: "single"}
MAX_WORDS, OVERLAP_WORDS, RRF_CONSTANT = 256, 32, 60
BRANCH_LIMITS = {"locomo": 40, "lme": 240}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")  # Frozen LoCoMo BM25 tokenization.
_WORDS = re.compile(r"\S+")


def _turn(turn, sid, index):
    role = str(turn.get("role", ""))
    speaker = str(turn.get("speaker") or ("User" if role == "user" else "Assistant"))
    dia_id = str(turn.get("dia_id") or "")
    return {"speaker": speaker, "role": role, "dia_id": dia_id,
            "source_id": dia_id or f"{sid}:T{index + 1}",
            "text": str(turn.get("text", turn.get("content", "")) or ""),
            "caption": str(turn.get("blip_caption") or "")}


def parse_dataset(name: str, raw: Sequence[Mapping]) -> list[dict]:
    """Normalize raw sessions and separate questions/gold into their own records."""
    contexts = []
    if name == "locomo":
        for ci, item in enumerate(raw):
            cid = str(item.get("sample_id", ci))
            conversation = item["conversation"]
            sids = sorted(int(m.group(1)) for key in conversation
                          if (m := re.fullmatch(r"session_(\d+)", key)))
            sessions = [{"session_id": str(sid), "source_session_id": str(sid),
                         "date": str(conversation.get(f"session_{sid}_date_time", "")),
                         "turns": [_turn(t, str(sid), i)
                                   for i, t in enumerate(conversation[f"session_{sid}"])]}
                        for sid in sids]
            questions = []
            for qi, q in enumerate(item["qa"]):
                category = q.get("category")
                if category not in CATEGORIES:
                    continue
                question = str(q["question"])
                questions.append({"question_id": f"{cid}-q{qi:03d}",
                    "question": question, "raw_question": question, "question_date": "",
                    "answer": str(q.get("answer", q.get("adversarial_answer", ""))),
                    "category": category, "category_name": CATEGORIES[category],
                    "evidence": list(q.get("evidence", [])), "is_abstention": False})
            contexts.append({"context_id": cid, "dataset": name,
                             "sessions": sessions, "questions": questions})
    elif name == "lme":
        for item in raw:
            qid = str(item["question_id"])
            dates, history, sids = (item["haystack_dates"], item["haystack_sessions"],
                                    item["haystack_session_ids"])
            if not len(dates) == len(history) == len(sids):
                raise ValueError(f"LME session/date/ID count mismatch: {qid}")
            # Fifteen original histories repeat one session identifier. Keep both
            # occurrences and their original ID; only the internal identity is
            # disambiguated. Never merge sessions or silently discard content.
            counts, occurrences, sessions = Counter(map(str, sids)), Counter(), []
            for date, turns, raw_sid in zip(dates, history, sids, strict=True):
                source_sid = str(raw_sid)
                occurrences[source_sid] += 1
                sid = (source_sid if counts[source_sid] == 1 else
                       f"{source_sid}::occ{occurrences[source_sid]}")
                sessions.append({"session_id": sid, "source_session_id": source_sid,
                                 "date": str(date),
                                 "turns": [_turn(t, sid, i) for i, t in enumerate(turns)]})
            raw_question, date = str(item["question"]), str(item["question_date"])
            q = {"question_id": qid,
                 "question": f"{raw_question} (question asked on {date})",
                 "raw_question": raw_question, "question_date": date,
                 "answer": str(item["answer"]), "category": str(item["question_type"]),
                 "category_name": str(item["question_type"]),
                 "evidence": list(item.get("answer_session_ids", [])),
                 "is_abstention": qid.endswith("_abs")}
            contexts.append({"context_id": qid, "dataset": name,
                             "sessions": sessions, "questions": [q]})
    else:
        raise ValueError("Dataset must be locomo or lme")
    cids = [c["context_id"] for c in contexts]
    qids = [q["question_id"] for c in contexts for q in c["questions"]]
    if len(cids) != len(set(cids)) or len(qids) != len(set(qids)):
        raise ValueError("Duplicate context/question identifiers")
    for context in contexts:
        sids = [s["session_id"] for s in context["sessions"]]
        if len(sids) != len(set(sids)):
            raise ValueError(f"Duplicate session IDs: {context['context_id']}")
    return contexts


def load_dataset(name: str, path: str | Path | None = None) -> list[dict]:
    if name not in DATASET_PATHS:
        raise ValueError("Dataset must be locomo or lme")
    with Path(path or DATASET_PATHS[name]).open(encoding="utf-8") as handle:
        contexts = parse_dataset(name, json.load(handle))
    if path is None:
        expected_contexts, expected_questions = ((10, 1540) if name == "locomo" else (500, 500))
        if (len(contexts), sum(len(c["questions"]) for c in contexts)) != (expected_contexts, expected_questions):
            raise ValueError("Unexpected full evaluation population")
    return contexts


def _turn_content(turn: Mapping) -> str:
    text = str(turn.get("text", turn.get("content", "")) or "")
    caption = str(turn.get("caption", turn.get("blip_caption", "")) or "")
    return f"{text} (shared image: {caption})".strip() if caption else text


def build_chunks(sessions: Sequence[Mapping], max_words: int = MAX_WORDS,
                 overlap_words: int = OVERLAP_WORDS) -> list[dict]:
    """Greedy session-local windows, repeating labels on long-turn continuation.

    Word normalization changes whitespace only. All source-content words survive.
    Empty sessions/turns create no spurious evidence-only header. If metadata is
    pathologically large, fail explicitly rather than dropping content or looping.
    IDs are unique and stable within a context, in source order.
    """
    if not isinstance(max_words, int) or not isinstance(overlap_words, int) or not 0 <= overlap_words < max_words:
        raise ValueError("Require integer 0 <= overlap_words < max_words")
    chunks = []
    seen_sessions = set()
    for session in sessions:
        sid, date = str(session["session_id"]), str(session.get("date", ""))
        if sid in seen_sessions:
            raise ValueError("Duplicate session ID")
        seen_sessions.add(sid)
        header = f"[Session {session.get('source_session_id', sid)} at {date}]"
        header_words = len(header.split())
        if header_words >= max_words:
            raise ValueError("Session metadata exhausts the word budget")
        words, owners, labels, source_ids = [], [], [], []
        for ti, turn in enumerate(session.get("turns", [])):
            content = _WORDS.findall(_turn_content(turn))
            speaker = str(turn.get("speaker") or ("User" if turn.get("role") == "user" else "Assistant"))
            dia_id = str(turn.get("dia_id") or "")
            source_id = str(turn.get("source_id") or dia_id or f"{sid}:T{ti + 1}")
            labels.append(f"{speaker} ({dia_id or 'T' + str(ti + 1)}):")
            source_ids.append(source_id)
            words.extend(content)
            owners.extend([ti] * len(content))
        start, previous_end = 0, 0
        while start < len(words):
            end, used, previous_owner = start, header_words, None
            fragments, included_sources = [], []
            while end < len(words):
                owner = owners[end]
                new_owner = owner != previous_owner
                extra = 1 + (len(labels[owner].split()) if new_owner else 0)
                if used + extra > max_words:
                    break
                if new_owner:
                    fragments.append([labels[owner], []])
                    included_sources.append(source_ids[owner])
                fragments[-1][1].append(words[end])
                used += extra
                previous_owner = owner
                end += 1
            if end == start:
                raise ValueError("Turn metadata leaves no room for source text")
            text = "\n".join([header] + [label + " " + " ".join(ws) for label, ws in fragments])
            assert len(text.split()) == used <= max_words
            chunks.append({"id": f"C{len(chunks) + 1:06d}", "text": text,
                "session_id": sid,
                "source_session_id": str(session.get("source_session_id", sid)),
                "source_ids": list(dict.fromkeys(included_sources)),
                "word_count": used, "content_word_start": start, "content_word_end": end,
                "overlap_content_words": max(0, previous_end - start)})
            if end == len(words):
                break
            if end - start <= overlap_words:
                raise ValueError("Metadata overhead prevents progress with the requested overlap")
            previous_end, start = end, end - overlap_words
    return chunks


class BM25Index:
    """Frozen LoCoMo BM25 formula, accelerated with a term-posting index.

    Token regex, repeated-query-token weighting, k1/b, avgdl floor, and positive
    match rule reproduce locomo81_graph_memory_v2.BM25Index numerically.
    """
    def __init__(self, chunks: Sequence[Mapping], k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.ids = [str(c["id"]) for c in chunks]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate chunk IDs")
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        self.avgdl = 0.0
        for index, chunk in enumerate(chunks):
            tokens = [s.lower() for s in _TOKEN_RE.findall(str(chunk["text"]))]
            self.lengths.append(len(tokens))
            self.avgdl = (self.avgdl * index + len(tokens)) / (index + 1)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((index, tf))

    def scores(self, query: str) -> list[float]:
        n = len(self.ids)
        scores = [0.0] * n
        for term in (s.lower() for s in _TOKEN_RE.findall(query or "")):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            for index, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1))
                scores[index] += idf * tf * (self.k1 + 1) / denom
        return scores


def cosine_scores(vectors: Sequence[Sequence[float]], query_vector: Sequence[float]) -> list[float]:
    q = np.asarray(query_vector, dtype=np.float64)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Query vector must be finite, nonempty, and nonzero")
    if len(vectors) == 0:
        return []
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(q) or not np.isfinite(matrix).all():
        raise ValueError("Document vector dimensions/values are invalid")
    norms = np.linalg.norm(matrix, axis=1)
    if (norms == 0).any():
        raise ValueError("Zero document embedding")
    return ((matrix @ q) / (norms * np.linalg.norm(q))).tolist()


def hybrid_candidates(chunks: Sequence[Mapping], dense_scores: Sequence[float],
                      lexical_scores: Sequence[float], branch_k: int,
                      union_cap: int, rrf_constant: float = RRF_CONSTANT) -> tuple[list, dict]:
    """Score-free RRF; lexical branch excludes zero matches, not padding docs.

    Both ranks start at1. Ties within branches and in fused scores use original
    chunk order. The union is capped after fusion, never before its two branches.
    Returned chunks retain original text/provenance; trace contains only scores.
    """
    if (not isinstance(branch_k, int) or not isinstance(union_cap, int)
            or branch_k <= 0 or union_cap <= 0 or not math.isfinite(rrf_constant) or rrf_constant <= 0):
        raise ValueError("Invalid retrieval limits/RRF constant")
    n = len(chunks)
    if len(dense_scores) != n or len(lexical_scores) != n:
        raise ValueError("Retrieval score count mismatch")
    if any(not math.isfinite(float(x)) for x in [*dense_scores, *lexical_scores]):
        raise ValueError("Nonfinite retrieval score")
    ids = [str(c["id"]) for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    dense = sorted(range(n), key=lambda i: (-float(dense_scores[i]), i))[:branch_k]
    lexical = sorted((i for i in range(n) if lexical_scores[i] > 0),
                     key=lambda i: (-float(lexical_scores[i]), i))[:branch_k]
    fused = defaultdict(float)
    for ranked in (dense, lexical):
        for rank, index in enumerate(ranked, 1):
            fused[index] += 1 / (rrf_constant + rank)
    selected = sorted(fused, key=lambda i: (-fused[i], i))[:union_cap]
    trace = {"branch_k": branch_k, "union_cap": union_cap, "rrf_constant": rrf_constant,
             "dense_ids": [ids[i] for i in dense], "bm25_ids": [ids[i] for i in lexical],
             "bm25_positive_only": True, "union_count_before_cap": len(fused),
             "candidate_ids": [ids[i] for i in selected],
             "candidates": [{"id": ids[i], "dense_score": float(dense_scores[i]),
                             "bm25_score": float(lexical_scores[i]), "rrf_score": fused[i]}
                            for i in selected],
             "tie_break": "original chunk order", "uses_gold": False}
    return [chunks[i] for i in selected], trace
