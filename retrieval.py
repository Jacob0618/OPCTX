"""Hybrid candidate generation, mathematical selection, and model reranking."""
from collections import Counter
import json
import math
import re
import numpy as np
import prompts
from selector import select_multichannel

def bm25(texts, query):
    tokenize = lambda s: re.findall(r"[a-zA-Z0-9_]+", s.lower())
    docs = [tokenize(s) for s in texts]
    counts = [Counter(d) for d in docs]
    df = Counter(t for d in docs for t in set(d))
    n, average = len(docs), sum(map(len, docs)) / max(1, len(docs))
    scores = np.zeros(n)
    for term in tokenize(query):
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, count in enumerate(counts):
            tf = count[term]
            if tf:
                scores[i] += idf * tf * 2.5 / (
                    tf + 1.5 * (0.25 + 0.75 * len(docs[i]) / max(average, 1)))
    return scores

def cosine(items, query_vector):
    if not items:
        return np.zeros(0)
    matrix = np.asarray([item["vector"] for item in items], dtype=float)
    vector = np.asarray(query_vector, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(vector):
        raise ValueError("Stored embedding dimension mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("Nonfinite stored embedding")
    return matrix @ vector  # All stored/query vectors are L2-normalized.

def top_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]

def hybrid_candidates(memory, question, query_vector, *,
                      summary_k, dense_k, bm25_k, candidate_cap=None):
    facts = memory["facts"]
    if not facts:
        return []
    dense = cosine(facts, query_vector)
    lexical = bm25([f["text"] for f in facts], question)
    # Session summaries and event-view summaries compete in the same branch.
    routes = memory["packets"] + memory["views"]
    route_scores = cosine(routes, query_vector)
    ids = {facts[i]["id"] for i in top_indices(dense, dense_k)}
    ids.update(facts[i]["id"] for i in top_indices(lexical, bm25_k) if lexical[i] > 0)
    for i in top_indices(route_scores, summary_k):
        ids.update(routes[i]["member_fact_ids"])
    ordered = [facts[i] for i in top_indices(dense, len(facts))
               if facts[i]["id"] in ids]
    return ordered if candidate_cap is None else ordered[:candidate_cap]

def memeye_candidates(memory, question, query_vector):
    """Requested branch: Top10 summaries + Top20 dense + Top20 BM25.

    A summary/view expands to its member facts; union and deduplicate by ID.
    Therefore the pool is NOT capped at 10+20+20=50 facts.
    """
    return hybrid_candidates(memory, question, query_vector,
                             summary_k=10, dense_k=20, bm25_k=20)

def fact_line(fact):
    return f"[{fact['id']}] {fact['text']}\n"

def packet_header(packet):
    return f"\nSessions {', '.join(packet['session_ids'])} | {packet['timestamp']}\n"

def render_facts(memory, facts):
    packets = {p["id"]: p for p in memory["packets"]}
    groups = {}
    for fact in facts:
        groups.setdefault(fact["packet_id"], []).append(fact)
    return "".join(packet_header(packets[key]) + "".join(map(fact_line, values))
                   for key, values in groups.items())

def project_question(question, client):
    """Original query plus exact-quote, object-anchored query projections."""
    result = client.generate(prompts.PROJECT + json.dumps(question, ensure_ascii=False),
                             max_tokens=2048)
    roles = ("entity", "relation", "attribute", "scope")
    if set(result) != set(roles):
        return [question]
    spans = {}
    for role in roles:
        values = result[role]
        if (not isinstance(values, list) or
            any(not isinstance(x, str) or not x.strip() for x in values) or
            len(values) != len(set(values))):
            return [question]
        spans[role] = []
        for quote in values:
            start = question.find(quote)
            if start < 0 or question.find(quote, start + 1) >= 0:
                return [question]
            spans[role].append((start, start + len(quote)))
    def combine(intervals):
        merged = []
        for a, b in sorted(set(intervals)):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        return " ".join(question[a:b] for a, b in merged)
    anchors = spans["entity"] + spans["relation"] + spans["scope"]
    queries = [question, combine([s for r in roles for s in spans[r]])]
    queries += [combine(anchors + [a]) for a in spans["attribute"]]
    return list(dict.fromkeys(q for q in queries if q))

def mathematical_select(memory, candidates, question, query_vector, client, config):
    if not candidates:
        return []
    queries = project_question(question, client)
    vectors = [query_vector] + client.embed(queries[1:], query=True)
    texts = [f["text"] for f in candidates]
    channels = [{"query_id": f"q{i}", "query": q,
                 "dense_scores": cosine(candidates, v).tolist(),
                 "bm25_scores": bm25(texts, q).tolist()}
                for i, (q, v) in enumerate(zip(queries, vectors))]
    units = [{"id": f["id"], "text": f["text"], "cost": len(fact_line(f)),
              "group": f["packet_id"], "sources": f["sources"]}
             for f in candidates]
    costs = {p["id"]: len(packet_header(p)) for p in memory["packets"]}
    # Dense50 budget reference is taken over ALL visible Writer facts,
    # not just the merged hybrid candidate pool.
    all_facts = memory["facts"]
    dense50 = [all_facts[i] for i in
               top_indices(cosine(all_facts, query_vector), config.dense_budget_k)]
    budget = len(render_facts(memory, dense50))
    result = select_multichannel(
        units, question, channels, max_units=config.final_k,
        char_budget=budget, group_costs=costs,
        time_limit=config.solver_seconds_per_stage)
    by_id = {f["id"]: f for f in candidates}
    return [by_id[i] for i in result["selected_ids"]]

def rerank(candidates, question, client, config):
    target = min(config.final_k, len(candidates))
    if not target:
        return []
    payload = [{"id": f["id"], "text": f["text"]} for f in candidates]
    prompt = prompts.RERANK.format(
        top_k=target, question=question,
        candidates_json=json.dumps(payload, ensure_ascii=False))
    result = client.generate(prompt, max_tokens=2048)
    ids = result.get("ranked_fact_ids", [])
    by_id = {f["id"]: f for f in candidates}
    if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
        or len(ids) != target or len(set(ids)) != target
        or any(i not in by_id for i in ids)):
        raise ValueError("Invalid reranker IDs; no silent padding or truncation")
    return [by_id[i] for i in ids]
