"""Public API: construct memory, select facts, organize context, answer."""
import json
import re
from config import PROFILES
from client import Gemini
from memory import empty_memory, append_sessions
import prompts
from retrieval import (hybrid_candidates, memeye_candidates, mathematical_select,
                       rerank, render_facts)

OPERATION_MODE = {
    "atomic_lookup": "full", "count": "full", "explain_cause": "full",
    "summarize_state": "bounded",
    "enumerate": "local", "compare": "local", "temporal_point": "local",
    "temporal_duration": "local", "temporal_order": "local",
    "trace_change": "local", "hypothetical": "local",
}

def fill(template, **fields):
    # One pass: placeholders inside user-supplied evidence are not expanded.
    return re.sub(r"\{(question|context)\}", lambda m: fields[m[1]], template)

def text_reader_context(memory, selected, question, client):
    plan = client.generate(
        prompts.PLAN + "\nQUESTION_JSON:\n" + json.dumps(question, ensure_ascii=False),
        max_tokens=2048)
    operation = plan.get("operation")
    if operation not in OPERATION_MODE:
        raise ValueError("Unknown Reader operation")
    mode = OPERATION_MODE[operation]
    selected_ids = {f["id"] for f in selected}
    packet_ids = {f["packet_id"] for f in selected}
    context = render_facts(memory, selected)
    summaries = [p["summary"] for p in memory["packets"] if p["id"] in packet_ids]
    context += "\nLocal session summaries:\n" + "\n".join(summaries)
    if mode != "local":
        eligible = []
        for view in memory["views"]:
            # All support IDs for a displayed claim must be in selected facts.
            claims = [c for c in view["summary_claims"]
                      if c["support_fact_ids"] and
                      set(c["support_fact_ids"]) <= selected_ids]
            overlap = len(selected_ids & set(view["member_fact_ids"]))
            if overlap >= 2 and claims:
                eligible.append((overlap, view["id"], claims))
        eligible.sort(key=lambda x: (-x[0], x[1]))
        limit = 2 if mode == "bounded" else 10
        for _, view_id, claims in eligible[:limit]:
            shown = claims[:3] if mode == "bounded" else claims
            context += f"\nSupported event {view_id}:\n" + "\n".join(
                f"{c['text']} [support: {', '.join(c['support_fact_ids'])}]"
                for c in shown)
    context += "\nQuestion operation: " + operation
    context += ("\nUse source dates, distinguish old and updated states, count "
                "distinct occurrences rather than repeated descriptions, and "
                "do not infer unsupported links.")
    return context

def memeye_reader_context(memory, selected, question, client, config):
    # No event-view prose or raw images enter this Reader. Online influences
    # the selected facts through the hybrid routing branch.
    context = render_facts(memory, selected)
    result = client.generate(
        fill(prompts.FOCUSED_READER, question=question, context=context),
        max_tokens=config.reader_tokens)
    # Evidence first; candidate answer last. QA receives only this aggregation.
    order = ("question_type", "target", "direct_evidence", "count_table",
             "timeline", "comparison_table", "answer_constraints",
             "missing_information", "should_answer", "candidate_answer")
    if "direct_evidence" not in result or "candidate_answer" not in result:
        raise ValueError("Incomplete Reader output")
    return "\n".join(
        key + ": " + json.dumps(result[key], ensure_ascii=False)
        for key in order if key in result and result[key] not in ("", [], {}))

class OpCtx:
    def __init__(self, dataset="locomo", *, memory=None, client=None):
        self.config = PROFILES[dataset]
        self.client = client if client is not None else Gemini(self.config)
        self.memory = memory if memory is not None else empty_memory(self.config)
        from dataclasses import asdict
        if (self.memory.get("format") != "opctx-minimal-v1" or
            self.memory.get("config") != asdict(self.config)):
            raise ValueError("Memory/config mismatch; use its original profile settings")

    def write(self, sessions, *, image_root="."):
        """Append complete sessions chronologically; never mix histories."""
        existing = {s for p in self.memory["packets"] for s in p["session_ids"]}
        ids = [s.get("id") for s in sessions]
        if (any(not isinstance(s, str) or not s for s in ids)
            or len(set(ids)) != len(ids) or existing.intersection(ids)):
            raise ValueError("Session IDs must be unique, nonempty, and new")
        size = self.config.sessions_per_write
        for start in range(0, len(sessions), size):
            self.memory = append_sessions(self.memory, sessions[start:start+size],
                                          self.client, self.config, image_root)
        return self.memory

    def answer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Nonempty question required")
        if not self.memory["facts"]:
            return {"answer": "The information is unavailable.", "selected_fact_ids": []}
        cfg, client = self.config, self.client
        vector = client.embed([question], query=True)[0]
        if cfg.dataset == "memeye":
            # The named helper exposes the requested 10/20/20 implementation.
            if (cfg.summary_k, cfg.dense_k, cfg.bm25_k) == (10, 20, 20):
                candidates = memeye_candidates(self.memory, question, vector)
            else:
                candidates = hybrid_candidates(
                    self.memory, question, vector, summary_k=cfg.summary_k,
                    dense_k=cfg.dense_k, bm25_k=cfg.bm25_k,
                    candidate_cap=cfg.candidate_cap)
            selected = mathematical_select(self.memory, candidates, question,
                                           vector, client, cfg)
            context = memeye_reader_context(self.memory, selected, question, client, cfg)
            template = prompts.MEMEYE_QA
        else:
            candidates = hybrid_candidates(
                self.memory, question, vector, summary_k=cfg.summary_k,
                dense_k=cfg.dense_k, bm25_k=cfg.bm25_k, candidate_cap=cfg.candidate_cap)
            selected = rerank(candidates, question, client, cfg)
            context = text_reader_context(self.memory, selected, question, client)
            template = prompts.TEXT_QA
        answer = client.generate(fill(template, context=context, question=question),
                                 json_output=False, max_tokens=cfg.qa_tokens)
        return {"answer": answer, "selected_fact_ids": [f["id"] for f in selected]}
