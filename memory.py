"""Append-only Writer facts and incrementally maintained event views."""
from copy import deepcopy
import json
import math
from pathlib import Path
import prompts
from retrieval import hybrid_candidates, cosine, top_indices

def empty_memory(config):
    from dataclasses import asdict
    return {"format": "opctx-minimal-v1", "config": asdict(config),
            "facts": [], "packets": [], "views": []}

def public_fact(fact):
    return {k: v for k, v in fact.items() if k != "vector"}

def public_view(view):
    return {k: v for k, v in view.items() if k != "vector"}

def session_turns(sessions, client, config, image_root):
    lines, source_sessions = [], {}
    for session in sessions:
        sid = session["id"]
        for index, turn in enumerate(session.get("turns", []), 1):
            # Namespace IDs by session, even if the dataset reuses local IDs.
            source = sid + "::" + str(turn.get("id", index))
            if source in source_sessions:
                raise ValueError(f"Duplicate source ID: {source}")
            text = turn.get("text", "").strip()
            caption = turn.get("caption", "").strip()
            if config.dataset == "memeye" and turn.get("image") and not caption:
                caption = client.generate(
                    prompts.CAPTION, image=Path(image_root) / turn["image"],
                    json_output=False, max_tokens=config.writer_tokens)
            if config.dataset != "memeye" and (caption or turn.get("image")):
                raise ValueError("Text profile accepts dialogue only")
            if not text and not caption:
                continue
            source_sessions[source] = sid
            lines.append(f"{turn.get('speaker', 'user')} ({source}, "
                         f"{session.get('timestamp', '')}): {text}"
                         + (f"\nImage caption: {caption}" if caption else ""))
    return lines, source_sessions

def write_packet(sessions, memory, client, config, image_root):
    lines, legal = session_turns(sessions, client, config, image_root)
    if not lines:
        return None, []
    template = prompts.VISUAL_WRITER if config.dataset == "memeye" else prompts.WRITER
    template = template.replace("one COMPLETE dialogue session",
                                "the following complete dialogue session(s)")
    template = template.replace("this session", "the supplied session(s)")
    count = len(lines)
    guidance = f"approximately {max(1, math.ceil(count/2))}-{max(3, math.ceil(1.5*count))} atomic facts"
    if config.dataset == "lme":
        # Scale the extraction target with the actual number of nonempty sessions.
        n = len(set(legal.values()))
        guidance = f"approximately {20*n}-{40*n} atomic facts, only when supported"
    response = client.generate(template.format(
        turn_count=count, fact_guidance=guidance,
        session_id=", ".join(s["id"] for s in sessions),
        timestamp="; ".join(s.get("timestamp", "") for s in sessions),
        iso_date="; ".join(s.get("timestamp", "") for s in sessions),
        batch="\n".join(lines)), max_tokens=config.writer_tokens)
    summary, atomic = response.get("batch_summary"), response.get("atomic_facts")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(atomic, list):
        raise ValueError("Invalid Writer packet")
    packet_id = f"P{len(memory['packets'])+1:05d}"
    facts = []
    for item in atomic:
        text, ids = item.get("fact"), item.get("source_dia_ids")
        if (not isinstance(text, str) or not text.strip() or not isinstance(ids, list)
            or not ids or any(not isinstance(i, str) or i not in legal for i in ids)):
            raise ValueError("Writer fact has missing or invalid source references")
        facts.append({"id": f"F{len(memory['facts'])+len(facts)+1:06d}",
                      "text": text.strip(), "sources": list(dict.fromkeys(ids)),
                      "session_ids": list(dict.fromkeys(legal[i] for i in ids)),
                      "packet_id": packet_id})
    vectors = client.embed([summary] + [f["text"] for f in facts])
    for fact, vector in zip(facts, vectors[1:]):
        fact["vector"] = vector
    packet = {"id": packet_id, "summary": summary, "vector": vectors[0],
              "session_ids": list(dict.fromkeys(legal.values())),
              "timestamp": "; ".join(s.get("timestamp", "") for s in sessions),
              "member_fact_ids": [f["id"] for f in facts]}
    return packet, facts

def update_views(memory, packet, current, client, config):
    if not current or not memory["facts"]:
        return memory["views"]
    query = packet["summary"] + "\n" + "\n".join(f["text"] for f in current)
    vector = client.embed([query], query=True)[0]
    prior = hybrid_candidates(memory, query, vector, summary_k=5,
                              dense_k=40, bm25_k=20)
    # Expose complete member sets of selected active views, within the same
    # historical fact budget, so extend cannot silently drop old members.
    by_id = {f["id"]: f for f in memory["facts"]}
    visible_ids, active = set(), []
    ranked_views = top_indices(cosine(memory["views"], vector), len(memory["views"]))
    for i in ranked_views:
        view = memory["views"][i]
        union = visible_ids | set(view["member_fact_ids"])
        if len(union) <= config.online_fact_limit and len(active) < config.online_view_limit:
            active.append(view)
            visible_ids = union
    for fact in prior:
        if len(visible_ids) >= config.online_fact_limit:
            break
        visible_ids.add(fact["id"])
    historical = [f for f in memory["facts"] if f["id"] in visible_ids]
    payload = {"current_session": {"session_id": packet["id"],
                                  "summary": packet["summary"],
                                  "facts": [public_fact(f) for f in current]},
               "prior_candidate_facts": [public_fact(f) for f in historical],
               "active_views": [public_view(v) for v in active],
               "cross_session_candidate_pairs": []}
    template = prompts.ONLINE_MEMEYE if config.dataset == "memeye" else prompts.ONLINE
    result = client.generate(template.format(
        frontier_session_id=packet["id"],
        online_input_json=json.dumps(payload, ensure_ascii=False)),
        max_tokens=config.online_tokens)
    operations = result.get("operations")
    limit = 8 if config.dataset == "memeye" else 12
    if not isinstance(operations, list) or len(operations) > limit:
        raise ValueError("Invalid Online operation count")
    views = deepcopy(memory["views"])
    targets = {v["id"]: v for v in views}
    active_ids = {v["id"] for v in active}
    current_ids = {f["id"] for f in current}
    visible = {**{f["id"]: f for f in historical}, **{f["id"]: f for f in current}}
    changed = []
    seen_targets = set()
    for operation in operations:
        kind, target = operation.get("operation"), operation.get("target_view_id")
        members, claims = operation.get("member_fact_ids"), operation.get("summary_claims")
        view_type = operation.get("view_type")
        if view_type not in {"same_event", "continuing_process", "update_chain"}:
            raise ValueError("Unsupported Online view type")
        if (not isinstance(members, list) or not members or
            any(not isinstance(i, str) for i in members) or len(members) != len(set(members))):
            raise ValueError("Invalid view members")
        member_set = set(members)
        if (not member_set <= visible.keys() or not member_set & current_ids
            or not member_set & visible_ids):
            raise ValueError("View must connect visible past and current facts")
        if config.dataset == "memeye" and len(members) > 24:
            raise ValueError("MemEye event exceeds 24 member facts")
        sessions = {s for i in members for s in visible[i]["session_ids"]}
        if len(sessions) < 2 or not isinstance(claims, list) or not claims:
            raise ValueError("View must span sessions and have grounded claims")
        support = set()
        for claim in claims:
            ids = claim.get("support_fact_ids")
            if (not isinstance(claim.get("text"), str) or not claim["text"].strip()
                or not isinstance(ids, list) or not ids
                or any(not isinstance(i, str) or i not in member_set for i in ids)):
                raise ValueError("Invalid claim-support references")
            support.update(ids)
        if support != member_set:
            raise ValueError("Every member must support a claim")
        if kind == "create" and target == "":
            if any(set(v["member_fact_ids"]) == member_set for v in views):
                raise ValueError("Duplicate event view")
            view = {"id": f"V{len(views)+1:05d}", "version": 1,
                    "view_type": view_type}
            views.append(view)
            targets[view["id"]] = view
        elif kind == "extend" and target in active_ids and target not in seen_targets:
            seen_targets.add(target)
            view = targets[target]
            if view["view_type"] != view_type or not set(view["member_fact_ids"]) < member_set:
                raise ValueError("Extend must preserve old members and add new evidence")
            view["version"] += 1
        else:
            raise ValueError("Invalid Online create/extend target")
        view.update(member_fact_ids=members, summary_claims=claims,
                    summary=" ".join(c["text"] for c in claims))
        changed.append(view)
    vectors = client.embed([v["summary"] for v in changed])
    for view, vector in zip(changed, vectors):
        view["vector"] = vector
    return views

def append_sessions(memory, sessions, client, config, image_root="."):
    packet, facts = write_packet(sessions, memory, client, config, image_root)
    if packet is None:
        return memory
    views = update_views(memory, packet, facts, client, config)
    # Commit only after Writer and Online validation have both succeeded.
    return {**memory, "facts": memory["facts"] + facts,
            "packets": memory["packets"] + [packet], "views": views}
