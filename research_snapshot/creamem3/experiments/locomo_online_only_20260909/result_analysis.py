"""Read-only descriptive paired analysis;never chooses or changes an arm.

Nominal paired-question tests are supplemented by conversation-cluster
resampling because questions from one conversation are not independent.
Ten clusters and a reused benchmark do not establish deployment generality.
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from creamem3.experiments.locomo_online_only_20260909 import run
from creamem3.experiments.locomo_online_only_20260909.runtime import read, sha, seal

MAIN = ["session_strong", "opctx_original", "relational_delta",
        "opctx_original_additive_routes_v2", "delta_clean"]
READERS = ["session_facts_only", "opctx_original_additive_routes_v2__facts_only"]


def exact_discordance_p(wins, losses):
    """Two-sided exact McNemar/binomial test, descriptive, not multiplicity adjusted."""
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n))


def cluster_interval(pairs, repeats=10000, seed=20260909):
    groups = defaultdict(list)
    for a, b in pairs:
        groups[a["sample_id"]].append(a["correct"] - b["correct"])
    aggregates = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    if not aggregates:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        selected = rng.choices(aggregates, k=len(aggregates))
        values.append(100 * sum(s for s, n in selected) / sum(n for s, n in selected))
    values.sort()
    return {"clusters":len(aggregates), "resamples":repeats, "seed":seed,
            "lower_pp":values[int(0.025 * (repeats - 1))],
            "upper_pp":values[int(0.975 * (repeats - 1))],
            "caveat":"Only ten conversations;descriptive uncertainty,not unseen-test generalization"}


def pair_summary(pairs):
    if not pairs:
        return {"n":0}
    wins = sum(a["correct"] > b["correct"] for a, b in pairs)
    losses = sum(a["correct"] < b["correct"] for a, b in pairs)
    return {"n":len(pairs), "a_correct":sum(a["correct"] for a, b in pairs),
            "b_correct":sum(b["correct"] for a, b in pairs),
            "wins":wins, "losses":losses, "net_pp":100 * (wins-losses) / len(pairs),
            "nominal_exact_mcnemar_p":exact_discordance_p(wins,losses),
            "conversation_cluster_interval":cluster_interval(pairs),
            "identical_contexts":sum(a["context_sha256"]==b["context_sha256"] for a,b in pairs),
            "mean_top20_overlap":statistics.mean(len(set(a["selected_fact_ids"]) & set(b["selected_fact_ids"])) for a,b in pairs),
            "mean_context_chars_a":statistics.mean(a["context_chars"] for a,b in pairs),
            "mean_context_chars_b":statistics.mean(b["context_chars"] for a,b in pairs),
            "gained_ids":[a["qa_id"] for a,b in pairs if a["correct"]>b["correct"]],
            "lost_ids":[a["qa_id"] for a,b in pairs if a["correct"]<b["correct"]]}


def analyze(population="dev", save=False):
    protocol = read(run.RUN / "protocol.json")
    dev = set(protocol["dev_ids"])
    wanted = dev if population=="dev" else set(protocol["all_ids"])
    if population=="remaining":
        wanted -= dev
    arm_rows = {}
    for directory in sorted((run.RUN/"rows").iterdir()):
        if directory.is_dir():
            arm_rows[directory.name] = {p.stem:read(p) for p in directory.glob("*.json") if p.stem in wanted}
    scores = {}
    for arm, rows in arm_rows.items():
        scores[arm] = {"n":len(rows), "expected":len(wanted), "complete":len(rows)==len(wanted),
                      "correct":sum(r["correct"] for r in rows.values()), "by_category":{}, "by_conversation":{}}
        for field, key in [("category_name","by_category"),("sample_id","by_conversation")]:
            for group in sorted({r[field] for r in rows.values()}):
                subset = [r for r in rows.values() if r[field]==group]
                scores[arm][key][group] = {"n":len(subset),"correct":sum(r["correct"] for r in subset)}
    comparisons = [(a,b) for a in MAIN[1:] for b in ["session_strong","opctx_original"] if a!=b]
    comparisons += [(READERS[1],READERS[0]),("opctx_original__facts_only",READERS[0])]
    paired = {}
    for a,b in comparisons:
        shared = sorted(arm_rows.get(a,{}).keys() & arm_rows.get(b,{}).keys())
        pairs = [(arm_rows[a][q],arm_rows[b][q]) for q in shared]
        paired[a+" vs "+b] = pair_summary(pairs)
    result = {"population":population,"protocol_sha256":sha(protocol),"scores":scores,"paired":paired,
              "incomplete_scores_are_order_biased":True,"no_algorithm_or_selection_changes":True}
    if save:
        required = MAIN+READERS if population!="dev" else MAIN
        if any(not scores.get(a,{}).get("complete") for a in required):
            raise ValueError("Do not seal final report before all preregistered arms complete")
        seal(run.RUN/"analysis"/("paired_result_"+population+".json"),result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--population",choices=["dev","full","remaining"],default="dev")
    p.add_argument("--save",action="store_true");args=p.parse_args()
    print(json.dumps(analyze(args.population,args.save),ensure_ascii=False))
