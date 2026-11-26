"""Read-only independent binding and transport audit; no experiment writes."""
from pathlib import Path
import re
from creamem3.experiments.locomo_online_only_20260909.runtime import read, sha, file_sha, response_text, BACKBONE, JUDGE
from eval.prompts import QA_PROMPT, JUDGE_PROMPT


def generation_identity(stage, prompt, max_tokens):
    body={"contents":[{"role":"user","parts":[{"text":prompt}]}],
          "generationConfig":{"temperature":0.0,"maxOutputTokens":max_tokens,
                              "thinkingConfig":{"thinkingLevel":"minimal"} if stage=="judge" else {"thinkingBudget":0}}}
    return {"stage":stage,"scope":"shared:"+sha(body),"model":JUDGE if stage=="judge" else BACKBONE,
            "method":"generateContent","body":body}


def bound_text(root, identity):
    directory=root/"calls/provider"/identity["stage"]/sha(identity)
    request=read(directory/"request.json")
    final=read(directory/"final.json")
    raw=read(directory/f"attempt_{final['attempt']:02d}.response.json")
    if request!=identity or final["identity"]!=identity or sha(raw)!=final["response_sha256"]:
        raise ValueError("QA/Judge physical binding mismatch")
    return response_text(raw["response"])

def audit(root):
    root=Path(root)
    protocol=read(root/"protocol.json")
    problems=[]
    for path,digest in protocol["historical_graphs"].items():
        if file_sha(path)!=digest:
            problems.append("source drift: "+path)
    if file_sha(protocol["query_vector_file"])!=protocol["query_vector_file_sha256"]:
        problems.append("query vector source drift")
    counts={"rows":0,"requests":0,"physical_attempts":0,"physical_errors":0,"recovered_requests":0,
            "unrecovered":0,"orphans":0,"max_attempt":0,"qa_judge_bindings":0,"graphs":0}
    for path in sorted((root/"rows").glob("*/*.json")):
        row=read(path)
        counts["rows"]+=1
        if row["qa_id"]!=path.stem or row["identity"]["arm"]!=path.parent.name:
            problems.append("row identity "+str(path))
        if row["identity"]["protocol_sha256"]!=sha(protocol):
            problems.append("protocol binding "+str(path))
        if sha(row["context"])!=row["context_sha256"] or len(row["context"])!=row["context_chars"]:
            problems.append("context binding "+str(path))
        if len(row["selected_fact_ids"])!=20 or len(set(row["selected_fact_ids"]))!=20:
            problems.append("selection "+str(path))
        tags=re.findall(r"\[\[(yes|no)\]\]",row["judge_raw"].lower())
        if row["correct"] not in [0,1] or len(tags)!=1 or (tags[0]=="yes")!=bool(row["correct"]):
            problems.append("verdict "+str(path))
        qa_identity=generation_identity("qa",QA_PROMPT.format(question=row["question"],retrieved_texts=row["context"]),8192)
        if bound_text(root,qa_identity)!=row["response"]:
            problems.append("QA body/response binding "+str(path))
        prompt=JUDGE_PROMPT.format(question=row["question"],answer=row["answer"],response=row["response"])
        judge_match=False
        for maximum in [2048,4096,8192]:
            identity=generation_identity("judge",prompt,maximum)
            if not (root/"calls/provider/judge"/sha(identity)/"final.json").exists():
                continue
            actual=bound_text(root,identity)
            actual_tags=re.findall(r"\[\[(yes|no)\]\]",actual.lower())
            if len(actual_tags)==1:
                judge_match=actual==row["judge_raw"]
                break
        if not judge_match:problems.append("Judge body/response binding "+str(path))
        counts["qa_judge_bindings"]+=2
        origin=row["identity"].get("source_row_sha256")
        if origin:
            parent=row["identity"]["arm"].split("__")[0]
            source=read(root/"rows"/parent/path.name)
            if sha(source)!=origin or source["selected_fact_ids"]!=row["selected_fact_ids"]:
                problems.append("reader selection/source drift "+str(path))
        else:
            parent="session_strong" if path.parent.name=="session_facts_only" else path.parent.name
            retrieval=read(root/"retrieval"/parent/path.name)
            if retrieval["trace"]["rerank"]["selected_fact_ids"]!=row["selected_fact_ids"]:
                problems.append("row/retrieval selection drift "+str(path))
    for path in sorted((root/"graphs").glob("*/conv*.json")):
        record=read(path)
        registration=read(root/"variants"/(path.parent.name+".json"))
        if record["identity"]["variant_sha256"]!=sha(registration):
            problems.append("graph registration drift "+str(path))
        if record["audit"]["status"]!="pass" or record["standard_audit"]["status"]!="pass":
            problems.append("graph causal/structural failure "+str(path))
        counts["graphs"]+=1
    providers=[root/"calls/provider"]+list((root/"variant_calls").glob("*/provider"))+list((root/"diagnostic_calls").glob("*/provider"))
    for path in sorted(p for provider in providers for p in provider.glob("*/*/request.json")):
        request=read(path)
        counts["requests"]+=1
        key=sha(request)
        if path.parent.name!=key:
            problems.append("request hash "+str(path))
        final_path=path.with_name("final.json")
        errors=list(path.parent.glob("*.error.json"))
        counts["physical_errors"]+=len(errors)
        if final_path.exists():
            final=read(final_path)
            if final["identity"]!=request or final["request_sha256"]!=key:
                problems.append("final binding "+str(path))
            response=read(path.with_name(f"attempt_{final['attempt']:02d}.response.json"))
            if sha(response)!=final["response_sha256"]:
                problems.append("response hash "+str(path))
            counts["max_attempt"]=max(counts["max_attempt"],final["attempt"])
            counts["recovered_requests"]+=bool(errors)
        else:
            counts["unrecovered"]+=1
        for intent in path.parent.glob("*.intent.json"):
            counts["physical_attempts"]+=1
            prefix=intent.name.replace(".intent.json","")
            if not intent.with_name(prefix+".response.json").exists() and not intent.with_name(prefix+".error.json").exists():
                counts["orphans"]+=1
        for error in errors:
            if not read(error)["retryable"]:
                problems.append("nonretryable "+str(error))
    if counts["unrecovered"] or counts["orphans"]:
        problems.append("unfinished requests; if runner alive these may be in flight, audit after completion")
    return {"counts":counts,"problems":problems,"pass":not problems}
