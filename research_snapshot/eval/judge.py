"""Unified binary LLM judge with auditable parse metadata."""
import os
import re

from gemini_client import chat, JUDGE_MODEL
from prompts import JUDGE_PROMPT


def _thinking_budget():
    """Use provider-default reasoning only when the run asks for it."""
    mode = os.environ.get("JUDGE_THINKING", "off").strip().lower()
    return None if mode in {"default", "on", "thinking"} else 0


def judge_one_detailed(question, answer, response, max_tokens=2048):
    prompt = JUDGE_PROMPT.format(question=question, answer=answer,
                                 response=response)
    out = chat(prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=max_tokens,
               thinking_budget=_thinking_budget(), tag="judge")
    verdict, parsed = parse_verdict_detailed(out)
    return verdict, out, parsed


def judge_one(question, answer, response):
    verdict, out, parsed = judge_one_detailed(question, answer, response)
    return verdict if parsed else parse_verdict(out)


def parse_verdict(out):
    verdict, parsed = parse_verdict_detailed(out)
    if parsed:
        return verdict
    # Legacy fallback retained for non-audited historical runners.
    m = re.search(r"\b(yes|no)\b", (out or "").lower())
    return 1 if m and m.group(1) == "yes" else 0


def parse_verdict_detailed(out):
    tags = re.findall(r"\[\[(yes|no)\]\]", (out or "").lower())
    if len(tags) == 1:
        return (1 if tags[0] == "yes" else 0), True
    return 0, False
