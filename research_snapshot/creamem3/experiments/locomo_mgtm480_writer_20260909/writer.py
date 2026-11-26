"""MGTM's exact local Mem0-style long extraction instructions, text adaptation.

The historical prompt is loaded as a literal, never executed. Its contextual
memory granularity is preserved; atomic_facts is only the legacy JSON field.
No previous memory, Online state, QA, gold, images or captions enter Writer.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from creamem3.memeye_adapter_v9 import _literal_constant, SOURCE_MEM0_FILE
from creamem3.experiments.locomo_online_only_20260909 import run as old

read, seal, sha, file_sha, StopRun = old.read, old.seal, old.sha, old.file_sha, old.StopRun
EXPECTED_PROMPT_SHA = 'ad19187a37813ef77ee156e714c0650e6ec749e0264bdc07d499bc9b24115155'
CURRENT_DATE = '2026-09-09'
ADAPTATION = r'''TEXT-ONLY SESSION GRAPH ADAPTATION (highest priority for this run)

Retain the preceding extraction rules and their contextually rich memory unit.
Do not revert to terse one-clause fragments merely because the output field is
named atomic_facts. Keep an event together with its specific objects, participants,
reasons, reactions, quantities and explicitly supported before/after transition.
Use the original 15–80-word guidance, up to 100 for rich content, without a hard
word or fact-count quota. Split unrelated topics, not the defining details away.

Only the COMPLETE named-speaker dialogue supplied under New Messages is evidence.
Both named speakers are real participants, not a generic assistant persona.
Attribute their statements to their actual names; do not call everyone User.
No image, caption or image URL is supplied. Do not infer unseen visual details.
Historical examples demonstrate the task, but are NEVER facts about this input.
Summary, Recently Extracted Memories, Existing Memories and Last k Messages are
empty: there is no earlier-session or Online feedback in this independent Writer.

Preserve the original relative-time wording AND its Observation Date anchor.
You may additionally ground it to an absolute date/interval as the long prompt
requires when uniquely supported. Keep vague times vague within the anchored
interval; never invent an exact day from recently or last week. Current Date is
not the dialogue's date. Preserve negation, uncertainty, plans and attribution.

Return one strict JSON object, replacing the historical memory/id/attributed_to/
linked_memory_ids serialization with this compatible provenance-bound schema:
{"batch_summary":"3–4 information-dense sentences about this session's main events,
distinctive names, activities, places, goals and date; a retrieval aid, not a
substitute for detailed memories","atomic_facts":[{"fact":"One contextually rich,
self-contained memory","source_dia_ids":["D1:1"]}]}
The schema illustration's line breaks are explanatory; actual output must be
valid JSON. Use exactly these fields. Every memory must cite one or more IDs from
the supplied dialogue that jointly support it. Copy IDs exactly. Do not invent
links, new fields or facts to fill a quota. Deduplicate only actual repetitions.
'''


def load_prompt():
    text = _literal_constant(SOURCE_MEM0_FILE, 'ADDITIVE_EXTRACTION_PROMPT')
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != EXPECTED_PROMPT_SHA: raise StopRun('Historical long prompt content drift')
    return text


def bundle():
    text = load_prompt()
    return {'historical_source': str(SOURCE_MEM0_FILE), 'historical_file_sha256': file_sha(SOURCE_MEM0_FILE),
        'historical_prompt_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'historical_prompt_lines': len(text.splitlines()),
        'adaptation_sha256': sha(ADAPTATION), 'current_date': CURRENT_DATE,
        'source_sha256': file_sha(__file__), 'full_prompt': text, 'adaptation': ADAPTATION,
        'text_only': True, 'memory_granularity': 'contextually_rich_not_terse_atomic',
        'empty_history_for_both_arms': True, 'original_examples_retained_not_evidence': True}


def isolated_api(name):
    spec = importlib.util.spec_from_file_location(name, old.FROZEN)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def source_record(record):
    return {'session_id': int(record['session_id']), 'timestamp': str(record.get('timestamp', '')),
        'turns': [{key: str(t.get(key, '')) for key in ('dia_id', 'speaker', 'text')} for t in record['turns']]}


def messages(record, api):
    safe = source_record(record); rendered = api.render_text_only_session(safe)
    prompt = '\n'.join(['## Summary', '""', '## Recently Extracted Memories', '[]', '## Existing Memories', '[]',
        '## Last k Messages', '[]', '## Observation Date', api.iso_date_from_timestamp(safe['timestamp']),
        '## Current Date', CURRENT_DATE, '## New Messages', rendered])
    return prompt, rendered


def make_graph_api(root, sample, runtime, registration):
    api = isolated_api('mgtm480_graph_' + sample.replace('-', '_'))
    prompt_bundle = bundle(); system = prompt_bundle['full_prompt'] + '\n\n' + ADAPTATION
    digest = sha(prompt_bundle)
    api.PROMPT_SHA256 = {**api.PROMPT_SHA256, 'session_packet': digest}
    api.PROMPT_SCHEMA_VERSION = 'locomo.mgtm480.contextual.v1'

    def extract(record, unused_chat_fn, *, max_attempts=3):
        safe = source_record(record); sid = safe['session_id']
        prompt, rendered = messages(safe, api)
        identity = {'registration_sha256': sha(registration), 'sample': sample, 'session_id': sid,
                    'record_sha256': sha(safe), 'prompt_bundle_sha256': digest}
        path = root / 'writer_packets' / sample / ('%04d.json' % sid)
        if path.exists():
            existing = read(path)
            if existing['identity'] != identity: raise StopRun('Writer packet identity drift')
            return existing['packet']
        legal = [t['dia_id'] for t in safe['turns']]
        if not legal: raise StopRun('Unexpected empty LoCoMo session; no silent memory loss')
        errors = []
        for attempt in range(1, max_attempts + 1):
            request_prompt = prompt
            if errors: request_prompt += '\nPrevious response failed the JSON/source contract: ' + errors[-1] + '\nReturn a corrected valid object.'
            raw = runtime.chat(sample + '/session%04d/attempt%d' % (sid, attempt), 'writer', request_prompt,
                               max_tokens=16384, json_mode=True, system=system)
            value = api._parse_json_object(raw)
            try:
                if value is None: raise api.GraphMemoryError('Writer returned no JSON object')
                summary, facts = api._validate_session_packet(value, legal)
            except api.GraphMemoryError as exc:
                errors.append(str(exc)[:500]); continue
            packet = {'session_id': sid, 'timestamp': safe['timestamp'],
                'iso_date': api.iso_date_from_timestamp(safe['timestamp']), 'turn_count': len(safe['turns']),
                'source_dia_ids': legal, 'batch_summary': summary, 'atomic_facts': facts,
                'writer_attempts': attempt, 'writer_prompt_sha256': digest,
                'text_only_input_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest()}
            seal(path, {'identity': identity, 'packet': packet, 'raw_response': raw,
                        'request_prompt_sha256': sha(request_prompt), 'system_sha256': sha(system), 'prior_validation_errors': errors})
            old.emit('MGTM480 WRITER ' + sample + '/%04d facts=%d' % (sid, len(facts)))
            return packet
        raise StopRun('MGTM480 Writer JSON/source attempts exhausted: ' + sample + '/' + str(sid) + '; ' + str(errors))

    api.extract_session_packet = extract
    return api
