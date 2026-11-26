"""Clarify existing state contracts; typed per-event evidence partitions.

New roots, prompts and manifests. No response from the stopped intake run is
imported or reinterpreted, and no evidence validation is weakened. A short
causal-prefix smoke precedes full dispatch. Cooperative stop drains responses.
"""
import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from creamem3.experiments.locomo_online_semantics_20260909 import stable_event_state as source
from creamem3.experiments.locomo_online_semantics_20260909 import ordinal_proof_reader as loader

old, read, seal, sha, file_sha, StopRun = source.old, source.read, source.seal, source.sha, source.file_sha, source.StopRun
RULES = """
MANDATORY OUTPUT CONTRACT, including ADD:
- EVERY operation, including ADD, has a nonempty reason explaining the concrete
  cross-session connection and reason_fact_ids containing at least one supplied
  CURRENT fact. Empty reason or [] is invalid even for ADD.
- For ADD: old_claim_ids=[], revision_kind="none", effective_time="".
- An unknown or inapplicable date is exactly the empty string "". Never output
  JSON null, the string "null", "unknown", or a guessed precise date. This rule
  applies to effective_time, valid_from and valid_to. A relation normally uses
  valid_from="", valid_to="" unless an exact event date is explicitly grounded.
- Use each target's current_member_fact_ids ONLY in current_fact_ids and its
  earlier_member_fact_ids ONLY in earlier_fact_ids. Both lists are required for
  every new claim. "Earlier" means an earlier SESSION, not another sentence of
  the current session. Do not assign an unrelated fact to a convenient target.
- If the target has no valid current/earlier pair, do not invent an ADD there.
- Unused string fields are "", unused arrays are [], but reason is never unused.
"""
PROMPT = source.PROMPT.replace('Return JSON only. No summaries of the task and no anticipated benchmark answers.\nDATA:\n',
                               RULES + '\nReturn JSON only. No anticipated benchmark answers.\nDATA:\n')
SCHEMA = json.loads(json.dumps(source.roles.SCHEMA))  # De-alias shared schema dictionaries before field-specific constraints.
OP = SCHEMA['properties']['operations']['items']['properties']
OP['reason_fact_ids']['minItems'] = 1
OP['reason']['minLength'] = 1
for item in [OP['effective_time'], OP['new_claims']['items']['properties']['valid_from'], OP['new_claims']['items']['properties']['valid_to']]:
    item['description'] = 'Exactly YYYY-MM-DD if known and applicable; otherwise exactly empty string. Never null or the string null.'


def root_for(thinking): return source.ROOT / ('stable_state_intake_v4_t' + str(thinking))


def make_input(step, prefix, state, facts):
    value, exposed, scopes = source.make_input(step, prefix, state, facts)
    current = set(exposed['current_fact_ids']); earlier = set(exposed['fact_ids']) - current
    for event in value['stable_event_scopes']:
        event['current_member_fact_ids'] = [f for f in event['member_fact_ids'] if f in current]
        event['earlier_member_fact_ids'] = [f for f in event['member_fact_ids'] if f in earlier]
    return value, exposed, scopes


class StateRuntime(source.StateRuntime):
    def structured(self, scope, prompt, schema, thinking):
        if (self.root.parent / 'STOP_REQUESTED.json').exists():
            self._halt('Cooperative stop requested; preserve in-flight receipts and checkpoints')
        return super().structured(scope, prompt, schema, thinking)


def configure():
    module = loader.load_isolated('stable_intake_v4_engine', source.__file__)
    module.root_for = root_for; module.PROMPT = PROMPT; module.make_input = make_input
    class RoleAdapter:
        normalize_operation = staticmethod(source.roles.normalize_operation)
        SCHEMA = globals()['SCHEMA']
        __file__ = source.roles.__file__
    module.roles = RoleAdapter
    def registered_seal(path, value):
        if path.name == 'registration.json':
            value.update(intake_wrapper_sha256=file_sha(__file__),
                typed_event_partition_is_exact_same_existing_evidence=True,
                embedded_model_input_JSON_canonical_sort_keys=True,
                source_rationale='Deterministic replay regardless of provider JSON object key order',
                clarified_ADD_reason_and_empty_date_not_relaxed=True,
                no_prior_stopped_responses_imported=True,
                smoke_prefix_can_resume_into_full_same_contract=True)
        return seal(path, value)
    module.seal = registered_seal
    return module


def build_one(module, sample, graph, root, registration, runtime, thinking, limit=None):
    facts = {f['id']: f for f in graph['fact_nodes']}; state = source.compiler.empty_state(); prefix = {}; digests = []
    graph_sha = sha(graph); registration_sha = sha(registration)
    steps = graph['online_fusion']['steps']; chosen_steps = steps if limit is None else steps[:limit]
    for step in chosen_steps:
        sid = step['session_id']
        for accepted in step['accepted']: prefix[accepted['resulting_view']['id']] = accepted['resulting_view']
        value, exposed, scopes = make_input(step, prefix, state, facts)
        identity = {'parent_graph_sha256': graph_sha, 'registration_sha256': registration_sha,
                    'source_step_sha256': sha(step), 'input_sha256': sha(value), 'before_state_sha256': sha(state), 'session_id': sid}
        path = root / 'checkpoints' / sample / (f'{sid:04d}.json')
        if path.exists():
            cached = read(path)
            if cached['identity'] != identity: raise StopRun('Typed-prefix checkpoint drift')
            response = cached['response']
        else:
            response = {'operations': []} if not exposed['view_ids'] or not exposed['current_fact_ids'] else runtime.structured(
                sample + '/' + str(sid), PROMPT + json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True), SCHEMA, thinking)
        def validate_revision(i, data):
            return runtime.structured(sample + '/' + str(sid) + '/revision_gate/' + str(i),
                source.GATE_PROMPT + json.dumps(data, ensure_ascii=False, separators=(',', ':'), sort_keys=True), source.GATE_SCHEMA, thinking)
        state, accepted, rejected, gates = source.apply_response(state, response, exposed, scopes, facts, sid, validate_revision)
        checkpoint = {'identity': identity, 'input': value, 'response': response, 'accepted': accepted, 'rejected': rejected,
                      'revision_gates': gates, 'state': state, 'state_sha256': sha(state)}
        seal(path, checkpoint); digests.append(sha(checkpoint))
        old.emit('TYPED STATE t' + str(thinking) + ' ' + sample + '/' + str(sid) + ' claims=' + str(len(state['claims']))
                 + ' accepted=' + str(len(accepted)) + ' rejected=' + str(len(rejected)))
    if any(c['view_id'] not in prefix for c in state['claims'].values()): raise StopRun('Unexpected index event handle')
    final = {'identity': {'parent_graph_sha256': graph_sha, 'registration_sha256': registration_sha},
             'state': state, 'step_sha256': digests, 'coarse_events': len(prefix), 'raw_writer_unchanged': True,
             'same_original_causal_routes': True, 'new_index_nodes': 0}
    if len(chosen_steps) == len(steps): seal(root / 'states' / (sample + '.json'), final)
    return {'sample': sample, 'claims': len(state['claims']), 'coarse_events': len(prefix), 'steps': len(chosen_steps),
            'revisions': sum(o['operation']['action'] != 'add' for o in state['ledger'])}


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'smoke', 'run']); p.add_argument('--thinking', type=int, choices=[0, 8192], required=True)
    a = p.parse_args(); module = configure(); root, loaded, protocol, registration, parents = module.prepare(a.thinking)
    if a.command == 'prepare': return
    if a.command == 'run':
        smoke = read(root / 'smoke_complete.json')
        if not smoke['pass'] or smoke['registration_sha256'] != sha(registration): raise StopRun('Smoke not passed')
    from dotenv import load_dotenv
    load_dotenv(old.ROOT / '.env', override=False)
    runtime = StateRuntime(root / 'calls', allow_model_calls=True, concurrency=10, timeout=180)
    with old.Lease(root):
        try:
            targets = {'conv-26': parents['conv-26']} if a.command == 'smoke' else parents
            with ThreadPoolExecutor(max_workers=10) as pool:
                stats = [f.result() for f in as_completed([pool.submit(build_one, module, s, g, root, registration, runtime,
                    a.thinking, 4 if a.command == 'smoke' else None) for s, g in targets.items()])]
            if a.command == 'smoke':
                errors = [r for path in (root / 'checkpoints/conv-26').glob('*.json') for r in read(path)['rejected']]
                contract_errors = [r for r in errors if r['reason'] in {'Date must be explicit ISO or empty', 'Revision reason lacks exposed current evidence', 'Missing reason'}]
                passed = stats[0]['claims'] > 0 and not contract_errors
                seal(root / 'smoke_complete.json', {'pass': passed, 'registration_sha256': sha(registration),
                    'stats': stats, 'contract_errors': contract_errors, 'other_rejections': len(errors) - len(contract_errors),
                    'not_an_accuracy_result': True, 'prefix_checkpoint_reuse_is_exact_same_contract': True})
                old.emit('TYPED SMOKE COMPLETE ' + str(passed))
            else:
                seal(root / 'build_complete.json', {'registration_sha256': sha(registration), 'stats': sorted(stats, key=lambda x: x['sample']),
                    'requires_independent_replay_and_physical_audit_before_QA': True})
                old.emit('TYPED BUILD COMPLETE ' + json.dumps(stats))
        except BaseException as exc:
            runtime.stopped.set(); seal(root / ('stop_' + str(time.time_ns()) + '.json'),
                {'utc': source.base.now(), 'type': type(exc).__name__, 'message': str(exc)[:500]}); raise


if __name__ == '__main__': main()




