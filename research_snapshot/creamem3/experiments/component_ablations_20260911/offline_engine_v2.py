"""Three fixed schedules over the same causal Online transition."""
import copy
import json
from creamem3.experiments.locomo_offline_20260908 import offline_engine_v2 as old

online = old.online
sha,read,seal,StopRun = old.sha,old.read,old.seal,old.StopRun
MODES = ('online','end_only','feedback6')
SCHEMA = 'locomo.offline.component.v2'

def complete_json_suffix(raw):
    """Append only missing container closers at EOF; never alter any existing byte."""
    text = raw.strip()
    if not text or text[-1] not in '}]':
        return None
    stack=[]; quoted=False; escape=False
    for char in text:
        if quoted:
            if escape: escape=False
            elif char=='\\': escape=True
            elif char=='"': quoted=False
        elif char=='"': quoted=True
        elif char in '{[': stack.append('}' if char=='{' else ']')
        elif char in '}]':
            if not stack or stack.pop()!=char: return None
    if quoted or escape or not stack: return None
    suffix=''.join(reversed(stack))
    try:
        value=json.loads(text+suffix)
    except ValueError:
        return None
    if not isinstance(value,dict) or set(value)!={'replacements'} or not isinstance(value['replacements'],list):
        return None
    return value, {'kind':'append_container_closers_only','suffix':suffix,
                   'raw_sha256':sha(raw),'completed_sha256':sha(text+suffix)}


def parse_response(tape,raw,pack):
    value=online._parse_json_object(raw)
    if isinstance(value,dict) and set(value)=={'replacements'} and isinstance(value['replacements'],list):
        return value, {'kind':'none'}
    completed=complete_json_suffix(raw)
    if completed is not None:
        return completed
    return old.parse_offline_response(tape,raw,pack), {'kind':'legacy_string_preserving_model_repair'}




def prompt_input(base, ordinal, state):
    frontier = base['session_ids'][ordinal]
    previous = set(base['session_ids'][:ordinal])
    return online._build_prompt_input(frontier,
        current_summary=base['summary_by_session'][frontier],current_facts=base['facts_by_session'][frontier],
        prior_summaries=[base['summary_by_session'][s] for s in base['session_ids'][:ordinal]],
        prior_facts=[f for f in base['facts'] if f['session_id'] in previous],
        active_views=state['active_views'],base=base)


def advance(base, ordinal, previous, tape, *, mode):
    if mode not in MODES: raise ValueError('Unknown fixed mode')
    inp = prompt_input(base,ordinal,previous)
    state,record = old.advance(base,ordinal,previous,tape,offline_enabled=False)
    if record['prompt_input_sha256'] != sha(inp): raise StopRun('Online input replay mismatch')
    record['exposed_offline_view_ids'] = [v for v in inp['allowed_ids']['candidate_active_view_ids'] if v.startswith('OF')]
    record['extended_offline_view_ids'] = [x['target_view_id'] for x in record['online_accepted']
        if x['operation']=='extend' and x['target_view_id'].startswith('OF')]
    record['mode'] = mode
    due = (mode=='feedback6' and (ordinal+1)%old.INTERVAL==0) or (mode=='end_only' and ordinal+1==len(base['session_ids']))
    record['checkpoint_due'] = due
    record['deferred_view_ids'] = []
    if due:
        frontier=base['session_ids'][ordinal]
        active=state['active_views']; serial=state['next_offline']
        packs,deferred=old.choose_packs(active,state['dirty'],base,frontier)
        for pack in packs:
            raw=tape.chat('offline',old.PROMPT+json.dumps(pack,ensure_ascii=False,separators=(',',':')),
                          max_tokens=16384,json_mode=True)
            value,format_repair=parse_response(tape,raw,pack)
            pre=sha(active)
            active,serial,ok,bad,retired=old.apply_corrections(value,pack,active,base,frontier,serial,tape.embed)
            record['offline'].append({'pack':pack,'pre_state_sha256':pre,'post_state_sha256':sha(active),
                                       'accepted':ok,'rejected':bad,'format_repair':format_repair})
            record['tombstones'].extend(retired)
        state.update(active_views=active,next_offline=serial,dirty=sorted(deferred))
        record['post_state_sha256']=sha(active)
        record['deferred_view_ids']=sorted(deferred)
    return state,record


def build_graph(session,sample,root,runtime,registration_sha,*,mode):
    base=online._base_parts(session); state=old.empty_state()
    genesis=sha({'schema':SCHEMA,'session_sha256':sha(session),'registration':registration_sha,'mode':mode})
    chain=genesis; history=[]
    for ordinal,frontier in enumerate(base['session_ids']):
        path=root/'checkpoints'/mode/sample/f'session_{frontier:03d}.json'
        if path.exists():
            saved=read(path)
            if saved['previous_chain_sha256']!=chain or saved['registration_sha256']!=registration_sha:
                raise StopRun('Checkpoint identity/chain drift')
            tape=old.Tape(recorded=saved['tape'])
            nxt,rec=advance(base,ordinal,state,tape,mode=mode)
            if nxt!=saved['state'] or rec!=saved['record'] or tape.position!=len(tape.entries):
                raise StopRun('Saved checkpoint replay drift')
        else:
            tape=old.Tape(lambda stage,prompt,**kw:runtime.chat(sample,stage,prompt,**kw),runtime.embed)
            nxt,rec=advance(base,ordinal,state,tape,mode=mode)
            verifier=old.Tape(recorded=tape.entries)
            checked,checked_rec=advance(base,ordinal,state,verifier,mode=mode)
            if checked!=nxt or checked_rec!=rec or verifier.position!=len(tape.entries):
                raise StopRun('New transition replay drift')
            saved={'registration_sha256':registration_sha,'previous_chain_sha256':chain,
                   'state':nxt,'record':rec,'tape':tape.entries}
            saved['chain_sha256']=sha(saved); seal(path,saved)
        if saved['chain_sha256']!=sha({k:v for k,v in saved.items() if k!='chain_sha256'}):
            raise StopRun('Checkpoint chain hash drift')
        chain=saved['chain_sha256']; state=nxt; history.append(rec)
        runtime.log({'kind':'frontier_complete','mode':mode,'sample':sample,'frontier':frontier,
                     'offline_calls':len(rec['offline']),'accepted_corrections':sum(len(x['accepted']) for x in rec['offline']),
                     'exposed_corrected':len(rec['exposed_offline_view_ids']),
                     'extended_corrected':len(rec['extended_offline_view_ids'])})
    graph=copy.deepcopy(session); views=[state['active_views'][v] for v in sorted(state['active_views'])]
    graph['graph_variant']='online'
    graph['summary_nodes']=copy.deepcopy(base['summaries'])+views
    graph['edges']=copy.deepcopy(base['edges'])+online._view_edges(state['active_views'])
    graph['architecture'].update(online_update_policy=SCHEMA,event_view_membership='versioned_offline_correction_over_immutable_facts')
    graph['stats'].update(event_view_nodes=len(views),summary_nodes=len(graph['summary_nodes']),edges=len(graph['edges']),
                          cross_session_view_edges=sum(len(v['member_fact_ids']) for v in views))
    graph['offline_feedback']={'schema':SCHEMA,'mode':mode,'steps':history,'final_chain_sha256':chain,
        'genesis_sha256':genesis,'writer_fact_identity_sha256':sha(session['fact_nodes']),
        'offline_prompt_sha256':sha(old.PROMPT),'transition_replay':'pass','synchronous_prefix_publication':True}
    graph['content_fingerprint']=sha({'session':session['content_fingerprint'],'chain':chain,'state':state})
    if graph['fact_nodes']!=session['fact_nodes'] or graph['session_packets']!=session['session_packets']:
        raise StopRun('Writer changed')
    return graph

