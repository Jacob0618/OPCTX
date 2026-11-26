"""One common, fully charged read representation for raw and event evidence."""
from collections import defaultdict
from creamem3.experiments.mec_transfer_20260906.run_memeye import session_header
from creamem3.experiments.memeye_online_state_20260914 import read_api


def build(graph, mode, memory=None):
    if mode not in {'session','old_online','revised_online'}:raise ValueError('Unknown architecture')
    raw={f['id']:f for f in graph['fact_nodes']};units=[];headers={};requires={};rejected=[]
    def append(fid,text,kind,group,sources,metadata=None):
        if not text.strip():raise ValueError('Empty evidence')
        line=f'[{fid}] '+(metadata+' ' if metadata else '')+text
        units.append(dict(id=fid,text=text,kind=kind,group=group,cost=len(line)+1,
                          sources=sorted(set(sources)),line=line))
    for f in graph['fact_nodes']:
        group='S'+str(f['session_id']);header=session_header(f)
        if group in headers and headers[group]!=header:raise ValueError('Inconsistent session header')
        headers[group]=header
        append(f['id'],f['text'],'raw',group,f['source_dia_ids'])
    group='derived';headers[group]='[Derived event evidence; source facts below are authoritative]'
    if mode=='old_online':
        for v in graph['summary_nodes']:
            if v['node_type']!='event_view':continue
            for i,c in enumerate(v.get('summary_claims',[])):
                cid='O:'+v['id']+':'+str(i);support=list(dict.fromkeys(c['support_fact_ids']))
                if not support or not set(support)<=raw.keys():raise ValueError('Invalid old view support')
                requires[cid]=support
                append(cid,c['text'],'derived',group,[d for f in support for d in raw[f]['source_dia_ids']],
                       'status=unversioned; supports='+','.join(support)+';')
    elif mode=='revised_online':
        if memory is None:raise ValueError('Revised memory missing')
        for cid,c in memory['claims'].items():
            if c['status']=='retracted':continue
            _,receipt=read_api.select_packages(memory,raw,[{'kind':'derived','id':cid}],
                max_items=len(raw)+len(memory['claims']),char_budget=10**9)
            if cid not in receipt['derived_fact_ids']:
                rejected.append(dict(id=cid,reason=receipt['skipped']));continue
            support=receipt['raw_fact_ids'];key='N:'+cid
            deps=['N:'+x for x in c['current_dependencies']]
            requires[key]=list(dict.fromkeys(support+deps))
            append(key,c['text'],'derived',group,[d for f in support for d in raw[f]['source_dia_ids']],
                f"status={c['status']}; scope={c['scope']}; valid_from={c['valid_from'] or 'unknown'}; "
                f"valid_to={c['valid_to'] or 'open'}; supports={','.join(support)};")
        existing={u['id'] for u in units}
        # A child of a rejected dependency is not a usable evidence package.
        while True:
            invalid={cid for cid,deps in requires.items() if not set(deps)<=existing}
            if not invalid:break
            rejected += [dict(id=c,reason='unreadable derived dependency') for c in sorted(invalid)]
            units=[u for u in units if u['id'] not in invalid]
            for cid in invalid:requires.pop(cid);existing.remove(cid)
    costs={g:len(h)+1 for g,h in headers.items()}
    return dict(units=units,group_costs=costs,headers=headers,support_requires=requires,
                rejected_derived_candidates=rejected,mode=mode)


def render(bundle,ids,*,char_budget,max_items=50):
    units={u['id']:u for u in bundle['units']}
    if not ids or len(ids)!=len(set(ids)) or len(ids)>max_items or not set(ids)<=units.keys():
        raise ValueError('Invalid evidence count/identity')
    chosen=set(ids)
    for cid,deps in bundle['support_requires'].items():
        if cid in chosen and not set(deps)<=chosen:raise ValueError('Incomplete relation support')
    groups=defaultdict(list)
    for fid in ids:groups[units[fid]['group']].append(fid)
    lines=[]
    for group,members in groups.items():
        lines.append(bundle['headers'][group]);lines += [units[f]['line'] for f in members]
    text='\n'.join(lines)
    cost=sum(units[f]['cost'] for f in ids)+sum(bundle['group_costs'][g] for g in groups)-1
    if cost!=len(text) or len(text)>char_budget:raise ValueError('Actual evidence budget exceeded')
    return text,dict(raw_fact_ids=[f for f in ids if units[f]['kind']=='raw'],
        derived_ids=[f for f in ids if units[f]['kind']=='derived'],items=len(ids),chars=len(text),support_closed=True)
