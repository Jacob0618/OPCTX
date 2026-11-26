from pathlib import Path
from creamem3.experiments.memeye_evidence_20260914 import core as authority
from creamem3.experiments.component_ablations_20260911 import math_selector as selector
from creamem3.experiments.mec_transfer_20260906 import run_memeye as adapter
from creamem3.experiments.memeye_failure_online_20260914 import run as reader

ROOT=Path(r'D:\creamem3_cache\memeye_budget_online_20260914_r1')
DEPTHS=(5,10,20,50)
ROUTES={'wide':(10,20,20),'narrow':(2,5,5)}
KINDS=('session','online')
MODES=tuple(f'{r}_k{k}_{a}' for k in DEPTHS for r in ('wide','narrow') for a in KINDS)
sha,file_sha,checked,put=authority.sha,authority.file_sha,authority.checked,authority.put
direct,frozen=authority.direct,authority.frozen

def split(mode):
    if mode not in MODES: raise ValueError('Unknown arm')
    route,k,kind=mode.split('_'); return route,int(k[1:]),kind

def hashes():
    from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy
    paths=[Path(__file__).with_name(n) for n in ('config.py','prepare.py','run.py','verify.py')]
    paths += [Path(m.__file__) for m in (authority,selector,selector.base,selector.numerical,
        selector.lexical,adapter,adapter.raw,reader,frozen,direct,candidate_context_policy,
        direct.graph_memory,direct.g35_runtime,direct.retry5)]
    return {str(p.resolve()):file_sha(p) for p in paths}
