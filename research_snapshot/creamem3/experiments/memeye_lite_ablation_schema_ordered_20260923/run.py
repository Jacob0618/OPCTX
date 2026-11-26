"""Order-only schema compilation compatibility, independent registered entry."""
from copy import deepcopy
from pathlib import Path
import psutil
from ..memeye_lite_ablation_schema_20260923 import run as parent

base,b,rt=parent.base,parent.b,parent.rt
HERE=Path(__file__).resolve().parent
PARENT_REC=parent.REC
REC=base.ROOT/'v8_schema_ordered_amendment_r2'
PARENT_HASHES=parent.hashes
UNORDERED=deepcopy(parent.SCHEMA)
NEW_MODULE=__package__+'.run'


def ordered(value):
    if isinstance(value,list): return [ordered(v) for v in value]
    if not isinstance(value,dict): return value
    out={k:ordered(v) for k,v in value.items()}
    if out.get('type')=='object': out['propertyOrdering']=list(out['properties'])
    return out


SCHEMA=ordered(UNORDERED)


def strip_order(value):
    if isinstance(value,list):return [strip_order(v) for v in value]
    if not isinstance(value,dict):return value
    return {k:strip_order(v) for k,v in value.items() if k!='propertyOrdering'}


def hashes():
    return PARENT_HASHES() | {str(p):b.file_sha(p) for p in sorted([*HERE.glob('*.py'),HERE/'protocol.md'])}


def validate_parent():
    reg=b.read(PARENT_REC/'registration.json')
    base.require(reg['code_hashes']==PARENT_HASHES(),'Parent amendment source drift')
    base.require(reg['parent_sha256']==base.validate()['seal_sha256'],'Original contract drift')
    for p,h in reg['protected'].items():base.require(b.file_sha(p)==h,'Protected output changed '+p)
    base.require(strip_order(SCHEMA)==reg['response_schema'],'Not purely property order')
    return reg


def register():
    reg=validate_parent()
    error=PARENT_REC/'http400_diagnostic/presend_retry_2/http_error.json'
    actual=b.read(error)
    base.require(actual['status']==400 and 'too many states' in actual['body'],'Wrong diagnosis')
    return b.save(REC/'registration.json',dict(schema='memeye.v8.schema_order_only.v2',
        parent_sha256=reg['seal_sha256'],base_sha256=reg['parent_sha256'],code_hashes=hashes(),
        response_schema=SCHEMA,diagnostic_path=str(error),diagnostic_sha256=b.file_sha(error),
        change='propertyOrdering only; all constraints and prompts unchanged',
        models=rt.MODELS,authorization=reg['authorized'],original_failures_preserved=True,
        successful_outputs_not_retried=True,max_total_workers=4))


def validate():
    old=validate_parent();reg=b.read(REC/'registration.json')
    base.require(reg['parent_sha256']==old['seal_sha256'],'Parent registration drift')
    base.require(reg['code_hashes']==hashes() and reg['response_schema']==SCHEMA,'Ordered amendment drift')
    base.require(b.file_sha(reg['diagnostic_path'])==reg['diagnostic_sha256'],'Diagnosis drift')
    return reg


def active_workers():
    modules={parent.OLD_MODULE,'creamem3.experiments.memeye_lite_ablation_schema_20260923.run',NEW_MODULE}
    out=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            if '-m' not in cmd:continue
            i=cmd.index('-m')
            if len(cmd)>i+2 and cmd[i+1] in modules and cmd[i+2] in ('live','probe'):out.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return out


def build_v8(root,graph,gen,embed):
    reg=b.read(REC/'registration.json')
    for old in (root,root/'schema_v8'):
        if (old/'v8/complete.json').exists():
            b.save(root/'v8_ordered_provenance.json',dict(mode='retain_successful_pre_ordering_update',
                complete_path=str(old/'v8/complete.json'),sha256=b.file_sha(old/'v8/complete.json'),
                amendment_sha256=reg['seal_sha256']))
            return parent.ORIGINAL_BUILD_V8(old,graph,gen,embed)
    dest=root/'schema_ordered_v8';inherited={}
    for step in graph['online_fusion']['steps']:
        name=f"{step['session_id']}.json"
        paths=[d/'v8/checkpoints'/name for d in (root/'schema_v8',root)]
        original=next((p for p in paths if p.exists()),None)
        if original is None:break
        data=b.read(original);data.pop('seal_sha256',None)
        b.save(dest/'v8/checkpoints'/name,data);inherited[str(original)]=b.file_sha(original)
    b.save(dest/'provenance.json',dict(amendment_sha256=reg['seal_sha256'],
        inherited_successful_prefix=inherited,original_root=str(root),schema_sha256=b.sha(SCHEMA)))
    gen=parent.SchemaRuntime(dest/'calls',gen.model,gen.allow,reg['seal_sha256'])
    return parent.ORIGINAL_BUILD_V8(dest,graph,gen,embed)


def configure():
    parent.REC=REC;parent.SCHEMA=SCHEMA;parent.NEW_MODULE=NEW_MODULE
    parent.register=register;parent.validate=validate
    parent.active_workers=active_workers;parent.build_v8=build_v8


if __name__=='__main__':
    configure();parent.main()
