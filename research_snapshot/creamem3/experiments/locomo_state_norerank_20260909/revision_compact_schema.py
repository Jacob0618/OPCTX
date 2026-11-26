"""New failed-schema successor; unchanged local caps and state semantics.

Only provider-side nested array length limits are removed. All original
operation/support/time checks still run locally. Previous HTTP400 receipts and
first-session checkpoints are retained in the original state_build directory.
"""
import copy
import importlib.util
import types
from creamem3.experiments.locomo_state_norerank_20260909 import run, revision
from creamem3.experiments.locomo_state_norerank_20260909.runtime import Runtime,read,seal,sha,file_sha

BUILD_ROOT = run.ROOT/"revision_compact_schema_v2"


def compact_schema(value):
    result=copy.deepcopy(value)
    def visit(item):
        if isinstance(item,dict):
            item.pop("maxItems",None)
            for nested in item.values():visit(nested)
        elif isinstance(item,list):
            for nested in item:visit(nested)
    visit(result)
    return result


class CompactRuntime(Runtime):
    def structured_online(self,scope,prompt,schema):
        return super().structured_online(scope,prompt,compact_schema(schema))


def main():
    probe=read(run.ROOT/"diagnostics/schema_unbounded/result.json")
    if probe["status"]!=200:raise RuntimeError("Compact schema has not passed its tiny diagnostic")
    _,_,protocol=run.prepare()
    spec=importlib.util.spec_from_file_location("isolated_revisable_online_v2",revision.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapter=types.SimpleNamespace(**{key:getattr(run,key) for key in dir(run) if not key.startswith("__")})
    adapter.ROOT=BUILD_ROOT;module.run=adapter;module.Runtime=CompactRuntime
    record={"variant":"revision","version":"compact_schema_v2","source_sha256":file_sha(__file__),
        "algorithm_sha256":file_sha(revision.__file__),"prompt":revision.PROMPT,
        "provider_schema":compact_schema(revision.SCHEMA),"local_validation_schema":revision.SCHEMA,
        "protocol_sha256":sha(protocol),"legacy_route_sha256":file_sha(revision.legacy.__file__),
        "local_caps_unchanged":{"operations":12,"new_claims_per_operation":12,"prior_facts":80},
        "changes":"Only remove provider maxItems constraints that caused schema-state explosion;local constraints unchanged",
        "failed_attempt_preserved":str(run.ROOT/"state_build"),"new_model_rerank_calls":0}
    def register():
        seal(BUILD_ROOT/"registration.json",record)
        seal(run.ROOT/"arms/revision_build_compact_schema.json",record)
        return record
    module.register=register
    module.main()
    loaded,parent,_=run.prepare()
    for conv in loaded:
        sample=conv["sample_id"]
        source=BUILD_ROOT/"graphs/revision"/(sample+".json")
        graph_record=read(source)
        if graph_record["registration_sha256"]!=sha(record):raise RuntimeError("Revision registration drift")
        if graph_record["base_sha256"]!=sha(run.old.get_source_graph(sample,"session_strong",parent)):
            raise RuntimeError("Revision Writer source drift")
        seal(run.ROOT/"graphs/revision"/(sample+".json"),{**graph_record,"source_path":str(source),"source_sha256":sha(graph_record)})
    seal(run.ROOT/"revision_ready.json",{"registration_sha256":sha(record),"build_root":str(BUILD_ROOT),"conversations":len(loaded)})


if __name__=="__main__":main()
