"""G2.5 maintenance/read runtime with explicit completed-request provenance."""
import copy
from pathlib import Path
from . import transport

prior = transport.prior
read, seal, sha, StopRun = transport.read, transport.seal, transport.sha, transport.StopRun
CAPS = (16384, 24576, 32768)


def finish_reason(response):
    values = response.get('candidates') or []
    return values[0].get('finishReason') if len(values) == 1 else 'INVALID_CANDIDATE_COUNT'


class Runtime(transport.Runtime):
    def __init__(self, root, *, control, allow_model_calls, sources=()):
        super().__init__(root, control=control, allow_model_calls=allow_model_calls)
        self.sources = tuple(Path(p) for p in sources)

    def import_exact(self, identity):
        stage, key = identity['stage'], sha(identity)
        own = self.root/'provider'/stage/key
        with self.mutex:
            if (own/'final.json').exists():
                return
            for source in self.sources:
                directory = source/'provider'/stage/key
                if not (directory/'final.json').exists():
                    continue
                _, _, _, raw = prior.full.recover.completed_receipt(directory, identity)
                if stage in {'online','offline','offline_format_repair'} and finish_reason(raw['response']) != 'STOP':
                    continue
                files = {p.name: prior.file_sha(p) for p in sorted(directory.glob('*.json'))}
                seal(self.root/'imports'/stage/(key+'.json'), {'source':str(directory),
                     'request_sha256':key, 'files':files, 'new_physical_call':False})
                for name in files:
                    seal(own/name, read(directory/name))
                return

    def exact_post(self, stage, scope, model, method, body):
        identity = dict(stage=stage,scope=scope,model=model,method=method,body=body)
        self.import_exact(identity)
        return super().post(stage,scope,model,method,body)

    def post(self, stage, scope, model, method, body):
        if stage not in {'online','offline','offline_format_repair'}:
            return self.exact_post(stage,scope,model,method,body)
        if body['generationConfig']['maxOutputTokens'] != 16384:
            self._halt('Unexpected maintenance output cap')
        logical = dict(stage=stage,scope=scope,model=model,method=method,body=body)
        # Prefer a completed source resolution at any declared cap, never the
        # score or content of its answer. This preserves the historical prefix.
        selected = None
        for cap in CAPS:
            effective = copy.deepcopy(body); effective['generationConfig']['maxOutputTokens'] = cap
            identity = {**logical,'body':effective}; self.import_exact(identity)
            directory = self.root/'provider'/stage/sha(identity)
            if (directory/'final.json').exists():
                _,_,_,raw = prior.full.recover.completed_receipt(directory,identity)
                reason = finish_reason(raw['response'])
                if reason == 'STOP': selected=(identity,raw['response'],cap); break
                if reason != 'MAX_TOKENS': self._halt('Abnormal cached maintenance finish '+str(reason))
        if selected is None:
            for cap in CAPS:
                effective = copy.deepcopy(body); effective['generationConfig']['maxOutputTokens'] = cap
                identity = {**logical,'body':effective}
                response = self.exact_post(**identity)
                reason = finish_reason(response)
                if reason == 'STOP': selected=(identity,response,cap); break
                if reason != 'MAX_TOKENS': self._halt('Abnormal maintenance finish '+str(reason))
                seal(self.root/'truncations'/(sha(identity)+'.json'),{'identity':identity,
                     'logical_sha256':sha(logical),'accepted_into_memory':False,'finish_reason':reason})
            if selected is None: self._halt('Maintenance remained truncated at32768')
        identity,response,cap = selected
        seal(self.root/'maintenance_resolutions'/(sha(logical)+'.json'),{
             'logical_identity':logical,'effective_identity':identity,'effective_cap':cap,
             'finish_reason':'STOP','response_sha256':sha(response)})
        return response
