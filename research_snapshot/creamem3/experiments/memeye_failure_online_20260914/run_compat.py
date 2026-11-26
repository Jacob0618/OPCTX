"""Fix pre-live registration digest reading without rewriting sealed inputs.

run.py hashes the unsealed registration payload. frozen.checked returns the
payload INCLUDING its seal field. Verify that field then remove it in a private
reader only for this registration. No selections/prompts/model settings change.
"""
from pathlib import Path
import argparse
import types
from . import run as original
from . import selection


def engine():
    path=Path(original.__file__)
    module=types.ModuleType(__package__+'._private_registration_compat')
    module.__file__=str(path); module.__package__=__package__
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    prior=module.frozen
    proxy=types.SimpleNamespace(**vars(prior))
    def checked(p,*args,**kwargs):
        obj=prior.checked(p,*args,**kwargs)
        if Path(p).resolve()==(selection.ROOT/'factorial_registration.json').resolve():
            obj=dict(obj); expected=obj.pop('seal_sha256')
            if selection.sha(obj)!=expected: raise RuntimeError('Registration integrity failure')
        return obj
    proxy.checked=checked; module.frozen=proxy
    selection.put(selection.ROOT/'runner_compat_registration.json',{
        'base_runner_sha256':selection.file_sha(path),'compat_runner_sha256':selection.file_sha(Path(__file__)),
        'purpose':'verify and exclude the registration seal before hashing its payload',
        'candidate_or_context_or_model_changes':False,'original_artifacts_unchanged':True})
    return module


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--allow-model-calls',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); engine().run(args.allow_model_calls,args.smoke)
