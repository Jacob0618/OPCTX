"""Serialize a complete shared maintenance resolution, including its receipt."""
import threading
from . import runtime as original

CAPS=original.CAPS


class Runtime(original.Runtime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.resolution_mutex=threading.Lock()
        self.resolution_locks={}

    def post(self,stage,scope,model,method,body):
        if stage not in {'online','offline','offline_format_repair'}:
            return super().post(stage,scope,model,method,body)
        key=original.sha(dict(stage=stage,scope=scope,model=model,method=method,body=body))
        with self.resolution_mutex:
            lock=self.resolution_locks.setdefault(key,threading.Lock())
        with lock:
            return super().post(stage,scope,model,method,body)
