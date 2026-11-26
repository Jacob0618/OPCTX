"""Explicit inspected-request repair; no generation or scoring retry."""
from dataclasses import asdict
import sys
from . import run as r
from creamem3.experiments.memeye_lite_score_repair_20260923.run import parse_verdict

def main(arm,rid,request):
    root=r.ROOT/'runs'/arm;key=r.b.sha(rid);sp=root/'scores'/f'{key}.json'
    assert not sp.exists()
    fail=r.b.read(root/'failures'/f'{key}.json');pred=r.b.read(root/'predictions'/f'{key}.json')
    assert fail['message']=='invalid MGTM judge JSON score'
    response=r.b.read(root/'judge_calls'/request[:2]/request/'response.json');raw=response['response']
    assert raw['candidates'][0]['finishReason']=='STOP'
    text=r.rt.prior.n.g._normalize_text(raw).strip();verdict,repair=parse_verdict(text,{r.b.sha(text)})
    assert repair and repair['score_unchanged'] and repair['reasoning_unchanged']
    receipt=r.b.save(r.ROOT/'repairs'/f'{arm}_{key}.json',dict(row_id=rid,arm=arm,request_sha256=request,
        source_failure_sha256=fail['seal_sha256'],response_sha256=response['response_sha256'],repair=repair,
        code_sha256=r.b.file_sha(__file__),new_model_calls=0))
    gold=r.b.read(r.source.SOURCE/'scoring/open.json')['rows'][rid]
    out=r.rt.prior.Reply(text,request,response['response_sha256'],r.old.JUDGE,raw.get('modelVersion',''),asdict(r.rt.prior.n.g._normalize_usage(raw)),True)
    r.b.save(sp,dict(row_id=rid,task=pred['task'],x=gold['x'],y=gold['y'],raw_score=verdict.raw_score,
        mapped_score=verdict.mapped_score,reasoning=verdict.reasoning,judge=r.old.JUDGE,generation=asdict(out),
        elapsed_s=None,prediction_sha256=pred['seal_sha256'],repair_sha256=receipt['seal_sha256']))
    print('LOSSLESS JUDGE REPAIR',arm,rid,'new_calls=0')

if __name__=='__main__':main(*sys.argv[1:])
