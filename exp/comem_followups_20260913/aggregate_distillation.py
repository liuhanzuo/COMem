from pathlib import Path
import gzip,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    root=HERE/'distillation';assert json.loads((root/'complete.json').read_text())['windows']==100
    rows=[json.loads(l) for l in (root/'windows.jsonl').read_text().splitlines()]
    assert len(rows)==200 and len({(r['id'],r['arm']) for r in rows})==200
    keys=[k for k in rows[0] if k not in ('id','book','arm','n_positions')]
    books=sorted({r['book'] for r in rows});assert len(books)==50
    rng=np.random.default_rng(20260913);ix=rng.integers(0,50,size=(10000,50));arms={};arrays={}
    details={a:{k:[] for k in keys} for a in ('without_lora','comem')}
    with gzip.open(root/'token_metrics.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            for k,v in r['metrics'].items():details[r['arm']][k].extend(v)
    for arm in ('without_lora','comem'):
        chosen=[r for r in rows if r['arm']==arm];assert len(chosen)==100
        x=np.array([[np.mean([r[k] for r in chosen if r['book']==b]) for k in keys] for b in books]);arrays[arm]=x
        boot=x[ix].mean(axis=1);result={}
        for col,k in enumerate(keys):
            tok=np.array(details[arm][k]);assert len(tok)==51200
            result[k]={'mean':float(x[:,col].mean()),'book_bootstrap_95ci':np.quantile(boot[:,col],[.025,.975]).tolist(),'token_quantiles_01_10_50_90_99':np.quantile(tok,[.01,.1,.5,.9,.99]).tolist()}
        result['perplexity']=math.exp(result['student_nll']['mean']);result['teacher_perplexity']=math.exp(result['teacher_nll']['mean']);arms[arm]=result
    delta=arrays['comem']-arrays['without_lora'];bdelta=delta[ix].mean(axis=1)
    result={'complete':True,'books':50,'windows':100,'positions_per_arm':51200,'resamples':10000,'unit':'book','arms':arms,'paired_comem_minus_without_lora':{k:{'mean':float(delta[:,j].mean()),'book_bootstrap_95ci':np.quantile(bdelta[:,j],[.025,.975]).tolist()} for j,k in enumerate(keys)}}
    (HERE/'distillation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({arm:{k:(v['mean'] if isinstance(v,dict) else v) for k,v in result['arms'][arm].items()} for arm in arms},indent=2))
if __name__=='__main__':main()
