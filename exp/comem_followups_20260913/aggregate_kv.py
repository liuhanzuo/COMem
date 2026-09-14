from pathlib import Path
import argparse,gzip,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from eval import longeval,longbench
CELLS=('longeval_8k','longeval_16k','longeval_32k','qasper')
def main():
    with gzip.open(HERE/'data/kv_samples.jsonl.gz','rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    if all((HERE/'kv_quality'/f'shard_{i:02d}'/'complete.json').exists() for i in range(4)):
        records=[]
        for p in (HERE/'kv_quality').glob('shard_*/predictions.jsonl'):records.extend(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
        assert len(records)==3000 and len({(r['id'],r['method'],r['adapter_on']) for r in records})==3000
        for r in records:
            sample=samples[r['id']];assert r['status']=='ok' and r['selected']==sample['selected']
            score=float(longeval.extract_prediction(r['prediction'])==sample['answers'][0]) if sample['benchmark']=='longeval' else longbench.compute_f1_multi(r['prediction'],sample['answers'])
            assert abs(score-r['score'])<1e-12
        result={};paired={};rng=np.random.default_rng(20260913)
        for on in (False,True):
            for method in ('replay','comem','chunkkv'):
                per={}
                for c in CELLS:
                    group=[r for r in records if r['cell']==c and r['method']==method and r['adapter_on']==on];assert len(group)==(200 if c=='qasper' else 100)
                    per[c]={'n':len(group),'score':float(np.mean([r['score'] for r in group])*100)}
                per['longeval_macro']=float(np.mean([per[c]['score'] for c in CELLS[:3]]));result[f'{method}:adapter_{"on" if on else "off"}']=per
            for ref in ('replay','chunkkv'):
                base={r['id']:r['score'] for r in records if r['method']==ref and r['adapter_on']==on};ours={r['id']:r['score'] for r in records if r['method']=='comem' and r['adapter_on']==on}
                diffs={c:np.array([ours[s['id']]-base[s['id']] for s in samples.values() if s['cell']==c]) for c in CELLS}
                for name,cells in [('longeval_macro',CELLS[:3]),('qasper',['qasper'])]:
                    boots=[d[rng.integers(0,len(d),(10000,len(d)))].mean(1) for d in [diffs[c] for c in cells]]
                    if name=='qasper':
                        source=[json.loads(l) for l in (HERE.parent/'comem_frozen_j12_20260912/data/longbench/qasper.jsonl').read_text(encoding='utf-8').splitlines()]
                        clusters={}
                        for sample in samples.values():
                            if sample['cell']=='qasper':
                                idx=int(sample['id'].split('_')[-1]);clusters.setdefault(source[idx]['context'],[]).append(ours[sample['id']]-base[sample['id']])
                        sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()]);draw=rng.integers(0,len(sums),(10000,len(sums)))
                        boots=[sums[draw].sum(1)/counts[draw].sum(1)]
                    paired[f'{name}:comem_minus_{ref}:adapter_{on}']={'delta_pp':float(np.mean([diffs[c].mean() for c in cells])*100),'paired_stratified_bootstrap_95ci':np.quantile(np.mean(boots,axis=0)*100,[.025,.975]).tolist()}
        summary={'complete':True,'records':3000,'unique_samples':500,'methods':result,'paired':paired,'bootstrap':'10000 sample resamples within length for LongEval; Qasper source-document cluster resamples; seed20260913'}
        (HERE/'kv_quality_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('KV quality complete:3000')
    if all((HERE/'kv_cost'/f'process_{i:02d}'/'complete.json').exists() for i in (1,2,3)):
        rows=[]
        for p in (HERE/'kv_cost').glob('process_*/records.jsonl'):rows.extend(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
        rows=[r for r in rows if not r['warmup']];assert len(rows)==1080
        assert len({(r['id'],r['method'],r['adapter_on'],r['process'],r['rep']) for r in rows})==1080
        fields=('prepare_ms','write_ms','selection_ms','fetch_ms','ttft_ms','online_ms','e2e_ms','persistent_bytes','peak_allocated_bytes','peak_reserved_bytes')
        for r in rows:
            if r['status']!='ok':assert r['status']=='OOM';continue
            assert len(r['generated_ids'])==128 and r['peak_reserved_bytes']<=28_000_000_000 and r['peak_allocated_bytes']<=28_000_000_000
            assert abs(r['prepare_ms']+r['online_ms']-r['e2e_ms'])<1e-5 and r['online_ms']>=r['ttft_ms']
            s=samples[r['id']];assert r['selected']==s['selected']
            computed=float(longeval.extract_prediction(r['prediction'])==s['answers'][0]) if s['benchmark']=='longeval' else longbench.compute_f1_multi(r['prediction'],s['answers']);assert abs(computed-r['score'])<1e-12
        summary={}
        for cell in CELLS:
            summary[cell]={}
            for method in ('replay','comem','chunkkv'):
                for on in (False,True):
                    group=[r for r in rows if r['cell']==cell and r['method']==method and r['adapter_on']==on];assert len(group)==45
                    ok=[r for r in group if r['status']=='ok'];n_oom=len(group)-len(ok)
                    d={'n_formal':45,'n_oom':n_oom,'unique_examples':5}
                    if ok:
                        d['medians']={k:float(np.median([r[k] for r in ok])) for k in fields};d['ranges']={k:[min(r[k] for r in ok),max(r[k] for r in ok)] for k in fields};d['process_medians']={str(i):{k:float(np.median([r[k] for r in ok if r['process']==i])) for k in fields} for i in (1,2,3)}
                        unique={}
                        for r in ok:
                            if r['id'] in unique:assert unique[r['id']]['generated_ids']==r['generated_ids'],'repeat output changed'
                            unique[r['id']]=r
                        d['local_subset_score']=float(np.mean([r['score'] for r in unique.values()])*100)
                    summary[cell][f'{method}:adapter_{"on" if on else "off"}']=d
        (HERE/'kv_cost_summary.json').write_text(json.dumps({'complete':True,'formal_records':1080,'aggregation':'median over 3 processes x 5 examples x 3 repetitions per cell; process medians/ranges retained','cells':summary},indent=2),encoding='utf-8');print('KV cost complete:1080')
if __name__=='__main__':main()
