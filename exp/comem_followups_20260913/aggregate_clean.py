"""Full/clean scores from the same outputs; macro over the same six datasets."""
from pathlib import Path
import gzip,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from eval.longbench import compute_f1_multi
TASKS=('narrativeqa','qasper','hotpotqa','2wikimqa','multifieldqa_en','musique')
def main():
    with gzip.open(HERE/'data/clean_samples.jsonl.gz','rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    result={};records=[]
    for file in (HERE.parent/'comem_frozen_j12_20260912/results').glob('shard_*/longbench_*.predictions.jsonl'):
        for line in file.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);sid=f"{r['task']}:{r['index']}";sample=samples[sid]
            assert r['answers']==sample['answers'] and r['selected_chunks']==sample['selected']
            rec={'id':sid,'task':r['task'],'arm':'without_lora','prediction':r['prediction'],'score':r['score']}
            records.append(rec)
    remote_complete=all((HERE/'clean_quality'/f'shard_{s:02d}'/'complete.json').exists() for s in range(4))
    if remote_complete:
        for f in (HERE/'clean_quality').glob('shard_*/predictions.jsonl'):records.extend(json.loads(l) for l in f.read_text(encoding='utf-8').splitlines())
    allscores={};assert len({(r['id'],r['arm']) for r in records})==len(records)
    for arm in sorted({r['arm'] for r in records}):
        group=[r for r in records if r['arm']==arm];assert len(group)==1150
        scored={};per_task={}
        for r in group:
            v=compute_f1_multi(r['prediction'],samples[r['id']]['answers']);assert abs(v-r['score'])<1e-10
            scored[r['id']]=v
        for task in TASKS:
            support=[r for r in samples.values() if r['task']==task];clean=[r for r in support if r['clean']]
            full=float(np.mean([scored[r['id']] for r in support])*100);c=float(np.mean([scored[r['id']] for r in clean])*100)
            per_task[task]={'full_n':len(support),'clean_n':len(clean),'full_f1':full,'clean_f1':c,'change_f1':c-full}
        result[arm]={'per_task':per_task,'full_macro_f1':float(np.mean([r['full_f1'] for r in per_task.values()])),'clean_macro_f1':float(np.mean([r['clean_f1'] for r in per_task.values()]))};allscores[arm]=scored
    # Document-cluster bootstrap within each dataset, preserving paired predictions.
    comparisons={};rng=np.random.default_rng(20260913)
    if remote_complete:
        for subset in ('full','clean'):
            for reference in ('without_lora','replay_no_adapter'):
                distributions=[];point=[]
                for task in TASKS:
                    raw=[json.loads(l) for l in (HERE.parent/'comem_frozen_j12_20260912/data/longbench'/f'{task}.jsonl').read_text(encoding='utf-8').splitlines()]
                    groups={}
                    for i,row in enumerate(raw):
                        sid=f'{task}:{i}'
                        if subset=='clean' and not samples[sid]['clean']:continue
                        groups.setdefault(row['context'],[]).append(allscores['comem'][sid]-allscores[reference][sid])
                    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);idx=rng.integers(0,len(sums),(10000,len(sums)))
                    distributions.append(sums[idx].sum(1)/counts[idx].sum(1));point.append(sums.sum()/counts.sum())
                boot=np.mean(distributions,axis=0)*100
                comparisons[f'{subset}:comem_minus_{reference}']={'mean_pp':float(np.mean(point)*100),'document_cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
    summary={'complete':remote_complete,'methods_complete':list(result),'full_n':1150,'clean_n':1053,'methods':result,'paired_macro_comparisons':comparisons,'bootstrap':'10000 paired document-cluster resamples stratified by dataset; seed20260913','missing_other_baseline_clean_predictions':True}
    (HERE/'clean_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
