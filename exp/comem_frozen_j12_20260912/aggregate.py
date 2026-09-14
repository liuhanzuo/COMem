"""Require the complete 5,250-example grid, rescore predictions, and aggregate."""
from pathlib import Path
import ast, collections, gzip, importlib.util, json, math, re, statistics, string
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
lengths=('8k','16k','32k','64k','128k')
expected={}
for task in ('narrativeqa','qasper','hotpotqa','2wikimqa','multifieldqa_en','musique'):
    expected[('longbench',task,'')]=150 if task=='multifieldqa_en' else 200
for length in ('0k','1k','2k','4k','8k','16k','32k'):
    for task in ('qa1','qa2','qa5'): expected[('babilong',task,length)]=100
for length in lengths:
    expected[('longeval','lines',length)]=100
    for task in ('niah_single_2','niah_multikey_1','variable_tracking'): expected[('ruler',task,length)]=100
# Extract only the official English QA metric's three pure functions so the CPU
# checker need not install unrelated Chinese/ROUGE/code-similarity dependencies.
official=HERE/'vendor/longbench/metrics.py'
tree=ast.parse(official.read_text(encoding='utf-8'))
names={'normalize_answer','f1_score','qa_f1_score'}
metric_tree=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
assert len(metric_tree.body)==3
ns={'re':re,'string':string,'Counter':collections.Counter}
exec(compile(metric_tree,str(official),'exec'),ns)
spec=importlib.util.spec_from_file_location('babi_metric',HERE/'vendor/babilong/babilong/metrics.py')
babi=importlib.util.module_from_spec(spec); spec.loader.exec_module(babi)
grouped=collections.defaultdict(dict)
workers=[]
for shard in range(4):
    folder=HERE/'results'/f'shard_{shard:02d}'
    done=json.loads((folder/'complete.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    check=json.loads((folder/'correctness.json').read_text())
    assert done['complete'] and done['cells']==47
    assert meta['adapter'] is None and meta['split']==12 and meta['topk']==12 and meta['hop_topk']==4
    assert not meta['smoke'] and check['j0_top1_equal'] and check['j12_outputs_equal']
    total=0
    files=list(folder.glob('*.predictions.jsonl')); assert len(files)==47
    for f in files:
        rows=[json.loads(line) for line in f.read_text(encoding='utf-8').splitlines() if line]
        source_path=f.with_name(f.name.replace('.predictions.jsonl','.inputs.jsonl.gz'))
        with gzip.open(source_path,'rt',encoding='utf-8') as inputs:
            source_rows=0
            for row,line in zip(rows,inputs,strict=True):
                source_rows+=1
                source=json.loads(line)
                key=(row['benchmark'],row['task'],row['length']); index=row['index']
                assert key in expected and 0<=index<expected[key] and index%4==shard
                assert index not in grouped[key] and row['status']=='ok'
                assert source['index']==index and source['source_tokens']==len(source['input_ids'])==row['source_tokens']
                assert source['selected_chunks']==row['selected_chunks'] and len(row['selected_chunks'])<=12
                assert row['selected_chunks']==sorted(set(row['selected_chunks']))
                pred,answers=row['prediction'],row['answers']
                if key[0]=='longbench': value=max(ns['qa_f1_score'](pred,a) for a in answers)
                elif key[0]=='babilong': value=float(babi.compare_answers(answers[0],pred,row['question'],babi.TASK_LABELS[key[1]]))
                elif key[0]=='longeval':
                    found=re.search(r'\d{4,}',pred); value=float((found.group(0) if found else '')==answers[0])
                else: value=sum(a.lower() in pred.lower() for a in answers)/len(answers)
                assert math.isclose(value,row['score'],rel_tol=0,abs_tol=1e-12),(key,index,value,row['score'])
                grouped[key][index]=value
        assert source_rows==len(rows)
        total+=len(rows)
    assert total==done['records']
    workers.append({'metadata':meta,'correctness':check,'complete':done})
assert set(grouped)==set(expected)
cell_scores={}
for key,n in expected.items():
    assert set(grouped[key])==set(range(n)),key
    cell_scores['/'.join(key)]={'benchmark':key[0],'task':key[1],'length':key[2],'n':n,'score':100*statistics.mean(grouped[key].values())}
scores={bench:statistics.mean(v['score'] for v in cell_scores.values() if v['benchmark']==bench) for bench in ('ruler','longeval','longbench','babilong')}
scores['locomo']=24.52
scores['average']=statistics.mean(scores.values())
summary={'complete':True,'formal_examples':sum(expected.values()),'cells':cell_scores,'scores':scores,'workers':workers,'locomo_source':{'status':'historical full-set frozen j12 Judge','score':24.52,'n':1986,'source':'arXiv tab_depth_tradeoff.tex and designated ARR depth ablation'},'new_input_pairing':'Synthetic inputs sampled independently from the adapted overview row','scope':'new frozen j12 scores, no new adapter accuracy or infrastructure timing claims'}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'scores':scores,'new_examples':sum(expected.values()),'complete_cells':len(cell_scores)},indent=2))
