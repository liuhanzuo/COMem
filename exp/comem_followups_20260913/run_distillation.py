"""Post-training diagnostic, no parameter updates; all vocabulary probabilities."""
from pathlib import Path
import argparse,contextlib,gzip,json,os,platform,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]),str(HERE/'reference')]
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from train_8b_baseline_executed import PublishedReader
from common import attach,dump

def metrics(t,s,labels):
    tp=torch.log_softmax(t.float(),-1);sq=torch.log_softmax(s.float(),-1)
    inds=t.topk(64,-1).indices
    pt=tp.gather(-1,inds);qs=sq.gather(-1,inds)
    ptc=pt-torch.logsumexp(pt,-1,keepdim=True);qsc=qs-torch.logsumexp(qs,-1,keepdim=True)
    forward=(tp.exp()*(tp-sq)).sum(-1);reverse=(sq.exp()*(sq-tp)).sum(-1)
    cforward=(ptc.exp()*(ptc-qsc)).sum(-1);creverse=(qsc.exp()*(qsc-ptc)).sum(-1)
    return {'teacher_top64_mass':pt.exp().sum(-1),'student_teacher_top64_mass':qs.exp().sum(-1),'full_kl_teacher_student':forward,'full_kl_student_teacher':reverse,'conditional_kl_teacher_student':cforward,'conditional_kl_student_teacher':creverse,'conditional_training_loss':.6*cforward+.4*creverse,'teacher_nll':-tp.gather(-1,labels.unsqueeze(-1)).squeeze(-1),'student_nll':-sq.gather(-1,labels.unsqueeze(-1)).squeeze(-1),'top1_agreement':t.argmax(-1).eq(s.argmax(-1)).float(),'teacher_top1_correct':t.argmax(-1).eq(labels).float(),'student_top1_correct':s.argmax(-1).eq(labels).float()}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(42)
    out=HERE/('distillation_smoke' if a.smoke else 'distillation');out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=attach(base,a.adapter);reader=PublishedReader(model,12,grad_checkpoint=False)
    bos=tok.bos_token_id if tok.bos_token_id is not None else model.config.bos_token_id
    dump(out/'metadata.json',{'job':os.environ['SLURM_JOB_ID'],'node':platform.node(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'teacher':'adapter off, full causal 4097 positions including BOS','students':['independent-chunk lower12, adapter off','independent-chunk lower12, principal adapter on'],'labels':'512 next tokens following each query position, including one held-out final token','bos':bos,'training':False})
    with gzip.open(HERE/'data/distillation_windows.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    if a.smoke:rows=rows[:1]
    began=time.time()
    with (out/'windows.jsonl').open('w',encoding='utf-8') as output,gzip.open(out/'token_metrics.jsonl.gz','wt',encoding='utf-8') as detail:
        for row in rows:
            ids=torch.tensor(row['input_and_next_ids'],dtype=torch.long);window=ids[:4096];labels=ids[3585:4097].to('cuda').unsqueeze(0)
            with wrapper.disable_adapter():th=reader.hidden(window,bos,512,7,teacher=True)
            for arm in ('without_lora','comem'):
                with wrapper.disable_adapter() if arm=='without_lora' else contextlib.nullcontext():sh=reader.hidden(window,bos,512,7)
                accum={}
                for start in range(0,512,32):
                    tl=model.lm_head(th[:,start:start+32]);sl=model.lm_head(sh[:,start:start+32]);m=metrics(tl,sl,labels[:,start:start+32])
                    if row is rows[0] and start==0:
                        identity=metrics(tl,tl,labels[:,:32]);assert identity['full_kl_teacher_student'].abs().max()<1e-6 and identity['conditional_training_loss'].abs().max()<1e-6
                    for k,v in m.items():
                        assert torch.isfinite(v).all(),k
                        accum.setdefault(k,[]).extend(v.flatten().cpu().tolist())
                rec={'id':row['id'],'book':row['book'],'arm':arm,'n_positions':512,**{k:sum(v)/len(v) for k,v in accum.items()}}
                output.write(json.dumps(rec)+'\n');output.flush();detail.write(json.dumps({'id':row['id'],'arm':arm,'metrics':accum})+'\n');detail.flush();del sh
            del th
            dump(out/'progress.json',{'windows':rows.index(row)+1,'total':len(rows),'elapsed_s':time.time()-began});print(row['id'],flush=True)
    dump(out/'complete.json',{'complete':True,'windows':len(rows),'elapsed_s':time.time()-began})
if __name__=='__main__':main()
