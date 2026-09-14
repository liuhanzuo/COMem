"""Shared exact sample construction and residual-repair paths; no training."""
from pathlib import Path
import gc, json, os, sys, time
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import torch
from comem import CoMem
from comem.selectors import iter_bm25_indices
from eval import longeval, longbench

ARMS = ('replay', 'w0', 'w32')

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')
    temp.replace(path)

def attach(base, path):
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):
        wrapper=PeftModel.from_pretrained(base,str(path),autocast_adapter_dtype=True)
    return wrapper, wrapper.base_model.model.eval()

def parts(row):
    chunks=list(torch.tensor(row['input_ids'],dtype=torch.long).split(512))
    assert len(chunks)>1
    return chunks[:-1],chunks[-1].tolist()

def selection(chunks, row):
    ix=iter_bm25_indices(chunks,row['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
    assert ix==row['selected'], (row['id'],ix,row['selected'])
    return ix

def write_one(cm, source, index, width):
    start=index*512; end=min(start+512,len(source))
    left=max(0,start-width)
    assert 0<=left<=start<end<=len(source)
    states=cm.write_chunk(source[left:end])[...,start-left:,:].contiguous()
    assert states.shape[1]==end-start
    return states

def quality_ids(ids,eos_id,budget):
    ids=list(ids[:budget])
    if eos_id in ids: ids=ids[:ids.index(eos_id)]
    return ids

def score(row,text):
    if row['benchmark']=='longeval':
        return float(longeval.extract_prediction(text)==row['answers'][0])
    return float(longbench.compute_f1_multi(text,row['answers']))

def timed_stamp():
    torch.cuda.synchronize()
    return time.perf_counter()

def decode(reader,sink,states,query,eos_id,budget,fixed=False):
    q,bottom,qpos=reader.write_prefill(query)
    logits,top,ppos=reader.read_prefill(sink,states,q)
    first=timed_stamp()
    next_logits=logits[0,-1].float()
    if eos_id is not None: next_logits[eos_id]=float('-inf')
    token=int(next_logits.argmax().item()); generated=[token]
    for _ in range(1,budget):
        logits=reader.decode_step(token,bottom,top,qpos,ppos)
        qpos+=1; ppos+=1
        token=int(logits[0,-1].float().argmax().item())
        if token==eos_id and not fixed: break
        generated.append(token)
    ended=timed_stamp()
    return generated,first,ended

@torch.inference_mode()
def check_paths(wrapper,model,tok):
    cm,rp=CoMem(model,12,tokenizer=tok),CoMem(model,0,tokenizer=tok)
    bos=tok.bos_token_id if tok.bos_token_id is not None else model.config.bos_token_id
    # Qwen tokenizer omits BOS; use the configured BOS explicitly in every arm.
    tok.bos_token_id=bos
    eos=tok.eos_token_id
    ids=[bos]+tok.encode('Alice keeps 42. Bob keeps 73. '*8,add_special_tokens=False)
    teacher={}
    # Current teacher is adapter-off, j=0, full causal pack; student remains adapted.
    with wrapper.disable_adapter():
        full=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits
        split=rp.read_core(rp.write_chunk(ids[:1]),[rp.write_chunk(ids[1:35])],rp.write_chunk(ids[35:]),logits_tail=1)
        teacher={'max_abs_difference':float((full-split).abs().max()),'top1_equal':bool(full.argmax(-1).eq(split.argmax(-1)).all())}
        assert teacher['top1_equal'] and teacher['max_abs_difference']<=.125,teacher
        del full,split
    prompt=('Alice keeps the number 42 in a blue notebook. '*110)+' What number does Alice keep? Answer:'
    full_ids=tok.encode(prompt,add_special_tokens=True)
    row={'id':'parity','input_ids':full_ids,'question_ids':tok.encode('Alice number',add_special_tokens=False)}
    chunks,query=parts(row)
    row['selected']=iter_bm25_indices(chunks,row['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
    source=torch.cat(chunks)
    checks={'teacher':teacher,'bos_token_id':bos,'arms':{}}
    for arm in ARMS:
        reader=rp if arm=='replay' else cm
        states=[rp.write_chunk(chunks[i]) if arm=='replay' else write_one(cm,source,i,32 if arm=='w32' else 0) for i in row['selected']]
        sink=reader.write_chunk([bos])
        actual,_,_=decode(reader,sink,states,query,eos,8)
        expected=reader._decode_from_pack(sink,states,query,eos,8,False)
        assert actual==expected,(arm,actual,expected)
        data={'cached_recompute_equal':True,'state_positions':sum(s.shape[1] for s in states)}
        if arm!='w32':
            trace={'capture_step_logits':True}
            original=reader.generate_from_ids(torch.tensor([full_ids],device='cuda'),chunk_size=512,max_new_tokens=8,selector='iter_bm25',topk=12,sink_tokens='bos',bare_question_ids=row['question_ids'],iter_hop_topk=4,iter_rounds=0,stats=trace)
            assert trace['generated_ids']==actual,(arm,trace['generated_ids'],actual)
            assert original==tok.decode(actual,skip_special_tokens=True).strip()
            data['released_path_equal']=True
            del trace
        checks['arms'][arm]=data
        del states,sink
    checks['same_persistent_state_shape']=checks['arms']['w0']['state_positions']==checks['arms']['w32']['state_positions']
    assert checks['same_persistent_state_shape']
    gc.collect();torch.cuda.empty_cache()
    return checks
