"""CacheBlend-style control: fixed HKVD set seeded at zero-based layer 1.

Layer-1 input has seen layer-0 attention. Both K and V deviations are measured.
Two full bootstrap layers; fixed selection thereafter (no gradual filtering or
fetch/compute overlap). This is a disclosed HF control, not the native engine.
"""
import math,torch
from transformers.cache_utils import DynamicCache
from comem.cacheblend import CacheBlend,_SparseWriteCache

class KVControl(CacheBlend):
    @torch.no_grad()
    def read(self,pack_ids,merged_kv,sink_len,query_len,recompute_ratio,stats=None,logits_tail=1):
        ids=self.cm._as_ids(pack_ids);H=ids.shape[1];L=self.num_layers
        assert L>=2
        emb=self.cm.embed_tokens(ids);positions=torch.arange(H,device=self.device).unsqueeze(0)
        mask,pe=self.cm._make_mask_and_rope(emb,positions)
        cache=DynamicCache(config=self.config)
        h=self.cm._run_layers(emb,slice(0,2),mask,positions,pe,past_key_values=cache,use_cache=True)
        fk,fv=cache.layers[1].keys,cache.layers[1].values
        dev=((fk.float()-merged_kv[1][0].float()).square()+(fv.float()-merged_kv[1][1].float()).square()).sum(dim=(1,3)).squeeze(0)
        start,end=int(sink_len),H-int(query_len);n=end-start;r=float(recompute_ratio)
        nrc=min(n,int(math.ceil(r*n)))
        R=torch.zeros(H,dtype=torch.bool,device=self.device);R[:start]=True;R[end:]=True
        if nrc:R[torch.topk(dev[start:end],nrc).indices+start]=True
        ix=torch.nonzero(R,as_tuple=False).squeeze(1)
        mixed=[(cache.layers[l].keys,cache.layers[l].values) for l in range(2)]
        mixed.extend((k.clone(),v.clone()) for k,v in merged_kv[2:])
        h=h[:,ix,:];pos=positions[:,ix];rpe=self.cm.rotary_emb(h,position_ids=pos)
        sparse=self._sparse_attn_mask(torch.arange(H,device=self.device).view(1,H)<=ix.view(-1,1))
        for l in range(2,L):
            c=_SparseWriteCache(*mixed[l],ix)
            h=self.cm._layer_out_hidden(self.cm.layers[l](h,attention_mask=sparse,position_ids=pos,position_embeddings=rpe,past_key_values=c,use_cache=True))
            mixed[l]=(c.keys,c.values)
        logits=self.cm.lm_head(self.cm.norm(h if logits_tail is None else h[:,-logits_tail:]))
        if stats is not None:stats.update({'bootstrap_full_layers':2,'deviation_layer_zero_based':1,'deviation':'squared K+V L2','n_recompute_ctx':nrc,'n_context_tokens':n,'recompute_ratio':r,'context_deviation_mean':float(dev[start:end].mean()) if n else 0,'context_deviation_max':float(dev[start:end].max()) if n else 0,'approx_context_layer_fraction':(2+(L-2)*r)/L})
        return logits,ix,mixed

def assemble(cb,chunks,query,bos):
    segments=[[bos]]+list(chunks)+[query];kvs=[];offsets=[];offset=0
    for segment in segments:
        kv,n=cb.prefill_chunk_full(segment);kvs.append(kv);offsets.append(offset);offset+=n
    merged=cb.concat_kv_reindex(kvs,offsets)
    pack=torch.cat([cb.cm._as_ids(s).reshape(-1) for s in segments]).view(1,-1)
    return pack,merged

def generate(cb,pack,merged,query_len,eos,budget,fixed=False):
    import time
    stats={};logits,_,mixed=cb.read(pack,merged,1,query_len,.15,stats=stats)
    cache=cb.decode_cache(mixed);scores=logits[0,-1].float()
    if eos is not None:scores[eos]=-float('inf')
    token=int(scores.argmax());generated=[token];torch.cuda.synchronize();first=time.perf_counter();pos=pack.shape[1]
    del logits,mixed
    for _ in range(1,budget):
        logits=cb.decode_step(token,cache,pos);pos+=1;token=int(logits[0,-1].float().argmax())
        if token==eos and not fixed:break
        generated.append(token)
    torch.cuda.synchronize();return generated,first,time.perf_counter(),stats

@torch.no_grad()
def checks(model,tok):
    from comem import CoMem
    cb=KVControl(CoMem(model,0,tokenizer=tok));bos=tok.bos_token_id
    seq=tok.encode('Alice has 42 blue books. Bob has 73 red books. '*3,add_special_tokens=False)
    chunks=[seq[:20],seq[20:40]];q=seq[40:];pack,merged=assemble(cb,chunks,q,bos)
    actual,ix,_=cb.read(pack,merged,1,len(q),1.0)
    expected=model(input_ids=pack,use_cache=False,logits_to_keep=1).logits
    error=float((actual.float()-expected.float()).abs().max());assert error<=.125 and torch.equal(actual.argmax(-1),expected.argmax(-1)),error
    # Layer-0 key deviations should be only numerical; layer-1 depends on context.
    meta={};cb.read(pack,merged,1,len(q),.15,meta);assert meta['context_deviation_max']>1e-4
    original,_,_=cb.read(pack,merged,1,len(q),.15)
    zero_query=[(k.clone(),v.clone()) for k,v in merged]
    for k,v in zero_query:k[...,-len(q):,:]=0;v[...,-len(q):,:]=0
    zero_result,_,_=cb.read(pack,zero_query,1,len(q),.15)
    assert torch.equal(original,zero_result),'query placeholders changed logits'
    # All-token recompute must also generate the same next-token trajectory.
    _,_,mixed=cb.read(pack,merged,1,len(q),1.0);cache=cb.decode_cache(mixed)
    stock=model(input_ids=pack,use_cache=True,logits_to_keep=1);refcache=stock.past_key_values;token=int(stock.logits[0,-1].argmax())
    for i in range(3):
        a=cb.decode_step(token,cache,pack.shape[1]+i)
        b=model(input_ids=torch.tensor([[token]],device=cb.device),past_key_values=refcache,use_cache=True,logits_to_keep=1)
        assert torch.equal(a.argmax(-1),b.logits.argmax(-1));refcache=b.past_key_values;token=int(b.logits[0,-1].argmax())
    return {'r1_stock_max_abs_logits':error,'r1_stock_decode_3_steps_equal':True,'zero_query_placeholders_logits_equal':True,'selection':meta,'persistent_KV_bytes_per_token':cb.kv_bytes_per_tok()}
