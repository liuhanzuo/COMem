"""CoMem correctness self-test (CPU, tiny random Qwen3, no weights).

Run: ``python -m comem.selftest``

Gates (all must pass; fp32, tolerance 1e-4):

  (A) j=0 packing == full forward. The write/read packing path at ``resume_j=0``
      reproduces a stock ``model(input_ids=packed)`` forward — because j=0 write is
      a bare embedding lookup and j=0 read resumes ``layers[0:]`` over the
      concatenated embeddings with contiguous positions (RoPE lives inside the
      layers). This is the load-bearing correctness claim of the resume primitive.

  (B) resume_forward_ids == full forward at several ``j`` on a single sequence
      (the resume identity holds at every depth when the whole sequence is one
      contiguous chunk).

  (C) encode + generate == generate_from_ids, token-for-token. The ergonomic
      ``encode_ids`` / ``generate_ids`` pair reproduces the monolithic reference
      path exactly when the context is a multiple of ``chunk_size`` and the query
      is one chunk (identical packing).

  (D) resumed-band KV-cache decode == recompute decode, token-for-token (same
      generated ids, max|logit diff| < tol). Speed must not change the output.
"""
from __future__ import annotations

import torch

from .model import CoMem


def build_tiny_qwen3(n_layers=6, hidden=64, vocab=256, seed=0):
    """Tiny random Qwen3 (no weights) for CPU plumbing correctness."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=8192,
        sliding_window=None,
        use_sliding_window=False,
        attn_implementation="sdpa",
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval(), cfg


class _TinyTok:
    """Minimal tokenizer stand-in (bos=1, EOS disabled so decode runs full budget)."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.bos_token_id = 1
        self.eos_token_id = None
        self.pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


@torch.no_grad()
def run(n_layers=6, hidden=64, vocab=256, chunk_size=8, tol=1e-4, verbose=True):
    torch.manual_seed(0)
    model, cfg = build_tiny_qwen3(n_layers, hidden, vocab)
    model = model.to(torch.float32).eval()
    device = torch.device("cpu")
    tok = _TinyTok(vocab)
    L = n_layers
    bos = tok.bos_token_id

    def rid(n):
        return torch.randint(2, vocab, (1, n), device=device)

    # ---- (A) + (B): j=0 packing == full forward, resume identity ----
    qc0 = CoMem(model, resume_j=0, tokenizer=tok)
    sink_ids = torch.tensor([[bos]], device=device)
    c1, c2, c3 = rid(7), rid(5), rid(9)
    q = rid(4)
    packed = torch.cat([sink_ids, c1, c2, c3, q], dim=1)
    ref = qc0.full_forward_logits(packed)

    sink_hj = qc0.write_chunk(sink_ids)
    ctx_hj = [qc0.write_chunk(c) for c in (c1, c2, c3)]
    q_hj = qc0.write_chunk(q)
    out_pack = qc0.read(sink_hj, ctx_hj, q_hj)
    diff_pack = (out_pack.float() - ref.float()).abs().max().item()

    out_resume = qc0.resume_forward_ids(packed)
    diff_resume = (out_resume.float() - ref.float()).abs().max().item()

    diffs_j = {}
    for j in (1, L // 2, L):
        qcj = CoMem(model, resume_j=j, tokenizer=tok)
        outj = qcj.resume_forward_ids(packed)
        diffs_j[j] = (outj.float() - ref.float()).abs().max().item()

    # ---- (C) + (D): encode+generate == generate_from_ids; kv == recompute ----
    resume_j = max(1, L // 2)
    qc = CoMem(model, resume_j=resume_j, tokenizer=tok)
    n_ctx = 5
    ctx_ids = rid(n_ctx * chunk_size)          # exact multiple of chunk_size
    q_len = chunk_size - 2
    query_ids = rid(q_len)
    full_ids = torch.cat([ctx_ids, query_ids], dim=1)
    query_list = query_ids[0].tolist()

    c_results = {}
    for selector in ("recency", "bm25", "reader_attn", "iter_reader_attn", "iter_bm25"):
        st_mono = {"capture_step_logits": True}
        mono = qc.generate_from_ids(
            full_ids, chunk_size=chunk_size, max_new_tokens=12,
            selector=selector, topk=3, sink_tokens="bos",
            bare_question_ids=query_list, use_kv_cache=True, stats=st_mono,
            tokenizer=tok,
        )
        qc.encode_ids(ctx_ids, chunk_size=chunk_size, sink_tokens="bos")
        st_erg = {"capture_step_logits": True}
        erg_ids = qc.generate_ids(
            query_list, selector=selector, topk=3, mode="comem",
            max_new_tokens=12, use_kv_cache=True, stats=st_erg,
        )
        tok_ok = (st_mono["generated_ids"] == erg_ids)
        n = min(len(st_mono["step_logits"]), len(st_erg["step_logits"]))
        md = max((st_mono["step_logits"][s] - st_erg["step_logits"][s]).abs().max().item()
                 for s in range(n)) if n else 0.0
        c_results[selector] = (tok_ok, md)

    # (D) kv vs recompute on the monolithic path
    st_kv = {"capture_step_logits": True}
    _ = qc.generate_from_ids(full_ids, chunk_size=chunk_size, max_new_tokens=12,
                             selector="recency", topk=3, sink_tokens="bos",
                             use_kv_cache=True, stats=st_kv, tokenizer=tok)
    st_rc = {"capture_step_logits": True}
    _ = qc.generate_from_ids(full_ids, chunk_size=chunk_size, max_new_tokens=12,
                             selector="recency", topk=3, sink_tokens="bos",
                             use_kv_cache=False, stats=st_rc, tokenizer=tok)
    kv_tok_ok = (st_kv["generated_ids"] == st_rc["generated_ids"])
    nkv = min(len(st_kv["step_logits"]), len(st_rc["step_logits"]))
    kv_md = max((st_kv["step_logits"][s] - st_rc["step_logits"][s]).abs().max().item()
                for s in range(nkv)) if nkv else 0.0

    ok = (diff_pack < tol and diff_resume < tol
          and all(d < tol for d in diffs_j.values())
          and all(t and (m < tol) for t, m in c_results.values())
          and kv_tok_ok and (kv_md < tol))

    if verbose:
        print("=" * 72)
        print(f"CoMem self-test (tiny Qwen3, fp32, L={L}, tol={tol:.0e})")
        print("=" * 72)
        print(f"  (A) j=0 write/read packing == full forward : {diff_pack:.3e}  "
              f"{'PASS' if diff_pack < tol else 'FAIL'}")
        print(f"  (B) resume_forward_ids (j=0)               : {diff_resume:.3e}  "
              f"{'PASS' if diff_resume < tol else 'FAIL'}")
        for j, d in diffs_j.items():
            print(f"      resume_forward_ids (j={j:>2})            : {d:.3e}  "
                  f"{'PASS' if d < tol else 'FAIL'}")
        for sel, (t, m) in c_results.items():
            print(f"  (C) encode+generate==monolithic [{sel:>16}]: "
                  f"tokens={'OK' if t else 'MISMATCH'} maxdiff={m:.3e}  "
                  f"{'PASS' if (t and m < tol) else 'FAIL'}")
        print(f"  (D) kv-cache decode == recompute decode    : "
              f"tokens={'OK' if kv_tok_ok else 'MISMATCH'} maxdiff={kv_md:.3e}  "
              f"{'PASS' if (kv_tok_ok and kv_md < tol) else 'FAIL'}")
        print("-" * 72)
        print(f"SELF-TEST: {'ALL PASS' if ok else 'FAILURE'}")
        print("=" * 72)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
