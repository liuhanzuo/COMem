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

  (E) the external BASELINE gates, so one command covers the whole repo:
      * ``comem.cacheblend`` — RoPE reindex exact, ``r=1.0`` == full prefill,
        ``r=0.0`` finite (see :func:`comem.cacheblend.run_self_test`);
      * ``comem.kvcompress`` — SnapKV / PyramidKV do not perturb the model when
        no compression fires, and honour their retained-KV budget
        (see :func:`comem.kvcompress.run_self_test`).

  (F) the ``dense_bge`` selector's plumbing without touching the network: a stub
      retriever exercises the dispatch, the deterministic ``(-score, idx)``
      tie-break, and the fail-closed guard when no retriever is supplied.
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


class _StubBGERetriever:
    """Offline stand-in for :class:`comem.selectors.DenseBGERetriever`.

    Exposes the same ``scores(context_texts, query_text)`` contract with a
    deterministic, weight-free score (normalised character-bigram overlap), so the
    ``dense_bge`` dispatch / detokenisation / tie-break can be gated on CPU with
    no network and no BGE checkpoint. The real retriever's numerics are validated
    separately against the frozen checkpoint."""

    @staticmethod
    def _bigrams(text):
        return {text[i:i + 2] for i in range(max(0, len(text) - 1))}

    def scores(self, context_texts, query_text):
        q = self._bigrams(query_text or "")
        out = []
        for t in context_texts:
            c = self._bigrams(t)
            out.append(len(q & c) / (len(q | c) or 1))
        return out


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

    # ---- (F) dense_bge selector plumbing (stub retriever, no network) ----
    from . import selectors as _sel
    stub = _StubBGERetriever()
    ctx_chunks = [torch.tensor(list(range(10, 10 + chunk_size))),   # unrelated
                  torch.tensor(query_list),                        # == the query
                  torch.tensor(list(range(90, 90 + chunk_size)))]  # unrelated
    dense_sel = _sel.select_context_chunk_indices(
        "dense_bge", ctx_chunks, query_list, 1,
        dense_retriever=stub, dense_tokenizer=tok)
    dense_hits_query_chunk = (dense_sel == [1])          # the query chunk wins
    # topk == n_ctx must return every chunk exactly once, in doc order
    dense_all = _sel.select_context_chunk_indices(
        "dense_bge", ctx_chunks, query_list, len(ctx_chunks),
        dense_retriever=stub, dense_tokenizer=tok)
    dense_full_ok = (dense_all == [0, 1, 2])
    try:                                                  # fail-closed guard
        _sel.select_context_chunk_indices("dense_bge", ctx_chunks, query_list, 1)
        dense_guard_ok = False
    except ValueError:
        dense_guard_ok = True
    dense_ok = dense_hits_query_chunk and dense_full_ok and dense_guard_ok
    # end-to-end through the model entry point (exercises model.py's wiring)
    _ = qc.generate_from_ids(
        full_ids, chunk_size=chunk_size, max_new_tokens=4, selector="dense_bge",
        topk=3, sink_tokens="bos", bare_question_ids=query_list,
        dense_retriever=stub, tokenizer=tok)

    # ---- (E) external baseline gates ----
    from . import cacheblend as _cb
    from . import kvcompress as _kvc
    cb_ok = _cb.run_self_test(tol=tol, verbose=False)
    kvc_ok = _kvc.run_self_test(tol=tol, verbose=False)

    ok = (diff_pack < tol and diff_resume < tol
          and all(d < tol for d in diffs_j.values())
          and all(t and (m < tol) for t, m in c_results.values())
          and kv_tok_ok and (kv_md < tol)
          and dense_ok and cb_ok and kvc_ok)

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
        print(f"  (E) baseline cacheblend (reindex/r=1/r=0)  : "
              f"{'PASS' if cb_ok else 'FAIL'}   "
              f"[python -m comem.cacheblend for detail]")
        print(f"  (E) baseline snapkv+pyramidkv (stock/budget): "
              f"{'PASS' if kvc_ok else 'FAIL'}   "
              f"[python -m comem.kvcompress for detail]")
        print(f"  (F) selector dense_bge dispatch/tie-break  : "
              f"top1={'OK' if dense_hits_query_chunk else 'WRONG'} "
              f"full={'OK' if dense_full_ok else 'WRONG'} "
              f"guard={'OK' if dense_guard_ok else 'MISSING'}  "
              f"{'PASS' if dense_ok else 'FAIL'}")
        print("-" * 72)
        print(f"SELF-TEST: {'ALL PASS' if ok else 'FAILURE'}")
        print("=" * 72)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
