"""Context-chunk selectors for CoMem (Comprehension Memory).

Given a set of context chunks (each cached as a depth-``j`` hidden ``h_j`` plus
its raw token ids) and a query, a *selector* returns the ordered indices of the
``topk`` chunks to pack into the read. Every selector here EXCEPT ``dense_bge``
is FORWARD-FREE beyond the bottom-``j`` writes CoMem already performs: they
consume only the cached ``h_j`` tensors and/or the raw token ids, so retrieval
adds no extra model forward and CoMem's compute saving is preserved.
``dense_bge`` is the deliberate exception (a small FROZEN external sentence
encoder), so its retrieval latency / index size are reported separately.

Selectors
---------
* ``recency``          — the last ``topk`` context chunks (positional).
* ``bm25``             — highest lexical BM25 overlap with the bare question
                         (pure CPU; IDF over the candidate pool).
* ``dense_bge``        — highest cosine between a FROZEN BGE sentence embedding
                         of the query and of each chunk's decoded text (CLS
                         pooling + L2 norm + cosine == dot). The single-variable
                         "dense retrieval instead of lexical BM25" arm; needs a
                         :class:`DenseBGERetriever` (``dense_retriever=``) and a
                         backbone tokenizer (``dense_tokenizer=``) to detokenise
                         the chunks back to text.
* ``iter_bm25``        — multi-hop BFS BM25: round 1 == single-shot ``bm25``,
                         later rounds re-query with the previous picks' token
                         text to walk a lexical reference chain (RULER vt).
* ``iter_bm25_adaptive`` — like ``iter_bm25`` but confidence-adaptive: no fixed
                         ``topk`` budget; stop when a round's best score drops
                         below ``conf_ratio``× the round-1 best (chain end /
                         spurious link) or ``max_chunks`` is hit, so short chains
                         don't hard-fill low-score noise chunks into the read.
* ``reader_attn``      — highest query-``h_j`` vs chunk-``h_j`` mean-pool cosine
                         (semantic; reuses the cached ``write_chunk`` hiddens).
* ``iter_reader_attn`` — multi-hop BFS over the cached ``h_j`` (round 1 from the
                         query, later rounds from the just-found chunks) to
                         follow reference chains; ``meanpool`` or ``maxsim`` score.
* ``oracle``           — the chunks that CONTAIN the gold answer (upper bound;
                         needs a located needle set, else falls back to recency).

Also exposes the oracle needle locator (``locate_needle_chunks`` /
``find_subsequence_ids``) so the eval drivers can build the oracle chunk set
without importing anything outside ``comem``.

All primitives are lifted verbatim (formulae + defaults) from the QCMem research
code: the BM25 scorer (``k1=1.5``, ``b=0.75``) and the needle locator match the
original ``run_babilong_mem_space`` helpers, the reader-attn / iterative
selectors match ``eval_qcmem_babilong``, and ``dense_bge`` ports the frozen
BGE-large-en-v1.5 retriever (CLS + L2 + cosine, official query instruction,
stable ``(-score, idx)`` tie-break) from ``eval_p1_9_dense_rag.DenseRetriever``
— so CoMem reproduces the published retrieval rankings bit-for-bit.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# lexical BM25 (pure CPU, no model forward)
# --------------------------------------------------------------------------- #
def bm25_scores(docs, query_ids, k1: float = 1.5, b: float = 0.75):
    """BM25 of ``query_ids`` (list[int]) against each candidate document's token
    IDs. Corpus == the candidate pool ``docs`` (list[list[int]]), so IDF is over
    that pool. Query terms are de-duplicated. Returns a python list[float] of
    length ``len(docs)`` (doc order preserved), or ``None`` if the pool is empty.
    """
    N = len(docs)
    if N <= 0:
        return None
    df = Counter()
    doc_tf = []
    doc_len = []
    for d in docs:
        c = Counter(d)
        doc_tf.append(c)
        doc_len.append(len(d))
        for t in c:
            df[t] += 1
    avgdl = (sum(doc_len) / N) if N > 0 else 0.0
    idf = {t: math.log((N - dft + 0.5) / (dft + 0.5) + 1.0) for t, dft in df.items()}
    qterms = set(int(t) for t in query_ids)
    scores = []
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            it = idf.get(t, 0.0)
            if avgdl > 0:
                denom = f + k1 * (1.0 - b + b * dl / avgdl)
            else:
                denom = f + k1
            s += it * (f * (k1 + 1.0)) / denom
        scores.append(s)
    return scores


# --------------------------------------------------------------------------- #
# dense_bge: frozen BGE sentence-embedding cosine over the chunks' decoded text
# --------------------------------------------------------------------------- #
# Official BGE query instruction (BAAI/bge-*-en-v1.5 model card): passages are
# encoded RAW, only the query gets the instruction prefix.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class DenseBGERetriever:
    """Frozen BGE sentence encoder (``BAAI/bge-large-en-v1.5`` and friends).

    CLS pooling, L2 normalisation, cosine similarity (== dot after the norm).
    Passages are encoded WITHOUT the instruction, the query WITH the official
    ``BGE_QUERY_INSTRUCTION``; both are truncated to the encoder's 512-token
    position budget (standard dense-RAG behaviour). Held frozen and used
    read-only — this is a RETRIEVER, not part of the CoMem backbone.

    The encoder is loaded lazily on first use so constructing the object (e.g.
    from an eval driver's argparse) never touches the disk.

    Parameters
    ----------
    retriever_path:
        Local HF directory of the BGE checkpoint (loaded ``local_files_only``).
    device / dtype:
        Where/how to run the encoder. Defaults to the CoMem eval convention
        (``cuda:0`` + ``bfloat16``); pass ``float32`` for a deterministic CPU run.
    batch_size:
        Chunk-encoding batch size (chunks are ~512 backbone tokens each).
    """

    def __init__(self, retriever_path, device="cuda:0", dtype=torch.bfloat16,
                 batch_size: int = 64):
        self.path = str(retriever_path)
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.model = None
        self.tokenizer = None
        self.max_len = 512
        self.hidden = None

    @property
    def dtype_bytes(self) -> int:
        return 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

    def load(self):
        """Load the frozen encoder + its tokenizer (idempotent). Fails closed if
        the checkpoint declares a non-CLS pooling contract, since the cosine
        scoring below hard-codes BGE's CLS+L2 recipe."""
        if self.model is not None:
            return self
        import json
        import os
        from transformers import AutoModel, AutoTokenizer

        pool_cfg = os.path.join(self.path, "1_Pooling", "config.json")
        if os.path.exists(pool_cfg):
            with open(pool_cfg) as f:
                pc = json.load(f)
            if not pc.get("pooling_mode_cls_token", False):
                raise ValueError(
                    f"retriever pooling config {pc} is not CLS; the dense_bge "
                    f"selector hard-codes the BGE CLS+L2+cosine contract")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,
                                                       local_files_only=True)
        max_len = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        self.max_len = 512 if (max_len > 512 or max_len <= 0) else max_len
        self.model = AutoModel.from_pretrained(
            self.path, torch_dtype=self.dtype, local_files_only=True
        ).to(self.device).eval()
        self.hidden = int(self.model.config.hidden_size)
        return self

    @torch.no_grad()
    def encode(self, texts, is_query: bool = False) -> torch.Tensor:
        """L2-normalised CLS embeddings ``[N, hidden]`` (float32, on CPU)."""
        self.load()
        if is_query:
            texts = [f"{BGE_QUERY_INSTRUCTION} {t}" for t in texts]
        embs = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(
                texts[s:s + self.batch_size], padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]         # CLS token
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            embs.append(cls.float().cpu())
        if not embs:
            return torch.zeros((0, int(self.hidden or 0)), dtype=torch.float32)
        return torch.cat(embs, dim=0)

    def scores(self, context_texts, query_text):
        """Cosine similarity of ``query_text`` to each chunk text (doc order)."""
        if not context_texts:
            return []
        ctx = self.encode(list(context_texts), is_query=False)      # [n, d]
        q = self.encode([query_text or ""], is_query=True)          # [1, d]
        return (ctx @ q[0]).tolist()

    def index_bytes(self, n_chunks: int) -> int:
        """Size of the chunk-embedding index this retriever would persist."""
        return int(n_chunks) * int(self.hidden or 0) * self.dtype_bytes


def dense_bge_scores(context_chunks, query_ids, retriever, tokenizer):
    """Frozen-BGE cosine of the query against each context chunk (doc order).

    ``context_chunks`` are backbone token-id tensors and ``query_ids`` the bare
    question's backbone token ids, so both are detokenised with the BACKBONE
    ``tokenizer`` (``skip_special_tokens=True``) before being handed to the
    retriever — the chunk text a text-space dense retriever would see. Returns
    ``list[float]`` aligned with ``context_chunks`` (higher == more relevant)."""
    ctx_texts = [tokenizer.decode(c.tolist() if torch.is_tensor(c) else list(c),
                                  skip_special_tokens=True)
                 for c in context_chunks]
    query_text = tokenizer.decode(list(query_ids), skip_special_tokens=True)
    return retriever.scores(ctx_texts, query_text)


# --------------------------------------------------------------------------- #
# reader-attn: cosine over cached depth-j hiddens (no extra forward)
# --------------------------------------------------------------------------- #
def reader_attn_scores(context_hj, query_hj):
    """Salience of each context chunk to the query, from cached depth-``j`` hidden
    states (``h_j`` == the ``write_chunk`` output; NO extra model forward).

    Score = cosine similarity between the mean-pooled (over the token axis) query
    ``h_j`` vector and each context chunk's mean-pooled ``h_j`` vector. Returns
    ``list[float]`` aligned with ``context_hj`` (higher == more salient).
    """
    q_vec = query_hj.float().mean(dim=1).squeeze(0)          # [d]
    q_vec = q_vec / (q_vec.norm() + 1e-8)
    scores = []
    for h in context_hj:
        if h is None or h.shape[1] == 0:
            scores.append(float("-inf"))
            continue
        c_vec = h.float().mean(dim=1).squeeze(0)             # [d]
        c_vec = c_vec / (c_vec.norm() + 1e-8)
        scores.append(float(torch.dot(q_vec, c_vec).item()))  # cosine similarity
    return scores


def _meanpool_reps(context_hj):
    """L2-normalised mean-pooled ``[d]`` vector per context chunk (None-safe)."""
    reps = []
    for h in context_hj:
        if h is None or h.shape[1] == 0:
            reps.append(None)
            continue
        v = h.float().mean(dim=1).squeeze(0)
        reps.append(v / (v.norm() + 1e-8))
    return reps


def _query_meanpool_rep(query_hj):
    v = query_hj.float().mean(dim=1).squeeze(0)
    return v / (v.norm() + 1e-8)


def _token_reps(context_hj):
    """Row-L2-normalised token matrix ``[T, d]`` per context chunk (None-safe)."""
    reps = []
    for h in context_hj:
        if h is None or h.shape[1] == 0:
            reps.append(None)
            continue
        m = h.float().squeeze(0)                              # [T, d]
        reps.append(m / (m.norm(dim=-1, keepdim=True) + 1e-8))
    return reps


# --------------------------------------------------------------------------- #
# iterative multi-hop SEMANTIC selector (iter_reader_attn)
# --------------------------------------------------------------------------- #
def iter_reader_attn_indices(
    context_hj,
    query_hj,
    topk: int,
    iter_rounds: int = 0,
    iter_hop_topk: int = 2,
    iter_score: str = "meanpool",
):
    """Iterative multi-hop chunk selection over cached ``h_j``.

    Propagates a FRONTIER of just-found chunks: round 1 scores every context
    chunk against the query ``h_j`` (== reader_attn), then each later round scores
    the still-unselected chunks against the JUST-ADDED chunks' ``h_j`` (max over
    the frontier), so the signal walks one hop along a reference chain per round.
    ``iter_rounds<=0`` -> auto ``ceil(topk/hop)``. Returns a sorted list of
    context-chunk indices, up to ``topk`` chunks accumulated over the rounds.
    """
    n = len(context_hj)
    k = max(0, int(topk))
    if n == 0 or k == 0:
        return []
    hop = max(1, int(iter_hop_topk))
    rounds = int(iter_rounds) if iter_rounds and iter_rounds > 0 else -(-k // hop)
    rounds = max(1, rounds)

    if iter_score == "maxsim":
        chunk_reps = _token_reps(context_hj)                  # list[[T,d]|None]
        qm = query_hj.float().squeeze(0)
        q_rep = qm / (qm.norm(dim=-1, keepdim=True) + 1e-8)   # [Tq, d]

        def score(frontier_reps, i):
            ci = chunk_reps[i]
            if ci is None:
                return float("-inf")
            best = float("-inf")
            for fr in frontier_reps:
                if fr is None:
                    continue
                s = float((ci @ fr.T).max().item())           # best token pair
                if s > best:
                    best = s
            return best
    else:  # meanpool (default)
        chunk_reps = _meanpool_reps(context_hj)               # list[[d]|None]
        q_rep = _query_meanpool_rep(query_hj)                 # [d]

        def score(frontier_reps, i):
            ci = chunk_reps[i]
            if ci is None:
                return float("-inf")
            best = float("-inf")
            for fr in frontier_reps:
                if fr is None:
                    continue
                s = float(torch.dot(ci, fr).item())
                if s > best:
                    best = s
            return best

    selected: list = []
    selected_set: set = set()
    frontier = [q_rep]                                        # round 1 == reader_attn
    for _ in range(rounds):
        remaining = k - len(selected)
        if remaining <= 0:
            break
        cand = [i for i in range(n) if i not in selected_set]
        if not cand:
            break
        scored = [(i, score(frontier, i)) for i in cand]
        scored = [(i, s) for (i, s) in scored if s != float("-inf")]
        if not scored:
            break
        scored.sort(key=lambda t: t[1], reverse=True)
        take = min(hop, remaining, len(scored))
        new_sel = [i for (i, _s) in scored[:take]]
        selected.extend(new_sel)
        selected_set.update(new_sel)
        frontier = [chunk_reps[i] for i in new_sel]
    return sorted(selected)


# --------------------------------------------------------------------------- #
# iterative multi-hop LEXICAL selector (iter_bm25)
# --------------------------------------------------------------------------- #
def iter_bm25_indices(
    context_chunks,        # list[LongTensor] == context chunks (doc order)
    query_ids,             # list[int] bare-question token ids (round-1 query)
    topk: int,
    iter_rounds: int = 0,
    iter_hop_topk: int = 2,
):
    """Iterative multi-hop lexical (BM25) chunk selection.

    Round 1 == single-shot ``bm25``; later rounds use the concatenated token text
    of the previous round's just-added chunks as the new BM25 query, walking a
    lexical reference chain (RULER vt: consecutive ``VAR Vc = VAR V(c-1)`` links).
    IDF is over the full context pool every round, so round 1 is bit-for-bit the
    same ranking as ``bm25``. Chunks with zero BM25 overlap are skipped
    (chain-end guard). ``iter_rounds<=0`` -> auto ``ceil(topk/hop)``.
    """
    n = len(context_chunks)
    k = max(0, int(topk))
    if n == 0 or k == 0:
        return []
    hop = max(1, int(iter_hop_topk))
    rounds = int(iter_rounds) if iter_rounds and iter_rounds > 0 else -(-k // hop)
    rounds = max(1, rounds)

    docs = [c.tolist() for c in context_chunks]     # BM25 corpus (fixed IDF pool)

    selected: list = []
    selected_set: set = set()
    frontier_query = list(query_ids)                # round 1 == single-shot bm25
    for _ in range(rounds):
        remaining = k - len(selected)
        if remaining <= 0:
            break
        if not frontier_query:
            break
        scores = bm25_scores(docs, frontier_query)
        if not scores:
            break
        cand = [i for i in range(n) if i not in selected_set]
        cand = [i for i in cand if scores[i] > 0.0]
        if not cand:
            break
        cand.sort(key=lambda i: scores[i], reverse=True)
        take = min(hop, remaining, len(cand))
        new_sel = cand[:take]
        selected.extend(new_sel)
        selected_set.update(new_sel)
        frontier_query = []
        for i in new_sel:
            frontier_query.extend(docs[i])
    return sorted(selected)


# --------------------------------------------------------------------------- #
# iterative multi-hop LEXICAL selector, adaptive stop (iter_bm25_adaptive)
# --------------------------------------------------------------------------- #
def iter_bm25_adaptive_indices(
    context_chunks,        # list[LongTensor] == context chunks (doc order)
    query_ids,             # list[int] bare-question token ids (round-1 query)
    iter_hop_topk: int = 4,
    conf_ratio: float = 0.3,
    max_chunks: int = 64,
):
    """Iterative multi-hop lexical (BM25) chunk selection with a CONFIDENCE-based
    adaptive stop (drop-in variant of :func:`iter_bm25_indices`).

    Same BFS walk as ``iter_bm25`` (round 1 == single-shot ``bm25``; later rounds
    re-query BM25 with the previous round's picks' concatenated token text; IDF is
    the full context pool every round, so round 1 is bit-for-bit the ``bm25``
    ranking), but replaces the fixed ``topk`` budget with an adaptive stop:

    * record ``s1`` = round 1's best BM25 score (the confidence reference);
    * each LATER round, if that round's best candidate score ``< conf_ratio * s1``
      the chain has ended / the next hop is a spurious lexical link -> **break**
      (do not pull the low-score chunk in, unlike the fixed-budget ``iter_bm25``
      which hard-fills to ``topk`` and drags noise into the read);
    * ``max_chunks`` is a safety cap on the accumulated chunk count.

    Every round still takes the top ``iter_hop_topk`` new chunks with ``score>0``
    (chain-end guard). The loop stops on the FIRST of: (1) confidence below
    ``conf_ratio * s1``, (2) ``max_chunks`` reached, (3) no ``score>0`` candidate.
    Returns a sorted list of context-chunk indices.
    """
    n = len(context_chunks)
    if n == 0:
        return []
    hop = max(1, int(iter_hop_topk))
    cap = max(1, int(max_chunks))
    ratio = float(conf_ratio)

    docs = [c.tolist() for c in context_chunks]     # BM25 corpus (fixed IDF pool)

    selected: list = []
    selected_set: set = set()
    frontier_query = list(query_ids)                # round 1 == single-shot bm25
    s1 = None                                       # round-1 best score (confidence ref)
    while True:
        if len(selected) >= cap:
            break
        if not frontier_query:
            break
        scores = bm25_scores(docs, frontier_query)
        if not scores:
            break
        cand = [i for i in range(n) if i not in selected_set]
        cand = [i for i in cand if scores[i] > 0.0]
        if not cand:
            break
        cand.sort(key=lambda i: scores[i], reverse=True)
        best = scores[cand[0]]
        if s1 is None:
            s1 = best                               # round 1 sets the reference
        elif best < ratio * s1:
            break                                   # relative-confidence early stop
        remaining = cap - len(selected)
        take = min(hop, remaining, len(cand))
        new_sel = cand[:take]
        selected.extend(new_sel)
        selected_set.update(new_sel)
        frontier_query = []
        for i in new_sel:
            frontier_query.extend(docs[i])
    return sorted(selected)


# --------------------------------------------------------------------------- #
# top-level dispatch
# --------------------------------------------------------------------------- #
def select_context_chunk_indices(
    selector: str,
    context_chunks,        # list[LongTensor] (doc order)
    query_ids,             # list[int] bare-question token ids
    topk: int,
    needle_chunk_set=None,  # set[int] doc-absolute chunk indices (oracle) or None
    context_hj=None,        # list[Tensor [1,T,d]] cached h_j  (reader_attn* only)
    query_hj=None,          # Tensor [1,T,d] query chunk h_j   (reader_attn* only)
    iter_rounds=0,
    iter_hop_topk=2,
    iter_score="meanpool",
    iter_conf_ratio=0.3,
    iter_max_chunks=64,
    dense_retriever=None,   # DenseBGERetriever                (dense_bge only)
    dense_tokenizer=None,   # backbone tokenizer (detokenise)  (dense_bge only)
):
    """Return a sorted list of context-chunk indices to pack into the read,
    chosen by the requested selector (see module docstring). Selectors that need
    the cached ``h_j`` (``reader_attn`` / ``iter_reader_attn``) fall back to
    ``recency`` if the caller did not supply ``context_hj`` / ``query_hj``;
    ``dense_bge`` needs ``dense_retriever`` + ``dense_tokenizer`` and raises if
    either is missing (silently degrading a retriever arm would corrupt the
    single-variable comparison).
    """
    n_ctx = len(context_chunks)
    if n_ctx == 0:
        return []
    k = max(0, int(topk))

    if selector == "recency":
        if k <= 0:
            return []
        return list(range(max(0, n_ctx - k), n_ctx))

    if selector == "oracle":
        if needle_chunk_set:
            sel = sorted(c for c in needle_chunk_set if 0 <= c < n_ctx)
            if sel:
                return sel
        if k <= 0:
            return []
        return list(range(max(0, n_ctx - k), n_ctx))

    if selector == "bm25":
        if k <= 0:
            return []
        docs = [c.tolist() for c in context_chunks]
        scores = bm25_scores(docs, list(query_ids))
        if not scores:
            return list(range(max(0, n_ctx - k), n_ctx))
        order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
        return sorted(order[:k])

    if selector == "dense_bge":
        # Frozen BGE dense retrieval instead of the lexical BM25 ranking; ties
        # break by ascending index so the ranking is deterministic given the
        # embeddings. Fail loudly rather than silently degrade to recency: this
        # selector IS the experimental variable of the dense-retriever arm.
        if k <= 0:
            return []
        if dense_retriever is None or dense_tokenizer is None:
            raise ValueError(
                "selector 'dense_bge' needs dense_retriever= (a "
                "comem.selectors.DenseBGERetriever) and dense_tokenizer= (the "
                "backbone tokenizer used to detokenise the chunks)")
        scores = dense_bge_scores(context_chunks, list(query_ids),
                                  dense_retriever, dense_tokenizer)
        if not scores:
            return list(range(max(0, n_ctx - k), n_ctx))
        order = sorted(range(n_ctx), key=lambda i: (-scores[i], i))
        return sorted(order[:k])

    if selector == "iter_bm25":
        if k <= 0:
            return []
        return iter_bm25_indices(
            context_chunks, list(query_ids), topk=k,
            iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk,
        )

    if selector == "iter_bm25_adaptive":
        # Adaptive-stop variant of iter_bm25: no fixed topk budget — walk the
        # lexical chain until confidence drops below iter_conf_ratio * (round-1
        # best score), max_chunks is hit, or no positive-overlap candidate remains.
        return iter_bm25_adaptive_indices(
            context_chunks, list(query_ids),
            iter_hop_topk=iter_hop_topk,
            conf_ratio=iter_conf_ratio, max_chunks=iter_max_chunks,
        )

    if selector == "reader_attn":
        if k <= 0:
            return []
        if not context_hj or query_hj is None or len(context_hj) != n_ctx:
            return list(range(max(0, n_ctx - k), n_ctx))
        scores = reader_attn_scores(context_hj, query_hj)
        order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
        return sorted(order[:k])

    if selector == "iter_reader_attn":
        if k <= 0:
            return []
        if not context_hj or query_hj is None or len(context_hj) != n_ctx:
            return list(range(max(0, n_ctx - k), n_ctx))
        return iter_reader_attn_indices(
            context_hj, query_hj, topk=k,
            iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk,
            iter_score=iter_score,
        )

    raise ValueError(f"unknown selector {selector!r}")


# --------------------------------------------------------------------------- #
# oracle needle locator (used by the eval drivers to build needle_chunk_set)
# --------------------------------------------------------------------------- #
def find_subsequence_ids(haystack_ids, needle_ids):
    """Locate ``needle_ids`` as a contiguous subsequence of ``haystack_ids``
    (both 1-D list[int]). Returns the START token index of the LAST (most recent)
    match or None. Robust to whitespace-merge tokenisation at the answer boundary:
    try the full needle, then progressively trim leading tokens, then fall back to
    the longest trailing run that matches."""
    H, N = haystack_ids, needle_ids
    nH, nN = len(H), len(N)
    if nN == 0 or nH == 0:
        return None

    def _scan(sub):
        ns = len(sub)
        if ns == 0 or ns > nH:
            return None
        for s in range(nH - ns, -1, -1):
            if H[s:s + ns] == sub:
                return s
        return None

    r = _scan(N)
    if r is not None:
        return r
    for drop in range(1, min(4, nN)):
        r = _scan(N[drop:])
        if r is not None:
            return r
    for keep in range(nN - 1, 0, -1):
        r = _scan(N[-keep:])
        if r is not None:
            return r
    return None


def locate_needle_chunks(input_ids, target, tokenizer, chunk_size):
    """Return the set of 0-based DOCUMENT-ABSOLUTE chunk indices that contain the
    gold answer (``target``) in ``input_ids`` (a [1, L] LongTensor), or None.

    chunk index = ``token_pos // chunk_size``. The answer string can tokenise
    differently in isolation vs in-context (space-prefix merges), so we try a few
    encodings and take the union of any matches. Multi-token answers straddling a
    chunk boundary contribute BOTH chunks."""
    ids = input_ids[0].tolist()
    L = len(ids)
    tgt = (target or "").strip()
    if not tgt:
        return None
    cands = []
    for variant in (tgt, " " + tgt):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            cands.append(enc)
    chunks = set()
    for needle_ids in cands:
        start = find_subsequence_ids(ids, needle_ids)
        if start is None:
            continue
        end = min(L - 1, start + len(needle_ids) - 1)
        for p in range(start, end + 1):
            chunks.add(p // chunk_size)
    return chunks or None
