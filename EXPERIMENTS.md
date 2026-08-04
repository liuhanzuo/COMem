# EXPERIMENTS — provenance index

Which experiment produces which paper table, how to reproduce it from this repo,
and where the research-side artifact lives. This is an **index**, not a write-up:
every claim/caveat belongs to the paper text, and every number below is a
one-line pointer so a reader can find the table and the CLI that regenerates it.

**Scope / conventions.** Unless a row says otherwise: Qwen3-8B, bf16 + SDPA,
greedy, `chat_template=False` (the backbone has no SFT/RL, so a chat template
would be unfair), `chunk_size=512`, `topk=12`, `sink=bos`, `seed=42`,
`n=100`/cell. Numbers live in the paper's `tab_*.tex` sources; **result JSON/CSV
never enters this repo** (`.gitignore` blocks `*_results/`), so the "artifact"
column names the research-repo path that holds the raw predictions.

**Where each table lives.** This repo's `paper/sections/` carries the 17-table
core subset: `tab_overview`, `tab_h2h`, `tab_babilong`, `tab_longbench`,
`tab_longeval`, `tab_locomo`, `tab_scaling`, `tab_scale`, `tab_slm`, `tab_depth`,
`tab_chunk`, `tab_crosschunk`, `tab_selector`, `tab_itervt`, `tab_eff`,
`tab_hy3_distill`, `tab_hy3_ruler`. Every **other** `tab_*` referenced below —
including all of §1's audit tables — is part of the ARR submission's extended
table set and is **not** checked into this repo; its name is given so a reader of
the submitted paper can line the two up, and the "artifact" column is then the
authoritative pointer.

`<M>` below = a local backbone path, `<OUT>` = an output dir.
"Research repo" = the internal experiment repo (`Mixture-of-Memory`); its
per-experiment task IDs (`#NNN`) and the audit plan are in `paperA/TODOList.md`.

---

## 1. External-baseline comparisons (the ARR audit additions)

| Experiment | Paper table | One-line result | Reproduce here | Research artifact |
|---|---|---|---|---|
| **CacheBlend-style full-depth chunk KV** — same selector/pack/sink as CoMem, only the cache object changes (full `L`-layer KV vs one depth-`j` residual); recompute sweep `r ∈ {0, .10, .15, .18}` | `tab_cacheblend_baseline` (`tab:cacheblend-baseline`) | RULER macro 67.8 (`r=0`) → 74.7 (`r=.18`), BABI-qa5 ≈49, LoCoMo 17.1–17.4, at **144 KiB/token**; CoMem 97.05 / 68.67 / 23.36 at **8 KiB/token** | `--baseline cacheblend --recompute_ratio {0,0.10,0.15,0.18} --selector iter_bm25 --iter_hop_topk 4` (module `comem/cacheblend.py`; gate `python -m comem.cacheblend`) | `#143`; `bench_results/cacheblend/aggregate.json`; `paperA/artifacts/cacheblend_143/`; notes `paperA/CACHEBLEND_{IMPL_NOTES,BASELINE_DESIGN}.md` |
| **Equal-retained-budget compressed KV** — SnapKV / PyramidKV at CoMem's 6,657-token read budget, plus their full-prefill systems cost | `tab_kvcompress` (`tab:kvcompress`) | Native-A macro SnapKV 72.33 / PyramidKV 72.32 vs CoMem+LoRA 97.05; both must full-prefill (128k: ≈51.5 s, peak 54.3 GB) while CoMem's peak is context-flat | `--baseline snapkv|pyramidkv --kv_budget 6657 --kv_window 32` (module `comem/kvcompress.py`; gate `python -m comem.kvcompress`) | `#120` / P1.6; `ruler_results/p16_{snapkv,pyramidkv}_{native,yarn}/`, `locomo_results/p16_*/scores.json`; timing `outputs/p16_timing/p16_timing_full.json`; provenance `src/baselines/PROVENANCE.md` |
| **Frozen dense retriever diagnostic** — BGE-large-en-v1.5 (CLS+L2+cosine) feeding a `j=0`, LoRA-off reader; recall@12 vs hit-conditional reading | `tab_dense_retriever` (`tab:dense-retriever`), and the left panel of `tab_selector_dense_combined` | recall@12 decays with length (qa1 16k .52, qa2 16k .45, LongEval 16k .58) while hit-conditional reading stays strong → the bottleneck is retrieval, not reading | `--selector dense_bge --retriever_path <bge dir> --baseline kvdirect` for the `j=0` reader (selector in `comem/selectors.py`) | `#140` / P1.9; `bench_results/p1_9_dense_rag/aggregate.json`; notes `paperA/P1_9_DENSE_RAG_NOTES.md` |
| **Dense-selector swap into CoMem** — the same frozen BGE ranking fills CoMem's `h_j` instead of iterative BM25 (selector is the ONLY variable) | feeds the "Frozen BGE" row of `tab_equal_latency` (`tab:equal-latency`) | Under equal latency the frozen-BGE replay arm is within 1.0 pt of CoMem and both dependence-aware ranges span zero (TIE); the lexical BM25 replay arm is −11.56 | `--selector dense_bge --retriever_path <bge dir> --j auto --adapter <flagship LoRA>` | `#144`; `bench_results/dense_selector/aggregate.json`; `#141` phase B `bench_results/p0_20_phaseB_dense/`; notes `paperA/P0_20_PHASEB_NOTES.md` |
| **Prior-art interface map** — which persistent object / tunable axis / post-hit work each reusable-context family exposes | `tab_priorart` (`tab:priorart`) | Positioning table; its one empirical anchor for the chunk-KV/PIC row is the CacheBlend arm above | no run (survey table) | `paper/qcmem.bib`; audit item `A-AUDIT-1` in `paperA/TODOList.md` |

---

## 2. Core CoMem quality / mechanism

| Experiment | Paper table | One-line result | Reproduce here | Research artifact |
|---|---|---|---|---|
| Headline benchmark sweep (RULER / BABILong / LongBench / LongEval / LoCoMo) | `tab_overview`, `tab_h2h`, `tab_babilong`, `tab_longbench`, `tab_longeval`, `tab_locomo`, `tab_scaling` | CoMem holds RULER ≈100 and LongEval ≈0.98 at 128k where full-context attention collapses past its RoPE window | `python -m eval.run --benchmark {ruler,babilong,longbench,longeval,locomo} --model <M> --j auto --out <OUT>` | per-benchmark `*_results/` dirs in the research repo |
| Matched `j=0` replay vs `j=12` (same pack/examples/LoRA) | `tab_replay_latency` (`tab:replay-latency`), `tab_core_tradeoff` (`tab:core-tradeoff`) | 15-cell RULER macro 99.19 (`j=0`) vs 96.07 (`j=12`), i.e. −3.12 pt (95% CI [2.36, 3.93]) for a 1.403× faster Read | `--baseline kvdirect` (builds `resume_j=0`) vs `--baseline none --j 12`, same `--selector/--topk` | `bench_p0_13_quality_latency.py` (P0.13) |
| Continuous-prefix `h_12` attribution oracle | `tab_h12_oracle` (`tab:h12-oracle`) | The oracle is bit-identical to full `j=0` replay (99.19), so skipping the lower layers costs nothing; the whole 3.12-pt gap is the chunk-local Write approximation | not shipped (per-query lower-layer recompute — an attribution bound, not a cache) | `#121` / P1.7; `scripts/bench_p1_7_h12_oracle.py`; `bench_results/p1_7_h12_oracle/` |
| Write-context scope (chunk-local vs overlap `w` vs document-context oracle) | `tab_write_context` (`tab:write-context`), `tab_context_position` (`tab:context-position`) | A 32-token Write overlap recovers 6.0 of the 7.5-pt local-context gap; changing Read position alone costs the chunk-local writer 4.5 pt (the two factors interact) | not shipped (Write-side ablation harness) | P0.16/P0.17/P0.18; `scripts/eval_p016_e0_write_control.py`, `eval_p017_e2_overlap_write.py`, `eval_p018_e4_2x2_writecontrol.py` |
| Selector ablation (BM25 / recency / reader-attn / oracle, and the iterative variants) | `tab_selector`, `tab_itervt`, `tab_selector_dense_combined` | BM25 is flat in length where recency/reader-attn decay (multikey 32k: 99 vs 54/60); RULER variable-tracking needs the multi-hop `iter_bm25` chain | `--selector {bm25,recency,reader_attn,oracle,iter_bm25,iter_bm25_adaptive,dense_bge}` | `_run_p0_19_ruler_paired.sh`; `ruler_results/*` |
| Adapter on/off at the same `j=12` | `tab_adapter`, `tab_adapter_hcache_combined` | Frozen `j=12` collapses (multikey 8/3/1) and the rank-32 LoRA restores it (91/95/92) → depth reuse needs interface adaptation | `--adapter <dir>` vs `--adapter none`; train with `train/distill.py` | flagship adapter `outputs/qcmem_distill_qwen_j12_r32_4k/final` |
| Split-depth and chunk-size sweeps | `tab_depth`, `tab_depth_tradeoff`, `tab_distilled_depth_curve`, `tab_chunk`, `tab_crosschunk` | Quality/latency trades monotonically with `j`; `j ≈ 0.33 L` is the shipped operating point (`comem/model_registry.py`) | `--j {0,6,12,18,...}`, `--chunk_size {256,512,1024}` | P2.4; `ruler_results/*` depth cohorts |
| Sparse-MoE backbone (Qwen3-30B-A3B) | `tab_moe` (`tab:moe`) | Prefill 1.4×/8.4× at 8k/32k; Dense OOMs at 128k while CoMem stays at a context-flat 63.0 GB and scores NIAH 100 | `comem.CoMemMoE` / `comem.load_moe_comem` | `scripts/qcmem_moe_selftest.py` |
| Model-family scale sweep | `tab_scale`, `tab_slm`, `tab_locomo_scale` | The depth partition holds across the Qwen3 family (0.6B → 32B / 30B-A3B) | `--model <family member> --j auto` | P2.3; `ruler_results/*` scale cohorts |
| Natural long-document stress test (InfiniteBench) | `tab_infbench` (`tab:infbench`) | CoMem completes all 351 QA items with a bounded read (F1 6.06 vs Dense 2.16); Dense OOMs on 194/229 choice documents | not shipped in `eval/` (InfiniteBench driver) | `scripts/eval_qcmem_infbench.py` |
| Cross-benchmark contamination audit + clean-subset rescore | `tab_priorart` discussion / statistics appendix | Audit-only; the headline conclusions survive the clean subset | not shipped (audit script) | P0.14 / `A-P1.2`; `paperA/sections/08_statistics_appendix.tex` |

---

## 3. Efficiency, storage and serving

| Experiment | Paper table | One-line result | Reproduce here | Research artifact |
|---|---|---|---|---|
| Store-ready online prefill at 8k–128k | `tab_online_prefill` (`tab:online-prefill`) | Prefill speedup 1.17× / 5.45× / 38.3× at 8k / 32k / 128k, with CoMem peak flat at ≈18.7 GB vs Dense 50.0 GB | `python bench/vs_dense.py --mode speed --model_path <M> --resume_j 12 --context_lengths 8k 32k 128k` | `bench_p0_13_quality_latency.py`; `tab_eff` cohort |
| Write/read/decode phase profile + Dense head-to-head | `tab_eff`, `tab_eff_lora`, `tab_mechanism_compact` | Batched Write is ~7× serial; the resumed-band KV decode is 4–16× faster per token than re-running the read every step | `python bench/vs_dense.py --mode all --model_path <M>` | `bench_qcmem_vs_dense_result.txt` |
| Unified quality–latency–storage operating points at 128k | `tab_pareto` (`tab:pareto`) | Full context 78.80 @ 147,456 B/token vs CoMem+LoRA 96.07 @ 8,192 B/token, 0.807 s/query after a 5.83 s one-time Write | combination of the rows above (not a single command) | `paperA/sections/08_appendix.tex` (efficiency appendix) |
| Equal-latency retrieval-budget frontier (BM25 and dense replay vs CoMem) | `tab_equal_latency` (`tab:equal-latency`), protocol in `tab_equal_latency_protocol` | At matched TTFT, BM25 replay beats CoMem by 11.56 pt; frozen-BGE replay is a statistical tie (CI spans zero) → selector choice is a first-order frontier variable | `--baseline kvdirect` (replay) vs `--baseline none`, sweeping `--topk` to match TTFT | `#137`/`#141` (P0.20 A/B); `bench_results/p0_20_{eqlat,phaseB_dense}/`; `paperA/P0_20_PHASEB_NOTES.md` |
| Repeated-query serving break-even `Q*` | `tab_serving_crossover` (`tab:serving-crossover`) | `Q* ≈ 8–28` at 32k/128k for small `G`; large-`G` cells are workload-specific, not a scaling law (some are `∞`) | not shipped (serving harness) | `#139` / P1.8; `scripts/bench_p1_8_serving_curve.py`; `paperA/artifacts/p1_8_serving/` |
| Persistent-store I/O tiers at a 16M-token store | `tab_store_io` (`tab:store-io`), `tab_store_scaling` | GPU residence fits 8M tokens but not 16M (128 GiB); off-GPU tiers keep transfer volume store-size-independent at rising latency (CPU-pinned 941 QPS → CEPH 42 QPS) | not shipped (storage microbenchmark) | P2.2; `scripts/bench_persistent_store_io.py` |
| Same-`j` LoRA self-distillation (teacher `j=0` → student `j`) | `tab_hy3_distill`, `tab_hy3_ruler`, `tab_distilled_depth_curve` | Distillation is what makes the frozen `j=12` interface usable (see the adapter ablation) | `python -m train.distill --model <M> --j auto --data <pg19> --out <OUT>` | `train/README.md`; `outputs/qcmem_distill_*` |
| Write-path distillation upper bound (LoRA on layers `[0:j)`) | `tab_write_context` (`†` row) | A trained chunk-local writer reaches 98.5, closing ~80% of the deployable Write gap; the residual to the 100 document-context oracle is not significant | not shipped (research-only upper bound; does not replace the flagship adapter) | `#142`/`#150` / P1.10; `paperA/P1_10_WRITEPATH_NOTES.md`; `bench_results/p0_18_e4_bbwl_step*/` |

---

## 4. Correctness gates (run these before trusting any number)

| Gate | Command | Checks |
|---|---|---|
| CoMem core + all baselines + dense selector | `python -m comem.selftest` | (A) `j=0` packing == stock forward, (B) resume identity at several `j`, (C) `encode`+`generate` == monolithic path per selector, (D) KV-cache decode == recompute decode, (E) the CacheBlend and SnapKV/PyramidKV gates, (F) `dense_bge` dispatch/tie-break/guard |
| CacheBlend only | `python -m comem.cacheblend` | RoPE reindex exact (chunk-local prefill + delta-rotate == prefill at the global offset), `r=1.0` == vanilla full prefill, `r=0.0` finite |
| SnapKV / PyramidKV only | `python -m comem.kvcompress` | No perturbation when the prompt is under budget (== stock attention), retained KV honours the budget (SnapKV uniform; PyramidKV pyramidal, averaging the budget) |
| Speed + decode correctness | `python bench/vs_dense.py` | KV-cache decode logits == recompute decode logits, then the Write/Read/decode profile and the Dense head-to-head |

Notes on the ported baselines:

- **`dense_bge`** and **`cacheblend`** are ports of the research implementations
  (`scripts/eval_p1_9_dense_rag.py::DenseRetriever` and the
  `QCMemModel.cacheblend_*` methods) and were cross-checked against them:
  identical top-k with max |Δscore| ≈ 4e-7 for the retriever, and **max |Δ| = 0**
  at every stage (per-chunk prefill, reindexed concat, read logits at
  `r ∈ {0, .15, .5, 1}`) for CacheBlend.
- **`snapkv` / `pyramidkv`** are re-implemented from the papers' specification
  because the upstream repos patch transformers 4.37 + Llama/Mistral, an API that
  no longer exists here. The selection rule was cross-checked against the
  vendored upstream `SnapKVCluster` / `PyramidKVCluster` and reproduces their
  retained K/V **bit-for-bit** (max |Δ| = 0 on every layer, both methods).
- **CacheBlend deviation to state when citing it:** the recompute set is seeded
  once at the bootstrap layer and held fixed across layers (the
  "faithful-minimal" variant) instead of being re-ranked every few layers. It is
  a *CacheBlend-style* baseline, and it does **not** compress storage — it caches
  the same bytes as a full KV cache (144 KiB/token on Qwen3-8B, 18× CoMem) and
  wins only on prefill/TTFT.
