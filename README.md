# CoMem — Comprehension Memory

**Fixed-size, mid-depth-resume long-context memory for a plain (un-patched) decoder LLM.**

CoMem lets an 8B model answer questions over arbitrarily long contexts with a
**constant-size, constant-cost read**, by exploiting a simple observation:

> A transformer completes most of its **comprehension** of a token span in its
> *lower* layers; the *upper* layers increasingly just produce the next-token
> distribution.

So CoMem splits the backbone at a depth `j` (`resume_j`):

- **WRITE** (once, per chunk, chunk-local): `embed → layers[0:j]` over each chunk in
  isolation, and **cache the depth-`j` hidden `h_j`** (the chunk's comprehended,
  mid-layer representation) plus its raw token ids (for retrieval).
- **READ** (per query): **retrieve** the `topk` most relevant cached chunks, **pack**
  `[sink ; h_j^{c1} ; … ; h_j^{ck} ; h_j^{query}]` into one sequence with fresh
  contiguous RoPE positions, and **resume** `layers[j:] → norm → lm_head`. Only the
  upper layers are recomputed, over a *fixed-size* pack — so read cost does not grow
  with the context length.

`j = 0` is the RAG upper bound (selective full re-forward); `j = L` is closed-book.

### Why it works / headline results (Qwen3-8B; see `paper/`)

- **Length robustness.** Full-context attention **collapses to 0** past its RoPE
  window, while CoMem holds **RULER ≈ 100** and **LongEval ≈ 0.98 at 128k tokens** —
  because the read pack is always a handful of chunks, never the whole context.
- **Decode speed.** The resumed-band **KV-cache decode** (prefill both bands once,
  then push one token/step) runs **4–16× faster per token** than re-running the whole
  read every step, with byte-identical output.
- **Cheap comprehension memory.** `h_j` is computed once per chunk with only the
  bottom `j` layers; retrieval is forward-free (lexical BM25 or cosine over the
  cached `h_j`), so adding memory adds almost no compute over the writes CoMem
  already does.

## Install

```bash
pip install -r requirements.txt
# BABILong eval additionally needs the `babilong` package (pip install babilong).
```

Requires a local causal-LM checkpoint (Llama-3-8B, Qwen3-8B, or an in-tree MoE).

## Minimal use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from comem import CoMem

tok = AutoTokenizer.from_pretrained(PATH)
lm  = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype="bfloat16").cuda().eval()

model = CoMem(lm, resume_j=12, tokenizer=tok)   # split the backbone at layer 12
model.encode(long_document)                     # comprehend once → cache h_j per chunk
answer = model.generate("What is X?",           # retrieve topk → resume → decode
                        selector="bm25", topk=12, max_new_tokens=32)
```

- `selector`: `auto` (**default for RULER — data-validated per-task routing**
  (8B RULER n=500): `variable_tracking` → **fixed `iter_bm25`** (multi-hop BFS on
  the literal VAR chain, `iter_hop_topk=4`), all `niah_*` → **`bm25`** (single-shot
  lexical). This replaces the old single universal `iter_bm25_adaptive` default,
  whose confidence early-stop collapsed the VT chain (31/25/22 vs `iter_bm25`
  97/97/98 on 8k/16k/32k) and hurt `niah_multikey` (91/70/32 vs `bm25` 91/91/92);
  pass an explicit `--selector` to override it on ALL tasks for controls), `bm25`
  (lexical), `reader_attn` (cosine over `h_j`), `iter_bm25` /
  `iter_reader_attn` (multi-hop BFS for reference chains, e.g. RULER variable
  tracking), `iter_bm25_adaptive` (confidence-adaptive `iter_bm25`: no fixed
  `topk` budget — walk the chain until a hop's best score drops below
  `--iter_conf_ratio`× the round-1 best or `--iter_max_chunks` is hit; **kept as
  an opt-in selector for ρ-tuning experiments, no longer the default**),
  `recency`, `oracle`, `dense_bge` (frozen BGE dense retrieval — CLS+L2+cosine
  over each chunk's decoded text; the single-variable "dense instead of lexical"
  arm, needs `--retriever_path`).
- `mode`: `comem` (retrieval, fixed read; default), `kvdirect` / `hcache`
  (no-retrieval baselines that pack **all** chunks — read grows O(context); build
  `CoMem(resume_j=0)` for a faithful `kvdirect`).
- Sharded MoE backbones: use `comem.CoMemMoE` / `comem.load_moe_comem`.

### Package layout

```
comem/
  model.py       # class CoMem: primitives (write/read/decode/resume) + encode/generate
  selectors.py   # bm25 / iter_bm25 / iter_bm25_adaptive / reader_attn / iter_reader_attn / recency / oracle / dense_bge
  moe.py         # CoMemMoE: device_map-sharded MoE variant
  cacheblend.py  # CacheBlend-style full-depth chunk-KV baseline (+ its own gate)
  kvcompress.py  # SnapKV / PyramidKV prefill-then-compress baselines (+ its own gate)
  selftest.py    # CPU correctness gate (python -m comem.selftest)
train/distill.py # LoRA self-distillation (teacher j=0 → student j) on PG19
eval/            # thin drivers: build CoMem + generate + official scoring
  ruler.py  babilong.py  longbench.py  locomo.py  longeval.py
bench/vs_dense.py# CoMem vs Dense speed/accuracy + decode correctness gate
paper/           # LaTeX source
EXPERIMENTS.md   # experiment -> paper table -> CLI flag provenance index
```

## Correctness

`generate` is byte-identical to the reference research implementation.
`python -m comem.selftest` (CPU, tiny random Qwen3, fp32) checks:

- **(A)** `j=0` write/read packing == a stock `model(input_ids)` forward (diff `0`),
- **(B)** `resume_forward_ids` == full forward at several `j`,
- **(C)** `encode`+`generate` == the monolithic `generate_from_ids` for every
  selector (identical tokens),
- **(D)** KV-cache decode == recompute decode (identical tokens, max|logit diff|
  `< 1e-4`).
- **(E)** the baseline gates — CacheBlend (RoPE reindex exact, `r=1` == full
  prefill, `r=0` finite) and SnapKV/PyramidKV (no perturbation when no
  compression fires, retained-KV budget honoured). Run them alone with
  `python -m comem.cacheblend` / `python -m comem.kvcompress`.
- **(F)** the `dense_bge` selector's dispatch, tie-break and fail-closed guard
  (via an offline stub encoder, so no checkpoint is needed).

## Reproducing the eval

Each `eval/*.py` builds a `CoMem`, runs `generate_from_ids` per sample (the fused
encode+write+select+decode over one prompt whose trailing chunk is the query), and
applies the benchmark's **official** metric.

### One command, any cell

Every driver shares one unified CLI, so a single habit works everywhere:

```
--model <hf_path_or_name>   --j <int|auto>   --lengths 8k,16k,32k,64k,128k
--n <samples>   --selector bm25   --adapter <path|none>
--baseline <none|dense|kvdirect|hcache|streamingllm|snapkv|pyramidkv|cacheblend>
--out <dir>
# baseline-specific: --recompute_ratio (cacheblend) --kv_budget/--kv_window
#                    (snapkv/pyramidkv) --retriever_path (--selector dense_bge)
```

Run through the dispatcher (routes `--benchmark` to the matching driver):

```bash
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --lengths 8k,16k,32k --n 100 --selector bm25 --out ruler_results/qwen3_8b
```

…or the convenience wrapper (`--j auto` and env-override defaults):

```bash
./run_cell.sh ruler /path/to/Qwen3-8B --lengths 8k,16k,32k --n 100
# env overrides: PYTHON_BIN=..  J=12  BASELINE=dense  SELECTOR=reader_attn
```

The old native flags (`--model_path`, `--resume_j`, `--limit`/`--num_samples`/
`--max_samples`, `--output_dir`, space-separated lengths) still work as aliases.

### Model → split depth (`--j auto`)

`--j auto` picks the per-backbone split depth from `comem/model_registry.py`
(`resume_j ≈ round(0.33 · num_hidden_layers)`); unknown models fall back to that
formula with a warning.

| Model         | Layers L | `--j` |
|---------------|:--------:|:-----:|
| Qwen3-0.6B    | 28       | 9     |
| Qwen3-1.7B    | 28       | 9     |
| Qwen3-4B      | 36       | 12    |
| Qwen3-8B      | 36       | 12    |
| Qwen3-14B     | 40       | 13    |
| Qwen3-32B     | 64       | 21    |
| Qwen3-30B-A3B | 48       | 16    |

### One-line run per benchmark (copy-paste)

```bash
# RULER   (NIAH + variable-tracking; synthetic length sweep)
# --selector defaults to `auto` (per-task routing): vt -> fixed iter_bm25, niah_* -> bm25 (data-validated on 8B RULER n=500)
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --lengths 8k,16k,32k,64k,128k --n 100 --out ruler_results/qwen3_8b

# BABILong (qa1..qa10 x lengths; needs `pip install babilong`)
python -m eval.run --benchmark babilong --model /path/to/Llama-3-8B --j auto \
    --tasks qa1,qa2,qa5 --lengths 0k,1k,2k,4k,8k,16k --n 100 --out babilong_results/llama3_8b

# LongBench (real long-doc QA; per-dataset SQuAD F1/EM)
python -m eval.run --benchmark longbench --model /path/to/Qwen3-8B --j auto \
    --tasks narrativeqa,qasper,hotpotqa,2wikimqa,musique,multifieldqa_en \
    --out longbench_results/qwen3_8b

# LongEval (LongChat lines-retrieval; exact-value accuracy)
python -m eval.run --benchmark longeval --model /path/to/Qwen3-8B --j auto \
    --lengths 4k,8k,16k,32k,64k,128k --n 50 --out longeval_results/qwen3_8b

# LoCoMo (long-conversation memory QA; F1/EM/acc by category)
python -m eval.run --benchmark locomo --model /path/to/Qwen3-8B --j auto \
    --locomo_data data/locomo10.json --out locomo_results/qwen3_8b
```

`--lengths` applies to RULER / BABILong / LongEval (synthetic sweeps); LongBench
and LoCoMo iterate their fixed datasets and use `--tasks` / `--locomo_data`.

### Results & official scoring

| Benchmark | Output (`--out`)                     | Metric / scorer |
|-----------|--------------------------------------|-----------------|
| RULER     | `<out>/{task}_{len}.csv` + `_summary.json` | `string_match` recall (RULER `string_match_all`; ref strings as case-insensitive substrings) |
| BABILong  | `<out>/{task}_{len}_..._csv`         | official `babilong.metrics` — `TASK_LABELS` + `compare_answers` (**never** bare `re.search`) |
| LongBench | `<out>/{ds}_*.jsonl` + `scores.json` | SQuAD-style token-F1 / EM (`--score_only` merges shards) |
| LongEval  | `<out>/longeval_{len}.json` + `_summary.json` | exact-value match accuracy |
| LoCoMo    | `<out>/preds*.jsonl` + `scores.json` | F1 / EM / substring-acc (cat-5 = abstention-correct; `--score_only` merges shards) |

`eval/{longbench,longeval,locomo}.py` support `--score_only` to merge shards.
LoRA distillation: `train/distill.py` then eval with `--adapter <dir>`.

### Baselines (`--baseline`)

| `--baseline`               | What it is                                                                                                                                | Stored per token |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `none`                     | CoMem itself (retrieval + fixed read at `--j`)                                                                                             | one depth-`j` residual (8 KiB on Qwen3-8B) |
| `kvdirect` / `hcache`      | no-retrieval CoMem packs (**all** chunks; read grows O(context))                                                                            | — |
| `dense`                    | stock full-context generation                                                                                                              | — (full prefill each query) |
| `streamingllm`             | sink + sliding-window truncation, then dense                                                                                               | — |
| `snapkv` / `pyramidkv`     | prefill-then-compress KV: **full (exact) prefill**, then evict to `--kv_budget` retained tokens/layer and decode from it (`comem.kvcompress`) | bounded KV, but the whole prompt is still prefilled |
| `cacheblend`               | CacheBlend-style full-depth chunk KV: same selector/pack/sink as CoMem, but caches every layer's chunk K/V, reindexes RoPE and recomputes a `--recompute_ratio` slice (`comem.cacheblend`) | full `L`-layer KV (144 KiB on Qwen3-8B, 18× CoMem) |

`snapkv`/`pyramidkv` default to `--kv_budget 6657` = CoMem's read pack
(BOS 1 + top-12 × 512 + query ≤ 512), which makes the quality row an
equal-retained-token diagnostic. `cacheblend` does **not** compress storage — it
caches the same bytes as a full KV cache and wins only on prefill/TTFT, so report
its 144 KiB/token tier alongside any latency win.

```bash
# CacheBlend-style arm, single-variable vs CoMem (same selector/chunk/topk/sink)
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --baseline cacheblend --recompute_ratio 0.15 --selector iter_bm25 --topk 12 \
    --out ruler_results/cacheblend_r015

# equal-retained-budget compressed-KV arms
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --baseline snapkv --kv_budget 6657 --kv_window 32 --out ruler_results/snapkv

# dense-retrieval selector swap (CoMem reader unchanged; only the ranking differs)
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --selector dense_bge --retriever_path /path/to/bge-large-en-v1.5 --topk 12 \
    --out ruler_results/dense_bge
```

See `EXPERIMENTS.md` for which paper table each of these reproduces.

### Division of labor

Eval is a fixed pipeline; only *(model, j, benchmark, length, selector, baseline)*
vary. Suggested split: **one person owns one model column** (`--model X --j auto`)
and sweeps all five benchmarks × all baselines for it, e.g.

```bash
for B in ruler babilong longbench longeval locomo; do
  for BASE in none dense kvdirect hcache streamingllm; do
    ./run_cell.sh $B /path/to/Qwen3-8B --baseline $BASE --out results/qwen3_8b/$B/$BASE
  done
done
```

