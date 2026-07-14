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

- `selector`: `bm25` (lexical), `reader_attn` (cosine over `h_j`), `iter_bm25` /
  `iter_reader_attn` (multi-hop BFS for reference chains, e.g. RULER variable
  tracking), `recency`, `oracle`.
- `mode`: `comem` (retrieval, fixed read; default), `kvdirect` / `hcache`
  (no-retrieval baselines that pack **all** chunks — read grows O(context); build
  `CoMem(resume_j=0)` for a faithful `kvdirect`).
- Sharded MoE backbones: use `comem.CoMemMoE` / `comem.load_moe_comem`.

### Package layout

```
comem/
  model.py       # class CoMem: primitives (write/read/decode/resume) + encode/generate
  selectors.py   # bm25 / iter_bm25 / reader_attn / iter_reader_attn / recency / oracle
  moe.py         # CoMemMoE: device_map-sharded MoE variant
  selftest.py    # CPU correctness gate (python -m comem.selftest)
train/distill.py # LoRA self-distillation (teacher j=0 → student j) on PG19
eval/            # thin drivers: build CoMem + generate + official scoring
  ruler.py  babilong.py  longbench.py  locomo.py  longeval.py
bench/vs_dense.py# CoMem vs Dense speed/accuracy + decode correctness gate
paper/           # LaTeX source
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

## Reproducing the eval

Each `eval/*.py` builds a `CoMem`, runs `generate_from_ids` per sample (the fused
encode+write+select+decode over one prompt whose trailing chunk is the query), and
applies the benchmark's **official** metric. Example (RULER):

```bash
python -m eval.ruler --model_path /path/to/Qwen3-8B \
    --resume_j 12 --selector bm25 --topk 12 \
    --ruler_tasks niah_single niah_multi vt --lengths 4k 8k 16k 32k \
    --limit 50 --output_dir ruler_results/comem_j12
```

`eval/{longbench,longeval,locomo}.py` support `--score_only` to merge shards.
Head-to-head baselines: `--baseline {kvdirect,hcache}`. LoRA distillation:
`train/distill.py` then eval with `--lora_adapter <dir>`.
