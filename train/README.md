# CoMem self-distillation (`train/distill.py`)

**The core CoMem read is training-free.** This LoRA self-distillation is an
*optional* enhancement: it lets you **resume from a deeper split `j`** (a cheaper,
smaller read pack) without paying the *depth cliff* on precise-localisation tasks,
by teaching the upper layers to reconstruct the full-model behaviour from the
shallow depth-`j` cache. In-window it pulls the resumed read back up to the dense
upper bound; out of window it inherits CoMem's constant-cost length robustness.

## Method — self-distillation (teacher `j=0`, student `j`)

One model instance, LoRA toggled on/off to be student/teacher, so **no second copy
in memory**:

- **Teacher** = CoMem read at `resume_j = 0` with the adapters **disabled**
  (`peft.disable_adapter()`) under `no_grad`. At `j=0` the packed read
  `[sink ; ctx… ; query]` is re-forwarded through the **whole** model with the
  query present — i.e. exactly the frozen base model on the packed sequence (the
  RAG upper bound, no loss).
- **Student** = CoMem read at `resume_j = j` (default 12) with LoRA **on**. The
  sink + context chunks are cached at depth `j` by the **frozen** bottom
  `layers[0:j]` (query-blind, `no_grad`); only the resume path `layers[j:]` — where
  the LoRA lives — is grad-bearing and learns to reconstruct the teacher from that
  shallow cache. The backbone is frozen; only the LoRA params train.
- **Loss** = bidirectional top-k KL on the teacher's top-`k` support over the
  query-segment tokens:
  `loss = λ·KL(p‖q) + (1−λ)·KL(q‖p)` (λ = `--distill_lambda`, default 0.6),
  optionally `+ ce_weight · CE(student, teacher_argmax)` (`--ce_weight`, default 0).
  The teacher carries no grad; grad flows only through the student readout.
- **Data** = PG19 natural text (`--data`), streamed and tokenised on the fly, packed
  into `(n_ctx+1)·chunk_size`-token windows = `n_ctx` context chunks + 1 query chunk.
  **Pure self-supervision — no eval data, no needles.** A JSONL file with a `"text"`
  field per line, or one raw document per line, both work. Each DDP rank streams a
  disjoint shard (`[rank::world_size]`).

Because the student read runs the decoder layers **directly** (not through
`DDP.forward`), DDP's gradient hooks would never fire — so the trainer broadcasts
the replicated adapter from rank 0 and does an **explicit grad all-reduce** (mean
over ranks) after `backward`. This is the correct pattern for a custom
non-`forward` autograd graph.

## Run

```bash
# single GPU
python -m train.distill --model /path/to/Qwen3-8B --j auto \
    --data data/pg19_train.jsonl --out outputs/comem_distill_j12

# 8-GPU DDP
torchrun --nproc_per_node 8 -m train.distill \
    --model /path/to/Qwen3-8B --j 12 --lora_rank 32 \
    --data data/pg19_train.jsonl --total_steps 1000 \
    --out outputs/comem_distill_j12

# correctness gate only (fp32, no training): teacher==full forward at j=0
python -m train.distill --model /path/to/Qwen3-8B --j 12 \
    --out /tmp/_ck --self_test
```

`--j auto` picks the per-backbone split depth from `comem/model_registry.py`
(Qwen3-8B → 12), matching the eval CLI.

## Feed the adapter to eval

The trainer writes `--out/step{N}/` checkpoints and a final `--out/final/`. Pass
either to any eval driver's `--adapter`:

```bash
python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \
    --adapter outputs/comem_distill_j12/final \
    --lengths 8k,16k,32k --n 100 --out ruler_results/qwen3_8b_distill
```

`eval/_common.load_backbone` applies the LoRA (`PeftModel.from_pretrained`) and
hands CoMem the wrapped `base_model.model`, so the delta is live when CoMem calls
`layers[j:]` at read time. Eval with `--adapter none` (or omitting it) is the
zero-training CoMem.

## CLI (unified with eval)

| Flag | Alias | Default | Meaning |
|------|-------|:-------:|---------|
| `--model` | `--model_path` | *(required)* | Local HF causal-LM path |
| `--j` | `--resume_j` | `12` | Student split depth (int or `auto`); teacher is always 0 |
| `--data` | `--pg19_path` | `data/pg19_train.jsonl` | PG19 / raw-text corpus |
| `--out` | `--output_dir` | *(required)* | Adapter output dir |
| `--lora_rank` / `--lora_alpha` | | `32` / `32` | LoRA rank / alpha on `layers[j:]` |
| `--chunk_size` / `--n_ctx` | | `512` / `7` | Chunk length / context chunks per window (→ 4096-tok window) |
| `--teacher_topk` | | `64` | Top-k teacher support for the KL |
| `--distill_lambda` / `--ce_weight` | | `0.6` / `0.0` | KL mix / optional CE-to-argmax |
| `--total_steps` / `--lr` / `--warmup_steps` | | `1000` / `1e-4` / `50` | Optimizer schedule |
| `--gradient_checkpointing` | `--no-gradient_checkpointing` | on | Grad-checkpoint the read layer loop |
| `--save_interval` | | `250` | Checkpoint cadence |
| `--top_prepay_b` | | `0` | Student top-prepay (0 = exact connective resume) |

## Cost

~1–2 GPU-hours for the 1000-step default on Qwen3-8B (8× GPU DDP, `n_ctx=7` →
4096-tok window, grad-checkpointed). Only the LoRA params (+ AdamW state) train;
the 8B backbone is frozen and shared between teacher and student.

## Self-contained

`train/distill.py` depends only on the local `comem` package (+ `torch`,
`transformers`, `peft`) — **no dependency on any research repo**. The `--self_test`
gate and `python -m comem.selftest` both verify the CoMem read/write packing
reproduces a stock full forward before you train.
