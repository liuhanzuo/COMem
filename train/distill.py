#!/usr/bin/env python
"""CoMem self-distillation — recover the mid-depth-resume depth cliff with LoRA.

The zero-training CoMem read already breaks the read-out gap at moderate ``j`` but
COLLAPSES on precise-localisation tasks (the "depth cliff"). This trains a LoRA
self-distillation to push the cliff back:

  * TEACHER = CoMem read at ``j = 0`` (RAG upper bound: retrieved chunks
    re-forwarded through the WHOLE model with the query present), adapters DISABLED
    under ``no_grad`` -> exactly the frozen base model on the packed sequence.
  * STUDENT = CoMem read at ``j = --resume_j``: chunks cached at depth ``j`` from
    the FROZEN bottom ``layers[0:j]``; LoRA on ``layers[j:]`` ONLY learns to
    reconstruct the teacher from the shallow cache.
  * LOSS = bidirectional top-k KL on the QUERY-segment tokens (+ optional CE to
    teacher argmax).
  * DATA = PG19 natural text (``--pg19_path``, one JSONL line per doc), streamed +
    tokenised on the fly, packed into ``[sink ; ctx chunks ; query chunk]``. PURE
    self-supervision — NO eval data.

Teacher + student share ONE model (adapters on/off), so no second copy in memory.
Single- or multi-GPU DDP (explicit grad all-reduce, since the read runs the layers
directly rather than through ``DDP.forward``). Load the resulting adapter with
``CoMem(..., lora_adapter=...)`` via ``eval._common.load_backbone``.

Usage:
    torchrun --nproc_per_node 8 -m train.distill \\
        --model_path /path/to/Qwen3-8B --resume_j 12 --lora_rank 16 \\
        --pg19_path data/pg19_train.jsonl --total_steps 1000 \\
        --output_dir outputs/comem_distill_j12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comem import CoMem  # noqa: E402


# --------------------------------------------------------------------------- #
# top-k KL distillation loss (Scheme-A bidirectional, on teacher top-k support)
# --------------------------------------------------------------------------- #
def _weighted_token_mean(loss, token_weight=None):
    if token_weight is None:
        return loss.mean()
    w = token_weight[:loss.shape[0]].to(device=loss.device, dtype=loss.dtype).detach()
    denom = w.sum().clamp_min(torch.finfo(loss.dtype).eps)
    return (loss * w).sum() / denom


def distill_logits_kl(student_logits, teacher_idx, teacher_val, lam=0.6,
                      token_weight=None):
    """Bidirectional KL on the teacher top-k support. For each answer token:
    ``loss = lam*KL(p||q) + (1-lam)*KL(q||p)`` where ``p`` = softmax of the teacher
    top-k logits and ``q`` = softmax of the student logits gathered at those same
    indices. Teacher (``p``) carries no grad; grad flows only through the readout."""
    teacher_val = teacher_val.to(student_logits.dtype)
    p = torch.softmax(teacher_val, dim=-1)
    q_logits = torch.gather(student_logits, -1, teacher_idx)
    log_q = torch.log_softmax(q_logits, dim=-1)
    log_p = torch.log_softmax(teacher_val, dim=-1)
    q = log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(dim=-1)
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)
    loss = lam * kl_pq + (1.0 - lam) * kl_qp
    return _weighted_token_mean(loss, token_weight)


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def _dist_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _is_main(rank):
    return rank == 0


# --------------------------------------------------------------------------- #
# PG19 streaming packer — builds [sink ; ctx chunks ; query chunk] windows
# --------------------------------------------------------------------------- #
class PG19Packer:
    """Stream a JSONL / raw-text corpus, tokenise on the fly, yield packed windows
    of ``(n_ctx + 1) * chunk_size`` tokens = ``n_ctx`` context chunks + 1 query
    chunk. Sharded across DDP ranks by ``[rank::world_size]`` over produced windows."""

    def __init__(self, path, tokenizer, chunk_size, n_ctx, rank, world_size, seed):
        self.path = path
        self.tok = tokenizer
        self.chunk_size = int(chunk_size)
        self.n_ctx = int(n_ctx)
        self.window_len = (self.n_ctx + 1) * self.chunk_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def _windows(self):
        buf: List[int] = []
        wcount = 0
        while True:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.extend(self.tok.encode(line, add_special_tokens=False))
                    while len(buf) >= self.window_len:
                        w = buf[: self.window_len]
                        buf = buf[self.window_len:]
                        if wcount % self.world_size == self.rank:
                            yield w
                        wcount += 1

    def stream(self):
        for w in self._windows():
            toks = torch.tensor(w, dtype=torch.long)
            chunks = list(toks.split(self.chunk_size))
            yield {"ctx": chunks[: self.n_ctx], "query": chunks[self.n_ctx]}


def main():
    p = argparse.ArgumentParser(description="CoMem LoRA self-distillation on PG19")
    p.add_argument("--model_path", required=True)
    p.add_argument("--resume_j", type=int, default=12)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--pg19_path", default="data/pg19_train.jsonl")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_ctx", type=int, default=3)
    p.add_argument("--query_loss_tokens", type=int, default=0)
    p.add_argument("--teacher_topk", type=int, default=64)
    p.add_argument("--distill_lambda", type=float, default=0.6)
    p.add_argument("--ce_weight", type=float, default=0.0)
    p.add_argument("--total_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_interval", type=int, default=250)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    rank, world_size, local_rank = _dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True).to(device)
    base.config.use_cache = False
    L = int(base.config.num_hidden_layers)

    for prm in base.parameters():
        prm.requires_grad = False
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets,
        layers_to_transform=list(range(args.resume_j, L)), layers_pattern="layers")
    peft_model = get_peft_model(base, lora_cfg)
    causal_lm = peft_model.base_model.model
    peft_model.train()
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    qc = CoMem(causal_lm, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
               tokenizer=tokenizer)
    qc.grad_checkpoint = bool(args.gradient_checkpointing)
    train_params = [prm for prm in peft_model.parameters() if prm.requires_grad]
    if _is_main(rank):
        print(f"[comem-distill] LoRA on layers[{args.resume_j}:{L}] targets={targets} "
              f"-> trainable {sum(x.numel() for x in train_params)/1e6:.2f}M", flush=True)

    def _allreduce_grads_mean():
        if world_size <= 1:
            return
        for prm in train_params:
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)
            dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
            prm.grad /= world_size

    opt = torch.optim.AdamW(train_params, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if world_size > 1:
        for prm in train_params:
            dist.broadcast(prm.data, src=0)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    packer = PG19Packer(args.pg19_path, tokenizer, args.chunk_size, args.n_ctx,
                        rank, world_size, args.seed)
    stream = packer.stream()
    os.makedirs(args.output_dir, exist_ok=True)
    if _is_main(rank):
        json.dump(vars(args), open(os.path.join(args.output_dir, "distill_args.json"), "w"),
                  indent=2)

    t0, running, seen, step, micro = time.time(), 0.0, 0, 0, 0
    opt.zero_grad(set_to_none=True)
    while step < args.total_steps:
        sample = next(stream)
        ctx_chunks = [c.to(device) for c in sample["ctx"]]
        query_chunk = sample["query"].to(device)
        T_q = int(query_chunk.shape[0])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # TEACHER: j=0 read (adapters disabled), no grad
        with torch.no_grad():
            with peft_model.disable_adapter():
                qc_t = CoMem(causal_lm, resume_j=0, top_prepay_b=0, tokenizer=tokenizer)
                t_sink = qc_t.write_chunk([bos_id])
                t_ctx = [qc_t.write_chunk(c) for c in ctx_chunks]
                t_q = qc_t.write_chunk(query_chunk)
                t_logits = qc_t.read_core(t_sink, t_ctx, t_q, logits_tail=T_q)[0].float()
                n_loss = min(args.query_loss_tokens, T_q) if args.query_loss_tokens > 0 else T_q
                t_loss_logits = t_logits[-n_loss:]
                tk = torch.topk(t_loss_logits,
                                k=min(args.teacher_topk, t_loss_logits.shape[-1]), dim=-1)
                teacher_idx, teacher_val = tk.indices, tk.values
                teacher_argmax = teacher_idx[:, 0]

        # STUDENT: j=resume_j read (adapters ON), grad only in layers[j:]
        with torch.no_grad():
            s_sink = qc.write_chunk([bos_id])
            s_ctx = [qc.write_chunk(c) for c in ctx_chunks]
            s_q = qc.write_chunk(query_chunk)
        s_logits = qc.read_core(s_sink, s_ctx, s_q, logits_tail=T_q)
        s_loss_logits = s_logits[0][-n_loss:].float()

        loss = distill_logits_kl(s_loss_logits, teacher_idx, teacher_val,
                                 lam=args.distill_lambda)
        if args.ce_weight > 0.0:
            loss = loss + args.ce_weight * F.cross_entropy(s_loss_logits, teacher_argmax)
        (loss / args.grad_accum).backward()
        running += float(loss.detach().item())
        seen += 1
        micro += 1

        if micro % args.grad_accum == 0:
            _allreduce_grads_mean()
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if _is_main(rank) and step % args.log_interval == 0:
                dt = time.time() - t0
                print(f"[comem-distill] step {step}/{args.total_steps} "
                      f"loss {running/max(1,seen):.4f} lr {lr_at(step):.2e} "
                      f"{seen*world_size/dt:.1f} samp/s", flush=True)
                running, seen, t0 = 0.0, 0, time.time()
            if _is_main(rank) and (step % args.save_interval == 0 or step == args.total_steps):
                sd = os.path.join(args.output_dir, f"step{step}")
                os.makedirs(sd, exist_ok=True)
                peft_model.save_pretrained(sd)
                print(f"[comem-distill] saved LoRA -> {sd}", flush=True)

    if _is_main(rank):
        fd = os.path.join(args.output_dir, "final")
        os.makedirs(fd, exist_ok=True)
        peft_model.save_pretrained(fd)
        print(f"[comem-distill] DONE -> {fd}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
