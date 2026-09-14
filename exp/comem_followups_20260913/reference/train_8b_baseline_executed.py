"""CoMem suffix distillation and the teacher/student reader used in diagnostics.

No GPU is selected implicitly. The principal recipe and visibility conventions
are specified in the current manuscript's experimental protocol appendix.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_text(line):
    line = line.strip()
    if not line:
        return ""
    if line.startswith(("{", "[")):
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("A JSONL record must be an object with text/content.")
        return obj.get("text") or obj.get("content") or ""
    return line


def prepare_tokens(data, tokenizer, destination, tokenizer_source):
    """Concatenation matches S21 windows(): no inserted EOS and no shuffling."""
    data, destination = Path(data), Path(destination)
    meta_path = destination.with_suffix(destination.suffix + ".json")
    spec = {"source_sha256": digest(data), "tokenizer_source": str(tokenizer_source),
            "tokenizer_vocab_size": len(tokenizer), "format": "uint32-le-concatenated-v1",
            "add_special_tokens": False, "insert_document_eos": False}
    if destination.exists() or meta_path.exists():
        if not (destination.exists() and meta_path.exists()):
            raise ValueError("Incomplete token cache; use a new --token-cache path.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in spec.items()):
            raise ValueError("Token cache source/tokenizer specification differs.")
        if destination.stat().st_size != old["tokens"] * 4:
            raise ValueError("Token cache is truncated.")
        return old
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    docs, n = [], 0
    with data.open(encoding="utf-8") as src, tmp.open("wb") as dst:
        for line_index, line in enumerate(src):
            text = source_text(line)
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            np.asarray(ids, dtype="<u4").tofile(dst)
            docs.append({"line": line_index + 1, "token_start": n, "tokens": len(ids)})
            n += len(ids)
    if n == 0:
        tmp.unlink()
        raise ValueError("Training source produced no tokens.")
    os.replace(tmp, destination)
    meta = {**spec, "source_path": str(data.resolve()), "source_bytes": data.stat().st_size,
            "tokens": n, "documents": len(docs), "document_offsets": docs}
    atomic_json(meta_path, meta)
    return meta


class TokenStream:
    """Exact cursor resume, including windows crossing corpus/document boundaries."""
    def __init__(self, tokens, window, cursor=0):
        self.tokens, self.window, self.cursor = tokens, int(window), int(cursor)
        if len(tokens) < 1 or window < 1:
            raise ValueError("Empty corpus/window.")

    def take(self):
        idx = (np.arange(self.window, dtype=np.int64) + self.cursor) % len(self.tokens)
        result = torch.from_numpy(np.asarray(self.tokens[idx], dtype=np.int64).copy())
        self.cursor += self.window
        return result


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, adapter_dtype):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=adapter_dtype,
                                         device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=adapter_dtype,
                                         device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale, self.enabled = alpha / rank, True

    def forward(self, x):
        base = self.base(x)
        if not self.enabled:
            return base
        # FP32 master adapters use bf16 matmuls under CUDA autocast; CPU tests use fp32.
        delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return base + delta.to(base.dtype) * self.scale


def projection_modules(model):
    for i, layer in enumerate(model.model.layers):
        for nm in TARGETS:
            parent = layer.self_attn if nm in TARGETS[:4] else layer.mlp
            yield i, nm, parent


def attach_lora(model, j, rank, alpha, adapter_dtype):
    model.requires_grad_(False)
    modules = {}
    for i, nm, parent in projection_modules(model):
        if i >= j:
            mod = LoRALinear(getattr(parent, nm), rank, alpha, adapter_dtype)
            setattr(parent, nm, mod)
            modules[f"layers.{i}.{nm}"] = mod
    return modules


def flat_state(modules):
    return {f"{key}.{ab}": getattr(mod, ab).detach().cpu().clone()
            for key, mod in modules.items() for ab in ("A", "B")}


def restore_flat(modules, named):
    expected = {f"{key}.{ab}" for key in modules for ab in ("A", "B")}
    if set(named) != expected:
        raise ValueError("Checkpoint layer/projection keys differ from this model.")
    with torch.no_grad():
        for key, mod in modules.items():
            for ab in ("A", "B"):
                dst = getattr(mod, ab)
                if dst.shape != named[f"{key}.{ab}"].shape:
                    raise ValueError("Checkpoint adapter shape mismatch.")
                dst.copy_(named[f"{key}.{ab}"])


@contextlib.contextmanager
def adapters(modules, enabled):
    previous = [m.enabled for m in modules.values()]
    for m in modules.values():
        m.enabled = enabled
    try:
        yield
    finally:
        for m, old in zip(modules.values(), previous):
            m.enabled = old


def distribute(model, devices, splits=None):
    """Single process, contiguous layer model parallelism; no replicated 8B copy."""
    devices = [torch.device(d) for d in devices]
    L = len(model.model.layers)
    if splits is None:
        splits = [round(L * i / len(devices)) for i in range(1, len(devices))]
    bounds = [0] + list(splits) + [L]
    if len(bounds) != len(devices) + 1 or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("--layer-splits must be increasing interior boundaries, one per device transition.")
    model.model.embed_tokens.to(devices[0])
    model.model.rotary_emb.to(devices[0])
    for device, start, end in zip(devices, bounds, bounds[1:]):
        for layer in model.model.layers[start:end]:
            layer.to(device)
    model.model.norm.to(devices[-1])
    model.lm_head.to(devices[-1])
    return {"devices": list(map(str, devices)), "layer_boundaries": bounds}


class PublishedReader:
    """Exact published packing, with training checkpointing and explicit device transfers."""
    def __init__(self, model, j, grad_checkpoint=True):
        from comem import CoMem
        self.model, self.j, self.grad_checkpoint = model, j, grad_checkpoint
        self.reference = CoMem(model, resume_j=j)
        self.first = model.model.embed_tokens.weight.device

    def layers(self, hidden, start, end, use_checkpoint=False):
        hidden = hidden.to(self.first)
        pos = torch.arange(hidden.shape[1], device=self.first).unsqueeze(0)
        mask, pe = self.reference._make_mask_and_rope(hidden, pos)
        for layer in self.model.model.layers[start:end]:
            device = next(layer.parameters()).device
            hidden = hidden.to(device)
            local_pos = pos.to(device)
            local_pe = tuple(x.to(device) for x in pe)
            local_mask = mask.to(device) if torch.is_tensor(mask) else mask

            def call(h, block=layer, p=local_pos, r=local_pe, m=local_mask):
                out = block(h, attention_mask=m, position_ids=p,
                            position_embeddings=r, use_cache=False)
                return out[0] if isinstance(out, (tuple, list)) else out

            # Non-reentrant checkpoint handles a detached input with trainable layer params.
            if use_checkpoint and self.grad_checkpoint and torch.is_grad_enabled():
                hidden = checkpoint(call, hidden, use_reentrant=False)
            else:
                hidden = call(hidden)
        return hidden

    def write(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.first).reshape(1, -1)
        with torch.no_grad():
            return self.layers(self.model.model.embed_tokens(ids), 0, self.j).to(self.first)

    def hidden(self, window, bos, chunk, n_ctx, teacher=False):
        segments = [torch.tensor([bos], dtype=torch.long)] + list(window.split(chunk))
        if len(segments) != n_ctx + 2:
            raise ValueError("Unexpected training window size.")
        if teacher:
            ids = torch.cat(segments).to(self.first).unsqueeze(0)
            h = self.layers(self.model.model.embed_tokens(ids), 0, len(self.model.model.layers))
        else:
            # Each chunk AND query is written alone, with NO write sink; one sink in read pack.
            h = torch.cat([self.write(ids) for ids in segments], dim=1)
            h = self.layers(h, self.j, len(self.model.model.layers), use_checkpoint=True)
        return self.model.model.norm(h[:, -chunk:, :])


def support_loss(logits, idx, val, lam, mode):
    """Published trainer normalizes both p and q on teacher's selected support."""
    log_p = F.log_softmax(val.float(), dim=-1)
    if mode == "published":
        log_q = F.log_softmax(logits.float().gather(-1, idx), dim=-1)
    elif mode == "legacy-s21":
        log_q = F.log_softmax(logits.float(), dim=-1).gather(-1, idx)
    else:
        raise ValueError(mode)
    p, q = log_p.exp(), log_q.exp()
    return (lam * (p * (log_p - log_q)).sum(-1)
            + (1 - lam) * (q * (log_q - log_p)).sum(-1)).mean()


def batch_loss(reader, modules, window, bos, args):
    with torch.no_grad(), adapters(modules, False):
        teacher_h = reader.hidden(window, bos, args.chunk, args.n_ctx, teacher=True)
        targets = []
        for part in teacher_h.split(args.logit_chunk, dim=1):
            logits = reader.model.lm_head(part).float()
            top = logits.topk(min(args.topk, logits.shape[-1]), dim=-1)
            targets.append((top.indices, top.values))
            del logits
        del teacher_h
    # Remains enabled through backward, including all checkpoint recomputations.
    student_h = reader.hidden(window, bos, args.chunk, args.n_ctx)
    losses = []
    for part, (idx, val) in zip(student_h.split(args.logit_chunk, dim=1), targets):
        def head_loss(h, indices=idx, values=val):
            return support_loss(reader.model.lm_head(h), indices, values, args.lam, args.loss)
        loss = checkpoint(head_loss, part, use_reentrant=False) if torch.is_grad_enabled() else head_loss(part)
        losses.append(loss * part.shape[1] / args.chunk)
    return torch.stack(losses).sum()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1, progress)))


def rng_state(devices):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state(d) for d in devices]}


def restore_rng(state, devices):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    for device, value in zip(devices, state["cuda"]):
        torch.cuda.set_rng_state(value, device)


def peft_state(named):
    result = {}
    for key, value in named.items():
        _, index, proj, ab = key.split(".")
        parent = "self_attn" if proj in TARGETS[:4] else "mlp"
        result[f"base_model.model.model.layers.{index}.{parent}.{proj}.lora_{ab}.weight"] = value.contiguous()
    return result


def export_adapter(out, payload, args, layers):
    from safetensors.torch import save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_save(out / "adapter.pt", payload)
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
              "base_model_name_or_path": args.model, "r": args.rank, "lora_alpha": args.alpha,
              "lora_dropout": 0.0, "bias": "none", "target_modules": list(TARGETS),
              "layers_to_transform": list(range(args.j, layers)), "layers_pattern": "layers"}
    save_file(peft_state(payload["named"]), str(out / "adapter_model.safetensors"))
    atomic_json(out / "adapter_config.json", config)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", help="Explicit comma-separated CUDA devices; no GPU discovery or waiting.")
    ap.add_argument("--layer-splits", help="For two devices, e.g. 18; defaults to even contiguous layers.")
    ap.add_argument("--token-cache")
    ap.add_argument("--prepare-only", action="store_true", help="Tokenize on CPU without loading model weights.")
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=7)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--loss", choices=["published", "legacy-s21"], default="published")
    ap.add_argument("--adapter-dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--stop-after", type=int, help="Stop after this completed optimizer step without changing schedule.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--logit-chunk", type=int, default=32)
    ap.add_argument("--skip-windows", type=int, default=8, help="Matches S21's initial diagnostic window reservation.")
    ap.add_argument("--resume", help="Path to a full restart checkpoint (e.g. <out>/last.pt).")
    return ap


def main():
    args = parser().parse_args()
    if min(args.chunk, args.n_ctx, args.steps, args.rank, args.alpha, args.grad_accum,
           args.save_every, args.log_every, args.logit_chunk, args.topk) <= 0:
        raise ValueError("Chunk/context/steps/rank/alpha/intervals must be positive.")
    if args.skip_windows < 0 or args.warmup < 0 or not 0 <= args.lam <= 1:
        raise ValueError("Invalid skip/warmup/lambda.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cache = Path(args.token_cache or out / "pg19_tokens.u32")
    data_meta = prepare_tokens(args.data, tokenizer, cache, args.model)
    print(json.dumps({"data_documents": data_meta["documents"], "data_tokens": data_meta["tokens"],
                      "token_cache": str(cache), "prepare_only": args.prepare_only}), flush=True)
    if args.prepare_only:
        return
    if not args.devices:
        raise ValueError("Training requires explicit --devices cuda:N[,cuda:M].")
    devices = [torch.device(x.strip()) for x in args.devices.split(",")]
    if any(d.type != "cuda" or d.index is None for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("Select unique explicit CUDA indices.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process with --devices; torchrun/DDP would replicate the full model.")
    for device in devices:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"bf16 unsupported on {device}.")
    if not args.resume and (out / "last.pt").exists():
        raise ValueError("Output has a checkpoint; use --resume or a new output directory.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    from transformers import AutoModelForCausalLM
    import transformers
    import transformers.integrations.sdpa_attention as sdpa
    # Shared with the S15 eval kernel policy; avoids unsupported native-GQA SDPA fallback.
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               attn_implementation="sdpa", local_files_only=True).eval()
    if model.config.model_type != "qwen3" or not 0 < args.j < len(model.model.layers):
        raise ValueError("Validated only for dense Qwen3 with an interior split.")
    if model.config.attention_dropout != 0:
        raise ValueError("This recipe assumes attention_dropout=0.")
    splits = list(map(int, args.layer_splits.split(","))) if args.layer_splits else None
    placement = distribute(model, devices, splits)
    modules = attach_lora(model, args.j, args.rank, args.alpha, getattr(torch, args.adapter_dtype))
    params = [p for m in modules.values() for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    reader = PublishedReader(model, args.j)
    bos = tokenizer.bos_token_id
    bos = int(tokenizer.eos_token_id if bos is None else bos)
    tokens = np.memmap(cache, dtype="<u4", mode="r")
    window = args.chunk * (args.n_ctx + 1)
    stream = TokenStream(tokens, window, args.skip_windows * window)
    config = {k: getattr(args, k) for k in ("j", "rank", "alpha", "chunk", "n_ctx", "topk", "lam",
              "loss", "adapter_dtype", "steps", "lr", "warmup", "seed", "grad_accum", "skip_windows")}
    config.update(model_config=model.config.to_dict(), source_sha256=data_meta["source_sha256"],
                  data_tokens=len(tokens), tokenizer_vocab_size=len(tokenizer))
    metadata = {"recipe": config, "args": vars(args), "data": data_meta, "placement": placement,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "trainable_parameters": sum(p.numel() for p in params), "path": "base",
                "checkpointing": "non-reentrant, each upper layer and query head chunk",
                "precision": "bf16 unquantized frozen backbone; bf16 CUDA autocast",
                "world_size": 1, "global_windows_per_step": args.grad_accum,
                "data_note": "Sequential cyclic PG19 concatenation; first eight windows skipped once. No heldout-generalization claim."}
    completed = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["metadata"]["recipe"] != config:
            raise ValueError("Resume recipe/model/data differs. Schedule length may not change on resume.")
        restore_flat(modules, saved["named"])
        opt.load_state_dict(saved["optimizer"])
        completed, stream.cursor = int(saved["step"]), int(saved["token_cursor"])
        # Logical device count may change; CPU/adapter RNG is restored; CUDA RNG states require same count.
        if len(saved["rng"]["cuda"]) != len(devices):
            raise ValueError("Exact RNG resume requires the same number of selected devices.")
        restore_rng(saved["rng"], devices)
    atomic_json(out / "metadata.json", metadata)
    last_loss = None
    def save(step, final=False):
        adapter = {"args": vars(args), "j": args.j, "path": "base", "rank": args.rank,
                   "alpha": args.alpha, "targets": list(TARGETS), "step": step,
                   "named": flat_state(modules), "metadata": metadata}
        restart = {**adapter, "optimizer": opt.state_dict(), "token_cursor": stream.cursor,
                   "rng": rng_state(devices)}
        atomic_save(out / "last.pt", restart)
        if step % args.save_every == 0 or final:
            export_adapter(out / f"step{step}", adapter, args, len(model.model.layers))
        if final:
            export_adapter(out / "final", adapter, args, len(model.model.layers))
        atomic_json(out / "status.json", {"step": step, "target_steps": args.steps,
                    "complete": final, "loss": last_loss, "token_cursor": stream.cursor,
                    "training_tokens_processed": step * args.grad_accum * window,
                    "training_windows_processed": step * args.grad_accum,
                    "corpus_passes_including_initial_skip": stream.cursor / len(tokens),
                    "checkpoint": str(out / "last.pt"),
                    "cuda_peak_allocated_gib": {str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in devices}})
    stop = min(args.steps, args.stop_after or args.steps)
    if stop < completed:
        raise ValueError("--stop-after precedes the resumed completed step.")
    started = time.monotonic()
    print(json.dumps({"trainable_parameters": metadata["trainable_parameters"], "placement": placement,
                      "start_step": completed, "stop_step": stop, "recipe": config}, default=str), flush=True)
    with (out / "train.jsonl").open("a", encoding="utf-8") as log:
        for step in range(completed, stop):
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args)
            losses = []
            for _ in range(args.grad_accum):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = batch_loss(reader, modules, stream.take(), bos, args)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step + 1}; resume previous checkpoint.")
                (loss / args.grad_accum).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True, foreach=False)
            opt.step()
            completed = step + 1
            last_loss = sum(losses) / len(losses)
            row = {"step": completed, "loss": last_loss, "lr": lr_at(step, args),
                   "grad_norm": float(norm), "elapsed_s": time.monotonic() - started,
                   "token_cursor": stream.cursor}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if completed == 1 or completed % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if completed % args.save_every == 0 or completed == stop:
                save(completed, final=completed == args.steps)
    if completed == stop and not (out / "last.pt").exists():
        save(completed, final=completed == args.steps)


if __name__ == "__main__":
    main()
