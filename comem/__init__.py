"""CoMem — Comprehension Memory.

Fixed-size mid-depth-resume long-context memory over a plain (un-patched) decoder
LLM. See :class:`comem.model.CoMem` for the full contract.

Quick start
-----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from comem import CoMem

    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path).cuda().eval()

    model = CoMem(lm, resume_j=12, tokenizer=tok)
    model.encode(long_document)                 # comprehend once (cache h_j)
    answer = model.generate("What is X?", selector="bm25", topk=12)
"""
from .model import CoMem
from .moe import CoMemMoE, load_moe_comem
from . import selectors

__all__ = ["CoMem", "CoMemMoE", "load_moe_comem", "selectors"]
