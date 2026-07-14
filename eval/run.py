#!/usr/bin/env python
"""Top-level CoMem eval dispatcher.

One entry point for every benchmark; routes ``--benchmark`` to the matching
``eval.<benchmark>`` driver and forwards all remaining args to that driver's own
(unified) CLI.

Usage
-----
    python -m eval.run --benchmark ruler --model /path/to/Qwen3-8B --j auto \\
        --lengths 8k,16k,32k --n 100 --selector bm25 --out ruler_results/qwen3_8b

    python -m eval.run --benchmark babilong --model /path/to/Llama-3-8B --j auto \\
        --tasks qa1,qa2,qa5 --lengths 0k,1k,2k,4k,8k,16k --out babilong_results/llama3_8b

Any flag the target driver understands (``python -m eval.<benchmark> --help``) may
be appended; they are passed through verbatim.
"""
from __future__ import annotations

import argparse
import importlib
import sys

BENCHMARKS = ["ruler", "babilong", "longbench", "longeval", "locomo"]


def main():
    p = argparse.ArgumentParser(
        description="CoMem eval dispatcher (routes --benchmark to eval.<benchmark>)",
        epilog="All other args are forwarded to the chosen driver "
               "(see: python -m eval.<benchmark> --help).")
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS,
                   help="which benchmark driver to run")
    args, rest = p.parse_known_args()

    mod = importlib.import_module(f"eval.{args.benchmark}")
    # Hand the driver a clean argv (its own argparse reads sys.argv).
    sys.argv = [f"eval.{args.benchmark}"] + rest
    mod.main()


if __name__ == "__main__":
    main()
