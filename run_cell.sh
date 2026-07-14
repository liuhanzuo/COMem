#!/usr/bin/env bash
# Run one CoMem eval cell (a single benchmark x model, all other knobs default
# to sensible values). Thin wrapper over `python -m eval.run`.
#
# Usage:
#   ./run_cell.sh <benchmark> <model> [extra driver args...]
#
# Examples:
#   ./run_cell.sh ruler    /path/to/Qwen3-8B  --lengths 8k,16k,32k --n 100
#   ./run_cell.sh babilong /path/to/Llama-3-8B --tasks qa1,qa2,qa5 --lengths 0k,4k,16k
#   ./run_cell.sh longbench /path/to/Qwen3-8B --tasks narrativeqa,hotpotqa
#   ./run_cell.sh longeval /path/to/Qwen3-8B  --lengths 4k,8k,16k,32k,64k,128k
#   ./run_cell.sh locomo   /path/to/Qwen3-8B  --out locomo_results/qwen3_8b
#
# Env overrides:
#   PYTHON_BIN  python interpreter (default: python)
#   J           split depth (default: auto -> comem.model_registry)
#   BASELINE    none|dense|kvdirect|hcache|streamingllm (default: none = CoMem)
#   SELECTOR    bm25|reader_attn|recency|oracle|... (default: driver default)
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <benchmark> <model> [extra driver args...]" >&2
    echo "  benchmark: ruler | babilong | longbench | longeval | locomo" >&2
    exit 1
fi

BENCH="$1"; shift
MODEL="$1"; shift

PYBIN="${PYTHON_BIN:-python}"
J_VAL="${J:-auto}"

cd "$(dirname "$0")"

CMD=("$PYBIN" -m eval.run --benchmark "$BENCH" --model "$MODEL" --j "$J_VAL")
[ -n "${BASELINE:-}" ] && CMD+=(--baseline "$BASELINE")
[ -n "${SELECTOR:-}" ] && CMD+=(--selector "$SELECTOR")

echo "[run_cell] ${CMD[*]} $*" >&2
exec "${CMD[@]}" "$@"
