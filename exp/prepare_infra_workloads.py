"""Prepare source/query-separated timing inputs from a local PG-19 JSONL corpus."""
from pathlib import Path
import argparse
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'comem_infra_20260912'))
from bench_local import prepare_workloads
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', type=Path, required=True,
                        help='PG-19 training-subset JSONL, one text field per book')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.model_max_length = 10**9
    for length in (8192, 16384, 32768, 131072):
        dense = HERE / 'comem_dense5090_20260913/workloads' / f'{length}.json'
        dense.parent.mkdir(parents=True, exist_ok=True)
        if dense.exists():
            raise FileExistsError(f'Keep the selected timing inputs fixed: {dense}')
        rows = prepare_workloads(tokenizer, args.source, dense, length)
        if length in (32768, 131072):
            e2e = HERE / 'comem_e2e_20260913/workloads' / dense.name
            e2e.parent.mkdir(parents=True, exist_ok=True)
            if e2e.exists():
                raise FileExistsError(e2e)
            shutil.copy2(dense, e2e)
        if length == 131072:
            target = HERE / 'comem_b300_recheck_20260912/workloads.json'
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(dense, target)
        print(length, [row['book_index'] for row in rows])


if __name__ == '__main__':
    main()
