"""CPU-only loader, official prompt, and official accuracy for the prepared BABILong grid."""
from __future__ import annotations

import argparse
import importlib.util
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "babilong"
VENDOR_DIR = ROOT / "vendor" / "babilong" / "babilong"
TASKS = ("qa1", "qa2", "qa3", "qa5")
LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k")
MAX_NEW_TOKENS = 20  # CoMem eval/babilong.py default, not a universal benchmark rule.


@lru_cache(maxsize=2)
def _module(name):
    spec = importlib.util.spec_from_file_location(f"official_babilong_{name}", VENDOR_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task(task: str, length: str, data_dir: str | Path | None = None) -> list[dict]:
    if task not in TASKS or length not in LENGTHS:
        raise ValueError(f"Prepared grid is {TASKS} x {LENGTHS}")
    path = (Path(data_dir) if data_dir is not None else DATA_DIR) / task / f"{length}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected official JSON list in {path}")
    return rows


def format_prompt(row: dict, task: str) -> str:
    """Exact CoMem default official template: instructions/examples/post-prompt, no chat."""
    prompts = _module("prompts")
    cfg = prompts.DEFAULT_PROMPTS[task]
    return prompts.get_formatted_input(row["input"], row["question"], cfg["examples"],
                                       cfg["instruction"], cfg["post_prompt"],
                                       template=prompts.DEFAULT_TEMPLATE)


def score_prediction(pred: str, row: dict, task: str) -> float:
    metrics = _module("metrics")
    return float(metrics.compare_answers(row["target"], pred, row["question"], metrics.TASK_LABELS[task]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = []
    for task in TASKS:
        for length in LENGTHS:
            rows = load_task(task, length, args.data_dir)
            errors = []
            if len(rows) != 100:
                errors.append(f"expected 100 rows, got {len(rows)}")
            for index, row in enumerate(rows):
                if score_prediction(row["target"], row, task) != 1:
                    errors.append(f"row {index}: own answer failed")
                if score_prediction("", row, task) != 0:
                    errors.append(f"row {index}: blank output passed")
                if row["question"].strip() not in format_prompt(row, task):
                    errors.append(f"row {index}: question missing from prompt")
            report.append({"task": task, "length": length, "count": len(rows),
                           "fields": list(rows[0]), "errors": errors})
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if any(row["errors"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
