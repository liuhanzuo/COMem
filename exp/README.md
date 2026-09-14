# ICLR 2027 supplementary experiments

These scripts implement the accuracy, context, KV-cache, distillation, and
infrastructure controls in [the current manuscript](../paper_iclr2027/main.pdf).
They retain the measured masks, seeds, adaptation conditions, timing boundaries,
and aggregation rules. Repository imports and local input paths have been made
relative/configurable. No new GPU results are implied by publishing these scripts.

| Directory | Purpose | Manuscript |
|---|---|---|
| `comem_frozen_j12_20260912` | Complete no-adapter depth-12 quality evaluation | Table 1, Appendix C.1 |
| `comem_overlap_20260913` | Fixed left-context overlap: LongEval/Qasper quality and Write-inclusive cost | Appendix B.2 |
| `comem_followups_20260913` | Both adapter settings for chunk KV; full-vocabulary distillation diagnostics; paired clean LongBench support | Appendices B.4, D.5, E.3 |
| `comem_infra_20260912` | Same-evidence selected-pack timing helpers | Appendix D.1 |
| `comem_dense5090_20260913` | Full-source Dense versus CoMem under a 28 GB allocator cap | Table 2(a) |
| `comem_e2e_20260913` | Source preparation, TTFT, and fixed-length generation | Appendix D.1 |
| `comem_b300_recheck_20260912` | Same-adapter full-context/store-ready prefill | Table 2(b) |

Small aggregate measurements are in `paper_iclr2027/*.json` and
`paper_iclr2027/figures/teaser_data.json`. Raw predictions, source texts, token
arrays, model weights, adapters, private judge responses, and cluster logs are
not distributed here. Aggregators require the raw records produced by the
corresponding runners; the paper aggregates alone cannot reconstruct them.

## Environments and inputs

Install `pip install -r exp/requirements.txt`; sample preparation uses `requests`.
Figure rendering uses `matplotlib`. The measured local environment uses Windows,
PyTorch 2.7.1+cu128, Transformers 5.16.1, bf16 model weights, fp32 unmerged LoRA,
and SDPA. Accuracy-only B300 workers use the separately documented Slurm
environment. Some runners use newer Transformers APIs than the base package.
See Appendix D for each hardware/kernel configuration before comparing timings.

Supply the same Qwen3-8B checkpoint and principal suffix adapter through
`--model` and `--adapter`. Preparation scripts use environment variable
`COMEM_MODEL`; `prepare_diagnostics.py` also requires `COMEM_TRAIN_MANIFEST`, a
JSON object with `objects[*].name` identifying the PG-19 training books.
The distillation preparation downloads the public PG-19 validation split and
checks training/validation book disjointness. No benchmark data is used for
adapter training.

Expected local benchmark layout:

```text
exp/comem_frozen_j12_20260912/
  data/longbench/{narrativeqa,qasper,hotpotqa,2wikimqa,multifieldqa_en,musique}.jsonl
  data/babilong/{qa1,qa2,qa5}/{0k,1k,2k,4k,8k,16k,32k}.json
  data/pg19_essay.txt
  vendor/longbench/metrics.py
  vendor/babilong/babilong/{metrics,prompts}.py
```

Use the official test files without reordering rows. The no-adapter aggregator
expects the official LongBench metric from
<https://github.com/THUDM/LongBench/blob/main/metrics.py> and the official
BABILong `metrics.py` and `prompts.py`. Supply these dependencies from their
upstream distributions; their source providers' terms apply. `prepare_babilong.py`
validates the prepared grid including qa3, so its validation command additionally
requires the qa3 files; the main-table runner evaluates qa1/qa2/qa5 only. Exact RULER reproduction requires the same saved
PG-19 haystack and `PYTHONHASHSEED=0`.

## Quality and context controls

Generate the 300 LongEval and 200 Qasper confirmation examples:

```bash
export COMEM_MODEL=/path/to/Qwen3-8B
python exp/comem_overlap_20260913/prepare_samples.py
```

The preparer fixes selection before evaluating methods and does not overwrite
existing confirmation samples. Copy its `samples.jsonl.gz` to
`exp/comem_followups_20260913/data/kv_samples.jsonl.gz` for the same-evidence KV
controls. Create the destination `data` directory first.

The quality runners require an actual single-GPU Slurm allocation, checked via
`SLURM_JOB_ID` and CUDA device count. Run disjoint shards 0 through 3:

```bash
python exp/comem_overlap_20260913/run_quality.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --shard 0 --nshards 4
python exp/comem_followups_20260913/run_kv_quality.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --shard 0 --nshards 4
python exp/comem_frozen_j12_20260912/run_accuracy.py \
  --model /path/to/Qwen3-8B --shard 0 --nshards 4
```

`--smoke` uses separate output directories. Each full runner also executes its
own continuation/scoring checks. Output directories must be new; do not combine
partial or differently configured runs. Run `aggregate.py` in the overlap and
no-adapter directories, and `aggregate_kv.py` for the KV controls, after every
required shard is complete. These scripts verify sample coverage and rescore.

## Held-out distillation and clean-subset analysis

```bash
export COMEM_TRAIN_MANIFEST=/path/to/pg19_train_64_manifest.json
python exp/comem_followups_20260913/prepare_diagnostics.py
python exp/comem_followups_20260913/run_distillation.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter
python exp/comem_followups_20260913/aggregate_distillation.py
python exp/comem_followups_20260913/prepare_clean.py
python exp/comem_followups_20260913/run_clean.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --shard 0 --nshards 4
python exp/comem_followups_20260913/aggregate_clean.py
```

Run all four clean shards. The clean-subset scorer also needs the completed
no-adapter LongBench predictions in the no-adapter experiment's `results/`
directory. The optional clean smoke comparison additionally requires
`data/clean_frozen_smoke.json`, containing the matching saved no-adapter sample
selection, prediction, and score. Its absence does not prevent the full run.
`reference/clean_subset_ids.json` records the evaluated membership by dataset ID
and row index; it contains no document texts. It supports the reported filter,
not a new contamination claim about unknown backbone pretraining.

## Full-source and Write-inclusive costs

First prepare source/query-separated timing inputs from the local PG-19
training-subset JSONL (one `text` field per book, fixed book order):

```bash
python exp/prepare_infra_workloads.py --model /path/to/Qwen3-8B \
  --source /path/to/pg19_train_64.jsonl
```

The paper's local timing sources are the first three books long enough for each
source length plus 512 query tokens. Source order determines this selection.
For each command below run fresh processes 1, 2, and 3 **serially**, after a
separate `--process 0 --smoke` run:

```bash
python exp/comem_dense5090_20260913/bench_short.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --process 1
python exp/comem_dense5090_20260913/bench_dense.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --process 1
python exp/comem_e2e_20260913/bench_e2e.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --process 1
python exp/comem_followups_20260913/run_kv_cost.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --process 1
python exp/comem_overlap_20260913/run_cost.py \
  --model /path/to/Qwen3-8B --adapter /path/to/adapter --process 1
```

Local cost runners use `gpu_gate.py`, including Windows `tasklist`, to wait for
an idle card and apply the allocator budget. This is not a multi-GPU scheduler
or a portable cluster lock. The measured memory caps and their decimal/binary
unit handling are retained in the individual scripts. Full-source Dense OOMs
are categorical outcomes and never converted into speedup values.

Run `aggregate_short.py` and `aggregate.py` for Dense, `aggregate.py` for E2E,
and `aggregate_kv.py` followed by `report_kv_cost.py` for KV. The last script
also regenerates the corresponding manuscript table. The overlap aggregator
handles quality and cost outputs together.

The B300 `bench_remote.py` requires a single-GPU Slurm allocation. It compares
full-source prefill with store-ready CoMem and excludes Write/fetch; it is not
the same-evidence depth-only measurement. Supply its prepared workloads and
adapter explicitly. Its aggregate is produced by `summarize.py`.
