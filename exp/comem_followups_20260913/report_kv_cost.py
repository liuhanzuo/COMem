"""Verify completed CPU records and generate the appendix cost table."""
from pathlib import Path
from statistics import median
import json

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1] / 'paper_iclr2027'
CELLS = ('longeval_8k', 'longeval_16k', 'longeval_32k', 'qasper')
FIELDS = ('prepare_ms', 'write_ms', 'selection_ms', 'fetch_ms', 'ttft_ms',
          'online_ms', 'e2e_ms', 'persistent_bytes', 'peak_allocated_bytes',
          'peak_reserved_bytes')
records = []
for process in (1, 2, 3):
    folder = HERE / 'kv_cost' / f'process_{process:02d}'
    complete = json.loads((folder / 'complete.json').read_text())
    checks = json.loads((folder / 'correctness.json').read_text())
    assert complete['complete'] and complete['records'] == 480
    assert checks['r1_stock_max_abs_logits'] == 0
    assert checks['r1_stock_decode_3_steps_equal']
    assert checks['zero_query_placeholders_logits_equal']
    rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 480 and all(r['process'] == process for r in rows)
    assert sum(not r['warmup'] for r in rows) == 360
    records.extend(rows)
assert len({(r['process'], r['id'], r['method'], r['adapter_on'], r['rep']) for r in records}) == 1440
for r in records:
    assert r['status'] == 'ok' and len(r['generated_ids']) == 128
    assert max(r['peak_allocated_bytes'], r['peak_reserved_bytes']) <= 28_000_000_000
    assert abs(r['prepare_ms'] + r['online_ms'] - r['e2e_ms']) < 1e-5
    assert 0 < r['ttft_ms'] <= r['online_ms']
    expected = {'replay': 8, 'comem': 8192, 'chunkkv': 147456}[r['method']]
    assert r['persistent_bytes'] == r['source_tokens'] * expected
formal = [r for r in records if not r['warmup']]
saved = json.loads((HERE / 'kv_cost_summary.json').read_text())
reported = {}
for cell in CELLS:
    reported[cell] = {}
    for on in (False, True):
        for method in ('replay', 'chunkkv', 'comem'):
            arm = f'{method}:adapter_{"on" if on else "off"}'
            rows = [r for r in formal if r['cell'] == cell and r['method'] == method and r['adapter_on'] == on]
            assert len(rows) == 45 and len({r['id'] for r in rows}) == 5
            for sid in {r['id'] for r in rows}:
                repeated = [r for r in rows if r['id'] == sid]
                assert len(repeated) == 9
                assert all(r['generated_ids'] == repeated[0]['generated_ids'] and r['selected'] == repeated[0]['selected'] for r in repeated)
            procs = {str(i): {k: median(r[k] for r in rows if r['process'] == i) for k in FIELDS} for i in (1, 2, 3)}
            for i, fields in procs.items():
                for k, value in fields.items():
                    assert abs(value - saved['cells'][cell][arm]['process_medians'][i][k]) < 1e-8
            reported[cell][arm] = {
                'n': 45, 'process_medians': procs,
                'median_of_process_medians': {k: median(v[k] for v in procs.values()) for k in FIELDS},
                'process_median_range': {k: [min(v[k] for v in procs.values()), max(v[k] for v in procs.values())] for k in FIELDS},
                'max_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
                'max_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows),
            }
report = {'complete': True, 'formal_records': 1080, 'warmup_records': 360,
          'description': 'Latency: median of three process medians; each process has five examples and three repetitions. GPU: maximum over formal records. Process ranges are descriptive, not confidence intervals.',
          'cells': reported}
(HERE / 'kv_cost_paper_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(PAPER / 'kv_cost_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
lines = [r'\begin{table}[p]', r'\centering', r'\footnotesize',
         r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrrrrrr@{}}',
         r'\toprule',
         r'Method & Adapter & Prep. & TTFT & Online$_{128}$ & E2E$_{128}$ & Store & GPU \\',
         r' & & (ms) & (ms) & (s) & (s) & (MiB) & (GB) \\', r'\midrule']
titles = dict(zip(CELLS, ('LongEval 8k', 'LongEval 16k', 'LongEval 32k', 'Qasper')))
for cell in CELLS:
    lines.append(r'\multicolumn{8}{@{}l}{\textit{' + titles[cell] + r' ($n=5$)}} \\')
    for on in (False, True):
        for method in ('replay', 'chunkkv', 'comem'):
            arm = f'{method}:adapter_{"on" if on else "off"}'
            d = reported[cell][arm]; m = d['median_of_process_medians']
            name = {'replay': 'Replay', 'chunkkv': 'Chunk-KV', 'comem': 'CoMem' if on else 'CoMem (without LoRA)'}[method]
            lines.append(f"{name} & {'on' if on else 'off'} & {m['prepare_ms']:.1f} & {m['ttft_ms']:.1f} & {m['online_ms']/1000:.3f} & {m['e2e_ms']/1000:.3f} & {m['persistent_bytes']/2**20:.2f} & {d['max_allocated_bytes']/1e9:.2f}" + r' \\')
        if not on: lines.append(r'\addlinespace[2pt]')
    if cell != CELLS[-1]: lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{Matched evidence costs on RTX 5090: 1,080 formal measurements over 20 prespecified examples, six method--adapter settings, three fresh processes, and three repetitions. Each cell reports the median of three process medians; each process median pools five examples and three repetitions. Preparation constructs a fresh whole-document pinned-CPU store. TTFT and Online start after preparation, include selection and fetch, and end at the first and 128th generated token, respectively. E2E is measured directly from preparation through token 128. Store is the median persistent payload, separate from peak GPU allocation including weights; all attempts fit the 28 GB allocator cap. CoMem uses $j=12,w=0$; replay processes the same selected evidence, not the full source. Chunk-KV is a synchronous CacheBlend-style reference. The 500-example quality evaluation is reported separately in Table~\ref{tab:chunkkv}.}',
          r'\label{tab:chunkkv-cost}', r'\end{table}', '']
(PAPER / 'sections/tab_chunkkv_cost.tex').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'verified_formal_records': 1080, 'verified_warmups': 360, 'table': str(PAPER / 'sections/tab_chunkkv_cost.tex')}))
