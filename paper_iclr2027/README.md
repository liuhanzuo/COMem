# MidCache — ICLR 2027 manuscript

**Cache the Encoder Within: Reusing Intermediate Representations across LLM Queries**

The current PDF is `main.pdf`. Editable LaTeX source for the current manuscript, using the ICLR 2027 anonymous submission style.

The central insight is the semantic-encoder role of early transformer layers: intermediate residuals retain the output of document encoding, which can be cached and reused by the upper reader across queries. The abstract, introduction, motivation, method, experiments, figures, and conclusion follow this encoder--cache--reader argument. Probe accessibility motivates the interface; matched continuation and quality/cost experiments test whether it is usable. The manuscript does not assert a universal layer at which all semantic processing ends.

Related Work explains each principal baseline and nearby method through its cached object and read operation, including InfLLM and prior intermediate-activation reuse. The experiments test answer quality, saved encoder work and amortization, cache-boundary fidelity, and additional evidence under calibrated latency budgets. Overlap-Write is a context diagnostic defined in Appendix B.2; its negative confirmation results remain in the main text.

- Entry point: `main.tex`.
- The lower-third split gives j=12 on the 36-layer Qwen3-8B. Section 5.3 explains the quality/cost rationale in Table 3; Appendix A.1 gives the default rule and Appendix B.3 reports the depth sweep.
- Main text: abstract through conclusion, 9 pages in the delivered build.
- Statements begin on page 10, references on page 10, and appendices on page 13.
- The complete PDF is 36 pages.
- Table 1 compares seven configurations over five benchmarks. MidCache and MidCache (without LoRA) both use j=12, specified in the caption. The benchmark detail tables use the same no-adapter operating point.
- Table 2 covers the full-source RTX5090 comparison under a 28GB budget, the B300 full-context comparison, and the B200 pipeline measurement.
- Table 3 compares j=6/9/12/18 on RULER, LoCoMo, Read latency, and Read speedup, with the principal j=12 row shaded.
- Figure 1 is the Introduction teaser; Figure 2 explains the architecture; Figure 3 reports source-length scaling, including E2E.
- Appendix B contains matched depth and context controls and the fixed-w32 LongEval/Qasper evaluation, including paired RTX5090 quality and Write-inclusive costs on a prespecified 20-example subset. Different sample sets and execution conditions are specified in the protocols.
- Appendix D.5 reports the matched chunk-KV quality and RTX5090 cost controls with both adapter settings: 3,000 quality predictions and 1,080 formal cost measurements. `kv_cost_results.json` contains all per-process cost medians and ranges used for the appendix table. The synchronous reference is distinguished from native CacheBlend performance.

## Build

Run in this directory:

```powershell
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

Keep the bundled ICLR style and bibliography files beside `main.tex`. The anonymous setting is active.

The abstract and Reproducibility statement link to the [anonymous code repository](https://anonymous.4open.science/r/COMem-Anonymous/). The manuscript includes an AI use statement.

Literature citations use clickable blue square-bracket numbers, such as [1] and [2], with the bibliography numbered in order of first citation. The author-requested numeric format uses bundled `unsrtnat.bst` and a preamble override; the ICLR layout style remains unmodified. Figure/table and section references also link to their PDF destinations. Bibliography URLs are clickable where provided.

## Figures

Vector PDF, editable SVG, and PNG versions are included. `figures/draw_architecture.py` redraws the architecture; `figures/draw_teaser.py` uses `figures/teaser_data.json` to redraw the teaser and scaling plot. OOM markers are categorical outcomes without extrapolated latency values.

The source archive contains the current manuscript and figure sources. Internal author notes, review reports, and raw experiment logs are not part of this directory.


The architecture is on page 5, the teaser on page 2, and the complete source-scaling figure on page 26. Table 1 compares seven methods over five benchmarks. Both MidCache variants use j=12, stated in the caption; their row labels are MidCache and MidCache (without LoRA). Benchmark detail tables use the same j=12 no-adapter results. Table 2 compares full-source Dense with MidCache on RTX 5090 under the 28 GB allocator cap, alongside the B300 full-context comparison and B200 pipeline. Same-pack local replay remains a separate appendix diagnostic.
