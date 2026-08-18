# ICASSP 2027 manuscript

Primary artifact: `main.tex`.

Generated evidence:

- `figures/figure1_protocol_maps.{pdf,svg,png}`
- `figures/figure2_performance_ped.{pdf,svg,png}`
- `tables/table1_main.tex`
- `statistical_report.json`

Rebuild figures, table, and clustered statistics:

```bash
UV_CACHE_DIR=/tmp/eeg-forgetting-uv-cache uv run python \
  paper/scripts/build_paper_artifacts.py
```

The draft was compiled with the official Tectonic 0.16.9 binary and its bundled
`IEEEtran.cls`. `main.pdf` is four US-Letter pages: pages 1--3 contain technical
content and page 4 contains references only. No overfull boxes, undefined
citations, or undefined references were reported. Recompile with the official
ICASSP 2027 author kit when it becomes available and do not change margins to
force the page limit. Replace the author/affiliation placeholders before
submission.
