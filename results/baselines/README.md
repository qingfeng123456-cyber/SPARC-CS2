# Real architecture-comparison evidence

These files are the completed development/Rio architecture-baseline study results. The sources are the original returned `summary.json`, `per_series.csv`, `comparison.csv`, `selected_configurations.json`, and `run_protocol.json`; they are preserved without changing numerical evidence. `scores.csv` is a small derived table of every mean and sample standard deviation over fitted seed scores. The source protocol hashes refer to the historical experiment code, which predates public-release path cleanup.

`FULL` is the fixed SPARC-CS2 model; `NEURAL`, `TREE`, and `BASE_MIX` are frozen component references. Seeds 2026/2027/2028 identify independent neural runs, while TREE is one shared deterministic model. The summary retains per-series values and paired bootstrap intervals, so `python baselines/replot.py` regenerates all three comparison figures without full prediction archives or checkpoints.

These are existing research measurements. The packaging verification does not claim that complete training was rerun, or that this release is already published online.
