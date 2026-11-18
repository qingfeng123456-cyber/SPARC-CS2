# Matched architecture baselines

This folder contains complete, original PyTorch implementations and training/evaluation code for MLP-Joint, GRU-Joint, TCN-Joint, and Transformer-Joint. Each predicts the same normalized 10,221-class outcome as SPARC-CS2: 20 time bins times 511 nonempty simultaneous mark sets, followed by one no-event tail. They are task-adapted architecture comparisons, not reproductions of unrelated published systems.

## Data and fixed protocol

`data.py` reconstructs the 482-dimensional observed feature union from `../data/matches` at run time: the neural 272 dimensions, 96 SITE dimensions for each site, and 18 extra risk-state dimensions. It reads the whitelist in `READ_KEYS`, with labels kept separately. MLP uses the cutoff summary; the sequence architectures receive every sampled frame from the same round's start through the cutoff. Padding after a cutoff cannot change the prediction.

The fixed split is 30/10/11 series, with an event-stratified chronological 22/8 split inside training for configuration selection. Eligible support is 650/210/215 rounds and 56,704/18,210/19,655 cutoffs. The primary score averages cutoffs inside each round, rounds inside each series, and series equally, then averages the three independently fitted seed scores (2026, 2027, 2028). The deterministic TREE reference is shared, rather than three independent fitted trees.

`protocol/dev_metadata.npz` and `protocol/rio_metadata.npz` hold only row identities and targets extracted from the frozen seed-2026 FULL archives. They verify exact cache alignment without bundling the original prediction archive. These processed targets are available for scoring and must not be used as input features.

## Commands

Run from the repository root after installing `requirements.txt`:

```sh
python baselines/main.py --help
python baselines/data.py
python baselines/verify_data.py
python baselines/main.py --smoke --output outputs/baseline_smoke
```

Building the local feature cache writes about 377 MB for `features.npy`; it is intentionally not part of this repository and must stay ignored. The smoke mode uses small subsets and a single seed, and does not produce formal research results. Full training retains the original width candidates, internal selection, final refits, early stopping, checkpoints, and resuming logic in `config.json`, `main.py`, and `training.py`:

```sh
python baselines/main.py --device auto --output outputs/baselines --references outputs/references
```

Before the full run's final comparison report, generate frozen core references with the repository's core evaluation commands for all three seeds and the NEURAL/TREE/BASE_MIX/FULL variants, using `outputs/dev` and `outputs/rio` as the cohort output folders. Then convert them:

```sh
python baselines/export_references.py --input outputs --output outputs/references
python baselines/report.py --output outputs/baselines --references outputs/references
```

The exporter expects the core evaluation `runs/VARIANT_seedSEED/cutoffs.npz` layout. Report generation verifies all 48 baseline/reference prediction archives, their row identities, class ordering, normalized probabilities, and the shared deterministic TREE prediction before scoring. It retains all comparisons, including intervals spanning zero and unfavorable series.

## Real result evidence and redraw

`../results/baselines` contains the completed study's real aggregate summary, per-series CSV, paired-comparison CSV, selected configurations, and original run protocol. `scores.csv` is derived directly from `summary.json` and includes sample standard deviations over independent fitted seeds. The TREE standard deviation is zero because all seed labels refer to the same fitted predictor. No baseline checkpoints or large per-cutoff prediction archives are bundled.

```sh
python baselines/replot.py
```

This command recreates the joint-loss forest, per-series heatmap, and temporal/mark score plane directly from the bundled real summary, without training or accessing match data. Its outputs go to `outputs/baseline_plots`. The summary is preserved as a historical evidence record; its source hashes refer to the original experiment, not to the public-release path cleanup.

## Checks performed for this release

`verification.json` records successful imports/help, three report tests covering aggregation and rejection of invalid alignment, a redraw of all three real-result figures, and small forward checks for all four architectures. Each architecture produced 10,221 probabilities summing to one, and changing padded frames after the cutoff left the prediction unchanged. One real round from each of training/development/Rio also matched every public core feature block exactly (272 neural + 192 SITE + 18 risk), with identical eligibility, unchanged causal prefixes, and independence from future labels. Complete cache construction and full baseline training were not rerun while packaging this release.

## 中文快速开始

安装仓库根目录依赖后，先运行 `python baselines/main.py --help`。执行 `python baselines/data.py` 会从本包派生 NPZ 数据现场生成约 377 MB 特征缓存，不需要原始 DEMO。执行 `python baselines/main.py --smoke --output outputs/baseline_smoke` 可小规模检查四类基线。若只重画已有真实结果，执行 `python baselines/replot.py`，无需重新训练。正式重训需要先用核心评估入口生成固定参考预测，再按上述 exporter/report 命令完成比较。
