# SPARC-CS2

**Spatial Attribution and Rule-Guided Next-Event Forecasting in Counter-Strike 2**

This repository provides the SPARC-CS2 model, complete training code, frozen
pretrained artifacts, four neural baselines, filtered derived match data,
and recorded result summaries with plotting data.

**Repository:** https://github.com/qingfeng123456-cyber/SPARC-CS2

**Authors:** Weijun Xia and Weihua Zhao. See
[author contributions](docs/AUTHOR_CONTRIBUTIONS.md) and `CITATION.cff` for
attribution. The accompanying manuscript is unpublished.

SPARC-CS2 predicts the next first event from a defender team's available
observation prefix. Its output is a normalized law over **20 one-second bins ×
511 nonempty simultaneous event sets, plus a no-event tail**. One deployed seed
uses one 55,577-parameter structured neural decoder, risk-set boosting, three
spatial attribution classifiers, and public-state/deadline projections. The
complete fitted hybrid has **985 boosting trees**, in addition to the decoder.
The three provided neural seeds are independent runs, rather than an ensemble
that must be deployed together.

## Quickstart

Use Python 3.11 in a new environment, then run these commands from this
directory. The checkout layout is intentional: data and weights are read
relative to the repository, with optional explicit overrides.

```bash
python -m pip install -r requirements.txt
python main.py --help
python -m sparc_cs2 data-stats
python -m unittest discover -s tests -v
python main.py evaluate --role validation --output outputs/smoke --seeds 2026 --max-rounds 1
```

The last command performs frozen inference on one development round. It writes
a **partial** result, suitable for checking the environment. Choose a fresh
output directory for every run. No command above fits a model.

中文快速开始：安装 Python 3.11 和 `requirements.txt`，在本文件所在目录运行
`python main.py --help`。先运行 `data-stats` 核对 30/10/11 场划分，再运行上述
单回合 `evaluate` 命令检查推理。完整模型已在 `weights/main/`，无需训练即可使用。
`train` 会重新训练完整模型并写入新目录；它不会覆盖附带权重。所有路径相对本目录，
不依赖原作者电脑。数据是派生数值记录，非原始 DEMO，许可说明见
[DATA_AND_THIRD_PARTY.md](DATA_AND_THIRD_PARTY.md)。

## Reproduce inference and training

```bash
# All frozen references and the full model, all three independent seeds.
python main.py evaluate --role validation --output outputs/dev --variants NEURAL TREE BASE_MIX SITE RULES FULL SHUFFLE_RULES
python main.py evaluate --role test --output outputs/rio --variants NEURAL TREE BASE_MIX SITE RULES FULL SHUFFLE_RULES

# Complete selected model training, with the original fixed optimization settings.
python main.py train --split main --device auto --output outputs/trained/main
python main.py train --split internal --device auto --output outputs/trained/internal

# Score independently trained artifacts, without changing the provided frozen model.
python main.py evaluate --role validation --weights outputs/trained/main --output outputs/retrained-dev
python main.py evaluate --role internal --weights outputs/trained/internal --output outputs/retrained-internal
```

Training uses exactly ten neural epochs, AdamW, the original weighted loss,
batch size 512 and cosine learning-rate schedule. The risk hazard uses 105
boosting iterations; the conditional type law uses 80. SITE uses three binary
80-iteration classifiers. Full training includes the matched shuffled spatial
control and training-fitted deadline offsets. The implementation has no
teacher, external checkpoint, calibration cache, or prior-run dependency.
The two 50:50 mixtures and 2% escape mass are fixed. Training does not tune on
the test cohort. `--seeds 2026` can select one independent run.
Internal held-out evaluation requires the corresponding 22-match-fitted
artifacts; scoring rejects overlap with the model's recorded training matches.

`NEURAL`, `TREE`, `BASE_MIX`, `SITE`, `RULES`, `FULL`, and `SHUFFLE_RULES` are
frozen-component comparisons. The deletion variants retain the same fitted
base predictors; they are not described as end-to-end retraining ablations.

## Data and scoring protocol

| Partition | Matches | Eligible rounds | Scored cutoffs |
|---|---:|---:|---:|
| Train | 30 | 650 | 56,704 |
| Development | 10 | 210 | 18,210 |
| Rio test | 11 | 215 | 19,655 |
| Internal fit, within original train | 22 | 479 | 41,521 |
| Internal held out, within original train | 8 | 171 | 15,183 |

The original match roles are fixed in `data/index.json`. The internal split
holds out the latest `ceil(n/5)` matches within each event, ordered by date and
public match ID. The eligibility test excludes whole rounds with degenerate
game clocks or invalid native tick/event order. Scored cutoffs satisfy
`cutoff_valid & (target_bin >= 0)`.

Features use the CT team state and enemy last-public-contact memory, along
with causal past differences. The compact neural representation has 272
dimensions. Baselines use the same raw observation boundary, fixed target,
cutoffs, match roles and weighting, with a 482-dimensional representation.
Labels and future-event arrays remain separate from model inputs.

The primary metric is joint negative log likelihood: mean over cutoffs within
each round, rounds within each match, matches equally, then independent seeds
equally. Time NLL and A/B ingress Brier scores at ten seconds use the same
aggregation. Definitions, mark bits and clock conventions are detailed in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

`data/matches/` contains 51 filtered numeric NPZ archives (about 35 MB total),
with public observation arrays and labels required for training and scoring.
It excludes hidden enemy trajectories, separate truth/visibility arrays and
native DEMOs. Public match IDs identify matches; roster account identifiers
are absent. Original raw recordings are listed by URL, byte count and SHA-256
in `data/raw_demo_manifest.csv`. See [docs/DATA.md](docs/DATA.md) for acquisition
and parsing. The code license does not automatically license replay data.

## Baselines and figures

The four baselines are MLP, causal TCN, GRU and causal Transformer, implemented
in [baselines/](baselines/README.md). Their large checkpoints and generated
482-dimensional cache are excluded; the cache is generated from bundled data.
`baselines/README.md` covers fixed splits, selection and full report commands.

```bash
python baselines/data.py
python baselines/main.py --smoke --device cpu --output outputs/baseline-smoke
python baselines/replot.py --output outputs/baseline-figures
python scripts/replot.py --output outputs/core-figures
```

`results/core/` contains real frozen development/Rio summaries, per-match
scores, and numerical figure inputs. `results/baselines/` contains real
baseline summaries. The figure entry points redraw saved comparisons and
mechanisms without training. No Valve radar image is bundled.

## Layout

```text
main.py                  Public command entry point
sparc_cs2/               Core features, NN, risk law, SITE, rules, training and inference
weights/main/            Three compact NN checkpoints, fitted trees and clock offsets
data/                    Fixed index, filtered match records, schema and provenance
source/portable/cs2ai/    Original native-DEMO parsing and label-definition source
baselines/               Four baseline architectures and training/evaluation tools
results/                 Real summaries and compact plotting inputs
scripts/                 Raw parsing and saved-result figure entry points
tests/                   Small protocol/probability/label-isolation checks
docs/                    Protocol, data, frozen configuration and source mapping
```

## Verification and environment

The local release check covers imports and command help, exact data counts,
one-round inference, all seven distributions' normalization, label-independent
forward prediction, comparison with the original frozen prediction factors,
saved-data figure regeneration, and a four-cutoff neural optimization check
using the unchanged ten-epoch schedule. See `docs/release_verification.json` for
the measured differences and tested commands. Full training and full-cohort
evaluation were not rerun while packaging this release; reported cohort scores
are copied from the completed frozen experiments.

Pretrained scikit-learn/joblib artifacts were verified with scikit-learn 1.9.0
and joblib 1.5.3, pinned in `requirements.txt`. Other scikit-learn versions may
not deserialize fitted trees compatibly. The loader handles the historical
`__main__.RiskLaw` class reference. PyTorch checkpoints contain state dictionaries
and are loaded with `weights_only=True`. GPU kernels or different library
versions can change newly trained results; supplied artifacts preserve the
frozen experiment. Exact local versions are recorded in the verification file.

Project-authored code and documentation are MIT licensed. See
[DATA_AND_THIRD_PARTY.md](DATA_AND_THIRD_PARTY.md) for data, model artifacts and
third-party content. Citation metadata is in `CITATION.cff`; publication URL,
DOI and personal author metadata are intentionally not invented.
