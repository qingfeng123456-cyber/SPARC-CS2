# Derived data, source acquisition and parsing

`data/index.json` fixes the 51 included matches and roles. `raw_demo_manifest.csv`
lists event, teams, date, original DEMO metadata, verified public download URL,
match page and source SHA-256. Local acquisition paths were removed. No download
occurs implicitly. Obtain recordings from their listed source under applicable
terms, preserving the source file bytes. About 25 GB of raw DEMOs are deliberately
excluded from this repository.

Each bundled `data/matches/{match_id}.npz` stores `metadata` plus round-prefixed
arrays such as `r0_x`. Retained fields are listed in `data/manifest.json`.
`x`, `pub`, `time`, `tick` and `cutoff_valid` define the causal observation and
cutoff boundary. The remaining arrays provide labels, merged event clocks or
diagnostics needed by training, scoring and conditional figures. Future labels
are intentionally included so the released code can train and evaluate; they
are kept outside forward-model feature arguments.

Original hidden `truth`, `alive`, `ct_xyz`, `seen` and `region` arrays were
removed. Neither account IDs nor raw roster names are present. Match IDs and
team/event names remain as public cohort provenance. File hashes in
`manifest.json` distinguish the original numeric archives from their filtered
public derivatives. NPZ recompression changes file hashes while preserving
retained array values and dtypes.

The original parser and full executable event/target definitions are retained
in `source/portable/cs2ai/data.py`. Its fixed geometric boxes are in the adjacent
`geometry.json`; these are operational research annotations, not official
navigation regions. The public `scripts/preprocess.py` wrapper calls the
original single-match parser and retains the same filtered schema. It validates
the native source size and SHA-256 before parsing.

```bash
python -m pip install -r requirements-parser.txt
python scripts/preprocess.py --help
python scripts/preprocess.py --demos /path/to/acquired/demos --output outputs/reparsed
```

Place each raw file at `DEMO_ROOT/{file_name from manifest}` or at
`DEMO_ROOT/{basename}`. `--matches 2393003` can select a single match. New parser
outputs must use an empty output directory and never replace the supplied data.
The full raw parsing procedure was not rerun during public packaging. The
provided derived archives were checked against their fixed scoring counts and
the frozen model predictions. The original parsing environment used
demoparser2 0.42.2 and pandas 3.1.1.

The included numeric data and model artifacts do not automatically inherit the
MIT code license. See `DATA_AND_THIRD_PARTY.md` for scope and upstream rights.
