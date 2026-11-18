The MIT license covers project-authored code and documentation. It does not
grant rights to Valve's game, maps, radar imagery, match broadcasts, player
likenesses, third-party datasets, or original tournament DEMO recordings.

`data/matches/` contains filtered **derived numeric research records**, not
native DEMO files. Their source matches and acquisition URLs are identified in
`data/raw_demo_manifest.csv`; source and derived hashes are in
`data/manifest.json`. These records are included for reproducibility and do not
automatically inherit the code license. Verify applicable upstream terms and
permissions before redistribution or commercial use. No original DEMO files,
Valve radar images, roster account identifiers, or player names are included.
Public match identifiers and team/event names identify the research cohort.

`weights/` contains project-trained research artifacts derived from that cohort;
they likewise do not confer any rights to underlying game or replay content.
No separate blanket data or model license is asserted here.

The source parser uses the separately distributed `demoparser2` dependency.
NumPy, PyTorch, SciPy, scikit-learn, joblib, threadpoolctl and Matplotlib retain
their respective licenses; they are installed as dependencies, not vendored.
