# Numerical inputs for frozen model figures

These are real values from the completed selected-model experiment, retained
without rerunning training. `python scripts/replot.py` redraws paired effects,
time/mark decomposition, conditional destination gain and deadline mass transfer.
The public generator uses the same numerical inputs in a standalone layout.
It does not reproduce the original paper's third-party radar image.

`main_results.csv` has the original means for all three partitions. `../scores.csv`
adds independent-seed sample SDs for development and Rio from the frozen raw
summary. TREE is a single fitted deterministic model shared across seed labels.
Paired effects are relative joint-NLL reduction in percent, with original
10,000-draw match-bootstrap intervals. Time/mark gains are absolute NLL units.
Destination gains condition on a pure entry/plant family and its realized future
time bin; seed probabilities are averaged before log scoring, then native events
are deduplicated within lead bands. Deadline changes average cutoffs within
represented rounds and matches; the NPZ contains the two cohorts' 4×21 grids.
These diagnostics have different conditioning and must not be treated as the
same unconditional forecast metric. Original numerical source hashes are in
`../provenance.json`.
