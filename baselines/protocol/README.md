# Frozen row alignment

The two NPZ files contain row metadata and labels from the original frozen FULL seed-2026 development/Rio predictions, without probabilities, losses, weights, or raw player identities. Keys are `series_ids`, `round_ids`, `counts`, `bins`, `marks`, `time`, and `cutoff_index`. The three evaluation seeds use the same rows. The public match-series identifiers are study group identifiers, not Steam account identifiers.

`baselines/data.py` checks a reconstructed cache against these arrays exactly. Their checksums and support counts are recorded in `baselines/verification.json`.
