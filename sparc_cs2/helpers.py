"""Series-balanced training weights and fixed training-internal split."""
import numpy as np
from collections import defaultdict
import math

def round_weights(counts, series):
    nseries = len(set(series))
    return np.concatenate([np.full(n, 1. / (n*nseries*np.count_nonzero(series == sid)))
                           for n, sid in zip(counts, series)])


def internal_split(train_rounds):
    """Return (fit rounds, held-out rounds), preserving the caller's round order."""
    by_event = defaultdict(set)
    meta = {}
    for r in train_rounds:
        if r["role"] != "train":
            raise ValueError("Internal assessment uses original training series only.")
        match, event, date = str(r["match_id"]), str(r["event"]), str(r["date"])
        if meta.setdefault(match, (event, date)) != (event, date):
            raise ValueError(f"Inconsistent match metadata for {match}")
        by_event[event].add((date, match))
    held_ids = set()
    for event, series in by_event.items():
        count = math.ceil(len(series) / 5)
        if count >= len(series):
            raise ValueError(f"No earlier training series for {event}")
        held_ids.update(match for date, match in sorted(series)[-count:])
    fit = [r for r in train_rounds if str(r["match_id"]) not in held_ids]
    held = [r for r in train_rounds if str(r["match_id"]) in held_ids]
    return fit, held

