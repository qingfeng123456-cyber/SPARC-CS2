"""Self-contained causal inputs and the frozen cutoff/round/series protocol.

Only ``prepare_data`` reads the original processed match archives. Runtime
datasets read the portable derived cache; no hidden opponent trajectories,
native future events, or labels are part of a model input.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

SCHEMA = "cs2-baseline-causal-prefix-v2-matched"
HORIZON, MARKS, N_CLASSES, DIMENSIONS = 20, 511, 10221, 482
# Frozen, training-defined operational geometry. No runtime dependency on the
# source project; prepare_data verifies these boxes against the source record.
SITE_GEOMETRY = {
    "siteA": [700, 1650, 1900, 3200, -30, 300],
    "siteB": [-2200, -950, 2100, 3350, -70, 250],
    "stageA": [400, 1800, 1200, 2200, -150, 350],
    "stageB": [-2200, -700, 1300, 2450, -150, 350],
}
READ_KEYS = ("x", "pub", "time", "tick", "cutoff_valid", "target_bin",
             "target_mask", "events", "event_seconds")
EXPECTED = {
    "train": {"series": 30, "rounds": 650, "cutoffs": 56704},
    "validation": {"series": 10, "rounds": 210, "cutoffs": 18210},
    "test": {"series": 11, "rounds": 215, "cutoffs": 19655},
    "inner_fit": {"series": 22, "cutoffs": 41521},
    "inner_val": {"series": 8, "cutoffs": 15183},
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def public_features(pub, players):
    """Exact public transform used by memory_trial/memory.py."""
    u, x = np.asarray(pub), np.asarray(players)
    elapsed, planted = 120 * u[:, 0], u[:, 3]
    age = 45 * u[:, 4]
    alive_ct, alive_t = 5 * u[:, 1], 5 * u[:, 2]
    exposed = np.maximum(0, 5 * u[:, 18])
    ct, foe = x[:, :5, :], x[:, 5:, :]
    dxy = ct[:, :, None, :2] - foe[:, None, :, :2]
    dist = np.sqrt(np.square(dxy).sum(-1)) * 4096
    known = (ct[:, :, None, 3] > .5) & (foe[:, None, :, 31] > .5)
    exposure = known & (foe[:, None, :, 4] > .5)
    close, fresh = np.exp(-dist / 850), np.exp(-30 * foe[:, None, :, 5] / 5)
    parts = [
        (1-planted)*np.minimum(elapsed, 100)/35,
        (1-planted)*np.exp(-np.maximum(115-elapsed, 0)/9)*1.6,
        planted*3, planted*age/11, planted*np.exp(-np.maximum(40-age, 0)/6),
        .45*alive_ct, .45*alive_t, .85*(alive_ct == 1), .85*(alive_t == 1),
        1.5*np.exp(-30*u[:, 5]/4), 1.5*np.exp(-30*u[:, 6]/4),
        (exposed > 0)*2., np.minimum(exposed, 3)*.32,
        1.2*(close*exposure).max(axis=(1, 2)), .65*(close*known*fresh).max(axis=(1, 2)),
        .55*u[:, 19], .8*(u[:, 1]-u[:, 15]), .30*u[:, 14],
        .4*u[:, 8], .4*u[:, 9], .45*u[:, 20], .45*u[:, 24],
    ]
    return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)


def features(r):
    """Exact 272-D causal summary from gain_v2/neural.py, at every frame."""
    u, players, time = r["pub"][:, :30], r["x"], r["time"]
    obs = public_features(u, players)
    ct, enemy = players[:, :5, :], players[:, 5:, :]
    alive = ct[:, :, 31]
    coverage = np.concatenate([ct[:, :, i:i+1]*alive[:, :, None] for i in (5, 7, 10, 11, 12)], axis=-1)
    motion = np.linalg.norm(ct[:, :, 13:15], axis=-1)*alive
    spread = np.linalg.norm(ct[:, :, :2] - u[:, None, 8:10], axis=-1)*alive
    close = np.linalg.norm(ct[:, :, None, :2] - enemy[:, None, :, :2], axis=-1)
    valid = (alive[:, :, None] > .5) & (enemy[:, None, :, 31] > .5)
    fresh = np.exp(-enemy[:, :, 5]*30/5)
    pressure = (np.exp(-close*5)*valid*fresh[:, None, :]).max(axis=-1)
    base = np.column_stack((u, obs, coverage.sum(1), coverage.max(1), motion.mean(1), motion.max(1),
                            spread.mean(1), spread.max(1), pressure.mean(1), pressure.max(1)))
    histories = []
    for age in (1.5, 5., 12.):
        earlier = np.maximum(0, np.searchsorted(time, time-age, side="right")-1)
        histories.append(base - base[earlier])
    return np.ascontiguousarray(np.column_stack([base, *histories]), dtype=np.float32)


def site_summary(x, side):
    """Exact 32-D SITE geometry transform from defense/run.py."""
    ct, enemy = x[:, :5], x[:, 5:]
    alive, known = ct[..., 31], enemy[..., 31]
    fresh = np.exp(-enemy[..., 5]*30/6) * known
    cols = []
    for region in ("site", "stage"):
        box = np.asarray(SITE_GEOMETRY[region+side]).reshape(3, 2)
        center = box[:2].mean(1)
        for team, mask, is_ct in ((ct, alive, True), (enemy, known, False)):
            xy = team[..., :2]*4096.
            toward = center - xy
            dist = np.linalg.norm(toward, axis=-1)
            loc = np.exp(-dist/800)*mask
            inside = (((xy >= box[:2, 0]) & (xy <= box[:2, 1])).all(-1))*mask
            cols += [loc.max(1), loc.sum(1), np.sort(loc, axis=1)[:, -2], inside.sum(1)]
            if is_ct:
                unit = toward / np.maximum(dist[..., None], 1.)
                facing = unit[..., 0]*ct[..., 9] + unit[..., 1]*ct[..., 8]
                speed = (unit*ct[..., 13:15]).sum(-1)
                cols += [(loc*np.maximum(0., facing)).sum(1), (loc*speed).sum(1),
                         (loc*ct[..., 5]).sum(1), (loc*ct[..., 10]).sum(1)]
            else:
                cols += [(loc*fresh).max(1), (loc*fresh).sum(1), (loc*enemy[..., 4]).sum(1),
                         (fresh*inside).sum(1)]
    return np.column_stack(cols)


def site_features(r):
    """Two sites × (current 32 + 3/10-second differences), at every frame."""
    x, t = r["x"], r["time"]
    now = np.stack([site_summary(x, side) for side in ("A", "B")], axis=1)
    prior = []
    for delay in (3., 10.):
        earlier = np.maximum(0, np.searchsorted(t, t-delay, side="right")-1)
        prior.append(now-now[earlier])
    return np.concatenate([now, *prior], axis=2).astype(np.float32)


def risk_public_state(r, ids):
    """Exact 70-D risk-tree public state from risk/run_risk.py."""
    u, x, time = r["pub"], r["x"], r["time"]
    p, players = u[ids, :30], x[ids]
    ct, foe = players[:, :5], players[:, 5:]
    present = ct[:, :, 31] > .5
    seen = (foe[:, :, 31] > .5) & (foe[:, :, 4] > .5)
    known = foe[:, :, 31] > .5

    def masked_min(vals, present, fallback=4.):
        return np.where(present, vals, fallback).min(1)

    def mean(z):
        return (z*present).sum(1)/np.maximum(present.sum(1), 1)

    extra = [mean(np.linalg.norm(ct[:, :, 13:15], axis=-1)), mean(ct[:, :, 10]),
             masked_min(ct[:, :, 5], present, 1.), masked_min(foe[:, :, 5], known, 4.)]
    for center in ([1175, 2550], [-1575, 2725]):
        c = np.asarray(center)/4096
        for unit, mask in ((ct, present), (foe, seen), (foe, known)):
            extra.append(masked_min(np.linalg.norm(unit[:, :, :2]-c, axis=-1), mask))
    for dt in (2., 6.):
        prev = np.maximum(0, np.searchsorted(time, time[ids]-dt, side="right")-1)
        extra.extend((p[:, 18]-u[prev, 18], p[:, 15]-u[prev, 15], p[:, 19]-u[prev, 19],
                      np.linalg.norm(p[:, 8:10]-u[prev, 8:10], axis=1)))
    return np.ascontiguousarray(np.column_stack([p, public_features(p, players), *extra]), dtype=np.float32)


def baseline_features(r):
    """482-D union of every observed feature used by the frozen full model.

    The original 272 dimensions remain first. SITE contributes 96 dimensions per
    site; the risk-tree transform contributes only its 18 extra columns, because
    its first 52 are already exactly the first 52 neural-summary columns.
    """
    original = features(r)
    spatial = site_features(r).reshape(len(original), 192)
    risk = risk_public_state(r, np.arange(len(original)))
    np.testing.assert_array_equal(risk[:, :52], original[:, :52])
    return np.ascontiguousarray(np.column_stack((original, spatial, risk[:, 52:])), dtype=np.float32)


def frame_indices(r):
    return np.flatnonzero(r["cutoff_valid"] & (r["target_bin"] >= 0))


def scoring_eligibility(r):
    """Frozen clock contract, equivalent to portable/cs2ai/data.py."""
    seconds = np.asarray(r["time"], dtype=np.float64)
    reasons = []
    if seconds.ndim != 1 or not len(seconds) or not np.isfinite(seconds).all():
        reasons.append("invalid_observation_game_clock")
    elif np.any(np.diff(seconds) <= 0):
        reasons.append("non_strict_observation_game_clock")
    tick = np.asarray(r.get("tick", []))
    if tick.shape != seconds.shape or np.any(np.diff(tick) <= 0):
        reasons.append("invalid_observation_tick_order")
    event_seconds = np.asarray(r.get("event_seconds", []), dtype=np.float64)
    events = np.asarray(r.get("events", []))
    if (event_seconds.ndim != 1 or not np.isfinite(event_seconds).all()
            or events.ndim != 2 or events.shape[1] != 2 or len(events) != len(event_seconds)):
        reasons.append("invalid_event_clock")
    elif np.any(np.diff(event_seconds) <= 0):
        reasons.append("non_strict_merged_event_game_clock")
    bins, cutoff = np.asarray(r.get("target_bin", [])), np.asarray(r.get("cutoff_valid", []), dtype=bool)
    count = 0
    if bins.shape != seconds.shape or cutoff.shape != seconds.shape:
        reasons.append("invalid_cutoff_shape")
    else:
        count = int(np.sum(cutoff & (bins >= 0)))
        if not count:
            reasons.append("no_scorable_cutoffs")
    return {"eligible": not reasons, "reasons": reasons, "cutoffs": count}


def internal_split(rows):
    """Frozen event-stratified chronological 22/8 split of training series."""
    by_event, metadata = defaultdict(set), {}
    for row in rows:
        if row["role"] != "train":
            raise ValueError("Inner tuning can use original training series only")
        sid, event, day = str(row["match_id"]), str(row["event"]), str(row["date"])
        if metadata.setdefault(sid, (event, day)) != (event, day):
            raise ValueError(f"Inconsistent series metadata: {sid}")
        by_event[event].add((day, sid))
    held = set()
    for event, series in by_event.items():
        count = math.ceil(len(series)/5)
        if count >= len(series):
            raise ValueError(f"No earlier fitting series in {event}")
        held.update(sid for _, sid in sorted(series)[-count:])
    return ([r for r in rows if str(r["match_id"]) not in held],
            [r for r in rows if str(r["match_id"]) in held])


def _support(rows):
    return {"series": len({str(r["match_id"]) for r in rows}), "rounds": len(rows),
            "cutoffs": sum(int(r["cutoff_count"]) for r in rows),
            "series_ids": sorted({str(r["match_id"]) for r in rows})}


def prepare_data(project_root, cache_dir, force=False):
    """Build a small, portable derivative, without opening raw DEMO files."""
    project_root, cache_dir = Path(project_root).resolve(), Path(cache_dir).resolve()
    if (cache_dir/"manifest.json").is_file() and not force:
        manifest = json.loads((cache_dir/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Cache schema mismatch")
        verify_cache(cache_dir)
        return manifest
    source = project_root/"data"
    geometry_path = project_root/"source/portable/cs2ai/geometry.json"
    source_geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    for name, box in SITE_GEOMETRY.items():
        np.testing.assert_array_equal(box, source_geometry[name])
    index = json.loads((source/"index.json").read_text(encoding="utf-8"))
    metadata, frame_parts, time_parts, cutoff_parts, opened, excluded = [], [], [], [], [], []
    frame_start, cutoff_start = 0, 0
    causal_checks = []
    for number, entry in enumerate(sorted(index, key=lambda r: str(r["match_id"]))):
        path = (source/entry["path"]).resolve()
        if source not in path.parents:
            raise ValueError("Match archive must be within the processed data directory")
        opened.append({"path": str(path.relative_to(project_root)).replace("\\", "/"),
                       "sha256": digest(path), "bytes": path.stat().st_size,
                       "role": entry["role"], "match_id": str(entry["match_id"])})
        with np.load(path, allow_pickle=False) as packed:
            metas = json.loads(packed["metadata"].item())
            ordered = sorted(enumerate(metas), key=lambda pair: str(pair[1]["round_uid"]))
            for round_in_file, meta in ordered:
                if str(meta["match_id"]) != str(entry["match_id"]) or meta["role"] != entry["role"]:
                    raise ValueError("Round and index identities disagree")
                prefix = f"r{round_in_file}_"
                # Explicit field whitelist: do not iterate through all NPZ keys.
                r = {**meta, **{key: packed[prefix+key] for key in READ_KEYS}}
                eligibility = scoring_eligibility(r)
                if not eligibility["eligible"]:
                    excluded.append({"match_id": str(meta["match_id"]),
                                     "round_uid": str(meta["round_uid"]), **eligibility})
                    continue
                values, take = baseline_features(r), frame_indices(r)
                if values.shape != (len(r["time"]), DIMENSIONS) or not np.isfinite(values).all():
                    raise ValueError("Invalid causal feature dimensions or nonfinite input")
                bins, marks = r["target_bin"][take].astype(np.int64), r["target_mask"][take].astype(np.int64)
                if not np.all((bins >= 0) & (bins <= HORIZON)):
                    raise ValueError("Invalid first-event time bin")
                if not np.all((bins == HORIZON) | ((marks >= 1) & (marks <= MARKS))):
                    raise ValueError("Invalid nonempty event mask")
                if len(causal_checks) < 5 and len(values) > 5:
                    stop = len(values)//2
                    prefix_row = {**r, "x": r["x"][:stop], "pub": r["pub"][:stop], "time": r["time"][:stop]}
                    np.testing.assert_array_equal(baseline_features(prefix_row), values[:stop])
                    causal_checks.append(str(meta["round_uid"]))
                rid = len(metadata)
                row = {key: str(meta[key]) for key in ("match_id", "round_uid", "role", "event", "date")}
                row.update(frame_start=frame_start, frame_count=len(values),
                           cutoff_start=cutoff_start, cutoff_count=len(take))
                metadata.append(row)
                frame_parts.append(values)
                time_parts.append(r["time"].astype(np.float32))
                cutoff_parts.append({"round_index": np.full(len(take), rid, dtype=np.int32),
                    "cutoff_index": take.astype(np.int32), "bins": bins.astype(np.int16),
                    "marks": marks.astype(np.int16), "time": r["time"][take].astype(np.float32),
                    "y": np.where(bins == HORIZON, HORIZON*MARKS, bins*MARKS + marks-1).astype(np.int64)})
                frame_start += len(values)
                cutoff_start += len(take)
        if (number+1) % 10 == 0:
            print(f"Prepared {number+1}/{len(index)} processed match archives", flush=True)
    train = [r for r in metadata if r["role"] == "train"]
    fit, held = internal_split(train)
    splits = {role: _support([r for r in metadata if r["role"] == role]) for role in ("train", "validation", "test")}
    splits.update(inner_fit=_support(fit), inner_val=_support(held))
    for role, expected in EXPECTED.items():
        for key, value in expected.items():
            if splits[role][key] != value:
                raise ValueError(f"Frozen support mismatch {role}/{key}: {splits[role][key]} != {value}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep both a hashed cache-local copy and the convenient package data copy.
    (cache_dir/"geometry.json").write_bytes(geometry_path.read_bytes())
    (cache_dir.parent/"geometry.json").write_bytes(geometry_path.read_bytes())
    np.save(cache_dir/"features.npy", np.concatenate(frame_parts))
    np.save(cache_dir/"frame_time.npy", np.concatenate(time_parts))
    cutoffs = {key: np.concatenate([part[key] for part in cutoff_parts]) for key in cutoff_parts[0]}
    np.savez_compressed(cache_dir/"cutoffs.npz", **cutoffs)
    write_json(cache_dir/"rounds.json", metadata)
    files = {name: {"sha256": digest(cache_dir/name), "bytes": (cache_dir/name).stat().st_size}
             for name in ("features.npy", "frame_time.npy", "cutoffs.npz", "rounds.json", "geometry.json")}
    original_sources = ["baselines/data.py", "source/portable/cs2ai/data.py",
                        "source/portable/cs2ai/geometry.json"]
    manifest = {"schema": SCHEMA, "dimensions": DIMENSIONS, "n_classes": N_CLASSES,
        "index_sha256": digest(source/"index.json"), "read_fields": list(READ_KEYS),
        "feature_input_fields": ["x (causal CT observation cache)", "pub[:, :30]", "time"],
        "label_only_fields": ["target_bin", "target_mask"],
        "eligibility_only_fields": ["tick", "cutoff_valid", "events", "event_seconds"],
        "excluded_hidden_fields": ["truth", "seen", "ct_xyz", "event_time", "ingress_first_time"],
        "history": "Every emitted sampled frame from this round's start through the cutoff, inclusive; no crop or truncation.",
        "current_summary": "482-D union: original neural 272 + SITE A/B 96 each + risk-tree unique 18.",
        "feature_blocks": [
            {"name": "neural_summary", "slice": [0, 272], "source": "gain_v2/neural.py:features", "meaning": "Current 68 and 1.5/5/12-second differences"},
            {"name": "site_A", "slice": [272, 368], "source": "scale_gate/defense/run.py:features", "meaning": "Current 32 and 3/10-second differences"},
            {"name": "site_B", "slice": [368, 464], "source": "scale_gate/defense/run.py:features", "meaning": "Current 32 and 3/10-second differences"},
            {"name": "tree_extra", "slice": [464, 482], "source": "risk/run_risk.py:public_state[:,52:]", "meaning": "Health/age minima, closest CT/exposed/known site distances and 2/6-second changes"},
        ],
        "input_matching": "All observation-derived feature blocks used by the frozen full hybrid are available to every new baseline. Rule/lead deterministic functions use the included public clocks and counts.",
        "site_geometry": source_geometry,
        "site_geometry_file": "geometry.json (cache-local); ../geometry.json (package companion)",
        "site_geometry_sha256": digest(geometry_path),
        "class_order": "time-major 20*511 event cells, then a single no-event tail at index 10220",
        "score": "Cutoff mean within round, round mean within series, equal series mean",
        "splits": splits, "frames": frame_start, "rounds": len(metadata), "cutoffs": cutoff_start,
        "max_prefix_length": int(cutoffs["cutoff_index"].max()+1),
        "max_round_frames": max(row["frame_count"] for row in metadata),
        "files": files, "sources": opened, "excluded_rounds": excluded,
        "source_algorithms": {p: digest(project_root/p) for p in original_sources},
        "causal_prefix_checks": causal_checks}
    write_json(cache_dir/"manifest.json", manifest)
    verify_cache(cache_dir)
    verify_frozen_alignment(cache_dir, project_root)
    return manifest


def verify_cache(cache_dir):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported cache schema")
    for filename, expected in manifest["files"].items():
        path = cache_dir/filename
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Derived cache checksum failed: {filename}")
    features_array = np.load(cache_dir/"features.npy", mmap_mode="r", allow_pickle=False)
    if features_array.shape != (manifest["frames"], DIMENSIONS):
        raise ValueError("Derived feature shape mismatch")
    return manifest


class CutoffDataset(Dataset):
    """Each item is a complete causal prefix ending at one eligible cutoff."""
    def __init__(self, cache_dir, split="train", series_ids=None, max_examples=None):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir/"manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported cache schema")
        self.split = split
        self.frames = np.load(self.cache_dir/"features.npy", mmap_mode="r", allow_pickle=False)
        self.input_dim = int(self.manifest["dimensions"])
        if self.frames.shape[1] != self.input_dim:
            raise ValueError("Feature dimension differs from the matched-input manifest")
        self.rounds = json.loads((self.cache_dir/"rounds.json").read_text(encoding="utf-8"))
        with np.load(self.cache_dir/"cutoffs.npz", allow_pickle=False) as archive:
            all_records = {key: archive[key] for key in archive.files}
        role = "train" if split in ("inner_fit", "inner_val") else split
        if role not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split {split}")
        permitted = set(self.manifest["splits"][split]["series_ids"])
        if series_ids is not None:
            requested = {str(sid) for sid in series_ids}
            if not requested <= permitted:
                raise ValueError("Requested series cross the selected protocol partition")
            permitted = requested
        selected_rounds = np.array([r["role"] == role and r["match_id"] in permitted for r in self.rounds])
        global_ids = np.flatnonzero(selected_rounds[all_records["round_index"]])
        if max_examples is not None:
            global_ids = global_ids[:int(max_examples)]
        if not len(global_ids):
            raise ValueError("Dataset partition has no eligible cutoffs")
        self.global_indices = global_ids
        self.cutoffs = {key: value[global_ids] for key, value in all_records.items()}
        rr = self.cutoffs["round_index"]
        self.records = {
            "global_index": global_ids,
            "series_id": np.asarray([self.rounds[i]["match_id"] for i in rr]),
            "round_id": np.asarray([self.rounds[i]["round_uid"] for i in rr]),
            "bins": self.cutoffs["bins"], "marks": self.cutoffs["marks"],
            "time": self.cutoffs["time"], "cutoff_index": self.cutoffs["cutoff_index"],
        }
        self.records["series_ids"] = self.records["series_id"]
        self.records["round_ids"] = self.records["round_id"]
        self.lengths = self.cutoffs["cutoff_index"].astype(np.int64)+1
        self.y = self.cutoffs["y"]
        self.weights = balanced_weights(self.records["series_id"], self.records["round_id"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        index = int(index)
        row = self.rounds[int(self.cutoffs["round_index"][index])]
        start = int(row["frame_start"])
        end = start + int(self.lengths[index])
        if end > start + int(row["frame_count"]):
            raise ValueError("Cutoff extends beyond its own round")
        return {"x": self.frames[start:end], "y": int(self.y[index]),
                "weight": float(self.weights[index]), "index": index}

    def aggregate(self, indices, values):
        return aggregate(self, indices, values)

    def archive_metadata(self):
        """Metadata layout identical to frozen per-cutoff forecast archives."""
        sid, rid = self.records["series_id"], self.records["round_id"]
        starts = np.r_[0, np.flatnonzero((sid[1:] != sid[:-1]) | (rid[1:] != rid[:-1]))+1]
        return {"series_ids": sid[starts], "round_ids": rid[starts],
                "counts": np.diff(np.r_[starts, len(self)]),
                **{key: self.records[key] for key in ("bins", "marks", "time", "cutoff_index")}}


def balanced_weights(series, rounds):
    """N times each cutoff's exact contribution to the primary macro score."""
    series, rounds = np.asarray(series), np.asarray(rounds)
    weights = np.empty(len(series), np.float64)
    unique_series = np.unique(series)
    for sid in unique_series:
        match = series == sid
        round_ids = np.unique(rounds[match])
        for rid in round_ids:
            take = match & (rounds == rid)
            weights[take] = len(series)/(len(unique_series)*len(round_ids)*int(take.sum()))
    np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-12)
    return weights.astype(np.float32)


def aggregate(dataset, indices, values):
    """Macro-average aligned per-cutoff values over rounds and then series."""
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if indices.ndim != 1 or len(indices) != len(values) or len(np.unique(indices)) != len(indices):
        raise ValueError("Aggregation needs one value per unique local dataset index")
    if not len(indices) or indices.min() < 0 or indices.max() >= len(dataset) or not np.isfinite(values).all():
        raise ValueError("Invalid aggregation indices or values")
    sid, rid = dataset.records["series_id"][indices], dataset.records["round_id"][indices]
    per_series, round_count = {}, 0
    for series in sorted(set(sid.tolist())):
        match = sid == series
        round_values = [values[match & (rid == r)].mean(axis=0) for r in sorted(set(rid[match].tolist()))]
        per_series[series] = np.mean(round_values, axis=0)
        round_count += len(round_values)
    mean = np.mean(list(per_series.values()), axis=0)
    encode = lambda value: float(value) if np.ndim(value) == 0 else np.asarray(value).tolist()
    report = {"mean": encode(mean), "per_series": {key: encode(v) for key, v in per_series.items()},
              "series": len(per_series), "rounds": round_count, "cutoffs": len(indices)}
    if np.ndim(mean) == 0:
        report["macro_nll"] = float(mean)
    return report


def collate_fn(items):
    lengths = torch.as_tensor([len(item["x"]) for item in items], dtype=torch.long)
    dimensions = items[0]["x"].shape[-1]
    if any(item["x"].shape[-1] != dimensions for item in items):
        raise ValueError("Mixed input dimensions in a minibatch")
    x = torch.zeros((len(items), int(lengths.max()), dimensions), dtype=torch.float32)
    for i, item in enumerate(items):
        x[i, :lengths[i]] = torch.from_numpy(np.array(item["x"], dtype=np.float32, copy=True))
    return {"x": x, "lengths": lengths,
            "y": torch.as_tensor([item["y"] for item in items], dtype=torch.long),
            "weight": torch.as_tensor([item["weight"] for item in items], dtype=torch.float32),
            "index": torch.as_tensor([item["index"] for item in items], dtype=torch.long)}


class LengthBucketBatchSampler(Sampler):
    """Shuffle all examples once, then sort local pools to reduce padding."""
    def __init__(self, lengths, batch_size, shuffle, seed, bucket_batches=20):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.shuffle, self.seed = int(batch_size), bool(shuffle), int(seed)
        self.bucket_batches, self.epoch = int(bucket_batches), 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.lengths)/self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        ids = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        pool_size = self.batch_size*self.bucket_batches
        for start in range(0, len(ids), pool_size):
            pool = ids[start:start+pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            batches.extend(pool[offset:offset+self.batch_size].tolist() for offset in range(0, len(pool), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def make_loader(dataset, batch_size=128, shuffle=False, seed=2026):
    sampler = LengthBucketBatchSampler(dataset.lengths, batch_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def verify_frozen_alignment(cache_dir, project_root):
    """Check row identities against the compact frozen protocol metadata.

    These arrays are the unchanged metadata from the existing FULL seed-2026
    forecast archives; they contain labels and row identities, no predictions.
    The original three seeds share this exact observation/label ordering.
    """
    checks = {}
    for split, cohort in (("validation", "dev"), ("test", "rio")):
        dataset = CutoffDataset(cache_dir, split=split)
        expected = dataset.archive_metadata()
        path = Path(project_root)/f"baselines/protocol/{cohort}_metadata.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen reference metadata is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            for key, value in expected.items():
                np.testing.assert_array_equal(value, archive[key], err_msg=f"{cohort}/{key}")
        checks[cohort] = {"sha256": digest(path), "aligned": True,
                          "reference_seeds_sharing_metadata": [2026, 2027, 2028]}
    write_json(Path(cache_dir)/"frozen_alignment.json", checks)
    return checks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent/"data/cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepared = prepare_data(args.project_root, args.cache_dir, args.force)
    print(json.dumps({"schema": prepared["schema"], "splits": prepared["splits"],
                      "frames": prepared["frames"], "cutoffs": prepared["cutoffs"]}, indent=2))
