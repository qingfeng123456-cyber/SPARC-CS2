"""Portable causal CT observations and independently derived replay labels.

Raw demos are only opened by prepare(). Runtime loading uses portable NPZ
caches. Labels and diagnostic truth are never concatenated to student input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time as wall_time
import traceback

import numpy as np

SCHEMA = "ct-event-cache-v1"
GEOMETRY_PATH = Path(__file__).with_name("geometry.json")
GEOMETRY = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
EVENT_NAMES = [
    "IngressA",
    "IngressB",
    "RegroupAB",
    "RegroupBA",
    "PlantA",
    "PlantB",
    "DeathT",
    "DeathCT",
    "RoundEnd",
]
FIELDS = [
    "X",
    "Y",
    "Z",
    "team_num",
    "is_alive",
    "is_warmup_period",
    "is_coach_team",
    "game_time",
    "approximate_spotted_by",
    "health",
    "armor_value",
    "current_equip_value",
    "yaw",
    "flash_duration",
    "has_defuser",
    "is_scoped",
    "team_clan_name",
]
NATIVE_FIELDS = [
    "X",
    "Y",
    "Z",
    "team_num",
    "is_alive",
    "is_warmup_period",
    "is_coach_team",
    "game_time",
    "is_freeze_period",
    "is_technical_timeout",
    "is_waiting_for_resume",
]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def split_role(day):
    return (
        "train"
        if day <= "2026-04-06"
        else "validation"
        if day <= "2026-04-10"
        else "test"
    )


def geometry_regions(xyz):
    xyz = np.asarray(xyz)
    result = np.zeros(xyz.shape[:-1], np.int8)
    for name, code in (("stageA", 1), ("stageB", 2), ("siteA", 3), ("siteB", 4)):
        result[inside(xyz, GEOMETRY[name])] = code
    return result


def inside(xyz, box, inset=0):
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    return (
        (x >= box[0] + inset)
        & (x <= box[1] - inset)
        & (y >= box[2] + inset)
        & (y <= box[3] - inset)
        & (z >= box[4])
        & (z <= box[5])
    )


def _sid(value):
    return str(int(value))


def spotters(value):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    return {_sid(v) for v in value if v is not None and int(v) != 0}


def targets_at(times, event_times, event_masks, event_first=None, horizon=20):
    """First event strictly after each cutoff; time bins are right closed."""
    times = np.asarray(times)
    event_times = np.asarray(event_times)
    loc = np.searchsorted(event_times, times + 1e-8, side="right")
    bins = np.full(len(times), -1, np.int16)
    masks = np.zeros(len(times), np.int16)
    next_time = np.full(len(times), np.nan, np.float32)
    first_time = np.full(len(times), np.nan, np.float32)
    for i, j in enumerate(loc):
        if j >= len(event_times):
            continue
        delta = event_times[j] - times[i]
        if delta <= horizon + 1e-6:
            bins[i] = max(0, min(horizon - 1, math.ceil(delta - 1e-7) - 1))
            masks[i] = event_masks[j]
            next_time[i] = event_times[j]
            if event_first is not None:
                first_time[i] = event_first[j]
        elif times[i] + horizon <= event_times[-1] + 1e-6:
            bins[i] = horizon
    return bins, masks, next_time, first_time


def crossing_events(ticks, seconds, xyz, alive):
    """Native-tick hysteretic region entry, two distinct surviving players."""
    output, evidence = [], []
    stage_entries = [[], []]
    for site_index, name in enumerate(("siteA", "siteB")):
        outer = inside(xyz, GEOMETRY[name])
        inner = inside(xyz, GEOMETRY[name], GEOMETRY["boundary_inset_units"])
        recent, last_emit, armed = {}, -math.inf, True
        state = outer[0].copy()
        for k in range(1, len(ticks)):
            now = float(seconds[k])
            if not outer[k, alive[k]].any():
                armed = True
            entered = np.flatnonzero(~state & inner[k] & alive[k] & alive[k - 1])
            if seconds[k] - seconds[k - 1] > GEOMETRY["maximum_crossing_gap_seconds"]:
                entered = []
            for player in entered:
                recent[int(player)] = now
                evidence.append((int(ticks[k]), int(player), site_index, now))
            recent = {
                p: t
                for p, t in recent.items()
                if now - t <= 3 and alive[k, p] and outer[k, p]
            }
            if armed and now - last_emit >= 5 and len(recent) >= 2:
                ordered = sorted(recent.items(), key=lambda v: (v[1], v[0]))
                # Last arrival in the first valid pair is the confirmation.
                first = ordered[-2][1]
                output.append((int(ticks[k]), 1 << site_index, first))
                last_emit, armed = now, False
                recent.clear()
            state = np.where(inner[k], True, np.where(~outer[k], False, state))
    for stage, name in enumerate(("stageA", "stageB")):
        mask = inside(xyz, GEOMETRY[name]) & alive
        arrivals = [dict(), dict()]
        history = []
        last = -math.inf
        for k in range(1, len(ticks)):
            now = float(seconds[k])
            for player in np.flatnonzero(mask[k] & ~mask[k - 1]):
                arrivals[0][int(player)] = now
            arrivals[0] = {
                p: t for p, t in arrivals[0].items() if now - t <= 3 and mask[k, p]
            }
            keys = sorted(arrivals[0])
            if len(keys) >= 2 and now - last >= 5:
                for i, p in enumerate(keys):
                    for q in keys[i + 1 :]:
                        history.append((int(ticks[k]), now, p, q))
                last = now
        stage_entries[stage] = history
    # Same pair must have left its source staging, survived and arrived together.
    for source, dest in ((0, 1), (1, 0)):
        last = -math.inf
        for tick, now, p, q in stage_entries[dest]:
            prior = [
                h
                for h in stage_entries[source]
                if h[2:] == (p, q) and 0 < now - h[1] <= 15
            ]
            if not prior or now - last < 5:
                continue
            origin = prior[-1]
            a = np.searchsorted(ticks, origin[0])
            b = np.searchsorted(ticks, tick)
            if not alive[a : b + 1, [p, q]].all():
                continue
            source_mask = inside(
                xyz[a : b + 1, [p, q]], GEOMETRY["stageA" if source == 0 else "stageB"]
            )
            if source_mask.all(axis=0).any():
                continue
            output.append((tick, 1 << (2 + source), np.nan))
            last = now
    return output, evidence


def causal_features(frames, ticks, times, public_events, ct_ids, t_ids):
    """Consume only past/current public events; enemy slots allocate on exposure."""
    x = np.zeros((len(ticks), 10, 32), np.float32)
    pub = np.zeros((len(ticks), 64), np.float32)
    truth = np.zeros((len(ticks), 5, 3), np.float32)
    truth_alive = np.zeros((len(ticks), 5), bool)
    ctpos = np.zeros((len(ticks), 5, 3), np.float32)
    seen = np.zeros((len(ticks), 5), bool)
    memory, slots, dead = {}, {}, set()
    past_deaths = {2: -math.inf, 3: -math.inf}
    planted = None
    event_index = 0
    previous = {}
    count_spot = missing_spot = 0
    for i, (tick, now) in enumerate(zip(ticks, times)):
        f = frames[int(tick)]
        f = f.set_index("steamid", drop=False)
        while (
            event_index < len(public_events) and public_events[event_index][0] <= tick
        ):
            e = public_events[event_index]
            if e[1] == "death":
                dead.add(e[2])
                past_deaths[e[3]] = float(now)
            elif e[1] == "plant":
                planted = float(now)
            event_index += 1
        alive_ct = {
            sid for sid in ct_ids if sid in f.index and bool(f.loc[sid, "is_alive"])
        }
        pub[i, 0] = now / 120
        pub[i, 1] = len(alive_ct) / 5
        pub[i, 2] = sum(s not in dead for s in t_ids) / 5
        pub[i, 3] = planted is not None
        pub[i, 4] = min(2, (now - planted) / 45) if planted is not None else 0
        pub[i, 5] = min(1, (now - past_deaths[2]) / 30)
        pub[i, 6] = min(1, (now - past_deaths[3]) / 30)
        for j, sid in enumerate(ct_ids):
            row = f.loc[sid]
            p = row[["X", "Y", "Z"]].to_numpy(np.float32)
            ctpos[i, j] = p
            alive = bool(row.is_alive)
            if alive and not np.isfinite(p).all():
                raise ValueError("Live CT has missing coordinates in observation prefix")
            v = x[i, j]
            v[3] = alive
            v[4] = 1
            v[31] = alive
            if not alive:
                continue
            v[:3] = p / 4096
            v[5] = float(row.health or 0) / 100
            v[6] = float(row.armor_value or 0) / 100
            v[7] = float(row.current_equip_value or 0) / 10000
            yaw = np.deg2rad(float(row.yaw or 0))
            v[8:10] = [np.sin(yaw), np.cos(yaw)]
            v[10] = min(5, float(row.flash_duration or 0)) / 5
            v[11] = bool(row.has_defuser)
            v[12] = bool(row.is_scoped)
            if sid in previous:
                prev, then = previous[sid]
                dt = now - then
                if 0 < dt <= 1:
                    v[13:16] = np.clip((p - prev) / (dt * 350), -3, 3)
            previous[sid] = (p.copy(), now)
            region = int(geometry_regions(p))
            v[16 + region] = 1
        known = []
        for j, sid in enumerate(t_ids):
            row = f.loc[sid]
            p = row[["X", "Y", "Z"]].to_numpy(np.float32)
            truth[i, j] = p
            truth_alive[i, j] = bool(row.is_alive)
            ids = spotters(row.approximate_spotted_by)
            missing_spot += ids is None
            exposed = bool(ids and ids.intersection(alive_ct) and row.is_alive)
            count_spot += exposed
            seen[i, j] = exposed
            if exposed:
                if sid not in slots:
                    slots[sid] = len(slots)
                memory[sid] = (p.copy(), float(now))
            if sid not in memory:
                continue
            last, then = memory[sid]
            v = x[i, 5 + slots[sid]]
            v[:3] = last / 4096
            v[3] = sid not in dead
            v[4] = exposed
            v[5] = min(4, (now - then) / 30)
            v[6] = 1
            v[7] = ids is None
            v[16 + int(geometry_regions(last))] = 1
            v[31] = sid not in dead
            if sid not in dead:
                known.append(last)
        ca = x[i, :5, 3] > 0
        if ca.any():
            pub[i, 8:11] = ctpos[i, ca].mean(axis=0) / 4096
        if known:
            pub[i, 11:14] = np.mean(known, axis=0) / 4096
        pub[i, 14:19] = [
            x[i, :5, 7].sum() / 5,
            x[i, :5, 5].sum() / 5,
            x[i, :5, 6].sum() / 5,
            x[i, :5, 11].sum() / 5,
            x[i, 5:, 4].sum() / 5,
        ]
        pub[i, 19] = len(known) / 5
        pub[i, 20:25] = np.bincount(geometry_regions(ctpos[i, ca]), minlength=5) / 5
        if known:
            pub[i, 25:30] = (
                np.bincount(geometry_regions(np.asarray(known)), minlength=5) / 5
            )
    return {
        "x": np.nan_to_num(x),
        "pub": np.nan_to_num(pub),
        "truth": truth,
        "alive": truth_alive,
        "ct_xyz": ctpos,
        "seen": seen,
        "spot_positive_rows": count_spot,
        "spot_missing_rows": missing_spot,
    }


def _parse_match(path, row, output):
    from demoparser2 import DemoParser

    parser = DemoParser(str(path))
    header = parser.parse_header()
    if header.get("map_name") != "de_dust2":
        raise ValueError("Map mismatch")
    start = wall_time.perf_counter()
    events = dict(
        parser.parse_events(
            [
                "round_start",
                "round_freeze_end",
                "round_end",
                "bomb_planted",
                "player_death",
            ],
            player=["team_num", "X", "Y", "Z"],
            other=["game_time"],
        )
    )
    native = parser.parse_ticks(NATIVE_FIELDS)
    native = native[
        native.team_num.isin([2, 3])
        & native.steamid.notna()
        & native.is_coach_team.eq(False)
    ]
    native["steamid"] = native["steamid"].map(_sid)
    native = native.sort_values(["tick", "steamid"]).drop_duplicates(
        ["tick", "steamid"], keep="last"
    )
    clock = (
        native[
            [
                "tick",
                "game_time",
                "is_freeze_period",
                "is_technical_timeout",
                "is_waiting_for_resume",
            ]
        ]
        .drop_duplicates("tick")
        .sort_values("tick")
    )
    ck = clock.tick.to_numpy(np.int64)
    gt = clock.game_time.to_numpy(float)
    dk = np.diff(ck)
    dt = np.diff(gt)
    ok = (dt > 0) & (dk > 0)
    rate = float(np.median(dk[ok] / dt[ok]))
    if (
        not 16 <= rate <= 256
        or np.mean(abs(dk[ok] / dt[ok] - rate) / rate < 0.03) < 0.99
    ):
        raise ValueError("Unstable native tick/game_time clock")
    paused = (
        clock[["is_freeze_period", "is_technical_timeout", "is_waiting_for_resume"]]
        .fillna(True)
        .any(axis=1)
        .to_numpy(bool)
    )
    freezes = sorted(set(map(int, events["round_freeze_end"].tick)))
    ends = sorted(set(map(int, events["round_end"].tick)))
    rounds = []
    excluded = []
    requests = []
    evidence = []
    plant_checks = []
    for ri, fr in enumerate(freezes):
        next_fr = freezes[ri + 1] if ri + 1 < len(freezes) else np.iinfo(np.int64).max
        en = next((e for e in ends if fr < e < next_fr), None)
        if en is None:
            excluded.append({"freeze": fr, "reason": "missing_end_before_next_freeze"})
            continue
        a = np.searchsorted(ck, fr)
        b = np.searchsorted(ck, en, side="right")
        tick = ck[a:b]
        game = gt[a:b]
        if len(tick) < 3 or tick[0] != fr or tick[-1] < en - 1:
            excluded.append({"freeze": fr, "reason": "incomplete_native_clock"})
            continue
        seconds = np.r_[
            0,
            np.cumsum(
                np.where(
                    paused[a + 1 : b] | paused[a : b - 1],
                    0,
                    np.maximum(0, np.diff(game)),
                )
            ),
        ]
        local = native[(native.tick >= fr) & (native.tick <= en)]
        initial = local[local.tick == fr]
        if len(initial) != 10 or initial.is_warmup_period.any():
            excluded.append({"freeze": fr, "reason": "warmup_or_incomplete_roster"})
            continue
        ct_ids = sorted(initial.loc[initial.team_num == 3, "steamid"])
        t_ids = sorted(initial.loc[initial.team_num == 2, "steamid"])
        if len(ct_ids) != 5 or len(t_ids) != 5:
            excluded.append({"freeze": fr, "reason": "invalid_roster"})
            continue
        truthframe = local[local.steamid.isin(t_ids)]
        vals = []
        for sid in t_ids:
            z = truthframe[truthframe.steamid == sid].set_index("tick").reindex(tick)
            vals.append(z[["X", "Y", "Z", "is_alive"]].to_numpy(float))
        vals = np.stack(vals, axis=1)
        if not np.isfinite(vals).all():
            excluded.append({"freeze": fr, "reason": "missing_T_native_rows"})
            continue
        raw_events, evd = crossing_events(
            tick, seconds, vals[:, :, :3], vals[:, :, 3].astype(bool)
        )
        evidence.extend(evd)
        public = []
        deaths = events["player_death"]
        for d in deaths[(deaths.tick >= fr) & (deaths.tick <= en)].itertuples(
            index=False
        ):
            if d.user_team_num not in (2, 3):
                continue
            raw_events.append(
                (int(d.tick), 1 << (6 if d.user_team_num == 2 else 7), np.nan)
            )
            public.append(
                (int(d.tick), "death", _sid(d.user_steamid), int(d.user_team_num))
            )
        plants = events["bomb_planted"]
        for p in plants[(plants.tick >= fr) & (plants.tick <= en)].itertuples(
            index=False
        ):
            xyz = np.array([p.user_X, p.user_Y, p.user_Z])
            site = 0 if xyz[0] > 0 else 1
            valid = bool(inside(xyz, GEOMETRY["siteA" if site == 0 else "siteB"]))
            plant_checks.append(
                {
                    "tick": int(p.tick),
                    "site_entity": int(p.site),
                    "site": "A" if site == 0 else "B",
                    "xyz": xyz.tolist(),
                    "contained": valid,
                }
            )
            if not valid:
                raise ValueError("Plant outside fixed site proxy: inspect geometry")
            raw_events.append((int(p.tick), 1 << (4 + site), np.nan))
            public.append((int(p.tick), "plant", "", 0))
        raw_events.append((en, 256, np.nan))
        public.sort()
        merged = {}
        first = {}
        for et, mask, ft in raw_events:
            merged[et] = merged.get(et, 0) | mask
            if np.isfinite(ft):
                first[et] = min(first.get(et, math.inf), ft)
        et = np.array(sorted(merged), np.int64)
        em = np.array([merged[t] for t in et], np.int16)
        es = seconds[np.minimum(np.searchsorted(tick, et), len(seconds) - 1)]
        ef = np.array([first.get(t, np.nan) for t in et], np.float32)
        # Only the regular public clock and public game notices can create input frames.
        grid = np.arange(0, float(seconds[-1]), 0.5)
        indices = np.maximum(0, np.searchsorted(seconds, grid, side="right") - 1)
        sampled = np.unique(
            np.r_[tick[indices], [v[0] for v in public if fr <= v[0] < en]]
        )
        sampled = sampled[sampled < en]
        ix = np.searchsorted(tick, sampled)
        sample_seconds = seconds[ix]
        requests.extend(sampled.tolist())
        rounds.append(
            {
                "match_id": str(row["match_id"]),
                "round_uid": f"{row['match_id']}_{ri:03d}",
                "role": split_role(row["match_date"]),
                "event": row["event"],
                "date": row["match_date"],
                "tick": sampled,
                "time": sample_seconds.astype(np.float32),
                "ct_ids": ct_ids,
                "t_ids": t_ids,
                "public": public,
                "events": np.stack([et, em], axis=1),
                "event_seconds": es.astype(np.float32),
                "event_first": ef,
                "end_tick": en,
                "end_time": float(seconds[-1]),
            }
        )
    del native
    if not rounds:
        raise ValueError("No complete formal rounds")
    df = parser.parse_ticks(FIELDS, ticks=sorted(set(requests)))
    df = df[
        df.team_num.isin([2, 3]) & df.steamid.notna() & df.is_coach_team.eq(False)
    ].copy()
    df["steamid"] = df.steamid.map(_sid)
    frames = {
        int(t): f.drop_duplicates("steamid", keep="last")
        for t, f in df.groupby("tick", sort=False)
    }
    arrays = {}
    meta = []
    spotted = missing = 0
    for rr in rounds:
        rid = rr["round_uid"]
        if any(
            int(t) not in frames
            or any(
                sid not in set(frames[int(t)].steamid)
                for sid in rr["ct_ids"] + rr["t_ids"]
            )
            for t in rr["tick"]
        ):
            excluded.append({"round_uid": rid, "reason": "missing_observation_rows"})
            continue
        f = causal_features(
            frames, rr["tick"], rr["time"], rr["public"], rr["ct_ids"], rr["t_ids"]
        )
        spotted += f.pop("spot_positive_rows")
        missing += f.pop("spot_missing_rows")
        tb, tm, et, ft = targets_at(
            rr["time"], rr["event_seconds"], rr["events"][:, 1], rr["event_first"]
        )
        # A cutoff is evaluated on the fixed integer-second grid, never event-selected.
        grid = np.arange(math.ceil(rr["time"][0]), rr["end_time"], 1)
        ci = np.maximum(0, np.searchsorted(rr["time"], grid, side="right") - 1)
        valid = np.zeros(len(rr["time"]), bool)
        valid[ci] = True
        valid &= tb >= 0
        f.update(
            time=rr["time"],
            tick=rr["tick"],
            target_bin=tb,
            target_mask=tm,
            event_time=et,
            ingress_first_time=ft,
            cutoff_valid=valid,
            events=rr["events"],
            event_seconds=rr["event_seconds"],
            region=geometry_regions(f["truth"]),
        )
        idx = len(meta)
        for k, v in f.items():
            arrays[f"r{idx}_{k}"] = v
        nameframes = frames[int(rr["tick"][0])]
        ctname = nameframes.loc[nameframes.team_num == 3, "team_clan_name"].mode()
        tname = nameframes.loc[nameframes.team_num == 2, "team_clan_name"].mode()
        meta.append(
            {
                k: rr[k]
                for k in (
                    "match_id",
                    "round_uid",
                    "role",
                    "date",
                    "event",
                    "end_tick",
                    "end_time",
                )
            }
            | {
                "ct_team": str(ctname.iloc[0]) if len(ctname) else "unknown",
                "target_team": str(tname.iloc[0]) if len(tname) else "unknown",
            }
        )
    arrays["metadata"] = np.array(json.dumps(meta, ensure_ascii=False))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(output)
    masks = np.concatenate([arrays[f"r{i}_events"][:, 1] for i in range(len(meta))])
    report = {
        "schema": SCHEMA,
        "parser_version": importlib.metadata.version("demoparser2"),
        "match_id": str(row["match_id"]),
        "source_sha256": row["sha256"],
        "source_bytes": int(row["bytes"]),
        "role": split_role(row["match_date"]),
        "header": header,
        "rounds": len(meta),
        "frames": sum(len(arrays[f"r{i}_time"]) for i in range(len(meta))),
        "cutoffs": sum(
            int(arrays[f"r{i}_cutoff_valid"].sum()) for i in range(len(meta))
        ),
        "event_counts": {
            name: int(((masks & (1 << i)) != 0).sum())
            for i, name in enumerate(EVENT_NAMES)
        },
        "raw_event_rows": {k: len(v) for k, v in events.items()},
        "excluded": excluded,
        "geometry_annotation_status": GEOMETRY["annotation_status"],
        "plant_geometry_checks": plant_checks,
        "crossing_records": len(evidence),
        "positive_spot_rows": spotted,
        "missing_spot_rows": missing,
        "tick_rate": rate,
        "seconds": round(wall_time.perf_counter() - start, 2),
    }
    save_json(output.with_suffix(".audit.json"), report)
    return report


def scoring_eligibility(round_data):
    """A common clock contract for teacher fitting, training and scoring.

    Native tick order cannot be inferred from equal compressed game seconds.
    Such a round is retained in the parsed cache and excluded as a whole from
    every downstream role. This includes paused/flag-degenerate clocks even
    when one zero-delay label was produced by the earlier target pass.
    """
    seconds = np.asarray(round_data["time"], dtype=np.float64)
    reasons = []
    if seconds.ndim != 1 or not len(seconds) or not np.isfinite(seconds).all():
        reasons.append("invalid_observation_game_clock")
    elif np.any(np.diff(seconds) <= 0):
        reasons.append("non_strict_observation_game_clock")
    tick = np.asarray(round_data.get("tick", []))
    if tick.shape != seconds.shape or np.any(np.diff(tick) <= 0):
        reasons.append("invalid_observation_tick_order")
    event_seconds = np.asarray(round_data.get("event_seconds", []), dtype=np.float64)
    events = np.asarray(round_data.get("events", []))
    if (event_seconds.ndim != 1 or not np.isfinite(event_seconds).all()
            or events.ndim != 2 or events.shape[1] != 2
            or len(events) != len(event_seconds)):
        reasons.append("invalid_event_clock")
    elif np.any(np.diff(event_seconds) <= 0):
        # Events at one native tick have already been merged by bitwise OR.
        # A remaining tie represents different native ticks at one game time.
        reasons.append("non_strict_merged_event_game_clock")
    bins = np.asarray(round_data.get("target_bin", []))
    cutoff = np.asarray(round_data.get("cutoff_valid", []), dtype=bool)
    count = 0
    if bins.shape != seconds.shape or cutoff.shape != seconds.shape:
        reasons.append("invalid_cutoff_shape")
    else:
        count = int(np.sum(cutoff & (bins >= 0)))
        if not count:
            reasons.append("no_scorable_cutoffs")
    return {"eligible": not reasons, "reasons": reasons, "cutoffs": count}


def load_rounds(root, role=None, include_unscorable=False):
    root = Path(root)
    paths = (
        sorted((root / "matches").glob("*.npz"))
        if (root / "matches").exists()
        else sorted(root.glob("*.npz"))
    )
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            for i, m in enumerate(meta):
                if role is not None and m["role"] != role:
                    continue
                prefix = f"r{i}_"
                arrays = {
                    k[len(prefix) :]: data[k].copy()
                    for k in data.files
                    if k.startswith(prefix)
                }
                round_data = {**m, **arrays}
                if include_unscorable or scoring_eligibility(round_data)["eligible"]:
                    result.append(round_data)
    return result


def audit_clock_eligibility(root, output=None):
    """Report parsed/scorable denominators without modifying the raw caches."""
    root = Path(root)
    rounds = load_rounds(root, include_unscorable=True)
    by_role, excluded = {}, []
    for round_data in rounds:
        role = round_data["role"]
        counts = by_role.setdefault(role, dict(parsed_rounds=0, parsed_cutoffs=0,
                                              scorable_rounds=0, scorable_cutoffs=0,
                                              excluded_rounds=0, excluded_cutoffs=0))
        eligibility = scoring_eligibility(round_data)
        counts["parsed_rounds"] += 1
        counts["parsed_cutoffs"] += eligibility["cutoffs"]
        key = "scorable" if eligibility["eligible"] else "excluded"
        counts[key + "_rounds"] += 1
        counts[key + "_cutoffs"] += eligibility["cutoffs"]
        if not eligibility["eligible"]:
            seconds = np.asarray(round_data["time"])
            event_seconds = np.asarray(round_data.get("event_seconds", []))
            excluded.append(dict(round_uid=round_data["round_uid"], match_id=round_data["match_id"],
                                 role=role, reasons=eligibility["reasons"],
                                 cutoffs=eligibility["cutoffs"],
                                 observation_time_ties=int(np.sum(np.diff(seconds) == 0)),
                                 event_time_ties=int(np.sum(np.diff(event_seconds) == 0))))
    index = root / "index.json"
    report = dict(schema="scoring-clock-eligibility-v1", roles=by_role, excluded=excluded,
                  parsed_rounds=len(rounds), scorable_rounds=len(rounds) - len(excluded),
                  index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
                  rule="Strictly increasing finite observation game time and native ticks; strictly increasing merged event game time; at least one eligible cutoff. Parsed caches retained.")
    if output is not None:
        save_json(output, report)
    return report


def prepare(project, output, match_ids=None):
    project, output = Path(project), Path(output)
    rows = json.loads(
        (project / "acquisition/mirror_selection.json").read_text(encoding="utf-8-sig")
    )["rows"]
    ledger = [
        json.loads(s)
        for s in (project / "acquisition/ledger.jsonl")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if s.strip()
    ]
    if match_ids is not None:
        rows = [r for r in rows if str(r["match_id"]) in set(map(str, match_ids))]
    cache_key = hashlib.sha256(
        (
            SCHEMA
            + GEOMETRY_PATH.read_text(encoding="utf-8")
            + importlib.metadata.version("demoparser2")
        ).encode()
    ).hexdigest()
    reports = []
    failures = []
    for i, r in enumerate(rows):
        mid = str(r["match_id"])
        dest = output / "matches" / (mid + ".npz")
        audit = dest.with_suffix(".audit.json")
        print(
            f"Prepare {i + 1}/{len(rows)} {mid} {split_role(r['match_date'])}",
            flush=True,
        )
        try:
            if dest.exists() and audit.exists():
                previous = json.loads(audit.read_text(encoding="utf-8"))
                if (
                    previous.get("cache_key") == cache_key
                    and previous.get("source_sha256") == r["sha256"]
                ):
                    reports.append(previous)
                    continue
            receipts = [
                a
                for a in ledger
                if a.get("source_url") == r["download_url"]
                and a.get("status") in ("verified_existing", "downloaded_verified")
                and a.get("sha256") == r["sha256"]
            ]
            raw = next(
                (
                    Path(a["path"])
                    for a in reversed(receipts)
                    if Path(a["path"]).is_file()
                    and Path(a["path"]).stat().st_size == int(r["bytes"])
                ),
                None,
            )
            if raw is None:
                raise FileNotFoundError(
                    "No matching successful receipt and existing raw file"
                )
            report = _parse_match(raw, r, dest)
            report["cache_key"] = cache_key
            save_json(audit, report)
            reports.append(report)
            print(
                f"  {report['rounds']} rounds / {report['cutoffs']} cutoffs / {report['seconds']}s",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "match_id": mid,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED {mid}: {exc}", flush=True)
            save_json(output / "failures.json", failures)
    report = {
        "schema": SCHEMA,
        "cache_key": cache_key,
        "selected": len(rows),
        "successful": len(reports),
        "failed": failures,
        "rounds": sum(r["rounds"] for r in reports),
        "cutoffs": sum(r["cutoffs"] for r in reports),
        "geometry": GEOMETRY,
        "matches": reports,
    }
    save_json(output / "audit.json", report)
    save_json(
        output / "feature_schema.json",
        {
            "schema": SCHEMA,
            "x_shape": "L,10,32",
            "pub_shape": "L,64",
            "player_order": "5 CT stable within round; 5 enemy slots assigned at first public exposure",
            "normalization": "Fixed physical scales, no validation/test fitted statistics",
            "ct": {
                "0:3": "XYZ / 4096",
                "3": "alive",
                "4": "own presence",
                "5": "health / 100",
                "6": "armor / 100",
                "7": "equipment value / 10000",
                "8:10": "sin(yaw),cos(yaw)",
                "10": "flash duration clipped 5s / 5",
                "11": "defuser",
                "12": "scoped",
                "13:16": "past finite-difference XYZ velocity / 350 clipped [-3,3]",
                "16:21": "one-hot current geometric region",
                "21:31": "reserved zero",
                "31": "present",
            },
            "enemy": {
                "0:3": "last exposed XYZ / 4096",
                "3": "not publicly dead",
                "4": "currently exposed",
                "5": "last-seen age /30 clipped 4",
                "6": "has ever been exposed",
                "7": "current radar data missing",
                "16:21": "one-hot last observed region",
                "31": "known and not publicly dead",
                "other": "zero; private equipment/health never copied",
            },
            "pub": {
                "0": "elapsed /120",
                "1:3": "CT,T public alive count /5",
                "3": "public plant confirmation",
                "4": "plant age /45 clipped2",
                "5:7": "time since last public T,CT death /30 clipped1",
                "8:11": "CT centroid /4096",
                "11:14": "known-enemy memory centroid /4096",
                "14:19": "CT aggregate equipment,health,armor,defuser and current exposed enemy count",
                "19": "known living enemy count /5",
                "20:25": "CT region counts /5",
                "25:30": "known enemy region counts /5",
                "30:64": "reserved zero",
            },
            "target_bin": "0..19 event in (j,j+1],20 no event through horizon,-1 unknown",
            "target_mask": "bits in EVENT_NAMES order",
            "event_names": EVENT_NAMES,
            "truth_boundary": "truth,alive,region,events,event_seconds,event_time,ingress_first_time are labels/diagnostics, never model features",
        },
    )
    save_json(
        output / "index.json",
        [
            {
                "match_id": r["match_id"],
                "role": r["role"],
                "path": f"matches/{r['match_id']}.npz",
                "rounds": r["rounds"],
            }
            for r in reports
        ],
    )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--matches", nargs="*")
    args = ap.parse_args()
    result = prepare(args.project, args.output, args.matches)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("matches", "geometry")},
            ensure_ascii=False,
        )
    )
    raise SystemExit(bool(result["failed"]))
