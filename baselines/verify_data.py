"""Verify reconstructed cache, fixed splits, aggregation, and causal inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
import data


def verify(cache_dir, project=None):
    manifest = data.verify_cache(cache_dir)
    checks = {"cache_checksums": True, "splits": {}}
    for split in ("train", "validation", "test", "inner_fit", "inner_val"):
        dataset = data.CutoffDataset(cache_dir, split)
        values = np.random.default_rng(20261117).normal(size=len(dataset))
        macro = data.aggregate(dataset, np.arange(len(dataset)), values)
        np.testing.assert_allclose(np.dot(values, dataset.weights)/len(dataset), macro["mean"], atol=2e-8)
        metadata = dataset.archive_metadata()
        np.testing.assert_array_equal(metadata["bins"], dataset.y//data.MARKS)
        event = dataset.y < data.HORIZON*data.MARKS
        np.testing.assert_array_equal((dataset.y[event] % data.MARKS)+1, metadata["marks"][event])
        for i in (0, len(dataset)//2, len(dataset)-1):
            item = dataset[i]
            row = dataset.rounds[int(dataset.cutoffs["round_index"][i])]
            start = row["frame_start"]
            assert len(item["x"]) == int(dataset.records["cutoff_index"][i])+1
            np.testing.assert_array_equal(item["x"], dataset.frames[start:start+len(item["x"])])
        checks["splits"][split] = {key: macro[key] for key in ("series", "rounds", "cutoffs")}
    try:
        data.CutoffDataset(cache_dir, "train", series_ids=manifest["splits"]["test"]["series_ids"][:1])
    except ValueError:
        checks["cross_partition_request_rejected"] = True
    else:
        raise AssertionError("Cross-partition series request was accepted")
    if project is not None:
        project = Path(project)
        checks["frozen_alignment"] = data.verify_frozen_alignment(cache_dir, project)
        index = json.loads((project/"data/index.json").read_text(encoding="utf-8"))
        roles_checked = set()
        for entry in index:
            if entry["role"] in roles_checked:
                continue
            with np.load(project/"data"/entry["path"], allow_pickle=False) as archive:
                metadata = json.loads(archive["metadata"].item())
                for number, meta in enumerate(metadata):
                    r = {**meta, **{key: archive[f"r{number}_{key}"] for key in data.READ_KEYS}}
                    if not data.scoring_eligibility(r)["eligible"] or len(r["time"]) < 4:
                        continue
                    values = data.baseline_features(r)
                    stop = len(r["time"])//2
                    prefix = {**r, "time": r["time"][:stop], "pub": r["pub"][:stop], "x": r["x"][:stop]}
                    np.testing.assert_array_equal(data.baseline_features(prefix), values[:stop])
                    altered = {**r, "target_bin": np.zeros_like(r["target_bin"]),
                               "target_mask": np.full_like(r["target_mask"], 511)}
                    np.testing.assert_array_equal(data.baseline_features(altered), values)
                    roles_checked.add(entry["role"])
                    break
        assert roles_checked == {"train", "validation", "test"}
        checks["causal_prefix_and_label_independence"] = sorted(roles_checked)
    data.write_json(Path(cache_dir)/"verification.json", checks)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent/"data/cache")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.cache_dir, args.project_root), indent=2))
