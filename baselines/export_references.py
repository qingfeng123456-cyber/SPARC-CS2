"""Convert frozen SPARC-CS2 cutoff archives to matched baseline references.

Input: INPUT/{dev,rio}/runs/{NEURAL,TREE,BASE_MIX,FULL}_seedSEED/cutoffs.npz.
Each archive has per-round series_ids/round_ids/counts, row-level bins/marks/
time/cutoff_index, and nll/time_nll/probability_a_b_10s/time_mass/tail.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import data


def export(input_dir, output_dir):
    manifest = []
    for cohort in ("dev", "rio"):
        for name in ("NEURAL", "TREE", "BASE_MIX", "FULL"):
            for seed in (2026, 2027, 2028):
                source = input_dir/cohort/"runs"/f"{name}_seed{seed}"/"cutoffs.npz"
                with np.load(source, allow_pickle=False) as z:
                    counts = z["counts"]
                    n = int(counts.sum())
                    out = {key:z[key] for key in ("bins", "marks", "time", "cutoff_index", "nll",
                           "time_nll", "probability_a_b_10s", "time_mass", "tail")}
                    out.update(series_id=np.repeat(z["series_ids"], counts),
                               round_id=np.repeat(z["round_ids"], counts), index=np.arange(n))
                    out["y"] = np.where(out["bins"]==20,10220,out["bins"]*511+out["marks"]-1)
                target = output_dir/cohort/f"{name}_seed{seed}.npz"
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,**out)
                manifest.append(dict(file=str(target.relative_to(output_dir)), sha256=data.digest(target),
                                     source_sha256=data.digest(source)))
    data.write_json(output_dir/"manifest.json",dict(references=manifest, fitting_operations=0))
    return len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=repo/"outputs/references")
    args = parser.parse_args()
    print(json.dumps(dict(reference_files=export(args.input,args.output))))
