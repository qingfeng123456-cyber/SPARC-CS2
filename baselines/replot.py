"""Redraw three completed-study comparison plots from the bundled real summary.

This consumes aggregated real results, never trains or opens match archives.
The saved JSON retains cohort/method/series/seed scores and bootstrap intervals.
"""
import argparse
import json
from pathlib import Path
import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--summary", type=Path, default=repo/"results/baselines/summary.json")
    parser.add_argument("--output", type=Path, default=repo/"outputs/baseline_plots")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("A complete real experiment summary is required")
    models = tuple(summary["models"])
    order = models + ("NEURAL", "TREE", "BASE_MIX", "FULL")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = report._figures(args.output, summary, order, models)
    print(json.dumps(dict(output=str(args.output.resolve()), figures=[f["stem"] for f in figures]),indent=2))


if __name__ == "__main__":
    main()
