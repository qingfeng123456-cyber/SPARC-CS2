"""Independent checks of aggregation, alignment failures, and report delivery.

Synthetic predictions used here are implementation fixtures, never research
measurements. Run: python -m unittest test_report -v
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import report


def fixture(delta: float = .4, seed_delta: float = 0.) -> dict:
    series = np.array(["s1"] * 3 + ["s2"] * 5)
    rounds = np.array(["r1", "r1", "r2", "r3", "r3", "r3", "r4", "r4"])
    cutoff = np.array([0, 1, 0, 0, 1, 2, 0, 1])
    bins = np.array([0, 2, 20, 1, 20, 0, 3, 20])
    marks = np.array([1, 2, 0, 1, 0, 2, 1, 0])
    time_mass = np.full((8, 20), .02)
    tail = np.full(8, .6)
    tll = -np.log(np.where(bins < 20, .02, .6))
    nll = tll + (bins < 20) * (delta + seed_delta + .02 * np.arange(8))
    return dict(series_id=series, round_id=rounds, bins=bins, marks=marks,
                time=cutoff.astype(float)+1, cutoff_index=cutoff, index=np.arange(8),
                y=np.where(bins < 20, bins*511+marks-1, 10220), nll=nll, time_nll=tll,
                probability_a_b_10s=np.tile([.1, .15], (8, 1)), time_mass=time_mass, tail=tail)


def save_fixture(path: Path, arrays: dict, model: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(dict(
        model=model, seed=seed, parameters=1234, inference_seconds=.08,
        prediction_examples=len(arrays["bins"]),
    )), encoding="utf-8")


class ReportTests(unittest.TestCase):
    def test_macro_round_then_series_is_not_naive_cutoff_mean(self):
        a = fixture()
        a["nll"] = np.array([0., 2., 5., 6., 6., 6., 8., 8.])
        a["time_nll"] = np.zeros(8)
        result = report._aggregate(a)
        # s1: mean(mean(0,2),5)=3; s2: mean(6,8)=7; macro=5.
        self.assertAlmostEqual(result["values"][:, 0].mean(), 5.)
        self.assertNotAlmostEqual(a["nll"].mean(), 5.)
        np.testing.assert_array_equal(result["round_counts"], [2, 2])
        np.testing.assert_array_equal(result["cutoff_counts"], [3, 5])

    def test_paired_bootstrap_keeps_negative_comparison(self):
        rng = np.random.default_rng(3)
        draws = rng.integers(0, 3, size=(10000, 3))
        full = np.array([3., 2., 4.])
        better_baseline = full - .2
        result = report._comparison(full, better_baseline, draws)
        self.assertAlmostEqual(result["difference"], -.2)
        self.assertLess(result["ci_high"], 0)
        self.assertEqual(result["favorable_series"], 0)

    def test_full_delivery_and_rejection_paths(self):
        # Explicitly verify the eventual recursive-cleanup target remains within
        # the intended test workspace before TemporaryDirectory can remove it.
        workspace = Path(__file__).resolve().parent.parent / "work" / "report_tests"
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="report-fixture-", dir=workspace) as temporary:
            root = Path(temporary).resolve()
            self.assertIn(workspace.resolve(), root.parents)
            output, references = root / "output", root / "references"
            order = report.DEFAULT_MODELS + report.REFERENCES
            for cohort in report.COHORTS:
                for model_number, model in enumerate(order):
                    for seed_number, seed in enumerate((2026, 2027, 2028)):
                        # One architecture intentionally beats Full; a report
                        # must retain it and highlight the observed winner.
                        delta = .1 if model == "MLP" else (.3 if model == "FULL" else .5+.01*model_number)
                        seed_delta = 0. if model == "TREE" else .002*seed_number
                        path = ((output / "predictions" / cohort) if model in report.DEFAULT_MODELS else (references / cohort)) / f"{model}_seed{seed}.npz"
                        save_fixture(path, fixture(delta, seed_delta), model, seed)
            result = report.generate_report(output, references)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["sources"]), 48)
            self.assertEqual(result["cohorts"]["rio"]["methods"]["TREE"]["training_seed_count"], 1)
            self.assertLess(result["cohorts"]["rio"]["comparisons"]["MLP"]["joint_nll"]["difference"], 0)
            for filename in ("summary.json", "per_series.csv", "comparison.csv", "external_baselines.tex", "report.md", "report.html"):
                self.assertGreater((output / filename).stat().st_size, 10)
            self.assertEqual(len(list((output / "figures").glob("*.pdf"))), 3)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 3)
            table = (output / "external_baselines.tex").read_text(encoding="utf-8")
            self.assertIn("MLP-Joint & \\textbf{", table)
            self.assertNotIn("Full model & \\textbf{", table)

            sample = output / "predictions" / "dev" / "GRU_seed2026.npz"
            valid = fixture(.51)
            invalid = {key: value.copy() for key, value in valid.items()}
            invalid["time_mass"][0, 0] += .1
            save_fixture(sample, invalid, "GRU", 2026)
            with self.assertRaisesRegex(report.PredictionError, "not normalized"):
                report.generate_report(output, references)
            self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"], "failed")
            save_fixture(sample, valid, "GRU", 2026)

            invalid = {key: value.copy() for key, value in valid.items()}
            invalid["time"][0] += .25
            save_fixture(sample, invalid, "GRU", 2026)
            with self.assertRaisesRegex(report.PredictionError, "alignment mismatch in time"):
                report.generate_report(output, references)
            save_fixture(sample, valid, "GRU", 2026)

            invalid = {key: value.copy() for key, value in valid.items()}
            invalid["y"][0] = 42
            save_fixture(sample, invalid, "GRU", 2026)
            with self.assertRaisesRegex(report.PredictionError, "flat y disagrees"):
                report.generate_report(output, references)
            save_fixture(sample, valid, "GRU", 2026)

            sample.unlink()
            with self.assertRaisesRegex(report.PredictionError, "Missing 1 required"):
                report.generate_report(output, references)


if __name__ == "__main__":
    unittest.main()
