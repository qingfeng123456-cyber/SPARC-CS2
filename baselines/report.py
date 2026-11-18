"""Strict, CPU-only comparison reports from already saved CS2 predictions.

No model fitting, prediction, data parsing, or network access occurs here.
Run from a returned experiment directory with ``python report.py --help``.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


COHORTS = ("dev", "rio")
REFERENCES = ("FULL", "TREE", "NEURAL", "BASE_MIX")
DEFAULT_MODELS = ("MLP", "GRU", "TCN", "TRANSFORMER")
METRICS = ("joint_nll", "time_nll", "mark_nll", "brier_a10", "brier_b10", "brier_ab10")
ALIGN_FIELDS = ("series_id", "round_id", "bins", "marks", "time", "cutoff_index", "index", "y")
REQUIRED_FIELDS = ALIGN_FIELDS + ("nll", "time_nll", "probability_a_b_10s", "time_mass", "tail")
DISPLAY = {
    "MLP": "MLP-Joint", "GRU": "GRU-Joint", "TCN": "TCN-Joint",
    "TRANSFORMER": "Transformer-Joint", "FULL": "SPARC-CS2",
    "TREE": "Tree component", "NEURAL": "Neural component", "BASE_MIX": "Base mixture",
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_261_117


class PredictionError(ValueError):
    """An incomplete or inconsistent saved prediction is not scorable."""


def _name(model: str) -> str:
    return DISPLAY.get(model, model)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PredictionError(f"No rows to write: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, stream.getvalue())


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_vector(value: np.ndarray, name: str, n: int, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (n,) or array.dtype.kind not in "iuf":
        raise PredictionError(f"{path}: {name} must be a numeric ({n},) vector")
    if not np.isfinite(array).all() or np.any(array != np.floor(array)):
        raise PredictionError(f"{path}: non-finite or non-integer {name}")
    return array.astype(np.int64)


def _load_prediction(path: Path, model: str, seed: int, metadata_required: bool) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_FIELDS) - set(archive.files))
        if missing:
            raise PredictionError(f"{path}: missing arrays {missing}")
        array = {key: archive[key].copy() for key in REQUIRED_FIELDS}
    if np.asarray(array["bins"]).ndim != 1 or len(array["bins"]) == 0:
        raise PredictionError(f"{path}: bins must be a nonempty vector")
    n = len(array["bins"])
    for key in ("series_id", "round_id"):
        value = np.asarray(array[key])
        if value.shape != (n,) or value.dtype.kind not in "USiu":
            raise PredictionError(f"{path}: {key} must contain one scalar ID per cutoff")
        array[key] = value.astype(str)
        if np.any(array[key] == ""):
            raise PredictionError(f"{path}: empty {key}")
    for key in ("bins", "marks", "cutoff_index", "index", "y"):
        array[key] = _integer_vector(array[key], key, n, path)
    if np.any((array["bins"] < 0) | (array["bins"] > 20)):
        raise PredictionError(f"{path}: unknown or out-of-range target bins")
    event = array["bins"] < 20
    if np.any((array["marks"][event] < 1) | (array["marks"][event] > 511)):
        raise PredictionError(f"{path}: event targets require nonempty masks 1..511")
    expected_y = np.where(event, array["bins"] * 511 + array["marks"] - 1, 10_220)
    if not np.array_equal(array["y"], expected_y):
        raise PredictionError(f"{path}: flat y disagrees with bins/marks/tail")
    if np.any(array["cutoff_index"] < 0) or np.any(array["index"] < 0):
        raise PredictionError(f"{path}: negative cutoff/index")
    for key, shape in (
        ("time", (n,)), ("nll", (n,)), ("time_nll", (n,)),
        ("tail", (n,)), ("time_mass", (n, 20)), ("probability_a_b_10s", (n, 2)),
    ):
        value = np.asarray(array[key])
        if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise PredictionError(f"{path}: {key} must be finite numeric array of shape {shape}")
        array[key] = value.astype(np.float64)
    if np.any(array["time"] < 0):
        raise PredictionError(f"{path}: negative game time")
    for key in ("tail", "time_mass", "probability_a_b_10s"):
        if np.any(array[key] < -2e-7) or np.any(array[key] > 1 + 2e-7):
            raise PredictionError(f"{path}: {key} outside probability range")
    mass = array["time_mass"].sum(axis=1) + array["tail"]
    if not np.allclose(mass, 1., atol=5e-5, rtol=0):
        raise PredictionError(f"{path}: temporal law is not normalized; max error={np.max(abs(mass - 1)):.3g}")
    if np.any(array["probability_a_b_10s"] > array["time_mass"][:, :10].sum(1)[:, None] + 5e-5):
        raise PredictionError(f"{path}: ten-second ingress mass exceeds the first-event time mass")
    if np.any(array["nll"] < -2e-6) or np.any(array["time_nll"] < -2e-6):
        raise PredictionError(f"{path}: a negative log loss is negative")
    if np.any(array["nll"] + 2e-4 < array["time_nll"]):
        raise PredictionError(f"{path}: joint log loss is below its temporal marginal log loss")
    chosen_time = np.where(
        event, array["time_mass"][np.arange(n), np.minimum(array["bins"], 19)], array["tail"]
    )
    # Extremely small serialized float32 probabilities can underflow; log scores
    # remain the authoritative saved values for those rows, without clipping them.
    check = chosen_time > 1e-28
    if not np.allclose(array["time_nll"][check], -np.log(chosen_time[check]), atol=2e-4, rtol=2e-5):
        raise PredictionError(f"{path}: time_nll does not score the saved temporal law")
    if not np.allclose(array["nll"][~event], array["time_nll"][~event], atol=2e-4, rtol=2e-5):
        raise PredictionError(f"{path}: joint and temporal tail scores disagree")
    keys = np.rec.fromarrays(
        (array["series_id"], array["round_id"], array["cutoff_index"]),
        names=("series", "round", "cutoff"),
    )
    if len(np.unique(keys)) != n:
        raise PredictionError(f"{path}: duplicate series/round/cutoff identities")
    metadata_path = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict):
            raise PredictionError(f"{metadata_path}: expected an object")
    elif metadata_required:
        raise PredictionError(f"Missing prediction metadata: {metadata_path}")
    if metadata_required:
        for key in ("parameters", "inference_seconds", "prediction_examples"):
            if key not in metadata:
                raise PredictionError(f"{metadata_path}: missing {key}")
        if not isinstance(metadata["parameters"], int) or isinstance(metadata["parameters"], bool) or metadata["parameters"] <= 0:
            raise PredictionError(f"{metadata_path}: parameters must be a positive integer")
        seconds = metadata["inference_seconds"]
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not np.isfinite(seconds) or seconds < 0:
            raise PredictionError(f"{metadata_path}: invalid inference_seconds")
        if metadata["prediction_examples"] != n:
            raise PredictionError(f"{metadata_path}: prediction_examples differs from the NPZ")
    if "seed" in metadata and metadata["seed"] != seed:
        raise PredictionError(f"{metadata_path}: seed differs from its filename")
    if "model" in metadata and str(metadata["model"]).upper() != model:
        raise PredictionError(f"{metadata_path}: model differs from its filename")
    metadata["time_score_rows_below_serialization_check"] = int((~check).sum())
    return array, metadata


def _assert_aligned(reference: dict, candidate: dict, path: Path) -> None:
    for key in ALIGN_FIELDS:
        if not np.array_equal(reference[key], candidate[key]):
            raise PredictionError(f"{path}: prediction alignment mismatch in {key}")


def _aggregate(array: dict) -> dict:
    """Equal cutoffs within round, equal rounds within series, equal series."""
    target = (array["bins"][:, None] < 10) & ((array["marks"][:, None] & [1, 2]) != 0)
    brier = np.square(array["probability_a_b_10s"] - target)
    values = np.column_stack((
        array["nll"], array["time_nll"], array["nll"] - array["time_nll"],
        brier[:, 0], brier[:, 1], brier.mean(1),
    ))
    keys = np.rec.fromarrays((array["series_id"], array["round_id"]), names=("series", "round"))
    rounds, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(rounds))
    round_values = np.column_stack([
        np.bincount(inverse, weights=values[:, i], minlength=len(rounds)) / counts
        for i in range(len(METRICS))
    ])
    series, series_inverse = np.unique(rounds["series"], return_inverse=True)
    round_counts = np.bincount(series_inverse, minlength=len(series))
    series_values = np.column_stack([
        np.bincount(series_inverse, weights=round_values[:, i], minlength=len(series)) / round_counts
        for i in range(len(METRICS))
    ])
    cutoff_counts = np.bincount(series_inverse, weights=counts, minlength=len(series)).astype(int)
    return {
        "series": series.astype(str), "values": series_values,
        "round_counts": round_counts, "cutoff_counts": cutoff_counts,
        "cutoffs": len(values), "rounds": len(rounds),
    }


def _comparison(full: np.ndarray, reference: np.ndarray, draws: np.ndarray) -> dict:
    difference = reference - full
    boot_difference = difference[draws].mean(1)
    lo, hi = np.quantile(boot_difference, [.025, .975])
    baseline_mean = float(reference.mean())
    relative = 100 * float(difference.mean()) / baseline_mean if baseline_mean > 1e-15 else None
    denominator = reference[draws].mean(1)
    if np.all(denominator > 1e-15):
        relative_lo, relative_hi = np.quantile(100 * boot_difference / denominator, [.025, .975]).tolist()
    else:
        relative_lo = relative_hi = None
    return {
        "full_score": float(full.mean()), "reference_score": baseline_mean,
        "difference": float(difference.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "relative_percent": relative, "relative_ci_low": relative_lo, "relative_ci_high": relative_hi,
        "favorable_series": int((difference > 0).sum()), "series": len(difference),
    }


def _latex_escape(text: str) -> str:
    return "".join({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}.get(c, c) for c in text)


def _write_latex(path: Path, summary: dict, order: tuple[str, ...], models: tuple[str, ...]) -> None:
    scores = summary["cohorts"]
    # Highlight the lowest observed score, with no preferred architecture.
    best = {cohort: min(scores[cohort]["methods"][m]["mean"]["joint_nll"] for m in order) for cohort in COHORTS}
    rows = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Architecture comparisons: series-macro joint negative log likelihood (nats; lower is better). "
        r"Neural methods average the fitted seed scores; the deterministic tree is shared.}",
        r"\label{tab:architecture-baselines}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrr}", r"\toprule", r"Predictor & Development & Rio \\", r"\midrule",
        r"\multicolumn{3}{l}{\emph{Task-adapted architecture baselines}}\\",
    ]
    for model in order:
        if model == "NEURAL":
            rows += [r"\midrule", r"\multicolumn{3}{l}{\emph{Frozen component references}}\\"]
        if model == "FULL":
            rows.append(r"\midrule")
        cells = []
        for cohort in COHORTS:
            value = scores[cohort]["methods"][model]["mean"]["joint_nll"]
            cell = f"{value:.5f}"
            cells.append(r"\textbf{" + cell + "}" if abs(value - best[cohort]) < 5e-10 else cell)
        rows.append(_latex_escape(_name(model)) + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_text(path, "\n".join(rows) + "\n")


def _figures(output: Path, summary: dict, order: tuple[str, ...], models: tuple[str, ...]) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "mathtext.fontset": "stix", "axes.titlepad": 10,
    })
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    comparators = tuple(model for model in order if model != "FULL")
    labels = [_name(model) for model in comparators]
    figures: list[dict] = []

    def save(fig, stem: str, caption: str) -> None:
        for extension in ("pdf", "png"):
            fig.savefig(folder / f"{stem}.{extension}", bbox_inches="tight", pad_inches=.06)
        plt.close(fig)
        figures.append({"stem": stem, "caption": caption, "source": "summary.json and comparison.csv"})

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True, constrained_layout=True)
    all_intervals = [
        summary["cohorts"][c]["comparisons"][m]["joint_nll"][k]
        for c in COHORTS for m in comparators for k in ("ci_low", "ci_high")
    ]
    low, high = min(0., min(all_intervals)), max(0., max(all_intervals))
    span = max(high - low, .01)
    for axis, cohort, panel in zip(axes, COHORTS, ("a", "b")):
        axis.axvline(0, color="#596273", linewidth=.9, linestyle=(0, (3, 3)))
        for y, model in enumerate(comparators):
            row = summary["cohorts"][cohort]["comparisons"][model]["joint_nll"]
            color = "#0072B2" if model in models else "#7A668F"
            crossing = row["ci_low"] <= 0 <= row["ci_high"]
            # Draw the percentile interval independently of its point estimate:
            # a skewed bootstrap need not contain the original point estimate.
            axis.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, linewidth=1.2, zorder=3)
            axis.vlines([row["ci_low"], row["ci_high"]], y-.07, y+.07, color=color, linewidth=1.2, zorder=3)
            axis.plot(row["difference"], y, color=color, marker="o", markersize=5.2,
                      markerfacecolor="white" if crossing else color, markeredgewidth=1.1, zorder=4)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_ylim(len(labels)-.5, -.5)
        axis.set_xlim(low - .08*span, high + .08*span)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.grid(axis="x", color="#E7EAF0", linewidth=.7)
        axis.set_axisbelow(True)
        title = "Development" if cohort == "dev" else "Rio"
        axis.set_title(f"({panel}) {title}: {summary['cohorts'][cohort]['series']} series", loc="left")
        axis.set_xlabel("Reference NLL − Full NLL (nats)")
    save(fig, "baseline_joint_forest",
         "Paired series-macro joint-loss differences. Positive values favor Full. "
         "Intervals are percentile 95% intervals from 10,000 paired series bootstrap draws after averaging seed differences; "
         "development and Rio are resampled separately. Open markers identify intervals containing zero. "
         "Blue denotes task-adapted architecture baselines; purple denotes frozen component references. Every comparison is shown.")

    heatmaps = {}
    for cohort in COHORTS:
        cohort_data = summary["cohorts"][cohort]
        full = cohort_data["methods"]["FULL"]["per_series"]
        sids = cohort_data["series_ids"]
        heatmaps[cohort] = np.array([
            [cohort_data["methods"][m]["per_series"][sid]["joint_nll"] - full[sid]["joint_nll"] for sid in sids]
            for m in comparators
        ])
    bound = max(1e-6, max(float(np.max(np.abs(v))) for v in heatmaps.values()))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True, constrained_layout=True)
    for axis, cohort, panel in zip(axes, COHORTS, ("a", "b")):
        matrix = heatmaps[cohort]
        heat = axis.imshow(matrix, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound), interpolation="nearest")
        sids = summary["cohorts"][cohort]["series_ids"]
        axis.set_xticks(np.arange(len(sids)), sids, rotation=65, ha="right", rotation_mode="anchor")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xticks(np.arange(len(sids)+1)-.5, minor=True)
        axis.set_yticks(np.arange(len(labels)+1)-.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=.8)
        axis.tick_params(which="both", length=0)
        axis.set_xlabel("HLTV series ID")
        title = "Development" if cohort == "dev" else "Rio"
        axis.set_title(f"({panel}) {title}", loc="left")
        for y in range(len(labels)):
            for x in range(len(sids)):
                value = matrix[y, x]
                axis.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=6.2,
                          color="white" if abs(value) > .58*bound else "#16213A")
    colorbar = fig.colorbar(heat, ax=axes, fraction=.025, pad=.025, shrink=.89)
    colorbar.set_label("Reference − Full joint NLL (nats)")
    save(fig, "baseline_series_heatmap",
         "Series-level joint-loss differences after averaging the fitted seed scores. "
         "Each column is one held-out HLTV series. Positive (blue) cells favor Full and negative (red) cells favor the reference. "
         "Both cohorts use the same symmetric, unclipped color scale; all series and all comparators are retained.")

    colors = {"MLP": "#0072B2", "GRU": "#E69F00", "TCN": "#009E73", "TRANSFORMER": "#CC79A7",
              "NEURAL": "#858C96", "TREE": "#53616C", "BASE_MIX": "#292D36", "FULL": "#11233F"}
    markers = {"MLP": "o", "GRU": "^", "TCN": "D", "TRANSFORMER": "s", "NEURAL": "x", "TREE": "P", "BASE_MIX": "o", "FULL": "*"}
    points = [
        (summary["cohorts"][cohort]["methods"][model]["mean"]["time_nll"], summary["cohorts"][cohort]["methods"][model]["mean"]["mark_nll"])
        for cohort in COHORTS for model in order
    ]
    xvalues, yvalues = np.array(points).T
    xrange = max(float(np.ptp(xvalues)), .03)
    yrange = max(float(np.ptp(yvalues)), .03)
    xlim = (float(xvalues.min())-.1*xrange, float(xvalues.max())+.1*xrange)
    ylim = (max(0., float(yvalues.min())-.12*yrange), float(yvalues.max())+.12*yrange)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1), sharex=True, sharey=True, constrained_layout=True)
    legend_handles, legend_labels = [], []
    for axis, cohort, panel in zip(axes, COHORTS, ("a", "b")):
        for total in np.linspace(xlim[0]+ylim[0], xlim[1]+ylim[1], 7)[1:-1]:
            grid = np.linspace(*xlim, 100)
            axis.plot(grid, total-grid, color="#DCE1E8", linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        for model in order:
            mean = summary["cohorts"][cohort]["methods"][model]["mean"]
            kwargs: dict[str, Any] = dict(s=45, color=colors.get(model, "#0072B2"), marker=markers.get(model, "o"), zorder=3)
            if model == "BASE_MIX":
                kwargs.update(facecolors="white", edgecolors=colors[model], linewidths=1.1)
                kwargs.pop("color")
            if model == "FULL":
                kwargs.update(s=150, zorder=5, edgecolors="white", linewidths=.7)
            handle = axis.scatter(mean["time_nll"], mean["mark_nll"], **kwargs)
            if cohort == "dev":
                legend_handles.append(handle)
                legend_labels.append(_name(model))
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.set_xlabel("Temporal NLL (nats)")
        title = "Development" if cohort == "dev" else "Rio"
        axis.set_title(f"({panel}) {title}", loc="left")
        axis.text(.03, .04, "Lower left: lower joint loss", transform=axis.transAxes, fontsize=7.5, color="#596273")
    axes[0].set_ylabel("Mark contribution: joint − temporal NLL (nats)")
    fig.legend(legend_handles, legend_labels, loc="outside lower center", ncol=4, frameon=False, columnspacing=1.6, handletextpad=.5)
    save(fig, "baseline_score_plane",
         "Temporal and mark contributions to joint log loss, using the same cutoff–round–series aggregation and mean of seed scores. "
         "The vertical contribution equals joint minus temporal NLL, including zero contribution at no-event-tail targets. "
         "Dashed diagonals represent constant joint loss. Points are score means, not confidence regions or causal effects.")
    _write_json(folder / "figure_manifest.json", {"schema": "baseline-report-figures-v1", "figures": figures})
    includes = []
    for figure in figures:
        includes += [
            r"\begin{figure*}[t]", r"\centering",
            r"\includegraphics[width=\textwidth]{figures/" + figure["stem"] + ".pdf}",
            r"\caption{" + _latex_escape(figure["caption"]) + "}",
            r"\label{fig:" + figure["stem"].replace("_", "-") + "}", r"\end{figure*}", "",
        ]
    _write_text(folder / "latex_includes.tex", "\n".join(includes))
    return figures


def _human_reports(output: Path, summary: dict, order: tuple[str, ...], figures: list[dict]) -> None:
    lines = [
        "# CS2 基线比较回传报告", "", "状态：**完整预测已核验，比较与图表已生成。**", "",
        "所有规定模型、种子和两个队列均已读取；目标、比赛、回合、时点和索引逐项对齐，概率与日志损失通过数值核验。", "",
        f"主指标先平均同回合预测，再平均同系列赛回合，最后对系列赛等权；{len(summary['seeds'])}个神经训练种子的分数取均值。", "",
        "| 模型 | 开发集 joint NLL | Rio joint NLL | 独立拟合数 |",
        "|---|---:|---:|---:|",
    ]
    for model in order:
        dev = summary["cohorts"]["dev"]["methods"][model]
        rio = summary["cohorts"]["rio"]["methods"][model]
        lines.append(f"| {_name(model)} | {dev['mean']['joint_nll']:.6f} | {rio['mean']['joint_nll']:.6f} | {dev['training_seed_count']} |")
    lines += [
        "", "TREE 为同一个确定性拟合，保存的三个参考文件仅用于与神经种子对齐；它不代表三次独立训练。", "",
        "## Full 与每个参照的配对比较", "",
        "差值 = 参照损失 − Full 损失；正值有利于 Full。区间在种子差值平均后，以完整系列赛为单位做10,000次配对重采样。两个队列分别计算。", "",
        "| 队列 | 参照 | joint NLL 差值 | 95% 区间 | 有利系列赛 / 总数 |",
        "|---|---|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        for model, values in summary["cohorts"][cohort]["comparisons"].items():
            row = values["joint_nll"]
            lines.append(f"| {cohort} | {_name(model)} | {row['difference']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['favorable_series']} / {row['series']} |")
    lines += ["", "## 预测数量与运行记录", ""]
    for cohort in COHORTS:
        values = summary["cohorts"][cohort]
        lines.append(f"- {cohort}：{values['series']} 场系列赛，{values['rounds']} 回合，{values['cutoffs']} 个预测时点。")
    lines += ["", "| 新基线 | 队列 | 参数量 | 平均推理秒数 | 平均每时点毫秒 |", "|---|---|---:|---:|---:|"]
    for cohort in COHORTS:
        for model in summary["models"]:
            values = summary["cohorts"][cohort]["methods"][model]
            lines.append(f"| {_name(model)} | {cohort} | {values['parameters']:,} | {values['inference_seconds_mean']:.3f} | {values['inference_ms_per_prediction_mean']:.5f} |")
    lines += ["", "推理用时取各种子旁侧记录的平均值，运行硬件和batch设置由原始预测元数据确定。", "", "## 图表", ""]
    for figure in figures:
        lines += [f"### {figure['stem']}", "", f"![{figure['stem']}](figures/{figure['stem']}.png)", "", figure["caption"], ""]
    lines += [
        "## 可复核文件", "",
        "- `summary.json`：完整配置、各模型/种子/系列赛分数、配对区间、输入文件校验。",
        "- `per_series.csv`：每个模型、种子、系列赛的五项原始指标及标记贡献。",
        "- `comparison.csv`：Full 与全部参照的逐队列配对比较，包含正负和跨零区间。",
        "- `external_baselines.tex`：真实结果表，按实际最小值突出，保留全部模型。",
        "- `figures/`：可直接使用的 PDF/300-dpi PNG、图注和 LaTeX 插入片段。", "",
        "A/B 十秒概率指首个未来事件包含对应入点标记；其 Brier 分数不等同于发生过中间事件后的任意入点概率。", "",
    ]
    _write_text(output / "report.md", "\n".join(lines))
    # A dependency-free HTML report keeps the returned directory easy to open.
    blocks = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>CS2 基线比较回传报告</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#18243a;background:#eef2f6;margin:0}main{max-width:1150px;margin:32px auto;padding:30px 38px;background:white;border-radius:14px}h1,h2{color:#11233f}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #dfe5ec;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f1f5f9}img{max-width:100%;height:auto}figure{margin:30px 0;padding:18px;border:1px solid #dfe5ec;border-radius:10px}figcaption{font-size:13px;color:#465268}.ok{color:#076b55;font-weight:700}a{color:#006cb7}code{background:#f1f5f9;padding:2px 5px}</style><main>',
        '<h1>CS2 基线比较回传报告</h1><p class="ok">完整预测已核验，比较与图表已生成。</p>',
        f'<p>主指标按预测时点→回合→系列赛逐层平均，神经模型再平均{len(summary["seeds"])}个独立训练种子的分数。TREE共享一个确定性拟合。</p>',
        '<h2>主指标</h2><table><thead><tr><th>模型</th><th>开发集 joint NLL</th><th>Rio joint NLL</th></tr></thead><tbody>',
    ]
    for model in order:
        d = summary["cohorts"]["dev"]["methods"][model]["mean"]["joint_nll"]
        r = summary["cohorts"]["rio"]["methods"][model]["mean"]["joint_nll"]
        blocks.append(f"<tr><td>{html.escape(_name(model))}</td><td>{d:.6f}</td><td>{r:.6f}</td></tr>")
    blocks += ["</tbody></table>", "<h2>配对比较</h2><p>差值为参照损失减Full损失，正值有利于Full。95%区间独立按各队列的完整系列赛重采样10,000次。</p>",
               "<table><thead><tr><th>队列 / 参照</th><th>joint NLL 差值</th><th>95% 区间</th><th>有利系列赛</th></tr></thead><tbody>"]
    for cohort in COHORTS:
        for model, values in summary["cohorts"][cohort]["comparisons"].items():
            v = values["joint_nll"]
            blocks.append(f"<tr><td>{cohort} / {html.escape(_name(model))}</td><td>{v['difference']:+.6f}</td><td>[{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]</td><td>{v['favorable_series']} / {v['series']}</td></tr>")
    blocks += ["</tbody></table>", "<h2>比较图</h2>"]
    for figure in figures:
        blocks.append(f'<figure><a href="figures/{figure["stem"]}.pdf"><img src="figures/{figure["stem"]}.png" alt="{figure["stem"]}"></a><figcaption>{html.escape(figure["caption"])}</figcaption></figure>')
    blocks += [
        '<h2>完整记录</h2><p><a href="report.md">文字报告</a> · <a href="summary.json">完整数值与文件校验</a> · <a href="per_series.csv">逐系列赛分数</a> · <a href="comparison.csv">全部配对比较</a> · <a href="external_baselines.tex">LaTeX结果表</a></p>',
        '<p>所有模型及比较均被保留，区间跨零和负向结果同样展示。图像由保存的预测与分组信息生成，无需重新训练。</p></main></html>',
    ]
    _write_text(output / "report.html", "\n".join(blocks))


def generate_report(
    output_dir: Path,
    reference_dir: Path,
    seeds=(2026, 2027, 2028),
    models=("MLP", "GRU", "TCN", "TRANSFORMER"),
) -> dict:
    """Validate the complete experiment and write reproducible result artifacts.

    New files are ``output_dir/predictions/{dev,rio}/MODEL_seedSEED.npz``;
    frozen files are ``reference_dir/{dev,rio}/REFERENCE_seedSEED.npz``.
    Every requested file is mandatory. The returned dictionary is the same
    object serialized to ``output_dir/summary.json`` only after successful QA.
    """
    output_dir, reference_dir = Path(output_dir).resolve(), Path(reference_dir).resolve()
    seeds = tuple(int(s) for s in seeds)
    models = tuple(str(m).upper() for m in models)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "report_status.json", {"status": "validating", "started_at_utc": stamp})
    try:
        if not seeds or len(set(seeds)) != len(seeds) or not models or len(set(models)) != len(models):
            raise PredictionError("Need nonempty, distinct seeds and models")
        if set(models) & set(REFERENCES):
            raise PredictionError("New-model names must differ from the frozen references")
        order = models + ("NEURAL", "TREE", "BASE_MIX", "FULL")
        source_paths = {
            (c, m, s): ((output_dir / "predictions" / c) if m in models else (reference_dir / c)) / f"{m}_seed{s}.npz"
            for c in COHORTS for m in order for s in seeds
        }
        missing = [str(p) for p in source_paths.values() if not p.is_file()]
        if missing:
            raise PredictionError(f"Missing {len(missing)} required prediction files:\n" + "\n".join(missing))
        summary: dict[str, Any] = {
            "schema": "cs2-baseline-comparison-v1", "status": "complete", "created_at_utc": stamp,
            "models": list(models), "seeds": list(seeds), "references": list(REFERENCES),
            "outcome_space": {"time_bins": 20, "nonempty_mark_sets": 511, "no_event_tail": True, "classes": 10_221},
            "aggregation": "mean cutoffs within each round; mean rounds within each series; equal series mean; mean neural-seed scores",
            "uncertainty": {"unit": "series", "paired": True, "seed_differences_averaged_first": True,
                            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                            "interval": "percentile 95%", "cohorts_resampled_separately": True,
                            "difference": "reference minus Full; positive favors Full"},
            "tree_seed_handling": "One shared deterministic fitted tree; seed-labelled copies are paired references, not independent training runs.",
            "cohorts": {}, "sources": [],
        }
        per_rows, comparison_rows = [], []
        for cohort_number, cohort in enumerate(COHORTS):
            anchor = None
            arrays, aggregated, metadata = {}, {}, {}
            for model in order:
                for seed in seeds:
                    path = source_paths[(cohort, model, seed)]
                    a, meta = _load_prediction(path, model, seed, metadata_required=model in models)
                    if anchor is None:
                        anchor = a
                    else:
                        _assert_aligned(anchor, a, path)
                    arrays[(model, seed)] = a
                    aggregated[(model, seed)] = _aggregate(a)
                    metadata[(model, seed)] = meta
                    summary["sources"].append({
                        "cohort": cohort, "model": model, "seed": seed,
                        "source_kind": "new_baseline" if model in models else "frozen_reference",
                        "file": f"{cohort}/{path.name}", "bytes": path.stat().st_size, "sha256": _digest(path),
                        "metadata": meta,
                    })
            assert anchor is not None
            tree = arrays[("TREE", seeds[0])]
            for seed in seeds[1:]:
                for field in ("nll", "time_nll", "time_mass", "tail", "probability_a_b_10s"):
                    if not np.array_equal(tree[field], arrays[("TREE", seed)][field]):
                        raise PredictionError(f"{cohort}: TREE seed-labelled references are not the same shared deterministic prediction")
            base = aggregated[("FULL", seeds[0])]
            series = base["series"].tolist()
            cohort_summary: dict[str, Any] = {
                "series": len(series), "series_ids": series, "rounds": int(base["rounds"]),
                "cutoffs": int(base["cutoffs"]), "methods": {}, "comparisons": {},
            }
            values_by_model = {}
            for model in order:
                seed_values = np.stack([aggregated[(model, s)]["values"] for s in seeds])
                values = seed_values.mean(0)
                values_by_model[model] = values
                means = values.mean(0)
                method: dict[str, Any] = {
                    "display_name": _name(model), "kind": "architecture_baseline" if model in models else "frozen_reference",
                    "training_seed_count": 1 if model == "TREE" else len(seeds),
                    "prediction_files": len(seeds), "mean": dict(zip(METRICS, means.tolist())),
                    "per_seed": {str(s): dict(zip(METRICS, seed_values[i].mean(0).tolist())) for i, s in enumerate(seeds)},
                    "per_series": {sid: dict(zip(METRICS, values[i].tolist())) for i, sid in enumerate(series)},
                }
                if model in models:
                    parameters = {metadata[(model, seed)]["parameters"] for seed in seeds}
                    if len(parameters) != 1:
                        raise PredictionError(f"{cohort}/{model}: architecture parameter count differs between seeds")
                    method["parameters"] = parameters.pop()
                    seconds = [float(metadata[(model, seed)]["inference_seconds"]) for seed in seeds]
                    method["inference_seconds_mean"] = float(np.mean(seconds))
                    method["inference_ms_per_prediction_mean"] = 1000 * float(np.mean(seconds)) / len(anchor["bins"])
                cohort_summary["methods"][model] = method
                for i, seed in enumerate(seeds):
                    for j, sid in enumerate(series):
                        per_rows.append({
                            "cohort": cohort, "model": model, "seed": seed, "series_id": sid,
                            "rounds": int(base["round_counts"][j]), "cutoffs": int(base["cutoff_counts"][j]),
                            **dict(zip(METRICS, seed_values[i, j].tolist())),
                        })
            rng = np.random.default_rng(BOOTSTRAP_SEED + cohort_number)
            draws = rng.integers(0, len(series), size=(BOOTSTRAP_DRAWS, len(series)))
            for model in order:
                if model == "FULL":
                    continue
                comparisons = {}
                for metric_index, metric in enumerate(METRICS):
                    row = _comparison(values_by_model["FULL"][:, metric_index], values_by_model[model][:, metric_index], draws)
                    comparisons[metric] = row
                    comparison_rows.append({"cohort": cohort, "reference": model, "metric": metric,
                                            "neural_seeds": len(seeds), **row})
                cohort_summary["comparisons"][model] = comparisons
            summary["cohorts"][cohort] = cohort_summary
        # The same selected architecture is evaluated in both cohorts.
        for model in models:
            if summary["cohorts"]["dev"]["methods"][model]["parameters"] != summary["cohorts"]["rio"]["methods"][model]["parameters"]:
                raise PredictionError(f"{model}: parameter count differs between development and Rio")
        _csv(output_dir / "per_series.csv", per_rows)
        _csv(output_dir / "comparison.csv", comparison_rows)
        _write_latex(output_dir / "external_baselines.tex", summary, order, models)
        figures = _figures(output_dir, summary, order, models)
        summary["figures"] = figures
        _human_reports(output_dir, summary, order, figures)
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "report_status.json", {
            "status": "complete", "created_at_utc": stamp, "validated_prediction_files": len(source_paths),
            "summary_sha256": _digest(output_dir / "summary.json"),
        })
        return summary
    except Exception as error:
        failed = {"schema": "cs2-baseline-comparison-v1", "status": "failed", "created_at_utc": stamp,
                  "error_type": type(error).__name__, "error": str(error)}
        _write_json(output_dir / "report_status.json", failed)
        _write_json(output_dir / "summary.json", failed)
        _write_text(output_dir / "report.md", "# CS2 基线比较回传报告\n\n状态：**未完成**。\n\n" + str(error) + "\n")
        _write_text(output_dir / "report.html", '<!doctype html><meta charset="utf-8"><h1>基线报告未完成</h1><pre>' + html.escape(str(error)) + "</pre>")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Experiment output directory containing predictions/")
    parser.add_argument("--references", required=True, type=Path, help="Frozen reference directory containing dev/ and rio/")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    summary = generate_report(args.output, args.references, tuple(args.seeds), tuple(args.models))
    print(f"Complete: {args.output.resolve() / 'report.html'}", flush=True)
    print(f"Validated {len(summary['sources'])} prediction files; three publication figures saved as PDF and PNG.", flush=True)


if __name__ == "__main__":
    main()
