#!/usr/bin/env python3
"""Plot the audited Stage 06 data. Reads CSV files; never runs robot policies.

Requires matplotlib. For Chinese labels, pass --font /path/to/CJK-font.ttc.
Without --font, uses English labels and the built-in DejaVu Sans font.
Default input: ../results; default output: ../media, relative to this script.
Writes only stage06-bc-vs-bc-rnn.png and .svg in the output directory.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    out = args.output_dir or Path(__file__).resolve().parent.parent / "media"
    out.mkdir(parents=True, exist_ok=True)
    zh = args.font is not None
    family = "DejaVu Sans"
    if zh:
        font_manager.fontManager.addfont(str(args.font))
        family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 11,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "svg.hashsalt": "lift-ph-stage06", "axes.titleweight": "bold",
        "axes.labelcolor": "#334155", "text.color": "#172338",
        "xtick.color": "#526174", "ytick.color": "#526174",
    })
    def label(cn, en):
        return cn if zh else en

    with (args.data_dir / "stage06-training-rollouts.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (args.data_dir / "stage06-independent-evaluation.csv").open(newline="", encoding="utf-8") as f:
        evaluation = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(50, 2001, 50)):
        raise ValueError("Expected exactly 40 ordered checkpoints: 50, 100, ..., 2000")
    series = [[float(row[key]) for row in rows] for key in ("bc_success_rate", "bc_rnn_success_rate")]
    names = ["State BC", "State BC-RNN"]
    colors = ["#2867B2", "#C65A22"]
    markers = ["o", "s"]
    if [r["method"] for r in evaluation] != names:
        raise ValueError("Unexpected evaluation methods or ordering")
    for name, rates, summary in zip(names, series, evaluation):
        if not all(0 <= rate <= 1 and abs(rate * 50 - round(rate * 50)) < 1e-8 for rate in rates):
            raise ValueError(f"{name}: invalid rates for 50 rollouts per checkpoint")
        selected = next(epoch for epoch, rate in zip(epochs, rates) if rate == max(rates))
        if selected != int(summary["selected_epoch"]):
            raise ValueError(f"{name}: selected epoch disagrees with the fixed selection rule")
        if abs(float(summary["success_rate"]) * int(summary["n_rollouts"]) - int(summary["num_success"])) > 1e-8:
            raise ValueError(f"{name}: inconsistent independent success count")

    fig = plt.figure(figsize=(16, 10.6), facecolor="#FAFCFF")
    grid = fig.add_gridspec(2, 2, left=.075, right=.97, top=.83, bottom=.20,
                           height_ratios=[1.28, 1], width_ratios=[1.62, 1],
                           hspace=.49, wspace=.26)
    full = fig.add_subplot(grid[0, :])
    zoom = fig.add_subplot(grid[1, 0])
    independent = fig.add_subplot(grid[1, 1])
    fig.text(.075, .959, "ROBOMIMIC  /  LIFT-PH  /  STAGE 06", color="#526174", fontsize=11)
    fig.text(.075, .912, label("State BC 与 State BC-RNN：训练及独立评估", "State BC vs State BC-RNN: training and independent evaluation"),
             fontsize=23, weight="bold")
    fig.text(.075, .874, label("每种方法仅一个训练种子；保留原始点，不平滑，不绘制跨种子误差带。",
                             "One training seed per method; raw points, no smoothing, no across-seed error bands."),
             fontsize=12, color="#526174")

    for ax in (full, zoom, independent):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E2E8F0", linewidth=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#CAD4E0")
        ax.tick_params(length=3)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))

    for name, values, color, marker in zip(names, series, colors, markers):
        pct = [100 * value for value in values]
        full.plot(epochs, pct, color=color, marker=marker, markersize=4.5,
                  markerfacecolor="white", markeredgewidth=1.15, linewidth=1.7, label=name)
        late = [(epoch, value) for epoch, value in zip(epochs, pct) if epoch >= 500]
        zoom.plot([p[0] for p in late], [p[1] for p in late], color=color,
                  marker=marker, markersize=4.2, markerfacecolor="white", linewidth=1.6)
    full.set_title(label("A  完整训练曲线 · 每点 50 次 rollout", "A  Full training curves | 50 rollouts per point"), loc="left", pad=15, fontsize=13)
    full.legend(loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.18))
    full.set_xlim(0, 2050)
    full.set_ylim(0, 107)
    full.set_yticks([0, 20, 40, 60, 80, 100])
    full.set_xticks(range(0, 2001, 250))
    full.set_xlabel(label("训练 epoch（每个 epoch 执行 100 次参数更新）", "Training epoch (100 gradient updates per epoch)"))
    full.set_ylabel(label("样本成功率", "Sample success rate"))
    for epoch, name, color, location in [(600, names[0], colors[0], (1000, 63)),
                                        (300, names[1], colors[1], (390, 36))]:
        full.scatter([epoch], [100], s=130, facecolors="none", edgecolors=color, linewidths=1.8, zorder=6)
        caption = label(f"{name} 选定 epoch {epoch}", f"{name}: selected epoch {epoch}")
        full.annotate(caption, (epoch, 100), xytext=location, color=color, fontsize=11,
                      bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "none", "alpha": .95},
                      arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1,
                                  "connectionstyle": "arc3,rad=-.10"})

    zoom.set_title(label("B  局部放大 · epoch 500–2000 · 纵轴从 84% 起", "B  Detail from epoch 500 | truncated y-axis starts at 84%"), loc="left", pad=16, fontsize=12)
    zoom.set_xlim(480, 2020)
    zoom.set_ylim(84, 102)
    zoom.set_yticks([84, 88, 92, 96, 100])
    zoom.set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000])
    zoom.set_xlabel(label("训练 epoch", "Training epoch"))
    zoom.set_ylabel(label("样本成功率", "Sample success rate"))

    independent.set_title(label("C  冻结后的独立评估 · 每种 100 次", "C  Independent evaluation | 100 rollouts each"), loc="left", pad=16, fontsize=12)
    rates = [float(row["success_rate"]) * 100 for row in evaluation]
    independent.bar([0, 1], rates, width=.5, color=colors, alpha=.93)
    for x, rate, summary in zip([0, 1], rates, evaluation):
        independent.text(x, rate + 3, f"{summary['num_success']}/{summary['n_rollouts']}",
                         ha="center", va="bottom", fontsize=16, weight="bold")
    independent.set_ylim(0, 116)
    independent.set_yticks([0, 25, 50, 75, 100])
    independent.set_xlim(-.7, 1.7)
    independent.set_xticks([0, 1], [f"{r['method']}\nepoch {r['selected_epoch']}" for r in evaluation])
    independent.set_ylabel(label("样本成功率", "Sample success rate"))

    footnotes = [
        label("训练：两种方法均为 seed 1、2000 epoch、200000 次参数更新；检查点按最高训练期成功率、并列取最早选择。",
              "Training: seed 1, 2000 epochs and 200000 updates each; select highest training success, then earliest epoch."),
        label("独立评估：seed 20260915；每种 100 次；horizon 上限 400；相同 seed 不保证逐条初始状态严格配对。",
              "Independent evaluation: seed 20260915, 100 rollouts each, horizon limit 400; same seed does not establish paired starts."),
        label("解读边界：99/100 与 100/100 相差一次成功，不能据此声称统计显著优势；不代表跨训练种子的稳定性。",
              "Interpretation: a one-success difference does not establish statistical superiority or robustness across training seeds."),
        label("数据来源：用户提供并通过检查的 Stage06ResultExtraction 终端输出；平均执行步数为 BC 49.10、BC-RNN 42.82（含失败）。",
              "Source: user-supplied, checked Stage06ResultExtraction output. Mean episode steps: BC 49.10, BC-RNN 42.82 (failures included)."),
    ]
    for y, text in zip([.129, .100, .071, .042], footnotes):
        fig.text(.075, y, text, fontsize=10.2, color="#526174")

    target = out / "stage06-bc-vs-bc-rnn"
    fig.savefig(target.with_suffix(".png"), dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(target.with_suffix(".svg"), facecolor=fig.get_facecolor(), metadata={"Date": None})
    # Strip trailing ASCII whitespace emitted by the SVG backend.
    # Newlines remain separators inside SVG path attributes.
    svg_path = target.with_suffix(".svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip(" \t") for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print("FigureDataValidation=PASS")
    print("TrainingCheckpointsPerMethod=40")
    print("SelectedEpochs=BC:600;BCRNN:300")
    print("PlotOutputs=" + str(target.with_suffix(".png")) + ";" + str(target.with_suffix(".svg")))


if __name__ == "__main__":
    main()
