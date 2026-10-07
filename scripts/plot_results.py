"""Render the released training curves from the original scalar metrics."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth(values, decay):
    total = mass = 0.0
    result = []
    for value in values:
        total = decay * total + (1 - decay) * value
        mass = decay * mass + (1 - decay)
        result.append(total / mass)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "results/train-loss.csv")
    parser.add_argument("--output", type=Path, default=root / "assets/training-loss")
    parser.add_argument("--smoothing", type=float, default=0.997)
    args = parser.parse_args()
    if not 0 <= args.smoothing < 1:
        parser.error("Smoothing must be in [0, 1).")
    with args.data.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.labelsize": 19, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 6.3), layout="constrained")
    colors = {"vanilla": "#D87728", "dual-slot": "#2463A8"}
    handles = []
    for method, color in colors.items():
        data = sorted((row for row in rows if row["method"] == method),
                      key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in data]
        values = [float(row["train_loss"]) for row in data]
        ax.plot(steps, values, color=color, alpha=0.07, linewidth=0.6)
        ax.plot(steps, smooth(values, args.smoothing), color=color,
                linewidth=3.5, solid_capstyle="round")
        handles.append(Line2D([], [], color=color, linewidth=4, label=method))
    ax.set(xlabel="Step", ylabel="Training loss", xlim=(0, 3000), ylim=(2.85, 5.15))
    ax.set_xticks([0, 600, 1200, 1800, 2400, 3000])
    ax.set_yticks([3.0, 3.5, 4.0, 4.5, 5.0])
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#A7AFB9")
    ax.tick_params(length=4, color="#A7AFB9", pad=8)
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=21, handlelength=2.6, labelspacing=0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        destination = args.output.with_suffix("." + extension)
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
