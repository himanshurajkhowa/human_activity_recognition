"""Draw the saved D60 confusion matrix without loading participant records."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ACTIVITIES = ("Walk", "Run", "Sit", "Stand", "Lie", "Upstairs", "Downstairs")


def plot_confusion(destination):
    counts = np.loadtxt(ROOT / "results/d60_confusion.csv", delimiter=",", dtype=int)
    percent = 100 * counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.1), layout="constrained")
    image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(7), ACTIVITIES, rotation=32, ha="right")
    ax.set_yticks(range(7), ACTIVITIES)
    ax.set(xlabel="Predicted activity", ylabel="Recorded activity", title="D60 development predictions")
    for row, group in enumerate(percent):
        for column, value in enumerate(group):
            ax.text(column, row, f"{value:.0f}%", ha="center", va="center",
                    color="white" if value >= 58 else "#172534")
    fig.colorbar(image, ax=ax, label="Share of recorded activity (%)", shrink=0.8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    destination = ROOT / "outputs" / "d60_confusion.png"
    plot_confusion(destination)
    print("Saved outputs/d60_confusion.png")
