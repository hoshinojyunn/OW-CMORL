from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test
from pymoo.indicators.hv import Hypervolume


RUNS = {
    "dynamic": PROJECT_ROOT / "results" / "smoke_dynamic2",
    "static": PROJECT_ROOT / "results" / "smoke_static",
    "no_dyn_ablation": PROJECT_ROOT / "results" / "smoke_ablate_no_dyn",
}

OUT_DIR = PROJECT_ROOT / "figures" / "smoke_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_front(path: Path) -> np.ndarray:
    return np.loadtxt(path / "final" / "objs.txt", delimiter=",")


def compute_metrics(front: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    hv = Hypervolume(ref_point=-ref).do(-front)
    prefs = generate_w_batch_test(front.shape[1], 0.5)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": float(eu), "SP": float(sp)}


def main() -> None:
    ref = np.zeros(3)
    rows = []
    fronts = {}
    for name, path in RUNS.items():
        front = load_front(path)
        fronts[name] = front
        metrics = compute_metrics(front, ref)
        metrics["run"] = name
        metrics["points"] = len(front)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "smoke_metrics.csv", index=False)

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    colors = {"dynamic": "tab:blue", "static": "tab:orange", "no_dyn_ablation": "tab:green"}
    for name, front in fronts.items():
        ax1.scatter(front[:, 0], front[:, 1], front[:, 2], s=30, alpha=0.85, label=name, color=colors[name])
    ax1.set_xlabel("Obj 1")
    ax1.set_ylabel("Obj 2")
    ax1.set_zlabel("Obj 3")
    ax1.set_title("Final Pareto Fronts")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(len(df))
    width = 0.25
    for idx, metric in enumerate(["HV", "EU", "SP"]):
        ax2.bar(x + (idx - 1) * width, df[metric].values, width=width, label=metric)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["run"].values, rotation=15)
    ax2.set_title("Smoke Comparison")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_comparison.png", dpi=200)
    plt.close(fig)

    print(df)


if __name__ == "__main__":
    main()
