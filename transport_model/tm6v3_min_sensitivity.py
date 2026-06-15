"""
TM6v3-min: compact local SALib robustness pass
=============================================

Цель:
  - получить reviewer-safe ranking ключевых параметров TM6v3-min
  - не раздувать scope в большой uncertainty study

Метрика:
  - A* (steady-state formate concentration, mM)

Важно:
  - это локальный robustness pass вокруг manuscript nominal regime
  - диапазоны намеренно умеренные, чтобы не превращать анализ
    в глобальный stress-test за пределами physically intended window

Артефакты:
  - tmp/tm6v3_min_sensitivity_report.json
  - tmp/tm6v3_min_sobol.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from tm6v3_minimal_poc import TM6v3MinParams, find_steady_state


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "tmp"
OUT_DIR.mkdir(exist_ok=True)

def local_bounds(nominal: float, half_width_log10: float = 0.5):
    center = np.log10(nominal)
    return center - half_width_log10, center + half_width_log10


PARAM_DEFS = [
    ("k1", 1e-4, *local_bounds(1e-4, 0.5)),
    ("kd_A", 1e-4, *local_bounds(1e-4, 0.5)),
    ("km", 3e-2, *local_bounds(3e-2, 0.5)),
    ("kd_m", 3e-6, *local_bounds(3e-6, 0.5)),
    ("k_fe_gen", 5e-5, *local_bounds(5e-5, 0.5)),
    ("fe_supply", 5e-8, *local_bounds(5e-8, 0.5)),
    ("f1", 5e-3, *local_bounds(5e-3, 0.5)),
    ("Ka", 7e-4, *local_bounds(7e-4, 0.5)),
]


def make_problem():
    return {
        "num_vars": len(PARAM_DEFS),
        "names": [p[0] for p in PARAM_DEFS],
        "bounds": [[p[2], p[3]] for p in PARAM_DEFS],
    }


def params_from_vector(x):
    p = TM6v3MinParams()
    for i, (name, _, _, _) in enumerate(PARAM_DEFS):
        setattr(p, name, 10 ** x[i])
    return p


def evaluate_astar(x):
    p = params_from_vector(x)
    try:
        ss, _ = find_steady_state(p, t_end=300 * 3600)
        a_star = max(float(ss["a"]) * 1e3, 0.0)
        alive = a_star > 0.1
        return a_star, alive
    except Exception:
        return 0.0, False


def save_plot(ranked):
    labels = [r["name"] for r in ranked]
    st_vals = [r["ST"] for r in ranked]
    s1_vals = [r["S1"] for r in ranked]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(labels))
    ax.barh(y + 0.18, st_vals, height=0.35, label="ST")
    ax.barh(y - 0.18, s1_vals, height=0.35, label="S1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index")
    ax.set_title("TM6v3-min: Sobol sensitivity for A*")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "tm6v3_min_sobol_local.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    N = 256
    t0 = time.time()
    problem = make_problem()

    print(f"[1/4] Sobol sampling, N={N}")
    X = sobol_sample.sample(problem, N, calc_second_order=False)
    n_samples = X.shape[0]
    print(f"  samples = {n_samples}")

    print("[2/4] Model evaluations")
    Y_astar = np.zeros(n_samples)
    alive_flags = np.zeros(n_samples, dtype=bool)
    for i, x in enumerate(X):
        a_star, alive = evaluate_astar(x)
        Y_astar[i] = a_star
        alive_flags[i] = alive
        if (i + 1) % 250 == 0 or i == n_samples - 1:
            print(f"  {i + 1}/{n_samples}")

    print("[3/4] Sobol analysis")
    Si = sobol.analyze(problem, Y_astar, calc_second_order=False, print_to_console=False)

    ranked = []
    for i, name in enumerate(problem["names"]):
        ranked.append(
            {
                "name": name,
                "nominal": PARAM_DEFS[i][1],
                "S1": float(Si["S1"][i]) if np.isfinite(Si["S1"][i]) else 0.0,
                "ST": float(Si["ST"][i]) if np.isfinite(Si["ST"][i]) else 0.0,
                "S1_conf": float(Si["S1_conf"][i]) if np.isfinite(Si["S1_conf"][i]) else 0.0,
                "ST_conf": float(Si["ST_conf"][i]) if np.isfinite(Si["ST_conf"][i]) else 0.0,
            }
        )
    ranked.sort(key=lambda r: r["ST"], reverse=True)

    print("[4/4] Saving artifacts")
    plot_path = save_plot(ranked)
    report = {
        "model": "TM6v3_min",
        "method": "Sobol_local",
        "N": N,
        "num_samples": int(n_samples),
        "metric": "A_star_mM",
        "alive_threshold_mM": 0.1,
        "bounds": {
            "type": "log10_local",
            "half_width": 0.5,
            "approx_multiplier": 3.16,
        },
        "alive_fraction": float(np.mean(alive_flags)),
        "A_star_stats_mM": {
            "mean": float(np.mean(Y_astar)),
            "median": float(np.median(Y_astar)),
            "min": float(np.min(Y_astar)),
            "max": float(np.max(Y_astar)),
        },
        "ranked": ranked,
        "top_5": ranked[:5],
        "runtime_s": round(time.time() - t0, 2),
        "artifacts": {
            "plot": str(plot_path),
            "json": str(OUT_DIR / "tm6v3_min_sensitivity_local_report.json"),
        },
    }

    out_json = OUT_DIR / "tm6v3_min_sensitivity_local_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"alive_fraction = {report['alive_fraction']:.3f}")
    for row in ranked[:5]:
        print(f"{row['name']}: ST={row['ST']:.3f}, S1={row['S1']:.3f}")
    print(f"JSON: {out_json}")
    print(f"PNG:  {plot_path}")


if __name__ == "__main__":
    main()
