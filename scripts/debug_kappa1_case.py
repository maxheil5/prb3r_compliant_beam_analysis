from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import RUN_CSV_DIR, RUN_VERIFICATION_FIGURE_DIR, save_figure
from src.prb3r_model import sweep_prb3r_for_beam_locus
from src.validation_metrics import add_error_columns

import matplotlib.pyplot as plt


CSV_PATH = RUN_CSV_DIR / "debug_kappa1_endpoint_and_sign.csv"
FIGURE_DIR = RUN_VERIFICATION_FIGURE_DIR
COLUMN_ORDER = [
    "endpoint_margin",
    "beta_sign",
    "theta0_ref",
    "alpha",
    "beta",
    "beta_for_prb",
    "kappa",
    "phi",
    "Qx_ref",
    "Qy_ref",
    "theta1",
    "theta2",
    "theta3",
    "Qx_prb",
    "Qy_prb",
    "theta0_prb",
    "success",
    "residual_norm",
    "tip_error_percent",
    "slope_error_percent",
    "delta_theta_norm",
    "branch_jump_flag",
]


def main() -> dict[str, list[Path] | int]:
    RUN_CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = _generate_debug_dataset()
    df.to_csv(CSV_PATH, index=False)

    figure_paths = []
    figure_paths.extend(_plot_error_by_endpoint_margin(df))
    figure_paths.extend(_plot_error_by_beta_sign(df))
    figure_paths.extend(_plot_joint_angles(df))
    figure_paths.extend(_plot_residual_norm(df))
    figure_paths.extend(_plot_locus_detail(df))

    failures = int((~df["success"]).sum())
    branch_jumps = int(df["branch_jump_flag"].sum())
    print(f"Saved kappa = 1 diagnostic CSV: {CSV_PATH}")
    return {
        "csv_files": [CSV_PATH],
        "local_figures": figure_paths,
        "paper_figures": [],
        "warnings": failures + branch_jumps,
    }


def _generate_debug_dataset() -> pd.DataFrame:
    frames = []
    for endpoint_margin in [0.02, 0.05, 0.10, 0.20]:
        for beta_sign in [1, -1]:
            frame = sweep_prb3r_for_beam_locus(
                kappa=1.0,
                phi=np.pi / 2.0,
                endpoint_margin=endpoint_margin,
                beta_sign=beta_sign,
            )
            frame["endpoint_margin"] = endpoint_margin
            frames.append(add_error_columns(frame))
    return pd.concat(frames, ignore_index=True)[COLUMN_ORDER]


def _plot_error_by_endpoint_margin(df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for (endpoint_margin, beta_sign), group in df.groupby(["endpoint_margin", "beta_sign"], sort=True):
        valid = group.dropna(subset=["theta0_ref", "tip_error_percent"]).sort_values("theta0_ref")
        ax.plot(
            valid["theta0_ref"],
            valid["tip_error_percent"],
            label=f"margin={endpoint_margin:g}, beta sign={beta_sign:+d}",
            linewidth=1.4,
        )
    ax.set_xlabel("theta_0, rad")
    ax.set_ylabel("tip position error, %")
    ax.set_title("Kappa 1 Error by Endpoint Margin")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_error_by_endpoint_margin.png")
    plt.close(fig)
    return paths


def _plot_error_by_beta_sign(df: pd.DataFrame) -> list[Path]:
    summary = df.groupby("beta_sign", sort=True)["tip_error_percent"].max()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar([f"{int(sign):+d}" for sign in summary.index], summary.values)
    ax.set_xlabel("beta sign")
    ax.set_ylabel("maximum tip position error, %")
    ax.set_title("Kappa 1 Error by Beta Sign")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_error_by_beta_sign.png")
    plt.close(fig)
    return paths


def _plot_joint_angles(df: pd.DataFrame) -> list[Path]:
    subset = df[np.isclose(df["endpoint_margin"], 0.02)]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for beta_sign, group in subset.groupby("beta_sign", sort=True):
        valid = group.sort_values("theta0_ref")
        for column in ["theta1", "theta2", "theta3"]:
            ax.plot(
                valid["theta0_ref"],
                valid[column],
                label=f"{column}, beta sign={int(beta_sign):+d}",
                linewidth=1.3,
            )
    ax.set_xlabel("theta_0, rad")
    ax.set_ylabel("joint angle, rad")
    ax.set_title("Kappa 1 PRB Joint Angles")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_joint_angles.png")
    plt.close(fig)
    return paths


def _plot_residual_norm(df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for (endpoint_margin, beta_sign), group in df.groupby(["endpoint_margin", "beta_sign"], sort=True):
        valid = group.dropna(subset=["theta0_ref", "residual_norm"]).sort_values("theta0_ref")
        ax.semilogy(
            valid["theta0_ref"],
            valid["residual_norm"],
            label=f"margin={endpoint_margin:g}, beta sign={int(beta_sign):+d}",
            linewidth=1.3,
        )
    ax.set_xlabel("theta_0, rad")
    ax.set_ylabel("residual norm")
    ax.set_title("Kappa 1 Residual Norm")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_residual_norm.png")
    plt.close(fig)
    return paths


def _plot_locus_detail(df: pd.DataFrame) -> list[Path]:
    subset = df[np.isclose(df["endpoint_margin"], 0.02)]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ref = subset[subset["beta_sign"] == 1].sort_values("theta0_ref")
    ax.plot(ref["Qx_ref"], ref["Qy_ref"], label="reference", linewidth=2.0)
    for beta_sign, group in subset.groupby("beta_sign", sort=True):
        valid = group.sort_values("theta0_ref")
        ax.plot(
            valid["Qx_prb"],
            valid["Qy_prb"],
            linestyle="--",
            linewidth=1.6,
            label=f"PRB 3R, beta sign={int(beta_sign):+d}",
        )
    ax.set_xlabel("a/l")
    ax.set_ylabel("b/l")
    ax.set_title("Kappa 1 Locus Detail")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_locus_detail.png")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
