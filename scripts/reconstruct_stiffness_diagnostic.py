from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beam_theory import sweep_beam_locus
from src.plotting import RUN_CSV_DIR, RUN_VERIFICATION_FIGURE_DIR, save_figure
from src.prb3r_model import PRB3RParameters, solve_prb3r
from src.stiffness_reconstruction import (
    reconstruct_stiffness_for_locus,
    solve_prb3r_with_custom_stiffness,
)
from src.validation_metrics import add_error_columns

import matplotlib.pyplot as plt


KAPPA_VALUES = [0, 0.1, 1, 2, 5, 50]
PHI = np.pi / 2.0
RECONSTRUCTION_CSV = RUN_CSV_DIR / "stiffness_reconstruction_by_kappa.csv"
SUMMARY_CSV = RUN_CSV_DIR / "stiffness_fit_summary.csv"
ERROR_COMPARISON_CSV = RUN_CSV_DIR / "stiffness_fit_error_comparison.csv"
FIGURE_DIR = RUN_VERIFICATION_FIGURE_DIR


def main() -> dict[str, list[Path] | int]:
    RUN_CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    params = PRB3RParameters()
    recon_df, summary_df = _run_reconstruction(params)
    error_df, kappa1_detail = _compare_table_and_fitted_errors(summary_df, params)

    recon_df.to_csv(RECONSTRUCTION_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    error_df.to_csv(ERROR_COMPARISON_CSV, index=False)

    figures = []
    figures.extend(_plot_stiffness_fit_vs_kappa(summary_df, params))
    figures.extend(_plot_stiffness_error_comparison(error_df))
    figures.extend(_plot_kappa1_locus(kappa1_detail))
    figures.extend(_plot_kappa1_error(kappa1_detail))

    warnings = int((~recon_df["ik_success"]).sum())
    print(f"Saved stiffness reconstruction CSV: {RECONSTRUCTION_CSV}")
    print(f"Saved stiffness fit summary CSV: {SUMMARY_CSV}")
    print(f"Saved stiffness fit error comparison CSV: {ERROR_COMPARISON_CSV}")
    return {
        "csv_files": [RECONSTRUCTION_CSV, SUMMARY_CSV, ERROR_COMPARISON_CSV],
        "local_figures": figures,
        "paper_figures": [],
        "warnings": warnings,
    }


def _run_reconstruction(params: PRB3RParameters) -> tuple[pd.DataFrame, pd.DataFrame]:
    recon_frames = []
    summary_rows = []

    for kappa in KAPPA_VALUES:
        reference = sweep_beam_locus(kappa=kappa, phi=PHI, endpoint_margin=0.02)
        reconstructed, fit = reconstruct_stiffness_for_locus(reference, params=params)
        recon_frames.append(reconstructed)

        summary_rows.append(
            {
                "kappa": float(kappa),
                "k1_fit": fit["k1_fit"],
                "k2_fit": fit["k2_fit"],
                "k3_fit": fit["k3_fit"],
                "k1_table": params.k1,
                "k2_table": params.k2,
                "k3_table": params.k3,
                "k1_difference": fit["k1_fit"] - params.k1,
                "k2_difference": fit["k2_fit"] - params.k2,
                "k3_difference": fit["k3_fit"] - params.k3,
                "max_reconstruction_error": float(reconstructed["reconstruction_error"].max()),
                "mean_reconstruction_error": float(reconstructed["reconstruction_error"].mean()),
            }
        )

    return pd.concat(recon_frames, ignore_index=True), pd.DataFrame(summary_rows)


def _compare_table_and_fitted_errors(
    summary_df: pd.DataFrame,
    params: PRB3RParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    kappa1_detail = None

    for _, summary in summary_df.iterrows():
        kappa = float(summary["kappa"])
        reference = sweep_beam_locus(kappa=kappa, phi=PHI, endpoint_margin=0.02)
        fitted_k = np.array([summary["k1_fit"], summary["k2_fit"], summary["k3_fit"]], dtype=float)

        table_df = _solve_against_reference(reference, params, fitted_k=None)
        fitted_df = _solve_against_reference(reference, params, fitted_k=fitted_k)

        max_error_table = float(table_df["tip_error_percent"].max())
        max_error_fitted = float(fitted_df["tip_error_percent"].max())
        comparison_rows.append(
            {
                "kappa": kappa,
                "max_error_table_stiffness": max_error_table,
                "max_error_fitted_stiffness": max_error_fitted,
                "error_reduction": max_error_table - max_error_fitted,
                "k1_fit": fitted_k[0],
                "k2_fit": fitted_k[1],
                "k3_fit": fitted_k[2],
            }
        )

        if np.isclose(kappa, 1.0):
            kappa1_detail = pd.concat(
                [
                    table_df.assign(stiffness_source="table"),
                    fitted_df.assign(stiffness_source="fitted"),
                ],
                ignore_index=True,
            )

    return pd.DataFrame(comparison_rows), kappa1_detail


def _solve_against_reference(
    reference: pd.DataFrame,
    params: PRB3RParameters,
    fitted_k: np.ndarray | None,
) -> pd.DataFrame:
    rows = []
    previous_theta = None
    for _, row in reference.iterrows():
        if fitted_k is None:
            solve = solve_prb3r(row["alpha"], row["beta"], row["phi"], params=params, initial_guess=previous_theta)
        else:
            solve = solve_prb3r_with_custom_stiffness(
                row["alpha"],
                row["beta"],
                row["phi"],
                fitted_k,
                params=params,
                initial_guess=previous_theta,
            )
        theta = np.asarray(solve["theta"], dtype=float)
        if solve["success"]:
            previous_theta = theta
        rows.append(
            {
                "theta0_ref": row["theta0"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "kappa": row["kappa"],
                "phi": row["phi"],
                "Qx_ref": row["Qx"],
                "Qy_ref": row["Qy"],
                "theta1": theta[0],
                "theta2": theta[1],
                "theta3": theta[2],
                "Qx_prb": solve["Qx"],
                "Qy_prb": solve["Qy"],
                "theta0_prb": solve["theta0"],
                "success": solve["success"],
                "residual_norm": solve["residual_norm"],
            }
        )
    return add_error_columns(pd.DataFrame(rows))


def _plot_stiffness_fit_vs_kappa(summary_df: pd.DataFrame, params: PRB3RParameters) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for column in ["k1_fit", "k2_fit", "k3_fit"]:
        ax.plot(summary_df["kappa"], summary_df[column], marker="o", linewidth=1.8, label=column)
    for value, label in [(params.k1, "k1 table"), (params.k2, "k2 table"), (params.k3, "k3 table")]:
        ax.axhline(value, linestyle="--", linewidth=1.0, label=label)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("kappa")
    ax.set_ylabel("fitted stiffness")
    ax.set_title("Reconstructed PRB Stiffness vs Load Ratio")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "stiffness_fit_vs_kappa.png")
    plt.close(fig)
    return paths


def _plot_stiffness_error_comparison(error_df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(error_df))
    width = 0.38
    ax.bar(x - width / 2.0, error_df["max_error_table_stiffness"], width, label="table stiffness")
    ax.bar(x + width / 2.0, error_df["max_error_fitted_stiffness"], width, label="fitted stiffness")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in error_df["kappa"]])
    ax.set_xlabel("kappa")
    ax.set_ylabel("maximum tip position error, %")
    ax.set_title("Table vs Fitted Stiffness Error")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "stiffness_error_comparison.png")
    plt.close(fig)
    return paths


def _plot_kappa1_locus(kappa1_detail: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    reference = kappa1_detail[kappa1_detail["stiffness_source"] == "table"].sort_values("theta0_ref")
    ax.plot(reference["Qx_ref"], reference["Qy_ref"], linewidth=2.0, label="reference")
    for source, group in kappa1_detail.groupby("stiffness_source", sort=True):
        valid = group.sort_values("theta0_ref")
        ax.plot(valid["Qx_prb"], valid["Qy_prb"], linestyle="--", linewidth=1.6, label=f"PRB {source}")
    ax.set_xlabel("a/l")
    ax.set_ylabel("b/l")
    ax.set_title("Kappa 1 Table vs Fitted Locus")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_table_vs_fitted_locus.png")
    plt.close(fig)
    return paths


def _plot_kappa1_error(kappa1_detail: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for source, group in kappa1_detail.groupby("stiffness_source", sort=True):
        valid = group.sort_values("theta0_ref")
        ax.plot(valid["theta0_ref"], valid["tip_error_percent"], linewidth=1.8, label=f"PRB {source}")
    ax.set_xlabel("theta_0, rad")
    ax.set_ylabel("tip position error, %")
    ax.set_title("Kappa 1 Table vs Fitted Error")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = save_figure(fig, FIGURE_DIR / "kappa1_table_vs_fitted_error.png")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
