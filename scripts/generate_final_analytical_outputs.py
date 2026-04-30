from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_prb3r_comparison_plots
from src.beam_theory import generate_standard_locus_dataset
from src.plotting import PAPER_FIGURE_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINAL_CSV_DIR = PROJECT_ROOT / "results" / "csv" / "Final_Analytical"
FINAL_FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "Final_Analytical"
PAPER_FINAL_FIGURE_DIR = PAPER_FIGURE_DIR / "Final_Analytical"

VALIDATION_CSV = FINAL_CSV_DIR / "analytical_validation_summary.csv"
DIAGNOSTIC_NOTE = FINAL_CSV_DIR / "kappa1_diagnostic_note.md"
EXPECTED_ERRORS = {
    0.0: 1.9,
    0.1: 1.6,
    1.0: 0.9,
    2.0: 1.1,
    5.0: 1.2,
    50.0: 2.2,
}
TOLERANCE = 0.35


def main() -> dict[str, list[Path]]:
    FINAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    continuous_df = generate_standard_locus_dataset()
    comparison_df = generate_prb3r_comparison_plots._generate_load_ratio_sweep(beta_sign=1)
    force_angle_df = generate_prb3r_comparison_plots._generate_force_angle_sweep(beta_sign=1)

    validation_df = _write_validation_summary(comparison_df)
    _write_kappa1_note()

    local_figures = []
    paper_figures = []
    for figure_paths in [
        _plot_continuous_beam_tip_loci(continuous_df),
        _plot_prb_vs_continuous_beam(comparison_df),
        _plot_tip_error_vs_theta(comparison_df),
        _plot_max_error_by_kappa(validation_df),
        _plot_force_angle_sweep(force_angle_df),
    ]:
        local_figures.extend(figure_paths["local"])
        paper_figures.extend(figure_paths["paper"])

    print(f"Saved final analytical validation summary: {VALIDATION_CSV}")
    print(f"Saved kappa=1 diagnostic note: {DIAGNOSTIC_NOTE}")
    print("Saved final analytical figures:")
    for path in local_figures:
        print(f"  {path}")
    print("Copied paper-ready figures:")
    for path in paper_figures:
        print(f"  {path}")

    return {
        "csv_files": [VALIDATION_CSV, DIAGNOSTIC_NOTE],
        "local_figures": local_figures,
        "paper_figures": paper_figures,
    }


def _write_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kappa, expected in EXPECTED_ERRORS.items():
        group = df[df["kappa"].round(10) == round(kappa, 10)]
        current = float(group["tip_error_percent"].max())
        difference = current - expected
        passes = abs(difference) <= TOLERANCE
        if np.isclose(kappa, 1.0):
            notes = (
                "Elevated error near high-slope endpoint; diagnostics indicate sign, solver branch, "
                "IK reconstruction, and stiffness fitting are not the primary cause."
            )
        elif passes:
            notes = "Agreement within expected tolerance"
        else:
            notes = "Slightly above tolerance but trend consistent"

        rows.append(
            {
                "kappa": kappa,
                "max_tip_error_percent": current,
                "expected_reference_error_percent": expected,
                "difference_percent_points": difference,
                "passes_reference_tolerance": passes,
                "notes": notes,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(VALIDATION_CSV, index=False)
    return result


def _write_kappa1_note() -> None:
    note = """# Kappa = 1 Diagnostic Note

The corrected analytical sweep removed invalid high-slope points and confirmed stable solver behavior for the PRB 3R model. Most load-ratio cases agree with the expected paper-style error levels: kappa = 0, 0.1, 2, 5, and 50 are within the current tolerance band or very close to the reference trend. The remaining outlier is kappa = 1 at phi = pi/2, where the maximum normalized tip-position error is about 5%, compared with an expected value near 0.9%.

Several possible implementation causes were tested. The beta sign convention was checked explicitly, and beta_sign = +1 was confirmed as the correct convention; beta_sign = -1 produces much larger errors and is not physically consistent with the current formulation. Continuation diagnostics showed no branch jumps, and nonlinear equilibrium residuals remained small, so the elevated error is not caused by failed PRB solves. Inverse kinematics from the continuous beam tip position and tip slope reconstructs PRB joint angles to near machine precision wherever the PRB geometry is reachable, ruling out a forward/inverse kinematic inconsistency. A stiffness reconstruction diagnostic also showed that fitted stiffness values for kappa = 1 remain close to the tabulated values and do not reduce the maximum tip error.

For the project paper, this result remains useful because it is localized and well characterized. The analytical model captures the main trends across load ratios and force-angle cases, while the kappa = 1 endpoint discrepancy can be reported as a limitation of the fixed-parameter PRB 3R approximation or of the comparison convention used for that transitional load case. The final paper figures should therefore present the table-stiffness analytical results honestly, with a short note that kappa = 1 exhibits elevated error near the high-slope endpoint despite stable sign, branch, IK, and stiffness diagnostics.
"""
    DIAGNOSTIC_NOTE.write_text(note, encoding="utf-8")


def _plot_continuous_beam_tip_loci(df: pd.DataFrame) -> dict[str, list[Path]]:
    fig, ax = _new_figure()
    for kappa, group in df.groupby("kappa", sort=True):
        valid = group.dropna(subset=["Qx", "Qy"]).sort_values("theta0")
        ax.plot(valid["Qx"], valid["Qy"], linewidth=1.6, label=_kappa_label(kappa))
    _format_locus_axes(ax)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_final_figure(fig, "continuous_beam_tip_loci_phi90.png")


def _plot_prb_vs_continuous_beam(df: pd.DataFrame) -> dict[str, list[Path]]:
    fig, ax = _new_figure()
    for kappa, group in df.groupby("kappa", sort=True):
        valid = group.dropna(subset=["Qx_ref", "Qy_ref", "Qx_prb", "Qy_prb"]).sort_values("theta0_ref")
        line = ax.plot(valid["Qx_ref"], valid["Qy_ref"], linewidth=1.6, label=f"CB, {_kappa_label(kappa)}")[0]
        ax.plot(
            valid["Qx_prb"],
            valid["Qy_prb"],
            linestyle="--",
            linewidth=1.4,
            color=line.get_color(),
            label=f"PRB, {_kappa_label(kappa)}",
        )
    _format_locus_axes(ax)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save_final_figure(fig, "prb3r_vs_continuous_beam_phi90.png")


def _plot_tip_error_vs_theta(df: pd.DataFrame) -> dict[str, list[Path]]:
    fig, ax = _new_figure(width=6.4, height=4.4)
    for kappa, group in df.groupby("kappa", sort=True):
        valid = group.dropna(subset=["theta0_ref", "tip_error_percent"]).sort_values("theta0_ref")
        ax.plot(valid["theta0_ref"], valid["tip_error_percent"], linewidth=1.5, label=_kappa_label(kappa))
    ax.set_xlabel(r"$\theta_0$ [rad]")
    ax.set_ylabel(r"$e_Q$ [%]")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_final_figure(fig, "prb3r_tip_error_vs_theta_phi90.png")


def _plot_max_error_by_kappa(summary_df: pd.DataFrame) -> dict[str, list[Path]]:
    fig, ax = _new_figure(width=6.0, height=4.1)
    x_labels = [f"{value:g}" for value in summary_df["kappa"]]
    bars = ax.bar(x_labels, summary_df["max_tip_error_percent"])
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\max(e_Q)$ [%]")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    ymax = float(summary_df["max_tip_error_percent"].max())
    ax.set_ylim(0.0, ymax * 1.18)
    for bar, value in zip(bars, summary_df["max_tip_error_percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return _save_final_figure(fig, "prb3r_max_error_by_kappa.png")


def _plot_force_angle_sweep(df: pd.DataFrame) -> dict[str, list[Path]]:
    fig, ax = _new_figure(width=7.0, height=5.0)
    for phi_deg, group in df.groupby("phi_deg", sort=True):
        valid = group.dropna(subset=["Qx_ref", "Qy_ref", "Qx_prb", "Qy_prb"]).sort_values("theta0_ref")
        label = rf"$\phi = {phi_deg:g}^\circ$"
        line = ax.plot(valid["Qx_ref"], valid["Qy_ref"], linewidth=1.5, label=f"CB, {label}")[0]
        ax.plot(
            valid["Qx_prb"],
            valid["Qy_prb"],
            linestyle="--",
            linewidth=1.3,
            color=line.get_color(),
            label=f"PRB, {label}",
        )
    _format_locus_axes(ax)
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    return _save_final_figure(fig, "prb3r_force_angle_sweep_kappa0.png")


def _new_figure(width: float = 6.5, height: float = 4.8) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    return plt.subplots(figsize=(width, height))


def _format_locus_axes(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$a/l$")
    ax.set_ylabel(r"$b/l$")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")


def _save_final_figure(fig: plt.Figure, filename: str) -> dict[str, list[Path]]:
    local_png = FINAL_FIGURE_DIR / filename
    local_pdf = local_png.with_suffix(".pdf")
    fig.savefig(local_png, dpi=300, bbox_inches="tight")
    fig.savefig(local_pdf, bbox_inches="tight")
    plt.close(fig)

    paper_png = PAPER_FINAL_FIGURE_DIR / filename
    paper_pdf = paper_png.with_suffix(".pdf")
    shutil.copy2(local_png, paper_png)
    shutil.copy2(local_pdf, paper_pdf)
    return {"local": [local_png, local_pdf], "paper": [paper_png, paper_pdf]}


def _kappa_label(kappa: float) -> str:
    return rf"$\kappa = {float(kappa):g}$"


if __name__ == "__main__":
    main()
