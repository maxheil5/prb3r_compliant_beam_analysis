from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PAPER_FIGURE_DIR = WORKSPACE_ROOT / "ME7751-Course-Project-2-Paper" / "Figures" / "FEA_Results"


def save_figure(fig: plt.Figure, local_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    local_png = Path(local_path)
    local_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_png, dpi=300, bbox_inches="tight")
    saved = [local_png]

    if local_png.suffix.lower() == ".png":
        local_pdf = local_png.with_suffix(".pdf")
        fig.savefig(local_pdf, bbox_inches="tight")
        saved.append(local_pdf)
    else:
        local_pdf = None

    if paper_filename is not None:
        PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        paper_png = PAPER_FIGURE_DIR / paper_filename
        if paper_png.suffix.lower() != ".png":
            paper_png = paper_png.with_suffix(".png")
        shutil.copy2(local_png, paper_png)
        saved.append(paper_png)
        if local_pdf is not None:
            paper_pdf = paper_png.with_suffix(".pdf")
            shutil.copy2(local_pdf, paper_pdf)
            saved.append(paper_pdf)

    return saved


def plot_tip_error_by_case(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure()
    bars = ax.bar(df["case_id"].astype(str), df["tip_error_percent"])
    ax.set_xlabel("Case ID")
    ax.set_ylabel(r"Tip-position error, $e_Q$ [%]")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    _add_bar_labels(ax, bars, df["tip_error_percent"])
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_qx_qy_comparison(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure(width=7.2, height=4.6)
    x = np.arange(len(df))
    width = 0.2
    ax.bar(x - 1.5 * width, df["Qx_ref"], width, label=r"$Q_{x,\mathrm{ref}}$")
    ax.bar(x - 0.5 * width, df["Qx_fea"], width, label=r"$Q_{x,\mathrm{FEA}}$")
    ax.bar(x + 0.5 * width, df["Qy_ref"], width, label=r"$Q_{y,\mathrm{ref}}$")
    ax.bar(x + 1.5 * width, df["Qy_fea"], width, label=r"$Q_{y,\mathrm{FEA}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_id"].astype(int).astype(str))
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Normalized tip coordinate")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_tip_loci_comparison(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure(width=6.2, height=5.0)
    ax.plot(df["Qx_ref"], df["Qy_ref"], "o", label="Continuous beam")
    ax.plot(df["Qx_fea"], df["Qy_fea"], "s", label="FEA")
    for _, row in df.iterrows():
        ax.plot([row["Qx_ref"], row["Qx_fea"]], [row["Qy_ref"], row["Qy_fea"]], linewidth=0.8, alpha=0.5)
        ax.text(row["Qx_fea"], row["Qy_fea"], str(int(row["case_id"])), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(r"$a/l$")
    ax.set_ylabel(r"$b/l$")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_max_stress_by_case(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure()
    bars = ax.bar(df["case_id"].astype(str), df["max_von_mises_mpa"])
    ax.axhline(30.0, linestyle="--", linewidth=1.2, label="Yield strength")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Max von Mises stress [MPa]")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    _add_bar_labels(ax, bars, df["max_von_mises_mpa"])
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_stress_safety_factor(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure()
    bars = ax.bar(df["case_id"].astype(str), df["stress_safety_factor"])
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Safety factor = 1")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Stress safety factor")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    _add_bar_labels(ax, bars, df["stress_safety_factor"])
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_tip_error_vs_theta(df: pd.DataFrame, output_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    fig, ax = _new_figure()
    for kappa, group in df.groupby("kappa", sort=True):
        ax.scatter(group["theta0_target_rad"], group["tip_error_percent"], label=rf"$\kappa={float(kappa):g}$")
    ax.set_xlabel(r"$\theta_0$ target [rad]")
    ax.set_ylabel(r"Tip-position error, $e_Q$ [%]")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def _new_figure(width: float = 6.4, height: float = 4.4) -> tuple[plt.Figure, plt.Axes]:
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


def _add_bar_labels(ax: plt.Axes, bars, values: pd.Series) -> None:
    max_value = float(np.nanmax(values))
    offset = max_value * 0.025 if max_value > 0 else 0.05
    ax.set_ylim(top=max_value * 1.16 if max_value > 0 else 1.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
