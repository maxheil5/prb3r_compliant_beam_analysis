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
PAPER_DIR = WORKSPACE_ROOT / "ME7751-Course-Project-2-Paper"
PAPER_FIGURE_DIR = PAPER_DIR / "Figures"
RUN_NAME = "Run 3"
PAPER_RUN_FIGURE_DIR = PAPER_FIGURE_DIR / RUN_NAME
LOCAL_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RUN_CSV_DIR = PROJECT_ROOT / "results" / "csv" / RUN_NAME
RUN_FIGURE_DIR = LOCAL_FIGURE_DIR / RUN_NAME
RUN_ANALYTICAL_FIGURE_DIR = RUN_FIGURE_DIR / "analytical"
RUN_VERIFICATION_FIGURE_DIR = RUN_FIGURE_DIR / "verification"


def save_figure(fig: plt.Figure, local_path: str | Path, paper_filename: str | None = None) -> list[Path]:
    local_png = _resolve_local_path(local_path)
    local_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_png, dpi=300, bbox_inches="tight")

    saved_paths = [local_png]
    local_pdf = None
    if local_png.suffix.lower() == ".png":
        local_pdf = local_png.with_suffix(".pdf")
        fig.savefig(local_pdf, bbox_inches="tight")
        saved_paths.append(local_pdf)

    if paper_filename is not None:
        PAPER_RUN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        paper_png = PAPER_RUN_FIGURE_DIR / paper_filename
        if paper_png.suffix.lower() != ".png":
            paper_png = paper_png.with_suffix(".png")
        shutil.copy2(local_png, paper_png)
        saved_paths.append(paper_png)

        if local_pdf is not None:
            paper_pdf = paper_png.with_suffix(".pdf")
            shutil.copy2(local_pdf, paper_pdf)
            saved_paths.append(paper_pdf)

    return saved_paths


def copy_figures_to_paper_run(local_figure_paths: list[Path]) -> list[Path]:
    PAPER_RUN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in local_figure_paths:
        source = Path(source)
        if source.suffix.lower() not in {".png", ".pdf"}:
            continue
        target = PAPER_RUN_FIGURE_DIR / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def plot_tip_locus_atlas(
    df: pd.DataFrame,
    output_path: str | Path,
    paper_filename: str | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for kappa, group in df.groupby("kappa", sort=True):
        valid = group.dropna(subset=["Qx", "Qy"]).sort_values("theta0")
        if valid.empty:
            continue
        ax.plot(valid["Qx"], valid["Qy"], label=f"kappa = {_format_number(kappa)}", linewidth=1.8)

    ax.set_xlabel("a/l", fontsize=11)
    ax.set_ylabel("b/l", fontsize=11)
    ax.set_title("Continuous Beam Tip Loci for Varying Load Ratio", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_prb_vs_reference(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    paper_filename: str | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    grouping = "kappa" if "kappa" in df.columns else None
    groups = df.groupby(grouping, sort=True) if grouping else [(None, df)]

    for key, group in groups:
        valid_ref = group.dropna(subset=["Qx_ref", "Qy_ref"]).sort_values("theta0_ref")
        valid_prb = group.dropna(subset=["Qx_prb", "Qy_prb"]).sort_values("theta0_ref")
        if valid_ref.empty:
            continue

        label_suffix = f", kappa = {_format_number(key)}" if key is not None else ""
        ref_line = ax.plot(
            valid_ref["Qx_ref"],
            valid_ref["Qy_ref"],
            linewidth=1.8,
            label=f"reference{label_suffix}",
        )[0]
        if not valid_prb.empty:
            ax.plot(
                valid_prb["Qx_prb"],
                valid_prb["Qy_prb"],
                linestyle="--",
                linewidth=1.5,
                color=ref_line.get_color(),
                label=f"PRB 3R{label_suffix}",
            )

    ax.set_xlabel("a/l", fontsize=11)
    ax.set_ylabel("b/l", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_error_vs_theta(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    paper_filename: str | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    grouping = "kappa" if "kappa" in df.columns else None
    groups = df.groupby(grouping, sort=True) if grouping else [(None, df)]

    for key, group in groups:
        valid = group.dropna(subset=["theta0_ref", "tip_error_percent"]).sort_values("theta0_ref")
        if valid.empty:
            continue
        label = f"kappa = {_format_number(key)}" if key is not None else "tip error"
        ax.plot(valid["theta0_ref"], valid["tip_error_percent"], linewidth=1.7, label=label)

    ax.set_xlabel("theta_0, rad", fontsize=11)
    ax.set_ylabel("tip position error, %", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_error_summary_by_kappa(
    df: pd.DataFrame,
    output_path: str | Path,
    paper_filename: str | None = None,
) -> list[Path]:
    valid = df.dropna(subset=["kappa", "tip_error_percent"])
    summary = valid.groupby("kappa", sort=True)["tip_error_percent"].max()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar([_format_number(kappa) for kappa in summary.index], summary.values)
    ax.set_xlabel("kappa", fontsize=11)
    ax.set_ylabel("maximum tip position error, %", fontsize=11)
    ax.set_title("Maximum PRB 3R Tip Error by Load Ratio", fontsize=12)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def plot_force_angle_sweep(
    df: pd.DataFrame,
    output_path: str | Path,
    paper_filename: str | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for phi_deg, group in df.groupby("phi_deg", sort=True):
        valid_ref = group.dropna(subset=["Qx_ref", "Qy_ref"]).sort_values("theta0_ref")
        valid_prb = group.dropna(subset=["Qx_prb", "Qy_prb"]).sort_values("theta0_ref")
        if valid_ref.empty:
            continue

        ref_line = ax.plot(
            valid_ref["Qx_ref"],
            valid_ref["Qy_ref"],
            linewidth=1.8,
            label=f"reference, phi = {_format_number(phi_deg)} deg",
        )[0]
        if not valid_prb.empty:
            ax.plot(
                valid_prb["Qx_prb"],
                valid_prb["Qy_prb"],
                linestyle="--",
                linewidth=1.5,
                color=ref_line.get_color(),
                label=f"PRB 3R, phi = {_format_number(phi_deg)} deg",
            )

    ax.set_xlabel("a/l", fontsize=11)
    ax.set_ylabel("b/l", fontsize=11)
    ax.set_title("PRB 3R Force Angle Sweep", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths = save_figure(fig, output_path, paper_filename)
    plt.close(fig)
    return paths


def _resolve_local_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"
