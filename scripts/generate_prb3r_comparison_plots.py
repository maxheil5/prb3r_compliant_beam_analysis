from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import (
    RUN_ANALYTICAL_FIGURE_DIR,
    RUN_CSV_DIR,
    plot_error_summary_by_kappa,
    plot_error_vs_theta,
    plot_force_angle_sweep,
    plot_prb_vs_reference,
)
from src.prb3r_model import sweep_prb3r_for_beam_locus
from src.validation_metrics import add_error_columns


CSV_DIR = RUN_CSV_DIR
FIGURE_DIR = RUN_ANALYTICAL_FIGURE_DIR


def main(beta_sign: int = 1) -> dict[str, list[Path] | int]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    load_ratio_df = _generate_load_ratio_sweep(beta_sign=beta_sign)
    load_ratio_csv = CSV_DIR / "prb3r_vs_beam_phi90.csv"
    load_ratio_df.to_csv(load_ratio_csv, index=False)

    local_and_paper_paths = []
    local_and_paper_paths.extend(
        plot_prb_vs_reference(
            load_ratio_df,
            FIGURE_DIR / "prb3r_vs_reference_phi90.png",
            "PRB 3R Model Compared with Continuous Beam Reference",
        )
    )
    local_and_paper_paths.extend(
        plot_error_vs_theta(
            load_ratio_df,
            FIGURE_DIR / "prb3r_tip_error_phi90.png",
            "PRB 3R Tip Position Error",
        )
    )
    local_and_paper_paths.extend(
        plot_error_summary_by_kappa(
            load_ratio_df,
            FIGURE_DIR / "prb3r_error_summary_by_kappa.png",
        )
    )

    angle_df = _generate_force_angle_sweep(beta_sign=beta_sign)
    angle_csv = CSV_DIR / "prb3r_force_angle_sweep.csv"
    angle_df.to_csv(angle_csv, index=False)
    local_and_paper_paths.extend(
        plot_force_angle_sweep(
            angle_df,
            FIGURE_DIR / "prb3r_force_angle_sweep.png",
        )
    )

    local_figures = [path for path in local_and_paper_paths if PROJECT_ROOT in path.parents]
    failures = int((~load_ratio_df["success"]).sum() + (~angle_df["success"]).sum())
    branch_jumps = int(load_ratio_df["branch_jump_flag"].sum() + angle_df["branch_jump_flag"].sum())
    warnings = failures + branch_jumps

    print(f"Saved PRB comparison CSV: {load_ratio_csv}")
    print(f"Saved force angle sweep CSV: {angle_csv}")
    return {
        "csv_files": [load_ratio_csv, angle_csv],
        "local_figures": local_figures,
        "paper_figures": [],
        "warnings": warnings,
    }


def _generate_load_ratio_sweep(beta_sign: int = 1) -> pd.DataFrame:
    kappa_values = [0, 0.1, 1, 2, 5, 50]
    phi = np.pi / 2.0
    frames = [
        sweep_prb3r_for_beam_locus(kappa=kappa, phi=phi, beta_sign=beta_sign)
        for kappa in kappa_values
    ]
    return add_error_columns(pd.concat(frames, ignore_index=True))


def _generate_force_angle_sweep(beta_sign: int = 1) -> pd.DataFrame:
    kappa = 0.0
    frames = []

    for phi_deg in [30, 60, 120, 135, 150, 175]:
        frame = sweep_prb3r_for_beam_locus(
            kappa=kappa,
            phi=np.deg2rad(phi_deg),
            beta_sign=beta_sign,
        )
        frame["phi_deg"] = float(phi_deg)
        frames.append(frame)

    return add_error_columns(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    main()
