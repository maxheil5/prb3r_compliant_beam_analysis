from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import beam_theory
from src.plotting import RUN_CSV_DIR
from src.prb3r_model import solve_prb3r
from src.validation_metrics import add_error_columns


CSV_PATH = RUN_CSV_DIR / "basic_verification.csv"


def main() -> dict[str, list[Path] | int]:
    rows = []
    phi = np.pi / 2.0

    for theta0 in [0.1, 0.5, 1.0, np.pi / 2.0]:
        reference = beam_theory.pure_moment_tip(theta0)
        solve = solve_prb3r(alpha=0.0, beta=theta0, phi=phi)
        rows.append(_row("pure_moment", reference, solve, np.inf, phi))

    for theta0 in [0.1, 0.5, 1.0]:
        reference = beam_theory.compute_tip_from_theta0(theta0, kappa=0.0, phi=phi)
        solve = solve_prb3r(alpha=reference["alpha"], beta=reference["beta"], phi=phi)
        rows.append(_row("pure_vertical_force", reference, solve, 0.0, phi))

    df = add_error_columns(pd.DataFrame(rows))
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    print("Basic verification")
    print(
        df[
            [
                "case_type",
                "theta0_ref",
                "alpha",
                "beta",
                "tip_error_percent",
                "slope_error_percent",
                "success",
                "residual_norm",
            ]
        ].to_string(index=False)
    )

    failures = int((~df["success"]).sum())
    return {"csv_files": [CSV_PATH], "local_figures": [], "paper_figures": [], "warnings": failures}


def _row(case_type: str, reference: dict[str, float], solve: dict[str, object], kappa: float, phi: float) -> dict[str, object]:
    return {
        "case_type": case_type,
        "theta0_ref": reference["theta0"],
        "kappa": kappa,
        "phi": phi,
        "alpha": reference["alpha"],
        "beta": reference["beta"],
        "Qx_ref": reference["Qx"],
        "Qy_ref": reference["Qy"],
        "Qx_prb": solve["Qx"],
        "Qy_prb": solve["Qy"],
        "theta0_prb": solve["theta0"],
        "success": solve["success"],
        "residual_norm": solve["residual_norm"],
    }


if __name__ == "__main__":
    main()
