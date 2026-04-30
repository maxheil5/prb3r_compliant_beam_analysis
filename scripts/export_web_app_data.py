from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_ROOT / "web_app" / "public" / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cases = _load_csv("results/csv/FEA_Results/fea_comparison_summary.csv")
    sweep = _load_csv("results/csv/Run 3/prb3r_vs_beam_phi90.csv")
    validation = _load_csv("results/csv/Final_Analytical/analytical_validation_summary.csv")

    case_records = _records(_select_cases(cases))
    sweep_records = _records(_select_sweep(sweep))
    validation_records = _records(validation)

    _write_json("fea_cases.json", case_records)
    _write_json("prb_sweep_phi90.json", sweep_records)
    _write_json("validation_summary.json", validation_records)
    _write_json(
        "app_manifest.json",
        {
            "case_count": len(case_records),
            "sweep_count": len(sweep_records),
            "validation_count": len(validation_records),
            "source": "ME7751 PRB 3R compliant beam analytical and SolidWorks FEA results",
        },
    )

    print(f"Exported web app data to {DATA_DIR}")


def _load_csv(relative_path: str) -> pd.DataFrame:
    path = ANALYSIS_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required source CSV: {path}")
    return pd.read_csv(path)


def _select_cases(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "kappa",
        "theta0_target_rad",
        "phi_deg",
        "alpha",
        "beta",
        "Qx_ref",
        "Qy_ref",
        "Qx_fea",
        "Qy_fea",
        "x_tip_ref_mm",
        "y_tip_ref_mm",
        "x_tip_fea_mm",
        "y_tip_fea_mm",
        "Fz_N",
        "Mx_Nm",
        "load_description",
        "tip_error_percent",
        "tip_error_mm",
        "max_von_mises_mpa",
        "max_equivalent_strain",
        "stress_safety_factor",
        "stress_margin_MPa",
    ]
    return df[columns].sort_values("case_id")


def _select_sweep(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "theta0_ref",
        "alpha",
        "beta",
        "kappa",
        "Qx_ref",
        "Qy_ref",
        "theta1",
        "theta2",
        "theta3",
        "Qx_prb",
        "Qy_prb",
        "theta0_prb",
        "tip_error_percent",
    ]
    sweep = df[df["success"].astype(bool)].copy()
    sweep = sweep[np.isclose(sweep["phi"], np.pi / 2.0)]
    sweep = sweep[np.isclose(sweep["beta_sign"], 1.0)]
    return sweep[columns].sort_values(["kappa", "theta0_ref"])


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    cleaned = df.replace({np.nan: None})
    return json.loads(cleaned.to_json(orient="records", double_precision=10))


def _write_json(filename: str, payload: object) -> None:
    path = DATA_DIR / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
