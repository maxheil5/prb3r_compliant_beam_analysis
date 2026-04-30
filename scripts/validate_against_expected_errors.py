from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_prb3r_comparison_plots
from src.plotting import RUN_CSV_DIR


INPUT_CSV = RUN_CSV_DIR / "prb3r_vs_beam_phi90.csv"
OUTPUT_CSV = RUN_CSV_DIR / "expected_error_validation.csv"
EXPECTED_MAX_ERRORS = {
    0.0: 1.9,
    0.1: 1.6,
    1.0: 0.9,
    2.0: 1.1,
    5.0: 1.2,
    50.0: 2.2,
}
TOLERANCE = 0.35


def main(mode: str = "table") -> dict[str, list[Path] | int | bool]:
    if mode not in {"table", "fitted"}:
        raise ValueError("mode must be 'table' or 'fitted'.")
    RUN_CSV_DIR.mkdir(parents=True, exist_ok=True)
    if mode == "table":
        if not INPUT_CSV.exists():
            df = generate_prb3r_comparison_plots._generate_load_ratio_sweep()
            df.to_csv(INPUT_CSV, index=False)
        else:
            df = pd.read_csv(INPUT_CSV)
        output_csv = OUTPUT_CSV
    else:
        comparison_csv = RUN_CSV_DIR / "stiffness_fit_error_comparison.csv"
        if not comparison_csv.exists():
            raise ValueError("Run stiffness reconstruction before validating fitted stiffnesses.")
        comparison_df = pd.read_csv(comparison_csv)
        df = comparison_df.rename(columns={"max_error_fitted_stiffness": "tip_error_percent"})
        output_csv = RUN_CSV_DIR / "expected_error_validation_fitted_stiffness.csv"

    rows = []
    for kappa, expected in EXPECTED_MAX_ERRORS.items():
        group = df[df["kappa"].round(10) == round(kappa, 10)]
        max_error = float(group["tip_error_percent"].max())
        difference = max_error - expected
        rows.append(
            {
                "kappa": kappa,
                "max_error_code": max_error,
                "max_error_expected": expected,
                "difference": difference,
                "passes_tolerance": abs(difference) <= TOLERANCE,
                "mode": mode,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    validation_passed = bool(result["passes_tolerance"].all())

    print("Expected error validation")
    print(result.to_string(index=False))
    print(f"Validation passed: {validation_passed}")

    return {
        "csv_files": [output_csv],
        "local_figures": [],
        "paper_figures": [],
        "warnings": int((~result["passes_tolerance"]).sum()),
        "validation_passed": validation_passed,
    }


if __name__ == "__main__":
    main()
