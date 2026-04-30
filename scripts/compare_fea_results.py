from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fea_plotting import (
    PAPER_FIGURE_DIR,
    plot_max_stress_by_case,
    plot_qx_qy_comparison,
    plot_stress_safety_factor,
    plot_tip_error_by_case,
    plot_tip_error_vs_theta,
    plot_tip_loci_comparison,
)
from src.fea_processing import (
    build_reference_cases,
    compute_fea_quantities,
    load_fea_results,
    merge_fea_with_reference,
    summarize_fea_comparison,
)


CSV_DIR = PROJECT_ROOT / "results" / "csv" / "FEA_Results"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "FEA_Results"
REFERENCE_CSV = CSV_DIR / "fea_reference_cases.csv"
COMPARISON_CSV = CSV_DIR / "fea_comparison_summary.csv"
SUMMARY_CSV = CSV_DIR / "fea_summary_statistics.csv"


def main() -> dict[str, list[Path] | dict[str, float | int]]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fea = load_fea_results()
    reference = build_reference_cases()
    fea_quantities = compute_fea_quantities(fea)
    comparison = merge_fea_with_reference(fea_quantities, reference)
    summary = summarize_fea_comparison(comparison)

    reference.to_csv(REFERENCE_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    figure_paths = []
    figure_paths.extend(plot_tip_error_by_case(comparison, FIGURE_DIR / "fea_tip_error_by_case.png", "fea_tip_error_by_case.png"))
    figure_paths.extend(plot_qx_qy_comparison(comparison, FIGURE_DIR / "fea_qx_qy_comparison.png", "fea_qx_qy_comparison.png"))
    figure_paths.extend(plot_tip_loci_comparison(comparison, FIGURE_DIR / "fea_tip_loci_comparison.png", "fea_tip_loci_comparison.png"))
    figure_paths.extend(plot_max_stress_by_case(comparison, FIGURE_DIR / "fea_max_stress_by_case.png", "fea_max_stress_by_case.png"))
    figure_paths.extend(plot_stress_safety_factor(comparison, FIGURE_DIR / "fea_stress_safety_factor.png", "fea_stress_safety_factor.png"))
    figure_paths.extend(plot_tip_error_vs_theta(comparison, FIGURE_DIR / "fea_tip_error_vs_theta.png", "fea_tip_error_vs_theta.png"))

    local_figures = [path for path in figure_paths if FIGURE_DIR == path.parent]
    paper_figures = [path for path in figure_paths if PAPER_FIGURE_DIR == path.parent]

    _print_summary(summary, [REFERENCE_CSV, COMPARISON_CSV, SUMMARY_CSV], local_figures, paper_figures)
    return {
        "csv_files": [REFERENCE_CSV, COMPARISON_CSV, SUMMARY_CSV],
        "local_figures": local_figures,
        "paper_figures": paper_figures,
        "summary": summary,
    }


def _print_summary(
    summary: dict[str, float | int],
    csv_files: list[Path],
    local_figures: list[Path],
    paper_figures: list[Path],
) -> None:
    print("FEA comparison summary")
    print(f"Cases processed: {summary['number_of_cases']}")
    print(
        f"Maximum tip error: {summary['max_tip_error_percent']:.3f}% "
        f"(case {summary['case_id_with_max_error']})"
    )
    print(
        f"Maximum stress: {summary['max_stress_MPa']:.3f} MPa "
        f"(case {summary['case_id_with_max_stress']})"
    )
    print(
        f"Minimum safety factor: {summary['min_stress_safety_factor']:.3f} "
        f"(case {summary['case_id_with_min_safety_factor']})"
    )
    print("CSV files created:")
    for path in csv_files:
        print(f"  {path}")
    print("Local figures created:")
    for path in local_figures:
        print(f"  {path}")
    print("Paper figures copied:")
    for path in paper_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
