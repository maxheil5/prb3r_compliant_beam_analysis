from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_prb3r_comparison_plots
import generate_tip_locus_atlas
import run_basic_verification
import validate_against_expected_errors
import debug_kappa1_case
import reconstruct_stiffness_diagnostic
from src.plotting import copy_figures_to_paper_run


def main() -> dict[str, list[Path] | int]:
    basic_summary = run_basic_verification.main()
    atlas_summary = generate_tip_locus_atlas.main()
    comparison_summary = generate_prb3r_comparison_plots.main()
    validation_summary = validate_against_expected_errors.main()
    debug_summary = debug_kappa1_case.main()
    stiffness_summary = reconstruct_stiffness_diagnostic.main()

    final_local_figures = atlas_summary["local_figures"] + comparison_summary["local_figures"]
    if validation_summary.get("validation_passed", False):
        paper_figures = copy_figures_to_paper_run(final_local_figures)
    else:
        paper_figures = []

    paper_summary = {
        "csv_files": [],
        "local_figures": [],
        "paper_figures": paper_figures,
        "warnings": 0,
    }
    summaries = [
        basic_summary,
        atlas_summary,
        comparison_summary,
        validation_summary,
        debug_summary,
        stiffness_summary,
        paper_summary,
    ]
    combined = _combine(summaries)
    combined["validation_passed"] = bool(validation_summary.get("validation_passed", False))
    _print_summary(combined)
    return combined


def _combine(summaries: list[dict[str, object]]) -> dict[str, object]:
    combined: dict[str, object] = {
        "csv_files": [],
        "local_figures": [],
        "paper_figures": [],
        "warnings": 0,
    }
    for summary in summaries:
        for key in ["csv_files", "local_figures", "paper_figures"]:
            combined[key].extend(summary.get(key, []))
        combined["warnings"] += int(summary.get("warnings", 0))
    return combined


def _print_summary(summary: dict[str, list[Path] | int]) -> None:
    print("")
    print("Generation summary")
    print("CSV files generated:")
    for path in summary["csv_files"]:
        print(f"  {path}")

    print("Local figures generated:")
    for path in summary["local_figures"]:
        print(f"  {path}")

    print("Paper figures copied:")
    for path in summary["paper_figures"]:
        print(f"  {path}")

    print(f"Validation passed: {summary.get('validation_passed', False)}")
    print(f"Solve failures or warnings: {summary['warnings']}")


if __name__ == "__main__":
    main()
