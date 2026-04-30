from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beam_theory import generate_standard_locus_dataset
from src.plotting import RUN_ANALYTICAL_FIGURE_DIR, RUN_CSV_DIR, plot_tip_locus_atlas


CSV_PATH = RUN_CSV_DIR / "tip_locus_atlas_phi90.csv"
FIGURE_PATH = RUN_ANALYTICAL_FIGURE_DIR / "tip_locus_atlas_phi90.png"


def main() -> dict[str, list[Path] | int]:
    df = generate_standard_locus_dataset()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    figure_paths = plot_tip_locus_atlas(df, FIGURE_PATH)
    local_figures = [path for path in figure_paths if PROJECT_ROOT in path.parents]
    invalid_rows = int(df[["Qx", "Qy"]].isna().any(axis=1).sum())

    print(f"Saved continuous beam locus CSV: {CSV_PATH}")
    print(f"Saved continuous beam locus figure: {FIGURE_PATH}")
    return {
        "csv_files": [CSV_PATH],
        "local_figures": local_figures,
        "paper_figures": [],
        "warnings": invalid_rows,
    }


if __name__ == "__main__":
    main()
