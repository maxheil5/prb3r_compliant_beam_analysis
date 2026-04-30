from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEA_RESULTS_PATH = PROJECT_ROOT / "results" / "csv" / "FEA_Results" / "fea_results_filled.csv"
YIELD_STRENGTH_MPA = 30.0


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    columns = []
    for column in result.columns:
        name = str(column).strip().lower()
        name = name.replace("word report file name", "word_report_file_name")
        name = re.sub(r"[()\\[\\]{}]", "", name)
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        columns.append(name)
    result.columns = columns
    return result


def load_fea_results(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = DEFAULT_FEA_RESULTS_PATH if path is None else Path(path)
    df = pd.read_csv(csv_path, skiprows=1)
    df = normalize_column_names(df)
    df = df.dropna(how="all").copy()

    for column in _numeric_columns():
        if column in df.columns:
            df[column] = _to_numeric_series(df[column])

    if "case_id" in df.columns:
        df = df.dropna(subset=["case_id"]).copy()
        df["case_id"] = df["case_id"].astype(int)

    return df


def build_reference_cases() -> pd.DataFrame:
    rows = [
        [1, 0, 0.5, 90, 0.54984, 0.00000, 0.93378, 0.32591, 93.37775, 32.59074, 0, -0.09164, 0, "pure force low deflection"],
        [2, 0, 1.0, 90, 1.54350, 0.00000, 0.73836, 0.61037, 73.83564, 61.03718, 0, -0.25725, 0, "pure force medium deflection"],
        [3, 0, 1.4, 90, 4.40321, 0.00000, 0.47308, 0.79655, 47.30779, 79.65534, 0, -0.73387, 0, "pure force high deflection"],
        [4, 1, 1.0, 90, 0.18537, 0.86110, 0.82919, 0.48186, 82.91890, 48.18620, 0, -0.03090, -0.00717585, "mixed load sensitive case"],
        [5, 1, 2.0, 90, 0.86948, 1.86492, 0.40943, 0.74275, 40.94263, 74.27531, 0, -0.14491, -0.01554098, "mixed load sensitive case"],
        [6, 1, 2.5, 90, 1.89869, 2.75586, 0.19181, 0.75806, 19.18147, 75.80551, 0, -0.31645, -0.02296548, "high-slope sensitive case"],
        [7, 2, 1.0, 90, 0.10582, 0.92007, 0.83449, 0.47249, 83.44889, 47.24924, 0, -0.01764, -0.00766721, "mixed force and moment"],
        [8, 2, 2.0, 90, 0.46042, 1.91921, 0.42953, 0.72779, 42.95277, 72.77887, 0, -0.07674, -0.01599340, "mixed force and moment"],
        [9, 50, 1.0, 90, 0.00496, 0.99621, 0.84115, 0.46031, 84.11455, 46.03062, 0, -0.00083, -0.00830178, "moment-dominant"],
        [10, 50, 2.5, 90, 0.03133, 2.50308, 0.23838, 0.72122, 23.83827, 72.12245, 0, -0.00522, -0.02085903, "high moment-dominant"],
    ]
    columns = [
        "case_id",
        "kappa",
        "theta0_target_rad",
        "phi_deg",
        "alpha",
        "beta",
        "Qx_ref",
        "Qy_ref",
        "x_tip_ref_mm",
        "y_tip_ref_mm",
        "Fx_N",
        "Fz_N",
        "Mx_Nm",
        "load_description",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_fea_quantities(fea_df: pd.DataFrame, l_mm: float = 100.0) -> pd.DataFrame:
    result = fea_df.copy()
    result["Qx_fea"] = (l_mm + result["uy_tip_mm"]) / l_mm
    result["Qy_fea"] = -result["uz_tip_mm"] / l_mm
    result["x_tip_fea_mm"] = result["Qx_fea"] * l_mm
    result["y_tip_fea_mm"] = result["Qy_fea"] * l_mm
    return result


def merge_fea_with_reference(fea_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    merged = ref_df.merge(fea_df, on="case_id", how="left", validate="one_to_one")
    merged["Qx_error_percent"] = (merged["Qx_fea"] - merged["Qx_ref"]).abs() * 100.0
    merged["Qy_error_percent"] = (merged["Qy_fea"] - merged["Qy_ref"]).abs() * 100.0
    merged["tip_error_percent"] = (
        np.sqrt((merged["Qx_fea"] - merged["Qx_ref"]) ** 2 + (merged["Qy_fea"] - merged["Qy_ref"]) ** 2) * 100.0
    )
    merged["x_tip_error_mm"] = (merged["x_tip_fea_mm"] - merged["x_tip_ref_mm"]).abs()
    merged["y_tip_error_mm"] = (merged["y_tip_fea_mm"] - merged["y_tip_ref_mm"]).abs()
    merged["tip_error_mm"] = np.sqrt(
        (merged["x_tip_fea_mm"] - merged["x_tip_ref_mm"]) ** 2
        + (merged["y_tip_fea_mm"] - merged["y_tip_ref_mm"]) ** 2
    )
    merged["stress_safety_factor"] = YIELD_STRENGTH_MPA / merged["max_von_mises_mpa"]
    merged["stress_margin_MPa"] = YIELD_STRENGTH_MPA - merged["max_von_mises_mpa"]
    return merged


def summarize_fea_comparison(merged_df: pd.DataFrame) -> dict[str, float | int]:
    max_error_index = merged_df["tip_error_percent"].idxmax()
    max_stress_index = merged_df["max_von_mises_mpa"].idxmax()
    min_safety_index = merged_df["stress_safety_factor"].idxmin()

    return {
        "number_of_cases": int(merged_df["case_id"].nunique()),
        "max_tip_error_percent": float(merged_df.loc[max_error_index, "tip_error_percent"]),
        "mean_tip_error_percent": float(merged_df["tip_error_percent"].mean()),
        "max_stress_MPa": float(merged_df.loc[max_stress_index, "max_von_mises_mpa"]),
        "min_stress_safety_factor": float(merged_df.loc[min_safety_index, "stress_safety_factor"]),
        "case_id_with_max_error": int(merged_df.loc[max_error_index, "case_id"]),
        "case_id_with_max_stress": int(merged_df.loc[max_stress_index, "case_id"]),
        "case_id_with_min_safety_factor": int(merged_df.loc[min_safety_index, "case_id"]),
    }


def _numeric_columns() -> list[str]:
    return [
        "case_id",
        "mesh_size_mm",
        "element_size_target_mm",
        "total_nodes",
        "total_elements",
        "uy_tip_mm",
        "uz_tip_mm",
        "max_resultant_displacement_mm",
        "max_von_mises_mpa",
        "max_equivalent_strain",
    ]


def _to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")
