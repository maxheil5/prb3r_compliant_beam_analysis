from __future__ import annotations

import numpy as np
import pandas as pd


def tip_position_error(Qx_model: float, Qy_model: float, Qx_ref: float, Qy_ref: float) -> float:
    return float(np.sqrt((Qx_model - Qx_ref) ** 2 + (Qy_model - Qy_ref) ** 2) * 100.0)


def slope_error(theta_model: float, theta_ref: float) -> float:
    if abs(theta_ref) < 1.0e-12:
        return float(np.nan)
    return float(abs(theta_model - theta_ref) / abs(theta_ref) * 100.0)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["tip_error_percent"] = np.sqrt(
        (result["Qx_prb"] - result["Qx_ref"]) ** 2 + (result["Qy_prb"] - result["Qy_ref"]) ** 2
    ) * 100.0
    result["slope_error_percent"] = np.where(
        np.abs(result["theta0_ref"]) < 1.0e-12,
        np.nan,
        np.abs(result["theta0_prb"] - result["theta0_ref"]) / np.abs(result["theta0_ref"]) * 100.0,
    )
    return result
