import numpy as np
import pandas as pd

from src.fea_processing import build_reference_cases, compute_fea_quantities, merge_fea_with_reference


def test_compute_fea_quantities_case1():
    df = pd.DataFrame({"case_id": [1], "uy_tip_mm": [-6.714], "uz_tip_mm": [-32.24]})
    result = compute_fea_quantities(df)
    assert np.isclose(result.loc[0, "Qx_fea"], 0.93286)
    assert np.isclose(result.loc[0, "Qy_fea"], 0.3224)


def test_build_reference_cases():
    result = build_reference_cases()
    required = {
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
    }
    assert len(result) == 10
    assert result["case_id"].tolist() == list(range(1, 11))
    assert required.issubset(result.columns)


def test_merge_fea_with_reference():
    fea = pd.DataFrame(
        {
            "case_id": [1],
            "uy_tip_mm": [-6.714],
            "uz_tip_mm": [-32.24],
            "max_von_mises_mpa": [5.599],
        }
    )
    fea = compute_fea_quantities(fea)
    ref = build_reference_cases().query("case_id == 1")
    result = merge_fea_with_reference(fea, ref)
    assert "tip_error_percent" in result.columns
    assert np.isfinite(result.loc[0, "tip_error_percent"])
