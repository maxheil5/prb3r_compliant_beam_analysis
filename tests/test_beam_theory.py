import numpy as np

from src.beam_theory import (
    compute_tip_from_theta0,
    generate_standard_locus_dataset,
    make_theta0_sweep,
    pure_moment_tip,
    valid_theta0_max,
)


def test_pure_moment_tip_small_theta_returns_straight_tip():
    result = pure_moment_tip(1.0e-12)
    assert result["Qx"] == 1.0
    assert result["Qy"] == 0.0


def test_pure_moment_tip_matches_circular_arc():
    theta0 = np.pi / 2.0
    result = pure_moment_tip(theta0)
    assert np.isclose(result["Qx"], np.sin(theta0) / theta0)
    assert np.isclose(result["Qy"], (1.0 - np.cos(theta0)) / theta0)


def test_compute_tip_from_theta0_pure_force_returns_finite_values():
    result = compute_tip_from_theta0(theta0=0.5, kappa=0.0, phi=np.pi / 2.0)
    assert np.isfinite(result["Qx"])
    assert np.isfinite(result["Qy"])
    assert np.isfinite(result["alpha"])
    assert result["beta"] == 0.0


def test_generate_standard_locus_dataset_returns_required_columns():
    result = generate_standard_locus_dataset()
    required = {"theta0", "kappa", "phi", "alpha", "beta", "Qx", "Qy"}
    assert not result.empty
    assert required.issubset(result.columns)


def test_valid_theta0_max_uses_force_limit():
    assert np.isclose(valid_theta0_max(0.0, np.pi / 2.0), np.pi / 2.0)


def test_valid_theta0_max_caps_large_kappa_at_pi():
    assert np.isclose(valid_theta0_max(5.0, np.pi / 2.0), np.pi)


def test_make_theta0_sweep_avoids_endpoint_margin():
    sweep = make_theta0_sweep(kappa=0.0, phi=np.pi / 2.0, n_points=5, endpoint_margin=0.02)
    assert len(sweep) == 5
    assert np.isclose(sweep[0], 0.02)
    assert sweep[-1] <= np.pi / 2.0 - 0.02


def test_standard_dataset_uses_valid_kappa_specific_tails():
    result = generate_standard_locus_dataset()
    kappa0 = result[result["kappa"] == 0.0]
    assert np.isclose(kappa0["theta0"].max(), np.pi / 2.0 - 0.02)
    assert not result[["Qx", "Qy", "alpha", "beta"]].isna().any().any()
