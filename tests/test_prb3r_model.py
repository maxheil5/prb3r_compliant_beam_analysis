import numpy as np

from src.prb3r_model import (
    PRB3RParameters,
    forward_kinematics,
    inverse_kinematics_from_tip,
    jacobian_transpose,
    joint_points,
    residual,
    solve_prb3r,
    sweep_prb3r_for_beam_locus,
)
from src.plotting import PAPER_RUN_FIGURE_DIR, RUN_ANALYTICAL_FIGURE_DIR, RUN_CSV_DIR


def test_default_parameter_validation_passes():
    PRB3RParameters().validate()


def test_forward_kinematics_zero_configuration():
    result = forward_kinematics(np.array([0.0, 0.0, 0.0]))
    assert np.isclose(result["Qx"], 1.0)
    assert np.isclose(result["Qy"], 0.0)
    assert np.isclose(result["theta0"], 0.0)


def test_joint_points_zero_configuration():
    points = joint_points(np.array([0.0, 0.0, 0.0]))
    assert points.shape == (5, 2)
    assert np.allclose(points[-1], np.array([1.0, 0.0]))


def test_jacobian_transpose_shape():
    result = jacobian_transpose(np.array([0.1, 0.2, 0.3]))
    assert result.shape == (3, 3)


def test_residual_shape():
    result = residual(np.array([0.1, 0.2, 0.3]), alpha=0.2, beta=0.1, phi=np.pi / 2.0)
    assert result.shape == (3,)


def test_solve_prb3r_pure_moment_returns_finite_solution():
    result = solve_prb3r(alpha=0.0, beta=0.5, phi=np.pi / 2.0)
    assert np.all(np.isfinite(result["theta"]))
    assert np.isfinite(result["Qx"])
    assert np.isfinite(result["Qy"])
    assert np.isfinite(result["theta0"])
    assert np.isfinite(result["residual_norm"])
    if result["success"]:
        assert result["residual_norm"] < 1.0e-6


def test_sweep_prb3r_auto_sweep_includes_beta_sign_and_branch_columns():
    result = sweep_prb3r_for_beam_locus(
        kappa=1.0,
        phi=np.pi / 2.0,
        n_points=5,
        endpoint_margin=0.05,
        beta_sign=-1,
    )
    required = {"beta_for_prb", "beta_sign", "delta_theta_norm", "branch_jump_flag"}
    assert required.issubset(result.columns)
    assert len(result) == 5
    assert np.all(result["beta_sign"] == -1)
    assert np.allclose(result["beta_for_prb"], -result["beta"])


def test_run_scoped_output_paths_use_run_2():
    assert RUN_CSV_DIR.name == "Run 3"
    assert "Run 3" in RUN_ANALYTICAL_FIGURE_DIR.parts
    assert PAPER_RUN_FIGURE_DIR.name == "Run 3"


def test_inverse_kinematics_round_trip_plus_branch():
    theta = np.array([0.25, 0.35, 0.15])
    tip = forward_kinematics(theta)
    result = inverse_kinematics_from_tip(tip["Qx"], tip["Qy"], tip["theta0"], branch="plus")
    assert result["branch_used"] == "plus"
    assert result["reconstruction_error"] < 1.0e-10
    assert np.allclose(result["theta"], theta)


def test_inverse_kinematics_auto_uses_previous_theta_continuity():
    plus_theta = np.array([0.25, 0.35, 0.15])
    tip = forward_kinematics(plus_theta)
    minus = inverse_kinematics_from_tip(tip["Qx"], tip["Qy"], tip["theta0"], branch="minus")
    result = inverse_kinematics_from_tip(
        tip["Qx"],
        tip["Qy"],
        tip["theta0"],
        branch="auto",
        previous_theta=minus["theta"],
    )
    assert result["branch_used"] == "minus"
