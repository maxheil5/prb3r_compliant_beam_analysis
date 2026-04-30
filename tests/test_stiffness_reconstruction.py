import numpy as np
import pandas as pd

from src.prb3r_model import PRB3RParameters, forward_kinematics, jacobian_transpose
from src.stiffness_reconstruction import compute_joint_torques, reconstruct_stiffness_for_locus


def test_compute_joint_torques_matches_jacobian_load_product():
    theta = np.array([0.2, 0.1, 0.05])
    alpha = 0.3
    beta = 0.2
    phi = np.pi / 2.0
    load = np.array([2.0 * alpha * np.cos(phi), 2.0 * alpha * np.sin(phi), beta])
    assert np.allclose(compute_joint_torques(theta, alpha, beta, phi), jacobian_transpose(theta) @ load)


def test_reconstruct_stiffness_recovers_synthetic_consistent_fit():
    params = PRB3RParameters()
    rows = []
    for theta in [np.array([0.12, 0.08, 0.05]), np.array([0.22, 0.15, 0.09]), np.array([0.32, 0.22, 0.13])]:
        tip = forward_kinematics(theta, params=params)
        load = np.linalg.solve(jacobian_transpose(theta, params), params.stiffnesses * theta)
        rows.append(
            {
                "theta0": tip["theta0"],
                "alpha": 0.5 * np.hypot(load[0], load[1]),
                "beta": load[2],
                "kappa": np.nan,
                "phi": np.arctan2(load[1], load[0]),
                "Qx": tip["Qx"],
                "Qy": tip["Qy"],
            }
        )

    reconstructed, fit = reconstruct_stiffness_for_locus(pd.DataFrame(rows), params=params, branch="plus")
    assert reconstructed["reconstruction_error"].max() < 1.0e-10
    assert np.isclose(fit["k1_fit"], params.k1)
    assert np.isclose(fit["k2_fit"], params.k2)
    assert np.isclose(fit["k3_fit"], params.k3)
