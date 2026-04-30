from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .prb3r_model import (
    PRB3RParameters,
    inverse_kinematics_from_tip,
    jacobian_transpose,
    solve_prb3r,
)


ANGLE_DIVISION_TOL = 1.0e-6


def compute_joint_torques(
    theta: np.ndarray,
    alpha: float,
    beta: float,
    phi: float,
    params: PRB3RParameters | None = None,
) -> np.ndarray:
    load_vec = np.array(
        [2.0 * alpha * math.cos(phi), 2.0 * alpha * math.sin(phi), beta],
        dtype=float,
    )
    return jacobian_transpose(np.asarray(theta, dtype=float), params) @ load_vec


def reconstruct_stiffness_for_locus(
    df: pd.DataFrame,
    params: PRB3RParameters | None = None,
    branch: str = "auto",
) -> tuple[pd.DataFrame, dict[str, float]]:
    params = PRB3RParameters() if params is None else params
    params.validate()

    rows = []
    previous_theta = None

    for _, row in df.iterrows():
        try:
            ik = inverse_kinematics_from_tip(
                row["Qx"],
                row["Qy"],
                row["theta0"],
                params=params,
                branch=branch,
                previous_theta=previous_theta,
            )
            theta = np.asarray(ik["theta"], dtype=float)
            tau = compute_joint_torques(theta, row["alpha"], row["beta"], row["phi"], params=params)
            previous_theta = theta
            output = row.to_dict()
            output.update(
                {
                    "theta1": theta[0],
                    "theta2": theta[1],
                    "theta3": theta[2],
                    "tau1": tau[0],
                    "tau2": tau[1],
                    "tau3": tau[2],
                    "k1_point": _pointwise_stiffness(tau[0], theta[0]),
                    "k2_point": _pointwise_stiffness(tau[1], theta[1]),
                    "k3_point": _pointwise_stiffness(tau[2], theta[2]),
                    "reconstruction_error": ik["reconstruction_error"],
                    "branch_used": ik["branch_used"],
                    "ik_success": True,
                }
            )
        except ValueError:
            output = row.to_dict()
            output.update(
                {
                    "theta1": np.nan,
                    "theta2": np.nan,
                    "theta3": np.nan,
                    "tau1": np.nan,
                    "tau2": np.nan,
                    "tau3": np.nan,
                    "k1_point": np.nan,
                    "k2_point": np.nan,
                    "k3_point": np.nan,
                    "reconstruction_error": np.nan,
                    "branch_used": "failed",
                    "ik_success": False,
                }
            )
        rows.append(output)

    reconstructed = pd.DataFrame(rows)
    fit = {
        "k1_fit": _least_squares_stiffness(reconstructed["theta1"], reconstructed["tau1"]),
        "k2_fit": _least_squares_stiffness(reconstructed["theta2"], reconstructed["tau2"]),
        "k3_fit": _least_squares_stiffness(reconstructed["theta3"], reconstructed["tau3"]),
    }
    return reconstructed, fit


def solve_prb3r_with_custom_stiffness(
    alpha: float,
    beta: float,
    phi: float,
    k_values: np.ndarray | list[float] | tuple[float, float, float],
    params: PRB3RParameters | None = None,
    initial_guess: np.ndarray | None = None,
) -> dict[str, object]:
    base = PRB3RParameters() if params is None else params
    base.validate()
    k_values = np.asarray(k_values, dtype=float)
    if k_values.shape != (3,) or not np.all(np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise ValueError("k_values must contain three positive finite stiffness values.")

    custom = PRB3RParameters(
        gamma0=base.gamma0,
        gamma1=base.gamma1,
        gamma2=base.gamma2,
        gamma3=base.gamma3,
        k1=float(k_values[0]),
        k2=float(k_values[1]),
        k3=float(k_values[2]),
    )
    return solve_prb3r(alpha, beta, phi, params=custom, initial_guess=initial_guess)


def _pointwise_stiffness(tau: float, theta: float) -> float:
    if not math.isfinite(theta) or abs(theta) < ANGLE_DIVISION_TOL:
        return float(np.nan)
    return float(tau / theta)


def _least_squares_stiffness(theta: pd.Series, tau: pd.Series) -> float:
    theta_values = theta.to_numpy(dtype=float)
    tau_values = tau.to_numpy(dtype=float)
    valid = np.isfinite(theta_values) & np.isfinite(tau_values) & (np.abs(theta_values) > ANGLE_DIVISION_TOL)
    if not np.any(valid):
        return float(np.nan)

    numerator = float(np.sum(theta_values[valid] * tau_values[valid]))
    denominator = float(np.sum(theta_values[valid] ** 2))
    if denominator <= 0.0:
        return float(np.nan)
    return numerator / denominator
