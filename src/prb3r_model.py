from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, root


SOLVE_TOL = 1.0e-6


@dataclass
class PRB3RParameters:
    gamma0: float = 0.10
    gamma1: float = 0.35
    gamma2: float = 0.40
    gamma3: float = 0.15
    k1: float = 3.51
    k2: float = 2.99
    k3: float = 2.58

    @property
    def gammas(self) -> np.ndarray:
        return np.array([self.gamma0, self.gamma1, self.gamma2, self.gamma3], dtype=float)

    @property
    def stiffnesses(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3], dtype=float)

    def validate(self) -> None:
        gammas = self.gammas
        stiffnesses = self.stiffnesses
        if not np.isclose(np.sum(gammas), 1.0, atol=1.0e-12):
            raise ValueError("PRB gamma values must sum to 1.")
        if np.any(gammas <= 0.0):
            raise ValueError("All PRB gamma values must be positive.")
        if np.any(stiffnesses <= 0.0):
            raise ValueError("All PRB stiffness values must be positive.")


def forward_kinematics(theta: np.ndarray, params: PRB3RParameters | None = None) -> dict[str, float]:
    params = _params(params)
    theta1, theta2, theta3 = _theta_array(theta)
    angle12 = theta1 + theta2
    angle123 = angle12 + theta3

    qx = (
        params.gamma0
        + params.gamma1 * math.cos(theta1)
        + params.gamma2 * math.cos(angle12)
        + params.gamma3 * math.cos(angle123)
    )
    qy = (
        params.gamma1 * math.sin(theta1)
        + params.gamma2 * math.sin(angle12)
        + params.gamma3 * math.sin(angle123)
    )

    return {"Qx": float(qx), "Qy": float(qy), "theta0": float(angle123)}


def joint_points(theta: np.ndarray, params: PRB3RParameters | None = None) -> np.ndarray:
    params = _params(params)
    theta1, theta2, theta3 = _theta_array(theta)
    angle12 = theta1 + theta2
    angle123 = angle12 + theta3

    points = np.zeros((5, 2), dtype=float)
    points[1] = points[0] + np.array([params.gamma0, 0.0])
    points[2] = points[1] + params.gamma1 * np.array([math.cos(theta1), math.sin(theta1)])
    points[3] = points[2] + params.gamma2 * np.array([math.cos(angle12), math.sin(angle12)])
    points[4] = points[3] + params.gamma3 * np.array([math.cos(angle123), math.sin(angle123)])
    return points


def inverse_kinematics_from_tip(
    Qx: float,
    Qy: float,
    theta0: float,
    params: PRB3RParameters | None = None,
    branch: str = "auto",
    previous_theta: np.ndarray | None = None,
) -> dict[str, object]:
    params = _params(params)
    Qx = float(Qx)
    Qy = float(Qy)
    theta0 = float(theta0)

    if branch not in {"auto", "plus", "minus"}:
        raise ValueError("branch must be 'auto', 'plus', or 'minus'.")
    if not np.all(np.isfinite([Qx, Qy, theta0])):
        raise ValueError("Qx, Qy, and theta0 must be finite.")

    px = Qx - params.gamma3 * math.cos(theta0) - params.gamma0
    py = Qy - params.gamma3 * math.sin(theta0)
    denominator = 2.0 * params.gamma1 * params.gamma2
    acos_arg = (px * px + py * py - params.gamma1**2 - params.gamma2**2) / denominator
    acos_arg = _clip_acos_arg(acos_arg, Qx, Qy, theta0)
    theta2_abs = math.acos(acos_arg)

    candidates = {
        "plus": _ik_candidate(px, py, theta0, theta2_abs, params),
        "minus": _ik_candidate(px, py, theta0, -theta2_abs, params),
    }

    if branch in {"plus", "minus"}:
        branch_used = branch
    else:
        branch_used = _choose_ik_branch(candidates, theta0, previous_theta)

    theta = candidates[branch_used]
    kinematics = forward_kinematics(theta, params)
    reconstruction_error = float(
        np.linalg.norm(
            np.array(
                [
                    kinematics["Qx"] - Qx,
                    kinematics["Qy"] - Qy,
                    kinematics["theta0"] - theta0,
                ],
                dtype=float,
            )
        )
    )

    return {
        "theta": theta,
        "reconstruction_error": reconstruction_error,
        "branch_used": branch_used,
    }


def jacobian_transpose(theta: np.ndarray, params: PRB3RParameters | None = None) -> np.ndarray:
    params = _params(params)
    theta1, theta2, theta3 = _theta_array(theta)
    angle12 = theta1 + theta2
    angle123 = angle12 + theta3

    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    c12 = math.cos(angle12)
    s12 = math.sin(angle12)
    c123 = math.cos(angle123)
    s123 = math.sin(angle123)

    return np.array(
        [
            [
                -params.gamma1 * s1 - params.gamma2 * s12 - params.gamma3 * s123,
                params.gamma1 * c1 + params.gamma2 * c12 + params.gamma3 * c123,
                1.0,
            ],
            [
                -params.gamma2 * s12 - params.gamma3 * s123,
                params.gamma2 * c12 + params.gamma3 * c123,
                1.0,
            ],
            [-params.gamma3 * s123, params.gamma3 * c123, 1.0],
        ],
        dtype=float,
    )


def residual(
    theta: np.ndarray,
    alpha: float,
    beta: float,
    phi: float,
    params: PRB3RParameters | None = None,
) -> np.ndarray:
    params = _params(params)
    theta_values = _theta_array(theta)
    generalized_load = np.array(
        [2.0 * alpha * math.cos(phi), 2.0 * alpha * math.sin(phi), beta],
        dtype=float,
    )
    spring_moment = params.stiffnesses * theta_values
    return spring_moment - jacobian_transpose(theta_values, params) @ generalized_load


def solve_prb3r(
    alpha: float,
    beta: float,
    phi: float,
    params: PRB3RParameters | None = None,
    initial_guess: np.ndarray | None = None,
) -> dict[str, object]:
    params = _params(params)
    alpha = float(alpha)
    beta = float(beta)
    phi = float(phi)

    guess = _initial_guess(alpha, beta) if initial_guess is None else _theta_array(initial_guess)

    def fun(theta: np.ndarray) -> np.ndarray:
        return residual(theta, alpha, beta, phi, params)

    solution = root(fun, guess, method="hybr")
    theta = np.asarray(solution.x, dtype=float)
    residual_norm = float(np.linalg.norm(fun(theta)))
    message = str(solution.message)
    solver_success = bool(solution.success)

    if (not solver_success) or residual_norm > SOLVE_TOL:
        fallback = least_squares(fun, theta, xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12, max_nfev=1000)
        theta = np.asarray(fallback.x, dtype=float)
        residual_norm = float(np.linalg.norm(fun(theta)))
        message = f"root: {message}; least_squares: {fallback.message}"
        solver_success = bool(fallback.success)

    success = bool(solver_success and residual_norm <= SOLVE_TOL and np.all(np.isfinite(theta)))
    kinematics = forward_kinematics(theta, params)

    return {
        "theta": theta,
        "Qx": kinematics["Qx"],
        "Qy": kinematics["Qy"],
        "theta0": kinematics["theta0"],
        "success": success,
        "message": message,
        "residual_norm": residual_norm,
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
    }


def sweep_prb3r_for_beam_locus(
    theta0_values: np.ndarray | None = None,
    kappa: float | None = None,
    phi: float | None = None,
    params: PRB3RParameters | None = None,
    n_points: int = 150,
    endpoint_margin: float = 0.02,
    beta_sign: int = 1,
) -> pd.DataFrame:
    from . import beam_theory

    if kappa is None or phi is None:
        raise ValueError("kappa and phi are required for PRB beam locus sweeps.")
    if beta_sign not in (-1, 1):
        raise ValueError("beta_sign must be +1 or -1.")
    if theta0_values is None:
        theta0_values = beam_theory.make_theta0_sweep(
            kappa,
            phi,
            n_points=n_points,
            endpoint_margin=endpoint_margin,
        )

    rows = []
    params = _params(params)
    previous_theta = None

    for theta0_ref in theta0_values:
        try:
            reference = beam_theory.compute_tip_from_theta0(float(theta0_ref), kappa, phi)
            beta_for_prb = beta_sign * reference["beta"]
            solve = solve_prb3r(
                reference["alpha"],
                beta_for_prb,
                phi,
                params=params,
                initial_guess=previous_theta,
            )
            theta = np.asarray(solve["theta"], dtype=float)

            delta_theta_norm = np.nan
            branch_jump_flag = False
            if solve["success"] and previous_theta is not None:
                delta_theta_norm = float(np.linalg.norm(theta - previous_theta))
                branch_jump_flag = bool(delta_theta_norm > 0.5)

            if solve["success"]:
                previous_theta = theta

            rows.append(
                {
                    "theta0_ref": float(theta0_ref),
                    "alpha": reference["alpha"],
                    "beta": reference["beta"],
                    "beta_for_prb": beta_for_prb,
                    "beta_sign": int(beta_sign),
                    "kappa": float(kappa),
                    "phi": float(phi),
                    "Qx_ref": reference["Qx"],
                    "Qy_ref": reference["Qy"],
                    "theta1": theta[0],
                    "theta2": theta[1],
                    "theta3": theta[2],
                    "Qx_prb": solve["Qx"],
                    "Qy_prb": solve["Qy"],
                    "theta0_prb": solve["theta0"],
                    "success": solve["success"],
                    "residual_norm": solve["residual_norm"],
                    "delta_theta_norm": delta_theta_norm,
                    "branch_jump_flag": branch_jump_flag,
                }
            )
        except ValueError:
            rows.append(_failed_sweep_row(theta0_ref, kappa, phi, beta_sign))

    return pd.DataFrame(rows)


def _params(params: PRB3RParameters | None) -> PRB3RParameters:
    value = PRB3RParameters() if params is None else params
    value.validate()
    return value


def _theta_array(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype=float)
    if values.shape != (3,):
        raise ValueError("theta must contain exactly three joint rotations.")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain finite values.")
    return values


def _initial_guess(alpha: float, beta: float) -> np.ndarray:
    if abs(beta) > max(2.0 * abs(alpha), 1.0e-12):
        return np.array([beta / 3.0, beta / 3.0, beta / 3.0], dtype=float)
    return np.array([0.05, 0.05, 0.05], dtype=float)


def _ik_candidate(px: float, py: float, theta0: float, theta2: float, params: PRB3RParameters) -> np.ndarray:
    theta1 = math.atan2(py, px) - math.atan2(
        params.gamma2 * math.sin(theta2),
        params.gamma1 + params.gamma2 * math.cos(theta2),
    )
    theta3 = theta0 - theta1 - theta2
    return np.array([theta1, theta2, theta3], dtype=float)


def _clip_acos_arg(value: float, Qx: float, Qy: float, theta0: float) -> float:
    clip_tol = 1.0e-9
    if not math.isfinite(value):
        raise ValueError(f"Invalid IK geometry for Qx={Qx}, Qy={Qy}, theta0={theta0}; acos argument is {value}")
    if value < -1.0 - clip_tol or value > 1.0 + clip_tol:
        raise ValueError(
            f"Tip is outside PRB IK reach for Qx={Qx}, Qy={Qy}, theta0={theta0}; "
            f"acos argument is {value}"
        )
    return float(np.clip(value, -1.0, 1.0))


def _choose_ik_branch(
    candidates: dict[str, np.ndarray],
    theta0: float,
    previous_theta: np.ndarray | None,
) -> str:
    if previous_theta is not None:
        previous = _theta_array(previous_theta)
        return min(candidates, key=lambda key: float(np.linalg.norm(candidates[key] - previous)))

    sign = math.copysign(1.0, theta0) if abs(theta0) > 1.0e-12 else 1.0
    monotonic = [
        key for key, theta in candidates.items()
        if np.all(sign * theta >= -1.0e-10)
    ]
    if monotonic:
        return min(monotonic, key=lambda key: float(np.linalg.norm(candidates[key])))

    return min(candidates, key=lambda key: float(np.linalg.norm(candidates[key])))


def _failed_sweep_row(theta0_ref: float, kappa: float, phi: float, beta_sign: int) -> dict[str, float | bool]:
    return {
        "theta0_ref": float(theta0_ref),
        "alpha": np.nan,
        "beta": np.nan,
        "beta_for_prb": np.nan,
        "beta_sign": int(beta_sign),
        "kappa": float(kappa),
        "phi": float(phi),
        "Qx_ref": np.nan,
        "Qy_ref": np.nan,
        "theta1": np.nan,
        "theta2": np.nan,
        "theta3": np.nan,
        "Qx_prb": np.nan,
        "Qy_prb": np.nan,
        "theta0_prb": np.nan,
        "success": False,
        "residual_norm": np.nan,
        "delta_theta_norm": np.nan,
        "branch_jump_flag": False,
    }
