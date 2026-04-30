from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning, quad
import warnings


SMALL_ANGLE_TOL = 1.0e-10
G_TOL = 1.0e-11


def g_theta(theta: float | np.ndarray, theta0: float, kappa: float, phi: float) -> float | np.ndarray:
    return np.cos(theta0 - phi) - np.cos(np.asarray(theta) - phi) + kappa


def valid_theta0_max(kappa: float, phi: float, cap_at_pi: bool = True) -> float:
    kappa = float(kappa)
    phi = float(phi)

    if not math.isfinite(phi) or phi <= 0.0:
        raise ValueError(f"phi must be a positive finite angle, got phi={phi}")

    if np.isinf(kappa):
        theta0_max = math.pi
    elif not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"kappa must be nonnegative or infinity, got kappa={kappa}")
    elif kappa <= 2.0:
        theta0_max = phi + math.acos(1.0 - kappa)
    else:
        theta0_max = math.pi

    if cap_at_pi:
        theta0_max = min(theta0_max, math.pi)

    return float(theta0_max)


def make_theta0_sweep(
    kappa: float,
    phi: float,
    n_points: int = 150,
    theta0_min: float = 0.02,
    endpoint_margin: float = 0.02,
) -> np.ndarray:
    if int(n_points) != n_points or n_points < 2:
        raise ValueError(f"n_points must be an integer greater than 1, got n_points={n_points}")
    if not math.isfinite(theta0_min) or theta0_min < 0.0:
        raise ValueError(f"theta0_min must be nonnegative and finite, got theta0_min={theta0_min}")
    if not math.isfinite(endpoint_margin) or endpoint_margin < 0.0:
        raise ValueError(f"endpoint_margin must be nonnegative and finite, got endpoint_margin={endpoint_margin}")

    theta0_max = valid_theta0_max(kappa, phi)
    theta0_upper = theta0_max - endpoint_margin
    if theta0_upper <= theta0_min:
        raise ValueError(
            f"No valid theta0 sweep interval for kappa={kappa}, phi={phi}, "
            f"theta0_min={theta0_min}, endpoint_margin={endpoint_margin}"
        )

    return np.linspace(theta0_min, theta0_upper, int(n_points))


def pure_moment_tip(theta0: float) -> dict[str, float]:
    theta0 = float(theta0)
    if abs(theta0) < SMALL_ANGLE_TOL:
        qx = 1.0
        qy = 0.0
    else:
        qx = math.sin(theta0) / theta0
        qy = (1.0 - math.cos(theta0)) / theta0

    return {
        "theta0": theta0,
        "kappa": np.inf,
        "phi": np.nan,
        "alpha": 0.0,
        "beta": theta0,
        "Qx": float(qx),
        "Qy": float(qy),
    }


def compute_tip_from_theta0(
    theta0: float,
    kappa: float,
    phi: float,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> dict[str, float]:
    theta0 = float(theta0)
    kappa = float(kappa)
    phi = float(phi)

    if np.isinf(kappa):
        return pure_moment_tip(theta0)

    if abs(theta0) < SMALL_ANGLE_TOL:
        return {
            "theta0": theta0,
            "kappa": kappa,
            "phi": phi,
            "alpha": 0.0,
            "beta": 0.0,
            "Qx": 1.0,
            "Qy": 0.0,
        }

    if kappa < 0:
        raise ValueError(f"kappa must be nonnegative for theta0={theta0}, kappa={kappa}, phi={phi}")

    _validate_g_interval(theta0, kappa, phi)

    i0 = _integrate_weighted(theta0, kappa, phi, lambda theta: 1.0, atol, rtol)
    ix = _integrate_weighted(theta0, kappa, phi, math.cos, atol, rtol)
    iy = _integrate_weighted(theta0, kappa, phi, math.sin, atol, rtol)

    if abs(i0) < SMALL_ANGLE_TOL:
        raise ValueError(f"I0 is too small for theta0={theta0}, kappa={kappa}, phi={phi}")

    qx = ix / i0
    qy = iy / i0
    alpha = (0.5 * i0) ** 2
    beta = 0.0 if abs(kappa) < G_TOL else 2.0 * math.sqrt(alpha * kappa)

    values = (i0, ix, iy, alpha, beta, qx, qy)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite beam solution for theta0={theta0}, kappa={kappa}, phi={phi}")

    return {
        "theta0": theta0,
        "kappa": kappa,
        "phi": phi,
        "alpha": float(alpha),
        "beta": float(beta),
        "Qx": float(qx),
        "Qy": float(qy),
    }


def sweep_beam_locus(
    theta0_values: np.ndarray | None = None,
    kappa: float | None = None,
    phi: float | None = None,
    n_points: int = 150,
    endpoint_margin: float = 0.02,
) -> pd.DataFrame:
    if kappa is None or phi is None:
        raise ValueError("kappa and phi are required for beam locus sweeps.")
    if theta0_values is None:
        theta0_values = make_theta0_sweep(kappa, phi, n_points=n_points, endpoint_margin=endpoint_margin)

    rows = []
    for theta0 in theta0_values:
        try:
            rows.append(compute_tip_from_theta0(float(theta0), kappa, phi))
        except ValueError:
            rows.append(
                {
                    "theta0": float(theta0),
                    "kappa": float(kappa),
                    "phi": float(phi),
                    "alpha": np.nan,
                    "beta": np.nan,
                    "Qx": np.nan,
                    "Qy": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["theta0", "kappa", "phi", "alpha", "beta", "Qx", "Qy"])


def generate_standard_locus_dataset() -> pd.DataFrame:
    kappa_values = [0, 0.1, 1, 2, 5, 50]
    phi = np.pi / 2.0
    frames = [sweep_beam_locus(kappa=kappa, phi=phi) for kappa in kappa_values]
    return pd.concat(frames, ignore_index=True)


def generate_force_angle_dataset() -> pd.DataFrame:
    kappa = 0.0
    phi_degrees = [30, 60, 120, 135, 150, 175]
    frames = []

    for phi_deg in phi_degrees:
        frame = sweep_beam_locus(kappa=kappa, phi=np.deg2rad(phi_deg))
        frame["phi_deg"] = float(phi_deg)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _validate_g_interval(theta0: float, kappa: float, phi: float) -> None:
    grid = np.linspace(0.0, theta0, 501)
    values = np.asarray(g_theta(grid, theta0, kappa, phi), dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite g(theta) for theta0={theta0}, kappa={kappa}, phi={phi}")

    minimum = float(np.min(values))
    if minimum < -G_TOL:
        raise ValueError(f"Invalid g(theta) for theta0={theta0}, kappa={kappa}, phi={phi}; min g={minimum}")

    near_zero = np.isclose(values, 0.0, atol=G_TOL, rtol=0.0)
    if np.any(near_zero[1:-1]):
        raise ValueError(f"Interior zero in g(theta) for theta0={theta0}, kappa={kappa}, phi={phi}")


def _integrate_weighted(
    theta0: float,
    kappa: float,
    phi: float,
    weight: Callable[[float], float],
    atol: float,
    rtol: float,
) -> float:
    if abs(kappa) < G_TOL:
        return _integrate_force_case(theta0, kappa, phi, weight, atol, rtol)

    def integrand(theta: float) -> float:
        g_value = _checked_g(theta, theta0, kappa, phi)
        return weight(theta) / math.sqrt(g_value)

    return _quad_checked(integrand, 0.0, theta0, theta0, kappa, phi, atol, rtol)


def _integrate_force_case(
    theta0: float,
    kappa: float,
    phi: float,
    weight: Callable[[float], float],
    atol: float,
    rtol: float,
) -> float:
    endpoint_factor = -math.sin(theta0 - phi) * theta0
    if endpoint_factor <= G_TOL:
        raise ValueError(f"Invalid endpoint singularity for theta0={theta0}, kappa={kappa}, phi={phi}")

    def transformed(u: float) -> float:
        if abs(u) < SMALL_ANGLE_TOL:
            return 2.0 * theta0 * weight(theta0) / math.sqrt(endpoint_factor)

        theta = theta0 * (1.0 - u * u)
        g_value = _checked_g(theta, theta0, kappa, phi)
        return 2.0 * theta0 * u * weight(theta) / math.sqrt(g_value)

    return _quad_checked(transformed, 0.0, 1.0, theta0, kappa, phi, atol, rtol)


def _checked_g(theta: float, theta0: float, kappa: float, phi: float) -> float:
    value = float(g_theta(theta, theta0, kappa, phi))
    if not math.isfinite(value) or value < -G_TOL:
        raise ValueError(f"Invalid g(theta) for theta0={theta0}, kappa={kappa}, phi={phi}; g={value}")
    return max(value, 0.0)


def _quad_checked(
    integrand: Callable[[float], float],
    lower: float,
    upper: float,
    theta0: float,
    kappa: float,
    phi: float,
    atol: float,
    rtol: float,
) -> float:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=IntegrationWarning)
            value, error = quad(integrand, lower, upper, epsabs=atol, epsrel=rtol, limit=200)
    except Exception as exc:
        raise ValueError(f"Integration failed for theta0={theta0}, kappa={kappa}, phi={phi}: {exc}") from exc

    if not math.isfinite(value) or not math.isfinite(error):
        raise ValueError(f"Non-finite integral for theta0={theta0}, kappa={kappa}, phi={phi}")

    return float(value)
