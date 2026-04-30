const DEFAULT_GAMMAS = [0.1, 0.35, 0.4, 0.15];

export function nearestSweepRow(rows, kappa, theta0) {
  if (!rows?.length) {
    return null;
  }
  let best = null;
  let bestDistance = Infinity;
  for (const row of rows) {
    const distance = Math.abs(Number(row.kappa) - Number(kappa)) * 0.4 + Math.abs(Number(row.theta0_ref) - Number(theta0));
    if (distance < bestDistance) {
      best = row;
      bestDistance = distance;
    }
  }
  return best;
}

export function jointPointsFromTheta(theta, gammas = DEFAULT_GAMMAS) {
  const [gamma0, gamma1, gamma2, gamma3] = gammas;
  const [theta1, theta2, theta3] = theta.map(Number);
  const a12 = theta1 + theta2;
  const a123 = a12 + theta3;
  const points = [{ x: 0, y: 0 }];
  points.push({ x: gamma0, y: 0 });
  points.push({ x: points[1].x + gamma1 * Math.cos(theta1), y: points[1].y + gamma1 * Math.sin(theta1) });
  points.push({ x: points[2].x + gamma2 * Math.cos(a12), y: points[2].y + gamma2 * Math.sin(a12) });
  points.push({ x: points[3].x + gamma3 * Math.cos(a123), y: points[3].y + gamma3 * Math.sin(a123) });
  return points;
}

export function scaledCase(caseRow, progress) {
  const p = clamp(Number(progress), 0, 1);
  return {
    ...caseRow,
    Qx_ref_display: lerp(1, Number(caseRow.Qx_ref), p),
    Qy_ref_display: lerp(0, Number(caseRow.Qy_ref), p),
    Qx_fea_display: lerp(1, Number(caseRow.Qx_fea), p),
    Qy_fea_display: lerp(0, Number(caseRow.Qy_fea), p),
  };
}

export function beamCurve(qx, qy, steps = 52) {
  const points = [];
  for (let i = 0; i < steps; i += 1) {
    const s = i / (steps - 1);
    points.push({
      x: s * Number(qx),
      y: smooth(s) * Number(qy),
    });
  }
  return points;
}

export function mirroredPath(points, side) {
  const sign = side === "upper" ? -1 : 1;
  const offset = side === "upper" ? 0.34 : -0.34;
  return points.map((point) => ({ x: point.x, y: offset + sign * point.y }));
}

export function prbMirrorPoints(sweepRow, progress, side) {
  if (!sweepRow) {
    return [];
  }
  const theta = [sweepRow.theta1, sweepRow.theta2, sweepRow.theta3].map((value) => Number(value) * Number(progress));
  return mirroredPath(jointPointsFromTheta(theta), side);
}

export function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return number.toFixed(digits);
}

export function interpretation(caseRow) {
  const error = Number(caseRow.tip_error_percent);
  const factor = Number(caseRow.stress_safety_factor);
  const agreement = error < 1 ? "excellent agreement" : error < 10 ? "moderate discrepancy" : "large discrepancy";
  const stress = factor > 1 ? "below yield" : "near or above yield";
  return `${agreement}; stress is ${stress}.`;
}

function smooth(s) {
  return 3 * s * s - 2 * s * s * s;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
